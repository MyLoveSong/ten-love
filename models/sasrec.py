#!/usr/bin/env python3
"""
SASRec model for sequential recommendation.

SASRec: Self-Attentive Sequential Recommendation
Adapted for general recommender system baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class SASRecEncoder(nn.Module):
    """
    SASRec encoder for sequential recommendation.

    Based on the paper: "Self-Attentive Sequential Recommendation" (Kang et al., ICDM 2018)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 50,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        item_embedding_dim: int = 256,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # Item embeddings
        self.item_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)

        # Self-attention layers (unidirectional for SASRec)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers * 2)
        ])

        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through SASRec encoder.

        Args:
            input_ids: [batch_size, seq_len] Input item sequences
            attention_mask: [batch_size, seq_len] Attention mask (optional)
            return_hidden_states: Whether to return hidden states

        Returns:
            logits: [batch_size, seq_len, vocab_size] Prediction logits
            hidden_states: [batch_size, seq_len, hidden_size] Hidden states (optional)
        """
        batch_size, seq_len = input_ids.size()

        # Get embeddings
        item_embeds = self.item_embeddings(input_ids)  # [batch_size, seq_len, hidden_size]
        position_embeds = self.position_embeddings(torch.arange(seq_len, device=input_ids.device))
        position_embeds = position_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, hidden_size]

        # Combine embeddings
        hidden_states = item_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Create causal mask for unidirectional attention
        if attention_mask is None:
            # Create causal mask: future positions are masked
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(input_ids.device)
        else:
            causal_mask = ~attention_mask.unsqueeze(1).expand(-1, seq_len, -1).bool()

        all_hidden_states = [] if return_hidden_states else None

        # Self-attention layers
        for i, (attention, feed_forward, norm1, norm2) in enumerate(zip(
            self.attention_layers, self.feed_forward, self.layer_norms[::2], self.layer_norms[1::2]
        )):
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            # Self-attention with residual connection
            attn_output, _ = attention(
                hidden_states, hidden_states, hidden_states,
                attn_mask=causal_mask
            )
            hidden_states = norm1(hidden_states + attn_output)

            # Feed-forward with residual connection
            ff_output = feed_forward(hidden_states)
            hidden_states = norm2(hidden_states + ff_output)

        # Output projection
        logits = self.output_projection(hidden_states)  # [batch_size, seq_len, vocab_size]

        return logits, all_hidden_states


class SASRecRecommender(nn.Module):
    """
    SASRec-based recommender for general recommendation tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 50,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # SASRec encoder
        self.encoder = SASRecEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        user_sequences: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_scores: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for recommendation.

        Args:
            user_sequences: [batch_size, seq_len] User interaction sequences
            candidate_items: [batch_size, num_candidates] Candidate items (optional)
            attention_mask: [batch_size, seq_len] Attention mask
            return_scores: Whether to return recommendation scores

        Returns:
            outputs: Dictionary containing predictions and scores
        """
        batch_size = user_sequences.size(0)

        # Get sequence representations
        logits, _ = self.encoder(user_sequences, attention_mask)  # [batch_size, seq_len, vocab_size]

        # Use last position for prediction
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

        outputs = {
            'sequence_logits': logits,
            'last_logits': last_logits,
        }

        if return_scores and candidate_items is not None:
            # Get scores for candidate items
            candidate_scores = []
            for i in range(candidate_items.size(1)):
                candidate_ids = candidate_items[:, i]  # [batch_size]
                scores = last_logits.gather(1, candidate_ids.unsqueeze(1)).squeeze(1)  # [batch_size]
                candidate_scores.append(scores)

            recommendation_scores = torch.stack(candidate_scores, dim=1)  # [batch_size, num_candidates]
            outputs['recommendation_scores'] = recommendation_scores
            outputs['top_k_items'] = torch.topk(recommendation_scores, k=min(10, recommendation_scores.size(1)), dim=1).indices

        return outputs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SASRecRecommender':
        """Create model from configuration."""
        return cls(
            vocab_size=config.get('vocab_size', 1000),
            max_seq_len=config.get('max_seq_len', 50),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 2),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
        )
