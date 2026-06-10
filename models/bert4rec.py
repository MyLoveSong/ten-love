#!/usr/bin/env python3
"""
BERT4Rec model for sequential recommendation with cultural features integration.

BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer
Adapted for cultural-aware medical recommendation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class BERT4RecEncoder(nn.Module):
    """
    BERT4Rec encoder with cultural feature integration.

    Based on the paper: "BERT4Rec: Sequential Recommendation with Bidirectional Encoder
    Representations from Transformer" (Sun et al., CIKM 2019)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 50,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        cultural_dim: int = 64,
        item_embedding_dim: int = 256,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.cultural_dim = cultural_dim

        # Item embeddings
        self.item_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)

        # Cultural feature processing
        if cultural_dim > 0:
            self.cultural_encoder = nn.Linear(cultural_dim, hidden_size)
            self.cultural_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layers
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        # Special tokens
        self.mask_token = vocab_size - 1  # [MASK] token
        self.pad_token = 0  # [PAD] token

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cultural_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] - Item sequence
            attention_mask: [batch_size, seq_len] - Attention mask
            cultural_features: [batch_size, cultural_dim] - User cultural features

        Returns:
            logits: [batch_size, seq_len, vocab_size] - Prediction logits
        """
        batch_size, seq_len = input_ids.size()

        # Item embeddings + position embeddings
        item_embeds = self.item_embeddings(input_ids)  # [batch_size, seq_len, item_embedding_dim]

        # Project to hidden size if needed
        if item_embeds.size(-1) != self.hidden_size:
            item_embeds = F.linear(item_embeds, self.item_embeddings.weight.T[:self.hidden_size])

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(positions)

        embeddings = item_embeds + position_embeds

        # Integrate cultural features
        if cultural_features is not None and self.cultural_dim > 0:
            # Encode cultural features
            cultural_embeds = self.cultural_encoder(cultural_features)  # [batch_size, hidden_size]
            cultural_embeds = cultural_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]

            # Apply gating mechanism
            combined = torch.cat([embeddings, cultural_embeds], dim=-1)  # [batch_size, seq_len, hidden_size*2]
            gate = self.cultural_gate(combined)  # [batch_size, seq_len, hidden_size]

            embeddings = embeddings * gate + cultural_embeds * (1 - gate)

        embeddings = self.dropout(embeddings)

        # Create attention mask for transformer
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token).float()

        # Convert to transformer format (batch_first=True expects [batch, seq, hidden])
        # For nn.TransformerEncoder, src_key_padding_mask should be [batch_size, seq_len]
        key_padding_mask = (input_ids == self.pad_token)

        # Transformer encoding
        encoded = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=key_padding_mask
        )

        # Output predictions
        logits = self.output_layer(encoded)  # [batch_size, seq_len, vocab_size]

        return logits

    def get_masked_lm_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked language modeling loss for BERT4Rec training.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len] - Original sequence
            mask_positions: [batch_size, seq_len] - Boolean mask of masked positions

        Returns:
            loss: Scalar loss value
        """
        # Only compute loss on masked positions
        masked_logits = logits[mask_positions]  # [num_masked, vocab_size]
        masked_targets = targets[mask_positions]  # [num_masked]

        loss = F.cross_entropy(
            masked_logits,
            masked_targets,
            ignore_index=self.pad_token
        )

        return loss


class BERT4RecRecommender(nn.Module):
    """
    BERT4Rec-based recommender with cultural awareness for medical applications.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 50,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        cultural_dim: int = 64,
        item_embedding_dim: int = 256,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # BERT4Rec encoder
        self.encoder = BERT4RecEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cultural_dim=cultural_dim,
            item_embedding_dim=item_embedding_dim,
        )

        # Special tokens
        self.mask_token = vocab_size - 1
        self.pad_token = 0

    def forward(
        self,
        user_profile: torch.Tensor,
        cultural_features: torch.Tensor,
        item_sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for recommendation.

        Args:
            user_profile: [batch_size, user_profile_dim] - Not used in BERT4Rec
            cultural_features: [batch_size, cultural_dim] - Cultural context
            item_sequences: [batch_size, seq_len] - Item interaction sequences
            attention_mask: [batch_size, seq_len] - Attention mask
            labels: [batch_size, seq_len] - Target items for next position prediction

        Returns:
            outputs: Dictionary with logits and loss
        """
        # Get BERT4Rec predictions
        logits = self.encoder(
            input_ids=item_sequences,
            attention_mask=attention_mask,
            cultural_features=cultural_features
        )

        outputs = {
            'logits': logits,
            'recommendation_scores': logits[:, -1, :]  # Use last position for ranking
        }

        # Compute loss if labels provided
        if labels is not None:
            # For BERT4Rec, we predict the next item in sequence
            # Shift labels to predict next positions
            shifted_logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
            shifted_labels = labels[:, 1:]      # [batch, seq_len-1]

            # Create mask for valid positions
            valid_mask = (shifted_labels != self.pad_token)

            loss = F.cross_entropy(
                shifted_logits[valid_mask],
                shifted_labels[valid_mask],
                ignore_index=self.pad_token
            )

            outputs['loss'] = loss

        return outputs

    def predict_next_items(
        self,
        cultural_features: torch.Tensor,
        item_sequences: torch.Tensor,
        top_k: int = 10
    ) -> torch.Tensor:
        """
        Predict next items for given sequences.

        Args:
            cultural_features: [batch_size, cultural_dim]
            item_sequences: [batch_size, seq_len]
            top_k: Number of recommendations

        Returns:
            recommendations: [batch_size, top_k] - Recommended item indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                user_profile=None,  # Not used
                cultural_features=cultural_features,
                item_sequences=item_sequences
            )

            scores = outputs['recommendation_scores']  # [batch_size, vocab_size]

            # Get top-k recommendations
            _, top_indices = torch.topk(scores, top_k, dim=-1)

            return top_indices
