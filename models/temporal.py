#!/usr/bin/env python3
"""
Temporal-aware modeling for GluFormer.

Implements temporal sequence encoding for user behavior patterns,
meal timing, and blood glucose response modeling.
Inspired by NSFC project: "面向个性化时序糖尿病膳食干预的提示工程"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for sequence modeling in GluFormer.

    Supports both LSTM and Transformer architectures for encoding
    temporal patterns in user behavior, meal timing, and health responses.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        architecture: str = "lstm",  # "lstm" or "transformer"
        max_seq_len: int = 50,
        num_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.max_seq_len = max_seq_len

        # Input projection to standardize dimensions
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        if architecture == "lstm":
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        elif architecture == "transformer":
            # Positional encoding
            self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)

            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)
            self.output_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode temporal sequences.

        Args:
            sequences: [batch_size, seq_len, input_dim] Input sequences
            attention_mask: [batch_size, seq_len] Attention mask (optional)
            return_hidden_states: Whether to return all hidden states

        Returns:
            encoded: [batch_size, output_dim] or [batch_size, seq_len, output_dim]
            hidden_states: Optional list of hidden states
        """
        batch_size, seq_len, _ = sequences.size()

        # Project input to hidden dimension
        x = self.input_projection(sequences)  # [batch_size, seq_len, hidden_dim]
        x = self.dropout(x)

        if self.architecture == "lstm":
            # Pack sequences if attention_mask is provided
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, (hidden, cell) = self.encoder(packed)

                # Unpack
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output, batch_first=True, total_length=seq_len
                )
            else:
                output, (hidden, cell) = self.encoder(x)

            # Use last hidden state as sequence representation
            if self.encoder.bidirectional:
                # Concatenate forward and backward
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, 2*hidden_dim]
            else:
                hidden = hidden[-1]  # [batch_size, hidden_dim]

            encoded = self.layer_norm(hidden)

        elif self.architecture == "transformer":
            # Add positional encoding
            x = self.pos_encoding(x)

            # Create attention mask
            if attention_mask is not None:
                # Convert to transformer format (True for masked positions)
                attn_mask = ~attention_mask.bool()  # [batch_size, seq_len]
                # Expand for multi-head attention
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                attn_mask = attn_mask.expand(-1, self.encoder.layers[0].self_attn.num_heads, seq_len, -1)
                attn_mask = attn_mask.reshape(batch_size * self.encoder.layers[0].self_attn.num_heads, seq_len, seq_len)
            else:
                attn_mask = None

            # Encode
            output = self.encoder(x, mask=attn_mask)  # [batch_size, seq_len, hidden_dim]

            # Use mean pooling as sequence representation
            encoded = output.mean(dim=1)  # [batch_size, hidden_dim]
            encoded = self.output_projection(encoded)
            encoded = self.layer_norm(encoded)

        return encoded, None  # Return None for hidden_states for now


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalUserProfile(nn.Module):
    """
    Temporal-aware user profile modeling.

    Models how user preferences and behaviors evolve over time,
    incorporating meal timing, cultural preferences, and health responses.
    """

    def __init__(
        self,
        base_profile_dim: int = 32,
        temporal_dim: int = 64,
        num_temporal_features: int = 10,  # meal timing, blood glucose, etc.
        architecture: str = "lstm"
    ):
        super().__init__()

        self.base_profile_dim = base_profile_dim
        self.temporal_dim = temporal_dim

        # Temporal encoder for user behavior sequences
        self.temporal_encoder = TemporalEncoder(
            input_dim=num_temporal_features,
            hidden_dim=temporal_dim,
            architecture=architecture,
            max_seq_len=30  # 30 days of history
        )

        # Fusion of static profile with temporal features
        # Use actual output dimension from temporal encoder
        temporal_output_dim = self.temporal_encoder.output_dim
        self.profile_fusion = nn.Sequential(
            nn.Linear(base_profile_dim + temporal_output_dim, base_profile_dim),
            nn.LayerNorm(base_profile_dim),
            nn.GELU(),
            nn.Linear(base_profile_dim, base_profile_dim)
        )

    def forward(
        self,
        static_profile: torch.Tensor,
        temporal_sequences: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Enhance static user profile with temporal information.

        Args:
            static_profile: [batch_size, profile_dim] Static user features
            temporal_sequences: [batch_size, seq_len, num_features] Temporal behavior sequences
            attention_mask: [batch_size, seq_len] Attention mask for temporal sequences

        Returns:
            enhanced_profile: [batch_size, profile_dim] Enhanced user profile
        """
        if temporal_sequences is not None:
            # Encode temporal patterns
            temporal_features, _ = self.temporal_encoder(
                temporal_sequences, attention_mask
            )  # [batch_size, temporal_dim]

            # Fuse static and temporal features
            combined = torch.cat([static_profile, temporal_features], dim=1)
            enhanced_profile = self.profile_fusion(combined)
        else:
            # No temporal information available
            enhanced_profile = static_profile

        return enhanced_profile


class MealTimingEncoder(nn.Module):
    """
    Specialized encoder for meal timing and blood glucose response patterns.

    Models the relationship between meal timing, food intake, and blood glucose responses.
    """

    def __init__(
        self,
        food_feature_dim: int = 64,
        timing_feature_dim: int = 8,  # time of day, meal type, etc.
        hidden_dim: int = 64,
        max_meals_per_day: int = 3
    ):
        super().__init__()

        self.food_feature_dim = food_feature_dim
        self.timing_feature_dim = timing_feature_dim
        self.hidden_dim = hidden_dim
        self.max_meals_per_day = max_meals_per_day

        # Encode meal timing
        self.timing_encoder = nn.Sequential(
            nn.Linear(timing_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Meal sequence encoder (daily meal patterns)
        self.meal_sequence_encoder = TemporalEncoder(
            input_dim=food_feature_dim + hidden_dim,
            hidden_dim=hidden_dim,
            architecture="lstm",
            max_seq_len=max_meals_per_day
        )

        # Blood glucose response predictor
        self.glucose_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # Predict glucose change
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(
        self,
        food_features: torch.Tensor,
        timing_features: torch.Tensor,
        meal_sequence_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode meal timing and predict glucose response.

        Args:
            food_features: [batch_size, max_meals, food_dim] Food features per meal
            timing_features: [batch_size, max_meals, timing_dim] Timing features per meal
            meal_sequence_mask: [batch_size, max_meals] Mask for valid meals

        Returns:
            meal_patterns: [batch_size, hidden_dim] Encoded meal patterns
            glucose_responses: [batch_size, max_meals] Predicted glucose responses
        """
        batch_size, max_meals, _ = food_features.size()

        # Encode timing features
        timing_encoded = self.timing_encoder(timing_features)  # [batch_size, max_meals, hidden_dim]

        # Combine food and timing features
        meal_features = torch.cat([food_features, timing_encoded], dim=2)  # [batch_size, max_meals, food_dim + hidden_dim]

        # Encode meal sequence
        meal_patterns, _ = self.meal_sequence_encoder(meal_features, meal_sequence_mask)  # [batch_size, hidden_dim]

        # Predict glucose responses for each meal
        glucose_responses = self.glucose_predictor(timing_encoded)  # [batch_size, max_meals, 1]
        glucose_responses = glucose_responses.squeeze(2)  # [batch_size, max_meals]

        return meal_patterns, glucose_responses


def create_temporal_features(
    user_history: Dict[str, Any],
    max_seq_len: int = 30
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create temporal feature sequences from user history.

    This is a utility function to convert raw user data into temporal sequences.
    In practice, this would be integrated into the data preprocessing pipeline.

    Args:
        user_history: Dictionary containing user behavior history
        max_seq_len: Maximum sequence length

    Returns:
        temporal_sequences: [seq_len, num_features] Temporal features
        attention_mask: [seq_len] Attention mask
    """
    # Placeholder implementation - would need actual data format
    # Features could include:
    # - Meal timing (hour of day)
    # - Food categories consumed
    # - Blood glucose levels
    # - Cultural preference indicators
    # - Health metrics

    num_features = 10  # Example: timing, glucose, preferences, etc.

    # Create dummy temporal sequences for demonstration
    temporal_sequences = torch.randn(max_seq_len, num_features)
    attention_mask = torch.ones(max_seq_len, dtype=torch.bool)

    return temporal_sequences, attention_mask


# Example usage and testing
if __name__ == "__main__":
    # Test TemporalEncoder
    print("Testing TemporalEncoder...")

    batch_size, seq_len, input_dim = 4, 20, 10
    hidden_dim = 64

    # Test LSTM
    lstm_encoder = TemporalEncoder(input_dim, hidden_dim, architecture="lstm")
    sequences = torch.randn(batch_size, seq_len, input_dim)
    output, _ = lstm_encoder(sequences)
    print(f"LSTM output shape: {output.shape}")

    # Test Transformer
    transformer_encoder = TemporalEncoder(input_dim, hidden_dim, architecture="transformer")
    output, _ = transformer_encoder(sequences)
    print(f"Transformer output shape: {output.shape}")

    # Test TemporalUserProfile
    print("\nTesting TemporalUserProfile...")
    profile_dim = 32
    temporal_profile = TemporalUserProfile(
        base_profile_dim=profile_dim,
        temporal_dim=hidden_dim
    )

    static_profile = torch.randn(batch_size, profile_dim)
    temporal_seq = torch.randn(batch_size, seq_len, 10)  # 10 temporal features
    enhanced = temporal_profile(static_profile, temporal_seq)
    print(f"Enhanced profile shape: {enhanced.shape}")

    print("\nTemporal modeling modules ready!")
