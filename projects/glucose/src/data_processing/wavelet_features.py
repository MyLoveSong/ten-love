"""
Wavelet-based feature extraction for glucose time series.
Provides multi-scale time-frequency analysis capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import pywt
import logging

logger = logging.getLogger(__name__)


class WaveletFeatureExtractor(nn.Module):
    """
    Extracts multi-scale wavelet features from glucose time series.
    Converts wavelet coefficients into learnable tokens.
    """

    def __init__(self, wavelet: str = 'db4', levels: int = 3,
                 token_dim: int = 64, device: Optional[torch.device] = None):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.token_dim = token_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Learnable projections for each decomposition level
        self.approx_projections = nn.ModuleList([
            nn.Linear(1, token_dim) for _ in range(levels)
        ])
        self.detail_projections = nn.ModuleList([
            nn.Linear(1, token_dim) for _ in range(levels)
        ])

        # Position encodings for multi-scale tokens
        self.scale_embeddings = nn.Parameter(torch.randn(levels * 2, token_dim))

        self.to(self.device)

    def _extract_wavelet_coeffs(self, signal: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract wavelet coefficients using PyWavelets."""
        try:
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels)
            approx_coeffs = [coeffs[0]]  # Approximation coefficients
            detail_coeffs = coeffs[1:]   # Detail coefficients
            return approx_coeffs, detail_coeffs
        except Exception as e:
            logger.warning(f"Wavelet decomposition failed: {e}, using zero coefficients")
            # Fallback: return zero coefficients
            seq_len = len(signal)
            approx_coeffs = [np.zeros(seq_len // (2 ** self.levels))]
            detail_coeffs = [np.zeros(seq_len // (2 ** (i + 1))) for i in range(self.levels)]
            return approx_coeffs, detail_coeffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert glucose sequences to wavelet tokens.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 1)

        Returns:
            Wavelet tokens of shape (batch_size, num_tokens, token_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Process each sequence in the batch
        all_tokens = []

        for i in range(batch_size):
            sequence = x[i, :, 0].cpu().numpy()  # Extract 1D sequence

            # Extract wavelet coefficients
            approx_coeffs, detail_coeffs = self._extract_wavelet_coeffs(sequence)

            # Convert coefficients to tokens
            tokens = []

            # Process approximation coefficients
            for level, coeffs in enumerate(approx_coeffs):
                if len(coeffs) > 0:
                    # Take mean of coefficients as representative value
                    coeff_value = torch.tensor([np.mean(coeffs)], dtype=torch.float32, device=self.device)
                    token = self.approx_projections[level](coeff_value.unsqueeze(0))
                    token = token + self.scale_embeddings[level].unsqueeze(0)
                    tokens.append(token)

            # Process detail coefficients
            for level, coeffs in enumerate(detail_coeffs):
                if len(coeffs) > 0:
                    # Take mean of coefficients as representative value
                    coeff_value = torch.tensor([np.mean(coeffs)], dtype=torch.float32, device=self.device)
                    token = self.detail_projections[level](coeff_value.unsqueeze(0))
                    token = token + self.scale_embeddings[self.levels + level].unsqueeze(0)
                    tokens.append(token)

            # Stack tokens for this sequence
            if tokens:
                sequence_tokens = torch.cat(tokens, dim=0)  # (num_tokens, token_dim)
            else:
                # Fallback: create zero tokens
                sequence_tokens = torch.zeros(self.levels * 2, self.token_dim, device=self.device)

            all_tokens.append(sequence_tokens)

        # Stack all sequences
        # Pad to same length if needed
        max_tokens = max(tokens.shape[0] for tokens in all_tokens)
        padded_tokens = []

        for tokens in all_tokens:
            if tokens.shape[0] < max_tokens:
                padding = torch.zeros(max_tokens - tokens.shape[0], self.token_dim, device=self.device)
                tokens = torch.cat([tokens, padding], dim=0)
            padded_tokens.append(tokens.unsqueeze(0))

        return torch.cat(padded_tokens, dim=0)  # (batch_size, max_tokens, token_dim)


class WaveletTokenizer(nn.Module):
    """
    Tokenizes glucose sequences using wavelet decomposition.
    Integrates with existing sequence processing pipelines.
    """

    def __init__(self, input_dim: int = 1, token_dim: int = 64,
                 wavelet: str = 'db4', levels: int = 3,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.input_dim = input_dim
        self.token_dim = token_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wavelet feature extractor
        self.wavelet_extractor = WaveletFeatureExtractor(
            wavelet=wavelet, levels=levels, token_dim=token_dim, device=device
        )

        # Optional: sequence-to-token projection for raw sequences
        self.sequence_projection = nn.Linear(input_dim, token_dim)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize input sequences using both raw and wavelet features.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tuple of (sequence_tokens, wavelet_tokens)
        """
        # Raw sequence tokens
        sequence_tokens = self.sequence_projection(x)  # (batch_size, seq_len, token_dim)

        # Wavelet tokens
        wavelet_tokens = self.wavelet_extractor(x)  # (batch_size, num_wavelet_tokens, token_dim)

        return sequence_tokens, wavelet_tokens
