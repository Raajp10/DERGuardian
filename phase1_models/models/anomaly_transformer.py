"""Reconstruction-based AnomalyTransformer for time-series anomaly detection.

This is a lightweight reconstruction-style transformer trained on benign
sequences. The full Anomaly Transformer (Xu et al., ICLR 2022) couples a
prior-association branch with the series-association branch and uses minimax
training; for our compute budget we use a simplified reconstruction-only
variant that retains the positional encoding and stacked encoder layers.
This makes the implementation reviewable without sacrificing the core
attention-based representation learning that the paper argues for.

Reference:
    Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with
    Association Discrepancy," ICLR 2022.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for fixed-length sequences.

    Allows the transformer to use absolute position information without
    learning a separate embedding for each position.
    """

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves with the module to GPU but is not a
        # learnable parameter.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AnomalyTransformer(nn.Module):
    """Reconstruction-based anomaly transformer.

    The model projects each timestep into a d_model latent space, adds
    positional encoding, and processes the sequence with a stack of
    transformer encoder layers. A linear head reconstructs the original
    feature vector at each timestep. Anomaly score is the mean squared
    reconstruction error per window.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        if dim_feedforward is None:
            dim_feedforward = d_model * 2

        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.recon_head = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an input sequence.

        Args:
            x: tensor of shape (B, T, F).

        Returns:
            Reconstruction of shape (B, T, F).
        """
        h = self.input_proj(x)               # (B, T, d_model)
        h = self.positional(h)
        h = self.encoder(h)                  # (B, T, d_model)
        return self.recon_head(h)            # (B, T, F)


def anomaly_transformer_score(
    model: AnomalyTransformer, x: torch.Tensor
) -> torch.Tensor:
    """Per-window anomaly score = mean squared reconstruction error."""
    model.eval()
    with torch.no_grad():
        recon = model(x)
        score = ((recon - x) ** 2).mean(dim=(1, 2))
    return score
