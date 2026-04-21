"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements neural models logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    """Structured object used by the Phase 1 detector modeling workflow."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, bottleneck_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class GRUClassifier(nn.Module):
    """Structured object used by the Phase 1 detector modeling workflow."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1]).squeeze(-1)


class LSTMClassifier(nn.Module):
    """Structured object used by the Phase 1 detector modeling workflow."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)


class TinyTransformerClassifier(nn.Module):
    """Structured object used by the Phase 1 detector modeling workflow."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.proj(x))
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class TokenBaselineClassifier(nn.Module):
    """Structured object used by the Phase 1 detector modeling workflow."""

    def __init__(self, vocab_size: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        pooled = embedded.mean(dim=(1, 2))
        return self.head(pooled).squeeze(-1)
