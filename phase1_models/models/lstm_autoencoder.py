"""LSTM-Autoencoder for sequence-level reconstruction-based anomaly detection.

This model trains an encoder-decoder LSTM on benign sequences only. At inference
time, anomalous sequences produce higher reconstruction error than the benign
distribution, and the resulting score is thresholded at the validation 99th
percentile to flag windows.

Architecture:
    encoder:    LSTM(input_dim -> hidden_dim, num_layers)
    bottleneck: Linear(hidden_dim -> latent_dim)
    decoder:    Linear(latent_dim -> hidden_dim) -> LSTM(hidden_dim -> hidden_dim) -> Linear(hidden_dim -> input_dim)

Reference:
    Malhotra et al., "LSTM-based Encoder-Decoder for Multi-sensor Anomaly
    Detection," ICML Workshop on Anomaly Detection, 2016.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder for time-series anomaly detection.

    The encoder compresses an input sequence (B, T, F) to a single latent vector
    of size latent_dim. The decoder unrolls T steps from a learned seed seeded
    with the latent vector, producing a reconstruction (B, T, F).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode a sequence of shape (B, T, F).

        Returns the reconstructed sequence with the same shape.
        """
        batch_size, seq_len, _ = x.shape

        # Encode
        _, (hidden, _) = self.encoder(x)
        # hidden: (num_layers, B, hidden_dim) -> take last layer
        latent = self.to_latent(hidden[-1])  # (B, latent_dim)

        # Prepare decoder initial state
        decoder_h0 = self.from_latent(latent).unsqueeze(0)  # (1, B, hidden_dim)
        if self.num_layers > 1:
            decoder_h0 = decoder_h0.repeat(self.num_layers, 1, 1)
        decoder_c0 = torch.zeros_like(decoder_h0)

        # Decoder input is a sequence of zeros of length seq_len; the LSTM
        # generates the reconstruction conditioned only on the latent state.
        decoder_input = torch.zeros(
            batch_size, seq_len, self.hidden_dim, device=x.device, dtype=x.dtype
        )
        decoded, _ = self.decoder(decoder_input, (decoder_h0, decoder_c0))
        reconstruction = self.output_proj(decoded)  # (B, T, F)
        return reconstruction


def lstm_autoencoder_score(model: LSTMAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Compute per-window reconstruction-error anomaly score.

    Args:
        model: trained LSTMAutoencoder.
        x:     input sequences of shape (B, T, F).

    Returns:
        Tensor of shape (B,) with mean squared error per window.
    """
    model.eval()
    with torch.no_grad():
        recon = model(x)
        # Per-window mean squared error across time and features
        score = ((recon - x) ** 2).mean(dim=(1, 2))
    return score
