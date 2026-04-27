"""Variational Autoencoder for probabilistic anomaly detection.

The VAE learns a latent distribution over benign system states. At inference,
the anomaly score combines reconstruction error and KL divergence from the
learned latent prior. Windows that the model cannot encode well receive higher
scores and are flagged when above the validation 99th percentile threshold.

Reference:
    Kingma & Welling, "Auto-Encoding Variational Bayes," ICLR 2014.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    """Standard MLP variational autoencoder for tabular residual features.

    The encoder outputs the mean and log-variance of a Gaussian posterior
    over the latent code. The reparameterization trick is used to sample
    the latent code so gradients flow through the sampling step.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a batch of inputs to (mu, logvar) of the latent posterior."""
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z from N(mu, exp(logvar)) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a latent code back to input space."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a full VAE forward pass.

        Returns:
            recon:  reconstruction of x, same shape as x.
            mu:     posterior mean, (B, latent_dim).
            logvar: posterior log-variance, (B, latent_dim).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative ELBO loss for VAE training.

    The reconstruction term is mean squared error and the regularizer is the
    KL divergence between the learned posterior and an isotropic Gaussian
    prior. The coefficient beta scales the KL term, allowing the user to bias
    the model toward sharper reconstructions (small beta) or stronger
    disentanglement (large beta).

    Returns:
        total: loss to backpropagate.
        recon_loss: reconstruction component (for logging).
        kl_loss: KL component (for logging).
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def vae_anomaly_score(
    model: VAE, x: torch.Tensor, beta: float = 0.1
) -> torch.Tensor:
    """Per-sample anomaly score combining reconstruction error and KL term.

    Higher score = more anomalous. The reconstruction term picks up patterns
    the decoder cannot reproduce, while the KL term picks up patterns whose
    encoded posterior is far from the learned prior.
    """
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(x)
        recon_err = ((recon - x) ** 2).mean(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        score = recon_err + beta * kl
    return score
