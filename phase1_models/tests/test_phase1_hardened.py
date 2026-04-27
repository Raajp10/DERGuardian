"""Smoke tests for the four new Phase 1 baseline detectors.

These tests verify shapes, gradient flow, and basic threshold-calibration
behavior without requiring the real DERGuardian dataset. Run with:

    pytest tests/test_phase1_hardened.py -v

The tests are deliberately lightweight (CPU-only, < 30 seconds total) so
they can be wired into a CI pipeline that gates merges to main.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1_models.models.lstm_autoencoder import LSTMAutoencoder, lstm_autoencoder_score
from phase1_models.models.vae import VAE, vae_anomaly_score, vae_loss
from phase1_models.models.ocsvm import OCSVMConfig, OCSVMDetector
from phase1_models.models.anomaly_transformer import (
    AnomalyTransformer,
    anomaly_transformer_score,
)


@pytest.fixture
def benign_2d() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic benign + anomalous tabular data in shape (N, F)."""
    rng = np.random.default_rng(42)
    benign = rng.normal(0.0, 1.0, size=(200, 32)).astype(np.float32)
    anomalous = rng.normal(3.0, 1.5, size=(50, 32)).astype(np.float32)
    return benign, anomalous


@pytest.fixture
def benign_3d() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic benign + anomalous sequence data in shape (N, T, F)."""
    rng = np.random.default_rng(43)
    benign = rng.normal(0.0, 1.0, size=(100, 8, 16)).astype(np.float32)
    anomalous = rng.normal(2.5, 1.2, size=(25, 8, 16)).astype(np.float32)
    return benign, anomalous


# ---------------------- LSTM-Autoencoder ----------------------

def test_lstm_autoencoder_shape(benign_3d):
    """The LSTM-AE must output the same shape it was given."""
    benign, _ = benign_3d
    model = LSTMAutoencoder(input_dim=16, hidden_dim=32, latent_dim=8)
    x = torch.tensor(benign[:4])
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_lstm_autoencoder_gradient_flow(benign_3d):
    """Loss must be differentiable and gradients must reach all parameters."""
    benign, _ = benign_3d
    model = LSTMAutoencoder(input_dim=16, hidden_dim=32, latent_dim=8)
    x = torch.tensor(benign[:8])
    recon = model(x)
    loss = ((recon - x) ** 2).mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_lstm_autoencoder_score_separates_anomalies(benign_3d):
    """After 200 epochs of fitting on benign-only data, anomalous sequences
    should produce strictly higher mean reconstruction error than benign."""
    benign, anomalous = benign_3d
    model = LSTMAutoencoder(input_dim=16, hidden_dim=32, latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x_train = torch.tensor(benign)
    for _ in range(200):
        optimizer.zero_grad()
        recon = model(x_train)
        loss = ((recon - x_train) ** 2).mean()
        loss.backward()
        optimizer.step()
    benign_score = lstm_autoencoder_score(model, torch.tensor(benign[:25])).mean().item()
    anom_score = lstm_autoencoder_score(model, torch.tensor(anomalous)).mean().item()
    assert anom_score > benign_score, (
        f"LSTM-AE failed to separate: benign={benign_score:.3f} anom={anom_score:.3f}"
    )


# ---------------------- VAE ----------------------

def test_vae_shape_and_outputs(benign_2d):
    """VAE forward must return three tensors with the right shapes."""
    benign, _ = benign_2d
    model = VAE(input_dim=32, hidden_dim=64, latent_dim=16)
    x = torch.tensor(benign[:4])
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_vae_loss_components(benign_2d):
    """vae_loss returns nonnegative recon and KL components."""
    benign, _ = benign_2d
    model = VAE(input_dim=32, hidden_dim=64, latent_dim=16)
    x = torch.tensor(benign[:8])
    recon, mu, logvar = model(x)
    total, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=0.1)
    assert recon_loss.item() >= 0.0
    assert kl_loss.item() >= 0.0


def test_vae_score_separates_anomalies(benign_2d):
    """After training on benign data, anomalies should score higher."""
    benign, anomalous = benign_2d
    model = VAE(input_dim=32, hidden_dim=64, latent_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x_train = torch.tensor(benign)
    for _ in range(150):
        optimizer.zero_grad()
        recon, mu, logvar = model(x_train)
        total, _, _ = vae_loss(recon, x_train, mu, logvar, beta=0.1)
        total.backward()
        optimizer.step()
    benign_score = vae_anomaly_score(model, torch.tensor(benign[:25])).mean().item()
    anom_score = vae_anomaly_score(model, torch.tensor(anomalous)).mean().item()
    assert anom_score > benign_score, (
        f"VAE failed to separate: benign={benign_score:.3f} anom={anom_score:.3f}"
    )


# ---------------------- OCSVM ----------------------

def test_ocsvm_fit_score_predict(benign_2d):
    """OCSVM should fit, score, and predict without raising."""
    benign, anomalous = benign_2d
    detector = OCSVMDetector(OCSVMConfig(nu=0.05))
    detector.fit(benign)
    benign_scores = detector.score(benign)
    anom_scores = detector.score(anomalous)
    threshold = float(np.quantile(benign_scores, 0.99))
    pred = detector.predict(anomalous, threshold)
    assert pred.shape == (anomalous.shape[0],)
    assert anom_scores.mean() > benign_scores.mean()


def test_ocsvm_raises_when_not_fitted(benign_2d):
    """Calling score() before fit() must raise a clear error."""
    benign, _ = benign_2d
    detector = OCSVMDetector()
    with pytest.raises(RuntimeError):
        detector.score(benign)


# ---------------------- AnomalyTransformer ----------------------

def test_anomaly_transformer_shape(benign_3d):
    """The anomaly transformer must reconstruct sequences of equal shape."""
    benign, _ = benign_3d
    model = AnomalyTransformer(input_dim=16, d_model=32, nhead=4, num_layers=2, max_seq_len=16)
    x = torch.tensor(benign[:4])
    out = model(x)
    assert out.shape == x.shape


def test_anomaly_transformer_score_separates_anomalies(benign_3d):
    """After training on benign data, anomalies should score higher."""
    benign, anomalous = benign_3d
    model = AnomalyTransformer(input_dim=16, d_model=32, nhead=4, num_layers=2, max_seq_len=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train = torch.tensor(benign)
    for _ in range(150):
        optimizer.zero_grad()
        recon = model(x_train)
        loss = ((recon - x_train) ** 2).mean()
        loss.backward()
        optimizer.step()
    benign_score = anomaly_transformer_score(model, torch.tensor(benign[:25])).mean().item()
    anom_score = anomaly_transformer_score(model, torch.tensor(anomalous)).mean().item()
    assert anom_score > benign_score, (
        f"AnomalyTransformer failed to separate: benign={benign_score:.3f} anom={anom_score:.3f}"
    )


# ---------------------- Checkpointing ----------------------

def test_checkpoint_round_trip(tmp_path):
    """A checkpoint should round-trip through save_checkpoint/load_checkpoint."""
    from phase1_models.checkpointing import (
        CheckpointState,
        load_checkpoint,
        save_checkpoint,
        utcnow_iso,
    )

    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = CheckpointState(
        epoch=3,
        train_loss=0.5,
        val_loss=0.42,
        val_metric=-0.42,
        best_metric=-0.42,
        is_best=True,
        timestamp_utc=utcnow_iso(),
        seed=1,
        model_name="test_linear",
        config={"hidden": 4},
    )
    ckpt_path = save_checkpoint(model, optimizer, state, tmp_path)
    assert ckpt_path.exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "training_history.jsonl").exists()

    new_model = torch.nn.Linear(8, 4)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    restored = load_checkpoint(ckpt_path, new_model, new_optimizer)
    assert restored.epoch == 3
    assert restored.is_best is True
    assert restored.model_name == "test_linear"

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
