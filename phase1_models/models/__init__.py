"""New baseline models added in the Phase 1 hardening pass.

Each module exposes a model class and helper scoring functions consistent
with the contract used by the rest of the detector suite: scores are
higher-is-more-anomalous and thresholds are calibrated on validation benign
scores at the 99th percentile by default.
"""

from phase1_models.models.lstm_autoencoder import LSTMAutoencoder, lstm_autoencoder_score
from phase1_models.models.vae import VAE, vae_loss, vae_anomaly_score
from phase1_models.models.ocsvm import OCSVMConfig, OCSVMDetector
from phase1_models.models.anomaly_transformer import (
    AnomalyTransformer,
    anomaly_transformer_score,
)

__all__ = [
    "LSTMAutoencoder",
    "lstm_autoencoder_score",
    "VAE",
    "vae_loss",
    "vae_anomaly_score",
    "OCSVMConfig",
    "OCSVMDetector",
    "AnomalyTransformer",
    "anomaly_transformer_score",
]
