"""One-Class SVM wrapper for non-deep classical anomaly detection.

This is the canonical classical baseline for novelty detection. It models the
support of the benign distribution with an RBF kernel and flags samples that
fall outside this support. Reviewers commonly request this baseline because it
is well-understood, deterministic given a seed, and does not require GPU.

Reference:
    Schoelkopf et al., "Estimating the support of a high-dimensional
    distribution," Neural Computation, 13(7), 1443-1471, 2001.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.svm import OneClassSVM


@dataclass
class OCSVMConfig:
    """Configuration for OneClassSVM training.

    Attributes:
        kernel: kernel type. RBF is the default and works well for high-
            dimensional residual features.
        gamma: kernel coefficient. 'scale' is 1 / (n_features * X.var()),
            which adapts to the feature scale automatically.
        nu: an upper bound on the fraction of training errors and a lower
            bound on the fraction of support vectors. Should be set near the
            expected fraction of outliers in the benign training set.
    """

    kernel: str = "rbf"
    gamma: str | float = "scale"
    nu: float = 0.05


class OCSVMDetector:
    """Lightweight wrapper around sklearn OneClassSVM for the Phase 1 pipeline.

    The wrapper provides a fit() that trains on benign-only data and a score()
    that returns higher-is-more-anomalous scores, matching the contract used
    by the rest of the detector suite.
    """

    def __init__(self, config: OCSVMConfig | None = None) -> None:
        self.config = config or OCSVMConfig()
        self.model: OneClassSVM | None = None

    def fit(self, x_benign: np.ndarray) -> None:
        """Fit the OCSVM on benign-only training data.

        Args:
            x_benign: array of shape (n_samples, n_features). Should contain
                only benign windows.
        """
        self.model = OneClassSVM(
            kernel=self.config.kernel,
            gamma=self.config.gamma,
            nu=self.config.nu,
        )
        self.model.fit(x_benign)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Higher = more anomalous.

        sklearn's decision_function returns positive values for inliers and
        negative for outliers; we negate to follow the project convention.
        """
        if self.model is None:
            raise RuntimeError("OCSVMDetector.fit() must be called before score().")
        return -self.model.decision_function(x).astype(float)

    def predict(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary predictions using a calibrated threshold.

        Anomalies are samples whose score exceeds the threshold (typically
        the 99th percentile of validation benign scores).
        """
        scores = self.score(x)
        return (scores >= threshold).astype(int)
