"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements thresholds logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

import numpy as np


def quantile_thresholds(scores: np.ndarray, quantiles: tuple[float, ...] = (0.95, 0.99, 0.995)) -> dict[str, float]:
    """Handle quantile thresholds within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    values = np.asarray(scores).astype(float)
    return {f"p{int(q * 1000) / 10:g}": float(np.quantile(values, q)) for q in quantiles}


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Handle apply threshold within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return (np.asarray(scores).astype(float) >= float(threshold)).astype(int)
