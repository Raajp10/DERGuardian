from __future__ import annotations

import numpy as np


def quantile_thresholds(scores: np.ndarray, quantiles: tuple[float, ...] = (0.95, 0.99, 0.995)) -> dict[str, float]:
    values = np.asarray(scores).astype(float)
    return {f"p{int(q * 1000) / 10:g}": float(np.quantile(values, q)) for q in quantiles}


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores).astype(float) >= float(threshold)).astype(int)
