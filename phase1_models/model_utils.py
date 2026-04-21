from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

CANONICAL_MODEL_ROOT = "models_full_run"
CANONICAL_ARTIFACT_ROOT = "model_full_run_artifacts"
READY_MODEL_ROOT = "phase1_ready_models"
LEGACY_MODEL_ROOT = "models_legacy_smoke"
LEGACY_ARTIFACT_ROOT = "model_legacy_smoke_artifacts"
MODEL_DISPLAY_NAMES = {
    "threshold_baseline": "Threshold Baseline",
    "autoencoder": "Autoencoder",
    "isolation_forest": "Isolation Forest",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "llm_baseline": "Tokenized LLM-Style Baseline",
    "detector_only": "Detector Only",
    "detector_plus_context": "Detector + Context",
    "detector_plus_context_plus_token": "Detector + Context + Token",
}


@dataclass(slots=True)
class ModelPaths:
    model_name: str
    model_dir: Path
    artifact_dir: Path


def ensure_model_paths(
    model_name: str,
    project_root: str | Path | None = None,
    model_root: str = LEGACY_MODEL_ROOT,
    artifact_root: str = LEGACY_ARTIFACT_ROOT,
) -> ModelPaths:
    root = Path(project_root) if project_root is not None else ROOT
    model_dir = root / "outputs" / model_root / model_name
    artifact_dir = root / "outputs" / "reports" / artifact_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return ModelPaths(model_name=model_name, model_dir=model_dir, artifact_dir=artifact_dir)


def write_json(payload: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_pickle(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_torch_state(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def write_predictions(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def parameter_count(model: object) -> int:
    if hasattr(model, "parameters"):
        return int(sum(parameter.numel() for parameter in model.parameters()))
    if hasattr(model, "estimators_"):
        estimators = getattr(model, "estimators_", [])
        return int(sum(getattr(estimator, "tree_", None).node_count for estimator in estimators if hasattr(estimator, "tree_")))
    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"))
        return int(coef.size)
    return 0


def memory_estimate_mb(model: object) -> float:
    return float(parameter_count(model) * 4 / (1024.0 ** 2))


def display_model_name(model_name: object) -> str:
    key = str(model_name)
    return MODEL_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def apply_model_display_names(frame: pd.DataFrame, columns: tuple[str, ...] = ("model_name",)) -> pd.DataFrame:
    formatted = frame.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(display_model_name)
    return formatted


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
