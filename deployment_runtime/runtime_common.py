"""Offline deployment-runtime support for DERGuardian.

This module implements runtime common logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import uuid

import numpy as np
import pandas as pd

from common.io_utils import read_dataframe, write_dataframe


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "deployment_runtime"
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config_runtime.json"
DEFAULT_RUNTIME_OUTPUT_ROOT = ROOT / "outputs" / "deployment_runtime"
DEFAULT_REPORT_ROOT = ROOT / "outputs" / "reports" / "deployment_runtime"

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(slots=True)
class ModelRuntimeProfile:
    """Structured object used by the offline deployment-runtime workflow."""

    runtime_name: str
    model_name: str
    artifact_dir: Path
    reference_windows_path: Path
    attacked_windows_path: Path | None
    predictions_path: Path | None
    results_path: Path | None
    window_seconds: int
    step_seconds: int
    deployment_role: str
    description: str
    enabled: bool = True
    feature_mode: str = "aligned_residual"


def resolve_repo_path(path: str | Path) -> Path:
    """Handle resolve repo path within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Handle ensure dir within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    directory = resolve_repo_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    """Read json for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


def write_json(payload: Any, path: str | Path) -> Path:
    """Write json for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    return output_path


def write_text(text: str, path: str | Path) -> Path:
    """Write text for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def load_table(path: str | Path) -> pd.DataFrame:
    """Load table for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return read_dataframe(resolve_repo_path(path))


def normalize_timestamp(value: object) -> pd.Timestamp:
    """Handle normalize timestamp within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def timestamp_to_z(value: object) -> str:
    """Handle timestamp to z within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return normalize_timestamp(value).isoformat().replace("+00:00", "Z")


def json_ready(value: Any) -> Any:
    """Handle json ready within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return timestamp_to_z(value)
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, pd.Series):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def parse_jsonish(value: object) -> object:
    """Handle parse jsonish within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def infer_assets_from_feature_names(feature_names: list[str]) -> list[str]:
    """Handle infer assets from feature names within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    assets: list[str] = []
    for name in feature_names:
        cleaned = str(name).replace("delta__", "")
        tokens = cleaned.split("__", 1)[0].split("_")
        if len(tokens) < 2:
            continue
        prefix = tokens[0]
        asset = tokens[1]
        if prefix == "bus":
            asset = f"bus{asset}"
        if prefix in {"pv", "bess", "bus", "regulator", "capacitor", "switch"} and asset not in assets:
            assets.append(asset)
    return assets


def severity_from_ratio(score: float, threshold: float) -> str:
    """Handle severity from ratio within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if threshold <= 0.0:
        return "medium"
    ratio = float(score) / float(threshold)
    if ratio >= 6.0:
        return "critical"
    if ratio >= 3.0:
        return "high"
    if ratio >= 1.25:
        return "medium"
    return "low"


def max_severity(values: list[str]) -> str:
    """Handle max severity within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if not values:
        return "low"
    return max(values, key=lambda item: SEVERITY_ORDER.get(str(item).lower(), -1))


def flatten_dict(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Handle flatten dict within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return {f"{prefix}{key}": value for key, value in payload.items()}


def make_alert_id(prefix: str, timestamp: pd.Timestamp | str | object) -> str:
    """Handle make alert id within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return f"{prefix}-{normalize_timestamp(timestamp).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def load_runtime_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load runtime config for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    resolved = resolve_repo_path(config_path or DEFAULT_CONFIG_PATH)
    config = read_json(resolved)
    config["config_path"] = str(resolved)
    config["output_root"] = str(resolve_repo_path(config.get("output_root", DEFAULT_RUNTIME_OUTPUT_ROOT)))
    config["report_root"] = str(resolve_repo_path(config.get("report_root", DEFAULT_REPORT_ROOT)))
    return config


def select_model_profiles(config: dict[str, Any], model_mode: str = "threshold_only") -> list[ModelRuntimeProfile]:
    """Select model profiles for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    raw_profiles = config.get("model_profiles", {})
    selected_names: list[str]
    if model_mode == "threshold_only":
        selected_names = ["threshold_baseline"]
    elif model_mode == "autoencoder_only":
        selected_names = ["autoencoder_zero_day"]
    elif model_mode == "threshold_plus_autoencoder":
        selected_names = ["threshold_baseline", "autoencoder_zero_day"]
    else:
        raise ValueError(f"Unsupported model_mode `{model_mode}`.")

    selected_profiles: list[ModelRuntimeProfile] = []
    for runtime_name in selected_names:
        payload = raw_profiles.get(runtime_name)
        if not payload:
            continue
        selected_profiles.append(
            ModelRuntimeProfile(
                runtime_name=runtime_name,
                model_name=str(payload.get("model_name", runtime_name)),
                artifact_dir=resolve_repo_path(payload["artifact_dir"]),
                reference_windows_path=resolve_repo_path(payload["reference_windows_path"]),
                attacked_windows_path=resolve_repo_path(payload["attacked_windows_path"]) if payload.get("attacked_windows_path") else None,
                predictions_path=resolve_repo_path(payload["predictions_path"]) if payload.get("predictions_path") else None,
                results_path=resolve_repo_path(payload["results_path"]) if payload.get("results_path") else None,
                window_seconds=int(payload["window_seconds"]),
                step_seconds=int(payload["step_seconds"]),
                deployment_role=str(payload.get("deployment_role", "")),
                description=str(payload.get("description", "")),
                enabled=bool(payload.get("enabled", True)),
                feature_mode=str(payload.get("feature_mode", "aligned_residual")),
            )
        )
    return [profile for profile in selected_profiles if profile.enabled]


def runtime_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Handle runtime output paths within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    output_root = ensure_dir(config.get("output_root", DEFAULT_RUNTIME_OUTPUT_ROOT))
    report_root = ensure_dir(config.get("report_root", DEFAULT_REPORT_ROOT))
    return {
        "output_root": output_root,
        "edge_root": ensure_dir(output_root / "edge"),
        "gateway_root": ensure_dir(output_root / "gateway"),
        "control_center_root": ensure_dir(output_root / "control_center"),
        "reports_root": report_root,
    }


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Handle dataframe from records within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return pd.DataFrame(records) if records else pd.DataFrame()


def write_records_csv(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write records csv for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    frame = dataframe_from_records(records)
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame()
    write_dataframe(frame, output_path, fmt="csv")
    return output_path
