"""Offline deployment-runtime support for DERGuardian.

This module implements load deployed models logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle
import time

import numpy as np
import pandas as pd
import torch

from deployment_runtime.runtime_common import (
    ModelRuntimeProfile,
    load_runtime_config,
    normalize_timestamp,
    resolve_repo_path,
    select_model_profiles,
)
from phase1_models.neural_models import MLPAutoencoder


@dataclass(slots=True)
class PredictionDetails:
    """Structured object used by the offline deployment-runtime workflow."""

    score: float
    threshold: float
    predicted: bool
    top_features: list[dict[str, Any]]
    feature_row: dict[str, float]
    raw_score: float
    inference_ms: float
    reference_found: bool


class ReferenceWindowStore:
    """Structured object used by the offline deployment-runtime workflow."""

    def __init__(self, frame: pd.DataFrame) -> None:
        if "window_start_utc" not in frame.columns:
            raise ValueError("Reference windows must contain `window_start_utc`.")
        prepared = frame.copy()
        prepared["window_start_utc"] = pd.to_datetime(prepared["window_start_utc"], utc=True)
        self.frame = prepared.sort_values("window_start_utc").reset_index(drop=True)
        self.lookup = {
            normalize_timestamp(row["window_start_utc"]): row.to_dict()
            for _, row in self.frame.iterrows()
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "ReferenceWindowStore":
        resolved = resolve_repo_path(path)
        suffix = resolved.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(resolved)
        elif suffix == ".csv":
            frame = pd.read_csv(resolved)
        else:
            raise ValueError(f"Unsupported reference window format: {resolved}")
        return cls(frame)

    def get(self, window_start_utc: object) -> dict[str, Any] | None:
        start = normalize_timestamp(window_start_utc)
        if start in self.lookup:
            return self.lookup[start]
        for offset_seconds in (1, -1):
            candidate = start + pd.Timedelta(seconds=offset_seconds)
            if candidate in self.lookup:
                return self.lookup[candidate]
        return None


class BaseDeployedModel:
    """Structured object used by the offline deployment-runtime workflow."""

    def __init__(
        self,
        profile: ModelRuntimeProfile,
        model_info: dict[str, Any],
        standardizer,
        calibrator,
        reference_store: ReferenceWindowStore,
    ) -> None:
        self.profile = profile
        self.model_info = model_info
        self.standardizer = standardizer
        self.calibrator = calibrator
        self.reference_store = reference_store
        self.feature_columns = list(model_info.get("feature_columns", []))
        self.threshold = float(model_info.get("threshold", 0.0))
        self.modality = str(model_info.get("feature_mode", profile.feature_mode))

    def predict(self, window_record: dict[str, Any]) -> PredictionDetails:
        raise NotImplementedError

    def build_feature_row(self, window_record: dict[str, Any], reference_record: dict[str, Any]) -> dict[str, float]:
        feature_row: dict[str, float] = {}
        for feature in self.feature_columns:
            if feature.startswith("delta__"):
                base_feature = feature[len("delta__") :]
                current_value = _safe_float(window_record.get(base_feature, 0.0))
                reference_value = _safe_float(reference_record.get(base_feature, 0.0))
                feature_row[feature] = current_value - reference_value
            else:
                feature_row[feature] = _safe_float(window_record.get(feature, 0.0))
        return feature_row

    def standardize(self, feature_row: dict[str, float]) -> np.ndarray:
        values = np.asarray([float(feature_row.get(column, 0.0)) for column in self.feature_columns], dtype=np.float32)
        mean = np.asarray(self.standardizer.mean, dtype=np.float32)
        std = np.asarray(self.standardizer.std, dtype=np.float32)
        std = np.where(std < 1e-8, 1.0, std)
        standardized = (values - mean) / std
        return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

    def apply_calibrator(self, raw_score: float) -> float:
        if self.calibrator is None:
            return float(raw_score)
        calibrated = self.calibrator.predict_proba(np.asarray([[float(raw_score)]], dtype=np.float32))[:, 1]
        return float(calibrated[0])

    def _reference_for_window(self, window_record: dict[str, Any]) -> dict[str, Any] | None:
        return self.reference_store.get(window_record["window_start_utc"])

    def _base_top_feature_record(
        self,
        feature_name: str,
        contribution: float,
        window_record: dict[str, Any],
        reference_record: dict[str, Any],
    ) -> dict[str, Any]:
        signal_name = feature_name[len("delta__") :] if feature_name.startswith("delta__") else feature_name
        if "__" in signal_name:
            signal, aggregation = signal_name.rsplit("__", 1)
        else:
            signal, aggregation = signal_name, "value"
        window_value = _safe_float(window_record.get(signal_name, window_record.get(feature_name, 0.0)))
        reference_value = _safe_float(reference_record.get(signal_name, reference_record.get(feature_name, 0.0)))
        residual = float(window_value - reference_value) if feature_name.startswith("delta__") else float(window_value)
        return {
            "feature": feature_name,
            "signal": signal,
            "aggregation": aggregation,
            "contribution": round(float(contribution), 6),
            "window_value": round(window_value, 6),
            "reference_value": round(reference_value, 6),
            "residual": round(residual, 6),
        }


class ThresholdBaselineDeployedModel(BaseDeployedModel):
    """Structured object used by the offline deployment-runtime workflow."""

    def predict(self, window_record: dict[str, Any]) -> PredictionDetails:
        reference_record = self._reference_for_window(window_record)
        if reference_record is None:
            return PredictionDetails(
                score=0.0,
                threshold=self.threshold,
                predicted=False,
                top_features=[],
                feature_row={},
                raw_score=0.0,
                inference_ms=0.0,
                reference_found=False,
            )
        feature_row = self.build_feature_row(window_record, reference_record)
        started = time.perf_counter()
        standardized = self.standardize(feature_row)
        raw_score = float(np.mean(np.abs(standardized)))
        calibrated_score = self.apply_calibrator(raw_score)
        inference_ms = (time.perf_counter() - started) * 1000.0
        contributions = np.abs(standardized)
        ordering = np.argsort(contributions)[::-1][:5]
        top_features = [
            self._base_top_feature_record(self.feature_columns[idx], float(contributions[idx]), window_record, reference_record)
            for idx in ordering
        ]
        return PredictionDetails(
            score=calibrated_score,
            threshold=self.threshold,
            predicted=bool(calibrated_score >= self.threshold),
            top_features=top_features,
            feature_row=feature_row,
            raw_score=raw_score,
            inference_ms=inference_ms,
            reference_found=True,
        )


class AutoencoderDeployedModel(BaseDeployedModel):
    """Structured object used by the offline deployment-runtime workflow."""

    def __init__(
        self,
        profile: ModelRuntimeProfile,
        model_info: dict[str, Any],
        standardizer,
        calibrator,
        reference_store: ReferenceWindowStore,
        model: MLPAutoencoder,
    ) -> None:
        super().__init__(profile, model_info, standardizer, calibrator, reference_store)
        self.model = model.eval()

    def predict(self, window_record: dict[str, Any]) -> PredictionDetails:
        reference_record = self._reference_for_window(window_record)
        if reference_record is None:
            return PredictionDetails(
                score=0.0,
                threshold=self.threshold,
                predicted=False,
                top_features=[],
                feature_row={},
                raw_score=0.0,
                inference_ms=0.0,
                reference_found=False,
            )
        feature_row = self.build_feature_row(window_record, reference_record)
        started = time.perf_counter()
        standardized = self.standardize(feature_row)
        batch = torch.tensor(standardized[None, :], dtype=torch.float32)
        with torch.no_grad():
            reconstruction = self.model(batch).cpu().numpy()[0]
        per_feature_error = np.square(reconstruction - standardized)
        raw_score = float(per_feature_error.mean())
        calibrated_score = self.apply_calibrator(raw_score)
        inference_ms = (time.perf_counter() - started) * 1000.0
        ordering = np.argsort(per_feature_error)[::-1][:5]
        top_features = [
            self._base_top_feature_record(self.feature_columns[idx], float(per_feature_error[idx]), window_record, reference_record)
            for idx in ordering
        ]
        for idx, item in zip(ordering, top_features):
            item["reconstruction_error"] = round(float(per_feature_error[idx]), 6)
            item["standardized_value"] = round(float(standardized[idx]), 6)
            item["reconstructed_value"] = round(float(reconstruction[idx]), 6)
        return PredictionDetails(
            score=calibrated_score,
            threshold=self.threshold,
            predicted=bool(calibrated_score >= self.threshold),
            top_features=top_features,
            feature_row=feature_row,
            raw_score=raw_score,
            inference_ms=inference_ms,
            reference_found=True,
        )


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return numeric


def _load_threshold_model(profile: ModelRuntimeProfile) -> ThresholdBaselineDeployedModel:
    payload = _load_pickle(profile.artifact_dir / "model.pkl")
    reference_store = ReferenceWindowStore.from_path(profile.reference_windows_path)
    model_info = payload.get("model_info", {})
    return ThresholdBaselineDeployedModel(
        profile=profile,
        model_info=model_info,
        standardizer=payload["standardizer"],
        calibrator=payload.get("calibrator"),
        reference_store=reference_store,
    )


def _load_autoencoder_model(profile: ModelRuntimeProfile) -> AutoencoderDeployedModel:
    payload = _load_pickle(profile.artifact_dir / "metadata.pkl")
    state_dict = torch.load(profile.artifact_dir / "model.pt", map_location="cpu")
    input_dim = int(state_dict["encoder.0.weight"].shape[1])
    hidden_dim = int(state_dict["encoder.0.weight"].shape[0])
    bottleneck_dim = int(state_dict["encoder.2.weight"].shape[0])
    model = MLPAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
    model.load_state_dict(state_dict)
    reference_store = ReferenceWindowStore.from_path(profile.reference_windows_path)
    model_info = payload.get("model_info", {})
    return AutoencoderDeployedModel(
        profile=profile,
        model_info=model_info,
        standardizer=payload["standardizer"],
        calibrator=payload.get("calibrator"),
        reference_store=reference_store,
        model=model,
    )


def expected_artifacts_for_profiles(profiles: list[ModelRuntimeProfile]) -> dict[str, list[str]]:
    """Handle expected artifacts for profiles within the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    expectations: dict[str, list[str]] = {}
    for profile in profiles:
        required = [
            str(profile.reference_windows_path),
            str(profile.artifact_dir),
        ]
        if profile.model_name == "threshold_baseline":
            required.append(str(profile.artifact_dir / "model.pkl"))
            required.append(str(profile.artifact_dir / "results.json"))
        elif profile.model_name == "autoencoder":
            required.append(str(profile.artifact_dir / "metadata.pkl"))
            required.append(str(profile.artifact_dir / "model.pt"))
            required.append(str(profile.artifact_dir / "results.json"))
        expectations[profile.runtime_name] = required
    return expectations


def load_deployed_models(
    config_path: str | Path | None = None,
    model_mode: str = "threshold_only",
) -> tuple[dict[str, BaseDeployedModel], list[str]]:
    """Load deployed models for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    config = load_runtime_config(config_path)
    profiles = select_model_profiles(config, model_mode=model_mode)
    loaded: dict[str, BaseDeployedModel] = {}
    warnings: list[str] = []
    for profile in profiles:
        try:
            if profile.model_name == "threshold_baseline":
                loaded[profile.runtime_name] = _load_threshold_model(profile)
            elif profile.model_name == "autoencoder":
                loaded[profile.runtime_name] = _load_autoencoder_model(profile)
            else:
                warnings.append(f"Unsupported deployed model `{profile.model_name}` for profile `{profile.runtime_name}`.")
        except FileNotFoundError as exc:
            warnings.append(f"Missing artifact for `{profile.runtime_name}`: {exc}")
        except Exception as exc:
            warnings.append(f"Unable to load `{profile.runtime_name}` from {profile.artifact_dir}: {exc}")
    return loaded, warnings
