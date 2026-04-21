"""Offline deployment-runtime support for DERGuardian.

This module implements build alert packet logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import jsonschema

from deployment_runtime.runtime_common import (
    PACKAGE_ROOT,
    infer_assets_from_feature_names,
    json_ready,
    make_alert_id,
    normalize_timestamp,
    severity_from_ratio,
    timestamp_to_z,
)


DEFAULT_SCHEMA_PATH = PACKAGE_ROOT / "alert_packet_schema.json"


def load_alert_packet_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    """Load alert packet schema for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    resolved = Path(schema_path or DEFAULT_SCHEMA_PATH)
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_alert_packet(packet: dict[str, Any], schema_path: str | Path | None = None) -> None:
    """Validate alert packet for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    jsonschema.validate(instance=json_ready(packet), schema=load_alert_packet_schema(schema_path))


def build_alert_packet(
    *,
    site_id: str,
    packet_source: str,
    model_name: str,
    model_profile: str,
    score: float,
    threshold: float,
    modality: str,
    window_start_utc: object,
    window_end_utc: object,
    top_features: list[dict[str, Any]],
    local_context_path: str | Path,
    metadata: dict[str, Any] | None = None,
    explanation_packet_path: str | Path | None = None,
    alert_id: str | None = None,
) -> dict[str, Any]:
    """Build alert packet for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    start = normalize_timestamp(window_start_utc)
    end = normalize_timestamp(window_end_utc)
    affected_assets = infer_assets_from_feature_names([str(item.get("feature", "")) for item in top_features])
    asset_id = affected_assets[0] if affected_assets else None
    payload = {
        "alert_id": alert_id or make_alert_id(f"{site_id}-{model_profile}", end),
        "timestamp_utc": timestamp_to_z(end),
        "site_id": site_id,
        "packet_source": packet_source,
        "model_name": model_name,
        "model_profile": model_profile,
        "score": float(score),
        "threshold": float(threshold),
        "severity": severity_from_ratio(float(score), float(threshold)),
        "modality": modality,
        "asset_id": asset_id,
        "affected_assets": affected_assets,
        "top_features": top_features,
        "window_start_utc": timestamp_to_z(start),
        "window_end_utc": timestamp_to_z(end),
        "local_context_path": str(local_context_path),
        "buffer_id": str(Path(local_context_path).stem),
        "explanation_packet_path": str(explanation_packet_path) if explanation_packet_path else None,
        "metadata": metadata or {},
    }
    validate_alert_packet(payload)
    return payload
