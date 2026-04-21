from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json
import math
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "phase3_explanations"
OUTPUT_ROOT = ROOT / "outputs" / "reports" / "explanation_artifacts"

ATTACK_FAMILY_TAXONOMY = [
    "false_data_injection",
    "unauthorized_command",
    "command_suppression",
    "telemetry_corruption",
    "curtailment_inconsistency",
    "coordinated_campaign",
    "physical_dispatch_anomaly",
    "cyber_physical_mismatch",
    "unknown_anomaly",
]

SAFE_OPERATOR_PREFIXES = (
    "inspect",
    "verify",
    "compare",
    "review",
    "check",
    "correlate",
    "confirm",
    "monitor",
    "validate",
    "collect",
    "escalate",
    "notify",
    "document",
)

UNSAFE_ACTION_TERMS = (
    "exploit",
    "inject",
    "pivot",
    "exfiltrate",
    "disable protection",
    "trip the feeder",
    "open the breaker",
    "disconnect customers",
    "shut down the grid",
    "destroy",
    "override controls",
    "bypass",
    "replay attack",
    "execute attack",
)

DEFAULT_PATHS = {
    "attacked_windows": ROOT / "outputs" / "windows" / "merged_windows_attacked.parquet",
    "clean_windows": ROOT / "outputs" / "windows" / "merged_windows_clean.parquet",
    "cyber_events": ROOT / "outputs" / "attacked" / "cyber_events.parquet",
    "attack_labels": ROOT / "outputs" / "attacked" / "attack_labels.parquet",
    "scenario_manifest": ROOT / "outputs" / "attacked" / "scenario_manifest.json",
    "feature_importance": ROOT / "outputs" / "reports" / "model_full_run_artifacts" / "feature_importance_table.csv",
    "threshold_predictions": ROOT / "outputs" / "models_full_run" / "threshold_baseline" / "predictions.parquet",
    "threshold_results": ROOT / "outputs" / "models_full_run" / "threshold_baseline" / "results.json",
}

ASSET_TOKEN_PATTERN = re.compile(r"\b(?:pv\d+|bess\d+|creg\d+[a-z]?|c\d+[a-z]?|sw\d+|bus[_ ]?\d+|feeder head|substation)\b", re.IGNORECASE)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    directory = resolve_repo_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


def write_json(payload: Any, path: str | Path) -> Path:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    return output_path


def write_text(text: str, path: str | Path) -> Path:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def load_table(path: str | Path) -> pd.DataFrame:
    resolved = resolve_repo_path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(resolved)
    if suffix == ".csv":
        return pd.read_csv(resolved)
    raise ValueError(f"Unsupported table format: {resolved}")


def parse_object(value: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return value
    return value


def load_attack_labels(path: str | Path | None = None) -> pd.DataFrame:
    labels = load_table(path or DEFAULT_PATHS["attack_labels"]).copy()
    for column in ("start_time_utc", "end_time_utc"):
        labels[column] = pd.to_datetime(labels[column], utc=True)
    for column in ("affected_assets", "affected_signals", "causal_metadata"):
        if column in labels.columns:
            labels[column] = labels[column].apply(parse_object)
    return labels


def load_cyber_events(path: str | Path | None = None) -> pd.DataFrame:
    cyber = load_table(path or DEFAULT_PATHS["cyber_events"]).copy()
    for column in ("timestamp_utc", "ingest_timestamp_utc"):
        if column in cyber.columns:
            cyber[column] = pd.to_datetime(cyber[column], utc=True)
    return cyber


def normalize_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def timestamp_to_z(value: object) -> str:
    return normalize_timestamp(value).isoformat().replace("+00:00", "Z")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()
            if converted is not value:
                return json_ready(converted)
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return timestamp_to_z(value)
    if isinstance(value, pd.Series):
        return {str(key): json_ready(item) for key, item in value.items()}
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


def load_scenario_lookup(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    manifest = read_json(path or DEFAULT_PATHS["scenario_manifest"])
    scenarios = manifest.get("applied_scenarios", [])
    if not scenarios:
        scenarios = read_json(ROOT / "phase2" / "research_attack_scenarios.json").get("scenarios", [])
    return {str(item["scenario_id"]): item for item in scenarios}


def infer_assets_from_signals(signals: list[str]) -> list[str]:
    assets: list[str] = []
    for signal in signals:
        parts = signal.split("_")
        if len(parts) >= 2 and parts[0] in {"pv", "bess", "regulator", "capacitor", "switch", "bus"}:
            if parts[0] == "bus":
                asset = f"bus{parts[1]}"
            else:
                asset = parts[1]
            if asset not in assets:
                assets.append(asset)
    return assets


def asset_mentions(text: str) -> list[str]:
    mentions = []
    for match in ASSET_TOKEN_PATTERN.findall(text or ""):
        normalized = match.lower().replace(" ", "")
        if normalized.startswith("bus_"):
            normalized = normalized.replace("_", "")
        mentions.append(normalized)
    return sorted(set(mentions))


def humanize_feature_name(feature: str) -> str:
    cleaned = feature.replace("delta__", "").replace("__", " / ")
    cleaned = cleaned.replace("_", " ")
    replacements = {
        "p kw": "P kW",
        "q kvar": "Q kvar",
        "soc": "SOC",
        "pu": "p.u.",
        "std": "std",
        "mean": "mean",
        "min": "min",
        "max": "max",
        "last": "last",
    }
    lowered = cleaned.lower()
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return lowered


def severity_from_score(score: float, threshold: float) -> str:
    if threshold <= 0:
        return "medium"
    ratio = score / threshold
    if ratio >= 8.0:
        return "critical"
    if ratio >= 4.0:
        return "high"
    if ratio >= 1.5:
        return "medium"
    return "low"


def confidence_band(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def allowed_operator_action(action: str) -> bool:
    text = action.strip().lower()
    if not text:
        return False
    if any(term in text for term in UNSAFE_ACTION_TERMS):
        return False
    return text.startswith(SAFE_OPERATOR_PREFIXES)


def build_safe_operator_actions(packet: dict[str, Any]) -> list[str]:
    assets = packet.get("affected_assets", [])
    top_asset = assets[0] if assets else "the affected DER asset"
    actions = [
        f"Inspect command logs for {top_asset} and verify that recent issued setpoints match the approved schedule.",
        "Compare operator-facing telemetry against local device logs or historian data for the same window.",
        "Review authentication, protocol source, and actor identity for suspicious writes or campaign-tagged events.",
        "Verify DER dispatch policy, curtailment policy, or charge-discharge schedule against expected operating conditions.",
    ]
    if len(assets) >= 2:
        actions.append("Escalate the event as a potentially coordinated anomaly because multiple assets show concurrent disturbance evidence.")
    else:
        actions.append("Monitor adjacent feeder assets for spillover effects while the incident is triaged.")
    return [action for action in actions if allowed_operator_action(action)]


def extract_allowed_assets_and_signals(packet: dict[str, Any]) -> tuple[set[str], set[str]]:
    assets = {str(item).lower() for item in packet.get("affected_assets", [])}
    signals: set[str] = set()
    for section in ("physical_evidence", "cyber_evidence", "top_features"):
        for item in packet.get(section, []):
            if isinstance(item, dict):
                asset = item.get("asset")
                if asset:
                    assets.add(str(asset).lower())
                signal = item.get("signal") or item.get("resource") or item.get("feature")
                if signal:
                    signals.add(str(signal))
                    for inferred in infer_assets_from_signals([str(signal)]):
                        assets.add(str(inferred).lower())
    scenario = packet.get("offline_evaluation", {}).get("scenario_label") or {}
    for asset in scenario.get("affected_assets", []) if isinstance(scenario.get("affected_assets"), list) else []:
        assets.add(str(asset).lower())
    for signal in scenario.get("affected_signals", []) if isinstance(scenario.get("affected_signals"), list) else []:
        signals.add(str(signal))
        for inferred in infer_assets_from_signals([str(signal)]):
            assets.add(str(inferred).lower())
    return assets, signals
