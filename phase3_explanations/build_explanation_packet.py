"""Build evidence-grounded explanation packets for Phase 3 alerts.

The packet joins detector predictions with clean/attacked window deltas, cyber
events, scenario metadata, feature importance, and conservative candidate-family
hints. It produces post-alert operator support only; it is not a human-like
root-cause-analysis system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3_explanations.classify_attack_family import classify_from_packet
from phase3_explanations.shared import (
    DEFAULT_PATHS,
    build_safe_operator_actions,
    confidence_band,
    ensure_dir,
    humanize_feature_name,
    infer_assets_from_signals,
    json_ready,
    load_attack_labels,
    load_cyber_events,
    load_scenario_lookup,
    load_table,
    normalize_timestamp,
    read_json,
    resolve_repo_path,
    severity_from_score,
    timestamp_to_z,
    write_json,
)


META_COLUMNS = {"window_start_utc", "window_end_utc", "window_seconds", "scenario_id", "run_id", "split_id", "attack_family"}
PHYSICAL_PREFIXES = ("feeder_", "pv_", "bess_", "bus_", "regulator_", "capacitor_", "switch_", "line_", "derived_")
CYBER_PREFIXES = ("cyber_",)


def select_incident_row(
    predictions: pd.DataFrame,
    scenario_id: str | None = None,
    window_start: str | None = None,
    incident_id: str | None = None,
) -> pd.Series:
    """Select the prediction row that should seed one explanation packet."""

    frame = predictions.copy()
    frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
    frame["window_end_utc"] = pd.to_datetime(frame["window_end_utc"], utc=True)
    if window_start:
        match_ts = normalize_timestamp(window_start)
        matched = frame[frame["window_start_utc"] == match_ts]
        if matched.empty:
            raise ValueError(f"No prediction row found for window_start_utc={window_start}")
        return matched.sort_values("score", ascending=False).iloc[0]
    if incident_id:
        token = incident_id.replace("incident-", "")
        matched = frame[frame["window_start_utc"].astype(str).str.contains(token, regex=False)]
        if not matched.empty:
            return matched.sort_values("score", ascending=False).iloc[0]
    if scenario_id:
        matched = frame[frame["scenario_id"] == scenario_id]
        if matched.empty:
            raise ValueError(f"No prediction rows found for scenario_id={scenario_id}")
        return matched.sort_values(["predicted", "score"], ascending=False).iloc[0]
    return frame.sort_values(["predicted", "score"], ascending=False).iloc[0]


def match_window_row(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Handle match window row within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    subset = frame.copy()
    subset["window_start_utc"] = pd.to_datetime(subset["window_start_utc"], utc=True)
    subset["window_end_utc"] = pd.to_datetime(subset["window_end_utc"], utc=True)
    matched = subset[(subset["window_start_utc"] == start) & (subset["window_end_utc"] == end)]
    if matched.empty:
        raise ValueError(f"Unable to match attacked/clean window for {start.isoformat()} -> {end.isoformat()}")
    return matched.iloc[0]


def load_feature_importance(path: str | Path | None) -> dict[str, dict[str, float]]:
    """Load feature importance for the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    resolved = resolve_repo_path(path or DEFAULT_PATHS["feature_importance"])
    if not resolved.exists():
        return {}
    frame = pd.read_csv(resolved)
    lookup: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        lookup[str(row["feature"])] = {
            "effect_size": float(row.get("effect_size", 0.0)),
            "correlation_abs": float(row.get("correlation_abs", 0.0)),
        }
    return lookup


def compute_top_features(
    attacked_row: pd.Series,
    clean_row: pd.Series,
    feature_lookup: dict[str, dict[str, float]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank physical/cyber window features by local attacked-vs-clean shift."""

    records: list[dict[str, Any]] = []
    numeric_columns = [
        column
        for column in attacked_row.index
        if column not in META_COLUMNS and pd.api.types.is_number(attacked_row[column]) and pd.api.types.is_number(clean_row[column])
    ]
    for column in numeric_columns:
        attacked_value = float(attacked_row[column])
        baseline_value = float(clean_row[column])
        delta = attacked_value - baseline_value
        scale = max(abs(attacked_value), abs(baseline_value), 1.0)
        relative_change = delta / scale
        importance = feature_lookup.get(f"delta__{column}", {})
        score = abs(relative_change) * (1.0 + importance.get("effect_size", 0.0))
        if score < 0.02:
            continue
        records.append(
            {
                "feature": f"delta__{column}",
                "signal": column.rsplit("__", 1)[0] if "__" in column else column,
                "aggregation": column.rsplit("__", 1)[1] if "__" in column else "value",
                "attacked_value": round(attacked_value, 6),
                "baseline_value": round(baseline_value, 6),
                "delta": round(delta, 6),
                "relative_change": round(relative_change, 6),
                "ranking_score": round(score, 6),
                "global_effect_size": round(importance.get("effect_size", 0.0), 6),
                "description": humanize_feature_name(f"delta__{column}"),
            }
        )
    records.sort(key=lambda item: item["ranking_score"], reverse=True)
    return records[:top_k]


def preferred_feature_candidates(signal: str) -> list[str]:
    """Handle preferred feature candidates within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return [f"{signal}__mean", f"{signal}__last", f"{signal}__max", f"{signal}__min", signal]


def pick_relevant_feature(top_features: list[dict[str, Any]], signal: str) -> dict[str, Any] | None:
    """Handle pick relevant feature within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    for candidate in preferred_feature_candidates(signal):
        for item in top_features:
            if item.get("signal") == candidate.replace("__mean", "").replace("__last", "").replace("__max", "").replace("__min", ""):
                return item
            if item.get("feature") == f"delta__{candidate}":
                return item
    for item in top_features:
        if str(item.get("signal", "")).startswith(signal):
            return item
    return None


def feature_record_from_rows(attacked_row: pd.Series, clean_row: pd.Series, signal: str) -> dict[str, Any] | None:
    """Handle feature record from rows within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    for candidate in preferred_feature_candidates(signal):
        if candidate not in attacked_row.index or candidate not in clean_row.index:
            continue
        attacked_value = float(attacked_row[candidate])
        baseline_value = float(clean_row[candidate])
        delta = attacked_value - baseline_value
        scale = max(abs(attacked_value), abs(baseline_value), 1.0)
        relative_change = delta / scale
        return {
            "feature": f"delta__{candidate}",
            "signal": signal,
            "aggregation": candidate.rsplit("__", 1)[1] if "__" in candidate else "value",
            "attacked_value": round(attacked_value, 6),
            "baseline_value": round(baseline_value, 6),
            "delta": round(delta, 6),
            "relative_change": round(relative_change, 6),
            "description": humanize_feature_name(f"delta__{candidate}"),
        }
    return None


def categorize_physical_signal(signal: str) -> str:
    """Handle categorize physical signal within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if signal.startswith("feeder_") and "losses" in signal:
        return "losses_increase"
    if signal.startswith("feeder_"):
        return "feeder_power_shift"
    if "terminal_v" in signal:
        return "telemetry_mismatch"
    if signal.startswith("pv_") and ("available_kw" in signal or "curtailment" in signal):
        return "curtailment_inconsistency"
    if signal.startswith("pv_"):
        return "pv_dispatch_drop"
    if signal.startswith("bess_") and "soc" in signal:
        return "soc_inconsistency"
    if signal.startswith("bess_"):
        return "bess_power_jump"
    if signal.startswith("regulator_"):
        return "regulator_tap_deviation"
    if signal.startswith("bus_") and "angle_deg" in signal:
        return "voltage_angle_shift"
    if signal.startswith("bus_"):
        return "voltage_profile_shift"
    return "physical_signal_shift"


def build_physical_evidence(
    attacked_row: pd.Series,
    clean_row: pd.Series,
    top_features: list[dict[str, Any]],
    scenario_label: dict[str, Any] | None,
    cyber_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble physical evidence records grounded in window-level deltas."""

    evidence: list[dict[str, Any]] = []
    referenced_signals: list[str] = []
    if scenario_label:
        metadata = scenario_label.get("causal_metadata", {}) if isinstance(scenario_label.get("causal_metadata"), dict) else {}
        referenced_signals.extend(str(item) for item in metadata.get("observable_signals", []))
        referenced_signals.extend(str(item) for item in scenario_label.get("affected_signals", []))
    referenced_signals.extend(str(item.get("resource")) for item in cyber_evidence if item.get("resource"))

    seen_pairs: set[tuple[str, str]] = set()
    for signal in dict.fromkeys(referenced_signals):
        feature = feature_record_from_rows(attacked_row, clean_row, signal) or pick_relevant_feature(top_features, signal)
        if feature:
            pair = (signal, feature["aggregation"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            evidence.append(
                {
                    "asset": infer_assets_from_signals([signal])[0] if infer_assets_from_signals([signal]) else "feeder",
                    "signal": signal,
                    "category": categorize_physical_signal(signal),
                    "aggregation": feature["aggregation"],
                    "observed_value": feature["attacked_value"],
                    "baseline_value": feature["baseline_value"],
                    "delta": feature["delta"],
                    "relative_change": feature["relative_change"],
                    "direction": "increase" if float(feature["delta"]) >= 0 else "decrease",
                    "source_feature": feature["feature"],
                }
            )

    used_signals = {item["signal"] for item in evidence}
    for feature in top_features:
        signal = str(feature["signal"])
        if not signal.startswith(PHYSICAL_PREFIXES) or signal in used_signals:
            continue
        asset = infer_assets_from_signals([signal])[0] if infer_assets_from_signals([signal]) else "feeder"
        if scenario_label and asset not in {"feeder"}:
            known_assets = set(str(item) for item in scenario_label.get("affected_assets", [])) if isinstance(scenario_label.get("affected_assets"), list) else set()
            observable_assets = set(infer_assets_from_signals(referenced_signals))
            # Prefer scenario-grounded evidence before falling back to global
            # high-delta features; this avoids unsupported asset claims.
            if asset not in known_assets and asset not in observable_assets:
                continue
        pair = (signal, feature["aggregation"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        evidence.append(
            {
                "asset": asset,
                "signal": signal,
                "category": categorize_physical_signal(signal),
                "aggregation": feature["aggregation"],
                "observed_value": feature["attacked_value"],
                "baseline_value": feature["baseline_value"],
                "delta": feature["delta"],
                "relative_change": feature["relative_change"],
                "direction": "increase" if float(feature["delta"]) >= 0 else "decrease",
                "source_feature": feature["feature"],
            }
        )
        if len(evidence) >= 6:
            break
    return evidence[:6]


def build_cyber_evidence(
    cyber: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    lookback_seconds: int,
) -> list[dict[str, Any]]:
    """Collect cyber events that overlap the alert window or short lookback."""

    lookback_start = window_start - pd.Timedelta(seconds=int(lookback_seconds))
    relevant = cyber[(cyber["timestamp_utc"] >= lookback_start) & (cyber["timestamp_utc"] <= window_end)].copy()
    if relevant.empty:
        return []

    evidence: list[dict[str, Any]] = []
    for _, row in relevant[relevant.get("attack_flag", 0).fillna(0) == 1].iterrows():
        evidence.append(
            {
                "kind": "attack_event",
                "timestamp_utc": timestamp_to_z(row["timestamp_utc"]),
                "family": row.get("attack_family"),
                "actor": row.get("actor"),
                "protocol": row.get("protocol"),
                "resource": row.get("resource") or row.get("command_type"),
                "campaign_id": row.get("campaign_id"),
                "detail": row.get("notes") or f"Attack-tagged {row.get('event_type')} event observed.",
            }
        )

    auth_failures = relevant[
        (relevant.get("event_type") == "authentication")
        & ((relevant.get("auth_result") == "failure") | (relevant.get("actor") == "intruder"))
    ]
    for _, row in auth_failures.head(2).iterrows():
        evidence.append(
            {
                "kind": "auth_failure",
                "timestamp_utc": timestamp_to_z(row["timestamp_utc"]),
                "actor": row.get("actor"),
                "protocol": row.get("protocol"),
                "detail": row.get("notes") or "Authentication precursor or failure observed before the alert window.",
            }
        )

    suspicious_commands = relevant[
        (relevant.get("event_type").isin(["command", "attack"]))
        & ((relevant.get("attack_flag", 0).fillna(0) == 1) | (relevant.get("actor") == "intruder"))
    ]
    for _, row in suspicious_commands.head(4).iterrows():
        evidence.append(
            {
                "kind": "suspicious_command",
                "timestamp_utc": timestamp_to_z(row["timestamp_utc"]),
                "actor": row.get("actor"),
                "protocol": row.get("protocol"),
                "resource": row.get("resource") or row.get("command_type"),
                "command_type": row.get("command_type"),
                "campaign_id": row.get("campaign_id"),
                "detail": row.get("notes") or "Command-path evidence overlaps the incident lookback window.",
            }
        )

    config_changes = relevant[(relevant.get("event_type") == "configuration") & (relevant.get("attack_flag", 0).fillna(0) == 1)]
    for _, row in config_changes.head(2).iterrows():
        evidence.append(
            {
                "kind": "config_change",
                "timestamp_utc": timestamp_to_z(row["timestamp_utc"]),
                "actor": row.get("actor"),
                "protocol": row.get("protocol"),
                "resource": row.get("resource"),
                "campaign_id": row.get("campaign_id"),
                "detail": row.get("notes") or "Configuration-path change overlaps the alert window.",
            }
        )

    for _, row in suspicious_commands.head(2).iterrows():
        resource = str(row.get("resource") or row.get("command_type") or "")
        if any(token in resource for token in ("terminal_v", "telemetry", "measured", "register")):
            evidence.append(
                {
                    "kind": "telemetry_corruption_flag",
                    "timestamp_utc": timestamp_to_z(row["timestamp_utc"]),
                    "actor": row.get("actor"),
                    "protocol": row.get("protocol"),
                    "resource": resource,
                    "campaign_id": row.get("campaign_id"),
                    "detail": "A telemetry-facing resource was written during the incident lookback window.",
                }
            )
    return evidence[:8]


def build_detector_candidate_hints(top_features: list[dict[str, Any]], physical_evidence: list[dict[str, Any]], cyber_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build detector candidate hints for the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    hints: list[dict[str, Any]] = []
    top_feature_names = {item["feature"] for item in top_features[:6]}
    categories = {item["category"] for item in physical_evidence}
    cyber_kinds = {item["kind"] for item in cyber_evidence}
    control_assets = {
        str(item.get("asset"))
        for item in physical_evidence
        if item.get("asset") and not str(item.get("asset")).lower().startswith(("bus", "feeder", "substation"))
    }

    if any("terminal_v_pu" in name for name in top_feature_names) and "telemetry_corruption_flag" in cyber_kinds:
        hints.append({"label": "false_data_injection", "weight": 0.25, "reason": "Telemetry-facing voltage feature moved with cyber telemetry evidence."})
    if "bess_power_jump" in categories and "suspicious_command" in cyber_kinds:
        hints.append({"label": "unauthorized_command", "weight": 0.3, "reason": "BESS dispatch changed alongside suspicious command evidence."})
    if "curtailment_inconsistency" in categories:
        hints.append({"label": "curtailment_inconsistency", "weight": 0.25, "reason": "PV availability and curtailment features diverged materially."})
    if len(control_assets) >= 2:
        hints.append({"label": "coordinated_campaign", "weight": 0.22, "reason": "Multiple assets appear in the strongest evidence list."})
    if "attack_event" in cyber_kinds and any(item.get("campaign_id") for item in cyber_evidence) and len(control_assets) >= 2:
        hints.append({"label": "coordinated_campaign", "weight": 0.18, "reason": "Campaign-tagged cyber evidence is present."})
    return hints


def find_overlapping_label(labels: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, scenario_id: str | None) -> dict[str, Any] | None:
    """Handle find overlapping label within the Phase 3 grounded explanation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    overlap = labels[(labels["start_time_utc"] <= end) & (labels["end_time_utc"] >= start)]
    if scenario_id:
        scenario_match = overlap[overlap["scenario_id"] == scenario_id]
        if not scenario_match.empty:
            overlap = scenario_match
    if overlap.empty:
        return None
    row = overlap.iloc[0]
    return json_ready(row.to_dict())


def build_packet(
    predictions_path: str | Path = DEFAULT_PATHS["threshold_predictions"],
    results_path: str | Path = DEFAULT_PATHS["threshold_results"],
    attacked_windows_path: str | Path = DEFAULT_PATHS["attacked_windows"],
    clean_windows_path: str | Path = DEFAULT_PATHS["clean_windows"],
    cyber_events_path: str | Path = DEFAULT_PATHS["cyber_events"],
    attack_labels_path: str | Path = DEFAULT_PATHS["attack_labels"],
    scenario_manifest_path: str | Path = DEFAULT_PATHS["scenario_manifest"],
    feature_importance_path: str | Path = DEFAULT_PATHS["feature_importance"],
    model_name: str = "threshold_baseline",
    scenario_id: str | None = None,
    window_start: str | None = None,
    incident_id: str | None = None,
    top_k_features: int = 10,
    cyber_lookback_seconds: int = 900,
) -> dict[str, Any]:
    """Create one explanation packet from existing detector and evidence files."""

    predictions = load_table(predictions_path)
    attacked_windows = load_table(attacked_windows_path)
    clean_windows = load_table(clean_windows_path)
    cyber = load_cyber_events(cyber_events_path)
    labels = load_attack_labels(attack_labels_path)
    scenario_lookup = load_scenario_lookup(scenario_manifest_path)
    feature_lookup = load_feature_importance(feature_importance_path)
    results = read_json(results_path)

    selected = select_incident_row(predictions, scenario_id=scenario_id, window_start=window_start, incident_id=incident_id)
    window_start_ts = normalize_timestamp(selected["window_start_utc"])
    window_end_ts = normalize_timestamp(selected["window_end_utc"])
    attacked_row = match_window_row(attacked_windows, window_start_ts, window_end_ts)
    clean_row = match_window_row(clean_windows, window_start_ts, window_end_ts)
    active_label = find_overlapping_label(labels, window_start_ts, window_end_ts, str(selected.get("scenario_id") or scenario_id or ""))
    scenario_meta = scenario_lookup.get(str(selected.get("scenario_id")), None)

    top_features = compute_top_features(attacked_row, clean_row, feature_lookup, top_k=top_k_features)
    cyber_evidence = build_cyber_evidence(cyber, window_start_ts, window_end_ts, cyber_lookback_seconds)
    physical_evidence = build_physical_evidence(attacked_row, clean_row, top_features, active_label, cyber_evidence)
    candidate_hints = build_detector_candidate_hints(top_features, physical_evidence, cyber_evidence)

    affected_assets = []
    if active_label and isinstance(active_label.get("affected_assets"), list):
        affected_assets.extend(str(item) for item in active_label["affected_assets"])
    affected_assets.extend(infer_assets_from_signals([item["signal"] for item in physical_evidence]))
    affected_assets.extend(infer_assets_from_signals([str(item.get("resource") or item.get("command_type") or "") for item in cyber_evidence]))
    affected_assets = list(dict.fromkeys(asset for asset in affected_assets if asset))

    threshold = float(results.get("metrics", {}).get("threshold", 0.0))
    score = float(selected["score"])
    packet = {
        "incident_id": incident_id or f"incident-{model_name}-{window_start_ts.strftime('%Y%m%dT%H%M%SZ')}",
        "window": {
            "start_time_utc": timestamp_to_z(window_start_ts),
            "end_time_utc": timestamp_to_z(window_end_ts),
            "duration_seconds": int((window_end_ts - window_start_ts).total_seconds()),
            "cyber_lookback_seconds": int(cyber_lookback_seconds),
        },
        "model_name": model_name,
        "anomaly_score": round(score, 6),
        "threshold": round(threshold, 6),
        "score_margin": round(score - threshold, 6),
        "predicted_alert": bool(int(selected.get("predicted", 0))),
        "severity": severity_from_score(score, threshold),
        "affected_assets": affected_assets,
        "top_features": top_features,
        "physical_evidence": physical_evidence,
        "cyber_evidence": cyber_evidence,
        "detector_evidence": {
            "prediction_source": str(resolve_repo_path(predictions_path)),
            "results_source": str(resolve_repo_path(results_path)),
            "modality_type": "fused",
            "candidate_family_hints": candidate_hints,
            "family_hint_confidence": confidence_band(max((item["weight"] for item in candidate_hints), default=0.0)),
        },
        "offline_evaluation": {
            "scenario_label": active_label,
            "scenario_metadata": scenario_meta,
        },
    }
    preclassification = classify_from_packet(packet)
    packet["candidate_families"] = preclassification["candidate_families"]
    packet["detector_evidence"]["operator_action_starter"] = build_safe_operator_actions(packet)
    return packet


def main() -> None:
    """CLI entrypoint for writing a grounded explanation packet JSON file."""

    parser = argparse.ArgumentParser(description="Build a grounded explanation packet from existing detector outputs and structured evidence.")
    parser.add_argument("--predictions", default=str(DEFAULT_PATHS["threshold_predictions"]))
    parser.add_argument("--results", default=str(DEFAULT_PATHS["threshold_results"]))
    parser.add_argument("--attacked-windows", default=str(DEFAULT_PATHS["attacked_windows"]))
    parser.add_argument("--clean-windows", default=str(DEFAULT_PATHS["clean_windows"]))
    parser.add_argument("--cyber-events", default=str(DEFAULT_PATHS["cyber_events"]))
    parser.add_argument("--attack-labels", default=str(DEFAULT_PATHS["attack_labels"]))
    parser.add_argument("--scenario-manifest", default=str(DEFAULT_PATHS["scenario_manifest"]))
    parser.add_argument("--feature-importance", default=str(DEFAULT_PATHS["feature_importance"]))
    parser.add_argument("--model-name", default="threshold_baseline")
    parser.add_argument("--scenario-id")
    parser.add_argument("--window-start")
    parser.add_argument("--incident-id")
    parser.add_argument("--top-k-features", type=int, default=10)
    parser.add_argument("--cyber-lookback-seconds", type=int, default=900)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "reports" / "explanation_artifacts" / "explanation_packet.json"))
    args = parser.parse_args()

    packet = build_packet(
        predictions_path=args.predictions,
        results_path=args.results,
        attacked_windows_path=args.attacked_windows,
        clean_windows_path=args.clean_windows,
        cyber_events_path=args.cyber_events,
        attack_labels_path=args.attack_labels,
        scenario_manifest_path=args.scenario_manifest,
        feature_importance_path=args.feature_importance,
        model_name=args.model_name,
        scenario_id=args.scenario_id,
        window_start=args.window_start,
        incident_id=args.incident_id,
        top_k_features=args.top_k_features,
        cyber_lookback_seconds=args.cyber_lookback_seconds,
    )
    ensure_dir(Path(args.output).parent)
    output_path = write_json(packet, args.output)
    print(json.dumps(json_ready(packet), indent=2))
    print(f"Explanation packet written to {output_path}")


if __name__ == "__main__":
    main()
