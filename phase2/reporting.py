"""Phase 2 reporting and analysis artifacts.

These reports are analysis outputs layered on top of the canonical executable
Phase 2 path. They do not change the schema-validated JSON -> compile actions
-> OpenDSS rerun when needed -> attacked outputs pipeline. Instead, they make
the resulting attacked bundle easier to audit, cite, and interpret in papers.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ProjectPaths

ANALYSIS_NUMERIC_EPSILON = 1e-6
ANALYSIS_TOP_K = 5
LAYER_METADATA_COLUMNS = {
    "timestamp_utc",
    "scenario_id",
    "simulation_index",
    "split_id",
    "run_id",
    "source_layer",
}
CYBER_GROUP_KEYS = ("event_type", "protocol", "asset", "observable_signal")


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    return int(round(_safe_float(value)))


def _coerce_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("("):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [text]
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return list(parsed)
        return [text]
    if hasattr(value, "tolist"):
        listed = value.tolist()
        if isinstance(listed, list):
            return listed
        return [listed]
    if isinstance(value, set):
        return list(value)
    return [value]


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, float) and pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}
    return {}


def _normalise_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    frame = labels_df.copy()
    for column in ("affected_assets", "observable_signals", "affected_layers"):
        if column in frame.columns:
            frame[column] = frame[column].apply(_coerce_sequence)
    if "causal_metadata" in frame.columns:
        frame["causal_metadata"] = frame["causal_metadata"].apply(_coerce_mapping)
    return frame


def _load_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("Scenario payload must include a scenarios list.")
    return scenarios


def _scenario_duration_seconds(scenario: dict[str, Any], label_row: pd.Series) -> int:
    if scenario.get("duration_seconds") is not None:
        return _safe_int(scenario.get("duration_seconds"))
    if label_row.get("duration_seconds") is not None:
        return _safe_int(label_row.get("duration_seconds"))
    if scenario.get("duration_minutes") is not None:
        return _safe_int(scenario.get("duration_minutes")) * 60
    return 0


def _resolve_target_column(target_component: str, target_asset: str, target_signal: str) -> tuple[str, str]:
    component = target_component.strip().lower()
    signal = target_signal.strip()
    asset = target_asset.strip()
    if component == "measured_layer":
        if asset.startswith("pv"):
            prefix = "pv"
        elif asset.startswith("bess"):
            prefix = "bess"
        elif asset.startswith("creg") or asset.startswith("reg"):
            prefix = "regulator"
        elif asset.startswith("c"):
            prefix = "capacitor"
        elif asset.startswith("sw"):
            prefix = "switch"
        elif asset.startswith("bus"):
            prefix = "bus"
        elif asset.startswith("line") or asset.startswith("l"):
            prefix = "line"
        elif asset.startswith("feeder"):
            prefix = "feeder"
        else:
            prefix = asset
        return "measured", f"{prefix}_{asset}_{signal}"
    return "truth", f"{component}_{asset}_{signal}"


def _subset_interval(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    interval_mask = (frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] <= end)
    return frame.loc[interval_mask].copy()


def _align_interval_for_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    aligned["comparison_timestamp_utc"] = pd.to_datetime(aligned["timestamp_utc"], utc=True).dt.floor("s")
    return aligned.sort_values("comparison_timestamp_utc").drop_duplicates("comparison_timestamp_utc", keep="last")


def _common_numeric_columns(clean_df: pd.DataFrame, attacked_df: pd.DataFrame) -> list[str]:
    common_columns = [column for column in clean_df.columns if column in attacked_df.columns and column not in LAYER_METADATA_COLUMNS]
    numeric_columns: list[str] = []
    for column in common_columns:
        if pd.api.types.is_numeric_dtype(clean_df[column]) and pd.api.types.is_numeric_dtype(attacked_df[column]):
            numeric_columns.append(column)
    return numeric_columns


def _summarise_numeric_layer(
    clean_df: pd.DataFrame,
    attacked_df: pd.DataFrame,
    start: str,
    end: str,
    *,
    epsilon: float,
) -> dict[str, Any]:
    clean_interval = _subset_interval(clean_df, start, end)
    attacked_interval = _subset_interval(attacked_df, start, end)
    if clean_interval.empty or attacked_interval.empty:
        return {
            "changed": False,
            "row_count": 0,
            "channel_count": 0,
            "changed_channel_count": 0,
            "mean_abs_delta": 0.0,
            "median_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "top_changed_signals": [],
            "top_changed_signal_stats": [],
        }

    clean_aligned = _align_interval_for_comparison(clean_interval)
    attacked_aligned = _align_interval_for_comparison(attacked_interval)
    merged = clean_aligned.merge(attacked_aligned, on="comparison_timestamp_utc", suffixes=("_clean", "_attacked"), how="inner")
    common_columns = _common_numeric_columns(clean_interval, attacked_interval)
    column_stats: list[dict[str, Any]] = []
    for column in common_columns:
        clean_column = pd.to_numeric(merged[f"{column}_clean"], errors="coerce").fillna(0.0)
        attacked_column = pd.to_numeric(merged[f"{column}_attacked"], errors="coerce").fillna(0.0)
        abs_delta = (attacked_column - clean_column).abs()
        max_abs_delta = float(abs_delta.max()) if not abs_delta.empty else 0.0
        column_stats.append(
            {
                "signal": column,
                "mean_abs_delta": float(abs_delta.mean()) if not abs_delta.empty else 0.0,
                "median_abs_delta": float(abs_delta.median()) if not abs_delta.empty else 0.0,
                "max_abs_delta": max_abs_delta,
                "changed": max_abs_delta > epsilon,
            }
        )

    changed_stats = [item for item in column_stats if item["changed"]]
    ordered_stats = sorted(changed_stats, key=lambda item: item["max_abs_delta"], reverse=True)
    if changed_stats:
        mean_abs_delta = float(sum(item["mean_abs_delta"] for item in changed_stats) / len(changed_stats))
        median_abs_delta = float(sum(item["median_abs_delta"] for item in changed_stats) / len(changed_stats))
        max_abs_delta = float(max(item["max_abs_delta"] for item in changed_stats))
    else:
        mean_abs_delta = 0.0
        median_abs_delta = 0.0
        max_abs_delta = 0.0

    return {
        "changed": bool(changed_stats),
        "row_count": int(len(merged)),
        "channel_count": int(len(common_columns)),
        "changed_channel_count": int(len(changed_stats)),
        "mean_abs_delta": mean_abs_delta,
        "median_abs_delta": median_abs_delta,
        "max_abs_delta": max_abs_delta,
        "top_changed_signals": [item["signal"] for item in ordered_stats[:ANALYSIS_TOP_K]],
        "top_changed_signal_stats": ordered_stats[:ANALYSIS_TOP_K],
    }


def _summarise_target_column(
    clean_df: pd.DataFrame,
    attacked_df: pd.DataFrame,
    start: str,
    end: str,
    target_column: str,
    *,
    epsilon: float,
) -> dict[str, Any]:
    if target_column not in clean_df.columns or target_column not in attacked_df.columns:
        return {
            "target_column": target_column,
            "target_column_available": False,
            "changed_as_intended": False,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }

    clean_interval = _subset_interval(clean_df[["timestamp_utc", target_column]], start, end)
    attacked_interval = _subset_interval(attacked_df[["timestamp_utc", target_column]], start, end)
    clean_aligned = _align_interval_for_comparison(clean_interval)
    attacked_aligned = _align_interval_for_comparison(attacked_interval)
    merged = clean_aligned.merge(attacked_aligned, on="comparison_timestamp_utc", suffixes=("_clean", "_attacked"), how="inner")
    if merged.empty:
        return {
            "target_column": target_column,
            "target_column_available": True,
            "changed_as_intended": False,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }

    clean_series = pd.to_numeric(merged[f"{target_column}_clean"], errors="coerce")
    attacked_series = pd.to_numeric(merged[f"{target_column}_attacked"], errors="coerce")
    if clean_series.notna().all() and attacked_series.notna().all():
        abs_delta = (attacked_series - clean_series).abs()
        mean_abs_delta = float(abs_delta.mean()) if not abs_delta.empty else 0.0
        max_abs_delta = float(abs_delta.max()) if not abs_delta.empty else 0.0
        changed = max_abs_delta > epsilon
    else:
        clean_text = merged[f"{target_column}_clean"].astype(str)
        attacked_text = merged[f"{target_column}_attacked"].astype(str)
        changed = bool((clean_text != attacked_text).any())
        mean_abs_delta = 1.0 if changed else 0.0
        max_abs_delta = mean_abs_delta
    return {
        "target_column": target_column,
        "target_column_available": True,
        "changed_as_intended": changed,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
    }


def _summarise_cyber_layer(clean_df: pd.DataFrame, attacked_df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    clean_interval = _subset_interval(clean_df, start, end)
    attacked_interval = _subset_interval(attacked_df, start, end)
    clean_event_count = int(len(clean_interval))
    attacked_event_count = int(len(attacked_interval))
    if not attacked_interval.empty and "attack_flag" in attacked_interval.columns:
        attack_event_count = int(pd.to_numeric(attacked_interval["attack_flag"], errors="coerce").fillna(0).sum())
    else:
        attack_event_count = 0

    clean_grouped = (
        clean_interval.groupby(list(CYBER_GROUP_KEYS)).size().to_dict()
        if not clean_interval.empty and all(key in clean_interval.columns for key in CYBER_GROUP_KEYS)
        else {}
    )
    attacked_grouped = (
        attacked_interval.groupby(list(CYBER_GROUP_KEYS)).size().to_dict()
        if not attacked_interval.empty and all(key in attacked_interval.columns for key in CYBER_GROUP_KEYS)
        else {}
    )
    changed_group_items: list[dict[str, Any]] = []
    all_keys = set(clean_grouped) | set(attacked_grouped)
    for key in all_keys:
        clean_count = int(clean_grouped.get(key, 0))
        attacked_count = int(attacked_grouped.get(key, 0))
        delta_count = attacked_count - clean_count
        if delta_count != 0:
            event_type, protocol, asset, observable_signal = key
            changed_group_items.append(
                {
                    "event_type": str(event_type),
                    "protocol": str(protocol),
                    "asset": str(asset),
                    "observable_signal": str(observable_signal),
                    "delta_count": int(delta_count),
                }
            )
    changed_group_items.sort(key=lambda item: abs(item["delta_count"]), reverse=True)
    changed = bool(attack_event_count > 0 or attacked_event_count != clean_event_count or changed_group_items)
    return {
        "changed": changed,
        "clean_event_count": clean_event_count,
        "attacked_event_count": attacked_event_count,
        "event_count_delta": int(attacked_event_count - clean_event_count),
        "attack_event_count": attack_event_count,
        "changed_event_type_count": int(len(changed_group_items)),
        "top_changed_events": changed_group_items[:ANALYSIS_TOP_K],
    }


def _extract_observable_signals(label_row: pd.Series) -> list[str]:
    values = _coerce_sequence(label_row.get("observable_signals"))
    if values:
        return [str(value) for value in values]
    causal_metadata = _coerce_mapping(label_row.get("causal_metadata"))
    return [str(value) for value in _coerce_sequence(causal_metadata.get("observable_signals"))]


def _primary_changed_signals(
    truth_summary: dict[str, Any],
    measured_summary: dict[str, Any],
    cyber_summary: dict[str, Any],
    target_column: str | None,
    observable_signals: list[str],
) -> list[str]:
    ordered_signals: list[str] = []
    if target_column:
        ordered_signals.append(target_column)
    ordered_signals.extend(str(signal) for signal in observable_signals)
    ordered_signals.extend(str(signal) for signal in truth_summary.get("top_changed_signals", []))
    ordered_signals.extend(str(signal) for signal in measured_summary.get("top_changed_signals", []))
    ordered_signals.extend(
        f"cyber:{item['event_type']}:{item['asset']}:{item['observable_signal']}"
        for item in cyber_summary.get("top_changed_events", [])
    )
    deduplicated: list[str] = []
    seen: set[str] = set()
    for signal in ordered_signals:
        if signal and signal not in seen:
            deduplicated.append(signal)
            seen.add(signal)
        if len(deduplicated) >= ANALYSIS_TOP_K:
            break
    return deduplicated


def build_phase2_coverage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    scenarios = payload.get("scenarios", [])
    family_counter = Counter(scenario.get("attack_family", scenario.get("category", "unknown")) for scenario in scenarios)
    component_counter = Counter(scenario.get("target_component", "unknown") for scenario in scenarios)
    asset_counter = Counter(scenario.get("target_asset", "unknown") for scenario in scenarios)
    durations = [_safe_int(scenario.get("duration_seconds", 0)) for scenario in scenarios]
    severity_counter = Counter(scenario.get("severity", "unknown") for scenario in scenarios)
    return {
        "scenario_count": len(scenarios),
        "family_count": len(family_counter),
        "attack_family_coverage": dict(sorted(family_counter.items())),
        "target_component_coverage": dict(sorted(component_counter.items())),
        "target_asset_coverage": dict(sorted(asset_counter.items())),
        "severity_distribution": dict(sorted(severity_counter.items())),
        "duration_range_seconds": {
            "min": min(durations) if durations else 0,
            "max": max(durations) if durations else 0,
            "median": float(pd.Series(durations).median()) if durations else 0.0,
        },
    }


def build_phase2_label_summary(labels_df: pd.DataFrame) -> dict[str, Any]:
    frame = _normalise_labels(labels_df)
    metadata_columns = [
        "scenario_id",
        "scenario_name",
        "attack_family",
        "severity",
        "start_time_utc",
        "end_time_utc",
        "affected_assets",
    ]
    completeness = {
        column: float(frame[column].notna().mean()) if column in frame.columns else 0.0 for column in metadata_columns
    }
    affected_assets_counter = Counter()
    for items in frame.get("affected_assets", pd.Series([], dtype=object)).tolist():
        for item in _coerce_sequence(items):
            affected_assets_counter[str(item)] += 1
    return {
        "label_count": int(len(frame)),
        "metadata_completeness": completeness,
        "attack_family_distribution": dict(sorted(Counter(frame.get("attack_family", [])).items())),
        "severity_distribution": dict(sorted(Counter(frame.get("severity", [])).items())),
        "affected_asset_frequency": dict(sorted(affected_assets_counter.items())),
    }


def build_phase2_effect_summary(
    payload: dict[str, Any],
    labels_df: pd.DataFrame,
    clean_truth_df: pd.DataFrame,
    attacked_truth_df: pd.DataFrame,
    clean_measured_df: pd.DataFrame,
    attacked_measured_df: pd.DataFrame,
    clean_cyber_df: pd.DataFrame,
    attacked_cyber_df: pd.DataFrame,
    *,
    epsilon: float = ANALYSIS_NUMERIC_EPSILON,
) -> list[dict[str, Any]]:
    scenarios = {scenario["scenario_id"]: scenario for scenario in _load_payload(payload)}
    labels = _normalise_labels(labels_df)
    effect_rows: list[dict[str, Any]] = []
    for _, label_row in labels.sort_values("scenario_id").iterrows():
        scenario_id = str(label_row["scenario_id"])
        scenario = scenarios.get(scenario_id, {})
        start_time = str(label_row["start_time_utc"])
        end_time = str(label_row["end_time_utc"])
        target_component = str(label_row.get("target_component", scenario.get("target_component", "unknown")))
        affected_assets = _coerce_sequence(label_row.get("affected_assets"))
        affected_signals = _coerce_sequence(label_row.get("affected_signals"))
        target_asset = str(scenario.get("target_asset", affected_assets[0] if affected_assets else "unknown"))
        target_signal = str(scenario.get("target_signal", affected_signals[0] if affected_signals else "unknown"))
        target_layer, target_column = _resolve_target_column(target_component, target_asset, target_signal)

        truth_summary = _summarise_numeric_layer(clean_truth_df, attacked_truth_df, start_time, end_time, epsilon=epsilon)
        measured_summary = _summarise_numeric_layer(clean_measured_df, attacked_measured_df, start_time, end_time, epsilon=epsilon)
        cyber_summary = _summarise_cyber_layer(clean_cyber_df, attacked_cyber_df, start_time, end_time)
        target_summary = _summarise_target_column(
            clean_measured_df if target_layer == "measured" else clean_truth_df,
            attacked_measured_df if target_layer == "measured" else attacked_truth_df,
            start_time,
            end_time,
            target_column,
            epsilon=epsilon,
        )
        affected_layers: list[str] = []
        if truth_summary["changed"]:
            affected_layers.append("truth")
        if measured_summary["changed"]:
            affected_layers.append("measured")
        if cyber_summary["changed"]:
            affected_layers.append("cyber")
        observable_signals = _extract_observable_signals(label_row)
        primary_changed_signals = _primary_changed_signals(
            truth_summary,
            measured_summary,
            cyber_summary,
            target_summary["target_column"] if target_summary["target_column_available"] else None,
            observable_signals,
        )
        effect_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": str(label_row.get("scenario_name", scenario.get("scenario_name", scenario_id))),
                "category": str(scenario.get("category", label_row.get("attack_family", "unknown"))),
                "attack_family": str(label_row.get("attack_family", scenario.get("category", "unknown"))),
                "target_component": target_component,
                "target_asset": target_asset,
                "target_signal": target_signal,
                "duration_seconds": _scenario_duration_seconds(scenario, label_row),
                "severity": str(label_row.get("severity", scenario.get("severity", "unknown"))),
                "physical_truth_changed": bool(truth_summary["changed"]),
                "measured_layer_changed": bool(measured_summary["changed"]),
                "cyber_layer_changed": bool(cyber_summary["changed"]),
                "affected_artifact_layers": affected_layers,
                "effect_presence_pass": bool(affected_layers),
                "primary_observed_changed_signals": primary_changed_signals,
                "observable_signals": observable_signals,
                "additional_target_count": len(_coerce_sequence(scenario.get("additional_targets"))),
                "target_signal_changed_as_intended": bool(target_summary["changed_as_intended"]),
                "target_signal_target_column": target_summary["target_column"],
                "total_changed_channels_above_epsilon": int(
                    _safe_int(truth_summary["changed_channel_count"])
                    + _safe_int(measured_summary["changed_channel_count"])
                    + _safe_int(cyber_summary["changed_event_type_count"])
                ),
                "target_signal_check": {
                    "target_layer": target_layer,
                    **target_summary,
                },
                "truth_change_stats": truth_summary,
                "measured_change_stats": measured_summary,
                "cyber_change_stats": cyber_summary,
            }
        )
    return effect_rows


def build_phase2_scenario_difficulty(effect_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not effect_summary:
        return []

    durations = [max(int(item.get("duration_seconds", 0)), 0) for item in effect_summary]
    max_duration = max(durations) if durations else 1

    effect_strengths: list[float] = []
    for item in effect_summary:
        truth_strength = math.log1p(_safe_float(item["truth_change_stats"]["max_abs_delta"]))
        measured_strength = math.log1p(_safe_float(item["measured_change_stats"]["max_abs_delta"]))
        cyber_strength = math.log1p(
            abs(_safe_int(item["cyber_change_stats"]["event_count_delta"]))
            + _safe_int(item["cyber_change_stats"]["attack_event_count"])
        )
        effect_strengths.append(max(truth_strength, measured_strength, cyber_strength))

    min_strength = min(effect_strengths) if effect_strengths else 0.0
    max_strength = max(effect_strengths) if effect_strengths else 1.0
    denominator = max(max_strength - min_strength, 1e-9)

    difficulty_rows: list[dict[str, Any]] = []
    for index, item in enumerate(effect_summary):
        normalized_strength = (effect_strengths[index] - min_strength) / denominator
        magnitude_factor = 1.0 - normalized_strength
        duration_factor = 1.0 - min(item["duration_seconds"] / max_duration, 1.0) if max_duration else 0.0
        observability_count = max(len(item.get("observable_signals", [])), len(item.get("primary_observed_changed_signals", [])))
        observability_factor = 1.0 - min(observability_count / 4.0, 1.0)
        changed_channel_total = (
            _safe_int(item["truth_change_stats"]["changed_channel_count"])
            + _safe_int(item["measured_change_stats"]["changed_channel_count"])
            + _safe_int(item["cyber_change_stats"]["changed_event_type_count"])
        )
        channel_spread_factor = 1.0 - min(changed_channel_total / 12.0, 1.0)
        layer_count = max(len(item.get("affected_artifact_layers", [])), 1)
        if layer_count >= 3:
            multi_layer_factor = 0.25
        elif layer_count == 2:
            multi_layer_factor = 0.6
        else:
            multi_layer_factor = 1.0
        target_directness_factor = 1.0 if _safe_int(item.get("additional_target_count", 0)) == 0 else 0.55

        # Higher scores indicate scenarios that are more likely to be subtle
        # within this repository's own attacked bundle. This is a deterministic,
        # bundle-relative heuristic for experiment interpretation, not a formal
        # stealth metric or universal attack-difficulty claim.
        difficulty_score = round(
            100.0
            * (
                0.30 * magnitude_factor
                + 0.18 * duration_factor
                + 0.16 * observability_factor
                + 0.16 * channel_spread_factor
                + 0.10 * multi_layer_factor
                + 0.10 * target_directness_factor
            ),
            2,
        )

        if difficulty_score >= 67.0:
            difficulty_band = "hard"
        elif difficulty_score >= 34.0:
            difficulty_band = "medium"
        else:
            difficulty_band = "easy"

        component_scores = {
            "magnitude_factor": round(magnitude_factor, 4),
            "duration_factor": round(duration_factor, 4),
            "observability_factor": round(observability_factor, 4),
            "channel_spread_factor": round(channel_spread_factor, 4),
            "multi_layer_factor": round(multi_layer_factor, 4),
            "target_directness_factor": round(target_directness_factor, 4),
        }
        ranked_components = sorted(component_scores.items(), key=lambda pair: pair[1], reverse=True)
        explanation = (
            f"Repository-specific difficulty heuristic: {difficulty_band} detectability because "
            f"{ranked_components[0][0]}={ranked_components[0][1]:.2f} and "
            f"{ranked_components[1][0]}={ranked_components[1][1]:.2f}. "
            f"Higher values here indicate subtler effects within this attacked bundle."
        )
        difficulty_rows.append(
            {
                "scenario_id": item["scenario_id"],
                "scenario_name": item["scenario_name"],
                "difficulty_score": difficulty_score,
                "difficulty_band": difficulty_band,
                "component_scores": component_scores,
                "rationale_fields": component_scores,
                "effect_strength_scalar": round(effect_strengths[index], 6),
                "changed_channel_total": changed_channel_total,
                "observability_signal_count": observability_count,
                "affected_layer_count": layer_count,
                "explanation": explanation,
            }
        )
    return difficulty_rows


def _render_coverage_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Phase 2 Scenario Coverage Summary",
        "",
        f"- Scenario count: {summary['scenario_count']}",
        f"- Family count: {summary['family_count']}",
        f"- Target component coverage: {summary['target_component_coverage']}",
        f"- Target asset coverage: {summary['target_asset_coverage']}",
        f"- Severity distribution: {summary['severity_distribution']}",
        f"- Duration range (seconds): {summary['duration_range_seconds']}",
    ]
    return lines


def _render_label_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Phase 2 Label Summary",
        "",
        f"- Label count: {summary['label_count']}",
        f"- Metadata completeness: {summary['metadata_completeness']}",
        f"- Attack family distribution: {summary['attack_family_distribution']}",
        f"- Severity distribution: {summary['severity_distribution']}",
        f"- Affected asset frequency: {summary['affected_asset_frequency']}",
    ]
    return lines


def _render_effect_markdown(effect_summary: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Phase 2 Effect Summary",
        "",
        "These scenario-level summaries compare attacked outputs against the clean baseline over each labeled interval.",
        "They are analysis artifacts for reporting and interpretation, not executable Phase 2 dependencies.",
        "",
    ]
    for entry in effect_summary:
        lines.extend(
            [
                f"## {entry['scenario_id']} - {entry['scenario_name']}",
                "",
                f"- Attack family: {entry['attack_family']}",
                f"- Target: {entry['target_component']} / {entry['target_asset']} / {entry['target_signal']}",
                f"- Duration seconds: {entry['duration_seconds']}",
                f"- Severity: {entry['severity']}",
                f"- Changed layers: {entry['affected_artifact_layers']}",
                f"- Effect presence pass: {entry['effect_presence_pass']}",
                f"- Primary observed changed signals: {entry['primary_observed_changed_signals']}",
                f"- Truth change stats: {entry['truth_change_stats']}",
                f"- Measured change stats: {entry['measured_change_stats']}",
                f"- Cyber change stats: {entry['cyber_change_stats']}",
                "",
            ]
        )
    return lines


def _render_difficulty_markdown(difficulty_summary: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Phase 2 Scenario Difficulty Summary",
        "",
        "This is a bounded, deterministic detectability heuristic derived from repository artifacts.",
        "It is intended for experiment interpretation and paper reporting, not as a universal stealth score.",
        "",
    ]
    for entry in difficulty_summary:
        lines.extend(
            [
                f"## {entry['scenario_id']} - {entry['scenario_name']}",
                "",
                f"- Difficulty score: {entry['difficulty_score']}",
                f"- Difficulty band: {entry['difficulty_band']}",
                f"- Component scores: {entry['component_scores']}",
                f"- Explanation: {entry['explanation']}",
                "",
            ]
        )
    return lines


def write_phase2_summary_reports(
    reports_root: Path,
    payload: dict[str, Any],
    labels_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    validation_checks: list[dict[str, Any]] | None = None,
    clean_truth_df: pd.DataFrame | None = None,
    attacked_truth_df: pd.DataFrame | None = None,
    clean_measured_df: pd.DataFrame | None = None,
    attacked_measured_df: pd.DataFrame | None = None,
    clean_cyber_df: pd.DataFrame | None = None,
    attacked_cyber_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    _ensure_directory(reports_root)

    coverage_summary = build_phase2_coverage_summary(payload)
    coverage_json = reports_root / "phase2_scenario_coverage_summary.json"
    coverage_md = reports_root / "phase2_scenario_coverage_summary.md"
    _write_json(coverage_json, coverage_summary)
    _write_markdown(coverage_md, _render_coverage_markdown(coverage_summary))

    label_summary = build_phase2_label_summary(labels_df)
    label_json = reports_root / "phase2_label_summary.json"
    label_md = reports_root / "phase2_label_summary.md"
    _write_json(label_json, label_summary)
    _write_markdown(label_md, _render_label_markdown(label_summary))

    output_paths = {
        "coverage_summary_json": str(coverage_json),
        "coverage_summary_md": str(coverage_md),
        "label_summary_json": str(label_json),
        "label_summary_md": str(label_md),
    }

    analysis_inputs_present = all(
        frame is not None
        for frame in (
            clean_truth_df,
            attacked_truth_df,
            clean_measured_df,
            attacked_measured_df,
            clean_cyber_df,
            attacked_cyber_df,
        )
    )
    if analysis_inputs_present:
        effect_summary = build_phase2_effect_summary(
            payload,
            labels_df,
            clean_truth_df,  # type: ignore[arg-type]
            attacked_truth_df,  # type: ignore[arg-type]
            clean_measured_df,  # type: ignore[arg-type]
            attacked_measured_df,  # type: ignore[arg-type]
            clean_cyber_df,  # type: ignore[arg-type]
            attacked_cyber_df,  # type: ignore[arg-type]
        )
        difficulty_summary = build_phase2_scenario_difficulty(effect_summary)

        effect_json = reports_root / "phase2_effect_summary.json"
        effect_md = reports_root / "phase2_effect_summary.md"
        difficulty_json = reports_root / "phase2_scenario_difficulty.json"
        difficulty_md = reports_root / "phase2_scenario_difficulty.md"

        _write_json(effect_json, effect_summary)
        _write_markdown(effect_md, _render_effect_markdown(effect_summary))
        _write_json(difficulty_json, difficulty_summary)
        _write_markdown(difficulty_md, _render_difficulty_markdown(difficulty_summary))

        output_paths.update(
            {
                "effect_summary_json": str(effect_json),
                "effect_summary_md": str(effect_md),
                "scenario_difficulty_json": str(difficulty_json),
                "scenario_difficulty_md": str(difficulty_md),
            }
        )

    if validation_checks is not None:
        validation_json = reports_root / "attacked_validation_summary.json"
        validation_md = reports_root / "attacked_validation_summary.md"
        normalized_checks = _json_safe(validation_checks)
        if not isinstance(normalized_checks, list):
            normalized_checks = []
        _write_json(validation_json, {"checks": normalized_checks})
        validation_lines = ["# Attacked Dataset Validation Summary", ""]
        for check in normalized_checks:
            status = "PASS" if check.get("passed") else "FAIL"
            validation_lines.append(
                f"- {check.get('name')}: {status} (metric={check.get('metric')}, threshold={check.get('threshold')})"
            )
        _write_markdown(validation_md, validation_lines)
        output_paths.update(
            {
                "validation_summary_json": str(validation_json),
                "validation_summary_md": str(validation_md),
            }
        )

    return output_paths


def generate_phase2_analysis_reports(
    *,
    clean_root: Path,
    attacked_root: Path,
    reports_root: Path,
    scenarios_path: Path | None = None,
) -> dict[str, str]:
    scenario_manifest_path = attacked_root / "scenario_manifest.json"
    payload = _read_json(scenarios_path) if scenarios_path else _read_json(scenario_manifest_path)
    if "scenarios" not in payload and "applied_scenarios" in payload:
        payload = {"dataset_id": payload.get("scenario_id", "attacked_bundle"), "scenarios": payload.get("applied_scenarios", [])}
    clean_truth_df = pd.read_parquet(clean_root / "truth_physical_timeseries.parquet")
    attacked_truth_df = pd.read_parquet(attacked_root / "truth_physical_timeseries.parquet")
    clean_measured_df = pd.read_parquet(clean_root / "measured_physical_timeseries.parquet")
    attacked_measured_df = pd.read_parquet(attacked_root / "measured_physical_timeseries.parquet")
    clean_cyber_df = pd.read_parquet(clean_root / "cyber_events.parquet")
    attacked_cyber_df = pd.read_parquet(attacked_root / "cyber_events.parquet")
    labels_df = pd.read_parquet(attacked_root / "attack_labels.parquet")
    windows_df = pd.read_parquet(attacked_root / "merged_windows.parquet")

    validation_summary_path = reports_root / "attacked_validation_summary.json"
    validation_payload = _read_json(validation_summary_path) if validation_summary_path.exists() else {}
    if isinstance(validation_payload, dict):
        validation_checks = validation_payload.get("checks")
    elif isinstance(validation_payload, list):
        validation_checks = validation_payload
    else:
        validation_checks = None

    return write_phase2_summary_reports(
        reports_root,
        payload,
        labels_df,
        windows_df,
        validation_checks=validation_checks if isinstance(validation_checks, list) else None,
        clean_truth_df=clean_truth_df,
        attacked_truth_df=attacked_truth_df,
        clean_measured_df=clean_measured_df,
        attacked_measured_df=attacked_measured_df,
        clean_cyber_df=clean_cyber_df,
        attacked_cyber_df=attacked_cyber_df,
    )


def main() -> None:
    paths = ProjectPaths().ensure()
    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase 2 reporting artifacts from existing clean and attacked bundles. "
            "These outputs strengthen auditability and interpretation without changing the canonical executable path."
        )
    )
    parser.add_argument("--clean-root", type=Path, default=paths.clean_output)
    parser.add_argument("--attacked-root", type=Path, default=paths.attacked_output)
    parser.add_argument("--reports-root", type=Path, default=paths.reports_output)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=None,
        help="Optional scenario JSON to use instead of attacked scenario_manifest.json.",
    )
    args = parser.parse_args()
    output_paths = generate_phase2_analysis_reports(
        clean_root=args.clean_root,
        attacked_root=args.attacked_root,
        reports_root=args.reports_root,
        scenarios_path=args.scenarios,
    )
    print("Generated Phase 2 reports:")
    for key, value in sorted(output_paths.items()):
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
