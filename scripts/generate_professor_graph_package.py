"""Repository orchestration script for DERGuardian.

This script runs or rebuilds generate professor graph package artifacts for audits, figures,
reports, or reproducibility checks. It is release-support code and must preserve
the separation between canonical benchmark, replay, heldout synthetic, and
extension experiment contexts.
"""

from __future__ import annotations

import ast
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "window_size_study" / "professor_graph_package"
TABLE_ROOT = OUTPUT_ROOT / "graph_data_tables"
EVAL_ROOT = ROOT / "outputs" / "window_size_study" / "improved_phase3" / "evaluations"
HELDOUT_ROOT = ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models"
HELDOUT_JSON_ROOT = ROOT / "phase2_llm_benchmark" / "heldout_llm_response"
CLEAN_MEASURED_PATH = ROOT / "outputs" / "clean" / "measured_physical_timeseries.parquet"
MULTI_MODEL_METRICS_PATH = ROOT / "multi_model_heldout_metrics.csv"
BALANCED_METRICS_PATH = ROOT / "balanced_heldout_metrics.csv"
BENCHMARK_VS_REPLAY_PATH = ROOT / "benchmark_vs_replay_metrics.csv"

LLM_GENERATORS = ["chatgpt", "claude", "gemini", "grok"]
FIXED_MODEL_ORDER = [
    ("threshold_baseline", "10s"),
    ("transformer", "60s"),
    ("lstm", "300s"),
]
MODEL_DISPLAY = {
    ("threshold_baseline", "10s"): "Threshold Baseline (10s)",
    ("transformer", "60s"): "Transformer (60s)",
    ("lstm", "300s"): "LSTM (300s)",
}
MODEL_SHORT = {
    ("threshold_baseline", "10s"): "Threshold\n10s",
    ("transformer", "60s"): "Transformer\n60s",
    ("lstm", "300s"): "LSTM\n300s",
}
MODEL_PRESENTATION = {
    ("threshold_baseline", "10s"): "Threshold (10s)",
    ("transformer", "60s"): "Transformer (60s)",
    ("lstm", "300s"): "LSTM (300s)",
}
MODEL_PRESENTATION_SHORT = {
    ("threshold_baseline", "10s"): "Threshold\n(10s)",
    ("transformer", "60s"): "Transformer\n(60s)",
    ("lstm", "300s"): "LSTM\n(300s)",
}
MODEL_COLORS = {
    ("threshold_baseline", "10s"): "#7a7a7a",
    ("transformer", "60s"): "#355c7d",
    ("lstm", "300s"): "#4f7c5d",
}
METRIC_COLORS = {
    "precision": "#4e79a7",
    "recall": "#59a14f",
    "f1": "#9c755f",
}
FONT_FAMILY = "DejaVu Sans"
GENERATOR_DISPLAY = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "grok": "Grok",
}
CATEGORY_DISPLAY = {
    "bess_soc_dispatch": "BESS dispatch",
    "capacitor_control": "Capacitor control",
    "feeder_voltage": "Feeder voltage",
    "pv_curtailment": "PV curtailment",
    "pv_dispatch": "PV dispatch",
    "feeder_power": "Feeder power",
    "mixed_multi_asset": "Multi-asset",
    "feeder_current": "Feeder current",
    "regulator_control": "Regulator control",
    "telemetry_replay": "Telemetry replay",
    "command_path": "Command path",
}
TIMELINE_LEGEND_LABELS = {
    ("threshold_baseline", "10s"): "Threshold detect",
    ("transformer", "60s"): "Transformer detect",
    ("lstm", "300s"): "LSTM detect",
}
IMPROVED_MEETING_ORDER = [
    "performance_f1_only_by_model.png",
    "latency_vs_model_all_llm_attacks_zoomed.png",
    "per_generator_performance_matrix_improved.png",
    "timeline_detection_scenario_1_improved.png",
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    return value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _load_effective_json(generator_source: str) -> dict[str, Any]:
    folder = HELDOUT_JSON_ROOT / generator_source
    if not folder.exists():
        raise FileNotFoundError(f"Heldout JSON folder not found for {generator_source}: {folder}")
    preferred = folder / "new_respnse.json"
    json_path = preferred if preferred.exists() else sorted(folder.glob("*.json"))[0]
    return json.loads(json_path.read_text(encoding="utf-8"))


def _parse_listish(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
        parts = [item.strip().strip("'\"") for item in inner.split(",")]
        return [item for item in parts if item]
    return [text]


def _parse_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, float) and math.isnan(value):
        return {}
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _resolve_measurement_target_column(target_asset: str | None, target_signal: str | None) -> str | None:
    if not target_signal:
        return None
    signal = str(target_signal)
    if signal.startswith(
        (
            "bus_",
            "feeder_",
            "pv_",
            "bess_",
            "line_",
            "regulator_",
            "capacitor_",
            "switch_",
            "breaker_",
            "relay_",
            "derived_",
        )
    ):
        return signal
    asset = str(target_asset or "")
    if asset.startswith("pv"):
        return f"pv_{asset}_{signal}"
    if asset.startswith("bess"):
        return f"bess_{asset}_{signal}"
    if asset.startswith("bus"):
        suffix = asset.replace("bus", "")
        return signal if signal.startswith("bus_") else f"bus_{suffix}_{signal}"
    return signal


def _category_from_metadata(row: dict[str, Any]) -> tuple[str, str]:
    signals = [item.lower() for item in row.get("affected_signals", []) + row.get("observable_signals", []) if item]
    target_signal = str(row.get("target_signal") or "").lower()
    target_component = str(row.get("target_component") or "").lower()
    target_asset = str(row.get("target_asset") or "").lower()
    attack_family = str(row.get("attack_family") or "").lower()
    additional_targets = row.get("additional_targets", [])
    affected_assets = [item.lower() for item in row.get("affected_assets", []) if item]

    if attack_family == "coordinated_campaign" or len(set(affected_assets)) > 1 or len(additional_targets) > 0:
        return "mixed_multi_asset", "multi-asset or coordinated metadata"
    if attack_family == "replay" or "replay" in target_signal or any("replay" in signal for signal in signals):
        return "telemetry_replay", "attack family or signal metadata indicates replayed telemetry"
    if target_component == "regulator" or target_asset.startswith("creg") or any(signal.startswith("regulator_") for signal in signals):
        return "regulator_control", "target component or observable signals are regulator-related"
    if target_component == "capacitor" or target_asset.startswith("c") and "cap" not in target_asset or any(signal.startswith("capacitor_") for signal in signals):
        return "capacitor_control", "target component or observable signals are capacitor-related"
    if target_component == "pv" or target_asset.startswith("pv"):
        if "curtailment" in target_signal or any("curtailment" in signal for signal in signals):
            return "pv_curtailment", "PV-targeted scenario with curtailment-oriented signal"
        return "pv_dispatch", "PV-targeted scenario affecting active/reactive output or availability"
    if target_component == "bess" or target_asset.startswith("bess"):
        return "bess_soc_dispatch", "BESS-targeted scenario affecting dispatch, SOC, or available charge/discharge"
    if any("feeder_i_" in signal or "terminal_i_" in signal or "current" in signal for signal in signals + [target_signal]):
        return "feeder_current", "signal metadata centers on feeder/current magnitude"
    if any("v_pu" in signal or "voltage" in signal for signal in signals + [target_signal]):
        return "feeder_voltage", "signal metadata centers on voltage telemetry"
    if any(
        token in signal
        for signal in signals + [target_signal]
        for token in ["p_kw", "q_kvar", "losses_kw", "losses_kvar"]
    ):
        return "feeder_power", "signal metadata centers on active/reactive power telemetry"
    if attack_family in {"command_delay", "command_suppression", "unauthorized_command"}:
        return "command_path", "command-oriented attack family without a more specific asset grouping"
    return attack_family or "command_path", "fallback to attack family label"


def _friendly_category_name(category: str, count: int | None = None) -> str:
    label = CATEGORY_DISPLAY.get(category, category.replace("_", " ").title())
    if count is not None:
        return f"{label} (n={count})"
    return label


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": FONT_FAMILY,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def _discover_complete_models() -> list[tuple[str, str]]:
    metrics_df = pd.read_csv(MULTI_MODEL_METRICS_PATH)
    llm_df = metrics_df.loc[metrics_df["generator_source"].isin(LLM_GENERATORS)].copy()
    counts = (
        llm_df.groupby(["model_name", "window_label"])["generator_source"]
        .nunique()
        .reset_index(name="generator_count")
    )
    complete = {
        (str(row["model_name"]), str(row["window_label"]))
        for row in counts.to_dict(orient="records")
        if int(row["generator_count"]) == len(LLM_GENERATORS)
    }
    ordered: list[tuple[str, str]] = []
    for model_key in FIXED_MODEL_ORDER:
        if model_key in complete:
            ordered.append(model_key)
    extras = sorted(model_key for model_key in complete if model_key not in ordered)
    ordered.extend(extras)
    return ordered


def _load_scenario_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for generator in LLM_GENERATORS:
        labels_path = HELDOUT_ROOT / generator / "datasets" / "attack_labels.parquet"
        labels_df = pd.read_parquet(labels_path).copy()
        bundle_payload = _load_effective_json(generator)
        scenario_lookup = {
            str(item.get("scenario_id")): item
            for item in bundle_payload.get("scenarios", [])
        }
        for record in labels_df.to_dict(orient="records"):
            scenario_id = str(record["scenario_id"])
            scenario_meta = scenario_lookup.get(scenario_id, {})
            causal_metadata = _parse_mapping(record.get("causal_metadata"))
            affected_assets = _parse_listish(record.get("affected_assets"))
            affected_signals = _parse_listish(record.get("affected_signals"))
            observable_signals = _parse_listish(scenario_meta.get("observable_signals") or causal_metadata.get("observable_signals"))
            target_component = scenario_meta.get("target_component") or record.get("target_component")
            target_asset = scenario_meta.get("target_asset") or (affected_assets[0] if affected_assets else None)
            target_signal = scenario_meta.get("target_signal") or _resolve_measurement_target_column(target_asset, scenario_meta.get("target_signal")) or (affected_signals[0] if affected_signals else None)
            injection_type = scenario_meta.get("injection_type") or causal_metadata.get("injection_type")
            additional_targets = scenario_meta.get("additional_targets", [])
            row = {
                "generator_source": generator,
                "scenario_id": scenario_id,
                "scenario_name": record.get("scenario_name") or scenario_meta.get("scenario_name"),
                "attack_family": scenario_meta.get("attack_family") or record.get("attack_family"),
                "target_component": target_component,
                "target_asset": target_asset,
                "target_signal": target_signal,
                "injection_type": injection_type,
                "start_time_utc": pd.to_datetime(record.get("start_time_utc"), utc=True),
                "end_time_utc": pd.to_datetime(record.get("end_time_utc"), utc=True),
                "duration_seconds": (
                    _safe_float(scenario_meta.get("duration_s"))
                    or _safe_float(scenario_meta.get("duration_seconds"))
                    or (
                        pd.to_datetime(record.get("end_time_utc"), utc=True) - pd.to_datetime(record.get("start_time_utc"), utc=True)
                    ).total_seconds()
                ),
                "affected_assets": affected_assets,
                "affected_signals": affected_signals,
                "observable_signals": observable_signals,
                "additional_targets": additional_targets,
                "causal_metadata": causal_metadata,
            }
            category, mapping_reason = _category_from_metadata(row)
            row["variable_group"] = category
            row["mapping_reason"] = mapping_reason
            rows.append(row)
    inventory_df = pd.DataFrame(rows)
    inventory_df.sort_values(["generator_source", "start_time_utc", "scenario_id"], inplace=True)
    return inventory_df


def _load_predictions_and_summaries(models: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_df = pd.read_csv(MULTI_MODEL_METRICS_PATH)
    llm_df = metrics_df.loc[metrics_df["generator_source"].isin(LLM_GENERATORS)].copy()
    dataset_map = llm_df.groupby("generator_source", as_index=False)["dataset_id"].first()
    dataset_lookup = dict(zip(dataset_map["generator_source"], dataset_map["dataset_id"]))

    prediction_frames: list[pd.DataFrame] = []
    scenario_frames: list[pd.DataFrame] = []
    latency_frames: list[pd.DataFrame] = []
    for generator in LLM_GENERATORS:
        dataset_id = str(dataset_lookup[generator])
        artifact_key = f"{generator}__existing_heldout_phase2_bundle__{dataset_id}"
        for model_name, window_label in models:
            eval_dir = EVAL_ROOT / artifact_key / window_label / model_name
            predictions = pd.read_parquet(eval_dir / "predictions.parquet").copy()
            predictions["generator_source"] = generator
            predictions["model_name"] = model_name
            predictions["window_label"] = window_label
            prediction_frames.append(predictions)

            scenario_df = pd.read_csv(eval_dir / "scenario_summary.csv").copy()
            scenario_df["generator_source"] = generator
            scenario_df["model_name"] = model_name
            scenario_df["window_label"] = window_label
            scenario_df["first_detection_time"] = pd.to_datetime(scenario_df["first_detection_time"], utc=True, errors="coerce")
            scenario_frames.append(scenario_df)

            latency_df = pd.read_csv(eval_dir / "latency_table.csv").copy()
            latency_df["generator_source"] = generator
            latency_df["model_name"] = model_name
            latency_df["window_label"] = window_label
            latency_df["attack_start_utc"] = pd.to_datetime(latency_df["attack_start_utc"], utc=True, errors="coerce")
            latency_df["first_detection_utc"] = pd.to_datetime(latency_df["first_detection_utc"], utc=True, errors="coerce")
            latency_frames.append(latency_df)
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(scenario_frames, ignore_index=True),
        pd.concat(latency_frames, ignore_index=True),
    )


def _binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["attack_present"].astype(int).to_numpy()
    y_pred = frame["predicted"].astype(int).to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _model_display(model_name: str, window_label: str) -> str:
    return MODEL_DISPLAY.get((model_name, window_label), f"{model_name} ({window_label})")


def _model_short(model_name: str, window_label: str) -> str:
    return MODEL_SHORT.get((model_name, window_label), f"{model_name}\n{window_label}")


def _model_presentation(model_name: str, window_label: str) -> str:
    return MODEL_PRESENTATION.get((model_name, window_label), f"{model_name} ({window_label})")


def _model_presentation_short(model_name: str, window_label: str) -> str:
    return MODEL_PRESENTATION_SHORT.get((model_name, window_label), f"{model_name}\n({window_label})")


def _ordered_model_display_labels(models: list[tuple[str, str]]) -> list[str]:
    return [_model_display(model_name, window_label) for model_name, window_label in models]


def _ordered_model_presentation_labels(models: list[tuple[str, str]]) -> list[str]:
    return [_model_presentation(model_name, window_label) for model_name, window_label in models]


def _pretty_generator(generator_source: str) -> str:
    return GENERATOR_DISPLAY.get(generator_source, generator_source.title())


def _performance_table(predictions_df: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, window_label in models:
        subset = predictions_df.loc[
            (predictions_df["model_name"] == model_name)
            & (predictions_df["window_label"] == window_label)
            & (predictions_df["generator_source"].isin(LLM_GENERATORS))
        ].copy()
        metrics = _binary_metrics(subset)
        rows.append(
            {
                "model_name": model_name,
                "window_label": window_label,
                "model_display": _model_display(model_name, window_label),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "window_count": int(len(subset)),
                "attack_window_count": int(subset["attack_present"].astype(int).sum()),
                "benign_window_count": int((subset["attack_present"].astype(int) == 0).sum()),
                "generator_count": len(LLM_GENERATORS),
                "source_scope": "Original improved heldout replay predictions across chatgpt, claude, gemini, and grok accepted bundles only.",
            }
        )
    performance_df = pd.DataFrame(rows)
    performance_df.to_csv(TABLE_ROOT / "performance_vs_model_all_llm_attacks.csv", index=False)
    return performance_df


def _latency_table(latency_df: pd.DataFrame, scenario_df: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    scenario_counts = (
        scenario_df.loc[scenario_df["generator_source"].isin(LLM_GENERATORS)]
        .groupby(["model_name", "window_label"])["scenario_id"]
        .nunique()
        .reset_index(name="scenario_count_total")
    )
    rows: list[dict[str, Any]] = []
    for model_name, window_label in models:
        subset = latency_df.loc[
            (latency_df["model_name"] == model_name)
            & (latency_df["window_label"] == window_label)
            & (latency_df["generator_source"].isin(LLM_GENERATORS))
        ].copy()
        total_row = scenario_counts.loc[
            (scenario_counts["model_name"] == model_name)
            & (scenario_counts["window_label"] == window_label)
        ]
        total_scenarios = int(total_row["scenario_count_total"].iloc[0]) if not total_row.empty else 0
        rows.append(
            {
                "model_name": model_name,
                "window_label": window_label,
                "model_display": _model_display(model_name, window_label),
                "mean_latency_seconds": float(subset["latency_seconds"].mean()) if not subset.empty else np.nan,
                "median_latency_seconds": float(subset["latency_seconds"].median()) if not subset.empty else np.nan,
                "detected_scenarios": int(subset["scenario_id"].nunique()),
                "total_scenarios": total_scenarios,
                "source_scope": "Detected-scenario latency from improved heldout replay latency tables across chatgpt, claude, gemini, and grok.",
            }
        )
    latency_summary_df = pd.DataFrame(rows)
    latency_summary_df.to_csv(TABLE_ROOT / "latency_vs_model_all_llm_attacks.csv", index=False)
    return latency_summary_df


def _build_generator_matrix(predictions_df: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for generator in LLM_GENERATORS:
        for model_name, window_label in models:
            subset = predictions_df.loc[
                (predictions_df["generator_source"] == generator)
                & (predictions_df["model_name"] == model_name)
                & (predictions_df["window_label"] == window_label)
            ].copy()
            metrics = _binary_metrics(subset)
            rows.append(
                {
                    "generator_source": generator,
                    "model_name": model_name,
                    "window_label": window_label,
                    "model_display": _model_display(model_name, window_label),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            )
    generator_matrix_df = pd.DataFrame(rows)
    generator_matrix_df.to_csv(TABLE_ROOT / "per_generator_performance_matrix.csv", index=False)
    return generator_matrix_df


def _merge_scenario_metadata(scenario_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    merged = scenario_df.merge(
        inventory_df[
            [
                "generator_source",
                "scenario_id",
                "scenario_name",
                "attack_family",
                "target_component",
                "target_asset",
                "target_signal",
                "start_time_utc",
                "end_time_utc",
                "duration_seconds",
                "affected_assets",
                "affected_signals",
                "observable_signals",
                "variable_group",
                "mapping_reason",
            ]
        ],
        on=["generator_source", "scenario_id"],
        how="left",
        suffixes=("", "_meta"),
    )
    merged["attack_family"] = merged["attack_family_meta"].fillna(merged["attack_family"])
    merged.drop(columns=[column for column in merged.columns if column.endswith("_meta")], inplace=True)
    return merged


def _write_attack_variable_mapping(inventory_df: pd.DataFrame) -> pd.DataFrame:
    mapping_df = inventory_df[
        [
            "generator_source",
            "scenario_id",
            "scenario_name",
            "attack_family",
            "target_component",
            "target_asset",
            "target_signal",
            "variable_group",
            "mapping_reason",
        ]
    ].copy()
    mapping_df["affected_assets"] = inventory_df["affected_assets"].apply(lambda items: "|".join(items))
    mapping_df["affected_signals"] = inventory_df["affected_signals"].apply(lambda items: "|".join(items))
    mapping_df["observable_signals"] = inventory_df["observable_signals"].apply(lambda items: "|".join(items))
    mapping_df.to_csv(TABLE_ROOT / "attack_variable_group_mapping.csv", index=False)
    mapping_df.to_csv(TABLE_ROOT / "llm_scenario_master_inventory.csv", index=False)
    return mapping_df


def _attack_variable_heatmap_table(merged_scenarios: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    long_df = (
        merged_scenarios.loc[merged_scenarios["generator_source"].isin(LLM_GENERATORS)]
        .groupby(["variable_group", "model_name", "window_label"], as_index=False)
        .agg(recall=("detected", "mean"), scenario_count=("scenario_id", "nunique"))
    )
    long_df["model_display"] = long_df.apply(lambda row: _model_display(str(row["model_name"]), str(row["window_label"])), axis=1)

    counts = (
        merged_scenarios.loc[merged_scenarios["generator_source"].isin(LLM_GENERATORS)]
        .groupby("variable_group")["scenario_id"]
        .nunique()
        .to_dict()
    )
    pivot_df = long_df.pivot(index="variable_group", columns="model_display", values="recall").fillna(0.0)
    order_columns = [_model_display(model_name, window_label) for model_name, window_label in models]
    pivot_df = pivot_df.reindex(columns=order_columns)
    pivot_df = pivot_df.reindex(sorted(pivot_df.index, key=lambda category: (-counts.get(category, 0), category)))
    export_df = pivot_df.copy()
    export_df.insert(0, "scenario_count", [counts.get(index, 0) for index in export_df.index])
    export_df.to_csv(TABLE_ROOT / "attack_variable_effect_heatmap_by_model.csv")
    return export_df


def _build_detection_time_distribution(latency_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    distribution_df = latency_df.merge(
        inventory_df[["generator_source", "scenario_id", "attack_family", "variable_group"]],
        on=["generator_source", "scenario_id"],
        how="left",
    )
    distribution_df = distribution_df.loc[distribution_df["generator_source"].isin(LLM_GENERATORS)].copy()
    distribution_df.to_csv(TABLE_ROOT / "detection_time_distribution_by_model.csv", index=False)
    return distribution_df


def _score_candidate_variable(
    variable_name: str,
    target_component: str,
    target_asset: str,
) -> float:
    score = 0.0
    lowered = variable_name.lower()
    if "feeder_p_kw_total" in lowered:
        score += 15.0
    if "feeder_q_kvar_total" in lowered:
        score += 13.0
    if "v_pu" in lowered:
        score += 11.0
    if "_p_kw" in lowered:
        score += 10.0
    if "_q_kvar" in lowered:
        score += 9.0
    if "soc" in lowered:
        score += 8.0
    if "terminal_i_a" in lowered or "feeder_i_" in lowered:
        score += 7.0
    if target_asset and target_asset.lower() in lowered:
        score += 4.0
    if target_component == "pv" and "pv_" in lowered:
        score += 3.0
    if target_component == "bess" and "bess_" in lowered:
        score += 3.0
    if "curtailment_frac" in lowered or "target_kw" in lowered:
        score -= 10.0
    if lowered.endswith("_state") or lowered.endswith("_status") or lowered.endswith("_mode"):
        score -= 20.0
    if "tap_pos" in lowered:
        score -= 6.0
    return score


def _choose_timeline_variable(
    metadata_row: dict[str, Any],
    attacked_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> tuple[str, str]:
    target_component = str(metadata_row.get("target_component") or "")
    target_asset = str(metadata_row.get("target_asset") or "")
    target_signal = metadata_row.get("target_signal")
    candidates: list[str] = []
    resolved_target = _resolve_measurement_target_column(target_asset, target_signal)
    if resolved_target:
        candidates.append(resolved_target)
    for collection in (
        metadata_row.get("affected_signals", []),
        metadata_row.get("observable_signals", []),
    ):
        for item in collection:
            if item and item not in candidates:
                candidates.append(str(item))

    if target_component == "pv":
        candidates.extend(
            [
                f"pv_{target_asset}_p_kw",
                f"pv_{target_asset}_q_kvar",
                "feeder_p_kw_total",
                "feeder_q_kvar_total",
                f"pv_{target_asset}_curtailment_frac",
            ]
        )
    elif target_component == "bess":
        candidates.extend(
            [
                f"bess_{target_asset}_p_kw",
                f"bess_{target_asset}_soc",
                "feeder_p_kw_total",
                "feeder_q_kvar_total",
            ]
        )
    elif target_component == "regulator":
        candidates.extend(
            [
                "feeder_v_pu_phase_a",
                "feeder_v_pu_phase_b",
                "feeder_v_pu_phase_c",
                f"regulator_{target_asset}_tap_pos",
            ]
        )
    elif target_component == "capacitor":
        candidates.extend(
            [
                "feeder_q_kvar_total",
                "feeder_v_pu_phase_a",
                f"capacitor_{target_asset}_state",
            ]
        )

    unique_candidates: list[str] = []
    seen = set()
    for candidate in candidates:
        candidate = str(candidate)
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    attacked_cols = set(attacked_df.columns)
    clean_cols = set(clean_df.columns)
    valid_candidates = [candidate for candidate in unique_candidates if candidate in attacked_cols and candidate in clean_cols]
    if not valid_candidates:
        raise KeyError(f"No valid timeline variable found for scenario {metadata_row.get('scenario_id')}")
    physical_candidates = [
        candidate
        for candidate in valid_candidates
        if not any(token in candidate.lower() for token in ["curtailment_frac", "target_kw", "_state", "_status", "_mode", "tap_pos"])
    ]
    if physical_candidates:
        valid_candidates = physical_candidates

    start_time = pd.Timestamp(metadata_row["start_time_utc"])
    end_time = pd.Timestamp(metadata_row["end_time_utc"])
    duration = max((end_time - start_time).total_seconds(), 60.0)
    pre_seconds = min(max(duration * 0.75, 180.0), 900.0)
    post_seconds = min(max(duration * 0.75, 180.0), 900.0)

    attacked_subset = attacked_df.copy()
    attacked_subset["timestamp_utc"] = pd.to_datetime(attacked_subset["timestamp_utc"], utc=True)
    attacked_subset = attacked_subset.loc[
        (attacked_subset["timestamp_utc"] >= start_time - pd.Timedelta(seconds=pre_seconds))
        & (attacked_subset["timestamp_utc"] <= end_time + pd.Timedelta(seconds=post_seconds))
    ].copy()

    clean_subset = clean_df[["simulation_index"] + valid_candidates].copy()
    merged = attacked_subset[["simulation_index"] + valid_candidates].merge(
        clean_subset,
        on="simulation_index",
        how="left",
        suffixes=("_attacked", "_clean"),
    )
    attack_window = attacked_subset.loc[
        (attacked_subset["timestamp_utc"] >= start_time) & (attacked_subset["timestamp_utc"] <= end_time)
    ].copy()
    attack_indices = set(attack_window["simulation_index"].tolist())

    best_candidate = valid_candidates[0]
    best_score = -float("inf")
    best_reason = "metadata-priority fallback"
    for candidate in valid_candidates:
        attacked_series = pd.to_numeric(merged[f"{candidate}_attacked"], errors="coerce")
        clean_series = pd.to_numeric(merged[f"{candidate}_clean"], errors="coerce")
        if attacked_series.notna().sum() < 5 or clean_series.notna().sum() < 5:
            continue
        attack_mask = merged["simulation_index"].isin(attack_indices)
        if attack_mask.sum() == 0:
            continue
        diff = (attacked_series - clean_series).abs()
        attack_diff = diff.loc[attack_mask].mean(skipna=True)
        baseline_scale = clean_series.loc[~attack_mask].std(skipna=True)
        deviation_score = float(attack_diff) / float((baseline_scale or 0.0) + 1e-6)
        relevance_score = _score_candidate_variable(candidate, target_component.lower(), target_asset)
        total_score = deviation_score + relevance_score
        if total_score > best_score:
            best_candidate = candidate
            best_score = total_score
            best_reason = (
                f"chosen from metadata candidates using deviation score {deviation_score:.3f} "
                f"and relevance bonus {relevance_score:.1f}"
            )
    return best_candidate, best_reason


def _build_scenario_difficulty(merged_scenarios: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    base_summary = (
        merged_scenarios.groupby(
            ["generator_source", "scenario_id"],
            dropna=False,
            as_index=False,
        )
        .agg(
            detected_models=("detected", "sum"),
            total_models=("detected", "count"),
            detection_rate=("detected", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
            min_latency_seconds=("latency_seconds", "min"),
            max_latency_seconds=("latency_seconds", "max"),
        )
    )
    metadata = merged_scenarios[
        [
            "generator_source",
            "scenario_id",
            "scenario_name",
            "attack_family",
            "target_component",
            "target_asset",
            "target_signal",
            "variable_group",
            "start_time_utc",
            "end_time_utc",
            "duration_seconds",
            "affected_assets",
            "affected_signals",
            "observable_signals",
        ]
    ].drop_duplicates(subset=["generator_source", "scenario_id"])
    scenario_summary = base_summary.merge(metadata, on=["generator_source", "scenario_id"], how="left")
    max_latency = float(scenario_summary["mean_latency_seconds"].dropna().max()) if scenario_summary["mean_latency_seconds"].notna().any() else 1.0
    filled_latency = scenario_summary["mean_latency_seconds"].fillna(max_latency + 60.0)
    latency_min = float(filled_latency.min()) if not filled_latency.empty else 0.0
    latency_range = float(filled_latency.max() - latency_min) if len(filled_latency) else 1.0
    scenario_summary["difficulty_score"] = (
        (1.0 - scenario_summary["detection_rate"].astype(float)) * 0.7
        + ((filled_latency - latency_min) / (latency_range + 1e-6)) * 0.3
    )
    scenario_summary.sort_values(["difficulty_score", "generator_source", "scenario_id"], inplace=True)
    return scenario_summary


def _select_representative_scenarios(
    scenario_difficulty: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows: list[dict[str, Any]] = []
    used = set()

    def add_row(row: pd.Series, reason: str) -> None:
        key = (str(row["generator_source"]), str(row["scenario_id"]))
        if key in used:
            return
        selected_rows.append(
            {
                "generator_source": str(row["generator_source"]),
                "scenario_id": str(row["scenario_id"]),
                "attack_family": str(row["attack_family"]),
                "variable_group": str(row["variable_group"]),
                "difficulty_score": float(row["difficulty_score"]),
                "selection_reason": reason,
            }
        )
        used.add(key)

    easy_candidates = scenario_difficulty.loc[scenario_difficulty["detection_rate"] == scenario_difficulty["detection_rate"].max()].copy()
    if not easy_candidates.empty:
        add_row(easy_candidates.sort_values(["difficulty_score", "mean_latency_seconds"]).iloc[0], "easy scenario: highest cross-model detection rate with low mean latency")

    if not scenario_difficulty.empty:
        median_difficulty = float(scenario_difficulty["difficulty_score"].median())
        moderate = scenario_difficulty.assign(distance=(scenario_difficulty["difficulty_score"] - median_difficulty).abs()).sort_values(
            ["distance", "generator_source", "scenario_id"]
        )
        for _, row in moderate.iterrows():
            if (str(row["generator_source"]), str(row["scenario_id"])) not in used:
                add_row(row, "moderate scenario: difficulty closest to the middle of the heldout replay range")
                break

    hard = scenario_difficulty.sort_values(["difficulty_score", "mean_latency_seconds"], ascending=[False, False])
    for _, row in hard.iterrows():
        if (str(row["generator_source"]), str(row["scenario_id"])) not in used:
            add_row(row, "hard scenario: weakest cross-model detection or slowest mean response")
            break

    fdi = scenario_difficulty.loc[scenario_difficulty["attack_family"] == "false_data_injection"].sort_values(
        ["difficulty_score", "generator_source", "scenario_id"]
    )
    for _, row in fdi.iterrows():
        if (str(row["generator_source"]), str(row["scenario_id"])) not in used:
            add_row(row, "family coverage: representative false-data-injection scenario")
            break

    control_families = {"command_delay", "command_suppression", "unauthorized_command", "coordinated_campaign"}
    control = scenario_difficulty.loc[
        scenario_difficulty["attack_family"].isin(control_families) | (scenario_difficulty["target_component"] != "measured_layer")
    ].sort_values(["difficulty_score", "generator_source", "scenario_id"])
    for _, row in control.iterrows():
        if (str(row["generator_source"]), str(row["scenario_id"])) not in used:
            add_row(row, "family coverage: representative command/control-path scenario")
            break

    selected_df = pd.DataFrame(selected_rows).drop_duplicates(subset=["generator_source", "scenario_id"]).head(5)
    return selected_df


def _plot_performance_chart(performance_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    long_df = performance_df.melt(
        id_vars=["model_name", "window_label", "model_display"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    ordered_labels = [_model_display(model_name, window_label) for model_name, window_label in models]
    x_positions = np.arange(len(ordered_labels))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    for index, metric in enumerate(["precision", "recall", "f1"]):
        subset = long_df.loc[long_df["metric"] == metric].set_index("model_display").reindex(ordered_labels).reset_index()
        positions = x_positions + (index - 1) * bar_width
        bars = ax.bar(positions, subset["score"], width=bar_width, color=METRIC_COLORS[metric], label=metric.capitalize())
        for bar, value in zip(bars, subset["score"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(value) + 0.02,
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_model_short(model_name, window_label) for model_name, window_label in models])
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Score")
    ax.set_xlabel("Frozen Model")
    ax.set_title("Frozen-Model Performance Across LLM-Generated Heldout Attacks", pad=18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUTPUT_ROOT / "performance_vs_model_all_llm_attacks.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_chart(latency_summary_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    ordered = latency_summary_df.set_index("model_display").reindex(
        [_model_display(model_name, window_label) for model_name, window_label in models]
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [MODEL_COLORS.get(model_key, "#4e79a7") for model_key in models]
    bars = ax.bar(ordered["model_display"], ordered["mean_latency_seconds"], color=colors, width=0.6)
    ax.set_xticks(np.arange(len(ordered)))
    for bar, value in zip(bars, ordered["mean_latency_seconds"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + max(ordered["mean_latency_seconds"]) * 0.015,
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticklabels([_model_short(model_name, window_label) for model_name, window_label in models])
    ax.set_ylabel("Mean Detection Latency (s)")
    ax.set_xlabel("Frozen Model")
    ax.set_title("Mean Detection Latency Across LLM-Generated Heldout Attacks")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "latency_vs_model_all_llm_attacks.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_attack_variable_heatmap(heatmap_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    data = heatmap_df.drop(columns=["scenario_count"]).copy()
    category_counts = heatmap_df["scenario_count"].to_dict()
    display_index = [_friendly_category_name(category, int(category_counts[category])) for category in data.index]
    display_columns = [_model_short(model_name, window_label).replace("\n", " ") for model_name, window_label in models]
    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.65 * len(display_index))))
    image = ax.imshow(data.values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(display_columns)))
    ax.set_xticklabels(display_columns, rotation=0)
    ax.set_yticks(np.arange(len(display_index)))
    ax.set_yticklabels(display_index, rotation=0)
    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            value = float(data.iloc[row_index, column_index])
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=9, color="#0f172a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Frozen Model")
    ax.set_ylabel("Attack Variable / Impact Category")
    ax.set_title("Scenario-Level Recall by Attack Variable Category and Frozen Model")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Scenario-Level Detection Recall")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "attack_variable_effect_heatmap_by_model.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_detection_time_distribution(distribution_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    plot_df = distribution_df.copy()
    plot_df["model_display"] = plot_df.apply(lambda row: _model_display(str(row["model_name"]), str(row["window_label"])), axis=1)
    order = [_model_display(model_name, window_label) for model_name, window_label in models]

    fig, ax = plt.subplots(figsize=(9, 6))
    series_list = [plot_df.loc[plot_df["model_display"] == label, "latency_seconds"].dropna().to_numpy() for label in order]
    boxplot = ax.boxplot(series_list, patch_artist=True, showfliers=True, widths=0.55)
    for patch, model_key in zip(boxplot["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(model_key, "#4e79a7"))
        patch.set_alpha(0.85)
    for median in boxplot["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Frozen Model")
    ax.set_ylabel("Detection Latency (s, log scale)")
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([_model_short(model_name, window_label) for model_name, window_label in models])
    ax.set_title("Per-Scenario Detection Latency Distribution Across LLM-Generated Heldout Attacks")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "detection_time_distribution_by_model.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_generator_matrix(generator_matrix_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    ordered_columns = [_model_display(model_name, window_label) for model_name, window_label in models]
    pivot = generator_matrix_df.pivot(index="generator_source", columns="model_display", values="f1").reindex(
        index=LLM_GENERATORS,
        columns=ordered_columns,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(pivot.values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(ordered_columns)))
    ax.set_xticklabels([_model_short(model_name, window_label).replace("\n", " ") for model_name, window_label in models], rotation=0)
    ax.set_yticks(np.arange(len(LLM_GENERATORS)))
    ax.set_yticklabels([label.title() for label in LLM_GENERATORS], rotation=0)
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = float(pivot.iloc[row_index, column_index])
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#0f172a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Frozen Model")
    ax.set_ylabel("Generator Source")
    ax.set_title("Generator-by-Model Heldout Replay F1 Matrix")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Window-Level F1")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "per_generator_performance_matrix.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _load_saved_professor_tables() -> dict[str, pd.DataFrame]:
    return {
        "performance": pd.read_csv(TABLE_ROOT / "performance_vs_model_all_llm_attacks.csv"),
        "latency": pd.read_csv(TABLE_ROOT / "latency_vs_model_all_llm_attacks.csv"),
        "generator_matrix": pd.read_csv(TABLE_ROOT / "per_generator_performance_matrix.csv"),
        "attack_heatmap": pd.read_csv(TABLE_ROOT / "attack_variable_effect_heatmap_by_model.csv", index_col=0),
        "distribution": pd.read_csv(TABLE_ROOT / "detection_time_distribution_by_model.csv"),
        "timeline_selection": pd.read_csv(TABLE_ROOT / "timeline_selection_summary.csv"),
        "scenario_difficulty": pd.read_csv(
            TABLE_ROOT / "llm_scenario_difficulty_summary.csv",
            parse_dates=["start_time_utc", "end_time_utc"],
        ),
    }


def _plot_performance_f1_only(performance_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    ordered_labels = _ordered_model_display_labels(models)
    ordered = performance_df.set_index("model_display").reindex(ordered_labels).reset_index()
    x_positions = np.arange(len(ordered))
    colors = [MODEL_COLORS.get(model_key, "#4e79a7") for model_key in models]

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    bars = ax.bar(x_positions, ordered["f1"], color=colors, width=0.56)
    for bar, value in zip(bars, ordered["f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + 0.025,
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_model_presentation_short(model_name, window_label) for model_name, window_label in models], fontsize=12)
    ax.set_ylim(0.0, 0.98)
    ax.set_ylabel("F1")
    ax.set_xlabel("Frozen Model")
    ax.set_title("Frozen-Model F1 Across LLM-Generated Heldout Attacks")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "performance_f1_only_by_model.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_performance_chart_improved(performance_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    long_df = performance_df.melt(
        id_vars=["model_name", "window_label", "model_display"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    ordered_labels = _ordered_model_display_labels(models)
    x_positions = np.arange(len(ordered_labels))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(11.2, 6.1))
    for index, metric in enumerate(["precision", "recall", "f1"]):
        subset = long_df.loc[long_df["metric"] == metric].set_index("model_display").reindex(ordered_labels).reset_index()
        positions = x_positions + (index - 1) * bar_width
        bars = ax.bar(
            positions,
            subset["score"],
            width=bar_width,
            color=METRIC_COLORS[metric],
            label=metric.capitalize(),
            edgecolor="white",
            linewidth=0.6,
        )
        if metric == "f1":
            for bar, value in zip(bars, subset["score"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    float(value) + 0.02,
                    f"{float(value):.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_model_presentation_short(model_name, window_label) for model_name, window_label in models], fontsize=12)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_xlabel("Frozen Model")
    ax.set_title("Frozen-Model Precision, Recall, and F1 Across Heldout Replay", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_ROOT / "performance_vs_model_all_llm_attacks_improved.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_chart_linear(latency_summary_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    ordered = latency_summary_df.set_index("model_display").reindex(_ordered_model_display_labels(models)).reset_index()
    x_positions = np.arange(len(ordered))
    colors = [MODEL_COLORS.get(model_key, "#4e79a7") for model_key in models]

    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    bars = ax.bar(x_positions, ordered["mean_latency_seconds"], color=colors, width=0.58)
    y_max = float(ordered["mean_latency_seconds"].max()) if not ordered.empty else 1.0
    for bar, value in zip(bars, ordered["mean_latency_seconds"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + y_max * 0.018,
            f"{float(value):.1f} s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_model_presentation_short(model_name, window_label) for model_name, window_label in models], fontsize=12)
    ax.set_ylabel("Mean Detection Latency (s)")
    ax.set_xlabel("Frozen Model")
    ax.set_title("Mean Detection Latency Across LLM-Generated Heldout Attacks")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "latency_vs_model_all_llm_attacks_linear.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_chart_zoomed(latency_summary_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    ordered = latency_summary_df.set_index("model_display").reindex(_ordered_model_display_labels(models)).reset_index()
    x_positions = np.arange(len(ordered))
    colors = [MODEL_COLORS.get(model_key, "#4e79a7") for model_key in models]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.15, 1.0]})
    full_ax, zoom_ax = axes
    bars = full_ax.bar(x_positions, ordered["mean_latency_seconds"], color=colors, width=0.58)
    full_max = float(ordered["mean_latency_seconds"].max()) if not ordered.empty else 1.0
    for bar, value in zip(bars, ordered["mean_latency_seconds"]):
        full_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + full_max * 0.018,
            f"{float(value):.1f} s",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    full_ax.set_xticks(x_positions)
    full_ax.set_xticklabels([_model_presentation_short(model_name, window_label) for model_name, window_label in models], fontsize=11.5)
    full_ax.set_ylabel("Mean Detection Latency (s)")
    full_ax.set_title("Full range")
    full_ax.grid(axis="y", linestyle="--", alpha=0.28)
    full_ax.spines["top"].set_visible(False)
    full_ax.spines["right"].set_visible(False)

    zoom_bars = zoom_ax.bar(x_positions, ordered["mean_latency_seconds"], color=colors, width=0.58)
    zoom_ax.set_ylim(0.0, 80.0)
    for index, (bar, value) in enumerate(zip(zoom_bars, ordered["mean_latency_seconds"])):
        if float(value) <= 80.0:
            zoom_ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(value) + 2.0,
                f"{float(value):.1f} s",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )
        else:
            zoom_ax.text(
                index,
                78.0,
                f"{float(value):.1f} s\ntruncated",
                ha="center",
                va="top",
                fontsize=9,
                color="#444444",
            )
    zoom_ax.set_xticks(x_positions)
    zoom_ax.set_xticklabels([_model_presentation_short(model_name, window_label) for model_name, window_label in models], fontsize=11.5)
    zoom_ax.set_title("Zoomed view (0-80 s)")
    zoom_ax.grid(axis="y", linestyle="--", alpha=0.28)
    zoom_ax.spines["top"].set_visible(False)
    zoom_ax.spines["right"].set_visible(False)

    fig.suptitle("Mean Detection Latency Across LLM-Generated Heldout Attacks", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_ROOT / "latency_vs_model_all_llm_attacks_zoomed.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_chart_horizontal(latency_summary_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    ordered = latency_summary_df.set_index("model_display").reindex(_ordered_model_display_labels(models)).reset_index()
    y_positions = np.arange(len(ordered))
    colors = [MODEL_COLORS.get(model_key, "#4e79a7") for model_key in models]

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    bars = ax.barh(y_positions, ordered["mean_latency_seconds"], color=colors, height=0.55)
    x_max = float(ordered["mean_latency_seconds"].max()) if not ordered.empty else 1.0
    for bar, value in zip(bars, ordered["mean_latency_seconds"]):
        ax.text(
            float(value) + x_max * 0.012,
            bar.get_y() + bar.get_height() / 2.0,
            f"{float(value):.1f} s",
            ha="left",
            va="center",
            fontsize=10,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_model_presentation(model_name, window_label) for model_name, window_label in models], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Detection Latency (s)")
    ax.set_title("Mean Detection Latency Across LLM-Generated Heldout Attacks")
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "latency_vs_model_all_llm_attacks_horizontal.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_generator_matrix_improved(generator_matrix_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    ordered_columns = _ordered_model_display_labels(models)
    pivot = generator_matrix_df.pivot(index="generator_source", columns="model_display", values="f1").reindex(columns=ordered_columns)
    row_means = pivot.mean(axis=1).sort_values()
    pivot = pivot.reindex(index=row_means.index)
    display_rows = [_pretty_generator(row_label) for row_label in pivot.index]
    display_cols = [_model_presentation_short(model_name, window_label).replace("\n", " ") for model_name, window_label in models]

    cmap = LinearSegmentedColormap.from_list(
        "heldout_f1_matrix",
        ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#084594"],
    )
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    image = ax.imshow(pivot.values, cmap=cmap, norm=PowerNorm(gamma=0.75, vmin=0.0, vmax=1.0), aspect="auto")
    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels(display_cols, fontsize=11)
    ax.set_yticks(np.arange(len(display_rows)))
    ax.set_yticklabels(display_rows, fontsize=11)
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = float(pivot.iloc[row_index, column_index])
            text_color = "white" if value >= 0.72 else "#0f172a"
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=10, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Frozen Model")
    ax.set_ylabel("Generator Source")
    ax.set_title("Heldout F1 by Generator and Model")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Window-Level F1")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "per_generator_performance_matrix_improved.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_attack_variable_heatmap_improved(heatmap_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    plot_df = heatmap_df.copy()
    counts = plot_df["scenario_count"].astype(int).to_dict()
    data = plot_df.drop(columns=["scenario_count"]).copy()
    row_means = data.mean(axis=1).sort_values()
    data = data.reindex(index=row_means.index)
    display_rows = [_friendly_category_name(category, counts.get(category)) for category in data.index]
    display_cols = [_model_presentation_short(model_name, window_label).replace("\n", " ") for model_name, window_label in models]

    cmap = LinearSegmentedColormap.from_list(
        "heldout_variable_heat",
        ["#fffaf0", "#fde0c5", "#fcb07e", "#e46f5e", "#8c2d04"],
    )
    fig, ax = plt.subplots(figsize=(10.6, max(5.6, 0.66 * len(display_rows))))
    image = ax.imshow(data.values, cmap=cmap, norm=PowerNorm(gamma=0.72, vmin=0.0, vmax=1.0), aspect="auto")
    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels(display_cols, fontsize=11)
    ax.set_yticks(np.arange(len(display_rows)))
    ax.set_yticklabels(display_rows, fontsize=11)
    if data.shape[0] <= 12 and data.shape[1] <= 5:
        for row_index in range(data.shape[0]):
            for column_index in range(data.shape[1]):
                value = float(data.iloc[row_index, column_index])
                text_color = "white" if value >= 0.78 else "#0f172a"
                ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=10, color=text_color)
                if value < 0.75:
                    ax.add_patch(
                        Rectangle(
                            (column_index - 0.5, row_index - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="#7f1d1d",
                            linewidth=1.6,
                        )
                    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Frozen Model")
    ax.set_ylabel("Attack Variable Group")
    ax.set_title("Heldout Recall by Attack Variable Group")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Scenario-Level Recall")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "attack_variable_effect_heatmap_by_model_improved.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_detection_time_distribution_improved(distribution_df: pd.DataFrame, models: list[tuple[str, str]]) -> None:
    plot_df = distribution_df.copy()
    plot_df["model_display"] = plot_df.apply(lambda row: _model_display(str(row["model_name"]), str(row["window_label"])), axis=1)
    order = _ordered_model_display_labels(models)
    series_list = [plot_df.loc[plot_df["model_display"] == label, "latency_seconds"].dropna().to_numpy() for label in order]

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    boxplot = ax.boxplot(series_list, patch_artist=True, showfliers=True, widths=0.52)
    for patch, model_key in zip(boxplot["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(model_key, "#4e79a7"))
        patch.set_alpha(0.82)
    for median in boxplot["medians"]:
        median.set_color("#111111")
        median.set_linewidth(2.0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([1, 3, 10, 30, 100, 300])
    ax.set_xlabel("Frozen Model")
    ax.set_ylabel("Detection Latency (s, log scale)")
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([_model_presentation_short(model_name, window_label) for model_name, window_label in models], fontsize=11.5)
    ax.set_title("Heldout Detection Latency Distribution by Frozen Model")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for index, series in enumerate(series_list, start=1):
        if len(series) == 0:
            continue
        median_value = float(np.median(series))
        ax.text(index, median_value * 1.12, f"median {median_value:.0f}s", ha="center", va="bottom", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "detection_time_distribution_by_model_improved.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_timeline_figures_improved(models: list[tuple[str, str]]) -> None:
    saved_tables = _load_saved_professor_tables()
    selection_df = saved_tables["timeline_selection"]
    distribution_df = saved_tables["distribution"]
    scenario_meta = saved_tables["scenario_difficulty"].drop_duplicates(subset=["generator_source", "scenario_id"])

    line_styles = {
        ("threshold_baseline", "10s"): ":",
        ("transformer", "60s"): "--",
        ("lstm", "300s"): "-.",
    }

    for index, row in enumerate(selection_df.to_dict(orient="records"), start=1):
        timeline_path = TABLE_ROOT / f"timeline_detection_scenario_{index}.csv"
        if not timeline_path.exists():
            continue
        timeline_df = pd.read_csv(timeline_path)
        meta_row = scenario_meta.loc[
            (scenario_meta["generator_source"] == row["generator_source"])
            & (scenario_meta["scenario_id"] == row["scenario_id"])
        ]
        if meta_row.empty:
            continue
        duration_minutes = float(meta_row["duration_seconds"].iloc[0]) / 60.0
        scenario_latency = distribution_df.loc[
            (distribution_df["generator_source"] == row["generator_source"])
            & (distribution_df["scenario_id"] == row["scenario_id"])
        ].copy()

        fig, ax = plt.subplots(figsize=(12.6, 4.9))
        ax.axvspan(0.0, duration_minutes, color="#dbe4ee", alpha=0.55, label="Attack window", zorder=0)
        ax.plot(
            timeline_df["relative_minutes"],
            timeline_df["clean_value"],
            label="Clean",
            color="#2f2f2f",
            linewidth=2.2,
            linestyle="--",
        )
        ax.plot(
            timeline_df["relative_minutes"],
            timeline_df["attacked_value"],
            label="Attacked",
            color="#355c7d",
            linewidth=2.6,
        )
        ax.axvline(0.0, color="#8b1e3f", linestyle="-", linewidth=2.4, label="Attack start")

        for model_name, window_label in models:
            latency_row = scenario_latency.loc[
                (scenario_latency["model_name"] == model_name)
                & (scenario_latency["window_label"] == window_label)
            ]
            latency_seconds = _safe_float(latency_row["latency_seconds"].iloc[0]) if not latency_row.empty else None
            if latency_seconds is None:
                continue
            ax.axvline(
                latency_seconds / 60.0,
                color=MODEL_COLORS.get((model_name, window_label), "#4e79a7"),
                linestyle=line_styles.get((model_name, window_label), "--"),
                linewidth=2.7,
                label=TIMELINE_LEGEND_LABELS.get((model_name, window_label), _model_presentation(model_name, window_label)),
            )

        handles, labels = ax.get_legend_handles_labels()
        ordered_handles: list[Any] = []
        ordered_labels: list[str] = []
        for label, handle in zip(labels, handles):
            if label not in ordered_labels:
                ordered_labels.append(label)
                ordered_handles.append(handle)
        ax.legend(
            ordered_handles,
            ordered_labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=10,
        )
        ax.set_xlabel("Minutes Relative to Attack Start")
        ax.set_ylabel(str(row["chosen_variable"]))
        ax.set_title(
            f"Scenario {index}: {row['scenario_id']} ({_pretty_generator(str(row['generator_source']))})\n"
            f"Variable: {row['chosen_variable']}"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(rect=(0, 0, 0.84, 1))
        fig.savefig(OUTPUT_ROOT / f"timeline_detection_scenario_{index}_improved.png", dpi=320, bbox_inches="tight")
        plt.close(fig)


def _timeline_detection_label(model_name: str, window_label: str) -> str:
    return TIMELINE_LEGEND_LABELS.get((model_name, window_label), _model_presentation(model_name, window_label))


def _timeline_annotation_label(model_name: str, window_label: str) -> str:
    label = _timeline_detection_label(model_name, window_label)
    return label.replace(" detect", "")


def _choose_seconds_tick_step(span_seconds: float) -> int:
    if span_seconds <= 60:
        return 5
    if span_seconds <= 240:
        return 10
    if span_seconds <= 480:
        return 20
    if span_seconds <= 900:
        return 50
    return 100


def _build_second_ticks(x_min: float, x_max: float) -> np.ndarray:
    span = float(x_max - x_min)
    step = _choose_seconds_tick_step(span)
    start = int(math.floor(x_min / step) * step)
    end = int(math.ceil(x_max / step) * step)
    return np.arange(start, end + step, step, dtype=float)


def _format_range_seconds(x_min: float, x_max: float) -> str:
    return f"[{int(round(x_min))}, {int(round(x_max))}]"


def _plot_single_timeline_seconds_view(
    timeline_df: pd.DataFrame,
    meta_row: pd.Series,
    scenario_latency: pd.DataFrame,
    models: list[tuple[str, str]],
    output_path: Path,
    x_range_seconds: tuple[float, float],
    zoom_suffix: str,
) -> dict[str, Any]:
    plot_df = timeline_df.copy()
    plot_df["relative_seconds"] = plot_df["relative_minutes"].astype(float) * 60.0
    variable_name = str(meta_row["chosen_variable"])
    duration_seconds = float(meta_row["duration_seconds"])

    detection_rows: list[dict[str, Any]] = []
    for model_name, window_label in models:
        latency_row = scenario_latency.loc[
            (scenario_latency["model_name"] == model_name)
            & (scenario_latency["window_label"] == window_label)
        ]
        latency_seconds = _safe_float(latency_row["latency_seconds"].iloc[0]) if not latency_row.empty else None
        if latency_seconds is None:
            continue
        detection_rows.append(
            {
                "model_name": model_name,
                "window_label": window_label,
                "latency_seconds": float(latency_seconds),
                "legend_label": _timeline_detection_label(model_name, window_label),
                "annotation_label": _timeline_annotation_label(model_name, window_label),
                "color": MODEL_COLORS.get((model_name, window_label), "#4e79a7"),
                "linestyle": {
                    ("threshold_baseline", "10s"): ":",
                    ("transformer", "60s"): "--",
                    ("lstm", "300s"): "-.",
                }.get((model_name, window_label), "--"),
            }
        )

    x_min, x_max = x_range_seconds
    ticks = _build_second_ticks(x_min, x_max)
    fig, ax = plt.subplots(figsize=(12.8, 5.1))
    ax.axvspan(0.0, duration_seconds, color="#dbe4ee", alpha=0.45, label="Attack window", zorder=0)
    ax.plot(
        plot_df["relative_seconds"],
        plot_df["clean_value"],
        label="Clean",
        color="#2f2f2f",
        linewidth=2.15,
        linestyle="--",
    )
    ax.plot(
        plot_df["relative_seconds"],
        plot_df["attacked_value"],
        label="Attacked",
        color="#355c7d",
        linewidth=2.5,
    )
    ax.axvline(0.0, color="#8b1e3f", linestyle="-", linewidth=2.5, label="Attack start")

    if 0.0 <= duration_seconds <= x_max:
        ax.text(
            duration_seconds,
            0.985,
            "Attack end",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=9,
            color="#4b5563",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(ticks)
    ax.set_xlabel("Seconds Relative to Attack Start")
    ax.set_ylabel(variable_name)
    ax.set_title(
        f"Scenario {int(meta_row['scenario_index'])}: {meta_row['scenario_id']} ({_pretty_generator(str(meta_row['generator_source']))})\n"
        f"Variable: {variable_name} | {zoom_suffix}"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    anchor_y = y_max - 0.04 * y_range
    outside_notes: list[str] = []
    for annotation_index, detection in enumerate(sorted(detection_rows, key=lambda item: item["latency_seconds"])):
        latency_seconds = float(detection["latency_seconds"])
        if latency_seconds < x_min or latency_seconds > x_max:
            outside_notes.append(f"{detection['annotation_label']}: {latency_seconds:.0f} s outside view")
            continue
        ax.axvline(
            latency_seconds,
            color=detection["color"],
            linestyle=detection["linestyle"],
            linewidth=2.8,
            label=detection["legend_label"],
        )
        x_offset = 8 if annotation_index % 2 == 0 else 10
        y_offset = -(18 + annotation_index * 16)
        ax.annotate(
            f"{detection['annotation_label']}: {latency_seconds:.0f} s",
            xy=(latency_seconds, anchor_y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8.8,
            color=detection["color"],
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": detection["color"], "alpha": 0.9},
        )

    if outside_notes:
        ax.text(
            0.985,
            0.97,
            "\n".join(outside_notes),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.92},
        )

    handles, labels = ax.get_legend_handles_labels()
    ordered_handles: list[Any] = []
    ordered_labels: list[str] = []
    for label, handle in zip(labels, handles):
        if label not in ordered_labels:
            ordered_labels.append(label)
            ordered_handles.append(handle)
    ax.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9.8,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)

    return {
        "x_range_seconds": _format_range_seconds(x_min, x_max),
    }


def regenerate_timeline_seconds_views(models: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Handle regenerate timeline seconds views within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    saved_tables = _load_saved_professor_tables()
    selection_df = saved_tables["timeline_selection"].copy()
    distribution_df = saved_tables["distribution"].copy()
    scenario_meta = saved_tables["scenario_difficulty"].drop_duplicates(subset=["generator_source", "scenario_id"]).copy()
    if models is None:
        models = _discover_complete_models()

    selection_df["scenario_index"] = np.arange(1, len(selection_df) + 1)
    selection_df = selection_df.merge(
        scenario_meta[
            [
                "generator_source",
                "scenario_id",
                "duration_seconds",
            ]
        ],
        on=["generator_source", "scenario_id"],
        how="left",
    )

    summary_rows: list[dict[str, Any]] = []
    for row in selection_df.to_dict(orient="records"):
        scenario_index = int(row["scenario_index"])
        timeline_path = TABLE_ROOT / f"timeline_detection_scenario_{scenario_index}.csv"
        if not timeline_path.exists():
            continue
        timeline_df = pd.read_csv(timeline_path)
        timeline_df["relative_seconds"] = timeline_df["relative_minutes"].astype(float) * 60.0
        scenario_latency = distribution_df.loc[
            (distribution_df["generator_source"] == row["generator_source"])
            & (distribution_df["scenario_id"] == row["scenario_id"])
        ].copy()

        detection_lookup = {
            "threshold_detect_seconds": None,
            "transformer_detect_seconds": None,
            "lstm_detect_seconds": None,
        }
        detection_values: list[float] = []
        for model_name, window_label in models:
            latency_row = scenario_latency.loc[
                (scenario_latency["model_name"] == model_name)
                & (scenario_latency["window_label"] == window_label)
            ]
            latency_seconds = _safe_float(latency_row["latency_seconds"].iloc[0]) if not latency_row.empty else None
            if latency_seconds is None:
                continue
            detection_values.append(float(latency_seconds))
            if (model_name, window_label) == ("threshold_baseline", "10s"):
                detection_lookup["threshold_detect_seconds"] = float(latency_seconds)
            elif (model_name, window_label) == ("transformer", "60s"):
                detection_lookup["transformer_detect_seconds"] = float(latency_seconds)
            elif (model_name, window_label) == ("lstm", "300s"):
                detection_lookup["lstm_detect_seconds"] = float(latency_seconds)

        full_x_min = min(float(timeline_df["relative_seconds"].min()), -5.0)
        full_x_max = max(
            float(timeline_df["relative_seconds"].max()),
            float(row.get("duration_seconds") or 0.0),
            max(detection_values) if detection_values else 0.0,
        )
        full_result = _plot_single_timeline_seconds_view(
            timeline_df=timeline_df,
            meta_row=pd.Series(row),
            scenario_latency=scenario_latency,
            models=models,
            output_path=OUTPUT_ROOT / f"timeline_detection_scenario_{scenario_index}_seconds_full.png",
            x_range_seconds=(full_x_min, full_x_max),
            zoom_suffix="Full-range view",
        )

        early_detection_values = [value for value in detection_values if value <= 240.0]
        if not early_detection_values and detection_values:
            early_detection_values = [min(detection_values)]
        last_detection = max(early_detection_values) if early_detection_values else 30.0
        zoom_x_min = max(float(timeline_df["relative_seconds"].min()), -30.0)
        zoom_x_max = min(
            float(timeline_df["relative_seconds"].max()),
            max(90.0, last_detection + 30.0, min(float(row.get("duration_seconds") or 0.0), 120.0)),
        )
        if zoom_x_max - zoom_x_min < 50.0:
            zoom_x_max = min(float(timeline_df["relative_seconds"].max()), zoom_x_min + 50.0)

        zoom_result = _plot_single_timeline_seconds_view(
            timeline_df=timeline_df,
            meta_row=pd.Series(row),
            scenario_latency=scenario_latency,
            models=models,
            output_path=OUTPUT_ROOT / f"timeline_detection_scenario_{scenario_index}_seconds_zoom.png",
            x_range_seconds=(zoom_x_min, zoom_x_max),
            zoom_suffix="Early-response zoom",
        )

        summary_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "generator_source": row["generator_source"],
                "variable_name": row["chosen_variable"],
                "attack_start_seconds": 0,
                "transformer_detect_seconds": detection_lookup["transformer_detect_seconds"],
                "lstm_detect_seconds": detection_lookup["lstm_detect_seconds"],
                "threshold_detect_seconds": detection_lookup["threshold_detect_seconds"],
                "chosen_full_plot_range_seconds": full_result["x_range_seconds"],
                "chosen_zoom_plot_range_seconds": zoom_result["x_range_seconds"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLE_ROOT / "timeline_detection_seconds_summary.csv", index=False)
    summary_df.to_csv(OUTPUT_ROOT / "timeline_detection_seconds_summary.csv", index=False)

    summary_lines = [
        "# Timeline Seconds Improvement Summary",
        "",
        "- Only axis units, tick labels, and timeline layout were changed.",
        "- The selected scenarios, clean and attacked signal values, attack windows, and model detection timestamps were not changed.",
        "- The new outputs use seconds relative to attack start, with second-based tick spacing and direct detection-delay annotations.",
        "- For live presentation, the zoomed seconds views are the clearest first look because the model delays are readable without mental conversion.",
        "",
        "## Best Timeline To Show Tomorrow",
        "",
        "- Best full-range view: `timeline_detection_scenario_4_seconds_full.png`",
        "- Best zoomed view: `timeline_detection_scenario_1_seconds_zoom.png`",
        "- Recommended first live timeline: `timeline_detection_scenario_1_seconds_zoom.png`",
    ]
    _write_markdown(OUTPUT_ROOT / "TIMELINE_SECONDS_IMPROVEMENT_SUMMARY.md", "\n".join(summary_lines))
    return summary_df

def _timeline_data_for_scenario(
    generator_source: str,
    scenario_row: dict[str, Any],
    clean_df: pd.DataFrame,
    attacked_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str, str]:
    if generator_source not in attacked_cache:
        attacked_path = HELDOUT_ROOT / generator_source / "datasets" / "measured_physical_timeseries.parquet"
        attacked_df = pd.read_parquet(attacked_path).copy()
        attacked_df["timestamp_utc"] = pd.to_datetime(attacked_df["timestamp_utc"], utc=True)
        attacked_cache[generator_source] = attacked_df
    attacked_df = attacked_cache[generator_source]
    variable_name, variable_reason = _choose_timeline_variable(scenario_row, attacked_df, clean_df)
    start_time = pd.Timestamp(scenario_row["start_time_utc"])
    end_time = pd.Timestamp(scenario_row["end_time_utc"])
    duration_seconds = float(scenario_row["duration_seconds"])
    pre_seconds = min(max(duration_seconds * 0.75, 180.0), 900.0)
    post_seconds = min(max(duration_seconds * 0.75, 180.0), 900.0)

    attacked_subset = attacked_df.loc[
        (attacked_df["timestamp_utc"] >= start_time - pd.Timedelta(seconds=pre_seconds))
        & (attacked_df["timestamp_utc"] <= end_time + pd.Timedelta(seconds=post_seconds))
    ][["timestamp_utc", "simulation_index", variable_name]].copy()

    clean_subset = clean_df.loc[
        clean_df["simulation_index"].isin(attacked_subset["simulation_index"])
    ][["simulation_index", variable_name]].copy()
    merged = attacked_subset.merge(clean_subset, on="simulation_index", how="left", suffixes=("_attacked", "_clean"))
    merged["relative_minutes"] = (merged["timestamp_utc"] - start_time).dt.total_seconds() / 60.0
    merged.rename(
        columns={
            f"{variable_name}_attacked": "attacked_value",
            f"{variable_name}_clean": "clean_value",
        },
        inplace=True,
    )
    merged["attacked_value"] = pd.to_numeric(merged["attacked_value"], errors="coerce")
    merged["clean_value"] = pd.to_numeric(merged["clean_value"], errors="coerce")
    return merged, variable_name, variable_reason


def _plot_timeline_figures(
    selected_scenarios: pd.DataFrame,
    latency_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    models: list[tuple[str, str]],
) -> pd.DataFrame:
    clean_df = pd.read_parquet(CLEAN_MEASURED_PATH).copy()
    clean_df["timestamp_utc"] = pd.to_datetime(clean_df["timestamp_utc"], utc=True)
    attacked_cache: dict[str, pd.DataFrame] = {}

    selection_rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected_scenarios.to_dict(orient="records"), start=1):
        inventory_row = inventory_df.loc[
            (inventory_df["generator_source"] == record["generator_source"])
            & (inventory_df["scenario_id"] == record["scenario_id"])
        ].iloc[0].to_dict()

        timeline_df, variable_name, variable_reason = _timeline_data_for_scenario(
            generator_source=str(record["generator_source"]),
            scenario_row=inventory_row,
            clean_df=clean_df,
            attacked_cache=attacked_cache,
        )
        timeline_df.to_csv(TABLE_ROOT / f"timeline_detection_scenario_{index}.csv", index=False)

        scenario_latency = latency_df.loc[
            (latency_df["generator_source"] == record["generator_source"])
            & (latency_df["scenario_id"] == record["scenario_id"])
        ].copy()
        duration_minutes = float(inventory_row["duration_seconds"]) / 60.0

        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(
            timeline_df["relative_minutes"],
            timeline_df["clean_value"],
            label="Clean reference",
            color="#333333",
            linewidth=1.8,
            linestyle="--",
        )
        ax.plot(
            timeline_df["relative_minutes"],
            timeline_df["attacked_value"],
            label="Attacked signal",
            color="#355c7d",
            linewidth=2.1,
        )
        ax.axvspan(0.0, duration_minutes, color="#d9d9d9", alpha=0.35, label="Attack window")
        ax.axvline(0.0, color="#b22222", linestyle="--", linewidth=1.5, label="Attack start")
        for model_name, window_label in models:
            latency_row = scenario_latency.loc[
                (scenario_latency["model_name"] == model_name)
                & (scenario_latency["window_label"] == window_label)
            ]
            if latency_row.empty:
                continue
            latency_minutes = float(latency_row["latency_seconds"].iloc[0]) / 60.0
            ax.axvline(
                latency_minutes,
                color=MODEL_COLORS.get((model_name, window_label), "#4e79a7"),
                linestyle=":",
                linewidth=1.8,
                label=f"{_model_display(model_name, window_label)} detection",
            )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, loc="best", ncol=2)
        ax.set_xlabel("Minutes Relative to Attack Start")
        ax.set_ylabel(variable_name)
        ax.set_title(
            f"Scenario {index}: {record['generator_source']} | {record['scenario_id']} | {variable_name}"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / f"timeline_detection_scenario_{index}.png", dpi=320, bbox_inches="tight")
        plt.close(fig)

        selection_rows.append(
            {
                "scenario_id": record["scenario_id"],
                "generator_source": record["generator_source"],
                "attack_family": record["attack_family"],
                "chosen_variable": variable_name,
                "why_selected": f"{record['selection_reason']}; {variable_reason}",
            }
        )
    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(TABLE_ROOT / "timeline_selection_summary.csv", index=False)
    return selection_df


def _build_caption_drafts(selected_timeline_df: pd.DataFrame, used_balanced: bool) -> None:
    lines = [
        "# Figure Caption Drafts",
        "",
        "All figures use the improved heldout replay outputs as the source of truth. Main cross-model figures use the original accepted LLM-generated heldout bundles (`chatgpt`, `claude`, `gemini`, `grok`) because complete frozen-model coverage exists there. Balanced repaired replay is not used for the cross-model figures because it only exists for the transformer path.",
        "",
        "## performance_vs_model_all_llm_attacks.png",
        "",
        "Paper caption: Window-level precision, recall, and F1 for frozen models aggregated across the accepted original LLM-generated heldout bundles from ChatGPT, Claude, Gemini, and Grok.",
        "Professor explanation: This graph shows which frozen model looked strongest overall when we pooled the LLM-generated heldout attacks together.",
        "Interpretation: The replay generalization winner differs from the benchmark winner, so benchmark model selection and heldout replay transfer should be reported separately.",
        "",
        "## latency_vs_model_all_llm_attacks.png",
        "",
        "Paper caption: Mean detection latency for detected heldout scenarios across the accepted original LLM-generated heldout bundles, reported separately for each frozen model.",
        "Professor explanation: This graph shows which frozen model reacts fastest on the heldout replay bundles.",
        "Interpretation: Faster detection and better heldout transfer are not the same objective, so latency must be discussed alongside F1 rather than replaced by it.",
        "",
        "## attack_variable_effect_heatmap_by_model.png",
        "",
        "Paper caption: Scenario-level detection recall by attack-variable category and frozen model across the accepted original LLM-generated heldout bundles.",
        "Professor explanation: This heatmap shows which kinds of variables or control channels were easier or harder for each frozen model.",
        "Interpretation: Heldout replay performance is not uniform across attack categories; variable-level grouping reveals where each model is strong or weak.",
        "",
        "## detection_time_distribution_by_model.png",
        "",
        "Paper caption: Distribution of per-scenario detection latencies for detected scenarios in the accepted original LLM-generated heldout bundles, shown on a log-scaled latency axis for readability.",
        "Professor explanation: This figure shows whether each model's mean latency is being driven by a tight distribution or by a few very slow detections.",
        "Interpretation: The latency spread matters because some frozen models are stable while others have heavy tails or outlier delays.",
        "",
    ]
    for idx, row in enumerate(selected_timeline_df.to_dict(orient="records"), start=1):
        lines.extend(
            [
                f"## timeline_detection_scenario_{idx}.png",
                "",
                f"Paper caption: Clean versus attacked measured trajectory for `{row['chosen_variable']}` in `{row['scenario_id']}` from `{row['generator_source']}`, with the attack onset and first detection times from the frozen replay models overlaid.",
                f"Professor explanation: This timeline shows what the attacked signal actually did and when the frozen models fired relative to the scenario start.",
                "Interpretation: Representative timelines make the latency tradeoff concrete by linking model detections to the physical or telemetry deviation that each scenario produced.",
                "",
            ]
        )
    lines.extend(
        [
            "## per_generator_performance_matrix.png",
            "",
            "Paper caption: Generator-source by frozen-model F1 matrix for the accepted original LLM-generated heldout bundles.",
            "Professor explanation: This optional matrix shows whether the same frozen model stays strong across all generator sources or only on some of them.",
            "Interpretation: Cross-generator variability should be treated as a core part of the heldout replay story, not as a small side note.",
        ]
    )
    _write_markdown(OUTPUT_ROOT / "figure_caption_drafts.md", "\n".join(lines))


def _build_professor_improved_caption_drafts(selected_timeline_df: pd.DataFrame) -> None:
    lines = [
        "# Improved Figure Caption Drafts",
        "",
        "All improved figures keep the same heldout replay aggregation scope and the same underlying numbers as the original professor graph package. The changes are visual only: layout, labeling, ordering, and readability.",
        "",
        "## performance_f1_only_by_model.png",
        "",
        "Paper caption: Window-level F1 for frozen models aggregated across the accepted original LLM-generated heldout bundles from ChatGPT, Claude, Gemini, and Grok.",
        "Professor explanation: This is the cleanest single chart for overall heldout replay quality.",
        "Takeaway: The LSTM at 300 seconds has the strongest pooled heldout replay F1, followed by the transformer at 60 seconds.",
        "",
        "## performance_vs_model_all_llm_attacks_improved.png",
        "",
        "Paper caption: Window-level precision, recall, and F1 for frozen models aggregated across the accepted original LLM-generated heldout bundles from ChatGPT, Claude, Gemini, and Grok.",
        "Professor explanation: This version keeps all three quality metrics while making the F1 comparison easier to read.",
        "Takeaway: The same heldout replay conclusion remains, but the precision-recall tradeoff is easier to explain alongside F1.",
        "",
        "## latency_vs_model_all_llm_attacks_linear.png",
        "",
        "Paper caption: Mean detection latency in seconds for detected heldout scenarios across the accepted original LLM-generated heldout bundles, shown on a linear scale.",
        "Professor explanation: This is the direct seconds-based latency comparison with no scaling tricks.",
        "Takeaway: Threshold is much slower on average, transformer is fastest, and LSTM sits between them on mean latency.",
        "",
        "## latency_vs_model_all_llm_attacks_zoomed.png",
        "",
        "Paper caption: Mean detection latency in seconds for detected heldout scenarios across the accepted original LLM-generated heldout bundles, shown with a full-range panel and a 0-80 second zoomed panel.",
        "Professor explanation: This is the best chart for showing both the extreme threshold delay and the practical transformer-versus-LSTM gap.",
        "Takeaway: The zoomed panel makes it obvious that transformer remains much faster than LSTM even though LSTM has the stronger pooled replay F1.",
        "",
        "## latency_vs_model_all_llm_attacks_horizontal.png",
        "",
        "Paper caption: Mean detection latency in seconds for detected heldout scenarios across the accepted original LLM-generated heldout bundles, shown as horizontal bars for presentation readability.",
        "Professor explanation: This is an alternate version in case horizontal ranking is easier to discuss live.",
        "Takeaway: The same latency ordering holds regardless of chart orientation.",
        "",
        "## per_generator_performance_matrix_improved.png",
        "",
        "Paper caption: Generator-source by frozen-model F1 matrix for the accepted original LLM-generated heldout bundles, ordered from harder to easier generator sources by mean F1.",
        "Professor explanation: This shows whether one frozen model stays strong across all generator sources or only in some bundles.",
        "Takeaway: Cross-generator variability is part of the story, but the overall LSTM-versus-transformer relationship is still visible.",
        "",
        "## attack_variable_effect_heatmap_by_model_improved.png",
        "",
        "Paper caption: Scenario-level detection recall by attack-variable group and frozen model across the accepted original LLM-generated heldout bundles, ordered from harder to easier groups by mean recall.",
        "Professor explanation: This heatmap shows which variable-impact groups are harder for each model.",
        "Takeaway: Performance is not uniform across variable groups, and the weaker regions are easier to spot in the improved layout.",
        "",
        "## detection_time_distribution_by_model_improved.png",
        "",
        "Paper caption: Distribution of per-scenario detection latencies for detected scenarios in the accepted original LLM-generated heldout bundles, shown on a log-scaled seconds axis with median annotations.",
        "Professor explanation: This shows whether average latency is representative or whether outliers are doing too much of the work.",
        "Takeaway: Latency spread matters, not just the mean, because different frozen models have very different tails.",
        "",
    ]
    for idx, row in enumerate(selected_timeline_df.to_dict(orient="records"), start=1):
        lines.extend(
            [
                f"## timeline_detection_scenario_{idx}_improved.png",
                "",
                f"Paper caption: Clean versus attacked measured trajectory for `{row['chosen_variable']}` in `{row['scenario_id']}` from `{row['generator_source']}`, with the attack window, attack onset, and frozen-model first-detection times overlaid using the same replay outputs as the original timeline.",
                "Professor explanation: This timeline lets us point to one physical trajectory and show exactly when each frozen model reacted.",
                "Takeaway: The improved layout makes the latency tradeoff concrete without changing any underlying detections or timestamps.",
                "",
            ]
        )
    _write_markdown(OUTPUT_ROOT / "figure_caption_drafts_professor_improved.md", "\n".join(lines))


def _build_manifest(models: list[tuple[str, str]], source_notes: dict[str, list[str]], based_on_note: str) -> None:
    manifest = [
        {
            "filename": "performance_vs_model_all_llm_attacks.png",
            "source_tables": source_notes["performance"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Overall frozen-model precision, recall, and F1 across the accepted original LLM-generated heldout bundles.",
        },
        {
            "filename": "latency_vs_model_all_llm_attacks.png",
            "source_tables": source_notes["latency"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Mean detection latency comparison across the accepted original LLM-generated heldout bundles.",
        },
        {
            "filename": "attack_variable_effect_heatmap_by_model.png",
            "source_tables": source_notes["heatmap"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Scenario-level detection recall by attack-variable category and frozen model.",
        },
        {
            "filename": "detection_time_distribution_by_model.png",
            "source_tables": source_notes["distribution"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Distribution of per-scenario detection latencies for detected heldout scenarios.",
        },
        {
            "filename": "per_generator_performance_matrix.png",
            "source_tables": source_notes["generator_matrix"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Optional generator-source by frozen-model F1 matrix.",
        },
        {
            "filename": "performance_f1_only_by_model.png",
            "source_tables": source_notes["performance"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Cleaner meeting-first F1 comparison across the accepted original LLM-generated heldout bundles.",
        },
        {
            "filename": "performance_vs_model_all_llm_attacks_improved.png",
            "source_tables": source_notes["performance"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Improved grouped precision, recall, and F1 comparison with cleaner labels.",
        },
        {
            "filename": "latency_vs_model_all_llm_attacks_linear.png",
            "source_tables": source_notes["latency"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Linear-scale seconds view of mean heldout replay latency.",
        },
        {
            "filename": "latency_vs_model_all_llm_attacks_zoomed.png",
            "source_tables": source_notes["latency"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Full-range plus zoomed heldout replay latency comparison to clarify transformer-versus-LSTM timing.",
        },
        {
            "filename": "latency_vs_model_all_llm_attacks_horizontal.png",
            "source_tables": source_notes["latency"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Presentation-friendly horizontal view of mean heldout replay latency.",
        },
        {
            "filename": "per_generator_performance_matrix_improved.png",
            "source_tables": source_notes["generator_matrix"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Improved generator-source by frozen-model F1 matrix ordered by generator difficulty.",
        },
        {
            "filename": "attack_variable_effect_heatmap_by_model_improved.png",
            "source_tables": source_notes["heatmap"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Improved attack-variable-group heatmap ordered by hardest-to-easiest recall.",
        },
        {
            "filename": "detection_time_distribution_by_model_improved.png",
            "source_tables": source_notes["distribution"],
            "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
            "based_on": based_on_note,
            "purpose": "Improved presentation version of the per-scenario latency distribution.",
        },
    ]
    for index in range(1, 6):
        png_path = OUTPUT_ROOT / f"timeline_detection_scenario_{index}.png"
        if png_path.exists():
            manifest.append(
                {
                    "filename": png_path.name,
                    "source_tables": source_notes["timelines"],
                    "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
                    "based_on": based_on_note,
                    "purpose": "Representative clean-versus-attacked timeline with attack onset and frozen-model detection markers.",
                }
            )
        improved_path = OUTPUT_ROOT / f"timeline_detection_scenario_{index}_improved.png"
        if improved_path.exists():
            manifest.append(
                {
                    "filename": improved_path.name,
                    "source_tables": source_notes["timelines"],
                    "models_included": [_model_display(model_name, window_label) for model_name, window_label in models],
                    "based_on": based_on_note,
                    "purpose": "Improved representative timeline with clearer attack-window shading and frozen-model detection markers.",
                }
            )
    _write_json(OUTPUT_ROOT / "figure_manifest_professor_graphs.json", manifest)


def _build_summary(models: list[tuple[str, str]], balanced_metrics_exists: bool) -> None:
    lines = [
        "# Professor Graph Package Summary",
        "",
        "## Figures Created",
        "",
        "- `performance_vs_model_all_llm_attacks.png`",
        "- `latency_vs_model_all_llm_attacks.png`",
        "- `attack_variable_effect_heatmap_by_model.png`",
        "- `detection_time_distribution_by_model.png`",
        "- `per_generator_performance_matrix.png`",
    ]
    for index in range(1, 6):
        if (OUTPUT_ROOT / f"timeline_detection_scenario_{index}.png").exists():
            lines.append(f"- `timeline_detection_scenario_{index}.png`")
    lines.extend(
        [
            "",
            "## Source Tables Used",
            "",
            "- `outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/predictions.parquet`",
            "- `outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/scenario_summary.csv`",
            "- `outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/latency_table.csv`",
            "- `phase2_llm_benchmark/new_respnse_result/models/<generator>/datasets/attack_labels.parquet`",
            "- `phase2_llm_benchmark/heldout_llm_response/<generator>/new_respnse.json`",
            "- `phase2_llm_benchmark/new_respnse_result/models/<generator>/datasets/measured_physical_timeseries.parquet`",
            "- `outputs/clean/measured_physical_timeseries.parquet`",
            "",
            "## Models Included",
            "",
        ]
    )
    for model_name, window_label in models:
        lines.append(f"- `{_model_display(model_name, window_label)}`")
    lines.extend(
        [
            "",
            "## Original vs Balanced Heldout Basis",
            "",
            "- The main cross-model figures are based on the original improved heldout replay for the accepted LLM-generated bundles from `chatgpt`, `claude`, `gemini`, and `grok`.",
            "- Balanced repaired heldout replay was not used for the cross-model graphs because the balanced repaired path only exists for the transformer and does not provide complete frozen-model coverage.",
        ]
    )
    if balanced_metrics_exists:
        lines.append("- `balanced_heldout_metrics.csv` was retained as context but not used as the main source for cross-model comparisons.")
    lines.extend(
        [
            "",
            "## What Each Graph Is Intended To Show",
            "",
            "- `performance_vs_model_all_llm_attacks.png`: which frozen model performs best overall across the pooled LLM-generated heldout attacks.",
            "- `latency_vs_model_all_llm_attacks.png`: which frozen model is fastest on mean detection time.",
            "- `attack_variable_effect_heatmap_by_model.png`: which attack-variable categories are easier or harder for each model.",
            "- `detection_time_distribution_by_model.png`: how detection-time spread differs by model beyond the mean.",
            "- `timeline_detection_scenario_*.png`: how representative attacked variables evolve and when each model first detects the event.",
            "- `per_generator_performance_matrix.png`: optional generator-source breakdown of F1 to show cross-generator variability.",
        ]
    )
    _write_markdown(OUTPUT_ROOT / "PROFESSOR_GRAPH_PACKAGE_SUMMARY.md", "\n".join(lines))


def _build_improvement_summary() -> None:
    lines = [
        "# Professor Graph Package Improvement Summary",
        "",
        "## What Stayed The Same",
        "",
        "- All metrics, aggregations, and heldout bundle scope stayed fixed.",
        "- No replay outputs were regenerated and no models were retrained.",
        "- The improved figures use the existing `graph_data_tables/` exports and the existing timeline selections as the source of truth.",
        "",
        "## Old Figures Kept",
        "",
        "- `performance_vs_model_all_llm_attacks.png`",
        "- `latency_vs_model_all_llm_attacks.png`",
        "- `attack_variable_effect_heatmap_by_model.png`",
        "- `detection_time_distribution_by_model.png`",
        "- `per_generator_performance_matrix.png`",
        "- `timeline_detection_scenario_1.png` through `timeline_detection_scenario_5.png`",
        "",
        "## Improved Figures Added",
        "",
        "- `performance_f1_only_by_model.png`",
        "- `performance_vs_model_all_llm_attacks_improved.png`",
        "- `latency_vs_model_all_llm_attacks_linear.png`",
        "- `latency_vs_model_all_llm_attacks_zoomed.png`",
        "- `latency_vs_model_all_llm_attacks_horizontal.png`",
        "- `per_generator_performance_matrix_improved.png`",
        "- `attack_variable_effect_heatmap_by_model_improved.png`",
        "- `detection_time_distribution_by_model_improved.png`",
        "- `timeline_detection_scenario_1_improved.png` through `timeline_detection_scenario_5_improved.png`",
        "",
        "## Which Figure To Show First",
        "",
        "- Start with `performance_f1_only_by_model.png`.",
        "",
        "## Recommended Display Order For Tomorrow",
        "",
    ]
    for index, filename in enumerate(IMPROVED_MEETING_ORDER, start=1):
        lines.append(f"{index}. `{filename}`")
    lines.extend(
        [
            "",
            "## Why This Order Works",
            "",
            "- `performance_f1_only_by_model.png`: fastest way to show who leads overall on pooled heldout replay.",
            "- `latency_vs_model_all_llm_attacks_zoomed.png`: immediately clarifies the speed tradeoff without hiding the threshold outlier.",
            "- `per_generator_performance_matrix_improved.png`: shows the cross-generator consistency story.",
            "- `timeline_detection_scenario_1_improved.png`: grounds the discussion in one concrete physical trajectory and one set of detection times.",
        ]
    )
    _write_markdown(OUTPUT_ROOT / "PROFESSOR_GRAPH_PACKAGE_IMPROVEMENT_SUMMARY.md", "\n".join(lines))


def main() -> None:
    """Run the command-line entrypoint for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    _configure_style()

    models = _discover_complete_models()
    inventory_df = _load_scenario_inventory()
    predictions_df, scenario_df, latency_df = _load_predictions_and_summaries(models)
    merged_scenarios = _merge_scenario_metadata(scenario_df, inventory_df)

    performance_df = _performance_table(predictions_df, models)
    latency_summary_df = _latency_table(latency_df, scenario_df, models)
    generator_matrix_df = _build_generator_matrix(predictions_df, models)
    _write_attack_variable_mapping(inventory_df)
    heatmap_df = _attack_variable_heatmap_table(merged_scenarios, models)
    distribution_df = _build_detection_time_distribution(latency_df, inventory_df)
    scenario_difficulty = _build_scenario_difficulty(merged_scenarios, models)
    scenario_difficulty.to_csv(TABLE_ROOT / "llm_scenario_difficulty_summary.csv", index=False)
    selected_scenarios = _select_representative_scenarios(scenario_difficulty)
    timeline_selection_df = _plot_timeline_figures(selected_scenarios, latency_df, inventory_df, models)

    _plot_performance_chart(performance_df, models)
    _plot_latency_chart(latency_summary_df, models)
    _plot_attack_variable_heatmap(heatmap_df, models)
    _plot_detection_time_distribution(distribution_df, models)
    _plot_generator_matrix(generator_matrix_df, models)
    _plot_timeline_figures_improved(models)

    saved_tables = _load_saved_professor_tables()
    _plot_performance_f1_only(saved_tables["performance"], models)
    _plot_performance_chart_improved(saved_tables["performance"], models)
    _plot_latency_chart_linear(saved_tables["latency"], models)
    _plot_latency_chart_zoomed(saved_tables["latency"], models)
    _plot_latency_chart_horizontal(saved_tables["latency"], models)
    _plot_generator_matrix_improved(saved_tables["generator_matrix"], models)
    _plot_attack_variable_heatmap_improved(saved_tables["attack_heatmap"], models)
    _plot_detection_time_distribution_improved(saved_tables["distribution"], models)

    based_on_note = (
        "Original improved heldout replay on accepted LLM-generated bundles from chatgpt, claude, gemini, and grok; canonical bundle and human-authored bundle excluded from the main graphs."
    )
    source_notes = {
        "performance": [
            "multi_model_heldout_metrics.csv (model discovery only)",
            "outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/predictions.parquet",
            "graph_data_tables/performance_vs_model_all_llm_attacks.csv",
        ],
        "latency": [
            "outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/latency_table.csv",
            "graph_data_tables/latency_vs_model_all_llm_attacks.csv",
        ],
        "heatmap": [
            "outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/scenario_summary.csv",
            "phase2_llm_benchmark/new_respnse_result/models/<generator>/datasets/attack_labels.parquet",
            "phase2_llm_benchmark/heldout_llm_response/<generator>/new_respnse.json",
            "graph_data_tables/attack_variable_effect_heatmap_by_model.csv",
            "graph_data_tables/attack_variable_group_mapping.csv",
        ],
        "distribution": [
            "outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/latency_table.csv",
            "graph_data_tables/detection_time_distribution_by_model.csv",
        ],
        "generator_matrix": [
            "outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/predictions.parquet",
            "graph_data_tables/per_generator_performance_matrix.csv",
        ],
        "timelines": [
            "phase2_llm_benchmark/new_respnse_result/models/<generator>/datasets/measured_physical_timeseries.parquet",
            "outputs/clean/measured_physical_timeseries.parquet",
            "outputs/window_size_study/improved_phase3/evaluations/<generator>__existing_heldout_phase2_bundle__<dataset_id>/<window>/<model>/latency_table.csv",
            "graph_data_tables/timeline_selection_summary.csv",
        ],
    }
    _build_manifest(models, source_notes, based_on_note)
    _build_caption_drafts(timeline_selection_df, used_balanced=False)
    _build_professor_improved_caption_drafts(timeline_selection_df)
    _build_summary(models, balanced_metrics_exists=BALANCED_METRICS_PATH.exists())
    _build_improvement_summary()


if __name__ == "__main__":
    main()
