"""Phase 3 evaluation and analysis support for DERGuardian.

This module implements experiment utils logic for detector evaluation, ablations,
zero-day-like heldout synthetic analysis, latency sweeps, or final reporting.
It keeps benchmark, replay, heldout synthetic, and extension results separated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import WindowConfig
from common.io_utils import ensure_dir, read_dataframe, write_dataframe, write_json
from phase1_models.model_loader import load_phase1_package, run_phase1_inference
from phase1.build_windows import build_merged_windows
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload, detection_latency_table, per_scenario_metrics
from phase1_models.neural_models import GRUClassifier, LSTMClassifier, TinyTransformerClassifier, TokenBaselineClassifier
from phase1_models.residual_dataset import (
    WINDOW_META_COLUMNS,
    build_aligned_residual_dataframe,
    canonical_residual_artifact_path,
    canonical_window_source_paths,
    load_persisted_residual_dataframe,
    residual_artifact_is_fresh,
    residual_artifact_is_usable,
)
from phase1_models.run_full_evaluation import (
    FullRunData,
    MODEL_PRIORITY,
    _plot_latency_by_scenario,
    _plot_model_comparison,
    _to_markdown,
    assign_attack_aware_split,
    prepare_full_run_data,
    rank_features_by_effect_size,
    run_autoencoder,
    run_isolation_forest,
    run_sequence_model,
    run_threshold_baseline,
)


PHYSICAL_PREFIXES = (
    "env_",
    "feeder_",
    "bus_",
    "line_",
    "pv_",
    "bess_",
    "load_",
    "regulator_",
    "switch_",
    "capacitor_",
    "breaker_",
    "derived_",
)
CYBER_PREFIXES = ("cyber_",)
PHASE3_MODEL_PRIORITY = [name for name in MODEL_PRIORITY]


@dataclass(slots=True)
class PreparedWindowBundle:
    """Structured object used by the Phase 3 evaluation workflow."""

    clean_windows: pd.DataFrame
    attacked_windows: pd.DataFrame
    labels_df: pd.DataFrame
    residual_df: pd.DataFrame
    clean_windows_path: Path
    attacked_windows_path: Path
    labels_path: Path
    residual_artifact_path: Path
    residual_artifact_used: bool


def load_optional_phase2_analysis_artifacts(root: Path) -> dict[str, object]:
    """Load optional Phase 2 analysis reports for interpretation only.

    These artifacts are not required detector inputs for Phase 3. They are
    bounded reporting layers that can support experiment discussion, scenario
    selection, and paper writing without changing the canonical model inputs.
    """

    reports_root = root / "outputs" / "reports"
    payload: dict[str, object] = {}
    effect_path = reports_root / "phase2_effect_summary.json"
    difficulty_path = reports_root / "phase2_scenario_difficulty.json"
    if effect_path.exists():
        payload["effect_summary"] = json.loads(effect_path.read_text(encoding="utf-8"))
    if difficulty_path.exists():
        payload["scenario_difficulty"] = json.loads(difficulty_path.read_text(encoding="utf-8"))
    return payload


def load_labels(root: Path, labels_path: Path | None = None) -> pd.DataFrame:
    """Load labels for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    resolved = labels_path or (root / "outputs" / "attacked" / "attack_labels.parquet")
    labels = read_dataframe(resolved)
    if not labels.empty:
        labels["start_time_utc"] = pd.to_datetime(labels["start_time_utc"], utc=True)
        labels["end_time_utc"] = pd.to_datetime(labels["end_time_utc"], utc=True)
    return labels


def prepare_window_bundle(
    root: Path,
    clean_windows_path: Path | None = None,
    attacked_windows_path: Path | None = None,
    labels_path: Path | None = None,
) -> PreparedWindowBundle:
    """Handle prepare window bundle within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    clean_path = clean_windows_path or (root / "outputs" / "windows" / "merged_windows_clean.parquet")
    attacked_path = attacked_windows_path or (root / "outputs" / "attacked" / "merged_windows.parquet")
    resolved_labels_path = labels_path or (root / "outputs" / "attacked" / "attack_labels.parquet")
    labels = load_labels(root, resolved_labels_path)
    clean = read_dataframe(clean_path)
    attacked = read_dataframe(attacked_path)
    for frame in [clean, attacked]:
        frame["window_start_utc"] = pd.to_datetime(frame["window_start_utc"], utc=True)
        frame["window_end_utc"] = pd.to_datetime(frame["window_end_utc"], utc=True)
    use_canonical_cache = clean_windows_path is None and attacked_windows_path is None and labels_path is None
    residual_path = canonical_residual_artifact_path(root)
    source_paths = canonical_window_source_paths(root)
    residual_artifact_used = False
    if use_canonical_cache and residual_artifact_is_usable(residual_path) and residual_artifact_is_fresh(residual_path, source_paths):
        residual_df = load_persisted_residual_dataframe(residual_path).drop(columns=["split_name"], errors="ignore")
        residual_artifact_used = True
    else:
        residual_df = build_residual_dataframe(clean, attacked, labels)
    return PreparedWindowBundle(
        clean_windows=clean,
        attacked_windows=attacked,
        labels_df=labels,
        residual_df=residual_df,
        clean_windows_path=clean_path,
        attacked_windows_path=attacked_path,
        labels_path=resolved_labels_path,
        residual_artifact_path=residual_path,
        residual_artifact_used=residual_artifact_used,
    )


def build_residual_dataframe(clean_windows: pd.DataFrame, attacked_windows: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Build residual dataframe for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return build_aligned_residual_dataframe(clean_windows, attacked_windows, labels_df).drop(columns=["split_name"], errors="ignore")


def annotate_scenario_windows(frame: pd.DataFrame, labels_df: pd.DataFrame, default_scenario: str = "benign") -> pd.DataFrame:
    """Handle annotate scenario windows within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    annotated = frame.copy()
    if "scenario_window_id" not in annotated.columns:
        annotated["scenario_window_id"] = default_scenario
    else:
        annotated["scenario_window_id"] = annotated["scenario_window_id"].fillna(default_scenario).astype(str)
    if labels_df.empty:
        return annotated
    for _, label in labels_df.iterrows():
        mask = (annotated["window_start_utc"] < label["end_time_utc"]) & (annotated["window_end_utc"] > label["start_time_utc"])
        annotated.loc[mask, "scenario_window_id"] = str(label["scenario_id"])
    return annotated


def build_full_run_data(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    split_df: pd.DataFrame,
    candidate_feature_columns: list[str],
    residual_artifact_path: Path | None = None,
) -> FullRunData:
    """Build full run data for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    ranking = rank_features_by_effect_size(
        split_df[split_df["split_name"] == "train"].copy(),
        candidate_feature_columns,
    )
    return FullRunData(
        residual_df=feature_df.copy(),
        labels_df=labels_df.copy(),
        split_df=split_df.copy(),
        feature_ranking=ranking,
        residual_artifact_path=residual_artifact_path or canonical_residual_artifact_path(ROOT),
    )


def infer_candidate_columns(frame: pd.DataFrame, mode: str = "residual") -> list[str]:
    """Handle infer candidate columns within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and column not in WINDOW_META_COLUMNS and column != "split_name"
    ]
    if mode == "residual":
        return [column for column in numeric_columns if column.startswith("delta__") or column.startswith("fdi__")]
    if mode == "physical_only":
        return [column for column in numeric_columns if _is_physical_feature(column)]
    if mode == "cyber_only":
        return [column for column in numeric_columns if _is_cyber_feature(column)]
    if mode == "fused":
        return [column for column in numeric_columns if _is_physical_feature(column) or _is_cyber_feature(column)]
    if mode == "fused_plus_residual":
        return [
            column
            for column in numeric_columns
            if _is_physical_feature(column) or _is_cyber_feature(column) or column.startswith("delta__") or column.startswith("fdi__")
        ]
    raise ValueError(f"Unsupported feature mode: {mode}")


def build_modality_frame(bundle: PreparedWindowBundle, mode: str) -> pd.DataFrame:
    """Build modality frame for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    attacked = annotate_scenario_windows(bundle.attacked_windows.copy(), bundle.labels_df)
    if mode == "physical_only":
        selected = infer_candidate_columns(attacked, mode="physical_only")
        return attacked[_metadata_columns(attacked) + selected].copy()
    if mode == "cyber_only":
        selected = infer_candidate_columns(attacked, mode="cyber_only")
        return attacked[_metadata_columns(attacked) + selected].copy()
    if mode == "fused":
        selected = infer_candidate_columns(attacked, mode="fused")
        return attacked[_metadata_columns(attacked) + selected].copy()
    if mode == "fused_plus_residual":
        attacked_selected = attacked[_metadata_columns(attacked) + infer_candidate_columns(attacked, mode="fused")].copy()
        residual_selected = bundle.residual_df[["window_start_utc"] + infer_candidate_columns(bundle.residual_df, mode="residual")].copy()
        merged = attacked_selected.merge(residual_selected, on="window_start_utc", how="inner")
        return merged
    raise ValueError(f"Unsupported modality frame mode: {mode}")


def build_window_bundle_for_config(root: Path, window_seconds: int, step_seconds: int, output_dir: Path) -> tuple[Path, Path]:
    """Build window bundle for config for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output = output_dir / f"clean_windows_{window_seconds}s_{step_seconds}s.parquet"
    attacked_output = output_dir / f"attacked_windows_{window_seconds}s_{step_seconds}s.parquet"
    if clean_output.exists() and attacked_output.exists():
        return clean_output, attacked_output

    clean_measured = read_dataframe(root / "outputs" / "clean" / "measured_physical_timeseries.parquet")
    clean_cyber = read_dataframe(root / "outputs" / "clean" / "cyber_events.parquet")
    attacked_measured = read_dataframe(root / "outputs" / "attacked" / "measured_physical_timeseries.parquet")
    attacked_cyber = read_dataframe(root / "outputs" / "attacked" / "cyber_events.parquet")
    labels_df = load_labels(root)
    empty_labels = pd.DataFrame(columns=labels_df.columns)
    windows = WindowConfig(window_seconds=window_seconds, step_seconds=step_seconds)
    clean_windows = build_merged_windows(clean_measured, clean_cyber, empty_labels, windows)
    attacked_windows = build_merged_windows(attacked_measured, attacked_cyber, labels_df, windows)
    write_dataframe(clean_windows, clean_output, fmt="parquet")
    write_dataframe(attacked_windows, attacked_output, fmt="parquet")
    return clean_output, attacked_output


def run_model_suite(
    root: Path,
    full_data: FullRunData,
    report_root_name: str,
    model_root_name: str,
    include_models: list[str],
    model_settings: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Run model suite for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    report_root = ensure_dir(root / "outputs" / "reports" / report_root_name)
    summary_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    per_scenario_frames: list[pd.DataFrame] = []
    latency_frames: list[pd.DataFrame] = []
    figure_manifest: list[dict[str, object]] = []
    for model_name in include_models:
        settings = model_settings.get(model_name, {})
        feature_counts = _as_int_list(settings.get("feature_counts"), default=[32, 64, 96])
        seq_lens = _as_int_list(settings.get("seq_lens"), default=[4, 8, 12])
        epochs = int(settings.get("epochs", 12))
        patience = int(settings.get("patience", 4))
        if model_name == "threshold_baseline":
            result = run_threshold_baseline(root, full_data, feature_counts, report_root, model_root_name, report_root_name)
        elif model_name == "autoencoder":
            result = run_autoencoder(root, full_data, feature_counts, epochs, patience, report_root, model_root_name, report_root_name)
        elif model_name == "isolation_forest":
            result = run_isolation_forest(root, full_data, feature_counts, report_root, model_root_name, report_root_name)
        elif model_name == "gru":
            result = run_sequence_model(
                root,
                full_data,
                model_name="gru",
                model_ctor=lambda input_dim: GRUClassifier(input_dim=input_dim, hidden_dim=64),
                feature_counts=feature_counts,
                seq_lens=seq_lens,
                epochs=epochs,
                patience=patience,
                token_input=False,
                report_root=report_root,
                model_root_name=model_root_name,
                artifact_root_name=report_root_name,
            )
        elif model_name == "transformer":
            result = run_sequence_model(
                root,
                full_data,
                model_name="transformer",
                model_ctor=lambda input_dim: TinyTransformerClassifier(input_dim=input_dim, d_model=64, nhead=4, num_layers=2),
                feature_counts=feature_counts,
                seq_lens=seq_lens,
                epochs=epochs,
                patience=patience,
                token_input=False,
                report_root=report_root,
                model_root_name=model_root_name,
                artifact_root_name=report_root_name,
            )
        elif model_name == "lstm":
            result = run_sequence_model(
                root,
                full_data,
                model_name="lstm",
                model_ctor=lambda input_dim: LSTMClassifier(input_dim=input_dim, hidden_dim=64),
                feature_counts=feature_counts,
                seq_lens=seq_lens,
                epochs=epochs,
                patience=patience,
                token_input=False,
                report_root=report_root,
                model_root_name=model_root_name,
                artifact_root_name=report_root_name,
            )
        elif model_name == "llm_baseline":
            result = run_sequence_model(
                root,
                full_data,
                model_name="llm_baseline",
                model_ctor=lambda _: TokenBaselineClassifier(vocab_size=16, embed_dim=32),
                feature_counts=feature_counts,
                seq_lens=seq_lens,
                epochs=epochs,
                patience=patience,
                token_input=True,
                report_root=report_root,
                model_root_name=model_root_name,
                artifact_root_name=report_root_name,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        summary_rows.append(result["summary"])
        sweep_rows.extend(result["sweep_rows"])
        per_scenario_frames.append(result["per_scenario"])
        latency_frames.append(result["latency"])
        figure_manifest.extend(result["figures"])
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["priority_rank"] = summary_df["model_name"].apply(
            lambda name: PHASE3_MODEL_PRIORITY.index(name) if name in PHASE3_MODEL_PRIORITY else len(PHASE3_MODEL_PRIORITY)
        )
        summary_df = summary_df.sort_values(["recall", "precision", "f1", "priority_rank"], ascending=[False, False, False, True]).reset_index(drop=True)
        summary_df = summary_df.drop(columns=["priority_rank"])
    return {
        "summary_df": summary_df,
        "sweep_df": pd.DataFrame(sweep_rows),
        "per_scenario_df": pd.concat(per_scenario_frames, ignore_index=True) if per_scenario_frames else pd.DataFrame(),
        "latency_df": pd.concat(latency_frames, ignore_index=True) if latency_frames else pd.DataFrame(),
        "figure_manifest": figure_manifest,
        "report_root": report_root,
    }


def run_saved_phase1_package(
    *,
    root: Path,
    full_data: FullRunData,
    package_dir: Path,
    holdout_scenario: str,
) -> dict[str, object]:
    """Run saved phase1 package for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    package = load_phase1_package(package_dir)
    test_frame = full_data.split_df[full_data.split_df["split_name"] == "test"].copy().reset_index(drop=True)
    inference = run_phase1_inference(package, test_frame)
    predictions = inference["predictions"].copy()
    threshold = float(inference["metadata"]["threshold"])
    model_name = str(inference["metadata"]["model_name"])
    metrics = compute_binary_metrics(
        predictions["attack_present"].to_numpy(),
        predictions["score"].to_numpy(),
        threshold,
    )
    curves = compute_curve_payload(
        predictions["attack_present"].to_numpy(),
        predictions["score"].to_numpy(),
    )
    latency = detection_latency_table(predictions, full_data.labels_df, model_name)
    per_scenario = per_scenario_metrics(predictions, full_data.labels_df, model_name)
    summary = {
        "model_name": model_name,
        "holdout_scenario": holdout_scenario,
        **metrics,
        "threshold": threshold,
    }
    return {
        "summary": summary,
        "per_scenario": per_scenario,
        "latency": latency,
        "predictions": predictions,
        "curves": curves,
        "package_manifest": package.manifest,
    }


def summarize_latency(latency_df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Handle summarize latency within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if latency_df.empty:
        return pd.DataFrame(columns=group_columns + ["mean_latency_seconds", "median_latency_seconds", "detected_scenarios", "total_scenarios"])
    frame = latency_df.copy()
    frame["detected_flag"] = frame["latency_seconds"].notna().astype(int)
    summary = (
        frame.groupby(group_columns, observed=False)
        .agg(
            mean_latency_seconds=("latency_seconds", "mean"),
            median_latency_seconds=("latency_seconds", "median"),
            detected_scenarios=("detected_flag", "sum"),
            total_scenarios=("scenario_id", "count"),
        )
        .reset_index()
    )
    return summary


def mean_metric_summary(summary_df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Handle mean metric summary within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if summary_df.empty:
        return pd.DataFrame()
    numeric_columns = [
        column
        for column in summary_df.columns
        if column not in set(group_columns) and pd.api.types.is_numeric_dtype(summary_df[column])
    ]
    aggregated = summary_df.groupby(group_columns, observed=False)[numeric_columns].mean().reset_index()
    if "holdout_scenario" in summary_df.columns:
        counts = summary_df.groupby(group_columns, observed=False)["holdout_scenario"].nunique().reset_index(name="holdout_scenario_count")
        aggregated = aggregated.merge(counts, on=group_columns, how="left")
    return aggregated


def write_phase3_report(path: Path, title: str, sections: list[tuple[str, str]]) -> Path:
    """Write phase3 report for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    lines = [title, ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append(body)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def plot_metric_bars(frame: pd.DataFrame, category_column: str, path: Path, metrics: list[str], title: str) -> Path:
    """Handle plot metric bars within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if frame.empty:
        ax.text(0.5, 0.5, "No results available", ha="center", va="center")
    else:
        x = np.arange(len(frame))
        width = 0.18 if len(metrics) >= 4 else 0.22
        for idx, metric in enumerate(metrics):
            ax.bar(x + idx * width, frame[metric].fillna(0.0), width=width, label=metric)
        ax.set_xticks(x + width * (len(metrics) - 1) / 2.0)
        ax.set_xticklabels(frame[category_column], rotation=30, ha="right")
        ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_latency_tradeoff(frame: pd.DataFrame, x_column: str, y_column: str, label_column: str, path: Path, title: str) -> Path:
    """Handle plot latency tradeoff within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    fig, ax = plt.subplots(figsize=(7, 5))
    if frame.empty:
        ax.text(0.5, 0.5, "No results available", ha="center", va="center")
    else:
        for _, row in frame.iterrows():
            ax.scatter(float(row.get(x_column, np.nan)), float(row.get(y_column, np.nan)), s=80)
            ax.annotate(str(row.get(label_column, "")), (float(row.get(x_column, 0.0)), float(row.get(y_column, 0.0))), fontsize=8)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_summary_bundle(
    summary_df: pd.DataFrame,
    per_scenario_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    report_root: Path,
    prefix: str,
) -> dict[str, Path]:
    """Handle save summary bundle within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    paths = {
        "summary_csv": report_root / f"{prefix}_model_summary.csv",
        "summary_md": report_root / f"{prefix}_model_summary.md",
        "per_scenario_csv": report_root / f"{prefix}_per_scenario_metrics.csv",
        "latency_csv": report_root / f"{prefix}_latency_analysis.csv",
        "sweep_csv": report_root / f"{prefix}_sweep_table.csv",
    }
    summary_df.to_csv(paths["summary_csv"], index=False)
    paths["summary_md"].write_text(_to_markdown(summary_df), encoding="utf-8")
    per_scenario_df.to_csv(paths["per_scenario_csv"], index=False)
    latency_df.to_csv(paths["latency_csv"], index=False)
    sweep_df.to_csv(paths["sweep_csv"], index=False)
    return paths


def clone_best_model_figures(source_report_dir: Path, destination_dir: Path, prefix: str) -> list[dict[str, str]]:
    """Handle clone best model figures within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    copied: list[dict[str, str]] = []
    destination_dir.mkdir(parents=True, exist_ok=True)
    for figure_name in ["confusion_matrix.png", "precision_recall_curve.png", "roc_curve.png"]:
        source = source_report_dir / figure_name
        if not source.exists():
            continue
        destination = destination_dir / f"{prefix}_{figure_name}"
        destination.write_bytes(source.read_bytes())
        copied.append({"source": str(source), "path": str(destination)})
    return copied


def to_markdown(frame: pd.DataFrame) -> str:
    """Handle to markdown within the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return _to_markdown(frame)


def _metadata_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "window_start_utc",
        "window_end_utc",
        "window_seconds",
        "scenario_id",
        "run_id",
        "split_id",
        "attack_present",
        "attack_family",
        "attack_severity",
        "attack_affected_assets",
        "scenario_window_id",
    ]
    return [column for column in preferred if column in frame.columns]


def _is_physical_feature(column: str) -> bool:
    base = column[6:] if column.startswith("delta__") else column
    return base.startswith(PHYSICAL_PREFIXES)


def _is_cyber_feature(column: str) -> bool:
    base = column[6:] if column.startswith("delta__") else column
    return base.startswith(CYBER_PREFIXES)


def _as_int_list(value: object, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parsed = [int(item) for item in value.split(",") if item.strip()]
        return parsed or list(default)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]
