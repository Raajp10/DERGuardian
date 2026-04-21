"""Run the Phase 1 detector window-size benchmark study.

The script rebuilds residual windows for selected time scales, trains/evaluates
the feasible detector set, writes benchmark artifacts, and packages the selected
ready-model outputs. It operates on the canonical benchmark path and must remain
separate from replay, heldout synthetic, and extension-only experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import WindowConfig
from common.io_utils import write_dataframe, write_json
from common.time_alignment import reconstruct_nominal_timestamps
from phase1.build_windows import build_merged_windows
from phase1_models.metrics import compute_curve_payload, compute_binary_metrics, detection_latency_table, per_scenario_metrics
from phase1_models.neural_models import GRUClassifier, LSTMClassifier, TinyTransformerClassifier, TokenBaselineClassifier
from phase1_models.ready_package_utils import build_split_summary
from phase1_models.residual_dataset import build_aligned_residual_dataframe
import phase1_models.ready_package_utils as ready_package_utils
import phase1_models.run_full_evaluation as full_eval


OUTPUT_ROOT = ROOT / "outputs" / "window_size_study"
WINDOW_SPECS = [
    {"tag": "5s", "window_seconds": 5, "step_seconds": 1},
    {"tag": "10s", "window_seconds": 10, "step_seconds": 2},
    {"tag": "60s", "window_seconds": 60, "step_seconds": 12},
    {"tag": "300s", "window_seconds": 300, "step_seconds": 60},
]
FEATURE_COUNTS = [32, 64, 96, 128]
SEQ_LENS = [4, 8, 12]
EPOCHS = 24
PATIENCE = 5
BUFFER_WINDOWS = 2
MIN_ATTACK_OVERLAP = 0.2

WINDOW_META_COLUMNS = {
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
}

CANONICAL_SOURCE_PATHS = {
    "clean_measured": ROOT / "outputs" / "clean" / "measured_physical_timeseries.parquet",
    "clean_cyber": ROOT / "outputs" / "clean" / "cyber_events.parquet",
    "attacked_measured": ROOT / "outputs" / "attacked" / "measured_physical_timeseries.parquet",
    "attacked_cyber": ROOT / "outputs" / "attacked" / "cyber_events.parquet",
    "attack_labels": ROOT / "outputs" / "attacked" / "attack_labels.parquet",
}


def main() -> None:
    """CLI entrypoint for the strict canonical window-size study."""

    parser = argparse.ArgumentParser(description="Run the strict canonical window-size study.")
    parser.add_argument("--window-sizes", default="5,10,60,300")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    selected_specs = select_window_specs(args.window_sizes)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_audit = validate_window_sources(selected_specs)
    write_run_manifest(selected_specs, source_audit)

    clean_measured = pd.read_parquet(CANONICAL_SOURCE_PATHS["clean_measured"])
    clean_cyber = pd.read_parquet(CANONICAL_SOURCE_PATHS["clean_cyber"])
    attacked_measured = pd.read_parquet(CANONICAL_SOURCE_PATHS["attacked_measured"])
    attacked_cyber = pd.read_parquet(CANONICAL_SOURCE_PATHS["attacked_cyber"])
    attack_labels = pd.read_parquet(CANONICAL_SOURCE_PATHS["attack_labels"])
    if not attack_labels.empty:
        attack_labels["start_time_utc"] = pd.to_datetime(attack_labels["start_time_utc"], utc=True)
        attack_labels["end_time_utc"] = pd.to_datetime(attack_labels["end_time_utc"], utc=True)
    empty_labels = attack_labels.iloc[0:0].copy()

    dataset_summary_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    final_best_rows: list[dict[str, Any]] = []
    best_prediction_frames: dict[str, pd.DataFrame] = {}
    best_per_scenario_frames: dict[str, pd.DataFrame] = {}
    best_curve_payloads: dict[str, dict[str, list[float]]] = {}
    for spec in selected_specs:
        print(f"[window-size-study] building window datasets for {spec['tag']}", flush=True)
        window_root = OUTPUT_ROOT / spec["tag"]
        data_root = window_root / "data"
        reports_root = window_root / "reports"
        figures_root = window_root / "figures"
        data_root.mkdir(parents=True, exist_ok=True)
        reports_root.mkdir(parents=True, exist_ok=True)
        figures_root.mkdir(parents=True, exist_ok=True)

        clean_windows = build_merged_windows(
            measured_df=clean_measured,
            cyber_df=clean_cyber,
            labels_df=empty_labels,
            windows=WindowConfig(
                window_seconds=spec["window_seconds"],
                step_seconds=spec["step_seconds"],
                min_attack_overlap_fraction=MIN_ATTACK_OVERLAP,
            ),
        )
        # Clean and attacked windows are regenerated per window size, but the
        # final comparison remains canonical benchmark evidence only.
        attacked_windows = build_merged_windows(
            measured_df=attacked_measured,
            cyber_df=attacked_cyber,
            labels_df=attack_labels,
            windows=WindowConfig(
                window_seconds=spec["window_seconds"],
                step_seconds=spec["step_seconds"],
                min_attack_overlap_fraction=MIN_ATTACK_OVERLAP,
            ),
        )

        clean_windows_path = data_root / "merged_windows_clean.parquet"
        attacked_windows_path = data_root / "merged_windows_attacked.parquet"
        write_dataframe(clean_windows, clean_windows_path, fmt="parquet")
        write_dataframe(attacked_windows, attacked_windows_path, fmt="parquet")

        dataset_summary_rows.extend(
            [
                dataset_summary_row(spec, "clean", clean_windows),
                dataset_summary_row(spec, "attacked", attacked_windows),
            ]
        )

        print(f"[window-size-study] building residual dataset for {spec['tag']}", flush=True)
        full_data = build_full_run_data_for_window(spec, clean_windows, attacked_windows, attack_labels, window_root)
        full_eval._write_diagnostic_tables(full_data, reports_root, FEATURE_COUNTS, window_seconds=spec["window_seconds"])
        write_json(
            {
                "window_seconds": spec["window_seconds"],
                "step_seconds": spec["step_seconds"],
                "feature_counts": FEATURE_COUNTS,
                "seq_lens": SEQ_LENS,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "buffer_windows": BUFFER_WINDOWS,
                "min_attack_overlap_fraction": MIN_ATTACK_OVERLAP,
                "selection_rule": "Canonical per-model validation search. Cross-window best-model selection uses highest test F1, then recall, precision, average_precision.",
            },
            window_root / "benchmark_config.json",
        )

        if args.skip_existing and (reports_root / "model_summary.csv").exists():
            print(f"[window-size-study] skipping existing model run for {spec['tag']}", flush=True)
            model_summary = pd.read_csv(reports_root / "model_summary.csv")
            all_window_rows.extend(model_summary.to_dict(orient="records"))
            per_scenario_existing = pd.read_csv(reports_root / "per_scenario_metrics.csv") if (reports_root / "per_scenario_metrics.csv").exists() else pd.DataFrame()
            best_row = select_best_model_row(model_summary)
            if best_row:
                final_best_rows.append(best_row)
                predictions_path = ROOT / "outputs" / "window_size_study" / spec["tag"] / "models" / str(best_row["model_name"]) / "predictions.parquet"
                if predictions_path.exists():
                    predictions = pd.read_parquet(predictions_path)
                    best_prediction_frames[spec["tag"]] = predictions
                    best_curve_payloads[spec["tag"]] = compute_curve_payload(
                        predictions["attack_present"].to_numpy(),
                        predictions["score"].to_numpy(),
                    )
                if not per_scenario_existing.empty:
                    best_per_scenario_frames[spec["tag"]] = per_scenario_existing[per_scenario_existing["model_name"] == best_row["model_name"]].copy()
            continue

        print(f"[window-size-study] training canonical model suite for {spec['tag']}", flush=True)
        model_rows, sweep_rows, per_scenario_rows, latency_rows, best_row, best_predictions, best_curves = run_models_for_window(
            spec=spec,
            full_data=full_data,
            reports_root=reports_root,
        )
        model_summary_df = pd.DataFrame(model_rows)
        all_sweeps_df = pd.DataFrame(sweep_rows)
        per_scenario_df = pd.DataFrame(per_scenario_rows)
        latency_df = pd.DataFrame(latency_rows)

        model_summary_df.to_csv(reports_root / "model_summary.csv", index=False)
        all_sweeps_df.to_csv(reports_root / "model_sweep_results.csv", index=False)
        per_scenario_df.to_csv(reports_root / "per_scenario_metrics.csv", index=False)
        latency_df.to_csv(reports_root / "latency_metrics.csv", index=False)
        (reports_root / "window_run_report.md").write_text(build_window_run_report(spec, model_summary_df, per_scenario_df, latency_df), encoding="utf-8")

        all_window_rows.extend(model_rows)
        if best_row:
            final_best_rows.append(best_row)
            best_prediction_frames[spec["tag"]] = best_predictions
            best_per_scenario_frames[spec["tag"]] = per_scenario_df[per_scenario_df["model_name"] == best_row["model_name"]].copy()
            best_curve_payloads[spec["tag"]] = best_curves

    dataset_summary_df = pd.DataFrame(dataset_summary_rows)
    dataset_summary_df.to_csv(OUTPUT_ROOT / "window_dataset_summary.csv", index=False)

    final_best_df = pd.DataFrame(final_best_rows).sort_values("window_seconds").reset_index(drop=True)
    attacked_summary = dataset_summary_df[dataset_summary_df["dataset_kind"] == "attacked"][
        ["window_label", "row_count", "column_count", "attack_window_count", "benign_window_count", "attack_rate", "feature_count"]
    ].copy()
    if not final_best_df.empty and not attacked_summary.empty:
        final_best_df = final_best_df.merge(attacked_summary, on="window_label", how="left")
    final_best_df.to_csv(OUTPUT_ROOT / "final_window_comparison.csv", index=False)
    (OUTPUT_ROOT / "final_window_comparison.md").write_text(build_final_window_comparison_md(final_best_df), encoding="utf-8")
    (OUTPUT_ROOT / "window_size_interpretation.md").write_text(build_window_size_interpretation(final_best_df), encoding="utf-8")

    figures_root = OUTPUT_ROOT / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    figure_manifest = build_final_figures(
        figures_root=figures_root,
        comparison_df=final_best_df,
        dataset_summary_df=dataset_summary_df,
        best_prediction_frames=best_prediction_frames,
        best_curve_payloads=best_curve_payloads,
        best_per_scenario_frames=best_per_scenario_frames,
    )
    write_json(figure_manifest, figures_root / "figure_manifest.json")

    best_package_info = package_best_model(final_best_df)
    phase2_inventory_df, phase2_comparison_df, phase2_md, phase2_figure_manifest = build_phase2_bundle_outputs(best_package_info)
    phase2_inventory_df.to_csv(OUTPUT_ROOT / "phase2_bundle_inventory.csv", index=False)
    phase2_comparison_df.to_csv(OUTPUT_ROOT / "phase2_bundle_comparison.csv", index=False)
    (OUTPUT_ROOT / "phase2_bundle_comparison.md").write_text(phase2_md, encoding="utf-8")
    write_json(phase2_figure_manifest, OUTPUT_ROOT / "phase2_bundle_figures" / "figure_manifest.json")

    write_top_level_deliverables(
        final_best_df=final_best_df,
        all_model_rows=pd.DataFrame(all_window_rows),
        phase2_comparison_df=phase2_comparison_df,
        phase2_comparison_md=phase2_md,
        best_package_info=best_package_info,
        dataset_summary_df=dataset_summary_df,
    )


def select_window_specs(window_sizes: str) -> list[dict[str, Any]]:
    """Select window specs for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    requested = {int(item.strip()) for item in window_sizes.split(",") if item.strip()}
    specs = [spec for spec in WINDOW_SPECS if spec["window_seconds"] in requested]
    if not specs:
        raise ValueError("No supported window sizes were requested.")
    return specs


def write_run_manifest(selected_specs: list[dict[str, Any]], source_audit: dict[str, Any]) -> None:
    """Write run manifest for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    write_json(
        {
            "selected_windows": selected_specs,
            "feature_counts": FEATURE_COUNTS,
            "seq_lens": SEQ_LENS,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "buffer_windows": BUFFER_WINDOWS,
            "min_attack_overlap_fraction": MIN_ATTACK_OVERLAP,
            "window_step_policy": "20 percent of window size, rounded to integer seconds, with a floor of 1 second. This preserves the canonical 300s->60s overlap ratio.",
            "canonical_sources": {
                "clean_measured": "outputs/clean/measured_physical_timeseries.parquet",
                "clean_cyber": "outputs/clean/cyber_events.parquet",
                "attacked_measured": "outputs/attacked/measured_physical_timeseries.parquet",
                "attacked_cyber": "outputs/attacked/cyber_events.parquet",
                "attack_labels": "outputs/attacked/attack_labels.parquet",
            },
            "source_validation_rule": "Physical source files must be pre-window time-series artifacts, not merged window files, and the inferred physical sampling interval must be less than or equal to the smallest requested window size.",
            "measured_timestamp_policy": "Physical windows are built on analysis_timestamp_utc reconstructed from timestamp_utc + simulation_index + sample_rate_seconds when those columns are present.",
            "detected_source_audit": source_audit,
        },
        OUTPUT_ROOT / "run_manifest.json",
    )


def validate_window_sources(selected_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate window sources for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    minimum_window_seconds = min(int(spec["window_seconds"]) for spec in selected_specs)
    print(
        f"[window-size-study] validating canonical pre-window sources for minimum requested window {minimum_window_seconds}s",
        flush=True,
    )
    measured_audit = {
        "clean_measured": audit_physical_source(CANONICAL_SOURCE_PATHS["clean_measured"]),
        "attacked_measured": audit_physical_source(CANONICAL_SOURCE_PATHS["attacked_measured"]),
    }
    cyber_audit = {
        "clean_cyber": audit_event_source(CANONICAL_SOURCE_PATHS["clean_cyber"]),
        "attacked_cyber": audit_event_source(CANONICAL_SOURCE_PATHS["attacked_cyber"]),
    }

    for label, audit in measured_audit.items():
        relative_path = str(Path(audit["path"]).relative_to(ROOT))
        print(
            (
                f"[window-size-study] source {label}: {relative_path} "
                f"nominal_interval={audit['nominal_interval_seconds']}s "
                f"observed_timestamp_interval={audit['observed_timestamp_interval_seconds']}s "
                f"windowed={audit['is_windowed']} "
                f"timestamp_column={audit['effective_timestamp_column']}"
            ),
            flush=True,
        )
        if audit["is_windowed"]:
            raise SystemExit(
                f"Invalid window-size study source: {relative_path} is already windowed or aggregated. "
                "Use the pre-window measured_physical_timeseries parquet instead."
            )
        interval_seconds = audit["nominal_interval_seconds"]
        if interval_seconds is None:
            raise SystemExit(
                f"Unable to infer the physical sampling interval for {relative_path}. "
                "The study cannot continue reproducibly."
            )
        if float(interval_seconds) > float(minimum_window_seconds):
            raise SystemExit(
                f"Invalid source resolution: {relative_path} has inferred physical interval {interval_seconds}s, "
                f"which is coarser than the smallest requested window {minimum_window_seconds}s."
            )

    for label, audit in cyber_audit.items():
        relative_path = str(Path(audit["path"]).relative_to(ROOT))
        print(
            (
                f"[window-size-study] source {label}: {relative_path} "
                f"event_stream={audit['event_stream']} "
                f"timestamp_mode_seconds={audit['timestamp_delta_mode_seconds']}"
            ),
            flush=True,
        )
        if audit["is_windowed"]:
            raise SystemExit(
                f"Invalid cyber source: {relative_path} is already windowed or aggregated. "
                "Use the canonical pre-window cyber_events parquet instead."
            )

    return {
        "minimum_requested_window_seconds": minimum_window_seconds,
        "measured": measured_audit,
        "cyber": cyber_audit,
        "attack_labels": str(CANONICAL_SOURCE_PATHS["attack_labels"].relative_to(ROOT)),
    }


def audit_physical_source(path: Path) -> dict[str, Any]:
    """Handle audit physical source within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    probe_columns = [column for column in ["timestamp_utc", "simulation_index", "sample_rate_seconds", "window_start_utc", "window_end_utc"] if column in columns]
    probe = pd.read_parquet(path, columns=probe_columns)
    if "timestamp_utc" in probe.columns:
        probe["timestamp_utc"] = pd.to_datetime(probe["timestamp_utc"], utc=True, errors="coerce")
    is_windowed = (
        "window_start_utc" in columns
        or "window_end_utc" in columns
        or any("__mean" in column or "__std" in column or "__last" in column for column in columns)
    )

    observed_interval = None
    if "timestamp_utc" in probe.columns and len(probe) > 1:
        deltas = probe["timestamp_utc"].sort_values().diff().dropna().dt.total_seconds()
        if not deltas.empty:
            observed_interval = float(deltas.mode().iloc[0])

    nominal_interval = None
    if {"timestamp_utc", "simulation_index", "sample_rate_seconds"}.issubset(probe.columns) and len(probe) > 1:
        analysis = reconstruct_nominal_timestamps(probe)
        deltas = analysis.sort_values().diff().dropna().dt.total_seconds()
        if not deltas.empty:
            nominal_interval = float(deltas.mode().iloc[0])
    elif "sample_rate_seconds" in probe.columns:
        sample_rate = pd.to_numeric(probe["sample_rate_seconds"], errors="coerce").dropna()
        if not sample_rate.empty:
            nominal_interval = float(sample_rate.mode().iloc[0])
    else:
        nominal_interval = observed_interval

    return {
        "path": str(path),
        "row_count": int(parquet.metadata.num_rows),
        "column_count": int(len(columns)),
        "effective_timestamp_column": "analysis_timestamp_utc" if {"timestamp_utc", "simulation_index", "sample_rate_seconds"}.issubset(probe.columns) else "timestamp_utc",
        "raw_timestamp_column": "timestamp_utc" if "timestamp_utc" in columns else None,
        "nominal_interval_seconds": nominal_interval,
        "observed_timestamp_interval_seconds": observed_interval,
        "is_windowed": bool(is_windowed),
        "has_simulation_index": "simulation_index" in columns,
        "has_sample_rate_seconds": "sample_rate_seconds" in columns,
    }


def audit_event_source(path: Path) -> dict[str, Any]:
    """Handle audit event source within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    probe_columns = [column for column in ["timestamp_utc", "window_start_utc", "window_end_utc"] if column in columns]
    probe = pd.read_parquet(path, columns=probe_columns)
    if "timestamp_utc" in probe.columns:
        probe["timestamp_utc"] = pd.to_datetime(probe["timestamp_utc"], utc=True, errors="coerce")

    deltas = pd.Series(dtype=float)
    if "timestamp_utc" in probe.columns and len(probe) > 1:
        deltas = probe["timestamp_utc"].sort_values().diff().dropna().dt.total_seconds()
    is_windowed = (
        "window_start_utc" in columns
        or "window_end_utc" in columns
        or any("__mean" in column or "__std" in column or "__last" in column for column in columns)
    )

    return {
        "path": str(path),
        "row_count": int(parquet.metadata.num_rows),
        "column_count": int(len(columns)),
        "effective_timestamp_column": "timestamp_utc" if "timestamp_utc" in columns else None,
        "timestamp_delta_mode_seconds": float(deltas.mode().iloc[0]) if not deltas.empty else None,
        "timestamp_delta_median_seconds": float(deltas.median()) if not deltas.empty else None,
        "event_stream": True,
        "is_windowed": bool(is_windowed),
    }


def dataset_summary_row(spec: dict[str, Any], dataset_kind: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Handle dataset summary row within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    feature_columns = [column for column in frame.columns if column not in WINDOW_META_COLUMNS]
    attack_count = int(frame["attack_present"].sum()) if "attack_present" in frame.columns else 0
    return {
        "window_label": spec["tag"],
        "dataset_kind": dataset_kind,
        "window_seconds": spec["window_seconds"],
        "step_seconds": spec["step_seconds"],
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "attack_window_count": attack_count,
        "benign_window_count": int(len(frame) - attack_count),
        "attack_rate": float(attack_count / len(frame)) if len(frame) else 0.0,
        "feature_count": int(len(feature_columns)),
        "feature_columns": "|".join(feature_columns),
    }


def build_full_run_data_for_window(
    spec: dict[str, Any],
    clean_windows: pd.DataFrame,
    attacked_windows: pd.DataFrame,
    attack_labels: pd.DataFrame,
    window_root: Path,
) -> full_eval.FullRunData:
    """Build full run data for window for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    residual_df = build_aligned_residual_dataframe(clean_windows, attacked_windows, attack_labels)
    split_df = full_eval.assign_attack_aware_split(residual_df, attack_labels, buffer_windows=BUFFER_WINDOWS)
    residual_path = window_root / "residual_windows.parquet"
    split_summary_path = window_root / "split_summary.json"
    feature_ranking_path = window_root / "feature_ranking.csv"
    write_dataframe(split_df.sort_values("window_start_utc").reset_index(drop=True), residual_path, fmt="parquet")
    write_json(build_split_summary(split_df), split_summary_path)
    feature_columns = [column for column in residual_df.columns if column.startswith("delta__")]
    feature_ranking = full_eval.rank_features_by_effect_size(split_df[split_df["split_name"] == "train"], feature_columns)
    feature_ranking.to_csv(feature_ranking_path, index=False)
    return full_eval.FullRunData(
        residual_df=residual_df,
        labels_df=attack_labels,
        split_df=split_df,
        feature_ranking=feature_ranking,
        residual_artifact_path=residual_path,
    )


def run_models_for_window(
    spec: dict[str, Any],
    full_data: full_eval.FullRunData,
    reports_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, pd.DataFrame, dict[str, list[float]]]:
    """Run models for window for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    model_root_name = str(Path("window_size_study") / spec["tag"] / "models")
    artifact_root_name = str(Path("..") / "window_size_study" / spec["tag"] / "reports")
    ready_root_name = str(Path("window_size_study") / spec["tag"] / "ready_packages")
    model_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    per_scenario_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []

    suite = [
        (
            "threshold_baseline",
            lambda: full_eval.run_threshold_baseline(
                ROOT,
                full_data,
                feature_counts=FEATURE_COUNTS,
                report_root=reports_root,
                model_root_name=model_root_name,
                artifact_root_name=artifact_root_name,
            ),
        ),
        (
            "isolation_forest",
            lambda: full_eval.run_isolation_forest(
                ROOT,
                full_data,
                feature_counts=FEATURE_COUNTS,
                report_root=reports_root,
                model_root_name=model_root_name,
                artifact_root_name=artifact_root_name,
            ),
        ),
        (
            "autoencoder",
            lambda: full_eval.run_autoencoder(
                ROOT,
                full_data,
                feature_counts=FEATURE_COUNTS,
                epochs=EPOCHS,
                patience=PATIENCE,
                report_root=reports_root,
                model_root_name=model_root_name,
                artifact_root_name=artifact_root_name,
            ),
        ),
        (
            "gru",
            lambda: full_eval.run_sequence_model(
                ROOT,
                full_data,
                model_name="gru",
                model_ctor=lambda input_dim: GRUClassifier(input_dim=input_dim, hidden_dim=64),
                feature_counts=FEATURE_COUNTS,
                seq_lens=SEQ_LENS,
                epochs=EPOCHS,
                patience=PATIENCE,
                token_input=False,
                report_root=reports_root,
                model_root_name=model_root_name,
                artifact_root_name=artifact_root_name,
            ),
        ),
        (
            "lstm",
            lambda: full_eval.run_sequence_model(
                ROOT,
                full_data,
                model_name="lstm",
                model_ctor=lambda input_dim: LSTMClassifier(input_dim=input_dim, hidden_dim=64),
                feature_counts=FEATURE_COUNTS,
                seq_lens=SEQ_LENS,
                epochs=EPOCHS,
                patience=PATIENCE,
                token_input=False,
                report_root=reports_root,
                model_root_name=model_root_name,
                artifact_root_name=artifact_root_name,
            ),
        ),
        (
            "transformer",
            lambda: full_eval.run_sequence_model(
                ROOT,
                full_data,
                model_name="transformer",
                model_ctor=lambda input_dim: TinyTransformerClassifier(input_dim=input_dim, d_model=64, nhead=4, num_layers=2),
                feature_counts=FEATURE_COUNTS,
                seq_lens=SEQ_LENS,
                epochs=EPOCHS,
                patience=PATIENCE,
                token_input=False,
                report_root=reports_root,
                model_root_name=model_root_name,
                artifact_root_name=artifact_root_name,
            ),
        ),
        (
            "llm_baseline",
            lambda: full_eval.run_sequence_model(
                ROOT,
                full_data,
                model_name="llm_baseline",
                model_ctor=lambda _: TokenBaselineClassifier(vocab_size=16, embed_dim=32),
                feature_counts=FEATURE_COUNTS,
                seq_lens=SEQ_LENS,
                epochs=EPOCHS,
                patience=PATIENCE,
                token_input=True,
                report_root=reports_root,
                model_root_name=model_root_name,
                artifact_root_name=artifact_root_name,
            ),
        ),
    ]

    with ready_model_root_override(ready_root_name):
        for model_name, runner in suite:
            existing = load_existing_model_result(spec, full_data, model_name, model_root_name, ready_root_name)
            if existing is not None:
                print(f"[window-size-study] reusing existing {model_name} artifacts for {spec['tag']}", flush=True)
                model_rows.append(existing["summary"])
                successful.append(
                    {
                        "summary": existing["summary"],
                        "predictions": existing["predictions"],
                        "per_scenario": existing["per_scenario"],
                        "curves": existing["curves"],
                    }
                )
                sweep_rows.extend(existing["sweep_rows"])
                per_scenario_rows.extend(existing["per_scenario"].to_dict(orient="records"))
                latency_rows.extend(existing["latency"].to_dict(orient="records"))
                continue
            print(f"[window-size-study] running {model_name} for {spec['tag']}", flush=True)
            try:
                result = runner()
                if model_name == "threshold_baseline":
                    threshold_inference_time = fix_threshold_baseline_inference_time(spec, full_data)
                    result["model_info"]["inference_time_seconds"] = threshold_inference_time
                summary = dict(result["summary"])
                summary.update(
                    {
                        "window_label": spec["tag"],
                        "window_seconds": spec["window_seconds"],
                        "step_seconds": spec["step_seconds"],
                        "status": "completed",
                        "training_time_seconds": float(result["model_info"].get("training_time_seconds") or 0.0),
                        "inference_time_seconds": float(result["model_info"].get("inference_time_seconds") or 0.0),
                        "inference_time_ms_per_prediction": (
                            1000.0 * float(result["model_info"].get("inference_time_seconds") or 0.0) / max(len(result["predictions"]), 1)
                        ),
                        "parameter_count": int(result["model_info"].get("parameter_count") or 0),
                        "mean_detection_latency_seconds": safe_mean(result["latency"]["latency_seconds"]) if not result["latency"].empty else math.nan,
                        "median_detection_latency_seconds": safe_median(result["latency"]["latency_seconds"]) if not result["latency"].empty else math.nan,
                        "scenario_f1_mean": safe_mean(result["per_scenario"]["f1"]) if not result["per_scenario"].empty else math.nan,
                        "scenario_f1_std": safe_std(result["per_scenario"]["f1"]) if not result["per_scenario"].empty else math.nan,
                        "model_dir": str(ROOT / "outputs" / model_root_name / model_name),
                        "report_dir": str(ROOT / "outputs" / "window_size_study" / spec["tag"] / "reports" / model_name),
                        "ready_package_dir": str(result["ready_package_dir"]),
                        "error": "",
                    }
                )
                model_rows.append(summary)
                successful.append(
                    {
                        "summary": summary,
                        "predictions": result["predictions"],
                        "per_scenario": result["per_scenario"],
                        "curves": compute_curve_payload(
                            result["predictions"]["attack_present"].to_numpy(),
                            result["predictions"]["score"].to_numpy(),
                        ),
                    }
                )
                sweep_rows.extend(append_window_fields(spec, result["sweep_rows"]))
                per_scenario_rows.extend(append_window_fields(spec, result["per_scenario"].to_dict(orient="records")))
                latency_rows.extend(append_window_fields(spec, result["latency"].to_dict(orient="records")))
            except Exception as exc:
                failure_path = reports_root / f"{model_name}_failure.txt"
                failure_path.write_text(traceback.format_exc(), encoding="utf-8")
                model_rows.append(
                    {
                        "window_label": spec["tag"],
                        "window_seconds": spec["window_seconds"],
                        "step_seconds": spec["step_seconds"],
                        "model_name": model_name,
                        "status": "failed",
                        "precision": math.nan,
                        "recall": math.nan,
                        "f1": math.nan,
                        "roc_auc": math.nan,
                        "average_precision": math.nan,
                        "macro_precision": math.nan,
                        "macro_recall": math.nan,
                        "weighted_precision": math.nan,
                        "weighted_recall": math.nan,
                        "threshold": math.nan,
                        "feature_count": math.nan,
                        "seq_len": math.nan,
                        "training_time_seconds": math.nan,
                        "inference_time_seconds": math.nan,
                        "inference_time_ms_per_prediction": math.nan,
                        "parameter_count": math.nan,
                        "mean_detection_latency_seconds": math.nan,
                        "median_detection_latency_seconds": math.nan,
                        "scenario_f1_mean": math.nan,
                        "scenario_f1_std": math.nan,
                        "model_dir": "",
                        "report_dir": str(reports_root / model_name),
                        "ready_package_dir": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    best_success = select_best_success(successful)
    if best_success is None:
        return model_rows, sweep_rows, per_scenario_rows, latency_rows, None, pd.DataFrame(), {}
    return (
        model_rows,
        sweep_rows,
        per_scenario_rows,
        latency_rows,
        best_success["summary"],
        best_success["predictions"],
        best_success["curves"],
    )


def load_existing_model_result(
    spec: dict[str, Any],
    full_data: full_eval.FullRunData,
    model_name: str,
    model_root_name: str,
    ready_root_name: str,
) -> dict[str, Any] | None:
    """Load existing model result for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    model_dir = ROOT / "outputs" / "window_size_study" / spec["tag"] / "models" / model_name
    predictions_path = model_dir / "predictions.parquet"
    results_path = model_dir / "results.json"
    if not predictions_path.exists() or not results_path.exists():
        return None
    predictions = pd.read_parquet(predictions_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    model_info = dict(results.get("model_info", {}))
    metrics = dict(results.get("metrics", {}))
    if model_name == "threshold_baseline" and float(model_info.get("inference_time_seconds") or 0.0) <= 0.0:
        model_info["inference_time_seconds"] = fix_threshold_baseline_inference_time(spec, full_data)
    threshold = float(metrics.get("threshold") or model_info.get("threshold") or 0.5)
    if not {"attack_present", "score"}.issubset(predictions.columns):
        return None
    if "predicted" not in predictions.columns:
        predictions["predicted"] = (predictions["score"].astype(float) >= threshold).astype(int)
    latency = detection_latency_table(predictions, full_data.labels_df, model_name)
    per_scenario = per_scenario_metrics(predictions, full_data.labels_df, model_name)
    latency = pd.DataFrame(append_window_fields(spec, latency.to_dict(orient="records")))
    per_scenario = pd.DataFrame(append_window_fields(spec, per_scenario.to_dict(orient="records")))
    summary = {
        "window_label": spec["tag"],
        "window_seconds": spec["window_seconds"],
        "step_seconds": spec["step_seconds"],
        "model_name": model_name,
        "status": "completed",
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "weighted_precision": metrics.get("weighted_precision"),
        "weighted_recall": metrics.get("weighted_recall"),
        "threshold": threshold,
        "feature_count": len(model_info.get("feature_columns", [])),
        "seq_len": model_info.get("seq_len"),
        "training_time_seconds": float(model_info.get("training_time_seconds") or 0.0),
        "inference_time_seconds": float(model_info.get("inference_time_seconds") or 0.0),
        "inference_time_ms_per_prediction": 1000.0 * float(model_info.get("inference_time_seconds") or 0.0) / max(len(predictions), 1),
        "parameter_count": int(model_info.get("parameter_count") or 0),
        "mean_detection_latency_seconds": safe_mean(latency["latency_seconds"]) if not latency.empty else math.nan,
        "median_detection_latency_seconds": safe_median(latency["latency_seconds"]) if not latency.empty else math.nan,
        "scenario_f1_mean": safe_mean(per_scenario["f1"]) if not per_scenario.empty else math.nan,
        "scenario_f1_std": safe_std(per_scenario["f1"]) if not per_scenario.empty else math.nan,
        "model_dir": str(model_dir),
        "report_dir": str(ROOT / "outputs" / "window_size_study" / spec["tag"] / "reports" / model_name),
        "ready_package_dir": str(ROOT / "outputs" / ready_root_name / model_name),
        "error": "",
    }
    return {
        "summary": summary,
        "predictions": predictions,
        "per_scenario": per_scenario,
        "latency": latency,
        "curves": compute_curve_payload(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy()),
        "sweep_rows": [],
    }


@contextmanager
def ready_model_root_override(temp_root: str):
    """Handle ready model root override within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    previous = ready_package_utils.READY_MODEL_ROOT
    ready_package_utils.READY_MODEL_ROOT = temp_root
    try:
        yield
    finally:
        ready_package_utils.READY_MODEL_ROOT = previous


def fix_threshold_baseline_inference_time(spec: dict[str, Any], full_data: full_eval.FullRunData) -> float:
    """Handle fix threshold baseline inference time within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    model_path = ROOT / "outputs" / "window_size_study" / spec["tag"] / "models" / "threshold_baseline" / "model.pkl"
    if not model_path.exists():
        return math.nan
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    test_frame = full_data.split_df[full_data.split_df["split_name"] == "test"].copy().reset_index(drop=True)
    feature_columns = list(payload["model_info"]["feature_columns"])
    started = time.perf_counter()
    raw_scores = full_eval._threshold_score(test_frame, feature_columns, payload["standardizer"])
    _ = full_eval.apply_calibrator(payload["calibrator"], raw_scores)
    elapsed = time.perf_counter() - started
    payload["model_info"]["inference_time_seconds"] = float(elapsed)
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    results_path = model_path.parent / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        results.setdefault("model_info", {})["inference_time_seconds"] = float(elapsed)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return float(elapsed)


def append_window_fields(spec: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Handle append window fields within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    enriched: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["window_label"] = spec["tag"]
        payload["window_seconds"] = spec["window_seconds"]
        payload["step_seconds"] = spec["step_seconds"]
        enriched.append(payload)
    return enriched


def select_best_success(successful: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select best success for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if not successful:
        return None
    return max(
        successful,
        key=lambda item: (
            safe_float(item["summary"].get("f1")),
            safe_float(item["summary"].get("recall")),
            safe_float(item["summary"].get("precision")),
            safe_float(item["summary"].get("average_precision")),
        ),
    )


def select_best_model_row(model_summary: pd.DataFrame) -> dict[str, Any] | None:
    """Select best model row for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if model_summary.empty:
        return None
    successful = model_summary[model_summary["status"] == "completed"].copy()
    if successful.empty:
        return None
    successful = successful.sort_values(["f1", "recall", "precision", "average_precision"], ascending=False)
    return successful.iloc[0].to_dict()


def safe_float(value: Any) -> float:
    """Handle safe float within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    try:
        numeric = float(value)
    except Exception:
        return float("-inf")
    return numeric if np.isfinite(numeric) else float("-inf")


def safe_mean(series: pd.Series) -> float:
    """Handle safe mean within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return float(cleaned.mean()) if not cleaned.empty else math.nan


def safe_median(series: pd.Series) -> float:
    """Handle safe median within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return float(cleaned.median()) if not cleaned.empty else math.nan


def safe_std(series: pd.Series) -> float:
    """Handle safe std within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    return float(cleaned.std(ddof=0)) if not cleaned.empty else math.nan


def build_window_run_report(
    spec: dict[str, Any],
    model_summary_df: pd.DataFrame,
    per_scenario_df: pd.DataFrame,
    latency_df: pd.DataFrame,
) -> str:
    """Build window run report for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    lines = [f"# Window Run Report: {spec['tag']}", ""]
    lines.append(f"- window_seconds: `{spec['window_seconds']}`")
    lines.append(f"- step_seconds: `{spec['step_seconds']}`")
    lines.append("- status: canonical benchmark suite over rebuilt aligned-residual windows")
    lines.append("")
    lines.append("## Model Summary")
    lines.append(to_markdown(model_summary_df))
    lines.append("")
    lines.append("## Per-Scenario Metrics")
    lines.append(to_markdown(per_scenario_df))
    lines.append("")
    lines.append("## Latency Metrics")
    lines.append(to_markdown(latency_df))
    lines.append("")
    return "\n".join(lines) + "\n"


def build_final_window_comparison_md(final_best_df: pd.DataFrame) -> str:
    """Build final window comparison md for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    lines = ["# Final Window Comparison", ""]
    lines.append("## Best Model Per Window")
    lines.append(to_markdown(final_best_df))
    lines.append("")
    return "\n".join(lines) + "\n"


def build_window_size_interpretation(final_best_df: pd.DataFrame) -> str:
    """Build window size interpretation for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if final_best_df.empty:
        return "# Window Size Interpretation\n\nNo successful runs were available.\n"
    best_overall = final_best_df.sort_values(["f1", "recall", "precision", "average_precision"], ascending=False).iloc[0]
    fastest = final_best_df.sort_values(["mean_detection_latency_seconds", "inference_time_seconds"], ascending=True).iloc[0]
    stable = final_best_df.sort_values(["scenario_f1_std", "scenario_f1_mean"], ascending=[True, False]).iloc[0]
    slowest = final_best_df.sort_values("mean_detection_latency_seconds", ascending=False).iloc[0]
    lines = ["# Window Size Interpretation", ""]
    lines.append(f"- Best overall window size: `{best_overall['window_label']}` with best model `{best_overall['model_name']}` and test F1 `{best_overall['f1']}`.")
    lines.append(f"- Fastest window size: `{fastest['window_label']}` by mean detection latency `{fastest['mean_detection_latency_seconds']}` seconds.")
    lines.append(f"- Most stable window size: `{stable['window_label']}` with lowest per-scenario F1 spread `{stable['scenario_f1_std']}`.")
    lines.append(
        f"- The slowest response among the best-per-window setups is `{slowest['window_label']}` at mean detection latency `{slowest['mean_detection_latency_seconds']}` seconds."
    )
    lines.append("")
    lines.append("## Plain-English Tradeoff")
    lines.append("- Smaller windows react faster because each detector sees shorter slices of the stream and can score earlier.")
    lines.append("- Smaller windows also expose more short-term noise, which can make residual features less stable.")
    lines.append("- Larger windows smooth noise and can improve stability, but they delay the first chance to detect an attack.")
    lines.append("- The 5-minute setting is the canonical baseline and is useful for stability, but this study checks whether that stability costs too much response time compared with 5s, 10s, and 60s.")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_final_figures(
    figures_root: Path,
    comparison_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    best_prediction_frames: dict[str, pd.DataFrame],
    best_curve_payloads: dict[str, dict[str, list[float]]],
    best_per_scenario_frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Build final figures for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    manifest: list[dict[str, Any]] = []
    if not comparison_df.empty:
        performance_path = figures_root / "performance_vs_window_size.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(comparison_df["window_label"], comparison_df["precision"], marker="o", label="precision")
        ax.plot(comparison_df["window_label"], comparison_df["recall"], marker="o", label="recall")
        ax.plot(comparison_df["window_label"], comparison_df["f1"], marker="o", label="f1")
        ax.set_title("Best-Model Performance vs Window Size")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(performance_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(performance_path), "description": "Best-model precision, recall, and F1 versus window size."})

        latency_path = figures_root / "latency_vs_window_size.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(comparison_df["window_label"], comparison_df["mean_detection_latency_seconds"], marker="o")
        ax.set_title("Best-Model Detection Latency vs Window Size")
        ax.set_ylabel("Mean detection latency (seconds)")
        fig.tight_layout()
        fig.savefig(latency_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(latency_path), "description": "Best-model detection latency versus window size."})

        training_path = figures_root / "training_time_vs_window_size.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(comparison_df["window_label"], comparison_df["training_time_seconds"])
        ax.set_title("Best-Model Training Time vs Window Size")
        ax.set_ylabel("Seconds")
        fig.tight_layout()
        fig.savefig(training_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(training_path), "description": "Best-model training time versus window size."})

        inference_path = figures_root / "inference_time_vs_window_size.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(comparison_df["window_label"], comparison_df["inference_time_seconds"])
        ax.set_title("Best-Model Inference Time vs Window Size")
        ax.set_ylabel("Seconds")
        fig.tight_layout()
        fig.savefig(inference_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(inference_path), "description": "Best-model inference time versus window size."})

        roc_path = figures_root / "best_model_roc_by_window_size.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        for tag, curves in best_curve_payloads.items():
            if curves.get("roc_fpr") and curves.get("roc_tpr"):
                ax.plot(curves["roc_fpr"], curves["roc_tpr"], label=tag)
        ax.set_title("Best-Model ROC Curves by Window Size")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(roc_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(roc_path), "description": "Best-model ROC curves by window size."})

        pr_path = figures_root / "best_model_pr_by_window_size.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        for tag, curves in best_curve_payloads.items():
            if curves.get("pr_recall") and curves.get("pr_precision"):
                ax.plot(curves["pr_recall"], curves["pr_precision"], label=tag)
        ax.set_title("Best-Model PR Curves by Window Size")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(pr_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(pr_path), "description": "Best-model precision-recall curves by window size."})

    attacked_summary = dataset_summary_df[dataset_summary_df["dataset_kind"] == "attacked"].copy()
    if not attacked_summary.empty:
        balance_path = figures_root / "class_balance_attack_coverage_by_window_size.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(attacked_summary))
        ax.bar(x - 0.2, attacked_summary["benign_window_count"], width=0.4, label="benign")
        ax.bar(x + 0.2, attacked_summary["attack_window_count"], width=0.4, label="attack")
        ax.set_xticks(x)
        ax.set_xticklabels(attacked_summary["window_label"])
        ax.set_title("Class Balance and Attack Coverage by Window Size")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(balance_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(balance_path), "description": "Benign and attack window counts by attacked-window size."})

    if best_per_scenario_frames:
        combined = []
        for tag, frame in best_per_scenario_frames.items():
            tagged = frame.copy()
            tagged["window_label"] = tag
            combined.append(tagged)
        combined_df = pd.concat(combined, ignore_index=True)
        pivot = combined_df.pivot_table(index="scenario_id", columns="window_label", values="f1", aggfunc="mean").fillna(0.0)
        scenario_path = figures_root / "per_scenario_performance_by_window_size.png"
        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.4)))
        heatmap = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_title("Best-Model Per-Scenario F1 by Window Size")
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(scenario_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(scenario_path), "description": "Per-scenario best-model F1 by window size."})

    return manifest


def package_best_model(final_best_df: pd.DataFrame) -> dict[str, Any]:
    """Handle package best model within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if final_best_df.empty:
        return {}
    best_row = final_best_df.sort_values(["f1", "recall", "precision", "average_precision"], ascending=False).iloc[0].to_dict()
    window_label = best_row["window_label"]
    model_name = best_row["model_name"]
    model_dir = ROOT / "outputs" / "window_size_study" / window_label / "models" / model_name
    ready_dir = ROOT / "outputs" / "window_size_study" / window_label / "ready_packages" / model_name
    report_dir = ROOT / "outputs" / "window_size_study" / window_label / "reports"
    package_root = OUTPUT_ROOT / "best_model_package"
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        shutil.copytree(model_dir, package_root / "model_artifacts")
    if ready_dir.exists():
        shutil.copytree(ready_dir, package_root / "ready_package")
    if report_dir.exists() and (report_dir / "model_summary.csv").exists():
        shutil.copy2(report_dir / "model_summary.csv", package_root / "window_model_summary.csv")
    metadata = {
        "window_label": window_label,
        "window_seconds": int(best_row["window_seconds"]),
        "step_seconds": int(best_row["step_seconds"]),
        "model_name": model_name,
        "precision": best_row.get("precision"),
        "recall": best_row.get("recall"),
        "f1": best_row.get("f1"),
        "average_precision": best_row.get("average_precision"),
        "roc_auc": best_row.get("roc_auc"),
        "training_time_seconds": best_row.get("training_time_seconds"),
        "inference_time_seconds": best_row.get("inference_time_seconds"),
        "criterion": "Highest test F1, then recall, precision, average_precision.",
    }
    write_json(metadata, package_root / "best_model_summary.json")
    readme_lines = [
        "# Best Model Package",
        "",
        f"- what won: `{model_name}` at window size `{window_label}`",
        f"- why it won: it achieved the highest test F1 in the window-size study, with tie-breaks on recall, precision, and average precision",
        "- how to rerun it: `python scripts/run_window_size_study.py --window-sizes "
        + str(int(best_row["window_seconds"]))
        + "`",
        "",
        "## Saved Contents",
        "",
        "- `model_artifacts/`: checkpoint/weights, model config, metrics, and predictions",
        "- `ready_package/`: preprocessing, thresholds, calibration, manifest, and packaged predictions",
        "- `best_model_summary.json`: exact winning setup metadata",
        "",
    ]
    (package_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return metadata


def build_phase2_bundle_outputs(best_package_info: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str, list[dict[str, Any]]]:
    """Build phase2 bundle outputs for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    inventory_rows: list[dict[str, Any]] = []
    manifests = sorted(ROOT.rglob("scenario_manifest.json"))
    bundle_manifests = [path for path in manifests if "outputs\\attacked" in str(path) or "outputs/attacked" in str(path)]
    if not bundle_manifests:
        inventory_df = pd.DataFrame(columns=["bundle_id", "manifest_path", "scenario_count", "dataset_id", "source_status"])
        comparison_df = pd.DataFrame(columns=["bundle_id", "window_label", "model_name", "precision", "recall", "f1", "average_precision", "roc_auc", "mean_latency_seconds", "note"])
        md = "# Phase 2 Bundle Comparison\n\nNo distinct attacked bundle manifests were found.\n"
        return inventory_df, comparison_df, md, []

    comparison_rows: list[dict[str, Any]] = []
    for path in bundle_manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bundle_id = str(payload.get("scenario_id") or path.parent.name)
        applied = payload.get("applied_scenarios", [])
        inventory_rows.append(
            {
                "bundle_id": bundle_id,
                "manifest_path": str(path),
                "scenario_count": int(len(applied)),
                "dataset_id": payload.get("scenario_id"),
                "source_status": "canonical_attacked_bundle",
            }
        )
        if best_package_info:
            window_label = str(best_package_info["window_label"])
            model_name = str(best_package_info["model_name"])
            predictions_path = ROOT / "outputs" / "window_size_study" / window_label / "models" / model_name / "predictions.parquet"
            labels_path = ROOT / "outputs" / "attacked" / "attack_labels.parquet"
            if predictions_path.exists() and labels_path.exists():
                predictions = pd.read_parquet(predictions_path)
                labels = pd.read_parquet(labels_path)
                if not labels.empty:
                    labels["start_time_utc"] = pd.to_datetime(labels["start_time_utc"], utc=True)
                    labels["end_time_utc"] = pd.to_datetime(labels["end_time_utc"], utc=True)
                metrics = compute_binary_metrics(
                    predictions["attack_present"].to_numpy(),
                    predictions["score"].to_numpy(),
                    float(pd.to_numeric(predictions["score"], errors="coerce").median()) if "predicted" not in predictions.columns else float(predictions.loc[predictions["predicted"] == 1, "score"].min()) if (predictions["predicted"] == 1).any() else 0.5,
                )
                if "predicted" in predictions.columns:
                    y_true = predictions["attack_present"].astype(int).to_numpy()
                    y_pred = predictions["predicted"].astype(int).to_numpy()
                    scores = predictions["score"].astype(float).to_numpy()
                    threshold = float(scores[y_pred == 1].min()) if (y_pred == 1).any() else float(np.quantile(scores, 0.95))
                    metrics = compute_binary_metrics(y_true, scores, threshold)
                latency = detection_latency_table(predictions, labels, model_name)
                comparison_rows.append(
                    {
                        "bundle_id": bundle_id,
                        "window_label": window_label,
                        "model_name": model_name,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "average_precision": metrics.get("average_precision"),
                        "roc_auc": metrics.get("roc_auc"),
                        "mean_latency_seconds": safe_mean(latency["latency_seconds"]) if not latency.empty else math.nan,
                        "note": "Only one distinct attacked bundle exists in the cleaned canonical workspace.",
                    }
                )

    inventory_df = pd.DataFrame(inventory_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    md_lines = ["# Phase 2 Bundle Comparison", ""]
    if len(inventory_df) == 1:
        md_lines.append("- Only one distinct attacked bundle exists in the current cleaned canonical workspace.")
    else:
        md_lines.append(f"- Distinct attacked bundles found: `{len(inventory_df)}`")
    md_lines.append("")
    md_lines.append("## Bundle Inventory")
    md_lines.append(to_markdown(inventory_df))
    md_lines.append("")
    md_lines.append("## Bundle Metrics")
    md_lines.append(to_markdown(comparison_df))
    md_lines.append("")

    figure_manifest = build_phase2_bundle_figures(inventory_df, comparison_df)
    return inventory_df, comparison_df, "\n".join(md_lines) + "\n", figure_manifest


def build_phase2_bundle_figures(inventory_df: pd.DataFrame, comparison_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build phase2 bundle figures for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    figure_root = OUTPUT_ROOT / "phase2_bundle_figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    if not comparison_df.empty:
        performance_path = figure_root / "performance_by_bundle.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(comparison_df["bundle_id"], comparison_df["f1"])
        ax.set_title("Best-Model F1 by Bundle")
        ax.set_ylim(0.0, 1.05)
        fig.tight_layout()
        fig.savefig(performance_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(performance_path), "description": "Best-model F1 by attacked bundle."})

        latency_path = figure_root / "latency_by_bundle.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(comparison_df["bundle_id"], comparison_df["mean_latency_seconds"])
        ax.set_title("Best-Model Mean Detection Latency by Bundle")
        ax.set_ylabel("Seconds")
        fig.tight_layout()
        fig.savefig(latency_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(latency_path), "description": "Best-model mean detection latency by attacked bundle."})

    labels_path = ROOT / "outputs" / "attacked" / "attack_labels.parquet"
    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        family_counts = labels.groupby("attack_family", observed=False).size().reset_index(name="scenario_count").sort_values("scenario_count", ascending=False)
        family_path = figure_root / "scenario_family_breakdown_by_bundle.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(family_counts["attack_family"], family_counts["scenario_count"])
        ax.set_title("Scenario-Family Breakdown for Canonical Bundle")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(family_path, dpi=180)
        plt.close(fig)
        manifest.append({"path": str(family_path), "description": "Scenario-family breakdown for the attacked bundle inventory."})
    return manifest


def write_top_level_deliverables(
    final_best_df: pd.DataFrame,
    all_model_rows: pd.DataFrame,
    phase2_comparison_df: pd.DataFrame,
    phase2_comparison_md: str,
    best_package_info: dict[str, Any],
    dataset_summary_df: pd.DataFrame,
) -> None:
    """Write top level deliverables for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    window_size_report = build_window_size_study_report(final_best_df, all_model_rows, dataset_summary_df, phase2_comparison_df, best_package_info)
    (ROOT / "WINDOW_SIZE_STUDY_REPORT.md").write_text(window_size_report, encoding="utf-8")

    model_table = all_model_rows.copy()
    model_table.to_csv(ROOT / "WINDOW_SIZE_MODEL_TABLE.csv", index=False)

    latency_table = all_model_rows[
        [
            "window_label",
            "window_seconds",
            "step_seconds",
            "model_name",
            "status",
            "mean_detection_latency_seconds",
            "median_detection_latency_seconds",
            "inference_time_seconds",
            "inference_time_ms_per_prediction",
        ]
    ].copy()
    latency_table.to_csv(ROOT / "WINDOW_SIZE_LATENCY_TABLE.csv", index=False)

    training_cost = all_model_rows[
        [
            "window_label",
            "window_seconds",
            "step_seconds",
            "model_name",
            "status",
            "training_time_seconds",
            "parameter_count",
            "feature_count",
            "seq_len",
        ]
    ].copy()
    training_cost.to_csv(ROOT / "WINDOW_SIZE_TRAINING_COST_TABLE.csv", index=False)

    best_lines = ["# Best Model Decision", ""]
    if best_package_info:
        best_lines.append(f"- winner: `{best_package_info['model_name']}` at `{best_package_info['window_label']}`")
        best_lines.append(f"- criterion: `{best_package_info['criterion']}`")
        best_lines.append(f"- precision: `{best_package_info['precision']}`")
        best_lines.append(f"- recall: `{best_package_info['recall']}`")
        best_lines.append(f"- f1: `{best_package_info['f1']}`")
    else:
        best_lines.append("- No successful best-model package was produced.")
    (ROOT / "BEST_MODEL_DECISION.md").write_text("\n".join(best_lines) + "\n", encoding="utf-8")

    (ROOT / "PHASE2_BUNDLE_COMPARISON_REPORT.md").write_text(phase2_comparison_md, encoding="utf-8")

    notes_lines = [
        "# Final Experiment Notes",
        "",
        "- Canonical source data were treated as read-only.",
        "- All rebuilt windows, residual datasets, model runs, figures, and comparison files were written under `outputs/window_size_study/`.",
        "- The stock Phase 1 threshold baseline stores a placeholder inference-time value; this study recomputed and corrected that value after each threshold run.",
        "- Cross-window best-model selection used highest test F1, with recall, precision, and average precision as tie-breaks.",
        "- If any model failed, that failure is recorded directly in the per-window `model_summary.csv` and the stack trace is saved in the corresponding window report folder.",
    ]
    (ROOT / "FINAL_EXPERIMENT_NOTES.md").write_text("\n".join(notes_lines) + "\n", encoding="utf-8")


def build_window_size_study_report(
    final_best_df: pd.DataFrame,
    all_model_rows: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    phase2_comparison_df: pd.DataFrame,
    best_package_info: dict[str, Any],
) -> str:
    """Build window size study report for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    lines = ["# Window Size Study Report", ""]
    lines.append("## Scope")
    lines.append("- Canonical inputs were rebuilt into four window sizes: 5s, 10s, 60s, and 300s.")
    lines.append("- Canonical model suite: threshold baseline, isolation forest, autoencoder, GRU, LSTM, transformer, and tokenized LLM-style baseline.")
    lines.append("- All experimental outputs were saved under `outputs/window_size_study/`.")
    lines.append("")
    lines.append("## Window Dataset Summary")
    lines.append(to_markdown(dataset_summary_df))
    lines.append("")
    lines.append("## Best Model Per Window")
    lines.append(to_markdown(final_best_df))
    lines.append("")
    lines.append("## All Model Runs")
    lines.append(to_markdown(all_model_rows))
    lines.append("")
    lines.append("## Best Model Package")
    if best_package_info:
        lines.append(f"- Winner: `{best_package_info['model_name']}` at `{best_package_info['window_label']}`")
        lines.append(f"- F1: `{best_package_info['f1']}`")
        lines.append(f"- Recall: `{best_package_info['recall']}`")
        lines.append(f"- Precision: `{best_package_info['precision']}`")
    else:
        lines.append("- No winning package was produced because no successful runs completed.")
    lines.append("")
    lines.append("## Phase 2 Bundle Comparison")
    lines.append(to_markdown(phase2_comparison_df))
    lines.append("")
    lines.append("## Why There Are So Many Variables")
    lines.append("- The project monitors a full feeder, not a single device, so it tracks voltages, angles, power flows, DER states, controls, environment, and cyber events across many assets.")
    lines.append("- Full-feeder monitoring creates hundreds of raw channels because each bus, line, phase, DER asset, and control element contributes separate measurements.")
    lines.append("- Windowing multiplies those channels into multiple features because each signal is turned into statistics such as mean, standard deviation, min, max, and last value.")
    lines.append("- Smaller windows can improve response latency because the detector can score more often, but they also expose more noise and short-lived fluctuations.")
    lines.append("- Five-minute windows can be more stable because they smooth transient noise, but they can be slower for operational response.")
    lines.append("")
    return "\n".join(lines) + "\n"


def to_markdown(df: pd.DataFrame) -> str:
    """Handle to markdown within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if df.empty:
        return "| empty |\n|---|\n"
    header = "| " + " | ".join(str(column) for column in df.columns) + " |"
    divider = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = [header, divider]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
