"""Repository orchestration script for DERGuardian.

This script runs or rebuilds build phase1 lineage audit artifacts for audits, figures,
reports, or reproducibility checks. It is release-support code and must preserve
the separation between canonical benchmark, replay, heldout synthetic, and
extension experiment contexts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_LINEAGE_PATH = ROOT / "PHASE1_DATA_LINEAGE_AUDIT.md"
TRAINING_INPUT_INVENTORY_PATH = ROOT / "phase1_training_input_inventory.csv"
CANONICAL_INPUT_SUMMARY_PATH = ROOT / "phase1_canonical_input_summary.md"


def rel(path: Path) -> str:
    """Handle rel within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    """Load json for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return json.loads(path.read_text(encoding="utf-8"))


def classify_column(column: str) -> tuple[str, str, bool]:
    """Handle classify column within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if column in {
        "window_start_utc",
        "window_end_utc",
        "window_seconds",
        "scenario_id",
        "run_id",
        "split_id",
        "scenario_window_id",
        "split_name",
    }:
        return "metadata", "metadata", False
    if column in {"attack_present", "attack_family", "attack_severity", "attack_affected_assets"}:
        return "label", "metadata", False
    if not column.startswith("delta__"):
        return "other", "metadata", False

    base = column[len("delta__") :]
    prefix = base.split("__", 1)[0]
    if prefix.startswith("env_"):
        return "environment", "feature", True
    if prefix.startswith("cyber_"):
        return "cyber", "feature", True
    if prefix.startswith("derived_"):
        return "derived", "feature", True
    if prefix.startswith("feeder_"):
        return "feeder", "feature", True
    if prefix.startswith("bus_"):
        return "bus", "feature", True
    if prefix.startswith("line_"):
        return "line", "feature", True
    if prefix.startswith("load_"):
        return "load", "feature", True
    if prefix.startswith("regulator_"):
        return "regulator", "feature", True
    if prefix.startswith("capacitor_"):
        return "capacitor", "feature", True
    if prefix.startswith("switch_"):
        return "switch", "feature", True
    if prefix.startswith("breaker_") or prefix.startswith("relay_"):
        return "protection", "feature", True
    if prefix.startswith("pv_"):
        return "pv", "feature", True
    if prefix.startswith("bess_"):
        return "bess", "feature", True
    if prefix in {"simulation_index", "sample_rate_seconds"}:
        return "timing", "feature", True
    return "other_feature", "feature", True


def source_artifact_for_column(column: str) -> str:
    """Handle source artifact for column within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if column in {"attack_present", "attack_family", "attack_severity", "attack_affected_assets", "scenario_window_id"}:
        return "outputs/attacked/attack_labels.parquet"
    if column.startswith("delta__cyber_"):
        return "outputs/clean/cyber_events.parquet + outputs/attacked/cyber_events.parquet"
    if column.startswith("delta__"):
        return "outputs/clean/measured_physical_timeseries.parquet + outputs/attacked/measured_physical_timeseries.parquet"
    return "outputs/window_size_study/60s/residual_windows.parquet"


def build_training_inventory() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build training inventory for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    residual_path = ROOT / "outputs" / "window_size_study" / "60s" / "residual_windows.parquet"
    residual_df = pd.read_parquet(residual_path)
    selected_features = json.loads(
        (ROOT / "outputs" / "window_size_study" / "60s" / "ready_packages" / "transformer" / "feature_columns.json").read_text(encoding="utf-8")
    )
    selected_set = set(selected_features)

    rows = []
    for column in residual_df.columns:
        category, metadata_or_feature, residualized = classify_column(column)
        rows.append(
            {
                "column_name": column,
                "category": category,
                "source_artifact": source_artifact_for_column(column),
                "used_in_training": bool(column in selected_set),
                "metadata_or_feature": metadata_or_feature,
                "residualized": bool(residualized),
            }
        )
    inventory = pd.DataFrame(rows).sort_values(["metadata_or_feature", "used_in_training", "category", "column_name"], ascending=[True, False, True, True]).reset_index(drop=True)
    inventory.to_csv(TRAINING_INPUT_INVENTORY_PATH, index=False)

    env_raw = [column for column in residual_df.columns if column.startswith("delta__env_")]
    env_selected = [column for column in selected_features if column.startswith("delta__env_")]
    summary = {
        "residual_path": rel(residual_path),
        "column_count": int(len(residual_df.columns)),
        "delta_feature_count": int(sum(column.startswith("delta__") for column in residual_df.columns)),
        "selected_transformer_feature_count": int(len(selected_features)),
        "env_residual_feature_count": int(len(env_raw)),
        "env_selected_feature_count": int(len(env_selected)),
    }
    return inventory, summary


def build_data_lineage() -> None:
    """Build data lineage for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    run_manifest = load_json(ROOT / "outputs" / "window_size_study" / "run_manifest.json")
    final_comparison = pd.read_csv(ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv")
    window_dataset_summary = pd.read_csv(ROOT / "outputs" / "window_size_study" / "window_dataset_summary.csv")
    canonical_full_summary = pd.read_csv(ROOT / "outputs" / "reports" / "model_full_run_artifacts" / "model_summary_table_full.csv")
    window_model_rows = []
    for window_label in ["5s", "10s", "60s", "300s"]:
        path = ROOT / "outputs" / "window_size_study" / window_label / "reports" / "model_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["window_label"] = window_label
            window_model_rows.append(frame)
    all_window_models = pd.concat(window_model_rows, ignore_index=True)
    inventory, training_summary = build_training_inventory()

    canonical_row = final_comparison[(final_comparison["model_name"] == "transformer") & (final_comparison["window_label"] == "60s")].iloc[0]
    benchmarked_models = sorted(all_window_models["model_name"].unique().tolist())
    raw_sources = run_manifest["canonical_sources"]
    selected_windows = run_manifest["selected_windows"]

    lines = [
        "# Phase 1 Data Lineage Audit",
        "",
        "This audit reconstructs the canonical Phase 1 path from raw clean/attacked data through windowing, residualization, split assignment, and benchmark model packaging.",
        "",
        "## Canonical raw sources",
        "",
        f"- Clean measured physical time series: `{raw_sources['clean_measured']}`",
        f"- Clean cyber event stream: `{raw_sources['clean_cyber']}`",
        f"- Attacked measured physical time series: `{raw_sources['attacked_measured']}`",
        f"- Attacked cyber event stream: `{raw_sources['attacked_cyber']}`",
        f"- Attack labels: `{raw_sources['attack_labels']}`",
        "",
        "The run manifest confirms these are pre-window time-series artifacts rather than merged-window artifacts.",
        "",
        "## Transformation chain",
        "",
        "1. Clean and attacked measured/cyber streams are loaded from `outputs/clean/` and `outputs/attacked/`.",
        "2. `phase1/build_windows.py` reconstructs `analysis_timestamp_utc` from `timestamp_utc + simulation_index * sample_rate_seconds` when available.",
        "3. `build_merged_windows(...)` aggregates physical signals into `mean/std/min/max/last` window statistics and appends cyber event-count features.",
        "4. Window labels are assigned by attack-label overlap using `min_attack_overlap_fraction = 0.2`.",
        "5. `build_aligned_residual_dataframe(...)` inner-joins attacked and clean windows on `window_start_utc` and computes `delta__* = attacked - clean`.",
        "6. `assign_attack_aware_split(...)` applies scenario-aware train/val/test splits with two benign buffer windows around attack windows.",
        "7. Per-model training selects a feature subset from the residual columns, calibrates scores on validation data, and writes a ready package.",
        "",
        "## Clean vs attacked separation",
        "",
        "- Clean windows are written under `outputs/window_size_study/<window>/data/merged_windows_clean.parquet`.",
        "- Attacked windows are written under `outputs/window_size_study/<window>/data/merged_windows_attacked.parquet`.",
        "- Residual windows are written under `outputs/window_size_study/<window>/residual_windows.parquet`.",
        "- The canonical cached full-run residual artifact is `outputs/reports/model_full_run_artifacts/residual_windows_full_run.parquet` for the older 300 s full-run pipeline.",
        "",
        "## Window-size contract",
        "",
    ]
    for spec in selected_windows:
        lines.append(
            f"- `{spec['tag']}`: window `{spec['window_seconds']}` s, step `{spec['step_seconds']}` s."
        )
    lines.extend(
        [
            "",
            f"- Step policy from `run_manifest.json`: {run_manifest['window_step_policy']}",
            "",
            "## Canonical benchmark winner",
            "",
            f"- Source-of-truth file: `outputs/window_size_study/final_window_comparison.csv`",
            f"- Winner: `{canonical_row['model_name']}` at `{canonical_row['window_label']}`",
            f"- Benchmark F1: `{canonical_row['f1']:.6f}`",
            f"- Ready package: `{rel(Path(canonical_row['ready_package_dir']))}`",
            "",
            "## Benchmarked model roster",
            "",
            f"- Implemented and benchmarked in the canonical window-size study: {', '.join(benchmarked_models)}.",
            "- `autoencoder` is an MLP autoencoder, not an LSTM autoencoder.",
            "- `llm_baseline` is a tokenized time-series baseline, not a generative LLM detector.",
            "- `arima` is not implemented in the current repo.",
            "- `ttm` is not part of the frozen canonical benchmark roster, but a separate extension benchmark now exists in `phase1_ttm_results.csv` and `phase1_ttm_eval_report.md`.",
            "",
            "## Metadata vs training inputs",
            "",
            f"- Residual 60 s dataset columns: `{training_summary['column_count']}`",
            f"- Residual `delta__*` feature columns: `{training_summary['delta_feature_count']}`",
            f"- Selected canonical transformer feature columns: `{training_summary['selected_transformer_feature_count']}`",
            f"- Canonical transformer input schema: `aligned_residual`, sequence length `12`, input dimension `64`.",
            "",
            "The column-level inventory is saved in `phase1_training_input_inventory.csv`.",
            "",
            "## Environment variables",
            "",
            f"- Raw window datasets include environment features (`env_*`) because `window_dataset_summary.csv` lists them in the merged-window feature columns.",
            f"- Residual datasets also contain `delta__env_*` columns (`{training_summary['env_residual_feature_count']}` total).",
            f"- Selected canonical transformer features use `{training_summary['env_selected_feature_count']}` environment residual columns.",
            "- So environment variables are simulated and present in the raw/model-ready window artifacts, but they are not part of the final selected 60 s transformer feature subset.",
            "",
            "## Label-generation consistency",
            "",
            "- Attack presence is assigned at the window level by overlap with `attack_labels.parquet`.",
            "- Residual windows keep `attack_present`, `attack_family`, `attack_severity`, and `attack_affected_assets` metadata from attacked windows.",
            "- Scenario-aware splitting uses `scenario_window_id` so attacked windows are split by scenario rather than by random row shuffling.",
            "",
            "## Diagram accuracy notes",
            "",
            "- `Baseline Models ML Models (LSTM, Autoencoder, Isolation Forest)` is incomplete because the benchmark suite also includes transformer, GRU, threshold baseline, and a tokenized LLM-style baseline.",
            "- The diagram label can overread `Autoencoder` as `LSTM AE`, but the actual repository autoencoder is MLP-based.",
            "- `Normal Data Generation OpenDSS Simulation (Voltage, Current, Power, SOC, Temp)` is directionally true, but the actual monitored channels also include feeder losses, control states, cyber counts, and derived residual features.",
            "- `LLM Analysis Context Learning of Normal Behavior` exists as context-summary and reasoning artifacts in the legacy full-run pipeline; it is not part of the frozen canonical benchmark winner selection.",
            "",
            "## Canonical vs legacy note",
            "",
            "- The older `outputs/reports/model_full_run_artifacts/` pipeline still exists and is useful for lineage/reference.",
            "- The frozen source-of-truth benchmark selection is the newer window-size study under `outputs/window_size_study/`.",
        ]
    )
    DATA_LINEAGE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    model_counts = all_window_models.groupby(["window_label", "model_name"]).size().reset_index(name="rows")
    canonical_input_lines = [
        "# Phase 1 Canonical Input Summary",
        "",
        "## Canonical winner path",
        "",
        f"- Winner: `{canonical_row['model_name']}` at `{canonical_row['window_label']}` from `outputs/window_size_study/final_window_comparison.csv`.",
        f"- Residual training input artifact: `{training_summary['residual_path']}`.",
        f"- Ready package: `{rel(Path(canonical_row['ready_package_dir']))}`.",
        f"- Model report directory: `{rel(Path(canonical_row['report_dir']))}`.",
        "",
        "## Input contract",
        "",
        f"- Residual windows rows: `{int(canonical_row['row_count'])}`",
        f"- Total residual columns: `{training_summary['column_count']}`",
        f"- Residual feature columns available before selection: `{training_summary['delta_feature_count']}`",
        f"- Selected feature columns used by the canonical transformer: `{training_summary['selected_transformer_feature_count']}`",
        f"- Sequence length: `12`",
        f"- Window / step: `60 s / 12 s`",
        "",
        "## Benchmarked model set in the canonical study",
        "",
        "```text",
        model_counts.sort_values(["window_label", "model_name"]).to_string(index=False),
        "```",
        "",
        "## Outside the canonical benchmark roster",
        "",
        "- `ARIMA`: no training script, result table, or ready package found.",
        "- `TTM`: implemented separately as an extension benchmark; it is not part of the frozen canonical window-size benchmark roster.",
    ]
    CANONICAL_INPUT_SUMMARY_PATH.write_text("\n".join(canonical_input_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_data_lineage()
