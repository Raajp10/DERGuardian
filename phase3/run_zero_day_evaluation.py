"""Phase 3 evaluation and analysis support for DERGuardian.

This module implements run zero day evaluation logic for detector evaluation, ablations,
zero-day-like heldout synthetic analysis, latency sweeps, or final reporting.
It keeps benchmark, replay, heldout synthetic, and extension results separated.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime, timezone
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from common.config import PipelineConfig
from common.io_utils import ensure_dir, write_json
from phase1_models.model_utils import READY_MODEL_ROOT
from phase3.experiment_utils import (
    mean_metric_summary,
    plot_metric_bars,
    prepare_window_bundle,
    run_model_suite,
    run_saved_phase1_package,
    summarize_latency,
    to_markdown,
)
from phase3.zero_day_report import build_zero_day_report
from phase3.zero_day_splitter import available_scenarios, build_zero_day_full_run_data


DEFAULT_MODELS = [
    "threshold_baseline",
    "autoencoder",
    "isolation_forest",
    "gru",
    "transformer",
    "lstm",
    "llm_baseline",
]

DEFAULT_RANDOM_SEED = int(PipelineConfig().random_seed)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional for some report-only paths.
        pass


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _artifact_path_payload(paths: dict[str, Path | str | list[str]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in paths.items():
        if isinstance(value, Path):
            payload[key] = str(value.resolve())
        elif isinstance(value, list):
            payload[key] = [str(Path(item).resolve()) for item in value]
        else:
            payload[key] = str(value)
    return payload


def main() -> None:
    """Run the command-line entrypoint for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Run leave-one-scenario-out zero-day evaluation for the DER anomaly benchmark.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--holdout-scenarios", default="")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--execution-mode", default="reduced_validation", choices=["reduced_validation", "full"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--buffer-windows", type=int, default=2)
    parser.add_argument("--reuse-ready-model", default="")
    parser.add_argument("--reuse-package-dir", default="")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    root = Path(args.project_root)
    _set_global_seed(int(args.seed))
    output_root = ensure_dir(root / "outputs" / "phase3_zero_day")
    artifact_root = ensure_dir(root / "outputs" / "reports" / "phase3_zero_day_artifacts")
    bundle = prepare_window_bundle(root)
    scenarios = available_scenarios(bundle.labels_df)
    holdout_scenarios = [item for item in args.holdout_scenarios.split(",") if item.strip()] if args.holdout_scenarios else scenarios
    model_names = [item for item in args.models.split(",") if item.strip()]
    if args.reuse_ready_model or args.reuse_package_dir:
        package_dir = (
            Path(args.reuse_package_dir)
            if args.reuse_package_dir
            else root / "outputs" / READY_MODEL_ROOT / str(args.reuse_ready_model)
        )
        run_saved_package_evaluation(
            root=root,
            bundle=bundle,
            holdout_scenarios=holdout_scenarios,
            package_dir=package_dir,
            buffer_windows=args.buffer_windows,
            seed=int(args.seed),
            command="python " + " ".join(sys.argv),
        )
        return

    model_settings = {
        "threshold_baseline": {"feature_counts": [32, 64, 96]},
        "autoencoder": {"feature_counts": [32, 64, 96], "epochs": args.epochs, "patience": args.patience},
        "isolation_forest": {"feature_counts": [32, 64]},
        "gru": {"feature_counts": [32], "seq_lens": [4], "epochs": args.epochs, "patience": args.patience},
        "transformer": {"feature_counts": [32], "seq_lens": [4], "epochs": args.epochs, "patience": args.patience},
        "lstm": {"feature_counts": [32], "seq_lens": [4], "epochs": args.epochs, "patience": args.patience},
        "llm_baseline": {"feature_counts": [96], "seq_lens": [4], "epochs": args.epochs, "patience": args.patience},
    }
    if args.execution_mode == "full":
        model_settings["gru"]["feature_counts"] = [32, 64]
        model_settings["transformer"]["feature_counts"] = [32, 64]
        model_settings["lstm"]["feature_counts"] = [32, 64]
        model_settings["gru"]["seq_lens"] = [4, 8]
        model_settings["transformer"]["seq_lens"] = [4, 8]
        model_settings["lstm"]["seq_lens"] = [4, 8]
        model_settings["llm_baseline"]["feature_counts"] = [64, 96]
        model_settings["llm_baseline"]["seq_lens"] = [4, 8]

    summary_long_rows: list[dict[str, object]] = []
    per_scenario_rows: list[pd.DataFrame] = []
    latency_rows: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []

    for holdout_scenario in holdout_scenarios:
        holdout_data = build_zero_day_full_run_data(
            feature_df=bundle.residual_df,
            labels_df=bundle.labels_df,
            holdout_scenario=holdout_scenario,
            buffer_windows=args.buffer_windows,
        )
        split_summary = (
            holdout_data.split_df.groupby(["split_name", "attack_present"], observed=False)
            .size()
            .reset_index(name="window_count")
        )
        split_summary["holdout_scenario"] = holdout_scenario
        split_rows.extend(split_summary.to_dict(orient="records"))

        result = run_model_suite(
            root=root,
            full_data=holdout_data,
            report_root_name=f"phase3_zero_day_artifacts/{holdout_scenario}",
            model_root_name=f"phase3_zero_day/zero_day_models/{holdout_scenario}",
            include_models=model_names,
            model_settings=model_settings,
        )
        summary_df = result["summary_df"].copy()
        if summary_df.empty:
            continue
        summary_df["holdout_scenario"] = holdout_scenario
        summary_long_rows.extend(summary_df.to_dict(orient="records"))

        per_scenario_df = result["per_scenario_df"].copy()
        per_scenario_df = per_scenario_df[per_scenario_df["scenario_id"] == holdout_scenario].copy()
        per_scenario_df["holdout_scenario"] = holdout_scenario
        per_scenario_rows.append(per_scenario_df)

        latency_df = result["latency_df"].copy()
        latency_df = latency_df[latency_df["scenario_id"] == holdout_scenario].copy()
        latency_df["holdout_scenario"] = holdout_scenario
        latency_rows.append(latency_df)

    summary_long_df = pd.DataFrame(summary_long_rows)
    zero_day_summary = mean_metric_summary(summary_long_df, ["model_name"]).sort_values(["recall", "precision", "f1"], ascending=[False, False, False]).reset_index(drop=True)
    per_scenario_df = pd.concat(per_scenario_rows, ignore_index=True) if per_scenario_rows else pd.DataFrame()
    latency_df = pd.concat(latency_rows, ignore_index=True) if latency_rows else pd.DataFrame()
    latency_summary = summarize_latency(latency_df, ["model_name"])
    if not zero_day_summary.empty and not latency_summary.empty:
        zero_day_summary = zero_day_summary.merge(latency_summary, on="model_name", how="left")

    summary_csv = artifact_root / "zero_day_model_summary.csv"
    summary_md = artifact_root / "zero_day_model_summary.md"
    per_scenario_csv = artifact_root / "zero_day_per_scenario_metrics.csv"
    latency_csv = artifact_root / "zero_day_latency_analysis.csv"
    long_csv = artifact_root / "zero_day_model_summary_long.csv"
    split_csv = artifact_root / "zero_day_split_summary.csv"

    zero_day_summary.to_csv(summary_csv, index=False)
    summary_md.write_text(to_markdown(zero_day_summary), encoding="utf-8")
    summary_long_df.to_csv(long_csv, index=False)
    per_scenario_df.to_csv(per_scenario_csv, index=False)
    latency_df.to_csv(latency_csv, index=False)
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)

    comparison_plot = artifact_root / "zero_day_comparison_plot.png"
    plot_metric_bars(zero_day_summary, "model_name", comparison_plot, ["precision", "recall", "f1", "average_precision"], "Zero-Day Leave-One-Scenario-Out Comparison")

    artifact_paths = {
        "summary_csv": summary_csv,
        "summary_markdown": summary_md,
        "summary_long_csv": long_csv,
        "per_scenario_metrics_csv": per_scenario_csv,
        "latency_analysis_csv": latency_csv,
        "split_summary_csv": split_csv,
        "comparison_plot_png": comparison_plot,
        "report_markdown": artifact_root / "zero_day_report.md",
        "phase3_model_output_root": output_root / "zero_day_models",
    }

    write_json(
        {
            "evaluation_timestamp_utc": _utc_now_z(),
            "random_seed": int(args.seed),
            "execution_mode": args.execution_mode,
            "command": "python " + " ".join(sys.argv),
            "project_root": str(root.resolve()),
            "evaluated_model_names": model_names,
            "evaluated_package_paths": [],
            "attacked_windows_path": str(bundle.attacked_windows_path.resolve()),
            "attack_labels_path": str(bundle.labels_path.resolve()),
            "clean_windows_path": str(bundle.clean_windows_path.resolve()),
            "residual_artifact_path": str(bundle.residual_artifact_path.resolve()),
            "holdout_scenarios": holdout_scenarios,
            "output_root": str(artifact_root.resolve()),
            "artifact_paths": _artifact_path_payload(artifact_paths),
            "metrics_summary_path": str(summary_csv.resolve()),
            "per_scenario_metrics_path": str(per_scenario_csv.resolve()),
            "latency_analysis_path": str(latency_csv.resolve()),
            "explanation_artifact_paths": [],
            "package_reuse_mode": False,
            "models": model_names,
            "epochs": args.epochs,
            "patience": args.patience,
            "buffer_windows": args.buffer_windows,
            "residual_artifact_used": bool(bundle.residual_artifact_used),
            "notes": "Zero-day evaluation trains on benign windows plus all non-held-out attack scenarios, then tests on the unseen hold-out scenario with local benign context.",
        },
        artifact_root / "zero_day_run_manifest.json",
    )
    build_zero_day_report(artifact_root)
    print(f"Zero-day artifacts written to {artifact_root}")


def run_saved_package_evaluation(
    *,
    root: Path,
    bundle,
    holdout_scenarios: list[str],
    package_dir: Path,
    buffer_windows: int,
    seed: int,
    command: str,
) -> None:
    """Run saved package evaluation for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    package_name = package_dir.name
    artifact_root = ensure_dir(root / "outputs" / "reports" / "phase3_package_reuse_artifacts" / package_name)
    summary_rows: list[dict[str, object]] = []
    per_scenario_frames: list[pd.DataFrame] = []
    latency_frames: list[pd.DataFrame] = []
    for holdout_scenario in holdout_scenarios:
        holdout_data = build_zero_day_full_run_data(
            feature_df=bundle.residual_df,
            labels_df=bundle.labels_df,
            holdout_scenario=holdout_scenario,
            buffer_windows=buffer_windows,
        )
        result = run_saved_phase1_package(
            root=root,
            full_data=holdout_data,
            package_dir=package_dir,
            holdout_scenario=holdout_scenario,
        )
        summary_rows.append(result["summary"])
        per_scenario_frame = result["per_scenario"].copy()
        per_scenario_frame["holdout_scenario"] = holdout_scenario
        per_scenario_frames.append(per_scenario_frame)
        latency_frame = result["latency"].copy()
        latency_frame["holdout_scenario"] = holdout_scenario
        latency_frames.append(latency_frame)

    summary_long_df = pd.DataFrame(summary_rows)
    summary_df = mean_metric_summary(summary_long_df, ["model_name"]).reset_index(drop=True)
    per_scenario_df = pd.concat(per_scenario_frames, ignore_index=True) if per_scenario_frames else pd.DataFrame()
    latency_df = pd.concat(latency_frames, ignore_index=True) if latency_frames else pd.DataFrame()
    summary_df.to_csv(artifact_root / "saved_package_summary.csv", index=False)
    summary_long_df.to_csv(artifact_root / "saved_package_summary_long.csv", index=False)
    per_scenario_df.to_csv(artifact_root / "saved_package_per_scenario_metrics.csv", index=False)
    latency_df.to_csv(artifact_root / "saved_package_latency_analysis.csv", index=False)
    (artifact_root / "saved_package_summary.md").write_text(to_markdown(summary_df), encoding="utf-8")
    plot_metric_bars(
        summary_df,
        "model_name",
        artifact_root / "saved_package_comparison_plot.png",
        ["precision", "recall", "f1", "average_precision"],
        f"Saved Phase 1 Package Reuse: {package_name}",
    )
    summary_csv = artifact_root / "saved_package_summary.csv"
    per_scenario_csv = artifact_root / "saved_package_per_scenario_metrics.csv"
    latency_csv = artifact_root / "saved_package_latency_analysis.csv"
    summary_md = artifact_root / "saved_package_summary.md"
    comparison_plot = artifact_root / "saved_package_comparison_plot.png"
    artifact_paths = {
        "summary_csv": summary_csv,
        "summary_markdown": summary_md,
        "summary_long_csv": artifact_root / "saved_package_summary_long.csv",
        "per_scenario_metrics_csv": per_scenario_csv,
        "latency_analysis_csv": latency_csv,
        "comparison_plot_png": comparison_plot,
    }
    write_json(
        {
            "evaluation_timestamp_utc": _utc_now_z(),
            "random_seed": int(seed),
            "execution_mode": "saved_package_reuse",
            "command": command,
            "project_root": str(root.resolve()),
            "evaluated_model_names": summary_df["model_name"].astype(str).tolist() if not summary_df.empty else [package_name],
            "evaluated_package_paths": [str(package_dir.resolve())],
            "mode": "saved_phase1_package_reuse",
            "package_dir": str(package_dir.resolve()),
            "attacked_windows_path": str(bundle.attacked_windows_path.resolve()),
            "attack_labels_path": str(bundle.labels_path.resolve()),
            "clean_windows_path": str(bundle.clean_windows_path.resolve()),
            "residual_artifact_path": str(bundle.residual_artifact_path.resolve()),
            "holdout_scenarios": holdout_scenarios,
            "output_root": str(artifact_root.resolve()),
            "artifact_paths": _artifact_path_payload(artifact_paths),
            "metrics_summary_path": str(summary_csv.resolve()),
            "per_scenario_metrics_path": str(per_scenario_csv.resolve()),
            "latency_analysis_path": str(latency_csv.resolve()),
            "explanation_artifact_paths": [],
            "package_reuse_mode": True,
            "buffer_windows": buffer_windows,
            "residual_artifact_used": bool(bundle.residual_artifact_used),
            "notes": "This Phase 3 path loads a saved Phase 1 package directly and runs inference on unseen holdout windows without retraining.",
        },
        artifact_root / "saved_package_run_manifest.json",
    )
    print(f"Saved-package Phase 3 artifacts written to {artifact_root}")


if __name__ == "__main__":
    main()
