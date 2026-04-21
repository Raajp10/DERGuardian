"""Phase 3 evaluation and analysis support for DERGuardian.

This module implements run sequence model sweep logic for detector evaluation, ablations,
zero-day-like heldout synthetic analysis, latency sweeps, or final reporting.
It keeps benchmark, replay, heldout synthetic, and extension results separated.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from common.io_utils import ensure_dir, write_json
from phase1_models.run_full_evaluation import prepare_full_run_data
from phase3.experiment_utils import plot_metric_bars, run_model_suite, to_markdown, write_phase3_report


def main() -> None:
    """Run the command-line entrypoint for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Run Phase 3 sequence-model tuning sweeps.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--feature-counts", default="32,64,96,128")
    parser.add_argument("--seq-lens", default="4,8,12,16")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.project_root)
    artifact_root = ensure_dir(root / "outputs" / "reports" / "phase3_zero_day_artifacts")
    feature_counts = [int(item) for item in args.feature_counts.split(",") if item.strip()]
    seq_lens = [int(item) for item in args.seq_lens.split(",") if item.strip()]

    full_data = prepare_full_run_data(root, buffer_windows=2)
    result = run_model_suite(
        root=root,
        full_data=full_data,
        report_root_name="phase3_zero_day_artifacts/sequence_model_sweep",
        model_root_name="phase3_zero_day/sequence_model_sweep",
        include_models=["transformer", "gru", "lstm"],
        model_settings={
            "transformer": {"feature_counts": feature_counts, "seq_lens": seq_lens, "epochs": args.epochs, "patience": args.patience},
            "gru": {"feature_counts": feature_counts, "seq_lens": seq_lens, "epochs": args.epochs, "patience": args.patience},
            "lstm": {"feature_counts": feature_counts, "seq_lens": seq_lens, "epochs": args.epochs, "patience": args.patience},
        },
    )

    sweep_df = result["sweep_df"].copy()
    summary_df = result["summary_df"].copy()
    threshold_baseline_path = root / "outputs" / "reports" / "model_full_run_artifacts" / "model_summary_table_full.csv"
    baseline_df = pd.read_csv(threshold_baseline_path) if threshold_baseline_path.exists() else pd.DataFrame()
    threshold_row = baseline_df[baseline_df["model_name"] == "threshold_baseline"].copy()

    sweep_csv = artifact_root / "sequence_model_sweep_table.csv"
    best_json = artifact_root / "best_sequence_configs.json"
    plot_path = artifact_root / "sequence_model_comparison.png"
    sweep_df.to_csv(sweep_csv, index=False)
    write_json(summary_df.to_dict(orient="records"), best_json)
    plot_metric_bars(summary_df, "model_name", plot_path, ["precision", "recall", "f1", "average_precision"], "Tuned Sequence Model Comparison")

    sections = [
        (
            "Sweep Setup",
            "\n".join(
                [
                    f"- feature_counts: `{feature_counts}`",
                    f"- seq_lens: `{seq_lens}`",
                    f"- epochs: `{args.epochs}`",
                    f"- patience: `{args.patience}`",
                ]
            ),
        ),
        ("Best Sequence Configurations", to_markdown(summary_df)),
        ("Threshold Baseline Reference", to_markdown(threshold_row)),
        ("Full Sequence Sweep Table", to_markdown(sweep_df)),
    ]
    write_phase3_report(artifact_root / "sequence_model_report.md", "# Phase 3 Sequence Model Sweep Report", sections)
    print(f"Sequence-model sweep artifacts written to {artifact_root}")


if __name__ == "__main__":
    main()

