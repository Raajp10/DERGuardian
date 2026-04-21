"""Phase 3 evaluation and analysis support for DERGuardian.

This module implements run fdi ablation logic for detector evaluation, ablations,
zero-day-like heldout synthetic analysis, latency sweeps, or final reporting.
It keeps benchmark, replay, heldout synthetic, and extension results separated.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from common.io_utils import ensure_dir
from phase3.experiment_utils import plot_metric_bars, prepare_window_bundle, run_model_suite, to_markdown, write_phase3_report
from phase3.fdi_feature_builder import augment_fdi_features, fdi_feature_candidates
from phase3.zero_day_splitter import build_zero_day_full_run_data


DEFAULT_HOLDOUT = "scn_false_data_injection_pv60_voltage"


def main() -> None:
    """Run the command-line entrypoint for the Phase 3 evaluation workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Run the Phase 3 stealth-FDI feature ablation study.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--holdout-scenario", default=DEFAULT_HOLDOUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.project_root)
    artifact_root = ensure_dir(root / "outputs" / "reports" / "phase3_zero_day_artifacts")
    bundle = prepare_window_bundle(root)
    augmented_df, fdi_columns = augment_fdi_features(bundle.residual_df, bundle.attacked_windows, bundle.clean_windows)

    variants = {
        "baseline_residual": bundle.residual_df,
        "fdi_augmented": augmented_df,
    }
    model_settings = {
        "threshold_baseline": {"feature_counts": [32, 64, 96]},
        "autoencoder": {"feature_counts": [32, 64, 96], "epochs": args.epochs, "patience": args.patience},
        "transformer": {"feature_counts": [32, 64], "seq_lens": [4, 8], "epochs": args.epochs, "patience": args.patience},
    }

    ablation_rows: list[dict[str, object]] = []
    importance_rows: list[pd.DataFrame] = []

    for variant_name, frame in variants.items():
        holdout_data = build_zero_day_full_run_data(
            feature_df=frame,
            labels_df=bundle.labels_df,
            holdout_scenario=args.holdout_scenario,
            buffer_windows=2,
            candidate_columns=fdi_feature_candidates(frame) if variant_name == "fdi_augmented" else None,
        )
        result = run_model_suite(
            root=root,
            full_data=holdout_data,
            report_root_name=f"phase3_zero_day_artifacts/fdi_{variant_name}",
            model_root_name=f"phase3_zero_day/fdi_ablation/{variant_name}",
            include_models=["threshold_baseline", "autoencoder", "transformer"],
            model_settings=model_settings,
        )
        summary_df = result["summary_df"].copy()
        summary_df["feature_variant"] = variant_name
        ablation_rows.extend(summary_df.to_dict(orient="records"))

        ranking = holdout_data.feature_ranking.copy()
        ranking = ranking[
            ranking["feature"].str.startswith("fdi__")
            | ranking["feature"].str.contains("pv60", regex=False)
            | ranking["feature"].str.contains("cyber_", regex=False)
        ].copy()
        ranking["feature_variant"] = variant_name
        importance_rows.append(ranking.head(30))

    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        ablation_df = ablation_df.sort_values(["recall", "precision", "f1"], ascending=[False, False, False]).reset_index(drop=True)
    importance_df = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()

    ablation_csv = artifact_root / "fdi_ablation_table.csv"
    importance_csv = artifact_root / "fdi_feature_importance.csv"
    plot_path = artifact_root / "FDI_detection_comparison.png"
    ablation_df.to_csv(ablation_csv, index=False)
    importance_df.to_csv(importance_csv, index=False)
    plot_metric_bars(
        ablation_df.assign(model_variant=ablation_df["model_name"] + " | " + ablation_df["feature_variant"]),
        "model_variant",
        plot_path,
        ["precision", "recall", "f1"],
        "Stealth FDI Detection Comparison",
    )

    sections = [
        (
            "Study Setup",
            "\n".join(
                [
                    f"- holdout_scenario: `{args.holdout_scenario}`",
                    "- compared feature sets: `baseline_residual`, `fdi_augmented`",
                    "- evaluated models: `threshold_baseline`, `autoencoder`, `transformer`",
                ]
            ),
        ),
        ("FDI Ablation Results", to_markdown(ablation_df)),
        ("FDI-Oriented Feature Ranking", to_markdown(importance_df.head(25))),
    ]
    write_phase3_report(artifact_root / "fdi_report.md", "# Phase 3 FDI Ablation Report", sections)
    print(f"FDI ablation artifacts written to {artifact_root}")


if __name__ == "__main__":
    main()

