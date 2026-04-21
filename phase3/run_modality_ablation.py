from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from common.io_utils import ensure_dir
from phase1_models.run_full_evaluation import assign_attack_aware_split
from phase3.experiment_utils import (
    build_full_run_data,
    build_modality_frame,
    infer_candidate_columns,
    plot_metric_bars,
    prepare_window_bundle,
    run_model_suite,
    to_markdown,
    write_phase3_report,
)


MODALITIES = ["physical_only", "cyber_only", "fused", "fused_plus_residual"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cyber-vs-physical modality ablations for Phase 3.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.project_root)
    artifact_root = ensure_dir(root / "outputs" / "reports" / "phase3_zero_day_artifacts")
    bundle = prepare_window_bundle(root)
    best_sequence_path = artifact_root / "best_sequence_configs.json"
    best_sequence_configs = json.loads(best_sequence_path.read_text(encoding="utf-8")) if best_sequence_path.exists() else []
    transformer_config = next((item for item in best_sequence_configs if item.get("model_name") == "transformer"), {})
    transformer_feature_count = int(transformer_config.get("feature_count", 32) or 32)
    transformer_seq_len = int(transformer_config.get("seq_len", 4) or 4)

    rows: list[dict[str, object]] = []
    for modality in MODALITIES:
        modality_frame = build_modality_frame(bundle, modality)
        split_df = assign_attack_aware_split(modality_frame, bundle.labels_df, buffer_windows=2)
        full_data = build_full_run_data(
            feature_df=modality_frame,
            labels_df=bundle.labels_df,
            split_df=split_df,
            candidate_feature_columns=infer_candidate_columns(modality_frame, mode=modality),
        )
        result = run_model_suite(
            root=root,
            full_data=full_data,
            report_root_name=f"phase3_zero_day_artifacts/modality_{modality}",
            model_root_name=f"phase3_zero_day/modality_ablation/{modality}",
            include_models=["threshold_baseline", "autoencoder", "transformer"],
            model_settings={
                "threshold_baseline": {"feature_counts": [32, 64, 96]},
                "autoencoder": {"feature_counts": [32, 64, 96], "epochs": args.epochs, "patience": args.patience},
                "transformer": {
                    "feature_counts": [transformer_feature_count],
                    "seq_lens": [transformer_seq_len],
                    "epochs": args.epochs,
                    "patience": args.patience,
                },
            },
        )
        summary_df = result["summary_df"].copy()
        summary_df["modality"] = modality
        rows.extend(summary_df.to_dict(orient="records"))

    ablation_df = pd.DataFrame(rows)
    ablation_csv = artifact_root / "modality_ablation_table.csv"
    plot_path = artifact_root / "modality_ablation_plot.png"
    ablation_df.to_csv(ablation_csv, index=False)
    plot_metric_bars(
        ablation_df.assign(model_modality=ablation_df["model_name"] + " | " + ablation_df["modality"]),
        "model_modality",
        plot_path,
        ["precision", "recall", "f1"],
        "Cyber / Physical Modality Ablation",
    )

    sections = [
        (
            "Study Setup",
            "\n".join(
                [
                    "- evaluated models: `threshold_baseline`, `autoencoder`, `transformer`",
                    f"- transformer config: feature_count `{transformer_feature_count}`, seq_len `{transformer_seq_len}`",
                    f"- modalities: `{MODALITIES}`",
                ]
            ),
        ),
        ("Modality Ablation Results", to_markdown(ablation_df)),
    ]
    write_phase3_report(artifact_root / "modality_ablation_report.md", "# Phase 3 Modality Ablation Report", sections)
    print(f"Modality ablation artifacts written to {artifact_root}")


if __name__ == "__main__":
    main()

