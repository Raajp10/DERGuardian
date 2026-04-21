from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from common.io_utils import ensure_dir
from phase1_models.run_full_evaluation import assign_attack_aware_split
from phase3.experiment_utils import (
    build_full_run_data,
    build_window_bundle_for_config,
    infer_candidate_columns,
    plot_latency_tradeoff,
    prepare_window_bundle,
    run_model_suite,
    summarize_latency,
    to_markdown,
    write_phase3_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 latency-vs-window-size experiments.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--configs", default="300:60,120:24,120:12")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.project_root)
    artifact_root = ensure_dir(root / "outputs" / "reports" / "phase3_zero_day_artifacts")
    window_root = ensure_dir(root / "outputs" / "phase3_zero_day" / "latency_window_sweep")

    rows: list[dict[str, object]] = []
    for config_token in [item for item in args.configs.split(",") if item.strip()]:
        window_seconds, step_seconds = [int(part) for part in config_token.split(":")]
        config_slug = f"w{window_seconds}_s{step_seconds}"
        if window_seconds == 300 and step_seconds == 60:
            bundle = prepare_window_bundle(root)
        else:
            clean_path, attacked_path = build_window_bundle_for_config(root, window_seconds, step_seconds, window_root / config_slug)
            bundle = prepare_window_bundle(root, clean_windows_path=clean_path, attacked_windows_path=attacked_path)

        split_df = assign_attack_aware_split(bundle.residual_df, bundle.labels_df, buffer_windows=2)
        full_data = build_full_run_data(
            feature_df=bundle.residual_df,
            labels_df=bundle.labels_df,
            split_df=split_df,
            candidate_feature_columns=infer_candidate_columns(bundle.residual_df, mode="residual"),
        )
        result = run_model_suite(
            root=root,
            full_data=full_data,
            report_root_name=f"phase3_zero_day_artifacts/latency_{config_slug}",
            model_root_name=f"phase3_zero_day/latency_window_sweep/{config_slug}",
            include_models=["threshold_baseline", "autoencoder", "transformer"],
            model_settings={
                "threshold_baseline": {"feature_counts": [32, 64, 96]},
                "autoencoder": {"feature_counts": [32, 64], "epochs": args.epochs, "patience": args.patience},
                "transformer": {"feature_counts": [32, 64], "seq_lens": [4, 8], "epochs": args.epochs, "patience": args.patience},
            },
        )
        summary_df = result["summary_df"].copy()
        latency_summary = summarize_latency(result["latency_df"], ["model_name"])
        merged = summary_df.merge(latency_summary, on="model_name", how="left")
        merged["window_seconds"] = window_seconds
        merged["step_seconds"] = step_seconds
        merged["window_config"] = f"{window_seconds}s/{step_seconds}s"
        rows.extend(merged.to_dict(orient="records"))

    sweep_df = pd.DataFrame(rows)
    sweep_csv = artifact_root / "latency_window_sweep_table.csv"
    plot_path = artifact_root / "latency_window_sweep_plot.png"
    sweep_df.to_csv(sweep_csv, index=False)
    plot_latency_tradeoff(
        sweep_df.assign(label=sweep_df["model_name"] + " " + sweep_df["window_config"]),
        x_column="mean_latency_seconds",
        y_column="f1",
        label_column="label",
        path=plot_path,
        title="Latency / F1 Trade-Off Across Window Configurations",
    )
    sections = [
        (
            "Configurations",
            "\n".join([f"- `{item}`" for item in [token for token in args.configs.split(",") if token.strip()]]),
        ),
        ("Latency Window Sweep", to_markdown(sweep_df)),
    ]
    write_phase3_report(artifact_root / "latency_tradeoff_report.md", "# Phase 3 Latency / Window Trade-Off Report", sections)
    print(f"Latency sweep artifacts written to {artifact_root}")


if __name__ == "__main__":
    main()

