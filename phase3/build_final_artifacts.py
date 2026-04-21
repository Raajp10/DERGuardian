from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.io_utils import ensure_dir, write_json
from phase3.experiment_utils import plot_latency_tradeoff, to_markdown, write_phase3_report
from phase1_models.model_utils import apply_model_display_names, display_model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble final Phase 3 paper-ready artifacts and report.")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.project_root)
    phase1_root = root / "outputs" / "reports" / "model_full_run_artifacts"
    phase3_root = root / "outputs" / "reports" / "phase3_zero_day_artifacts"
    final_root = ensure_dir(root / "outputs" / "reports" / "final_paper_artifacts")

    same_df = pd.read_csv(phase1_root / "model_summary_table_full.csv") if (phase1_root / "model_summary_table_full.csv").exists() else pd.DataFrame()
    zero_day_df = pd.read_csv(phase3_root / "zero_day_model_summary.csv") if (phase3_root / "zero_day_model_summary.csv").exists() else pd.DataFrame()
    latency_df = pd.read_csv(phase3_root / "latency_window_sweep_table.csv") if (phase3_root / "latency_window_sweep_table.csv").exists() else pd.DataFrame()
    modality_df = pd.read_csv(phase3_root / "modality_ablation_table.csv") if (phase3_root / "modality_ablation_table.csv").exists() else pd.DataFrame()
    fdi_df = pd.read_csv(phase3_root / "fdi_ablation_table.csv") if (phase3_root / "fdi_ablation_table.csv").exists() else pd.DataFrame()
    zero_day_latency = pd.read_csv(phase3_root / "zero_day_latency_analysis.csv") if (phase3_root / "zero_day_latency_analysis.csv").exists() else pd.DataFrame()
    error_summary = _read_json(root / "outputs" / "reports" / "phase3_error_analysis_summary.json")
    case_studies_path = root / "outputs" / "reports" / "phase3_case_studies.md"

    final_comparison = _build_final_comparison_table(same_df, zero_day_df)
    final_zero_day = zero_day_df.copy()
    final_latency = latency_df.copy()
    final_modality = modality_df.copy()

    _write_table(final_comparison, final_root / "final_model_comparison_table.csv", final_root / "final_model_comparison_table.md")
    final_zero_day.to_csv(final_root / "final_zero_day_table.csv", index=False)
    final_latency.to_csv(final_root / "final_latency_table.csv", index=False)
    final_modality.to_csv(final_root / "final_modality_ablation_table.csv", index=False)

    recommendation = _build_recommendation_text(same_df, zero_day_df)
    (final_root / "final_best_model_recommendation.md").write_text(recommendation, encoding="utf-8")
    (final_root / "final_results_discussion.md").write_text(_build_discussion_text(same_df, zero_day_df, latency_df, modality_df, fdi_df, error_summary), encoding="utf-8")
    (final_root / "final_limitations.md").write_text(_build_limitations_text(zero_day_df, latency_df, fdi_df), encoding="utf-8")

    figures_manifest: list[dict[str, str]] = []
    tables_manifest: list[dict[str, str]] = []

    same_zero_plot = final_root / "same_scenario_vs_zero_day_comparison.png"
    _plot_same_vs_zero_day(same_df, zero_day_df, same_zero_plot)
    figures_manifest.append({"path": str(same_zero_plot), "description": "Same-scenario vs zero-day F1 comparison."})

    latency_model_plot = final_root / "latency_by_model.png"
    _plot_latency_by_model(zero_day_latency, latency_model_plot)
    figures_manifest.append({"path": str(latency_model_plot), "description": "Mean zero-day latency by model."})

    latency_tradeoff_plot = final_root / "latency_vs_f1_tradeoff.png"
    plot_latency_tradeoff(
        latency_df.assign(label=latency_df["model_name"] + " " + latency_df["window_config"]) if not latency_df.empty else latency_df,
        x_column="mean_latency_seconds",
        y_column="f1",
        label_column="label",
        path=latency_tradeoff_plot,
        title="Latency vs F1 Trade-Off",
    )
    figures_manifest.append({"path": str(latency_tradeoff_plot), "description": "Latency vs F1 trade-off from the window sweep."})

    fdi_plot_src = phase3_root / "FDI_detection_comparison.png"
    if fdi_plot_src.exists():
        fdi_plot_dst = final_root / "FDI_detection_comparison.png"
        shutil.copy2(fdi_plot_src, fdi_plot_dst)
        figures_manifest.append({"path": str(fdi_plot_dst), "description": "Stealth FDI detection comparison."})

    modality_plot_src = phase3_root / "modality_ablation_plot.png"
    if modality_plot_src.exists():
        modality_plot_dst = final_root / "modality_ablation_comparison.png"
        shutil.copy2(modality_plot_src, modality_plot_dst)
        figures_manifest.append({"path": str(modality_plot_dst), "description": "Cyber vs physical modality ablation comparison."})

    best_model_name = _best_model_name(same_df)
    best_model_dir = phase1_root / best_model_name if best_model_name else None
    if best_model_dir and best_model_dir.exists():
        _copy_if_exists(best_model_dir / "confusion_matrix.png", final_root / "best_model_confusion_matrix.png", figures_manifest, "Best model confusion matrix.")
        _copy_if_exists(best_model_dir / "precision_recall_curve.png", final_root / "best_model_pr_curve.png", figures_manifest, "Best model precision-recall curve.")
        _copy_if_exists(best_model_dir / "roc_curve.png", final_root / "best_model_roc_curve.png", figures_manifest, "Best model ROC curve.")

    for path, description in [
        (final_root / "final_model_comparison_table.csv", "Combined same-scenario and zero-day comparison table."),
        (final_root / "final_model_comparison_table.md", "Combined same-scenario and zero-day comparison table in Markdown."),
        (final_root / "final_zero_day_table.csv", "Zero-day aggregate results."),
        (final_root / "final_latency_table.csv", "Latency window sweep summary."),
        (final_root / "final_modality_ablation_table.csv", "Cyber-vs-physical modality ablation summary."),
        (final_root / "final_best_model_recommendation.md", "Recommended model for thesis/paper discussion."),
        (final_root / "final_results_discussion.md", "Final Phase 3 results discussion."),
        (final_root / "final_limitations.md", "Final Phase 3 limitations statement."),
    ]:
        tables_manifest.append({"path": str(path), "description": description})
    for source, description in [
        (root / "outputs" / "reports" / "phase3_error_analysis_summary.json", "Phase 3 machine-readable error-analysis summary."),
        (root / "outputs" / "reports" / "phase3_error_analysis_summary.md", "Phase 3 markdown error-analysis summary."),
        (root / "outputs" / "reports" / "phase3_false_positive_analysis.csv", "Per-model false-positive analysis across unseen-scenario holdouts."),
        (root / "outputs" / "reports" / "phase3_false_negative_analysis.csv", "Per-model false-negative analysis across unseen-scenario holdouts."),
        (root / "outputs" / "reports" / "phase3_per_scenario_error_patterns.csv", "Per-scenario error-pattern summary across models."),
        (case_studies_path, "Compact Phase 3 case studies grounded in the expanded holdout evaluation."),
    ]:
        if source.exists():
            destination = final_root / source.name
            shutil.copy2(source, destination)
            tables_manifest.append({"path": str(destination), "description": description})

    write_json(figures_manifest, final_root / "final_figures_manifest.json")
    write_json(tables_manifest, final_root / "final_tables_manifest.json")

    report_sections = [
        ("Phase 3 Scripts Added", _phase3_script_list(root)),
        ("Zero-Day Construction", _zero_day_construction_text()),
        ("Best Same-Scenario Model", _best_same_scenario_text(same_df)),
        ("Best Zero-Day Model", _best_zero_day_text(zero_day_df)),
        ("Fusion Helpfulness", _fusion_help_text(modality_df)),
        ("Latency Improvement", _latency_text(latency_df)),
        ("FDI Improvement", _fdi_text(fdi_df)),
        ("Error Analysis Highlights", _error_analysis_text(error_summary)),
        ("Paper-Usable Results", _paper_usable_text(same_df, zero_day_df)),
    ]
    write_phase3_report(root / "reports" / "phase3_final_report.md", "# Phase 3 Final Research Report", report_sections)
    print(f"Final paper artifacts written to {final_root}")


def _build_final_comparison_table(same_df: pd.DataFrame, zero_day_df: pd.DataFrame) -> pd.DataFrame:
    same_table = same_df[["model_name", "precision", "recall", "f1", "average_precision", "roc_auc"]].copy() if not same_df.empty else pd.DataFrame()
    if not same_table.empty:
        same_table["evaluation_regime"] = "same_scenario"
    zero_table = zero_day_df[["model_name", "precision", "recall", "f1", "average_precision", "roc_auc"]].copy() if not zero_day_df.empty else pd.DataFrame()
    if not zero_table.empty:
        zero_table["evaluation_regime"] = "zero_day"
    combined = pd.concat([same_table, zero_table], ignore_index=True) if not same_table.empty or not zero_table.empty else pd.DataFrame()
    if not combined.empty:
        combined = combined[["evaluation_regime", "model_name", "precision", "recall", "f1", "average_precision", "roc_auc"]]
    return combined


def _write_table(frame: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(apply_model_display_names(frame)), encoding="utf-8")


def _build_recommendation_text(same_df: pd.DataFrame, zero_day_df: pd.DataFrame) -> str:
    same_name = _best_model_name(same_df)
    zero_name = _best_model_name(zero_day_df)
    return "\n".join(
        [
            "# Final Best-Model Recommendation",
            "",
            f"- same_scenario_recommendation: `{display_model_name(same_name)}`",
            f"- zero_day_recommendation: `{display_model_name(zero_name)}`",
            "- Recommendation: use the same-scenario threshold baseline as the operational anchor, and cite the zero-day winner separately for generalization analysis.",
            "",
        ]
    )


def _build_discussion_text(
    same_df: pd.DataFrame,
    zero_day_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    modality_df: pd.DataFrame,
    fdi_df: pd.DataFrame,
    error_summary: dict[str, object],
) -> str:
    return "\n".join(
        [
            "# Final Results Discussion",
            "",
            _best_same_scenario_text(same_df),
            _best_zero_day_text(zero_day_df),
            _latency_text(latency_df),
            _fusion_help_text(modality_df),
            _fdi_text(fdi_df),
            _error_analysis_text(error_summary),
            "",
        ]
    )


def _build_limitations_text(zero_day_df: pd.DataFrame, latency_df: pd.DataFrame, fdi_df: pd.DataFrame) -> str:
    holdout_count = int(zero_day_df["total_scenarios"].max()) if not zero_day_df.empty and "total_scenarios" in zero_day_df.columns else 0
    lines = [
        "# Final Limitations",
        "",
        f"- Zero-day-like evaluation is leave-one-scenario-out over {holdout_count or 'the currently available'} synthetic scenarios; it is stronger than same-scenario testing but still bounded to repository-generated holdouts.",
        "- Smaller windows were studied for latency reduction, but not every model was fully re-tuned for every window configuration.",
        "- FDI-focused features improve interpretability and targeted analysis, but stealth telemetry corruption remains the hardest class.",
    ]
    if zero_day_df.empty:
        lines.append("- Zero-day aggregate outputs were unavailable at final-assembly time.")
    if latency_df.empty:
        lines.append("- Latency sweep outputs were unavailable at final-assembly time.")
    if fdi_df.empty:
        lines.append("- FDI ablation outputs were unavailable at final-assembly time.")
    lines.append("")
    return "\n".join(lines)


def _plot_same_vs_zero_day(same_df: pd.DataFrame, zero_day_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if same_df.empty or zero_day_df.empty:
        ax.text(0.5, 0.5, "Comparison unavailable", ha="center", va="center")
    else:
        merged = same_df[["model_name", "f1"]].merge(zero_day_df[["model_name", "f1"]], on="model_name", how="inner", suffixes=("_same", "_zero"))
        x = np.arange(len(merged))
        width = 0.35
        ax.bar(x - width / 2.0, merged["f1_same"], width=width, label="same_scenario")
        ax.bar(x + width / 2.0, merged["f1_zero"], width=width, label="zero_day")
        ax.set_xticks(x)
        ax.set_xticklabels(merged["model_name"].map(display_model_name), rotation=30, ha="right")
        ax.legend(loc="best")
    ax.set_ylabel("F1")
    ax.set_title("Same-Scenario vs Zero-Day Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_latency_by_model(latency_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if latency_df.empty:
        ax.text(0.5, 0.5, "Latency results unavailable", ha="center", va="center")
    else:
        frame = latency_df.copy()
        frame = frame.groupby("model_name", observed=False)["latency_seconds"].mean().reset_index()
        x = np.arange(len(frame))
        ax.bar(x, frame["latency_seconds"])
        ax.set_xticks(x)
        ax.set_xticklabels(frame["model_name"].map(display_model_name), rotation=30, ha="right")
    ax.set_ylabel("Mean latency (s)")
    ax.set_title("Latency by Model")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _copy_if_exists(source: Path, destination: Path, manifest: list[dict[str, str]], description: str) -> None:
    if source.exists():
        shutil.copy2(source, destination)
        manifest.append({"path": str(destination), "description": description})


def _best_model_name(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return str(frame.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]["model_name"])


def _best_same_scenario_text(same_df: pd.DataFrame) -> str:
    if same_df.empty:
        return "- Same-scenario benchmark summary unavailable."
    row = same_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    return f"- Best same-scenario model: `{display_model_name(row['model_name'])}` with precision `{row['precision']}`, recall `{row['recall']}`, F1 `{row['f1']}`."


def _best_zero_day_text(zero_day_df: pd.DataFrame) -> str:
    if zero_day_df.empty:
        return "- Zero-day summary unavailable."
    row = zero_day_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    return f"- Best zero-day model: `{display_model_name(row['model_name'])}` with precision `{row['precision']}`, recall `{row['recall']}`, F1 `{row['f1']}`."


def _fusion_help_text(modality_df: pd.DataFrame) -> str:
    if modality_df.empty:
        return "- Modality ablation unavailable."
    best = modality_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    return f"- Best modality condition: `{best['modality']}` for model `{display_model_name(best['model_name'])}` with F1 `{best['f1']}`."


def _latency_text(latency_df: pd.DataFrame) -> str:
    if latency_df.empty:
        return "- Latency sweep unavailable."
    threshold_rows = latency_df[latency_df["model_name"] == "threshold_baseline"].copy()
    if not threshold_rows.empty:
        baseline = threshold_rows.sort_values("window_seconds", ascending=False).iloc[0]
        best_f1 = threshold_rows.sort_values(["f1", "mean_latency_seconds"], ascending=[False, True]).iloc[0]
        min_latency = threshold_rows.sort_values("mean_latency_seconds").iloc[0]
        if float(min_latency["mean_latency_seconds"]) < float(baseline["mean_latency_seconds"]):
            return (
                f"- Smaller windows reduced latency for the threshold baseline to `{min_latency['mean_latency_seconds']}` seconds at "
                f"`{min_latency['window_config']}`, while the best threshold F1 was `{best_f1['f1']}` at `{best_f1['window_config']}`."
            )
        return (
            f"- Smaller windows did not produce a clear latency win for the strongest threshold model. "
            f"The baseline `300s/60s` setting stayed lower-latency at `{baseline['mean_latency_seconds']}` seconds, while the best threshold F1 "
            f"`{best_f1['f1']}` occurred at `{best_f1['window_config']}` with mean latency `{best_f1['mean_latency_seconds']}` seconds."
        )
    best = latency_df.sort_values(["f1", "mean_latency_seconds"], ascending=[False, True]).iloc[0]
    return f"- Best latency trade-off observed at `{best['window_config']}` for `{display_model_name(best['model_name'])}` with F1 `{best['f1']}` and mean latency `{best['mean_latency_seconds']}` seconds."


def _fdi_text(fdi_df: pd.DataFrame) -> str:
    if fdi_df.empty:
        return "- FDI ablation unavailable."
    best = fdi_df.sort_values(["recall", "precision", "f1"], ascending=[False, False, False]).iloc[0]
    return f"- Best stealth-FDI result: `{display_model_name(best['model_name'])}` under `{best['feature_variant']}` with precision `{best['precision']}`, recall `{best['recall']}`, F1 `{best['f1']}`."


def _paper_usable_text(same_df: pd.DataFrame, zero_day_df: pd.DataFrame) -> str:
    if same_df.empty or zero_day_df.empty:
        return "- Final paper-usable claim is limited because one or more aggregate result tables are missing."
    return "- The project now has paper-usable same-scenario and unseen-scenario result tables, plus latency, FDI, modality, and error-analysis studies suitable for a thesis or paper results section."


def _phase3_script_list(root: Path) -> str:
    scripts = sorted(path.name for path in (root / "phase3").glob("*.py"))
    return "\n".join(f"- `{name}`" for name in scripts)


def _zero_day_construction_text() -> str:
    return "\n".join(
        [
            "- For each attack scenario, the held-out scenario attack windows were reserved for test together with local benign context windows.",
            "- Training used benign windows plus all non-held-out attack scenarios.",
            "- Validation used only non-held-out scenarios, preserving leave-one-scenario-out zero-day integrity.",
        ]
    )


def _error_analysis_text(error_summary: dict[str, object]) -> str:
    if not error_summary:
        return "- Phase 3 error-analysis artifacts were unavailable at final-assembly time."
    findings = error_summary.get("key_findings", [])
    if isinstance(findings, list) and findings:
        return "- " + " ".join(str(item) for item in findings[:3])
    return "- Phase 3 error-analysis artifacts exist, but no summary findings were recorded."


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
