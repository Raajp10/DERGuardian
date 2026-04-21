"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements generate paper figures logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from pathlib import Path
import argparse
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
from sklearn.metrics import confusion_matrix

from common.io_utils import ensure_dir, read_dataframe, write_json
from phase1_models.model_utils import display_model_name


MODEL_ORDER = [
    "threshold_baseline",
    "isolation_forest",
    "autoencoder",
    "gru",
    "lstm",
    "transformer",
    "llm_baseline",
]
NEURAL_MODELS = {"autoencoder", "gru", "lstm", "transformer", "llm_baseline"}
MODEL_COLORS = {
    "threshold_baseline": "#1f77b4",
    "isolation_forest": "#ff7f0e",
    "autoencoder": "#2ca02c",
    "gru": "#d62728",
    "lstm": "#9467bd",
    "transformer": "#8c564b",
    "llm_baseline": "#e377c2",
}
METRIC_COLORS = {
    "precision": "#1f77b4",
    "recall": "#ff7f0e",
    "f1": "#2ca02c",
    "average_precision": "#d62728",
}


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Generate paper-ready Phase 1 figures from existing full-run artifacts.")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.project_root)
    model_root = root / "outputs" / "models_full_run"
    report_root = root / "outputs" / "reports" / "model_full_run_artifacts"
    paper_root = ensure_dir(root / "outputs" / "reports" / "paper_figures")
    per_model_root = ensure_dir(paper_root / "per_model")
    comparison_root = ensure_dir(paper_root / "comparison")

    _configure_style()

    model_summary = pd.read_csv(report_root / "model_summary_table_full.csv")
    latency_df = pd.read_csv(report_root / "detection_latency_analysis_full.csv")
    fusion_df = pd.read_csv(report_root / "fusion_ablation_table.csv")

    model_artifacts: dict[str, dict[str, object]] = {}
    generated_paths: list[str] = []
    failures: list[str] = []

    for model_name in MODEL_ORDER:
        try:
            artifacts = _load_model_artifacts(model_root, model_name)
            model_artifacts[model_name] = artifacts
            output_dir = ensure_dir(per_model_root / model_name)
            generated_paths.extend(_generate_per_model_figures(model_name, artifacts, output_dir))
        except Exception as exc:  # pragma: no cover - defensive reporting
            failures.append(f"{model_name}: {exc}")

    comparison_paths = _generate_comparison_figures(
        model_summary=model_summary,
        model_artifacts=model_artifacts,
        fusion_df=fusion_df,
        latency_df=latency_df,
        comparison_root=comparison_root,
        paper_root=paper_root,
    )
    generated_paths.extend(comparison_paths)

    manifest = {
        "paper_root": str(paper_root),
        "generated_paths": generated_paths,
        "failures": failures,
        "model_count": len(model_artifacts),
    }
    write_json(manifest, paper_root / "phase1_paper_figures_manifest.json")
    print(json.dumps(manifest, indent=2))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def _load_model_artifacts(model_root: Path, model_name: str) -> dict[str, object]:
    result_path = model_root / model_name / "results.json"
    prediction_path = model_root / model_name / "predictions.parquet"
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    predictions = read_dataframe(prediction_path)
    return {
        "result": result,
        "predictions": predictions,
    }


def _generate_per_model_figures(model_name: str, artifacts: dict[str, object], output_dir: Path) -> list[str]:
    result = dict(artifacts["result"])
    predictions = artifacts["predictions"].copy()
    metrics = result["metrics"]
    curves = result["curves"]
    history = result.get("history", {})
    display_name = display_model_name(model_name)
    color = MODEL_COLORS[model_name]
    generated: list[str] = []

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(curves.get("roc_fpr", []), curves.get("roc_tpr", []), color=color, linewidth=2.4, label=f"AUC = {float(metrics.get('roc_auc', 0.0)):.3f}")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.2)
    ax.set_title(f"{display_name} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    generated.extend(_save_figure(fig, output_dir / "roc_curve"))

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(curves.get("pr_recall", []), curves.get("pr_precision", []), color=color, linewidth=2.4, label=f"AP = {float(metrics.get('average_precision', 0.0)):.3f}")
    ax.set_title(f"{display_name} Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    generated.extend(_save_figure(fig, output_dir / "precision_recall_curve"))

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    _plot_confusion_matrix(ax, predictions, display_name)
    generated.extend(_save_figure(fig, output_dir / "confusion_matrix"))

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    _plot_score_distribution(ax, predictions, float(metrics.get("threshold", 0.5) or 0.5), display_name)
    generated.extend(_save_figure(fig, output_dir / "score_distribution"))

    if model_name in NEURAL_MODELS:
        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        _plot_training_curve(ax, history, display_name)
        generated.extend(_save_figure(fig, output_dir / "training_curve"))

    return generated


def _generate_comparison_figures(
    *,
    model_summary: pd.DataFrame,
    model_artifacts: dict[str, dict[str, object]],
    fusion_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    comparison_root: Path,
    paper_root: Path,
) -> list[str]:
    generated: list[str] = []
    ordered = model_summary.copy()
    ordered["order"] = ordered["model_name"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    ordered = ordered.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    _plot_model_comparison(ax, ordered)
    generated.extend(_save_figure(fig, comparison_root / "model_performance_comparison"))

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _plot_roc_overlay(ax, model_artifacts)
    generated.extend(_save_figure(fig, comparison_root / "roc_overlay"))

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _plot_pr_overlay(ax, model_artifacts)
    generated.extend(_save_figure(fig, comparison_root / "precision_recall_overlay"))

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    _plot_fusion_ablation(ax, fusion_df)
    generated.extend(_save_figure(fig, comparison_root / "fusion_ablation_summary"))

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    _plot_latency_comparison(ax, latency_df)
    generated.extend(_save_figure(fig, comparison_root / "offline_detection_latency_comparison"))

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    _plot_model_comparison(axes[0, 0], ordered)
    _plot_roc_overlay(axes[0, 1], model_artifacts)
    _plot_pr_overlay(axes[1, 0], model_artifacts)
    _plot_fusion_ablation(axes[1, 1], fusion_df)
    fig.suptitle("Phase 1 Paper Overview Panel", y=1.02)
    generated.extend(_save_figure(fig, paper_root / "phase1_paper_overview_panel"))

    return generated


def _plot_confusion_matrix(ax: plt.Axes, predictions: pd.DataFrame, display_name: str) -> None:
    y_true = predictions["attack_present"].astype(int).to_numpy()
    y_pred = predictions["predicted"].astype(int).to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Normal", "Anomaly"])
    ax.set_yticks([0, 1], labels=["Normal", "Anomaly"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{display_name} Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n{normalized[i, j]*100:.1f}%",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
                fontsize=11,
                fontweight="bold",
            )


def _plot_score_distribution(ax: plt.Axes, predictions: pd.DataFrame, threshold: float, display_name: str) -> None:
    benign = predictions.loc[predictions["attack_present"] == 0, "score"].astype(float)
    attack = predictions.loc[predictions["attack_present"] == 1, "score"].astype(float)
    bins = 28
    if not benign.empty:
        ax.hist(benign, bins=bins, alpha=0.6, density=True, label="Normal", color="#4c78a8")
    if not attack.empty:
        ax.hist(attack, bins=bins, alpha=0.55, density=True, label="Anomaly", color="#f58518")
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=2.0, label=f"Threshold = {threshold:.3f}")
    ax.set_title(f"{display_name} Score Distribution")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")


def _plot_training_curve(ax: plt.Axes, history: dict[str, list[float]], display_name: str) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = np.arange(1, max(len(train_loss), len(val_loss)) + 1)
    if train_loss:
        ax.plot(epochs[: len(train_loss)], train_loss, linewidth=2.3, color="#1f77b4", label="Train Loss")
    if val_loss:
        ax.plot(epochs[: len(val_loss)], val_loss, linewidth=2.3, color="#d62728", label="Validation Loss")
    ax.set_title(f"{display_name} Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")


def _plot_model_comparison(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    labels = [display_model_name(name) for name in summary_df["model_name"]]
    x = np.arange(len(summary_df))
    width = 0.24
    for idx, metric in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (idx - 1) * width, summary_df[metric], width=width, label=metric.title(), color=METRIC_COLORS[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="upper right")


def _plot_roc_overlay(ax: plt.Axes, model_artifacts: dict[str, dict[str, object]]) -> None:
    for model_name in MODEL_ORDER:
        if model_name not in model_artifacts:
            continue
        result = model_artifacts[model_name]["result"]
        curves = result["curves"]
        roc_auc = float(result["metrics"].get("roc_auc", 0.0))
        ax.plot(
            curves.get("roc_fpr", []),
            curves.get("roc_tpr", []),
            linewidth=2.0,
            color=MODEL_COLORS[model_name],
            label=f"{display_model_name(model_name)} (AUC={roc_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.2)
    ax.set_title("ROC Overlay Across Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)


def _plot_pr_overlay(ax: plt.Axes, model_artifacts: dict[str, dict[str, object]]) -> None:
    for model_name in MODEL_ORDER:
        if model_name not in model_artifacts:
            continue
        result = model_artifacts[model_name]["result"]
        curves = result["curves"]
        ap = float(result["metrics"].get("average_precision", 0.0))
        ax.plot(
            curves.get("pr_recall", []),
            curves.get("pr_precision", []),
            linewidth=2.0,
            color=MODEL_COLORS[model_name],
            label=f"{display_model_name(model_name)} (AP={ap:.3f})",
        )
    ax.set_title("Precision-Recall Overlay Across Models")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)


def _plot_fusion_ablation(ax: plt.Axes, fusion_df: pd.DataFrame) -> None:
    ordered_modes = ["detector_only", "detector_plus_context", "detector_plus_context_plus_token"]
    plot_df = fusion_df.set_index("fusion_mode").loc[ordered_modes].reset_index()
    labels = [label.replace("_", "\n") for label in plot_df["fusion_mode"]]
    x = np.arange(len(plot_df))
    width = 0.18
    metrics = ["precision", "recall", "f1", "average_precision"]
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1.5) * width, plot_df[metric], width=width, label=("AP" if metric == "average_precision" else metric.title()), color=METRIC_COLORS.get(metric, "#777777"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Fusion Ablation Summary")
    ax.legend(loc="upper right", ncol=2)


def _plot_latency_comparison(ax: plt.Axes, latency_df: pd.DataFrame) -> None:
    plot_df = latency_df.dropna(subset=["latency_seconds"]).copy()
    grouped_data = []
    labels = []
    colors = []
    for model_name in MODEL_ORDER:
        group = plot_df.loc[plot_df["model_name"] == model_name, "latency_seconds"].astype(float).to_numpy()
        if len(group) == 0:
            continue
        grouped_data.append(group)
        labels.append(display_model_name(model_name))
        colors.append(MODEL_COLORS[model_name])
    if grouped_data:
        box = ax.boxplot(grouped_data, patch_artist=True, tick_labels=labels, showfliers=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in box["medians"]:
            median.set_color("#222222")
            median.set_linewidth(2.0)
    ax.set_title("Offline Detection Latency Comparison")
    ax.set_ylabel("Latency (seconds)")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)


def _save_figure(fig: plt.Figure, output_stem: Path) -> list[str]:
    fig.tight_layout()
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


if __name__ == "__main__":
    main()
