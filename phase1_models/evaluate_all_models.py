"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements evaluate all models logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from common.io_utils import read_dataframe
from phase1_models.dataset_loader import load_window_dataset_bundle
from phase1_models.metrics import detection_latency_table
from phase1_models.model_utils import (
    display_model_name,
    CANONICAL_ARTIFACT_ROOT,
    CANONICAL_MODEL_ROOT,
    LEGACY_ARTIFACT_ROOT,
    LEGACY_MODEL_ROOT,
)
from phase1_models.run_full_evaluation import run_full_benchmark
from phase1_models.train_autoencoder import train_autoencoder_model
from phase1_models.train_isolation_forest import train_isolation_forest
from phase1_models.train_llm_baseline import train_llm_style_baseline
from phase1_models.train_threshold_baseline import train_threshold_model
from phase1_models.sequence_train_utils import train_sequence_classifier_model
from phase1_models.neural_models import GRUClassifier, LSTMClassifier, TinyTransformerClassifier


MODEL_NAMES = ["threshold_baseline", "isolation_forest", "autoencoder", "gru", "lstm", "transformer", "llm_baseline"]


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Canonical full-run and legacy-smoke evaluation pipeline for Phase 1 DER anomaly models.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--run-mode", default="full", choices=["full", "smoke", "legacy-smoke"])
    parser.add_argument("--smoke-train", action="store_true")
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--feature-counts", default="32,64,96,128")
    parser.add_argument("--seq-lens", default="4,8,12")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--buffer-windows", type=int, default=2)
    parser.add_argument("--window-seconds", type=int, default=300)
    args = parser.parse_args()

    root = Path(args.project_root)
    run_mode = "legacy-smoke" if args.run_mode == "smoke" else args.run_mode
    if args.run_mode == "smoke":
        print(
            "Deprecated run-mode `smoke` selected; writing legacy smoke artifacts to "
            f"`outputs/reports/{LEGACY_ARTIFACT_ROOT}/` and models to `outputs/{LEGACY_MODEL_ROOT}/`."
        )
    if run_mode == "full":
        feature_counts = [int(item) for item in args.feature_counts.split(",") if item.strip()]
        seq_lens = [int(item) for item in args.seq_lens.split(",") if item.strip()]
        report_root = run_full_benchmark(
            root=root,
            feature_counts=feature_counts,
            seq_lens=seq_lens,
            epochs=args.epochs,
            patience=args.patience,
            buffer_windows=args.buffer_windows,
            window_seconds=args.window_seconds,
            report_root_name=CANONICAL_ARTIFACT_ROOT,
            model_root_name=CANONICAL_MODEL_ROOT,
        )
        print(f"Full evaluation artifacts written to {report_root}")
        return

    artifact_root = root / "outputs" / "reports" / LEGACY_ARTIFACT_ROOT
    model_root = root / "outputs" / LEGACY_MODEL_ROOT
    artifact_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    if args.smoke_train:
        _run_smoke_training(root, args.max_features, args.seq_len, args.epochs)

    bundle = load_window_dataset_bundle(root)
    model_summaries: list[dict[str, object]] = []
    inference_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    figure_manifest: list[dict[str, object]] = []
    table_manifest: list[dict[str, object]] = []
    latency_frames: list[pd.DataFrame] = []

    for model_name in MODEL_NAMES:
        model_dir = model_root / model_name
        results_path = model_dir / "results.json"
        predictions_path = model_dir / "predictions.parquet"
        if not results_path.exists() or not predictions_path.exists():
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        predictions = pd.read_parquet(predictions_path)
        model_artifact_dir = artifact_root / model_name
        model_artifact_dir.mkdir(parents=True, exist_ok=True)

        figure_manifest.extend(_generate_model_plots(model_name, results, predictions, model_artifact_dir))
        latency_frames.append(detection_latency_table(predictions, bundle.attack_labels, model_name))

        metrics = results.get("metrics", {})
        model_info = results.get("model_info", {})
        model_summaries.append(
            {
                "model_name": model_name,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "average_precision": metrics.get("average_precision"),
                "threshold": metrics.get("threshold"),
            }
        )
        inference_rows.append(
            {
                "model_name": model_name,
                "params": int(model_info.get("parameter_count", 0)),
                "training_time_seconds": float(model_info.get("training_time_seconds", 0.0)),
                "inference_time_seconds": float(model_info.get("inference_time_seconds", 0.0)),
                "memory_estimate_mb": float(int(model_info.get("parameter_count", 0)) * 4 / (1024.0 ** 2)),
            }
        )
        ablation_rows.append(
            {
                "model_name": model_name,
                "window_size": 300,
                "feature_count": len(model_info.get("feature_columns", [])),
                "threshold_type": model_info.get("selected_threshold", model_info.get("threshold", "fixed")),
                "training_mode": "legacy_smoke" if args.smoke_train else "manual",
            }
        )

    summary_df = pd.DataFrame(model_summaries).sort_values("f1", ascending=False)
    inference_df = pd.DataFrame(inference_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    latency_df = pd.concat(latency_frames, ignore_index=True) if latency_frames else pd.DataFrame()

    summary_csv = artifact_root / "model_summary_table.csv"
    summary_md = artifact_root / "model_summary_table.md"
    inference_csv = artifact_root / "inference_cost_table.csv"
    ablation_csv = artifact_root / "ablation_table.csv"
    strengths_md = artifact_root / "strengths_weakness_table.md"
    latency_csv = artifact_root / "detection_latency_analysis.csv"

    summary_df.to_csv(summary_csv, index=False)
    summary_md.write_text(_to_markdown(summary_df), encoding="utf-8")
    inference_df.to_csv(inference_csv, index=False)
    ablation_df.to_csv(ablation_csv, index=False)
    latency_df.to_csv(latency_csv, index=False)

    comparison_plot = artifact_root / "model_comparison_plot.png"
    figure_manifest.append({"path": str(_plot_model_comparison(summary_df, comparison_plot)), "description": "Model precision/recall/F1/AUC comparison."})
    latency_plot = artifact_root / "detection_latency_analysis.png"
    figure_manifest.append({"path": str(_plot_detection_latency(latency_df, latency_plot)), "description": "Detection latency comparison across models and scenarios."})
    corr_plot = artifact_root / "correlation_heatmap.png"
    figure_manifest.append({"path": str(_plot_feature_correlation(bundle.attacked_windows, corr_plot)), "description": "Correlation heatmap over attacked merged-window features."})

    strengths_md.write_text(_strengths_weakness_markdown(), encoding="utf-8")
    table_manifest.extend(
        [
            {"path": str(summary_csv), "description": "Model summary table in CSV."},
            {"path": str(summary_md), "description": "Model summary table in Markdown."},
            {"path": str(inference_csv), "description": "Inference cost table."},
            {"path": str(ablation_csv), "description": "Ablation configuration table."},
            {"path": str(latency_csv), "description": "Detection latency analysis table."},
            {"path": str(strengths_md), "description": "Strengths and weaknesses discussion table."},
        ]
    )

    (artifact_root / "paper_figures_manifest.json").write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")
    (artifact_root / "paper_tables_manifest.json").write_text(json.dumps(table_manifest, indent=2), encoding="utf-8")
    print(f"Legacy smoke artifacts written to {artifact_root}")


def _run_smoke_training(root: Path, max_features: int, seq_len: int, epochs: int) -> None:
    train_threshold_model(project_root=root, max_features=max_features, threshold_name="p99", run_mode="legacy-smoke")
    train_isolation_forest(project_root=root, max_features=max_features, run_mode="legacy-smoke")
    train_autoencoder_model(project_root=root, max_features=max_features, epochs=epochs, run_mode="legacy-smoke")
    train_sequence_classifier_model("gru", lambda input_dim: GRUClassifier(input_dim=input_dim, hidden_dim=64), root, max_features=max_features, seq_len=seq_len, epochs=epochs)
    train_sequence_classifier_model("lstm", lambda input_dim: LSTMClassifier(input_dim=input_dim, hidden_dim=64), root, max_features=max_features, seq_len=seq_len, epochs=epochs)
    train_sequence_classifier_model("transformer", lambda input_dim: TinyTransformerClassifier(input_dim=input_dim, d_model=64, nhead=4, num_layers=2), root, max_features=max_features, seq_len=seq_len, epochs=epochs)
    train_llm_style_baseline(project_root=root, max_features=min(max_features, 48), seq_len=seq_len, epochs=epochs, run_mode="legacy-smoke")


def _generate_model_plots(model_name: str, results: dict[str, object], predictions: pd.DataFrame, output_dir: Path) -> list[dict[str, object]]:
    metrics = results.get("metrics", {})
    curves = results.get("curves", {})
    history = results.get("history", {})
    figure_manifest: list[dict[str, object]] = []
    display_name = display_model_name(model_name)

    roc_path = output_dir / "roc_curve.png"
    pr_path = output_dir / "precision_recall_curve.png"
    cm_path = output_dir / "confusion_matrix.png"
    dist_path = output_dir / "anomaly_score_distribution.png"
    loss_path = output_dir / "training_validation_loss.png"
    threshold_path = output_dir / "threshold_visualization.png"

    figure_manifest.append({"path": str(_plot_curve(curves.get("roc_fpr", []), curves.get("roc_tpr", []), roc_path, f"{display_name} ROC", "FPR", "TPR")), "description": f"ROC curve for {display_name}."})
    figure_manifest.append({"path": str(_plot_curve(curves.get("pr_recall", []), curves.get("pr_precision", []), pr_path, f"{display_name} Precision-Recall", "Recall", "Precision")), "description": f"Precision-recall curve for {display_name}."})
    figure_manifest.append({"path": str(_plot_confusion_matrix(metrics.get("confusion_matrix", [[0, 0], [0, 0]]), cm_path, display_name)), "description": f"Confusion matrix for {display_name}."})
    figure_manifest.append({"path": str(_plot_score_distribution(predictions, dist_path, display_name)), "description": f"Anomaly score distribution for {display_name}."})
    figure_manifest.append({"path": str(_plot_loss_curves(history, loss_path, display_name)), "description": f"Training and validation loss for {display_name}."})
    figure_manifest.append({"path": str(_plot_threshold(predictions, float(metrics.get("threshold", 0.5)), threshold_path, display_name)), "description": f"Threshold visualization for {display_name}."})
    return figure_manifest


def _plot_curve(x: list[float], y: list[float], path: Path, title: str, xlabel: str, ylabel: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    if x and y:
        ax.plot(x, y)
    else:
        ax.text(0.5, 0.5, "Curve unavailable", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_confusion_matrix(matrix: list[list[int]], path: Path, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=["benign", "attack"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_score_distribution(predictions: pd.DataFrame, path: Path, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    if predictions.empty:
        ax.text(0.5, 0.5, "No predictions available", ha="center", va="center")
    else:
        benign = predictions[predictions["attack_present"] == 0]["score"]
        attack = predictions[predictions["attack_present"] == 1]["score"]
        if not benign.empty:
            ax.hist(benign, bins=20, alpha=0.6, label="benign")
        if not attack.empty:
            ax.hist(attack, bins=20, alpha=0.6, label="attack")
        ax.legend()
    ax.set_title(f"{model_name} anomaly score distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_loss_curves(history: dict[str, list[float]], path: Path, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if train_loss or val_loss:
        if train_loss:
            ax.plot(train_loss, label="train")
        if val_loss:
            ax.plot(val_loss, label="val")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Not applicable for non-iterative model", ha="center", va="center")
    ax.set_title(f"{model_name} training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_threshold(predictions: pd.DataFrame, threshold: float, path: Path, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    if predictions.empty:
        ax.text(0.5, 0.5, "No predictions available", ha="center", va="center")
    else:
        ordered = predictions.sort_values("window_start_utc").reset_index(drop=True)
        ax.plot(ordered["score"], label="score")
        ax.axhline(threshold, color="red", linestyle="--", label="threshold")
        ax.legend()
    ax.set_title(f"{model_name} threshold visualization")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_model_comparison(summary_df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    if summary_df.empty:
        ax.text(0.5, 0.5, "No model results available", ha="center", va="center")
    else:
        metrics = ["precision", "recall", "f1", "roc_auc"]
        x = np.arange(len(summary_df))
        width = 0.18
        for idx, metric in enumerate(metrics):
            ax.bar(x + idx * width, summary_df[metric].fillna(0.0), width=width, label=metric)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(summary_df["model_name"].map(display_model_name), rotation=30)
        ax.legend()
    ax.set_title("Model Comparison Plot")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_detection_latency(latency_df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    if latency_df.empty:
        ax.text(0.5, 0.5, "No latency data available", ha="center", va="center")
    else:
        summary = latency_df.groupby("model_name", observed=False)["latency_seconds"].mean().reset_index()
        ax.bar(summary["model_name"].map(display_model_name), summary["latency_seconds"].fillna(0.0))
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Detection Latency Analysis")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_feature_correlation(attacked_windows: pd.DataFrame, path: Path) -> Path:
    numeric = attacked_windows.select_dtypes(include=[np.number])
    selected = numeric.columns[:20]
    corr = numeric[selected].corr() if len(selected) else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 6))
    if corr.empty:
        ax.text(0.5, 0.5, "No correlation data available", ha="center", va="center")
    else:
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(selected)))
        ax.set_xticklabels(selected, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(selected)))
        ax.set_yticklabels(selected, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Window Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _strengths_weakness_markdown() -> str:
    rows = [
        "# Strengths and Weaknesses",
        "",
        "| model | strengths | weaknesses |",
        "|---|---|---|",
        "| threshold_baseline | Fast, transparent, normal-only calibration. | Sensitive to feature scaling and threshold choice. |",
        "| isolation_forest | Handles nonlinear tabular anomalies with little tuning. | Score stability can vary across random seeds and contamination assumptions. |",
        "| autoencoder | Learns compact normal reconstruction patterns. | Reconstruction error can blur subtle but high-impact attacks. |",
        "| gru | Captures short temporal dynamics in ordered window sequences. | May underperform on long-range dependencies. |",
        "| lstm | Better memory than simple recurrent baselines. | Higher training cost than GRU for similar window-level tasks. |",
        "| transformer | Strong at flexible temporal context aggregation. | Heavier than recurrent baselines and data-hungry. |",
        "| tokenized_sequence_baseline | Tokenized sequence approximation offers interpretable discrete pattern modeling. | It is only an approximation and not a true foundation LLM. |",
    ]
    return "\n".join(rows) + "\n"


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| model_name |\n|---|\n"
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    divider = "|---" * len(df.columns) + "|"
    rows = [header, divider]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
