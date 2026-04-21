"""Compute DERGuardian's repo-specific Alert Overlap Index metric.

AOI is defined here as a window-level Jaccard overlap between predicted alert
windows and ground-truth attack windows:

    AOI = true_positive_windows / (true_positive_windows + false_positive_windows + false_negative_windows)

The metric is intentionally reported as experimental and repo-specific. It is
computed only from prediction artifacts that contain real ``attack_present`` and
``predicted`` columns, and it does not change canonical benchmark model
selection.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_predictions(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _compute_row(path: Path, context: str, model_name: str, window_label: str, notes: str) -> dict[str, Any] | None:
    frame = _read_predictions(path)
    if frame is None or frame.empty or "attack_present" not in frame.columns or "predicted" not in frame.columns:
        return None
    y_true = pd.to_numeric(frame["attack_present"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred = pd.to_numeric(frame["predicted"], errors="coerce").fillna(0).astype(int).to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    denominator = tp + fp + fn
    aoi = float(tp / denominator) if denominator else np.nan
    return {
        "evaluation_context": context,
        "model_name": model_name,
        "window_label": window_label,
        "prediction_artifact": _repo_rel(path),
        "aoi_alert_overlap_index": aoi,
        "true_positive_windows": tp,
        "false_positive_windows": fp,
        "false_negative_windows": fn,
        "true_negative_windows": tn,
        "attack_window_count": int((y_true == 1).sum()),
        "predicted_alert_window_count": int((y_pred == 1).sum()),
        "evaluated_window_count": int(len(frame)),
        "alert_coverage": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "alert_precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
        "notes": notes,
    }


def _collect_prediction_artifacts(root: Path) -> list[tuple[Path, str, str, str, str]]:
    items: list[tuple[Path, str, str, str, str]] = []
    for window_dir in sorted((root / "outputs" / "window_size_study").glob("*s")):
        if not window_dir.is_dir():
            continue
        window_label = window_dir.name
        for pred_path in sorted((window_dir / "ready_packages").glob("*/predictions.parquet")):
            model_name = pred_path.parent.name
            if model_name == "llm_baseline":
                continue
            items.append(
                (
                    pred_path,
                    "canonical_benchmark_prediction_artifact",
                    model_name,
                    window_label,
                    "Canonical benchmark prediction artifact; AOI is an additional repo-specific metric only.",
                )
            )

    extension_root = root / "outputs" / "phase1_lstm_autoencoder_extension" / "predictions"
    for pred_path in sorted(extension_root.glob("lstm_autoencoder_*_canonical_test.parquet")):
        window_label = pred_path.name.split("_")[2]
        items.append(
            (
                pred_path,
                "lstm_autoencoder_extension_prediction_artifact",
                "lstm_autoencoder",
                window_label,
                "New LSTM autoencoder extension prediction artifact.",
            )
        )

    ttm_path = root / "outputs" / "phase1_ttm_extension" / "ttm_extension_predictions.parquet"
    if ttm_path.exists():
        items.append(
            (
                ttm_path,
                "ttm_extension_prediction_artifact",
                "ttm_extension",
                "60s",
                "TTM extension prediction artifact; extension-only comparison.",
            )
        )

    zero_day_root = root / "outputs" / "window_size_study" / "detector_window_zero_day_audit" / "zero_day_full_matrix" / "evaluations"
    for pred_path in sorted(zero_day_root.glob("*/*/*/predictions.parquet")):
        try:
            window_label = pred_path.parent.parent.name
            model_name = pred_path.parent.name
        except IndexError:
            continue
        items.append(
            (
                pred_path,
                "heldout_synthetic_zero_day_like_prediction_artifact",
                model_name,
                window_label,
                "Heldout synthetic zero-day-like prediction artifact; not real-world zero-day evidence.",
            )
        )
    return items


def _write_definition(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# AOI Metric Definition",
                "",
                "**Status:** implemented as an experimental, repo-specific metric.",
                "",
                "In this repository, AOI means **Alert Overlap Index**. It is defined at the window level as:",
                "",
                "`AOI = TP_windows / (TP_windows + FP_windows + FN_windows)`",
                "",
                "This is the Jaccard overlap between the set of windows predicted as alerts and the set of windows labeled as attacks.",
                "It is computable from existing prediction artifacts that contain `attack_present` and `predicted` columns.",
                "",
                "Important constraints:",
                "",
                "- AOI is not used to replace precision, recall, F1, ROC-AUC, or average precision.",
                "- AOI is not claimed as a standard DER cybersecurity metric here.",
                "- AOI does not alter the frozen canonical benchmark winner: transformer @ 60s.",
                "- AOI rows are reported only where real prediction artifacts exist.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_report(results: pd.DataFrame, path: Path, csv_path: Path) -> None:
    summary = (
        results.groupby(["evaluation_context", "model_name"], dropna=False)["aoi_alert_overlap_index"]
        .mean()
        .reset_index()
        .sort_values("aoi_alert_overlap_index", ascending=False)
    )
    lines = [
        "# AOI Evaluation Report",
        "",
        "AOI was computed only from prediction files with actual `attack_present` and `predicted` columns.",
        "The metric is experimental and repo-specific. It does not upgrade the project to an AOI-standard benchmark claim.",
        "",
        "## Mean AOI By Context And Model",
        "",
        summary.head(30).to_markdown(index=False) if not summary.empty else "No AOI rows were computed.",
        "",
        "## Scientific Use",
        "",
        "AOI is useful here as an additional overlap view of alert timing and over-alerting. It should be described as an experimental Alert Overlap Index, not as a standard metric unless future work establishes that external definition.",
        f"Source CSV: `{_repo_rel(csv_path)}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(results: pd.DataFrame, path: Path) -> None:
    plot_frame = (
        results.loc[results["evaluation_context"].isin(["canonical_benchmark_prediction_artifact", "lstm_autoencoder_extension_prediction_artifact", "ttm_extension_prediction_artifact"])]
        .groupby(["model_name"], dropna=False)["aoi_alert_overlap_index"]
        .mean()
        .reset_index()
        .sort_values("aoi_alert_overlap_index", ascending=False)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.8))
    if plot_frame.empty:
        plt.text(0.5, 0.5, "No AOI rows available", ha="center", va="center")
        plt.axis("off")
    else:
        plt.bar(plot_frame["model_name"], plot_frame["aoi_alert_overlap_index"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Mean AOI")
        plt.ylim(0, max(0.05, min(1.0, float(plot_frame["aoi_alert_overlap_index"].max()) * 1.2)))
        plt.title("Experimental Alert Overlap Index")
        plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    """Compute and write AOI artifacts."""

    artifact_dir = ROOT / "artifacts" / "extensions"
    docs_dir = ROOT / "docs" / "reports"
    figure_dir = ROOT / "docs" / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pred_path, context, model_name, window_label, notes in _collect_prediction_artifacts(ROOT):
        row = _compute_row(pred_path, context, model_name, window_label, notes)
        if row is not None:
            rows.append(row)
    results = pd.DataFrame(rows)
    csv_path = artifact_dir / "aoi_results.csv"
    results.to_csv(csv_path, index=False)
    _write_definition(docs_dir / "AOI_METRIC_DEFINITION.md")
    _write_report(results, docs_dir / "aoi_eval_report.md", csv_path)
    _write_figure(results, figure_dir / "aoi_comparison_figure.png")
    print({"aoi_rows": int(len(results)), "csv": _repo_rel(csv_path)})


if __name__ == "__main__":
    main()
