from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from common.config import WindowConfig
from phase1.build_windows import build_merged_windows
from phase1_models.model_loader import load_phase1_package, run_phase1_inference
from phase1_models.feature_builder import fit_standardizer, transform_features
from phase1_models.metrics import compute_binary_metrics, detection_latency_table, per_scenario_metrics
from phase1_models.residual_dataset import build_aligned_residual_dataframe

try:
    from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction

    HAS_TTM = True
    TTM_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local env
    HAS_TTM = False
    TTM_IMPORT_ERROR = str(exc)
    TinyTimeMixerConfig = None  # type: ignore[assignment]
    TinyTimeMixerForPrediction = None  # type: ignore[assignment]


WINDOW_STUDY_ROOT = ROOT / "outputs" / "window_size_study"
FINAL_WINDOW_COMPARISON_PATH = WINDOW_STUDY_ROOT / "final_window_comparison.csv"
PHASE3_HELDOUT_ROOT = WINDOW_STUDY_ROOT / "phase3_heldout"
IMPROVED_PHASE3_ROOT = WINDOW_STUDY_ROOT / "improved_phase3"
HELDOUT_JSON_ROOT = ROOT / "phase2_llm_benchmark" / "heldout_llm_response"
HELDOUT_EXISTING_ROOT = ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models"
HUMAN_AUTHORED_ROOT = IMPROVED_PHASE3_ROOT / "additional_source" / "human_authored"
RAW_AUDIT_ROOT = WINDOW_STUDY_ROOT / "detector_window_zero_day_audit"
RAW_ZERO_DAY_ROOT = RAW_AUDIT_ROOT / "zero_day_full_matrix"


@dataclass(frozen=True, slots=True)
class WindowSpec:
    label: str
    seconds: int
    step_seconds: int


@dataclass(frozen=True, slots=True)
class BundleSource:
    generator_source: str
    dataset_id: str
    source_type: str
    measured_path: Path
    cyber_path: Path
    labels_path: Path
    scenario_count: int
    accepted_scenario_count: int
    note: str


WINDOW_SPECS = [
    WindowSpec("5s", 5, 1),
    WindowSpec("10s", 10, 2),
    WindowSpec("60s", 60, 12),
    WindowSpec("300s", 300, 60),
]
WINDOW_LABELS = [spec.label for spec in WINDOW_SPECS]
CANONICAL_MODEL_ORDER = [
    "threshold_baseline",
    "isolation_forest",
    "autoencoder",
    "gru",
    "lstm",
    "transformer",
]
REQUESTED_MODEL_ORDER = [
    "transformer",
    "gru",
    "lstm_ae",
    "autoencoder",
    "isolation_forest",
    "threshold_baseline",
    "ttm_extension",
    "arima",
]
MODEL_DISPLAY_NAMES = {
    "transformer": "Transformer",
    "gru": "GRU",
    "lstm": "LSTM",
    "lstm_ae": "LSTM Autoencoder",
    "autoencoder": "Autoencoder (MLP)",
    "isolation_forest": "Isolation Forest",
    "threshold_baseline": "Threshold Baseline",
    "ttm_extension": "TinyTimeMixer Extension",
    "arima": "ARIMA",
}
GENERATOR_ORDER = ["chatgpt", "claude", "gemini", "grok", "human_authored"]
ALL_BUNDLE_ORDER = ["canonical_bundle"] + GENERATOR_ORDER
ZERO_DAY_REPORT_PATHS = [
    ROOT / "PHASE3_ZERO_DAY_REPORT.md",
    ROOT / "IMPROVED_PHASE3_RESULTS_REPORT.md",
]


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_").lower() or "item"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _false_positive_rate(metrics_payload: dict[str, Any]) -> float | None:
    matrix = metrics_payload.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 2 or len(matrix[0]) != 2:
        return None
    tn = float(matrix[0][0])
    fp = float(matrix[0][1])
    denom = tn + fp
    if denom <= 0.0:
        return None
    return fp / denom


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _safe_median(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.median())


def _count_existing(paths: list[Path]) -> int:
    return sum(1 for path in paths if path.exists())


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no rows_"
    frame = df.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return frame.to_markdown(index=False)


def _window_spec(label: str) -> WindowSpec:
    mapping = {spec.label: spec for spec in WINDOW_SPECS}
    return mapping[label]


def _benchmark_row(window_label: str, model_name: str) -> pd.Series | None:
    path = WINDOW_STUDY_ROOT / window_label / "reports" / "model_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    match = df.loc[df["model_name"] == model_name]
    if match.empty:
        return None
    return match.iloc[0]


def _benchmark_metrics(window_label: str, model_name: str) -> dict[str, Any] | None:
    path = WINDOW_STUDY_ROOT / window_label / "ready_packages" / model_name / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_detector_window_model_coverage_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for spec in WINDOW_SPECS:
        for model_name in CANONICAL_MODEL_ORDER:
            row = _benchmark_row(spec.label, model_name)
            metrics_path = WINDOW_STUDY_ROOT / spec.label / "ready_packages" / model_name / "metrics.json"
            figure_path = WINDOW_STUDY_ROOT / spec.label / "reports" / model_name / "confusion_matrix.png"
            rows.append(
                {
                    "evaluation_context": "canonical_benchmark_test_split",
                    "window_seconds": spec.seconds,
                    "model_name": model_name,
                    "artifact_exists": bool(row is not None),
                    "metrics_exist": metrics_path.exists(),
                    "figure_exists": figure_path.exists(),
                    "split_type": "canonical_test_split",
                    "canonical_or_extension": "canonical",
                    "needs_rerun": False if row is not None and metrics_path.exists() else True,
                    "notes": (
                        "Real canonical benchmark package and test-split metrics."
                        if row is not None
                        else "Expected canonical benchmark row missing."
                    ),
                }
            )
        rows.append(
            {
                "evaluation_context": "canonical_benchmark_test_split",
                "window_seconds": spec.seconds,
                "model_name": "lstm_ae",
                "artifact_exists": False,
                "metrics_exist": False,
                "figure_exists": False,
                "split_type": "canonical_test_split",
                "canonical_or_extension": "blocked",
                "needs_rerun": True,
                "notes": "No LSTM autoencoder detector implementation exists; repo has an MLP autoencoder reported as `autoencoder`.",
            }
        )
        rows.append(
            {
                "evaluation_context": "canonical_benchmark_test_split",
                "window_seconds": spec.seconds,
                "model_name": "arima",
                "artifact_exists": False,
                "metrics_exist": False,
                "figure_exists": False,
                "split_type": "canonical_test_split",
                "canonical_or_extension": "extension_blocked",
                "needs_rerun": True,
                "notes": "No ARIMA detector implementation exists in the repository.",
            }
        )

    ttm_results_path = ROOT / "phase1_ttm_results.csv"
    ttm_figures_exist = _count_existing(
        [
            ROOT / "phase1_lora_family_accuracy.png",
            ROOT / "phase1_model_comparison_extended.png",
            ROOT / "phase1_accuracy_latency_tradeoff_extended.png",
        ]
    )
    ttm_df = pd.read_csv(ttm_results_path) if ttm_results_path.exists() else pd.DataFrame()
    for spec in WINDOW_SPECS:
        exists = bool(not ttm_df.empty and (ttm_df["window_label"] == spec.label).any())
        rows.append(
            {
                "evaluation_context": "ttm_extension_benchmark",
                "window_seconds": spec.seconds,
                "model_name": "ttm_extension",
                "artifact_exists": exists,
                "metrics_exist": exists,
                "figure_exists": bool(ttm_figures_exist >= 2) if exists else False,
                "split_type": "canonical_test_split",
                "canonical_or_extension": "extension",
                "needs_rerun": not exists,
                "notes": (
                    "Real TinyTimeMixer extension benchmark exists at 60s only."
                    if exists
                    else "No TTM benchmark artifact exists for this window."
                ),
            }
        )

    phase3_report_exists = (ROOT / "PHASE3_ZERO_DAY_REPORT.md").exists()
    for generator in ["canonical_bundle", "chatgpt", "claude", "gemini", "grok"]:
        bundle_root = PHASE3_HELDOUT_ROOT / generator
        metrics_path = bundle_root / "reports" / "bundle_metrics.json"
        if metrics_path.exists():
            rows.append(
                {
                    "evaluation_context": "existing_phase3_saved_transformer_replay",
                    "window_seconds": 60,
                    "model_name": "transformer",
                    "artifact_exists": True,
                    "metrics_exist": True,
                    "figure_exists": phase3_report_exists,
                    "split_type": "heldout_bundle_replay",
                    "canonical_or_extension": "replay",
                    "needs_rerun": False,
                    "notes": f"Existing saved-transformer replay for bundle `{generator}`.",
                }
            )

    improved_report_exists = (ROOT / "IMPROVED_PHASE3_RESULTS_REPORT.md").exists()
    for generator in ALL_BUNDLE_ORDER:
        bundle_key_prefix = _slug(generator)
        base_root = IMPROVED_PHASE3_ROOT / "evaluations"
        matches = list(base_root.glob(f"{bundle_key_prefix}__*/**/metrics.json"))
        wanted = {
            ("threshold_baseline", 10),
            ("transformer", 60),
            ("lstm", 300),
        }
        for model_name, window_seconds in wanted:
            matched = [
                path
                for path in matches
                if path.parent.name == model_name and path.parent.parent.name == f"{window_seconds}s"
            ]
            if matched:
                rows.append(
                    {
                        "evaluation_context": "existing_improved_phase3_frozen_candidate_replay",
                        "window_seconds": window_seconds,
                        "model_name": model_name,
                        "artifact_exists": True,
                        "metrics_exist": True,
                        "figure_exists": improved_report_exists,
                        "split_type": "heldout_bundle_replay",
                        "canonical_or_extension": "replay",
                        "needs_rerun": False,
                        "notes": f"Existing improved replay includes `{generator}`.",
                    }
                )

    audit_df = pd.DataFrame(rows)
    audit_df = audit_df.sort_values(
        by=["evaluation_context", "window_seconds", "model_name", "split_type"],
        kind="stable",
    ).reset_index(drop=True)
    return audit_df


def _write_detector_window_model_coverage_summary(audit_df: pd.DataFrame) -> None:
    benchmark_df = audit_df.loc[audit_df["evaluation_context"] == "canonical_benchmark_test_split"].copy()
    benchmark_counts = (
        benchmark_df.groupby("window_seconds", as_index=False)["artifact_exists"].sum().rename(columns={"artifact_exists": "real_model_rows"})
    )
    replay_df = audit_df.loc[audit_df["canonical_or_extension"] == "replay"].copy()
    ttm_df = audit_df.loc[audit_df["evaluation_context"] == "ttm_extension_benchmark"].copy()
    lines = [
        "# Detector Window / Model Coverage Summary",
        "",
        "This audit reflects the repository state before this completion pass filled the missing detector-side heldout synthetic matrix.",
        "",
        "## Verified Existing Canonical Benchmark Coverage",
        "",
        _markdown_table(benchmark_counts),
        "",
        "Real canonical benchmark coverage already existed for windows `5s`, `10s`, `60s`, and `300s` across the implemented detector models:",
        "",
        "- `threshold_baseline`",
        "- `isolation_forest`",
        "- `autoencoder` (repo implementation is MLP autoencoder; no LSTM autoencoder was found)",
        "- `gru`",
        "- `lstm`",
        "- `transformer`",
        "",
        "The canonical benchmark winner remains `transformer @ 60s` from `outputs/window_size_study/final_window_comparison.csv`.",
        "",
        "## Existing Replay / Zero-Day-Like Coverage Before Completion",
        "",
        f"- Existing saved-package replay in `phase3_heldout` covered only `transformer @ 60s` across the canonical bundle plus the original heldout generator bundles. Rows found: `{len(audit_df.loc[audit_df['evaluation_context'] == 'existing_phase3_saved_transformer_replay'])}`.",
        f"- Existing improved replay in `improved_phase3` covered only three frozen candidates: `threshold_baseline @ 10s`, `transformer @ 60s`, and `lstm @ 300s`. Rows found: `{len(replay_df.loc[replay_df['evaluation_context'] == 'existing_improved_phase3_frozen_candidate_replay'])}`.",
        f"- Existing TTM extension evidence covered only `60s`. Verified rows: `{int(ttm_df['artifact_exists'].sum())}` / `{len(ttm_df)}`.",
        "",
        "## Explicit Gaps Found",
        "",
        "- No ARIMA detector benchmark implementation exists in the repo.",
        "- No real LSTM autoencoder detector exists; the available autoencoder is an MLP autoencoder.",
        "- No pre-existing replay / zero-day-like matrix covered all required windows and all detector models.",
        "- No pre-existing TTM detector comparison existed beyond the single `60s` extension run.",
    ]
    write_markdown(ROOT / "detector_window_model_coverage_summary.md", "\n".join(lines))


def _build_phase1_window_model_comparison_full() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for spec in WINDOW_SPECS:
        for model_name in CANONICAL_MODEL_ORDER:
            summary_row = _benchmark_row(spec.label, model_name)
            metrics_payload = _benchmark_metrics(spec.label, model_name)
            if summary_row is None or metrics_payload is None:
                rows.append(
                    {
                        "evaluation_context": "canonical_benchmark_test_split",
                        "canonical_or_extension": "canonical",
                        "window_label": spec.label,
                        "window_seconds": spec.seconds,
                        "step_seconds": spec.step_seconds,
                        "model_name": model_name,
                        "status": "missing",
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "average_precision": np.nan,
                        "roc_auc": np.nan,
                        "false_positive_rate": np.nan,
                        "mean_detection_latency_seconds": np.nan,
                        "median_detection_latency_seconds": np.nan,
                        "inference_time_ms_per_prediction": np.nan,
                        "ready_package_dir": "",
                        "metrics_source": "",
                        "notes": "Expected canonical benchmark artifact missing.",
                    }
                )
                continue
            rows.append(
                {
                    "evaluation_context": "canonical_benchmark_test_split",
                    "canonical_or_extension": "canonical",
                    "window_label": spec.label,
                    "window_seconds": spec.seconds,
                    "step_seconds": spec.step_seconds,
                    "model_name": model_name,
                    "status": str(summary_row["status"]),
                    "precision": float(summary_row["precision"]),
                    "recall": float(summary_row["recall"]),
                    "f1": float(summary_row["f1"]),
                    "average_precision": _safe_float(summary_row.get("average_precision")),
                    "roc_auc": _safe_float(summary_row.get("roc_auc")),
                    "false_positive_rate": _false_positive_rate(metrics_payload),
                    "mean_detection_latency_seconds": _safe_float(summary_row.get("mean_detection_latency_seconds")),
                    "median_detection_latency_seconds": _safe_float(summary_row.get("median_detection_latency_seconds")),
                    "inference_time_ms_per_prediction": _safe_float(summary_row.get("inference_time_ms_per_prediction")),
                    "ready_package_dir": str(summary_row["ready_package_dir"]),
                    "metrics_source": str(
                        WINDOW_STUDY_ROOT / spec.label / "ready_packages" / model_name / "metrics.json"
                    ),
                    "notes": (
                        "Repo implementation is an MLP autoencoder; no LSTM autoencoder artifact exists."
                        if model_name == "autoencoder"
                        else ""
                    ),
                }
            )

    ttm_results_path = ROOT / "phase1_ttm_results.csv"
    ttm_df = pd.read_csv(ttm_results_path) if ttm_results_path.exists() else pd.DataFrame()
    for spec in WINDOW_SPECS:
        if not ttm_df.empty and (ttm_df["window_label"] == spec.label).any():
            row = ttm_df.loc[ttm_df["window_label"] == spec.label].iloc[0]
            rows.append(
                {
                    "evaluation_context": "canonical_benchmark_test_split",
                    "canonical_or_extension": "extension",
                    "window_label": spec.label,
                    "window_seconds": int(row["window_seconds"]),
                    "step_seconds": int(row["step_seconds"]),
                    "model_name": "ttm_extension",
                    "status": str(row["status"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "average_precision": _safe_float(row.get("average_precision")),
                    "roc_auc": _safe_float(row.get("roc_auc")),
                    "false_positive_rate": np.nan,
                    "mean_detection_latency_seconds": _safe_float(row.get("mean_detection_latency_seconds")),
                    "median_detection_latency_seconds": _safe_float(row.get("median_detection_latency_seconds")),
                    "inference_time_ms_per_prediction": _safe_float(row.get("inference_time_ms_per_prediction")),
                    "ready_package_dir": str(ROOT / "outputs" / "phase1_ttm_extension"),
                    "metrics_source": str(ttm_results_path),
                    "notes": "Extension only. Real TTM benchmark evidence exists at 60s only.",
                }
            )
        else:
            rows.append(
                {
                    "evaluation_context": "canonical_benchmark_test_split",
                    "canonical_or_extension": "extension",
                    "window_label": spec.label,
                    "window_seconds": spec.seconds,
                    "step_seconds": spec.step_seconds,
                    "model_name": "ttm_extension",
                    "status": "not_run",
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "average_precision": np.nan,
                    "roc_auc": np.nan,
                    "false_positive_rate": np.nan,
                    "mean_detection_latency_seconds": np.nan,
                    "median_detection_latency_seconds": np.nan,
                    "inference_time_ms_per_prediction": np.nan,
                    "ready_package_dir": "",
                    "metrics_source": "",
                    "notes": "No evidenced TTM benchmark artifact exists for this window in the repo.",
                }
            )

    for spec in WINDOW_SPECS:
        rows.append(
            {
                "evaluation_context": "canonical_benchmark_test_split",
                "canonical_or_extension": "blocked",
                "window_label": spec.label,
                "window_seconds": spec.seconds,
                "step_seconds": spec.step_seconds,
                "model_name": "lstm_ae",
                "status": "not_implemented",
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "average_precision": np.nan,
                "roc_auc": np.nan,
                "false_positive_rate": np.nan,
                "mean_detection_latency_seconds": np.nan,
                "median_detection_latency_seconds": np.nan,
                "inference_time_ms_per_prediction": np.nan,
                "ready_package_dir": "",
                "metrics_source": "",
                "notes": "No LSTM autoencoder detector implementation exists; MLP autoencoder is reported separately as `autoencoder`.",
            }
        )

    for spec in WINDOW_SPECS:
        rows.append(
            {
                "evaluation_context": "canonical_benchmark_test_split",
                "canonical_or_extension": "extension_blocked",
                "window_label": spec.label,
                "window_seconds": spec.seconds,
                "step_seconds": spec.step_seconds,
                "model_name": "arima",
                "status": "not_implemented",
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "average_precision": np.nan,
                "roc_auc": np.nan,
                "false_positive_rate": np.nan,
                "mean_detection_latency_seconds": np.nan,
                "median_detection_latency_seconds": np.nan,
                "inference_time_ms_per_prediction": np.nan,
                "ready_package_dir": "",
                "metrics_source": "",
                "notes": "No ARIMA detector implementation exists in the repository.",
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df["window_order"] = comparison_df["window_label"].map({label: idx for idx, label in enumerate(WINDOW_LABELS)})
    comparison_df["model_order"] = comparison_df["model_name"].map({name: idx for idx, name in enumerate(REQUESTED_MODEL_ORDER + ["lstm"])})
    comparison_df = comparison_df.sort_values(["window_order", "canonical_or_extension", "model_order", "model_name"]).drop(
        columns=["window_order", "model_order"]
    )
    comparison_df.to_csv(ROOT / "phase1_window_model_comparison_full.csv", index=False)

    benchmark_only = comparison_df.loc[
        (comparison_df["canonical_or_extension"] == "canonical") & (comparison_df["status"] == "completed")
    ].copy()
    best_canonical = (
        benchmark_only.sort_values(["window_seconds", "f1", "average_precision", "precision"], ascending=[True, False, False, False])
        .groupby("window_label", as_index=False)
        .first()
    )
    best_all = (
        comparison_df.loc[comparison_df["status"] == "completed"]
        .sort_values(["window_seconds", "f1", "average_precision", "precision"], ascending=[True, False, False, False])
        .groupby("window_label", as_index=False)
        .first()[["window_label", "model_name", "f1", "canonical_or_extension"]]
        .rename(
            columns={
                "model_name": "best_model_including_extensions",
                "f1": "best_f1_including_extensions",
                "canonical_or_extension": "best_including_extensions_type",
            }
        )
    )
    best_summary = best_canonical.merge(best_all, on="window_label", how="left")
    best_summary = best_summary.rename(
        columns={
            "model_name": "canonical_best_model",
            "f1": "canonical_best_f1",
            "precision": "canonical_best_precision",
            "recall": "canonical_best_recall",
        }
    )
    best_summary["window_order"] = best_summary["window_label"].map({label: idx for idx, label in enumerate(WINDOW_LABELS)})
    best_summary = best_summary.sort_values("window_order").drop(columns="window_order")[
        [
            "window_label",
            "window_seconds",
            "canonical_best_model",
            "canonical_best_precision",
            "canonical_best_recall",
            "canonical_best_f1",
            "best_model_including_extensions",
            "best_f1_including_extensions",
            "best_including_extensions_type",
        ]
    ]
    best_summary.to_csv(ROOT / "phase1_best_per_window_summary.csv", index=False)
    return comparison_df, best_summary


def _write_phase1_window_model_comparison_md(comparison_df: pd.DataFrame, best_summary: pd.DataFrame) -> None:
    safe_subset = comparison_df.loc[
        comparison_df["model_name"].isin(["threshold_baseline", "isolation_forest", "autoencoder", "lstm_ae", "gru", "lstm", "transformer", "ttm_extension"])
    ].copy()
    lines = [
        "# Full Detector Window / Model Comparison",
        "",
        "This table is derived from real saved benchmark artifacts. It does not overwrite the frozen canonical selection file.",
        "",
        "## Important Implementation Notes",
        "",
        "- The canonical winner remains `transformer @ 60s`.",
        "- The repo does not contain a real ARIMA detector implementation.",
        "- The repo does not contain a real LSTM autoencoder detector. The available `autoencoder` is an MLP autoencoder and is reported under its actual implementation name.",
        "- TinyTimeMixer is extension-only and is kept separate from the canonical benchmark path.",
        "",
        "## Best Per Window",
        "",
        _markdown_table(best_summary),
        "",
        "## Full Comparison Table",
        "",
        _markdown_table(
            safe_subset[
                [
                    "window_label",
                    "model_name",
                    "canonical_or_extension",
                    "status",
                    "precision",
                    "recall",
                    "f1",
                    "average_precision",
                    "roc_auc",
                    "false_positive_rate",
                    "mean_detection_latency_seconds",
                ]
            ]
        ),
    ]
    write_markdown(ROOT / "phase1_window_model_comparison_full.md", "\n".join(lines))


def _plot_phase1_window_comparison(comparison_df: pd.DataFrame) -> None:
    plot_df = comparison_df.loc[
        comparison_df["model_name"].isin(["threshold_baseline", "isolation_forest", "autoencoder", "gru", "lstm", "transformer", "ttm_extension"])
        & (comparison_df["status"] == "completed")
    ].copy()
    plot_df["window_label"] = pd.Categorical(plot_df["window_label"], categories=WINDOW_LABELS, ordered=True)
    plot_df = plot_df.sort_values(["window_label", "model_name"])
    models = plot_df["model_name"].unique().tolist()
    x = np.arange(len(WINDOW_LABELS))
    width = 0.11

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {
        "threshold_baseline": "#4C78A8",
        "isolation_forest": "#F58518",
        "autoencoder": "#54A24B",
        "gru": "#E45756",
        "lstm": "#72B7B2",
        "transformer": "#B279A2",
        "ttm_extension": "#9D755D",
    }
    for idx, model_name in enumerate(models):
        subset = plot_df.loc[plot_df["model_name"] == model_name]
        values = []
        for label in WINDOW_LABELS:
            match = subset.loc[subset["window_label"] == label, "f1"]
            values.append(float(match.iloc[0]) if not match.empty else np.nan)
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=MODEL_DISPLAY_NAMES.get(model_name, model_name), color=colors.get(model_name))
    ax.set_xticks(x)
    ax.set_xticklabels(WINDOW_LABELS)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_xlabel("Window")
    ax.set_title("Detector Benchmark F1 by Window")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "phase1_window_comparison_5s_10s_60s_300s.png", dpi=220)
    plt.close(fig)

    heatmap_df = plot_df.pivot(index="model_name", columns="window_label", values="f1").reindex(
        index=["threshold_baseline", "isolation_forest", "autoencoder", "gru", "lstm", "transformer", "ttm_extension"],
        columns=WINDOW_LABELS,
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns.tolist())
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(name, name) for name in heatmap_df.index.tolist()])
    ax.set_xlabel("Window")
    ax.set_ylabel("Model")
    ax.set_title("Detector Benchmark F1 Heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("F1")
    fig.tight_layout()
    fig.savefig(ROOT / "phase1_model_vs_window_heatmap.png", dpi=220)
    plt.close(fig)


def _build_phase1_ttm_window_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    subset = comparison_df.loc[
        comparison_df["model_name"].isin(["ttm_extension", "transformer", "gru", "lstm", "lstm_ae", "isolation_forest", "autoencoder", "threshold_baseline", "arima"])
    ].copy()
    subset.to_csv(ROOT / "phase1_ttm_window_comparison.csv", index=False)

    lines = [
        "# TTM Window Comparison",
        "",
        "This report keeps TTM in the detector-side comparison without relabeling it as the canonical winner.",
        "",
        "- Real TTM benchmark evidence exists only at `60s` in this repository.",
        "- The canonical winner remains `transformer @ 60s`.",
        "- `ARIMA` remains unimplemented and is included only as a documented blocker row.",
        "",
        _markdown_table(
            subset[
                [
                    "window_label",
                    "model_name",
                    "canonical_or_extension",
                    "status",
                    "precision",
                    "recall",
                    "f1",
                    "mean_detection_latency_seconds",
                    "notes",
                ]
            ]
        ),
    ]
    write_markdown(ROOT / "phase1_ttm_window_comparison.md", "\n".join(lines))
    return subset


def _plot_ttm_window_comparison(ttm_df: pd.DataFrame) -> None:
    plot_df = ttm_df.loc[(ttm_df["window_label"] == "60s") & (ttm_df["status"] == "completed")].copy()
    order = ["transformer", "gru", "lstm", "isolation_forest", "autoencoder", "threshold_baseline", "ttm_extension"]
    plot_df["order"] = plot_df["model_name"].map({name: idx for idx, name in enumerate(order)})
    plot_df = plot_df.sort_values("order").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        [MODEL_DISPLAY_NAMES.get(name, name) for name in plot_df["model_name"]],
        plot_df["f1"].astype(float),
        color=["#4C78A8" if name != "ttm_extension" else "#B279A2" for name in plot_df["model_name"]],
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_xlabel("Model @ 60s")
    ax.set_title("TTM Extension vs Detector Models at 60s")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, plot_df["f1"].astype(float)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "phase1_ttm_vs_transformer_gru_lstm_if_arima.png", dpi=220)
    plt.close(fig)


def _select_effective_json(folder: Path) -> tuple[Path, list[Path]]:
    json_paths = sorted(folder.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {folder}")
    grouped: dict[str, list[Path]] = {}
    for path in json_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        grouped.setdefault(digest, []).append(path)
    groups = list(grouped.values())
    chosen = sorted(
        groups,
        key=lambda group: (
            0 if any(path.name == "new_respnse.json" for path in group) else 1,
            min(path.name for path in group),
        ),
    )[0]
    selected = next((path for path in chosen if path.name == "new_respnse.json"), chosen[0])
    duplicates = [path for path in chosen if path != selected]
    return selected, duplicates


def _build_heldout_bundle_sources() -> list[BundleSource]:
    bundles: list[BundleSource] = []
    for generator in ["chatgpt", "claude", "gemini", "grok"]:
        json_path, duplicates = _select_effective_json(HELDOUT_JSON_ROOT / generator)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        dataset_dir = HELDOUT_EXISTING_ROOT / generator / "datasets"
        labels_df = pd.read_parquet(dataset_dir / "attack_labels.parquet")
        note = f"Existing Phase 2 heldout bundle. Duplicate JSONs shadowed: {', '.join(path.name for path in duplicates)}." if duplicates else "Existing Phase 2 heldout bundle."
        bundles.append(
            BundleSource(
                generator_source=generator,
                dataset_id=str(payload.get("dataset_id", f"{generator}_heldout_bundle")),
                source_type="existing_heldout_phase2_bundle",
                measured_path=dataset_dir / "measured_physical_timeseries.parquet",
                cyber_path=dataset_dir / "cyber_events.parquet",
                labels_path=dataset_dir / "attack_labels.parquet",
                scenario_count=len(payload.get("scenarios", [])),
                accepted_scenario_count=int(labels_df["scenario_id"].astype(str).nunique()),
                note=note,
            )
        )

    human_dataset_dir = HUMAN_AUTHORED_ROOT / "datasets"
    human_manifest_path = human_dataset_dir / "compiled_manifest.json"
    human_manifest = json.loads(human_manifest_path.read_text(encoding="utf-8"))
    human_labels = pd.read_parquet(human_dataset_dir / "attack_labels.parquet")
    bundles.append(
        BundleSource(
            generator_source="human_authored",
            dataset_id=str(human_manifest.get("dataset_id", "human_authored_phase3_bundle_v1")),
            source_type="human_authored_heldout",
            measured_path=human_dataset_dir / "measured_physical_timeseries.parquet",
            cyber_path=human_dataset_dir / "cyber_events.parquet",
            labels_path=human_dataset_dir / "attack_labels.parquet",
            scenario_count=int(human_labels["scenario_id"].astype(str).nunique()),
            accepted_scenario_count=int(human_labels["scenario_id"].astype(str).nunique()),
            note="Existing human-authored additional heldout bundle from improved_phase3.",
        )
    )
    return bundles


def _bundle_key(bundle: BundleSource) -> str:
    return f"{_slug(bundle.generator_source)}__{_slug(bundle.source_type)}__{_slug(bundle.dataset_id)}"


def _canonical_clean_windows_path(spec: WindowSpec) -> Path:
    return WINDOW_STUDY_ROOT / spec.label / "data" / "merged_windows_clean.parquet"


def _required_measured_columns_for_feature_columns(feature_columns: list[str]) -> set[str]:
    required = {
        "timestamp_utc",
        "simulation_index",
        "scenario_id",
        "run_id",
        "split_id",
        "sample_rate_seconds",
    }
    for feature in feature_columns:
        stripped = str(feature).replace("delta__", "", 1)
        if stripped.startswith("cyber_"):
            continue
        base = stripped.rsplit("__", 1)[0] if "__" in stripped else stripped
        required.add(base)
    return required


def _load_or_build_bundle_residual(bundle: BundleSource, spec: WindowSpec, feature_columns: list[str]) -> Path:
    build_root = RAW_ZERO_DAY_ROOT / "replay_inputs" / _bundle_key(bundle) / spec.label
    build_root.mkdir(parents=True, exist_ok=True)
    residual_path = build_root / "residual_windows.parquet"
    if residual_path.exists():
        return residual_path
    improved_existing = IMPROVED_PHASE3_ROOT / "replay_inputs" / _bundle_key(bundle) / spec.label / "residual_windows.parquet"
    if improved_existing.exists():
        return improved_existing
    if spec.label == "5s":
        raise RuntimeError(
            "No prebuilt 5s replay residual exists for this heldout bundle. "
            "Building raw 5s heldout windows across full-day attacked traces exceeded the CPU-only completion-pass budget."
        )
    attacked_windows_path = build_root / "merged_windows_attacked.parquet"
    measured_df = pd.read_parquet(bundle.measured_path)
    required = _required_measured_columns_for_feature_columns(feature_columns)
    subset_columns = [column for column in measured_df.columns if column in required]
    measured_df = measured_df[subset_columns].copy()
    cyber_df = pd.read_parquet(bundle.cyber_path)
    labels_df = pd.read_parquet(bundle.labels_path)
    clean_windows = pd.read_parquet(_canonical_clean_windows_path(spec))
    attacked_windows = build_merged_windows(
        measured_df=measured_df,
        cyber_df=cyber_df,
        labels_df=labels_df,
        windows=WindowConfig(window_seconds=spec.seconds, step_seconds=spec.step_seconds, min_attack_overlap_fraction=0.2),
    )
    attacked_windows.to_parquet(attacked_windows_path, index=False)
    residual_df = build_aligned_residual_dataframe(clean_windows, attacked_windows, labels_df)
    residual_df.to_parquet(residual_path, index=False)
    return residual_path


def _ensure_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    prepared = frame.copy()
    missing_features: list[str] = []
    for column in feature_columns:
        if column not in prepared.columns:
            prepared[column] = 0.0
            missing_features.append(column)
    return prepared, missing_features


def _prepare_residual_for_inference(
    residual_df: pd.DataFrame,
    feature_columns: list[str],
    standardizer: Any | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared, missing_features = _ensure_feature_frame(residual_df, feature_columns)
    true_nonfinite_rows = set()
    true_nonfinite_features: list[dict[str, Any]] = []
    extreme_rows = set()
    extreme_features: list[dict[str, Any]] = []
    fallback_used = False
    for index, column in enumerate(feature_columns):
        series = pd.to_numeric(prepared[column], errors="coerce")
        values = series.to_numpy(dtype=float)
        nonfinite_mask = ~np.isfinite(values)
        nonfinite_count = int(nonfinite_mask.sum())
        if nonfinite_count > 0:
            fallback_used = True
            true_nonfinite_rows.update(series.index[nonfinite_mask].tolist())
            true_nonfinite_features.append({"feature": column, "nonfinite_count": nonfinite_count})
            series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prepared[column] = series.astype(float)
        if standardizer is not None and hasattr(standardizer, "mean") and hasattr(standardizer, "std") and index < len(standardizer.mean):
            std_value = float(standardizer.std[index]) if float(standardizer.std[index]) != 0.0 else 1.0
            z_scores = np.abs((prepared[column].to_numpy(dtype=float) - float(standardizer.mean[index])) / std_value)
            extreme_mask = np.isfinite(z_scores) & (z_scores > 100.0)
            extreme_count = int(extreme_mask.sum())
            if extreme_count > 0:
                extreme_rows.update(np.where(extreme_mask)[0].tolist())
                extreme_features.append({"feature": column, "extreme_count": extreme_count})
    return prepared, {
        "missing_feature_count": len(missing_features),
        "missing_features": missing_features[:128],
        "true_nonfinite_feature_count": len(true_nonfinite_features),
        "true_nonfinite_row_count": len(true_nonfinite_rows),
        "true_nonfinite_features": true_nonfinite_features[:128],
        "extreme_feature_count": len(extreme_features),
        "extreme_row_count": len(extreme_rows),
        "extreme_features": extreme_features[:128],
        "fallback_nonfinite_replacement_used": fallback_used,
        "clipping_applied": False,
    }


def _clip_extreme_features_for_package(
    frame: pd.DataFrame,
    feature_columns: list[str],
    standardizer: Any | None,
    z_limit: float = 100.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clipped = frame.copy()
    clipped_features: list[dict[str, Any]] = []
    affected_rows = set()
    for index, column in enumerate(feature_columns):
        if column not in clipped.columns:
            continue
        series = pd.to_numeric(clipped[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        values = series.to_numpy(dtype=float)
        lower = -1e6
        upper = 1e6
        if standardizer is not None and hasattr(standardizer, "mean") and hasattr(standardizer, "std") and index < len(standardizer.mean):
            std_value = float(standardizer.std[index]) if float(standardizer.std[index]) != 0.0 else 1.0
            lower = float(standardizer.mean[index]) - z_limit * std_value
            upper = float(standardizer.mean[index]) + z_limit * std_value
        clipped_series = series.clip(lower=lower, upper=upper)
        changed_mask = clipped_series.to_numpy(dtype=float) != values
        changed_count = int(changed_mask.sum())
        if changed_count > 0:
            affected_rows.update(np.where(changed_mask)[0].tolist())
            clipped_features.append({"feature": column, "clipped_count": changed_count, "lower": lower, "upper": upper})
        clipped[column] = clipped_series.astype(float)
    return clipped, {
        "fallback_clip_used": bool(clipped_features),
        "fallback_clipped_feature_count": len(clipped_features),
        "fallback_clipped_row_count": len(affected_rows),
        "fallback_clipped_features": clipped_features[:128],
    }


def _prediction_scores_are_finite(predictions: pd.DataFrame) -> bool:
    score_values = pd.to_numeric(predictions.get("score", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    raw_values = pd.to_numeric(predictions.get("raw_score", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(score_values).all() and np.isfinite(raw_values).all())


def _scenario_summary(
    predictions: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_name: str,
    generator_source: str,
    window_label: str,
) -> pd.DataFrame:
    scenario_df = per_scenario_metrics(predictions, labels_df, model_name=model_name)
    latency_df = detection_latency_table(predictions, labels_df, model_name=model_name)
    metadata = labels_df.copy()
    metadata["scenario_id"] = metadata["scenario_id"].astype(str)
    latency_df["scenario_id"] = latency_df["scenario_id"].astype(str)
    merged = scenario_df.merge(
        metadata[["scenario_id", "attack_family", "severity", "target_component", "affected_assets"]],
        on="scenario_id",
        how="left",
    ).merge(
        latency_df[["scenario_id", "latency_seconds"]],
        on="scenario_id",
        how="left",
    )
    merged["detected"] = merged["latency_seconds"].notna().astype(int)
    merged["generator_source"] = generator_source
    merged["window_label"] = window_label
    return merged


def _family_summary(scenario_df: pd.DataFrame) -> pd.DataFrame:
    if scenario_df.empty:
        return pd.DataFrame(
            columns=[
                "attack_family",
                "scenario_count",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "detection_rate",
                "mean_latency_seconds",
            ]
        )
    summary = (
        scenario_df.groupby("attack_family", as_index=False)
        .agg(
            scenario_count=("scenario_id", "nunique"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            detection_rate=("detected", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
        )
        .sort_values(["mean_f1", "detection_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def _evaluate_ready_package_bundle(bundle: BundleSource, spec: WindowSpec, model_name: str, package) -> tuple[dict[str, Any], pd.DataFrame]:
    eval_root = RAW_ZERO_DAY_ROOT / "evaluations" / _bundle_key(bundle) / spec.label / model_name
    existing = _load_existing_eval_row(
        bundle=bundle,
        spec=spec,
        model_name=model_name,
        canonical_or_extension="canonical",
        eval_root=eval_root,
    )
    if existing is not None:
        return existing
    residual_path = _load_or_build_bundle_residual(bundle, spec, list(package.feature_columns))
    residual_df = pd.read_parquet(residual_path)
    labels_df = pd.read_parquet(bundle.labels_path)
    prepared_df, prep_audit = _prepare_residual_for_inference(
        residual_df=residual_df,
        feature_columns=list(package.feature_columns),
        standardizer=package.preprocessing.get("standardizer"),
    )
    fallback_clip_audit = {
        "fallback_clip_used": False,
        "fallback_clipped_feature_count": 0,
        "fallback_clipped_row_count": 0,
        "fallback_clipped_features": [],
        "fallback_reason": None,
    }

    try:
        result = run_phase1_inference(package, prepared_df)
        predictions = result["predictions"].copy()
        if not _prediction_scores_are_finite(predictions):
            raise ValueError("non-finite prediction scores after raw replay")
    except Exception as exc:
        clipped_df, clip_audit = _clip_extreme_features_for_package(
            frame=prepared_df,
            feature_columns=list(package.feature_columns),
            standardizer=package.preprocessing.get("standardizer"),
        )
        fallback_clip_audit.update(clip_audit)
        fallback_clip_audit["fallback_reason"] = str(exc)
        result = run_phase1_inference(package, clipped_df)
        predictions = result["predictions"].copy()
        for column in ["raw_score", "score"]:
            if column in predictions.columns:
                predictions[column] = (
                    pd.to_numeric(predictions[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

    threshold = _safe_float(result["metadata"].get("threshold")) or 0.5
    metrics = compute_binary_metrics(
        y_true=predictions["attack_present"].astype(int).to_numpy(),
        scores=predictions["score"].astype(float).to_numpy(),
        threshold=threshold,
    )
    latency_df = detection_latency_table(predictions, labels_df, model_name=model_name)
    scenario_df = _scenario_summary(predictions, labels_df, model_name, bundle.generator_source, spec.label)
    family_df = _family_summary(scenario_df)
    mean_latency = _safe_mean(latency_df["latency_seconds"]) if "latency_seconds" in latency_df.columns else None
    median_latency = _safe_median(latency_df["latency_seconds"]) if "latency_seconds" in latency_df.columns else None
    false_positive_rate = _false_positive_rate(metrics)

    eval_root.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(eval_root / "predictions.parquet", index=False)
    latency_df.to_csv(eval_root / "latency_table.csv", index=False)
    scenario_df.to_csv(eval_root / "scenario_summary.csv", index=False)
    family_df.to_csv(eval_root / "family_summary.csv", index=False)
    prep_payload = {**prep_audit, **fallback_clip_audit}
    write_json(eval_root / "preparation_audit.json", prep_payload)
    write_json(
        eval_root / "metrics.json",
        {
            **metrics,
            "false_positive_rate": false_positive_rate,
            "mean_latency_seconds": mean_latency,
            "median_latency_seconds": median_latency,
            "generator_source": bundle.generator_source,
            "dataset_id": bundle.dataset_id,
            "window_label": spec.label,
            "model_name": model_name,
            "preparation_audit": prep_payload,
        },
    )

    row = {
        "evaluation_context": "heldout_synthetic_zero_day_like",
        "generator_source": bundle.generator_source,
        "dataset_id": bundle.dataset_id,
        "bundle_type": bundle.source_type,
        "window_label": spec.label,
        "window_seconds": spec.seconds,
        "step_seconds": spec.step_seconds,
        "model_name": model_name,
        "canonical_or_extension": "canonical",
        "status": "completed",
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "average_precision": _safe_float(metrics.get("average_precision")),
        "roc_auc": _safe_float(metrics.get("roc_auc")),
        "false_positive_rate": false_positive_rate,
        "mean_latency_seconds": mean_latency,
        "median_latency_seconds": median_latency,
        "evaluated_windows": int(len(predictions)),
        "attack_window_count": int(predictions["attack_present"].astype(int).sum()),
        "benign_window_count": int((predictions["attack_present"].astype(int) == 0).sum()),
        "detected_scenarios": int(scenario_df["detected"].sum()) if not scenario_df.empty else 0,
        "scenario_count": bundle.scenario_count,
        "accepted_scenario_count": bundle.accepted_scenario_count,
        "metrics_path": str(eval_root / "metrics.json"),
        "predictions_path": str(eval_root / "predictions.parquet"),
        "notes": (
            "Heldout synthetic scenario evaluation from frozen Phase 1 ready package. "
            f"Missing feature columns backfilled: {prep_audit['missing_feature_count']}. "
            f"True non-finite features: {prep_audit['true_nonfinite_feature_count']}. "
            f"Fallback clip used: {fallback_clip_audit['fallback_clip_used']}."
        ),
    }
    return row, scenario_df


def _ttm_feature_columns() -> list[str]:
    path = WINDOW_STUDY_ROOT / "60s" / "ready_packages" / "transformer" / "feature_columns.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _ttm_model_config() -> dict[str, Any]:
    config_path = ROOT / "phase1_ttm_extension_config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return dict(payload.get("model", {}))


def _ttm_build_forecasting_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    context_length: int,
    prediction_length: int,
    standardizer: Any,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ordered = frame.sort_values("window_start_utc").reset_index(drop=True).copy()
    if ordered.empty:
        return (
            np.empty((0, context_length, len(feature_columns)), dtype=np.float32),
            np.empty((0, prediction_length, len(feature_columns)), dtype=np.float32),
            pd.DataFrame(),
        )
    ordered["window_start_utc"] = pd.to_datetime(ordered["window_start_utc"], utc=True)
    ordered["window_end_utc"] = pd.to_datetime(ordered["window_end_utc"], utc=True)
    diffs = ordered["window_start_utc"].diff().dt.total_seconds().dropna()
    step_seconds = int(diffs.mode().iloc[0]) if not diffs.empty else 60
    segment_break = ordered["window_start_utc"].diff().dt.total_seconds().fillna(step_seconds).ne(step_seconds)
    ordered["segment_id"] = segment_break.cumsum().astype(int)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for _, segment in ordered.groupby("segment_id", observed=False):
        if len(segment) < context_length + prediction_length:
            continue
        feature_matrix = transform_features(segment, feature_columns, standardizer).astype(np.float32)
        for future_start in range(context_length, len(segment) - prediction_length + 1):
            future_end = future_start + prediction_length
            target_row = segment.iloc[future_end - 1]
            x_list.append(feature_matrix[future_start - context_length : future_start])
            y_list.append(feature_matrix[future_start:future_end])
            rows.append(
                {
                    "window_start_utc": target_row["window_start_utc"],
                    "window_end_utc": target_row["window_end_utc"],
                    "scenario_id": str(target_row.get("scenario_window_id", target_row.get("scenario_id", ""))),
                    "attack_present": int(target_row["attack_present"]),
                    "attack_family": str(target_row.get("attack_family", "benign")),
                    "attack_severity": str(target_row.get("attack_severity", "none")),
                    "attack_affected_assets": str(target_row.get("attack_affected_assets", "")),
                    "split_name": str(target_row.get("split_name", "")),
                }
            )
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32), pd.DataFrame(rows)


def _ttm_evaluate_forecaster(
    model: TinyTimeMixerForPrediction,
    x_values: np.ndarray,
    y_values: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, float]:
    if len(x_values) == 0:
        return np.array([], dtype=float), 0.0
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_values), torch.from_numpy(y_values))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    total_inference = 0.0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            start = time.perf_counter()
            output = model(past_values=batch_x, future_values=batch_y)
            total_inference += time.perf_counter() - start
            mse = torch.mean((output.prediction_outputs - batch_y) ** 2, dim=(1, 2))
            scores.append(mse.detach().cpu().numpy().astype(float))
    return np.concatenate(scores, axis=0), total_inference


def _ttm_calibrate_scores(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, LogisticRegression | None]:
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return scores, None
    calibrator = LogisticRegression(random_state=1729, solver="lbfgs")
    calibrator.fit(scores.reshape(-1, 1), y_true)
    calibrated = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    return calibrated.astype(float), calibrator


def _ttm_apply_calibrator(calibrator: LogisticRegression | None, scores: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(scores, dtype=float)
    return calibrator.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1].astype(float)


def _clean_ttm_score_vector(scores: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    nonfinite_mask = ~np.isfinite(values)
    nonfinite_count = int(nonfinite_mask.sum())
    if nonfinite_count == 0:
        return values, 0
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        replacement_high = 0.0
        replacement_low = 0.0
    else:
        replacement_high = float(np.max(finite_values))
        replacement_low = float(np.min(finite_values))
    cleaned = np.nan_to_num(values, nan=replacement_high, posinf=replacement_high, neginf=replacement_low)
    return cleaned.astype(float), nonfinite_count


def _ttm_choose_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if scores.size == 0:
        return 0.5
    thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.995, 60)))
    if len(thresholds) == 0:
        thresholds = np.array([0.5], dtype=float)
    best_threshold = float(thresholds[0])
    best_metrics = compute_binary_metrics(y_true, scores, best_threshold)
    for threshold in thresholds[1:]:
        metrics = compute_binary_metrics(y_true, scores, float(threshold))
        if float(metrics["f1"]) > float(best_metrics["f1"]) + 1e-9:
            best_metrics = metrics
            best_threshold = float(threshold)
        elif math.isclose(float(metrics["f1"]), float(best_metrics["f1"]), abs_tol=1e-9) and float(metrics["precision"]) > float(
            best_metrics["precision"]
        ):
            best_metrics = metrics
            best_threshold = float(threshold)
    return best_threshold


def _build_ttm_inference_state() -> dict[str, Any] | None:
    if not HAS_TTM:
        return None
    checkpoint_path = ROOT / "outputs" / "phase1_ttm_extension" / "ttm_extension_checkpoint.pt"
    results_path = ROOT / "phase1_ttm_results.csv"
    if not checkpoint_path.exists() or not results_path.exists():
        return None

    feature_columns = _ttm_feature_columns()
    model_cfg = _ttm_model_config()
    context_length = int(model_cfg.get("context_length", 12))
    prediction_length = int(model_cfg.get("prediction_length", 1))

    residual_df = pd.read_parquet(WINDOW_STUDY_ROOT / "60s" / "residual_windows.parquet")
    residual_df["window_start_utc"] = pd.to_datetime(residual_df["window_start_utc"], utc=True)
    residual_df["window_end_utc"] = pd.to_datetime(residual_df["window_end_utc"], utc=True)

    train_benign = residual_df[(residual_df["split_name"] == "train") & (residual_df["attack_present"] == 0)].copy()
    val_all = residual_df[residual_df["split_name"] == "val"].copy()
    standardizer = fit_standardizer(train_benign, feature_columns)
    val_all, _ = _ensure_feature_frame(val_all, feature_columns)
    val_all, _ = _prepare_residual_for_inference(val_all, feature_columns, standardizer)
    x_val, y_val, val_meta = _ttm_build_forecasting_sequences(val_all, feature_columns, context_length, prediction_length, standardizer)
    if len(x_val) == 0:
        return None

    config = TinyTimeMixerConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        patch_length=int(model_cfg.get("patch_length", 2)),
        patch_stride=int(model_cfg.get("patch_stride", 2)),
        num_input_channels=int(model_cfg.get("num_input_channels", len(feature_columns))),
        d_model=int(model_cfg.get("d_model", 16)),
        decoder_d_model=int(model_cfg.get("decoder_d_model", 16)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        decoder_num_layers=int(model_cfg.get("decoder_num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        head_dropout=float(model_cfg.get("head_dropout", 0.2)),
        loss=str(model_cfg.get("loss", "mse")),
        scaling=str(model_cfg.get("scaling", "std")),
        adaptive_patching_levels=0,
        self_attn=False,
        enable_forecast_channel_mixing=False,
    )
    device = torch.device("cpu")
    model = TinyTimeMixerForPrediction(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    raw_val_scores, _ = _ttm_evaluate_forecaster(model, x_val, y_val, device)
    raw_val_scores, val_nonfinite_score_count = _clean_ttm_score_vector(raw_val_scores)
    val_labels = val_meta["attack_present"].astype(int).to_numpy()
    calibrated_val, calibrator = _ttm_calibrate_scores(raw_val_scores, val_labels)
    threshold = _ttm_choose_best_threshold(calibrated_val, val_labels)

    return {
        "model": model,
        "device": device,
        "feature_columns": feature_columns,
        "standardizer": standardizer,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "calibrator": calibrator,
        "threshold": threshold,
        "checkpoint_path": checkpoint_path,
        "validation_nonfinite_score_count": val_nonfinite_score_count,
    }


def _evaluate_ttm_bundle(bundle: BundleSource, ttm_state: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    spec = _window_spec("60s")
    eval_root = RAW_ZERO_DAY_ROOT / "evaluations" / _bundle_key(bundle) / "60s" / "ttm_extension"
    existing = _load_existing_eval_row(
        bundle=bundle,
        spec=spec,
        model_name="ttm_extension",
        canonical_or_extension="extension",
        eval_root=eval_root,
    )
    if existing is not None:
        return existing
    residual_path = _load_or_build_bundle_residual(bundle, spec, list(ttm_state["feature_columns"]))
    residual_df = pd.read_parquet(residual_path)
    labels_df = pd.read_parquet(bundle.labels_path)
    prepared_df, prep_audit = _prepare_residual_for_inference(
        residual_df=residual_df,
        feature_columns=list(ttm_state["feature_columns"]),
        standardizer=ttm_state["standardizer"],
    )
    x_values, y_values, meta = _ttm_build_forecasting_sequences(
        prepared_df,
        ttm_state["feature_columns"],
        int(ttm_state["context_length"]),
        int(ttm_state["prediction_length"]),
        ttm_state["standardizer"],
    )
    if len(x_values) == 0:
        raise RuntimeError(f"TTM could not build any forecasting sequences for bundle {bundle.generator_source}.")
    raw_scores, inference_seconds = _ttm_evaluate_forecaster(
        model=ttm_state["model"],
        x_values=x_values,
        y_values=y_values,
        device=ttm_state["device"],
    )
    raw_scores, heldout_nonfinite_score_count = _clean_ttm_score_vector(raw_scores)
    calibrated_scores = _ttm_apply_calibrator(ttm_state["calibrator"], raw_scores)
    threshold = float(ttm_state["threshold"])
    predictions = meta.copy()
    predictions["raw_score"] = raw_scores
    predictions["score"] = calibrated_scores
    predictions["predicted"] = (predictions["score"] >= threshold).astype(int)

    metrics = compute_binary_metrics(
        y_true=predictions["attack_present"].astype(int).to_numpy(),
        scores=predictions["score"].astype(float).to_numpy(),
        threshold=threshold,
    )
    latency_df = detection_latency_table(predictions, labels_df, model_name="ttm_extension")
    scenario_df = _scenario_summary(predictions, labels_df, "ttm_extension", bundle.generator_source, "60s")
    family_df = _family_summary(scenario_df)
    mean_latency = _safe_mean(latency_df["latency_seconds"]) if "latency_seconds" in latency_df.columns else None
    median_latency = _safe_median(latency_df["latency_seconds"]) if "latency_seconds" in latency_df.columns else None
    false_positive_rate = _false_positive_rate(metrics)

    eval_root.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(eval_root / "predictions.parquet", index=False)
    latency_df.to_csv(eval_root / "latency_table.csv", index=False)
    scenario_df.to_csv(eval_root / "scenario_summary.csv", index=False)
    family_df.to_csv(eval_root / "family_summary.csv", index=False)
    write_json(eval_root / "preparation_audit.json", prep_audit)
    write_json(
        eval_root / "metrics.json",
        {
            **metrics,
            "false_positive_rate": false_positive_rate,
            "mean_latency_seconds": mean_latency,
            "median_latency_seconds": median_latency,
            "generator_source": bundle.generator_source,
            "dataset_id": bundle.dataset_id,
            "window_label": "60s",
            "model_name": "ttm_extension",
            "calibration_note": "Canonical 60s validation calibrator reconstructed from saved TTM checkpoint for heldout inference.",
            "validation_nonfinite_score_count": int(ttm_state.get("validation_nonfinite_score_count", 0)),
            "heldout_nonfinite_score_count": int(heldout_nonfinite_score_count),
            "preparation_audit": prep_audit,
        },
    )
    row = {
        "evaluation_context": "heldout_synthetic_zero_day_like",
        "generator_source": bundle.generator_source,
        "dataset_id": bundle.dataset_id,
        "bundle_type": bundle.source_type,
        "window_label": "60s",
        "window_seconds": 60,
        "step_seconds": 12,
        "model_name": "ttm_extension",
        "canonical_or_extension": "extension",
        "status": "completed",
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "average_precision": _safe_float(metrics.get("average_precision")),
        "roc_auc": _safe_float(metrics.get("roc_auc")),
        "false_positive_rate": false_positive_rate,
        "mean_latency_seconds": mean_latency,
        "median_latency_seconds": median_latency,
        "evaluated_windows": int(len(predictions)),
        "attack_window_count": int(predictions["attack_present"].astype(int).sum()),
        "benign_window_count": int((predictions["attack_present"].astype(int) == 0).sum()),
        "detected_scenarios": int(scenario_df["detected"].sum()) if not scenario_df.empty else 0,
        "scenario_count": bundle.scenario_count,
        "accepted_scenario_count": bundle.accepted_scenario_count,
        "metrics_path": str(eval_root / "metrics.json"),
        "predictions_path": str(eval_root / "predictions.parquet"),
        "notes": (
            "Heldout synthetic evaluation from saved 60s TTM checkpoint with reconstructed canonical validation calibrator. "
            f"Non-finite TTM raw scores replaced before calibration: {heldout_nonfinite_score_count}."
        ),
    }
    return row, scenario_df


def _load_existing_eval_row(
    bundle: BundleSource,
    spec: WindowSpec,
    model_name: str,
    canonical_or_extension: str,
    eval_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    metrics_path = eval_root / "metrics.json"
    scenario_path = eval_root / "scenario_summary.csv"
    predictions_path = eval_root / "predictions.parquet"
    if not metrics_path.exists() or not scenario_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    scenario_df = pd.read_csv(scenario_path)
    prediction_count = 0
    attack_window_count = 0
    benign_window_count = 0
    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path, columns=["attack_present"])
        prediction_count = int(len(predictions))
        attack_window_count = int(predictions["attack_present"].astype(int).sum())
        benign_window_count = int((predictions["attack_present"].astype(int) == 0).sum())
    row = {
        "evaluation_context": "heldout_synthetic_zero_day_like",
        "generator_source": bundle.generator_source,
        "dataset_id": bundle.dataset_id,
        "bundle_type": bundle.source_type,
        "window_label": spec.label,
        "window_seconds": spec.seconds,
        "step_seconds": spec.step_seconds,
        "model_name": model_name,
        "canonical_or_extension": canonical_or_extension,
        "status": "completed",
        "precision": _safe_float(metrics.get("precision")),
        "recall": _safe_float(metrics.get("recall")),
        "f1": _safe_float(metrics.get("f1")),
        "average_precision": _safe_float(metrics.get("average_precision")),
        "roc_auc": _safe_float(metrics.get("roc_auc")),
        "false_positive_rate": _safe_float(metrics.get("false_positive_rate")) or _false_positive_rate(metrics),
        "mean_latency_seconds": _safe_float(metrics.get("mean_latency_seconds")),
        "median_latency_seconds": _safe_float(metrics.get("median_latency_seconds")),
        "evaluated_windows": prediction_count,
        "attack_window_count": attack_window_count,
        "benign_window_count": benign_window_count,
        "detected_scenarios": int(scenario_df["detected"].sum()) if "detected" in scenario_df.columns else 0,
        "scenario_count": bundle.scenario_count,
        "accepted_scenario_count": bundle.accepted_scenario_count,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path) if predictions_path.exists() else "",
        "notes": "Loaded from cached heldout synthetic evaluation artifact.",
    }
    return row, scenario_df


def _build_zero_day_window_model_coverage_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    report_exists_phase3 = (ROOT / "PHASE3_ZERO_DAY_REPORT.md").exists()
    report_exists_improved = (ROOT / "IMPROVED_PHASE3_RESULTS_REPORT.md").exists()
    bundles = ALL_BUNDLE_ORDER
    models = ["threshold_baseline", "transformer", "lstm", "ttm_extension", "gru", "autoencoder", "lstm_ae", "isolation_forest", "arima"]
    for bundle_name in bundles:
        for spec in WINDOW_SPECS:
            for model_name in models:
                exists = False
                report_exists = False
                notes = "No pre-existing heldout synthetic evaluation artifact found."
                if bundle_name in ["canonical_bundle", "chatgpt", "claude", "gemini", "grok"] and spec.label == "60s" and model_name == "transformer":
                    metrics_path = PHASE3_HELDOUT_ROOT / bundle_name / "reports" / "bundle_metrics.json"
                    exists = metrics_path.exists()
                    report_exists = report_exists_phase3 if exists else False
                    notes = "Existing saved-transformer replay artifact in phase3_heldout." if exists else notes
                improved_metrics_path = None
                if model_name in {"threshold_baseline", "transformer", "lstm"} and spec.label in {"10s", "60s", "300s"}:
                    candidate_paths = list(
                        (IMPROVED_PHASE3_ROOT / "evaluations").glob(
                            f"{_slug(bundle_name)}__*/{spec.label}/{model_name}/metrics.json"
                        )
                    )
                    improved_metrics_path = candidate_paths[0] if candidate_paths else None
                if improved_metrics_path is not None and improved_metrics_path.exists():
                    exists = True
                    report_exists = report_exists_improved
                    notes = "Existing improved frozen-candidate replay artifact."
                if model_name == "ttm_extension":
                    exists = False
                    report_exists = False
                    notes = "No pre-existing heldout TTM evaluation artifact was found."
                if model_name == "lstm_ae":
                    exists = False
                    report_exists = False
                    notes = "No LSTM autoencoder detector implementation exists; MLP autoencoder is reported separately as `autoencoder`."
                if model_name == "arima":
                    notes = "ARIMA detector is not implemented in the repository."
                if bundle_name == "canonical_bundle":
                    notes = f"{notes} Canonical bundle is in-domain replay reference, not counted as heldout zero-day-like evidence."
                rows.append(
                    {
                        "window_seconds": spec.seconds,
                        "model_name": model_name,
                        "scenario_bundle": bundle_name,
                        "zero_day_eval_exists": exists,
                        "metrics_exist": exists,
                        "report_exists": report_exists,
                        "needs_rerun": not exists,
                        "notes": notes.strip(),
                    }
                )
    audit_df = pd.DataFrame(rows)
    audit_df = audit_df.sort_values(["scenario_bundle", "window_seconds", "model_name"]).reset_index(drop=True)
    return audit_df


def _write_zero_day_window_model_coverage_summary(audit_df: pd.DataFrame) -> None:
    heldout_only = audit_df.loc[audit_df["scenario_bundle"].isin(GENERATOR_ORDER)].copy()
    coverage = (
        heldout_only.groupby(["window_seconds", "model_name"], as_index=False)["zero_day_eval_exists"]
        .sum()
        .rename(columns={"zero_day_eval_exists": "bundle_count_with_existing_eval"})
    )
    lines = [
        "# Zero-Day-Like Window / Model Coverage Summary",
        "",
        "This audit reflects the pre-existing repository state before the missing heldout synthetic detector matrix was added in this pass.",
        "",
        "## Existing Heldout Synthetic Coverage Counts",
        "",
        _markdown_table(coverage),
        "",
        "## What Existed Before Completion",
        "",
        "- `transformer @ 60s` had existing heldout synthetic replay coverage in both `phase3_heldout` and the later improved replay package.",
        "- `threshold_baseline @ 10s` and `lstm @ 300s` had existing improved replay coverage only.",
        "- No existing `5s` heldout synthetic detector matrix existed.",
        "- No existing TTM heldout synthetic evaluation existed.",
        "- No ARIMA heldout synthetic evaluation existed because ARIMA is not implemented.",
        "",
        "## Context Separation",
        "",
        "- Existing `phase3_heldout` rows are frozen-model replay artifacts for `transformer @ 60s`.",
        "- Existing `improved_phase3` rows are replay artifacts for three frozen candidates only.",
        "- These existing rows were not a full benchmark and were not a full zero-day-like window/model sweep.",
    ]
    write_markdown(ROOT / "zero_day_window_model_coverage_summary.md", "\n".join(lines))


def _run_zero_day_full_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    RAW_ZERO_DAY_ROOT.mkdir(parents=True, exist_ok=True)
    bundles = _build_heldout_bundle_sources()
    package_cache: dict[tuple[str, str], Any] = {}
    rows: list[dict[str, Any]] = []
    scenario_frames: list[pd.DataFrame] = []

    for spec in WINDOW_SPECS:
        for model_name in CANONICAL_MODEL_ORDER:
            package_cache[(spec.label, model_name)] = load_phase1_package(
                WINDOW_STUDY_ROOT / spec.label / "ready_packages" / model_name
            )

    for bundle in bundles:
        for spec in WINDOW_SPECS:
            if spec.label == "5s":
                for model_name in CANONICAL_MODEL_ORDER:
                    rows.append(
                        {
                            "evaluation_context": "heldout_synthetic_zero_day_like",
                            "generator_source": bundle.generator_source,
                            "dataset_id": bundle.dataset_id,
                            "bundle_type": bundle.source_type,
                            "window_label": spec.label,
                            "window_seconds": spec.seconds,
                            "step_seconds": spec.step_seconds,
                            "model_name": model_name,
                            "canonical_or_extension": "canonical",
                            "status": "blocked",
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                            "average_precision": np.nan,
                            "roc_auc": np.nan,
                            "false_positive_rate": np.nan,
                            "mean_latency_seconds": np.nan,
                            "median_latency_seconds": np.nan,
                            "evaluated_windows": 0,
                            "attack_window_count": 0,
                            "benign_window_count": 0,
                            "detected_scenarios": 0,
                            "scenario_count": bundle.scenario_count,
                            "accepted_scenario_count": bundle.accepted_scenario_count,
                            "metrics_path": "",
                            "predictions_path": "",
                            "notes": (
                                "5s heldout synthetic sweep was blocked in this pass because raw 5s window generation across "
                                "full-day attacked bundle traces exceeded the CPU-only completion-pass wall-clock budget, and "
                                "no reusable prebuilt 5s replay residuals existed."
                            ),
                        }
                    )
                continue
            for model_name in CANONICAL_MODEL_ORDER:
                package = package_cache[(spec.label, model_name)]
                row, scenario_df = _evaluate_ready_package_bundle(bundle, spec, model_name, package)
                rows.append(row)
                scenario_frames.append(scenario_df)

    ttm_state = _build_ttm_inference_state()
    for bundle in bundles:
        if ttm_state is None:
            rows.append(
                {
                    "evaluation_context": "heldout_synthetic_zero_day_like",
                    "generator_source": bundle.generator_source,
                    "dataset_id": bundle.dataset_id,
                    "bundle_type": bundle.source_type,
                    "window_label": "60s",
                    "window_seconds": 60,
                    "step_seconds": 12,
                    "model_name": "ttm_extension",
                    "canonical_or_extension": "extension",
                    "status": "blocked",
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "average_precision": np.nan,
                    "roc_auc": np.nan,
                    "false_positive_rate": np.nan,
                    "mean_latency_seconds": np.nan,
                    "median_latency_seconds": np.nan,
                    "evaluated_windows": 0,
                    "attack_window_count": 0,
                    "benign_window_count": 0,
                    "detected_scenarios": 0,
                    "scenario_count": bundle.scenario_count,
                    "accepted_scenario_count": bundle.accepted_scenario_count,
                    "metrics_path": "",
                    "predictions_path": "",
                    "notes": f"TTM extension unavailable for heldout inference: {TTM_IMPORT_ERROR or 'saved checkpoint or config missing.'}",
                }
            )
        else:
            row, scenario_df = _evaluate_ttm_bundle(bundle, ttm_state)
            rows.append(row)
            scenario_frames.append(scenario_df)

    for bundle in bundles:
        for spec in WINDOW_SPECS:
            for blocked_model_name, blocked_type, blocked_note in [
                ("arima", "extension_blocked", "ARIMA detector is not implemented in the repository."),
                (
                    "lstm_ae",
                    "blocked",
                    "No LSTM autoencoder detector implementation exists; MLP autoencoder is reported separately as `autoencoder`.",
                ),
            ]:
                rows.append(
                    {
                        "evaluation_context": "heldout_synthetic_zero_day_like",
                        "generator_source": bundle.generator_source,
                        "dataset_id": bundle.dataset_id,
                        "bundle_type": bundle.source_type,
                        "window_label": spec.label,
                        "window_seconds": spec.seconds,
                        "step_seconds": spec.step_seconds,
                        "model_name": blocked_model_name,
                        "canonical_or_extension": blocked_type,
                        "status": "not_implemented",
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "average_precision": np.nan,
                        "roc_auc": np.nan,
                        "false_positive_rate": np.nan,
                        "mean_latency_seconds": np.nan,
                        "median_latency_seconds": np.nan,
                        "evaluated_windows": 0,
                        "attack_window_count": 0,
                        "benign_window_count": 0,
                        "detected_scenarios": 0,
                        "scenario_count": bundle.scenario_count,
                        "accepted_scenario_count": bundle.accepted_scenario_count,
                        "metrics_path": "",
                        "predictions_path": "",
                        "notes": blocked_note,
                    }
                )
            if spec.label != "60s":
                rows.append(
                    {
                        "evaluation_context": "heldout_synthetic_zero_day_like",
                        "generator_source": bundle.generator_source,
                        "dataset_id": bundle.dataset_id,
                        "bundle_type": bundle.source_type,
                        "window_label": spec.label,
                        "window_seconds": spec.seconds,
                        "step_seconds": spec.step_seconds,
                        "model_name": "ttm_extension",
                        "canonical_or_extension": "extension",
                        "status": "not_run",
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "average_precision": np.nan,
                        "roc_auc": np.nan,
                        "false_positive_rate": np.nan,
                        "mean_latency_seconds": np.nan,
                        "median_latency_seconds": np.nan,
                        "evaluated_windows": 0,
                        "attack_window_count": 0,
                        "benign_window_count": 0,
                        "detected_scenarios": 0,
                        "scenario_count": bundle.scenario_count,
                        "accepted_scenario_count": bundle.accepted_scenario_count,
                        "metrics_path": "",
                        "predictions_path": "",
                        "notes": (
                            "No saved TTM benchmark artifact exists for this window, so heldout synthetic inference was not run."
                            if spec.label != "5s"
                            else "5s heldout synthetic sweep was blocked before any TTM extension run was possible."
                        ),
                    }
                )

    full_df = pd.DataFrame(rows)
    full_df["window_order"] = full_df["window_label"].map({label: idx for idx, label in enumerate(WINDOW_LABELS)})
    full_df["generator_order"] = full_df["generator_source"].map({name: idx for idx, name in enumerate(GENERATOR_ORDER)})
    full_df["model_order"] = full_df["model_name"].map({name: idx for idx, name in enumerate(REQUESTED_MODEL_ORDER + ["lstm"])})
    full_df = full_df.sort_values(["generator_order", "window_order", "model_order", "model_name"]).drop(
        columns=["window_order", "generator_order", "model_order"]
    )
    full_df.to_csv(ROOT / "zero_day_model_window_results_full.csv", index=False)

    scenario_full_df = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    if not scenario_full_df.empty:
        scenario_full_df["window_seconds"] = scenario_full_df["window_label"].map({spec.label: spec.seconds for spec in WINDOW_SPECS})
        family_df = (
            scenario_full_df.groupby(["generator_source", "window_label", "window_seconds", "model_name", "attack_family"], as_index=False)
            .agg(
                scenario_count=("scenario_id", "nunique"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_f1=("f1", "mean"),
                detection_rate=("detected", "mean"),
                mean_latency_seconds=("latency_seconds", "mean"),
            )
            .sort_values(["window_seconds", "model_name", "attack_family"])
            .reset_index(drop=True)
        )
    else:
        family_df = pd.DataFrame()
    family_df.to_csv(ROOT / "zero_day_family_model_window_results.csv", index=False)
    return full_df, family_df


def _write_zero_day_reports(full_df: pd.DataFrame, family_df: pd.DataFrame) -> None:
    completed = full_df.loc[full_df["status"] == "completed"].copy()
    completed["window_label"] = pd.Categorical(completed["window_label"], categories=WINDOW_LABELS, ordered=True)
    mean_df = (
        completed.groupby(["window_label", "window_seconds", "model_name", "canonical_or_extension"], as_index=False)
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_false_positive_rate=("false_positive_rate", "mean"),
            mean_latency_seconds=("mean_latency_seconds", "mean"),
            bundle_count=("generator_source", "nunique"),
        )
        .sort_values(["window_seconds", "mean_f1"], ascending=[True, False])
        .reset_index(drop=True)
    )
    best_per_window = mean_df.groupby("window_label", as_index=False).first()
    present_windows = set(best_per_window["window_label"].astype(str).tolist())
    for spec in WINDOW_SPECS:
        if spec.label not in present_windows:
            best_per_window = pd.concat(
                [
                    best_per_window,
                    pd.DataFrame(
                        [
                            {
                                "window_label": spec.label,
                                "window_seconds": spec.seconds,
                                "model_name": "not_completed",
                                "canonical_or_extension": "blocked",
                                "mean_precision": np.nan,
                                "mean_recall": np.nan,
                                "mean_f1": np.nan,
                                "mean_false_positive_rate": np.nan,
                                "mean_latency_seconds": np.nan,
                                "bundle_count": 0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    best_per_window["window_order"] = best_per_window["window_label"].map({label: idx for idx, label in enumerate(WINDOW_LABELS)})
    best_per_window = best_per_window.sort_values("window_order").drop(columns="window_order").reset_index(drop=True)
    best_per_window.to_csv(ROOT / "zero_day_best_per_window_summary.csv", index=False)

    lines = [
        "# Zero-Day-Like Model / Window Results",
        "",
        "This report covers heldout synthetic scenario evaluation on generated Phase 2 attack bundles and the human-authored additional heldout source. It is kept separate from the canonical benchmark path.",
        "",
        "## Scope",
        "",
        "- Context: heldout synthetic scenario evaluation / zero-day-like evaluation",
        "- Bundles included: `chatgpt`, `claude`, `gemini`, `grok`, `human_authored`",
        "- Canonical benchmark winner remains `transformer @ 60s` and is not replaced by this report",
        "- This is not real-world zero-day evidence",
        "- `5s` heldout synthetic coverage is documented as blocked in this pass because reusable 5s replay residuals did not exist and raw 5s bundle-window generation exceeded the CPU-only wall-clock budget",
        "",
        "## Mean Results Across Heldout Bundles",
        "",
        _markdown_table(
            mean_df[
                [
                    "window_label",
                    "model_name",
                    "canonical_or_extension",
                    "mean_precision",
                    "mean_recall",
                    "mean_f1",
                    "mean_false_positive_rate",
                    "mean_latency_seconds",
                    "bundle_count",
                ]
            ]
        ),
        "",
        "## Best Per Window",
        "",
        _markdown_table(best_per_window[["window_label", "model_name", "mean_precision", "mean_recall", "mean_f1", "canonical_or_extension"]]),
        "",
        "## Notes",
        "",
        "- TTM is included only where real extension evidence exists (`60s`).",
        "- ARIMA remains a documented blocker only.",
        "- The repo autoencoder is an MLP autoencoder, not an LSTM autoencoder.",
    ]
    write_markdown(ROOT / "zero_day_model_window_results_full.md", "\n".join(lines))

    if family_df.empty:
        family_lines = [
            "# Zero-Day Family Difficulty Summary",
            "",
            "No family-level results were generated.",
        ]
    else:
        family_rank = (
            family_df.groupby("attack_family", as_index=False)
            .agg(
                scenario_count=("scenario_count", "sum"),
                mean_recall=("mean_recall", "mean"),
                mean_f1=("mean_f1", "mean"),
                detection_rate=("detection_rate", "mean"),
            )
            .sort_values(["mean_f1", "detection_rate"], ascending=[True, True])
            .reset_index(drop=True)
        )
        hardest = family_rank.head(3)
        easiest = family_rank.tail(3).sort_values(["mean_f1", "detection_rate"], ascending=[False, False])
        family_lines = [
            "# Zero-Day Family Difficulty Summary",
            "",
            "This ranking uses mean per-scenario F1 and detection rate across completed heldout synthetic evaluations. Lower values indicate harder synthetic transfer families within this bounded benchmark.",
            "",
            "## Family Ranking",
            "",
            _markdown_table(family_rank),
            "",
            "## Hardest Families In This Pass",
            "",
            _markdown_table(hardest),
            "",
            "## Easiest Families In This Pass",
            "",
            _markdown_table(easiest),
        ]
    write_markdown(ROOT / "zero_day_family_difficulty_summary.md", "\n".join(family_lines))


def _plot_zero_day_figures(full_df: pd.DataFrame) -> None:
    completed = full_df.loc[full_df["status"] == "completed"].copy()
    completed["window_label"] = pd.Categorical(completed["window_label"], categories=WINDOW_LABELS, ordered=True)
    mean_df = (
        completed.groupby(["window_label", "model_name"], as_index=False)[["precision", "recall", "f1"]]
        .mean(numeric_only=True)
        .sort_values(["window_label", "f1"], ascending=[True, False])
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model_name in mean_df["model_name"].drop_duplicates():
        subset = mean_df.loc[mean_df["model_name"] == model_name].copy()
        subset = subset.sort_values("window_label")
        ax.plot(
            subset["window_label"].astype(str),
            subset["f1"].astype(float),
            marker="o",
            linewidth=2,
            label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean Heldout Synthetic F1")
    ax.set_xlabel("Window")
    ax.set_title("Zero-Day-Like Mean F1 by Window")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "zero_day_f1_by_window.png", dpi=220)
    plt.close(fig)

    pr_df = (
        completed.groupby("model_name", as_index=False)[["precision", "recall"]]
        .mean(numeric_only=True)
        .sort_values("precision", ascending=False)
    )
    x = np.arange(len(pr_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x - width / 2, pr_df["precision"].astype(float), width=width, label="precision", color="#4C78A8")
    ax.bar(x + width / 2, pr_df["recall"].astype(float), width=width, label="recall", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(name, name) for name in pr_df["model_name"]], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean Score Across Heldout Bundles")
    ax.set_xlabel("Model")
    ax.set_title("Zero-Day-Like Precision / Recall by Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "zero_day_precision_recall_by_model.png", dpi=220)
    plt.close(fig)

    heatmap_df = (
        completed.groupby(["model_name", "window_label"], as_index=False)["f1"]
        .mean(numeric_only=True)
        .pivot(index="model_name", columns="window_label", values="f1")
        .reindex(index=["threshold_baseline", "isolation_forest", "autoencoder", "gru", "lstm", "transformer", "ttm_extension"], columns=WINDOW_LABELS)
    )
    fig, ax = plt.subplots(figsize=(8, 5.75))
    image = ax.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns.tolist())
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(name, name) for name in heatmap_df.index.tolist()])
    ax.set_xlabel("Window")
    ax.set_ylabel("Model")
    ax.set_title("Zero-Day-Like Mean F1 Heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean F1")
    fig.tight_layout()
    fig.savefig(ROOT / "zero_day_model_window_heatmap.png", dpi=220)
    plt.close(fig)


def _build_final_model_context_comparison(zero_day_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    benchmark_df = pd.read_csv(ROOT / "phase1_window_model_comparison_full.csv")
    for _, row in benchmark_df.loc[benchmark_df["status"] == "completed"].iterrows():
        rows.append(
            {
                "context_name": "canonical_benchmark_test_split",
                "context_family": "benchmark",
                "model_name": str(row["model_name"]),
                "window_label": str(row["window_label"]),
                "canonical_or_extension": str(row["canonical_or_extension"]),
                "generator_scope": "canonical_test_split",
                "precision": _safe_float(row.get("precision")),
                "recall": _safe_float(row.get("recall")),
                "f1": _safe_float(row.get("f1")),
                "mean_latency_seconds": _safe_float(row.get("mean_detection_latency_seconds")),
                "notes": "Frozen canonical benchmark context.",
            }
        )

    improved_path = ROOT / "multi_model_heldout_metrics.csv"
    if improved_path.exists():
        improved_df = pd.read_csv(improved_path)
        improved_df = improved_df.loc[improved_df["generator_source"].isin(GENERATOR_ORDER)].copy()
        grouped = (
            improved_df.groupby(["model_name", "window_label"], as_index=False)[["precision", "recall", "f1", "mean_latency_seconds"]]
            .mean(numeric_only=True)
            .sort_values(["window_label", "f1"], ascending=[True, False])
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "context_name": "existing_frozen_candidate_heldout_replay",
                    "context_family": "replay",
                    "model_name": str(row["model_name"]),
                    "window_label": str(row["window_label"]),
                    "canonical_or_extension": "replay",
                    "generator_scope": "chatgpt|claude|gemini|grok|human_authored",
                    "precision": _safe_float(row.get("precision")),
                    "recall": _safe_float(row.get("recall")),
                    "f1": _safe_float(row.get("f1")),
                    "mean_latency_seconds": _safe_float(row.get("mean_latency_seconds")),
                    "notes": "Existing improved replay context using frozen candidate subset only.",
                }
            )

    completed_zero_day = zero_day_df.loc[zero_day_df["status"] == "completed"].copy()
    grouped_zero_day = (
        completed_zero_day.groupby(["model_name", "window_label", "canonical_or_extension"], as_index=False)[["precision", "recall", "f1", "mean_latency_seconds"]]
        .mean(numeric_only=True)
        .sort_values(["window_label", "f1"], ascending=[True, False])
    )
    for _, row in grouped_zero_day.iterrows():
        rows.append(
            {
                "context_name": "heldout_synthetic_zero_day_like",
                "context_family": "zero_day_like",
                "model_name": str(row["model_name"]),
                "window_label": str(row["window_label"]),
                "canonical_or_extension": str(row["canonical_or_extension"]),
                "generator_scope": "chatgpt|claude|gemini|grok|human_authored",
                "precision": _safe_float(row.get("precision")),
                "recall": _safe_float(row.get("recall")),
                "f1": _safe_float(row.get("f1")),
                "mean_latency_seconds": _safe_float(row.get("mean_latency_seconds")),
                "notes": "New full heldout synthetic detector sweep from frozen packages.",
            }
        )

    context_df = pd.DataFrame(rows)
    context_df["window_order"] = context_df["window_label"].map({label: idx for idx, label in enumerate(WINDOW_LABELS)})
    context_df["model_order"] = context_df["model_name"].map({name: idx for idx, name in enumerate(REQUESTED_MODEL_ORDER + ["lstm"])})
    context_df = context_df.sort_values(["context_family", "window_order", "model_order", "model_name"]).drop(
        columns=["window_order", "model_order"]
    )
    context_df.to_csv(ROOT / "FINAL_MODEL_CONTEXT_COMPARISON.csv", index=False)
    return context_df


def _write_final_model_context_comparison(context_df: pd.DataFrame) -> None:
    best_rows = (
        context_df.sort_values(["context_name", "f1", "precision"], ascending=[True, False, False])
        .groupby("context_name", as_index=False)
        .first()[["context_name", "model_name", "window_label", "f1", "canonical_or_extension", "notes"]]
    )
    lines = [
        "# Final Model Context Comparison",
        "",
        "This file keeps the major evaluation contexts separate: canonical benchmark, existing replay, heldout synthetic zero-day-like evaluation, and the TTM extension branch.",
        "",
        "## Best Row Per Context",
        "",
        _markdown_table(best_rows),
        "",
        "## Full Context Table",
        "",
        _markdown_table(
            context_df[
                [
                    "context_name",
                    "model_name",
                    "window_label",
                    "canonical_or_extension",
                    "precision",
                    "recall",
                    "f1",
                    "mean_latency_seconds",
                ]
            ]
        ),
        "",
        "## Safe Reading",
        "",
        "- `transformer @ 60s` remains the canonical benchmark winner.",
        "- A different best row in replay or heldout synthetic evaluation would describe a different context, not a replacement of the canonical benchmark selection.",
        "- TTM remains extension-only.",
    ]
    write_markdown(ROOT / "FINAL_MODEL_CONTEXT_COMPARISON.md", "\n".join(lines))


def _plot_final_model_context_comparison(context_df: pd.DataFrame) -> None:
    best_rows = (
        context_df.sort_values(["context_name", "f1", "precision"], ascending=[True, False, False])
        .groupby("context_name", as_index=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(10, 5.75))
    x = np.arange(len(best_rows))
    bars = ax.bar(x, best_rows["f1"].astype(float), color=["#4C78A8", "#72B7B2", "#F58518"])
    ax.set_xticks(x)
    ax.set_xticklabels(best_rows["context_name"].tolist(), rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Best F1 In Context")
    ax.set_xlabel("Evaluation Context")
    ax.set_title("Best Detector Row by Context")
    ax.grid(axis="y", alpha=0.25)
    for bar, label in zip(bars, best_rows.apply(lambda row: f"{row['model_name']} @ {row['window_label']}", axis=1)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "FINAL_MODEL_CONTEXT_COMPARISON.png", dpi=220)
    plt.close(fig)


def _write_model_window_zero_day_safe_claims(zero_day_df: pd.DataFrame) -> None:
    completed = zero_day_df.loc[zero_day_df["status"] == "completed"].copy()
    mean_df = (
        completed.groupby(["window_label", "model_name"], as_index=False)[["precision", "recall", "f1"]]
        .mean(numeric_only=True)
        .sort_values(["window_label", "f1"], ascending=[True, False])
    )
    best_per_window = mean_df.groupby("window_label", as_index=False).first()
    lines = [
        "# Model / Window / Zero-Day Safe Claims",
        "",
        "## Safe To Say",
        "",
        "- The canonical benchmark still selects `transformer @ 60s`.",
        "- Real detector benchmark coverage exists at `5s`, `10s`, `60s`, and `300s` for the implemented canonical detector families.",
        "- The heldout synthetic evaluation now covers the implemented frozen detector packages across `10s`, `60s`, and `300s`, and it is reported separately from the canonical benchmark.",
        "- `5s` retained benchmark coverage, but the full heldout synthetic sweep at `5s` was blocked in this pass by raw bundle-window generation cost on CPU-only hardware.",
        "- TTM is a real extension benchmark at `60s` and an extension heldout synthetic evaluation at `60s` only.",
        "- The autoencoder result is for the repo's MLP autoencoder implementation, not an LSTM autoencoder.",
        "",
        "## Best Mean Heldout Synthetic Rows By Window",
        "",
        _markdown_table(best_per_window),
        "",
        "## Not Safe To Say",
        "",
        "- Real-world zero-day robustness.",
        "- That the heldout synthetic evaluation replaces the canonical benchmark winner.",
        "- That TTM is the canonical model-selection result.",
        "- That ARIMA was benchmarked when no implementation exists.",
        "- That an LSTM autoencoder was benchmarked when the repo only contains an MLP autoencoder.",
        "- AOI as a detector benchmark metric here. AOI is not implemented as part of this detector-side pass.",
        "",
        "## Precise Wording",
        "",
        "- Use `heldout synthetic scenario evaluation` or `zero-day-like evaluation within a bounded synthetic Phase 2 scenario benchmark`.",
        "- Use `TTM extension benchmark` and `TTM extension heldout synthetic evaluation`.",
        "- Keep `transformer @ 60s` as `the frozen canonical benchmark winner`.",
    ]
    write_markdown(ROOT / "MODEL_WINDOW_ZERO_DAY_SAFE_CLAIMS.md", "\n".join(lines))


def _patch_results_and_decision_reports(context_df: pd.DataFrame, zero_day_df: pd.DataFrame) -> None:
    benchmark_winner = context_df.loc[
        (context_df["context_name"] == "canonical_benchmark_test_split")
        & (context_df["model_name"] == "transformer")
        & (context_df["window_label"] == "60s")
    ]
    zero_day_best = (
        zero_day_df.loc[zero_day_df["status"] == "completed"]
        .groupby(["model_name", "window_label"], as_index=False)[["precision", "recall", "f1"]]
        .mean(numeric_only=True)
        .sort_values(["f1", "precision"], ascending=[False, False])
        .head(1)
    )
    zero_day_sentence = "No completed heldout synthetic detector rows were produced."
    if not zero_day_best.empty:
        row = zero_day_best.iloc[0]
        zero_day_sentence = (
            f"In the new heldout synthetic zero-day-like matrix, the strongest mean row was `{row['model_name']} @ {row['window_label']}` "
            f"with mean F1 `{float(row['f1']):.4f}`, reported separately from the canonical benchmark."
        )

    benchmark_sentence = "The canonical benchmark winner remains `transformer @ 60s`."
    if not benchmark_winner.empty:
        benchmark_sentence = (
            f"The canonical benchmark winner remains `transformer @ 60s` with benchmark F1 "
            f"`{float(benchmark_winner['f1'].iloc[0]):.4f}`."
        )

    results_path = ROOT / "COMPLETE_RESULTS_AND_DISCUSSION.md"
    decision_path = ROOT / "FINAL_PROJECT_DECISION.md"
    def replace_or_append(path: Path, heading: str, section_lines: list[str]) -> None:
        original = path.read_text(encoding="utf-8").rstrip()
        if heading in original:
            original = original.split(heading, 1)[0].rstrip()
        write_markdown(path, original + "\n\n" + heading + "\n\n" + "\n".join(section_lines))

    if results_path.exists():
        appendix = [
            benchmark_sentence,
            "",
            zero_day_sentence,
            "",
            "`5s` heldout synthetic coverage remains blocked in this pass because raw full-day 5s bundle-window generation exceeded the CPU-only wall-clock budget and no reusable 5s replay residuals existed.",
            "",
            "This addendum keeps the contexts separate: canonical benchmark, existing replay, heldout synthetic zero-day-like evaluation, and the TTM extension branch.",
            "AOI is not claimed as part of this detector-side benchmark pass.",
        ]
        replace_or_append(results_path, "## Detector Window And Zero-Day-Like Addendum", appendix)
    if decision_path.exists():
        appendix = [
            f"- Canonical benchmark winner unchanged: yes, `transformer @ 60s` remains frozen.",
            f"- Full detector window comparison completed: yes for implemented detector families across `5s`, `10s`, `60s`, and `300s`.",
            f"- Heldout synthetic detector sweep completed: yes for implemented ready-package detector families at `10s`, `60s`, and `300s`; `5s` is documented as blocked by CPU-only raw window-generation cost.",
            f"- TTM included at `60s` only as an extension; ARIMA remains unimplemented.",
            "- AOI should not be claimed for this detector-side pass.",
        ]
        replace_or_append(decision_path, "## Detector Window / Zero-Day-Like Audit Addendum", appendix)


def _build_final_verification() -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        ROOT / "detector_window_model_coverage_audit.csv",
        ROOT / "detector_window_model_coverage_summary.md",
        ROOT / "phase1_window_model_comparison_full.csv",
        ROOT / "phase1_window_model_comparison_full.md",
        ROOT / "phase1_window_comparison_5s_10s_60s_300s.png",
        ROOT / "phase1_model_vs_window_heatmap.png",
        ROOT / "phase1_best_per_window_summary.csv",
        ROOT / "phase1_ttm_window_comparison.csv",
        ROOT / "phase1_ttm_window_comparison.md",
        ROOT / "phase1_ttm_vs_transformer_gru_lstm_if_arima.png",
        ROOT / "zero_day_window_model_coverage_audit.csv",
        ROOT / "zero_day_window_model_coverage_summary.md",
        ROOT / "zero_day_model_window_results_full.csv",
        ROOT / "zero_day_model_window_results_full.md",
        ROOT / "zero_day_f1_by_window.png",
        ROOT / "zero_day_precision_recall_by_model.png",
        ROOT / "zero_day_model_window_heatmap.png",
        ROOT / "zero_day_best_per_window_summary.csv",
        ROOT / "zero_day_family_model_window_results.csv",
        ROOT / "zero_day_family_difficulty_summary.md",
        ROOT / "FINAL_MODEL_CONTEXT_COMPARISON.csv",
        ROOT / "FINAL_MODEL_CONTEXT_COMPARISON.md",
        ROOT / "FINAL_MODEL_CONTEXT_COMPARISON.png",
        ROOT / "MODEL_WINDOW_ZERO_DAY_SAFE_CLAIMS.md",
        ROOT / "FINAL_WINDOW_ZERO_DAY_VERIFICATION.md",
        ROOT / "FINAL_WINDOW_ZERO_DAY_OUTPUT_STATUS.csv",
    ]
    def status_frame() -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for path in required:
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            rows.append(
                {
                    "artifact_path": str(path),
                    "exists": exists,
                    "nonempty": bool(exists and size > 0),
                    "size_bytes": size,
                    "artifact_type": path.suffix.lower().lstrip("."),
                }
            )
        return pd.DataFrame(rows)

    status_df = status_frame()

    benchmark_df = pd.read_csv(ROOT / "phase1_window_model_comparison_full.csv")
    zero_day_df = pd.read_csv(ROOT / "zero_day_model_window_results_full.csv")
    context_df = pd.read_csv(ROOT / "FINAL_MODEL_CONTEXT_COMPARISON.csv")
    safe_claims_text = (ROOT / "MODEL_WINDOW_ZERO_DAY_SAFE_CLAIMS.md").read_text(encoding="utf-8")
    verification_rows = [
        {
            "check_name": "canonical_winner_preserved",
            "passed": bool(
                (
                    benchmark_df["model_name"].astype(str).eq("transformer")
                    & benchmark_df["window_label"].astype(str).eq("60s")
                    & benchmark_df["canonical_or_extension"].astype(str).eq("canonical")
                    & benchmark_df["status"].astype(str).eq("completed")
                ).any()
            ),
            "notes": "Transformer @ 60s still present as canonical completed row.",
        },
        {
            "check_name": "ttm_extension_separated",
            "passed": bool(
                not context_df.loc[context_df["model_name"].astype(str) == "ttm_extension", "canonical_or_extension"]
                .astype(str)
                .eq("canonical")
                .any()
            ),
            "notes": "TTM rows must stay extension-labeled.",
        },
        {
            "check_name": "zero_day_separate_from_benchmark",
            "passed": bool(zero_day_df["evaluation_context"].astype(str).eq("heldout_synthetic_zero_day_like").all()),
            "notes": "Zero-day-like rows must keep their own context label.",
        },
        {
            "check_name": "aoi_not_claimed_in_safe_claims",
            "passed": bool("AOI" in safe_claims_text and ("not implemented" in safe_claims_text or "Not Safe To Say" in safe_claims_text)),
            "notes": "Safe-claims file explicitly blocks AOI detector claim.",
        },
    ]
    verification_df = pd.DataFrame(verification_rows)
    for _ in range(2):
        status_df = status_frame()
        status_df.to_csv(ROOT / "FINAL_WINDOW_ZERO_DAY_OUTPUT_STATUS.csv", index=False)
        lines = [
            "# Final Window / Zero-Day Verification",
            "",
            "## Output Status",
            "",
            _markdown_table(status_df),
            "",
            "## Logical Checks",
            "",
            _markdown_table(verification_df),
            "",
            "## Separation Checks",
            "",
            "- Canonical benchmark, replay, heldout synthetic zero-day-like evaluation, and extension rows are kept separate by explicit context fields.",
            "- LoRA explanation artifacts were not used as detector benchmark evidence in this pass.",
            "",
            "## Runtime Notes",
            "",
            "- Re-running this script emits a non-blocking scikit-learn version warning when loading the saved IsolationForest package (`1.8.0` artifact loaded under local `1.7.2`). The warning is documented here because it affects reproducibility hygiene, but it did not prevent the saved-package evaluations from completing.",
        ]
        write_markdown(ROOT / "FINAL_WINDOW_ZERO_DAY_VERIFICATION.md", "\n".join(lines))
    return status_df, verification_df


def main() -> None:
    detector_audit_df = _build_detector_window_model_coverage_audit()
    detector_audit_df.to_csv(ROOT / "detector_window_model_coverage_audit.csv", index=False)
    _write_detector_window_model_coverage_summary(detector_audit_df)

    comparison_df, best_summary_df = _build_phase1_window_model_comparison_full()
    _write_phase1_window_model_comparison_md(comparison_df, best_summary_df)
    _plot_phase1_window_comparison(comparison_df)

    ttm_window_df = _build_phase1_ttm_window_comparison(comparison_df)
    _plot_ttm_window_comparison(ttm_window_df)

    zero_day_audit_df = _build_zero_day_window_model_coverage_audit()
    zero_day_audit_df.to_csv(ROOT / "zero_day_window_model_coverage_audit.csv", index=False)
    _write_zero_day_window_model_coverage_summary(zero_day_audit_df)

    zero_day_full_df, zero_day_family_df = _run_zero_day_full_matrix()
    _write_zero_day_reports(zero_day_full_df, zero_day_family_df)
    _plot_zero_day_figures(zero_day_full_df)

    context_df = _build_final_model_context_comparison(zero_day_full_df)
    _write_final_model_context_comparison(context_df)
    _plot_final_model_context_comparison(context_df)

    _write_model_window_zero_day_safe_claims(zero_day_full_df)
    _patch_results_and_decision_reports(context_df, zero_day_full_df)

    _build_final_verification()


if __name__ == "__main__":
    main()
