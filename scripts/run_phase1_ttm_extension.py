"""Repository orchestration script for DERGuardian.

This script runs or rebuilds run phase1 ttm extension artifacts for audits, figures,
reports, or reproducibility checks. It is release-support code and must preserve
the separation between canonical benchmark, replay, heldout synthetic, and
extension experiment contexts. TTM outputs are extension-only and do not replace the canonical benchmark winner.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
import sys

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
from torch.utils.data import DataLoader, TensorDataset
from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction

from phase1_models.feature_builder import fit_standardizer, transform_features
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload, detection_latency_table


SEED = 1729
WINDOW_LABEL = "60s"
CONTEXT_LENGTH = 12
PREDICTION_LENGTH = 1
BATCH_SIZE = 128
MAX_EPOCHS = 24
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

RESIDUAL_PATH = ROOT / "outputs" / "window_size_study" / WINDOW_LABEL / "residual_windows.parquet"
FEATURE_COLUMNS_PATH = (
    ROOT / "outputs" / "window_size_study" / WINDOW_LABEL / "ready_packages" / "transformer" / "feature_columns.json"
)
ATTACK_LABELS_PATH = ROOT / "outputs" / "attacked" / "attack_labels.parquet"
FINAL_COMPARISON_PATH = ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv"
MODEL_SUMMARY_GLOB = ROOT / "outputs" / "window_size_study"

ARTIFACT_ROOT = ROOT / "outputs" / "phase1_ttm_extension"
CHECKPOINT_PATH = ARTIFACT_ROOT / "ttm_extension_checkpoint.pt"
PREDICTIONS_PATH = ARTIFACT_ROOT / "ttm_extension_predictions.parquet"
TRAINING_HISTORY_PATH = ARTIFACT_ROOT / "ttm_extension_training_history.csv"
LATENCY_PATH = ARTIFACT_ROOT / "ttm_extension_latency.csv"
CURVES_PATH = ARTIFACT_ROOT / "ttm_extension_curves.json"

CONFIG_PATH = ROOT / "phase1_ttm_extension_config.yaml"
RESULTS_PATH = ROOT / "phase1_ttm_results.csv"
REPORT_PATH = ROOT / "phase1_ttm_eval_report.md"
COMPARE_PATH = ROOT / "phase1_ttm_vs_all_models.csv"
FIG_MODEL_COMPARISON = ROOT / "phase1_model_comparison_extended.png"
FIG_TRADEOFF = ROOT / "phase1_accuracy_latency_tradeoff_extended.png"


def set_seed(seed: int = SEED) -> None:
    """Handle set seed within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calibrate_scores(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, LogisticRegression | None]:
    """Handle calibrate scores within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return scores, None
    calibrator = LogisticRegression(random_state=SEED, solver="lbfgs")
    calibrator.fit(scores.reshape(-1, 1), y_true)
    calibrated = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    return calibrated.astype(float), calibrator


def apply_calibrator(calibrator: LogisticRegression | None, scores: np.ndarray) -> np.ndarray:
    """Handle apply calibrator within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    scores = np.asarray(scores, dtype=float)
    if calibrator is None:
        return scores
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1].astype(float)


def choose_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, dict[str, object]]:
    """Handle choose best threshold within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if scores.size == 0:
        return 0.5, compute_binary_metrics(y_true, scores, 0.5)
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.995, 60)))
    if len(candidate_thresholds) == 0:
        candidate_thresholds = np.array([0.5], dtype=float)
    best_threshold = float(candidate_thresholds[0])
    best_metrics: dict[str, object] | None = None
    for threshold in candidate_thresholds:
        metrics = compute_binary_metrics(y_true, scores, float(threshold))
        if best_metrics is None:
            best_metrics = metrics
            best_threshold = float(threshold)
            continue
        old = float(best_metrics["f1"])
        new = float(metrics["f1"])
        if new > old + 1e-9:
            best_metrics = metrics
            best_threshold = float(threshold)
        elif math.isclose(new, old, rel_tol=0.0, abs_tol=1e-9) and float(metrics["precision"]) > float(
            best_metrics["precision"]
        ):
            best_metrics = metrics
            best_threshold = float(threshold)
    assert best_metrics is not None
    return best_threshold, best_metrics


def build_forecasting_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    context_length: int,
    prediction_length: int,
    standardizer,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build forecasting sequences for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    ordered = frame.sort_values("window_start_utc").reset_index(drop=True).copy()
    if ordered.empty:
        return (
            np.empty((0, context_length, len(feature_columns)), dtype=np.float32),
            np.empty((0, prediction_length, len(feature_columns)), dtype=np.float32),
            pd.DataFrame(),
        )
    ordered["window_start_utc"] = pd.to_datetime(ordered["window_start_utc"], utc=True)
    step_seconds = int(ordered["window_start_utc"].diff().dt.total_seconds().dropna().mode().iloc[0]) if len(ordered) > 1 else 60
    segment_break = ordered["window_start_utc"].diff().dt.total_seconds().fillna(step_seconds).ne(step_seconds)
    ordered["segment_id"] = segment_break.cumsum().astype(int)

    sequences_x: list[np.ndarray] = []
    sequences_y: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for _, segment in ordered.groupby("segment_id", observed=False):
        if len(segment) < context_length + prediction_length:
            continue
        feature_matrix = transform_features(segment, feature_columns, standardizer).astype(np.float32)
        for future_start in range(context_length, len(segment) - prediction_length + 1):
            future_end = future_start + prediction_length
            target_row = segment.iloc[future_end - 1]
            sequences_x.append(feature_matrix[future_start - context_length : future_start])
            sequences_y.append(feature_matrix[future_start:future_end])
            rows.append(
                {
                    "window_start_utc": target_row["window_start_utc"],
                    "window_end_utc": target_row["window_end_utc"],
                    "scenario_id": str(target_row.get("scenario_id", target_row.get("scenario_window_id", ""))),
                    "attack_present": int(target_row["attack_present"]),
                    "attack_family": str(target_row.get("attack_family", "benign")),
                    "attack_severity": str(target_row.get("attack_severity", "none")),
                    "attack_affected_assets": str(target_row.get("attack_affected_assets", "[]")),
                    "split_name": str(target_row.get("split_name", "")),
                }
            )
    return np.asarray(sequences_x, dtype=np.float32), np.asarray(sequences_y, dtype=np.float32), pd.DataFrame(rows)


def evaluate_forecaster(
    model: TinyTimeMixerForPrediction,
    x_values: np.ndarray,
    y_values: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Handle evaluate forecaster within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if len(x_values) == 0:
        return np.array([], dtype=float), 0.0
    dataset = TensorDataset(torch.from_numpy(x_values), torch.from_numpy(y_values))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
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


def train_ttm(
    model: TinyTimeMixerForPrediction,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> tuple[TinyTimeMixerForPrediction, list[dict[str, float]], float]:
    """Handle train ttm within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_left = PATIENCE
    training_start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(past_values=batch_x, future_values=batch_y)
            loss = output.loss
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(past_values=batch_x, future_values=batch_y)
                val_losses.append(float(output.loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    training_time = time.perf_counter() - training_start
    return model, history, training_time


def load_all_window_model_rows() -> pd.DataFrame:
    """Load all window model rows for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    rows: list[pd.DataFrame] = []
    for model_summary_path in sorted(MODEL_SUMMARY_GLOB.glob("*/reports/model_summary.csv")):
        frame = pd.read_csv(model_summary_path)
        frame["source_file"] = str(model_summary_path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_comparison_table(ttm_row: dict[str, object]) -> pd.DataFrame:
    """Build comparison table for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    full_model_rows = load_all_window_model_rows()
    comparison_rows: list[dict[str, object]] = []
    if not full_model_rows.empty:
        full_model_rows["comparison_role"] = "family_best"
        best_per_model = (
            full_model_rows.sort_values(["f1", "average_precision", "precision"], ascending=False)
            .groupby("model_name", as_index=False)
            .first()
        )
        comparison_rows.extend(best_per_model.to_dict(orient="records"))

    final_window_winners = pd.read_csv(FINAL_COMPARISON_PATH)
    if not final_window_winners.empty:
        final_window_winners = final_window_winners.copy()
        final_window_winners["comparison_role"] = "window_winner"
        comparison_rows.extend(final_window_winners.to_dict(orient="records"))

    comparison_rows.append(
        {
            "window_label": WINDOW_LABEL,
            "window_seconds": 60,
            "step_seconds": 12,
            "model_name": "ttm_extension",
            "status": "completed",
            "precision": ttm_row["precision"],
            "recall": ttm_row["recall"],
            "f1": ttm_row["f1"],
            "roc_auc": ttm_row["roc_auc"],
            "average_precision": ttm_row["average_precision"],
            "macro_precision": ttm_row["macro_precision"],
            "macro_recall": ttm_row["macro_recall"],
            "weighted_precision": ttm_row["weighted_precision"],
            "weighted_recall": ttm_row["weighted_recall"],
            "threshold": ttm_row["threshold"],
            "feature_count": ttm_row["feature_count"],
            "seq_len": ttm_row["seq_len"],
            "training_time_seconds": ttm_row["training_time_seconds"],
            "inference_time_seconds": ttm_row["inference_time_seconds"],
            "inference_time_ms_per_prediction": ttm_row["inference_time_ms_per_prediction"],
            "parameter_count": ttm_row["parameter_count"],
            "mean_detection_latency_seconds": ttm_row["mean_detection_latency_seconds"],
            "median_detection_latency_seconds": ttm_row["median_detection_latency_seconds"],
            "scenario_f1_mean": np.nan,
            "scenario_f1_std": np.nan,
            "model_dir": str(ARTIFACT_ROOT),
            "report_dir": str(ROOT),
            "ready_package_dir": "",
            "error": "",
            "comparison_role": "extension",
            "training_supervision": "benign-only_forecasting_anomaly",
            "notes": "Randomly initialized TinyTimeMixer trained on benign residual windows only.",
        }
    )
    comparison_rows.append(
        {
            "window_label": "",
            "window_seconds": np.nan,
            "step_seconds": np.nan,
            "model_name": "arima",
            "status": "not_implemented",
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "average_precision": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "weighted_precision": np.nan,
            "weighted_recall": np.nan,
            "threshold": np.nan,
            "feature_count": np.nan,
            "seq_len": np.nan,
            "training_time_seconds": np.nan,
            "inference_time_seconds": np.nan,
            "inference_time_ms_per_prediction": np.nan,
            "parameter_count": np.nan,
            "mean_detection_latency_seconds": np.nan,
            "median_detection_latency_seconds": np.nan,
            "scenario_f1_mean": np.nan,
            "scenario_f1_std": np.nan,
            "model_dir": "",
            "report_dir": "",
            "ready_package_dir": "",
            "error": "No ARIMA implementation was found in the repository audit.",
            "comparison_role": "not_implemented",
            "training_supervision": "",
            "notes": "Documented blocker, no fabricated metrics.",
        }
    )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.drop_duplicates(subset=["model_name", "window_label", "comparison_role"], keep="first")
    comparison_df = comparison_df.sort_values(
        by=["comparison_role", "f1", "average_precision"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return comparison_df


def plot_comparison_figures(comparison_df: pd.DataFrame) -> None:
    """Handle plot comparison figures within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    family_best = comparison_df[comparison_df["comparison_role"].isin(["family_best", "extension", "not_implemented"])].copy()
    plot_df = family_best[family_best["status"].isin(["completed"])].copy()
    plot_df["label"] = plot_df.apply(
        lambda row: "TTM extension @ 60s"
        if row["model_name"] == "ttm_extension"
        else f"{row['model_name']} @ {row['window_label']}",
        axis=1,
    )
    plot_df = plot_df.sort_values("f1", ascending=False)

    plt.figure(figsize=(11, 5.5))
    colors = ["#2f5597" if name != "ttm_extension" else "#b55d1f" for name in plot_df["model_name"]]
    plt.bar(plot_df["label"], plot_df["f1"], color=colors)
    plt.ylabel("F1")
    plt.xlabel("Model / context")
    plt.title("Extended Model Comparison Including TinyTimeMixer Extension")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, max(0.9, float(plot_df["f1"].max()) + 0.08))
    for idx, value in enumerate(plot_df["f1"]):
        plt.text(idx, float(value) + 0.015, f"{float(value):.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_MODEL_COMPARISON, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5.5))
    for _, row in plot_df.iterrows():
        plt.scatter(
            float(row["inference_time_ms_per_prediction"]),
            float(row["f1"]),
            s=120 if row["model_name"] == "ttm_extension" else 90,
            color="#b55d1f" if row["model_name"] == "ttm_extension" else "#2f5597",
        )
        plt.annotate(
            row["label"],
            (float(row["inference_time_ms_per_prediction"]), float(row["f1"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    plt.xlabel("Inference time (ms / prediction)")
    plt.ylabel("F1")
    plt.title("Accuracy / Latency Trade-off With TinyTimeMixer Extension")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIG_TRADEOFF, dpi=200)
    plt.close()


def write_report(
    ttm_row: dict[str, object],
    comparison_df: pd.DataFrame,
    history_df: pd.DataFrame,
    split_summary: dict[str, int],
    training_note: str,
) -> None:
    """Write report for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    family_best = comparison_df[comparison_df["comparison_role"] == "family_best"].copy()
    family_best = family_best[family_best["status"] == "completed"].sort_values("f1", ascending=False)
    top_family = family_best.head(6)[["model_name", "window_label", "precision", "recall", "f1", "average_precision"]]

    lines = [
        "# Phase 1 TinyTimeMixer Extension Evaluation",
        "",
        "## Scope",
        "- This branch is an extension benchmark only.",
        "- The frozen canonical benchmark winner remains `transformer @ 60s` from `outputs/window_size_study/final_window_comparison.csv`.",
        "- The TTM result below must not be relabeled as the canonical benchmark winner.",
        "",
        "## Implementation Reality Check",
        f"- Architecture: `TinyTimeMixerForPrediction` from IBM Granite TSFM / `tsfm_public`.",
        f"- Data contract: canonical `60s` aligned residual windows using the frozen transformer feature subset (`{int(ttm_row['feature_count'])}` features).",
        f"- Training mode: benign-only forecasting on train split, validation-threshold calibration on heldout validation windows, test metrics on heldout test windows.",
        f"- Context / forecast horizon: `{int(ttm_row['seq_len'])}` input windows -> `{PREDICTION_LENGTH}` predicted window.",
        f"- {training_note}",
        "",
        "## Split Summary",
        f"- Train benign forecasting sequences: `{split_summary['train_sequences']}`",
        f"- Validation sequences: `{split_summary['val_sequences']}` (`{split_summary['val_positive_sequences']}` positive)",
        f"- Test sequences: `{split_summary['test_sequences']}` (`{split_summary['test_positive_sequences']}` positive)",
        "",
        "## TTM Extension Result",
        f"- Precision: `{float(ttm_row['precision']):.6f}`",
        f"- Recall: `{float(ttm_row['recall']):.6f}`",
        f"- F1: `{float(ttm_row['f1']):.6f}`",
        f"- PR-AUC: `{float(ttm_row['average_precision']):.6f}`",
        f"- ROC-AUC: `{float(ttm_row['roc_auc']):.6f}`" if pd.notna(ttm_row["roc_auc"]) else "- ROC-AUC: `n/a`",
        f"- Threshold after validation calibration: `{float(ttm_row['threshold']):.6f}`",
        f"- Parameter count: `{int(ttm_row['parameter_count'])}`",
        f"- Training time: `{float(ttm_row['training_time_seconds']):.2f}` s",
        f"- Inference latency: `{float(ttm_row['inference_time_ms_per_prediction']):.6f}` ms / prediction",
        f"- Mean detected attack latency: `{float(ttm_row['mean_detection_latency_seconds']):.3f}` s"
        if pd.notna(ttm_row["mean_detection_latency_seconds"])
        else "- Mean detected attack latency: `n/a`",
        "",
        "## Comparison Snapshot",
        top_family.to_markdown(index=False),
        "",
        "## Interpretation",
        "- This is a real TTM run, not a placeholder manifest.",
        "- It should be read as a secondary architectural comparison on the canonical residual input contract.",
        "- Because no repo-native public TTM checkpoint matched the short `12 -> 1` residual-window setup, the run used a randomly initialized tiny TTM and trained it locally.",
        "- If this branch underperforms the frozen canonical detector, that is evidence, not failure to report.",
        "",
        "## Training History",
        history_df.to_markdown(index=False),
        "",
        "## Output Files",
        f"- Results: `{RESULTS_PATH.name}`",
        f"- Comparison table: `{COMPARE_PATH.name}`",
        f"- Figures: `{FIG_MODEL_COMPARISON.name}`, `{FIG_TRADEOFF.name}`",
        f"- Raw artifacts: `{ARTIFACT_ROOT}`",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the command-line entrypoint for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    set_seed()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    residual_df = pd.read_parquet(RESIDUAL_PATH)
    residual_df["window_start_utc"] = pd.to_datetime(residual_df["window_start_utc"], utc=True)
    residual_df["window_end_utc"] = pd.to_datetime(residual_df["window_end_utc"], utc=True)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))

    train_benign = residual_df[(residual_df["split_name"] == "train") & (residual_df["attack_present"] == 0)].copy()
    val_all = residual_df[residual_df["split_name"] == "val"].copy()
    val_benign = val_all[val_all["attack_present"] == 0].copy()
    test_all = residual_df[residual_df["split_name"] == "test"].copy()

    standardizer = fit_standardizer(train_benign, feature_columns)
    x_train, y_train, _ = build_forecasting_sequences(
        train_benign, feature_columns, CONTEXT_LENGTH, PREDICTION_LENGTH, standardizer
    )
    x_val_benign, y_val_benign, _ = build_forecasting_sequences(
        val_benign, feature_columns, CONTEXT_LENGTH, PREDICTION_LENGTH, standardizer
    )
    x_val_all, y_val_all, val_meta = build_forecasting_sequences(
        val_all, feature_columns, CONTEXT_LENGTH, PREDICTION_LENGTH, standardizer
    )
    x_test_all, y_test_all, test_meta = build_forecasting_sequences(
        test_all, feature_columns, CONTEXT_LENGTH, PREDICTION_LENGTH, standardizer
    )

    if len(x_train) == 0 or len(x_val_benign) == 0 or len(x_val_all) == 0 or len(x_test_all) == 0:
        raise RuntimeError("TTM extension could not build non-empty forecasting splits from the canonical 60 s residual dataset.")

    device = torch.device("cpu")
    model_config = TinyTimeMixerConfig(
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        patch_length=2,
        patch_stride=2,
        num_input_channels=len(feature_columns),
        d_model=16,
        decoder_d_model=16,
        num_layers=3,
        decoder_num_layers=2,
        dropout=0.2,
        head_dropout=0.2,
        loss="mse",
        scaling="std",
        adaptive_patching_levels=0,
        self_attn=False,
        enable_forecast_channel_mixing=False,
    )
    model = TinyTimeMixerForPrediction(model_config).to(device)

    model, history, training_time = train_ttm(model, x_train, y_train, x_val_benign, y_val_benign, device)
    torch.save(model.state_dict(), CHECKPOINT_PATH)

    val_scores_raw, val_inference_time = evaluate_forecaster(model, x_val_all, y_val_all, device)
    test_scores_raw, test_inference_time = evaluate_forecaster(model, x_test_all, y_test_all, device)

    val_labels = val_meta["attack_present"].astype(int).to_numpy()
    test_labels = test_meta["attack_present"].astype(int).to_numpy()
    calibrated_val, calibrator = calibrate_scores(val_scores_raw, val_labels)
    threshold, _ = choose_best_threshold(calibrated_val, val_labels)
    calibrated_test = apply_calibrator(calibrator, test_scores_raw)

    predictions = test_meta.copy()
    predictions["score"] = calibrated_test
    predictions["predicted"] = (predictions["score"] >= threshold).astype(int)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)

    metrics = compute_binary_metrics(test_labels, calibrated_test, threshold)
    curves = compute_curve_payload(test_labels, calibrated_test)
    CURVES_PATH.write_text(json.dumps(curves, indent=2), encoding="utf-8")

    labels_df = pd.read_parquet(ATTACK_LABELS_PATH)
    latency_df = detection_latency_table(predictions, labels_df, "ttm_extension")
    latency_df.to_csv(LATENCY_PATH, index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(TRAINING_HISTORY_PATH, index=False)

    mean_latency = float(latency_df["latency_seconds"].dropna().mean()) if latency_df["latency_seconds"].notna().any() else np.nan
    median_latency = (
        float(latency_df["latency_seconds"].dropna().median()) if latency_df["latency_seconds"].notna().any() else np.nan
    )
    inference_ms = (test_inference_time / max(len(test_meta), 1)) * 1000.0
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    model_size_bytes = CHECKPOINT_PATH.stat().st_size

    results_row = {
        "model_name": "ttm_extension",
        "window_label": WINDOW_LABEL,
        "window_seconds": 60,
        "step_seconds": 12,
        "status": "completed",
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]) if metrics["roc_auc"] is not None else np.nan,
        "average_precision": float(metrics["average_precision"]) if metrics["average_precision"] is not None else np.nan,
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "weighted_precision": float(metrics["weighted_precision"]),
        "weighted_recall": float(metrics["weighted_recall"]),
        "threshold": float(threshold),
        "feature_count": int(len(feature_columns)),
        "seq_len": int(CONTEXT_LENGTH),
        "prediction_length": int(PREDICTION_LENGTH),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(test_inference_time),
        "inference_time_ms_per_prediction": float(inference_ms),
        "validation_inference_time_seconds": float(val_inference_time),
        "parameter_count": parameter_count,
        "model_size_bytes": int(model_size_bytes),
        "mean_detection_latency_seconds": mean_latency,
        "median_detection_latency_seconds": median_latency,
        "train_sequences": int(len(x_train)),
        "val_sequences": int(len(x_val_all)),
        "test_sequences": int(len(x_test_all)),
        "val_positive_sequences": int(val_labels.sum()),
        "test_positive_sequences": int(test_labels.sum()),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "predictions_path": str(PREDICTIONS_PATH),
        "notes": (
            "Randomly initialized TinyTimeMixer forecast-error detector over canonical 60 s residual windows. "
            "Extension branch only; canonical transformer @ 60 s remains the benchmark winner."
        ),
    }
    pd.DataFrame([results_row]).to_csv(RESULTS_PATH, index=False)

    comparison_df = build_comparison_table(results_row)
    comparison_df.to_csv(COMPARE_PATH, index=False)
    plot_comparison_figures(comparison_df)

    config_payload = {
        "experiment_name": "phase1_ttm_extension",
        "branch_type": "extension_benchmark_only",
        "canonical_status": {
            "canonical_benchmark_winner": "transformer @ 60s",
            "canonical_source_of_truth": str(FINAL_COMPARISON_PATH),
            "overwritten": False,
        },
        "data": {
            "residual_path": str(RESIDUAL_PATH),
            "feature_columns_source": str(FEATURE_COLUMNS_PATH),
            "feature_count": int(len(feature_columns)),
            "split_policy": "existing scenario-aware split_name from canonical residual dataset",
            "train_subset": "train benign only",
            "validation_subset": "all val windows for threshold calibration; benign val subset for early stopping",
            "test_subset": "all test windows",
        },
        "model": {
            "implementation": "tsfm_public.TinyTimeMixerForPrediction",
            "initialization": "random_init_local",
            "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "patch_length": 2,
            "patch_stride": 2,
            "num_input_channels": int(len(feature_columns)),
            "d_model": 16,
            "decoder_d_model": 16,
            "num_layers": 3,
            "decoder_num_layers": 2,
            "dropout": 0.2,
            "head_dropout": 0.2,
            "loss": "mse",
            "scaling": "std",
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "device": str(device),
            "seed": SEED,
        },
        "notes": [
            "The public IBM Granite TTM package was installed locally from source.",
            "No compatible repo-native public pre-trained TTM checkpoint matched the short 12->1 residual-window contract, so this run uses a locally trained randomly initialized tiny TTM.",
            "This output is extension evidence only and does not replace the canonical benchmark winner.",
        ],
    }
    CONFIG_PATH.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    split_summary = {
        "train_sequences": int(len(x_train)),
        "val_sequences": int(len(x_val_all)),
        "test_sequences": int(len(x_test_all)),
        "val_positive_sequences": int(val_labels.sum()),
        "test_positive_sequences": int(test_labels.sum()),
    }
    training_note = (
        "A compatible public pre-trained TTM checkpoint for this short residual-window contract was not available, "
        "so the architecture was trained locally from random initialization."
    )
    write_report(results_row, comparison_df, history_df, split_summary, training_note)


if __name__ == "__main__":
    main()
