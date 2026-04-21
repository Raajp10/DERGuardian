from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
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
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

from common.io_utils import read_dataframe
from phase1_models.context.phase1_context_builder import build_and_persist_context_summaries
from phase1_models.context.phase1_fusion import run_fusion_suite
from phase1_models.context.phase1_llm_reasoner import build_and_persist_reasoning_outputs
from phase1_models.feature_builder import fit_discretizer, fit_standardizer, transform_features, transform_to_tokens
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload, detection_latency_table, per_scenario_metrics
from phase1_models.model_utils import (
    apply_model_display_names,
    CANONICAL_ARTIFACT_ROOT,
    CANONICAL_MODEL_ROOT,
    display_model_name,
    ensure_model_paths,
    write_json,
    write_pickle,
    write_predictions,
    write_torch_state,
)
from phase1_models.neural_models import GRUClassifier, LSTMClassifier, MLPAutoencoder, TinyTransformerClassifier, TokenBaselineClassifier
from phase1_models.neural_training import predict_autoencoder_errors, predict_classifier_scores, train_autoencoder, train_classifier
from phase1_models.ready_package_utils import (
    architecture_from_model,
    build_input_schema_summary,
    build_split_summary,
    discretizer_bin_count,
    export_ready_model_package,
    serialize_calibration,
)
from phase1_models.residual_dataset import (
    build_aligned_residual_dataframe,
    canonical_residual_artifact_path,
    canonical_window_source_paths,
    load_persisted_residual_dataframe,
    persist_residual_dataframe,
    read_window_sources,
    residual_artifact_is_fresh,
    residual_artifact_is_usable,
)


THRESHOLD_LEVELS = {"p95": 0.95, "p99": 0.99, "p99.5": 0.995}
MODEL_PRIORITY = [
    "threshold_baseline",
    "autoencoder",
    "isolation_forest",
    "gru",
    "transformer",
    "lstm",
    "llm_baseline",
]
PRIMARY_FUSION_DETECTOR = "threshold_baseline"
PRIMARY_FUSION_TOKEN_MODEL = "llm_baseline"

@dataclass(slots=True)
class FullRunData:
    residual_df: pd.DataFrame
    labels_df: pd.DataFrame
    split_df: pd.DataFrame
    feature_ranking: pd.DataFrame
    residual_artifact_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Phase 1 DER anomaly-model diagnostics, training, and paper artifact generation.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--feature-counts", default="32,64,96,128")
    parser.add_argument("--seq-lens", default="4,8,12")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--buffer-windows", type=int, default=2)
    parser.add_argument("--window-seconds", type=int, default=300)
    args = parser.parse_args()

    feature_counts = [int(item) for item in args.feature_counts.split(",") if item.strip()]
    seq_lens = [int(item) for item in args.seq_lens.split(",") if item.strip()]
    run_full_benchmark(
        root=Path(args.project_root),
        feature_counts=feature_counts,
        seq_lens=seq_lens,
        epochs=args.epochs,
        patience=args.patience,
        buffer_windows=args.buffer_windows,
        window_seconds=args.window_seconds,
    )


def run_full_benchmark(
    root: Path,
    feature_counts: list[int],
    seq_lens: list[int],
    epochs: int = 24,
    patience: int = 5,
    buffer_windows: int = 2,
    window_seconds: int = 300,
    report_root_name: str = CANONICAL_ARTIFACT_ROOT,
    model_root_name: str = CANONICAL_MODEL_ROOT,
) -> Path:
    report_root = root / "outputs" / "reports" / report_root_name
    report_root.mkdir(parents=True, exist_ok=True)
    artifact_root_name = report_root_name

    full_data = prepare_full_run_data(root, buffer_windows=buffer_windows)
    _write_diagnostic_tables(full_data, report_root, feature_counts, window_seconds=window_seconds)

    model_results: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    per_scenario_frames: list[pd.DataFrame] = []
    latency_frames: list[pd.DataFrame] = []
    figure_manifest: list[dict[str, object]] = []
    table_manifest: list[dict[str, object]] = []
    model_result_map: dict[str, dict[str, object]] = {}

    threshold_result = run_threshold_baseline(
        root,
        full_data,
        feature_counts=feature_counts,
        report_root=report_root,
        model_root_name=model_root_name,
        artifact_root_name=artifact_root_name,
    )
    threshold_rows.extend(threshold_result["sweep_rows"])
    model_results.append(threshold_result["summary"])
    per_scenario_frames.append(threshold_result["per_scenario"])
    latency_frames.append(threshold_result["latency"])
    figure_manifest.extend(threshold_result["figures"])
    model_result_map["threshold_baseline"] = threshold_result

    autoencoder_result = run_autoencoder(
        root,
        full_data,
        feature_counts=feature_counts,
        epochs=epochs,
        patience=patience,
        report_root=report_root,
        model_root_name=model_root_name,
        artifact_root_name=artifact_root_name,
    )
    threshold_rows.extend(autoencoder_result["sweep_rows"])
    model_results.append(autoencoder_result["summary"])
    per_scenario_frames.append(autoencoder_result["per_scenario"])
    latency_frames.append(autoencoder_result["latency"])
    figure_manifest.extend(autoencoder_result["figures"])
    model_result_map["autoencoder"] = autoencoder_result

    isolation_result = run_isolation_forest(
        root,
        full_data,
        feature_counts=feature_counts,
        report_root=report_root,
        model_root_name=model_root_name,
        artifact_root_name=artifact_root_name,
    )
    threshold_rows.extend(isolation_result["sweep_rows"])
    model_results.append(isolation_result["summary"])
    per_scenario_frames.append(isolation_result["per_scenario"])
    latency_frames.append(isolation_result["latency"])
    figure_manifest.extend(isolation_result["figures"])
    model_result_map["isolation_forest"] = isolation_result

    for model_name, ctor in [
        ("gru", lambda input_dim: GRUClassifier(input_dim=input_dim, hidden_dim=64)),
        ("transformer", lambda input_dim: TinyTransformerClassifier(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)),
        ("lstm", lambda input_dim: LSTMClassifier(input_dim=input_dim, hidden_dim=64)),
        ("llm_baseline", lambda _: TokenBaselineClassifier(vocab_size=16, embed_dim=32)),
    ]:
        result = run_sequence_model(
            root,
            full_data,
            model_name=model_name,
            model_ctor=ctor,
            feature_counts=feature_counts,
            seq_lens=seq_lens,
            epochs=epochs,
            patience=patience,
            token_input=model_name == "llm_baseline",
            report_root=report_root,
            model_root_name=model_root_name,
            artifact_root_name=artifact_root_name,
        )
        threshold_rows.extend(result["sweep_rows"])
        model_results.append(result["summary"])
        per_scenario_frames.append(result["per_scenario"])
        latency_frames.append(result["latency"])
        figure_manifest.extend(result["figures"])
        model_result_map[model_name] = result

    context_summaries, context_table, context_paths = build_and_persist_context_summaries(
        root=root,
        top_k=8,
        artifact_root_name=artifact_root_name,
    )
    reasoning_records, reasoning_table, reasoning_paths = build_and_persist_reasoning_outputs(
        root=root,
        context_summaries=context_summaries,
        artifact_root_name=artifact_root_name,
        model_root_name=model_root_name,
    )

    fusion_summary_df = pd.DataFrame()
    fusion_results: dict[str, object] = {}
    detector_result = model_result_map.get(PRIMARY_FUSION_DETECTOR)
    token_result = model_result_map.get(PRIMARY_FUSION_TOKEN_MODEL)
    if detector_result is not None:
        fusion_results, fusion_summary_df = run_fusion_suite(
            detector_model_name=PRIMARY_FUSION_DETECTOR,
            labels_df=full_data.labels_df,
            detector_val=detector_result["validation_predictions"],
            detector_test=detector_result["predictions"],
            reasoning_table=reasoning_table,
            token_val=token_result["validation_predictions"] if token_result is not None else None,
            token_test=token_result["predictions"] if token_result is not None else None,
            output_model_dir=root / "outputs" / model_root_name / "fusion",
            output_report_dir=report_root / "fusion",
            split_summary=build_split_summary(full_data.split_df),
        )
        for mode_name, fusion_result in fusion_results.items():
            figure_manifest.extend(
                generate_model_plots(
                    mode_name,
                    fusion_result.metrics,
                    fusion_result.curves,
                    {"train_loss": [], "val_loss": []},
                    fusion_result.test_predictions,
                    report_root / "fusion" / mode_name,
                )
            )
        fusion_per_scenario = pd.concat([item.per_scenario for item in fusion_results.values()], ignore_index=True)
        fusion_latency = pd.concat([item.latency for item in fusion_results.values()], ignore_index=True)
        per_scenario_frames.append(fusion_per_scenario)
        latency_frames.append(fusion_latency)

    summary_df = pd.DataFrame(model_results)
    summary_df["priority_rank"] = summary_df["model_name"].apply(lambda name: MODEL_PRIORITY.index(name) if name in MODEL_PRIORITY else len(MODEL_PRIORITY))
    summary_df = summary_df.sort_values(["recall", "precision", "f1", "priority_rank"], ascending=[False, False, False, True]).reset_index(drop=True)
    summary_df = summary_df.drop(columns=["priority_rank"])
    per_scenario_df = pd.concat(per_scenario_frames, ignore_index=True) if per_scenario_frames else pd.DataFrame()
    latency_df = pd.concat(latency_frames, ignore_index=True) if latency_frames else pd.DataFrame()
    threshold_df = pd.DataFrame(threshold_rows)

    summary_csv = report_root / "model_summary_table_full.csv"
    summary_md = report_root / "model_summary_table_full.md"
    per_scenario_csv = report_root / "per_scenario_metrics.csv"
    threshold_csv = report_root / "threshold_sweep_table.csv"
    feature_sep_csv = report_root / "feature_separability_table.csv"
    feature_imp_csv = report_root / "feature_importance_table.csv"
    latency_csv = report_root / "detection_latency_analysis_full.csv"
    context_table_path = context_paths[1]
    reasoning_table_path = reasoning_paths[1]
    fusion_csv = report_root / "fusion_ablation_table.csv"

    summary_df.to_csv(summary_csv, index=False)
    summary_md.write_text(_to_markdown(summary_df), encoding="utf-8")
    per_scenario_df.to_csv(per_scenario_csv, index=False)
    threshold_df.to_csv(threshold_csv, index=False)
    full_data.feature_ranking.to_csv(feature_sep_csv, index=False)
    full_data.feature_ranking.head(64).to_csv(feature_imp_csv, index=False)
    latency_df.to_csv(latency_csv, index=False)
    if not fusion_summary_df.empty:
        fusion_summary_df.to_csv(fusion_csv, index=False)

    comparison_path = report_root / "model_comparison_plot_full.png"
    figure_manifest.append({"path": str(_plot_model_comparison(summary_df, comparison_path)), "description": "Full-run model comparison plot."})
    latency_path = report_root / "latency_by_scenario.png"
    figure_manifest.append({"path": str(_plot_latency_by_scenario(latency_df, latency_path)), "description": "Detection latency by scenario and model."})
    if not fusion_summary_df.empty:
        fusion_plot_path = report_root / "fusion_ablation_plot.png"
        figure_manifest.append({"path": str(_plot_fusion_ablation(fusion_summary_df, fusion_plot_path)), "description": "Fusion ablation comparison across detector-only, context-augmented, and token-augmented modes."})
        overview_path = report_root / "phase1_upgrade_overview_panel.png"
        figure_manifest.append({"path": str(_plot_phase1_upgrade_overview(summary_df, fusion_summary_df, latency_df, per_scenario_df, overview_path)), "description": "Combined Phase 1 upgrade overview panel for paper-ready reporting."})

    table_manifest.extend(
        [
            {"path": str(full_data.residual_artifact_path), "description": "Persisted canonical aligned residual dataset with split assignments."},
            {"path": str(summary_csv), "description": "Full-run model summary table in CSV."},
            {"path": str(summary_md), "description": "Full-run model summary table in Markdown."},
            {"path": str(per_scenario_csv), "description": "Per-scenario evaluation metrics."},
            {"path": str(threshold_csv), "description": "Threshold and configuration sweep table."},
            {"path": str(feature_sep_csv), "description": "Per-feature separability statistics."},
            {"path": str(feature_imp_csv), "description": "Top discriminative feature ranking."},
            {"path": str(latency_csv), "description": "Per-scenario detection latency table."},
            {"path": str(context_paths[0]), "description": "Structured JSONL context summaries for every aligned residual window."},
            {"path": str(context_table_path), "description": "Flattened context-summary table for joins and analysis."},
            {"path": str(reasoning_paths[0]), "description": "Optional local prompt-style reasoning outputs in JSONL."},
            {"path": str(reasoning_table_path), "description": "Flattened reasoning output table for downstream fusion and reporting."},
        ]
    )
    if not fusion_summary_df.empty:
        table_manifest.extend(
            [
                {"path": str(fusion_csv), "description": "Fusion ablation table across detector-only, detector-plus-context, and detector-plus-context-plus-token modes."},
                {"path": str(root / "outputs" / model_root_name / "fusion" / "detector_only" / "results.json"), "description": "Detector-only fusion calibration metadata and metrics."},
                {"path": str(root / "outputs" / model_root_name / "fusion" / "detector_plus_context" / "results.json"), "description": "Detector-plus-context fusion calibration metadata and metrics."},
                {"path": str(root / "outputs" / model_root_name / "fusion" / "detector_plus_context_plus_token" / "results.json"), "description": "Detector-plus-context-plus-token fusion calibration metadata and metrics."},
            ]
        )

    write_json(figure_manifest, report_root / "paper_figures_manifest_full.json")
    write_json(table_manifest, report_root / "paper_tables_manifest_full.json")
    _write_model_report_full(
        report_root=report_root,
        summary_df=summary_df,
        threshold_df=threshold_df,
        per_scenario_df=per_scenario_df,
        diagnostics=full_data,
        fusion_summary_df=fusion_summary_df,
        context_summary_count=len(context_summaries),
        reasoning_summary_count=len(reasoning_records),
        context_paths=context_paths,
        reasoning_paths=reasoning_paths,
    )
    print(f"Full-run artifacts written to {report_root}")
    return report_root


def prepare_full_run_data(
    root: Path,
    buffer_windows: int = 2,
    use_persisted_residual: bool = True,
    persist_residual: bool = True,
    artifact_root_name: str = CANONICAL_ARTIFACT_ROOT,
) -> FullRunData:
    """Load the canonical full-run inputs and persist the canonical residual dataset.

    The cached residual parquet is only reused for the canonical full-run request
    (`buffer_windows == 2` and the canonical artifact root) when it is fresh
    relative to the clean windows, attacked windows, and attack labels. Any other
    request recomputes the residual frame to avoid stale-artifact confusion.
    """
    clean_path, attacked_path, labels_path = canonical_window_source_paths(root)
    residual_path = canonical_residual_artifact_path(root, artifact_root_name=artifact_root_name)
    labels = read_dataframe(labels_path)
    if not labels.empty:
        labels["start_time_utc"] = pd.to_datetime(labels["start_time_utc"], utc=True)
        labels["end_time_utc"] = pd.to_datetime(labels["end_time_utc"], utc=True)
    is_canonical_request = buffer_windows == 2 and artifact_root_name == CANONICAL_ARTIFACT_ROOT

    if (
        is_canonical_request
        and use_persisted_residual
        and residual_artifact_is_usable(residual_path)
        and residual_artifact_is_fresh(residual_path, (clean_path, attacked_path, labels_path))
    ):
        persisted = load_persisted_residual_dataframe(residual_path)
        split_df = persisted.copy()
        residual_df = persisted.drop(columns=["split_name"], errors="ignore").copy()
    else:
        clean, attacked, labels = read_window_sources(root, clean_path, attacked_path, labels_path)
        residual_df = build_aligned_residual_dataframe(clean, attacked, labels)
        split_df = assign_attack_aware_split(residual_df, labels, buffer_windows=buffer_windows)
        if persist_residual and is_canonical_request:
            persist_residual_dataframe(split_df, residual_path)

    feature_columns = [column for column in residual_df.columns if column.startswith("delta__")]
    feature_ranking = rank_features_by_effect_size(split_df[split_df["split_name"] == "train"], feature_columns)
    return FullRunData(
        residual_df=residual_df,
        labels_df=labels,
        split_df=split_df,
        feature_ranking=feature_ranking,
        residual_artifact_path=residual_path,
    )


def assign_attack_aware_split(residual_df: pd.DataFrame, labels_df: pd.DataFrame, buffer_windows: int = 2) -> pd.DataFrame:
    frame = residual_df.sort_values("window_start_utc").reset_index(drop=True)
    frame["split_name"] = ""
    position_map = {int(idx): pos for pos, idx in enumerate(frame.index)}
    for _, label in labels_df.iterrows():
        scenario_id = str(label["scenario_id"])
        scenario_idx = frame.index[frame["scenario_window_id"] == scenario_id].tolist()
        if not scenario_idx:
            continue
        train_idx, val_idx, test_idx = _split_indices(scenario_idx)
        for split_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            frame.loc[idxs, "split_name"] = split_name
            context_positions: list[int] = []
            for idx in idxs:
                pos = position_map[int(idx)]
                context_positions.extend(range(max(0, pos - buffer_windows), min(len(frame), pos + buffer_windows + 1)))
            context_idx = [
                frame.index[pos]
                for pos in sorted(set(context_positions))
                if frame.loc[frame.index[pos], "attack_present"] == 0 and frame.loc[frame.index[pos], "split_name"] == ""
            ]
            if context_idx:
                frame.loc[context_idx, "split_name"] = split_name

    benign_idx = frame.index[frame["split_name"] == ""].tolist()
    train_idx, val_idx, test_idx = _split_indices(benign_idx, ratios=(0.6, 0.2, 0.2))
    frame.loc[train_idx, "split_name"] = "train"
    frame.loc[val_idx, "split_name"] = "val"
    frame.loc[test_idx, "split_name"] = "test"
    return frame


def _split_indices(indices: list[int], ratios: tuple[float, float, float] = (0.5, 0.2, 0.3)) -> tuple[list[int], list[int], list[int]]:
    if not indices:
        return [], [], []
    ordered = list(indices)
    n_rows = len(ordered)
    train_end = max(int(round(n_rows * ratios[0])), 1)
    val_size = max(int(round(n_rows * ratios[1])), 1) if n_rows >= 3 else max(n_rows - train_end - 1, 0)
    val_end = min(train_end + val_size, n_rows - 1) if n_rows > 2 else train_end
    train_idx = ordered[:train_end]
    val_idx = ordered[train_end:val_end]
    test_idx = ordered[val_end:]
    if not val_idx and len(ordered) >= 3:
        val_idx = [ordered[train_end]]
        test_idx = ordered[train_end + 1 :]
    if not test_idx:
        test_idx = ordered[-1:]
        if val_idx:
            val_idx = val_idx[:-1]
        elif train_idx:
            train_idx = train_idx[:-1]
    return train_idx, val_idx, test_idx


def rank_features_by_effect_size(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    benign = frame[frame["attack_present"] == 0]
    attack = frame[frame["attack_present"] == 1]
    rows: list[dict[str, object]] = []
    for column in feature_columns:
        benign_values = benign[column].astype(float).fillna(0.0).to_numpy()
        attack_values = attack[column].astype(float).fillna(0.0).to_numpy()
        if len(benign_values) < 2 or len(attack_values) < 2:
            continue
        pooled_std = math.sqrt((benign_values.var(ddof=0) + attack_values.var(ddof=0)) / 2.0) + 1e-9
        effect_size = abs(float(attack_values.mean() - benign_values.mean()) / pooled_std)
        feature_values = frame[column].astype(float).fillna(0.0).to_numpy()
        labels = frame["attack_present"].astype(int).to_numpy()
        if np.allclose(feature_values.var(ddof=0), 0.0) or np.allclose(labels.var(ddof=0), 0.0):
            corr = 0.0
        else:
            corr = abs(float(np.nan_to_num(np.corrcoef(feature_values, labels)[0, 1], nan=0.0)))
        rows.append(
            {
                "feature": column,
                "effect_size": effect_size,
                "correlation_abs": corr,
                "attack_mean": float(attack_values.mean()),
                "benign_mean": float(benign_values.mean()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["effect_size", "correlation_abs"], ascending=False).reset_index(drop=True)
    return ranking


def run_threshold_baseline(
    root: Path,
    full_data: FullRunData,
    feature_counts: list[int],
    report_root: Path,
    model_root_name: str,
    artifact_root_name: str,
) -> dict[str, object]:
    model_name = "threshold_baseline"
    paths = ensure_model_paths(model_name, root, model_root=model_root_name, artifact_root=artifact_root_name)
    splits = _split_frames(full_data.split_df)
    sweep_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for feature_count in feature_counts:
        feature_columns = _top_features(full_data.feature_ranking, feature_count)
        train_benign = splits["train"][splits["train"]["attack_present"] == 0]
        standardizer = fit_standardizer(train_benign, feature_columns)
        val_scores = _threshold_score(splits["val"], feature_columns, standardizer)
        y_val = splits["val"]["attack_present"].astype(int).to_numpy()
        calibrated_val, calibrator = calibrate_scores(val_scores, y_val)
        benign_val_scores = calibrated_val[splits["val"]["attack_present"] == 0]
        for threshold_name, quantile in THRESHOLD_LEVELS.items():
            threshold_value = float(np.quantile(benign_val_scores, quantile)) if len(benign_val_scores) else float(np.quantile(calibrated_val, quantile))
            metrics = compute_binary_metrics(y_val, calibrated_val, threshold_value)
            row = {
                "model_name": model_name,
                "feature_count": len(feature_columns),
                "seq_len": None,
                "threshold_name": threshold_name,
                "threshold": threshold_value,
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
                "config_type": "threshold",
            }
            sweep_rows.append(row)
            if _is_better_configuration(metrics, best["metrics"] if best else None):
                best = {
                    "feature_columns": feature_columns,
                    "standardizer": standardizer,
                    "calibrator": calibrator,
                    "threshold": threshold_value,
                    "threshold_name": threshold_name,
                    "metrics": metrics,
                }

    assert best is not None
    best_val_scores = _threshold_score(splits["val"], best["feature_columns"], best["standardizer"])
    calibrated_val = apply_calibrator(best["calibrator"], best_val_scores)
    test_scores = _threshold_score(splits["test"], best["feature_columns"], best["standardizer"])
    calibrated_test = apply_calibrator(best["calibrator"], test_scores)
    val_predictions = _prediction_frame(splits["val"], calibrated_val, best["threshold"])
    predictions = _prediction_frame(splits["test"], calibrated_test, best["threshold"])
    metrics = compute_binary_metrics(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy(), best["threshold"])
    curves = compute_curve_payload(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy())
    latency = detection_latency_table(predictions, full_data.labels_df, model_name)
    per_scenario = per_scenario_metrics(predictions, full_data.labels_df, model_name)
    training_history = {"train_loss": [], "val_loss": []}
    model_info = {
        "model_name": model_name,
        "feature_columns": best["feature_columns"],
        "selected_threshold": best["threshold_name"],
        "threshold": best["threshold"],
        "training_time_seconds": 0.0,
        "inference_time_seconds": float(len(predictions)),
        "parameter_count": len(best["feature_columns"]) * 2,
        "feature_mode": "aligned_residual",
        "run_mode": "full",
        "training_supervision": "normal_oriented_with_validation_calibration",
        "architecture_parameters": {},
    }
    write_pickle({"standardizer": best["standardizer"], "calibrator": best["calibrator"], "model_info": model_info}, paths.model_dir / "model.pkl")
    write_json({"metrics": metrics, "curves": curves, "model_info": model_info, "history": training_history}, paths.model_dir / "results.json")
    write_predictions(predictions, paths.model_dir / "predictions.parquet")
    ready_package_dir = export_ready_model_package(
        root=root,
        package_name=model_name,
        model_name=model_name,
        model_family="statistical_threshold_detector",
        checkpoint_kind="pickle",
        checkpoint_payload={"score_method": "mean_abs_standardized"},
        preprocessing_payload={"standardizer": best["standardizer"], "discretizer": None},
        feature_columns=best["feature_columns"],
        thresholds_payload={
            "primary_threshold": float(best["threshold"]),
            "selection_method": "benign_val_quantile_after_calibration",
            "selected_threshold_name": best["threshold_name"],
        },
        calibration_payload=serialize_calibration(best["calibrator"], input_names=["raw_score"]),
        config_payload={
            "feature_mode": "aligned_residual",
            "sequence_length": None,
            "token_input": False,
            "score_method": "mean_abs_standardized",
            "architecture_parameters": {},
            "training_supervision": "normal_oriented_with_validation_calibration",
            "notes": "Normal-oriented threshold detector calibrated on validation windows.",
        },
        training_history=training_history,
        metrics_payload=metrics,
        predictions=predictions,
        split_summary=build_split_summary(full_data.split_df),
        notes="Normal-oriented threshold baseline over aligned residual windows.",
        training_supervision="normal_oriented_with_validation_calibration",
        input_schema_summary=build_input_schema_summary(best["feature_columns"], {"feature_mode": "aligned_residual"}),
    )
    figures = generate_model_plots(model_name, metrics, curves, {"train_loss": [], "val_loss": []}, predictions, paths.artifact_dir)
    return {
        "summary": _summary_row(model_name, metrics, model_info),
        "sweep_rows": sweep_rows,
        "latency": latency,
        "per_scenario": per_scenario,
        "figures": figures,
        "validation_predictions": val_predictions,
        "predictions": predictions,
        "model_info": model_info,
        "ready_package_dir": ready_package_dir,
    }


def run_isolation_forest(
    root: Path,
    full_data: FullRunData,
    feature_counts: list[int],
    report_root: Path,
    model_root_name: str,
    artifact_root_name: str,
) -> dict[str, object]:
    model_name = "isolation_forest"
    paths = ensure_model_paths(model_name, root, model_root=model_root_name, artifact_root=artifact_root_name)
    splits = _split_frames(full_data.split_df)
    sweep_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for feature_count in feature_counts:
        feature_columns = _top_features(full_data.feature_ranking, feature_count)
        train_benign = splits["train"][splits["train"]["attack_present"] == 0]
        standardizer = fit_standardizer(train_benign, feature_columns)
        x_train = transform_features(train_benign, feature_columns, standardizer)
        x_val = transform_features(splits["val"], feature_columns, standardizer)

        model = IsolationForest(n_estimators=300, contamination=0.05, random_state=1729, n_jobs=-1)
        start = time.perf_counter()
        model.fit(x_train)
        training_time = time.perf_counter() - start
        val_scores = -model.score_samples(x_val)
        y_val = splits["val"]["attack_present"].astype(int).to_numpy()
        calibrated_val, calibrator = calibrate_scores(val_scores, y_val)
        threshold, metrics = choose_best_threshold(calibrated_val, y_val)
        sweep_rows.append(
            {
                "model_name": model_name,
                "feature_count": len(feature_columns),
                "seq_len": None,
                "threshold_name": "val_scan",
                "threshold": threshold,
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
                "config_type": "unsupervised",
            }
        )
        if _is_better_configuration(metrics, best["metrics"] if best else None):
            best = {
                "feature_columns": feature_columns,
                "standardizer": standardizer,
                "model": model,
                "calibrator": calibrator,
                "threshold": threshold,
                "metrics": metrics,
                "training_time": training_time,
            }

    assert best is not None
    x_val_best = transform_features(splits["val"], best["feature_columns"], best["standardizer"])
    val_scores = -best["model"].score_samples(x_val_best)
    calibrated_val = apply_calibrator(best["calibrator"], val_scores)
    x_test = transform_features(splits["test"], best["feature_columns"], best["standardizer"])
    inference_start = time.perf_counter()
    test_scores = -best["model"].score_samples(x_test)
    inference_time = time.perf_counter() - inference_start
    calibrated_test = apply_calibrator(best["calibrator"], test_scores)
    val_predictions = _prediction_frame(splits["val"], calibrated_val, best["threshold"])
    predictions = _prediction_frame(splits["test"], calibrated_test, best["threshold"])
    metrics = compute_binary_metrics(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy(), best["threshold"])
    curves = compute_curve_payload(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy())
    latency = detection_latency_table(predictions, full_data.labels_df, model_name)
    per_scenario = per_scenario_metrics(predictions, full_data.labels_df, model_name)
    model_info = {
        "model_name": model_name,
        "feature_columns": best["feature_columns"],
        "threshold": best["threshold"],
        "training_time_seconds": best["training_time"],
        "inference_time_seconds": inference_time,
        "parameter_count": int(best["model"].estimators_[0].tree_.node_count * len(best["model"].estimators_)) if hasattr(best["model"], "estimators_") else 0,
        "feature_mode": "aligned_residual",
        "run_mode": "full",
        "training_supervision": "normal_oriented_with_validation_calibration",
        "architecture_parameters": {
            "n_estimators": int(getattr(best["model"], "n_estimators", 0)),
            "contamination": float(getattr(best["model"], "contamination", 0.0)),
            "random_state": int(getattr(best["model"], "random_state", 0) or 0),
        },
    }
    write_pickle({"model": best["model"], "standardizer": best["standardizer"], "calibrator": best["calibrator"], "model_info": model_info}, paths.model_dir / "model.pkl")
    training_history = {"train_loss": [], "val_loss": []}
    write_json({"metrics": metrics, "curves": curves, "model_info": model_info, "history": training_history}, paths.model_dir / "results.json")
    write_predictions(predictions, paths.model_dir / "predictions.parquet")
    ready_package_dir = export_ready_model_package(
        root=root,
        package_name=model_name,
        model_name=model_name,
        model_family="isolation_forest_detector",
        checkpoint_kind="pickle",
        checkpoint_payload={"model": best["model"]},
        preprocessing_payload={"standardizer": best["standardizer"], "discretizer": None},
        feature_columns=best["feature_columns"],
        thresholds_payload={
            "primary_threshold": float(best["threshold"]),
            "selection_method": "validation_threshold_scan",
        },
        calibration_payload=serialize_calibration(best["calibrator"], input_names=["raw_score"]),
        config_payload={
            "feature_mode": "aligned_residual",
            "sequence_length": None,
            "token_input": False,
            "architecture_parameters": model_info["architecture_parameters"],
            "training_supervision": "normal_oriented_with_validation_calibration",
            "notes": "Benign-trained isolation forest over aligned residual windows.",
        },
        training_history=training_history,
        metrics_payload=metrics,
        predictions=predictions,
        split_summary=build_split_summary(full_data.split_df),
        notes="Isolation Forest trained on benign residual windows and calibrated on validation data.",
        training_supervision="normal_oriented_with_validation_calibration",
        input_schema_summary=build_input_schema_summary(best["feature_columns"], {"feature_mode": "aligned_residual"}),
    )
    figures = generate_model_plots(model_name, metrics, curves, {"train_loss": [], "val_loss": []}, predictions, paths.artifact_dir)
    return {
        "summary": _summary_row(model_name, metrics, model_info),
        "sweep_rows": sweep_rows,
        "latency": latency,
        "per_scenario": per_scenario,
        "figures": figures,
        "validation_predictions": val_predictions,
        "predictions": predictions,
        "model_info": model_info,
        "ready_package_dir": ready_package_dir,
    }


def run_autoencoder(
    root: Path,
    full_data: FullRunData,
    feature_counts: list[int],
    epochs: int,
    patience: int,
    report_root: Path,
    model_root_name: str,
    artifact_root_name: str,
) -> dict[str, object]:
    model_name = "autoencoder"
    paths = ensure_model_paths(model_name, root, model_root=model_root_name, artifact_root=artifact_root_name)
    splits = _split_frames(full_data.split_df)
    sweep_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for feature_count in feature_counts:
        feature_columns = _top_features(full_data.feature_ranking, feature_count)
        train_benign = splits["train"][splits["train"]["attack_present"] == 0]
        val_benign = splits["val"][splits["val"]["attack_present"] == 0]
        standardizer = fit_standardizer(train_benign, feature_columns)
        x_train = transform_features(train_benign, feature_columns, standardizer).astype(np.float32)
        x_val_benign = transform_features(val_benign, feature_columns, standardizer).astype(np.float32)
        x_val = transform_features(splits["val"], feature_columns, standardizer).astype(np.float32)
        model = MLPAutoencoder(input_dim=len(feature_columns), hidden_dim=min(128, max(32, len(feature_columns))), bottleneck_dim=max(16, len(feature_columns) // 4))
        model, history, training_time = train_autoencoder(model, x_train, x_val_benign, epochs=epochs, patience=patience)
        val_scores = predict_autoencoder_errors(model, x_val)
        y_val = splits["val"]["attack_present"].astype(int).to_numpy()
        calibrated_val, calibrator = calibrate_scores(val_scores, y_val)
        threshold, metrics = choose_best_threshold(calibrated_val, y_val)
        sweep_rows.append(
            {
                "model_name": model_name,
                "feature_count": len(feature_columns),
                "seq_len": None,
                "threshold_name": "val_scan",
                "threshold": threshold,
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
                "config_type": "unsupervised",
            }
        )
        if _is_better_configuration(metrics, best["metrics"] if best else None):
            best = {
                "feature_columns": feature_columns,
                "standardizer": standardizer,
                "model": model,
                "calibrator": calibrator,
                "threshold": threshold,
                "metrics": metrics,
                "history": history,
                "training_time": training_time,
            }

    assert best is not None
    x_val_best = transform_features(splits["val"], best["feature_columns"], best["standardizer"]).astype(np.float32)
    val_scores = predict_autoencoder_errors(best["model"], x_val_best)
    calibrated_val = apply_calibrator(best["calibrator"], val_scores)
    x_test = transform_features(splits["test"], best["feature_columns"], best["standardizer"]).astype(np.float32)
    inference_start = time.perf_counter()
    test_scores = predict_autoencoder_errors(best["model"], x_test)
    inference_time = time.perf_counter() - inference_start
    calibrated_test = apply_calibrator(best["calibrator"], test_scores)
    val_predictions = _prediction_frame(splits["val"], calibrated_val, best["threshold"])
    predictions = _prediction_frame(splits["test"], calibrated_test, best["threshold"])
    metrics = compute_binary_metrics(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy(), best["threshold"])
    curves = compute_curve_payload(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy())
    latency = detection_latency_table(predictions, full_data.labels_df, model_name)
    per_scenario = per_scenario_metrics(predictions, full_data.labels_df, model_name)
    model_info = {
        "model_name": model_name,
        "feature_columns": best["feature_columns"],
        "threshold": best["threshold"],
        "training_time_seconds": best["training_time"],
        "inference_time_seconds": inference_time,
        "parameter_count": int(sum(parameter.numel() for parameter in best["model"].parameters())),
        "feature_mode": "aligned_residual",
        "run_mode": "full",
        "training_supervision": "normal_oriented_with_validation_calibration",
        "architecture_parameters": architecture_from_model(
            best["model"],
            model_name,
            input_dim=len(best["feature_columns"]),
        ),
    }
    write_pickle({"standardizer": best["standardizer"], "calibrator": best["calibrator"], "model_info": model_info}, paths.model_dir / "metadata.pkl")
    write_torch_state(best["model"], paths.model_dir / "model.pt")
    write_json({"metrics": metrics, "curves": curves, "model_info": model_info, "history": best["history"]}, paths.model_dir / "results.json")
    write_predictions(predictions, paths.model_dir / "predictions.parquet")
    ready_package_dir = export_ready_model_package(
        root=root,
        package_name=model_name,
        model_name=model_name,
        model_family="autoencoder_detector",
        checkpoint_kind="torch",
        checkpoint_payload=best["model"],
        preprocessing_payload={"standardizer": best["standardizer"], "discretizer": None},
        feature_columns=best["feature_columns"],
        thresholds_payload={
            "primary_threshold": float(best["threshold"]),
            "selection_method": "validation_threshold_scan",
        },
        calibration_payload=serialize_calibration(best["calibrator"], input_names=["raw_score"]),
        config_payload={
            "feature_mode": "aligned_residual",
            "sequence_length": None,
            "token_input": False,
            "architecture_parameters": model_info["architecture_parameters"],
            "training_supervision": "normal_oriented_with_validation_calibration",
            "notes": "Benign-trained autoencoder over aligned residual windows.",
        },
        training_history=best["history"],
        metrics_payload=metrics,
        predictions=predictions,
        split_summary=build_split_summary(full_data.split_df),
        notes="Autoencoder trained on benign residual windows and calibrated on validation data.",
        training_supervision="normal_oriented_with_validation_calibration",
        input_schema_summary=build_input_schema_summary(best["feature_columns"], {"feature_mode": "aligned_residual"}),
    )
    figures = generate_model_plots(model_name, metrics, curves, best["history"], predictions, paths.artifact_dir)
    return {
        "summary": _summary_row(model_name, metrics, model_info),
        "sweep_rows": sweep_rows,
        "latency": latency,
        "per_scenario": per_scenario,
        "figures": figures,
        "validation_predictions": val_predictions,
        "predictions": predictions,
        "model_info": model_info,
        "ready_package_dir": ready_package_dir,
    }


def run_sequence_model(
    root: Path,
    full_data: FullRunData,
    model_name: str,
    model_ctor,
    feature_counts: list[int],
    seq_lens: list[int],
    epochs: int,
    patience: int,
    token_input: bool,
    report_root: Path,
    model_root_name: str,
    artifact_root_name: str,
) -> dict[str, object]:
    paths = ensure_model_paths(model_name, root, model_root=model_root_name, artifact_root=artifact_root_name)
    splits = _split_frames(full_data.split_df)
    sweep_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None

    for feature_count in feature_counts:
        feature_columns = _top_features(full_data.feature_ranking, feature_count)
        train_benign = splits["train"][splits["train"]["attack_present"] == 0]
        standardizer = fit_standardizer(train_benign, feature_columns)
        discretizer = fit_discretizer(splits["train"], feature_columns, bins=16) if token_input else None

        for seq_len in seq_lens:
            x_train, y_train, _ = build_sequence_dataset_segments(splits["train"], feature_columns, seq_len, standardizer, discretizer)
            x_val, y_val, _ = build_sequence_dataset_segments(splits["val"], feature_columns, seq_len, standardizer, discretizer)
            if len(x_train) == 0 or len(x_val) == 0:
                continue

            model = model_ctor(len(feature_columns))
            pos_weight = float(max((y_train == 0).sum(), 1) / max((y_train == 1).sum(), 1))
            model, history, training_time = train_classifier(
                model,
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=epochs,
                token_input=token_input,
                pos_weight=pos_weight,
                patience=patience,
            )
            val_scores = predict_classifier_scores(model, x_val, token_input=token_input)
            calibrated_val, calibrator = calibrate_scores(val_scores, y_val)
            threshold, metrics = choose_best_threshold(calibrated_val, y_val)
            sweep_rows.append(
                {
                    "model_name": model_name,
                    "feature_count": len(feature_columns),
                    "seq_len": seq_len,
                    "threshold_name": "val_scan",
                    "threshold": threshold,
                    "val_precision": metrics["precision"],
                    "val_recall": metrics["recall"],
                    "val_f1": metrics["f1"],
                    "config_type": "sequence",
                }
            )
            if _is_better_configuration(metrics, best["metrics"] if best else None):
                best = {
                    "feature_columns": feature_columns,
                    "standardizer": standardizer,
                    "discretizer": discretizer,
                    "model": model,
                    "calibrator": calibrator,
                    "threshold": threshold,
                    "metrics": metrics,
                    "history": history,
                    "training_time": training_time,
                    "seq_len": seq_len,
                    "token_input": token_input,
                }

    assert best is not None
    x_test, y_test, test_meta = build_sequence_dataset_segments(
        splits["test"],
        best["feature_columns"],
        best["seq_len"],
        best["standardizer"],
        best["discretizer"],
    )
    inference_start = time.perf_counter()
    test_scores = predict_classifier_scores(best["model"], x_test, token_input=best["token_input"])
    inference_time = time.perf_counter() - inference_start
    x_val_best, _, val_meta = build_sequence_dataset_segments(
        splits["val"],
        best["feature_columns"],
        best["seq_len"],
        best["standardizer"],
        best["discretizer"],
    )
    val_scores = predict_classifier_scores(best["model"], x_val_best, token_input=best["token_input"]) if len(x_val_best) else np.array([], dtype=float)
    calibrated_val = apply_calibrator(best["calibrator"], val_scores)
    calibrated_test = apply_calibrator(best["calibrator"], test_scores)
    val_predictions = val_meta.copy()
    if not val_predictions.empty:
        val_predictions["score"] = calibrated_val
        val_predictions["predicted"] = (val_predictions["score"] >= best["threshold"]).astype(int)
    predictions = test_meta.copy()
    predictions["score"] = calibrated_test
    predictions["predicted"] = (predictions["score"] >= best["threshold"]).astype(int)
    metrics = compute_binary_metrics(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy(), best["threshold"])
    curves = compute_curve_payload(predictions["attack_present"].to_numpy(), predictions["score"].to_numpy())
    latency = detection_latency_table(predictions, full_data.labels_df, model_name)
    per_scenario = per_scenario_metrics(predictions, full_data.labels_df, model_name)
    model_info = {
        "model_name": model_name,
        "feature_columns": best["feature_columns"],
        "seq_len": best["seq_len"],
        "threshold": best["threshold"],
        "training_time_seconds": best["training_time"],
        "inference_time_seconds": inference_time,
        "parameter_count": int(sum(parameter.numel() for parameter in best["model"].parameters())),
        "feature_mode": "aligned_residual",
        "run_mode": "full",
        "training_supervision": "supervised_sequence_classifier",
        "notes": "Token baseline uses discretized residual features." if model_name == "llm_baseline" else None,
        "architecture_parameters": architecture_from_model(
            best["model"],
            model_name,
            input_dim=len(best["feature_columns"]),
            seq_len=best["seq_len"],
            token_bins=discretizer_bin_count(best["discretizer"]),
        ),
    }
    payload = {"standardizer": best["standardizer"], "discretizer": best["discretizer"], "calibrator": best["calibrator"], "model_info": model_info}
    write_pickle(payload, paths.model_dir / "metadata.pkl")
    write_torch_state(best["model"], paths.model_dir / "model.pt")
    write_json({"metrics": metrics, "curves": curves, "model_info": model_info, "history": best["history"]}, paths.model_dir / "results.json")
    write_predictions(predictions, paths.model_dir / "predictions.parquet")
    ready_package_dir = export_ready_model_package(
        root=root,
        package_name=model_name,
        model_name=model_name,
        model_family="token_sequence_classifier" if model_name == "llm_baseline" else "sequence_classifier",
        checkpoint_kind="torch",
        checkpoint_payload=best["model"],
        preprocessing_payload={"standardizer": best["standardizer"], "discretizer": best["discretizer"]},
        feature_columns=best["feature_columns"],
        thresholds_payload={
            "primary_threshold": float(best["threshold"]),
            "selection_method": "validation_threshold_scan",
        },
        calibration_payload=serialize_calibration(best["calibrator"], input_names=["raw_score"]),
        config_payload={
            "feature_mode": "aligned_residual",
            "sequence_length": int(best["seq_len"]),
            "token_input": bool(best["token_input"]),
            "architecture_parameters": model_info["architecture_parameters"],
            "training_supervision": "supervised_sequence_classifier",
            "notes": model_info.get("notes"),
            "tokenizer_settings": (
                {
                    "strategy": "per_feature_quantile_digitization",
                    "requested_bins": 16,
                    "actual_max_bins": discretizer_bin_count(best["discretizer"]),
                }
                if model_name == "llm_baseline"
                else None
            ),
        },
        training_history=best["history"],
        metrics_payload=metrics,
        predictions=predictions,
        split_summary=build_split_summary(full_data.split_df),
        notes=(
            "Tokenized time-series language-style baseline using discretized residual tokens."
            if model_name == "llm_baseline"
            else "Sequence classifier over aligned residual windows."
        ),
        training_supervision="supervised_sequence_classifier",
        input_schema_summary=build_input_schema_summary(best["feature_columns"], {"feature_mode": "aligned_residual"}),
    )
    figures = generate_model_plots(model_name, metrics, curves, best["history"], predictions, paths.artifact_dir)
    return {
        "summary": _summary_row(model_name, metrics, model_info),
        "sweep_rows": sweep_rows,
        "latency": latency,
        "per_scenario": per_scenario,
        "figures": figures,
        "validation_predictions": val_predictions,
        "predictions": predictions,
        "model_info": model_info,
        "ready_package_dir": ready_package_dir,
    }


def build_sequence_dataset_segments(
    frame: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    standardizer,
    discretizer=None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ordered = frame.sort_values("window_start_utc").reset_index(drop=True).copy()
    if ordered.empty:
        return np.empty((0, seq_len, len(feature_columns))), np.array([], dtype=np.float32), pd.DataFrame()
    ordered["window_start_utc"] = pd.to_datetime(ordered["window_start_utc"], utc=True)
    step_seconds = int(ordered["window_start_utc"].diff().dt.total_seconds().dropna().mode().iloc[0]) if len(ordered) > 1 else 60
    segment_break = ordered["window_start_utc"].diff().dt.total_seconds().fillna(step_seconds).ne(step_seconds)
    ordered["segment_id"] = segment_break.cumsum().astype(int)

    sequences: list[np.ndarray] = []
    labels: list[float] = []
    rows: list[dict[str, object]] = []
    for _, segment in ordered.groupby("segment_id", observed=False):
        if len(segment) < seq_len:
            continue
        if discretizer is not None:
            feature_matrix = transform_to_tokens(segment, feature_columns, discretizer).astype(np.int64)
        else:
            feature_matrix = transform_features(segment, feature_columns, standardizer).astype(np.float32)
        for end_idx in range(seq_len - 1, len(segment)):
            start_idx = end_idx - seq_len + 1
            sequences.append(feature_matrix[start_idx : end_idx + 1])
            labels.append(float(segment.iloc[end_idx]["attack_present"]))
            rows.append(
                {
                    "window_start_utc": segment.iloc[end_idx]["window_start_utc"],
                    "window_end_utc": segment.iloc[end_idx]["window_end_utc"],
                    "scenario_id": segment.iloc[end_idx]["scenario_window_id"],
                    "attack_present": int(segment.iloc[end_idx]["attack_present"]),
                    "attack_family": str(segment.iloc[end_idx].get("attack_family", "benign")),
                    "attack_severity": str(segment.iloc[end_idx].get("attack_severity", "none")),
                    "attack_affected_assets": str(segment.iloc[end_idx].get("attack_affected_assets", "[]")),
                    "split_name": str(segment.iloc[end_idx].get("split_name", "")),
                }
            )
    dtype = np.int64 if discretizer is not None else np.float32
    return np.asarray(sequences, dtype=dtype), np.asarray(labels, dtype=np.float32), pd.DataFrame(rows)


def calibrate_scores(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, LogisticRegression | None]:
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return scores, None
    calibrator = LogisticRegression(random_state=1729, solver="lbfgs")
    calibrator.fit(scores.reshape(-1, 1), y_true)
    calibrated = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    return calibrated.astype(float), calibrator


def apply_calibrator(calibrator: LogisticRegression | None, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if calibrator is None:
        return scores
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1].astype(float)


def choose_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, dict[str, object]]:
    candidate_thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.995, 60)))
    best_threshold = float(candidate_thresholds[0]) if len(candidate_thresholds) else 0.5
    best_metrics: dict[str, object] | None = None
    for threshold in candidate_thresholds:
        metrics = compute_binary_metrics(y_true, scores, float(threshold))
        if _is_better_configuration(metrics, best_metrics):
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _is_better_configuration(metrics: dict[str, object], best_metrics: dict[str, object] | None) -> bool:
    if best_metrics is None:
        return True
    candidate = (
        float(metrics.get("recall") or 0.0),
        float(metrics.get("precision") or 0.0),
        float(metrics.get("f1") or 0.0),
        float(metrics.get("average_precision") or 0.0),
    )
    incumbent = (
        float(best_metrics.get("recall") or 0.0),
        float(best_metrics.get("precision") or 0.0),
        float(best_metrics.get("f1") or 0.0),
        float(best_metrics.get("average_precision") or 0.0),
    )
    return candidate > incumbent


def _threshold_score(frame: pd.DataFrame, feature_columns: list[str], standardizer) -> np.ndarray:
    transformed = transform_features(frame, feature_columns, standardizer)
    return np.mean(np.abs(transformed), axis=1)


def _top_features(ranking: pd.DataFrame, feature_count: int) -> list[str]:
    if ranking.empty:
        return []
    return ranking["feature"].head(min(feature_count, len(ranking))).tolist()


def _split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split_name: frame[frame["split_name"] == split_name].copy().reset_index(drop=True)
        for split_name in ["train", "val", "test"]
    }


def _prediction_frame(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    selected_columns = [
        column
        for column in [
            "window_start_utc",
            "window_end_utc",
            "scenario_window_id",
            "attack_present",
            "attack_family",
            "attack_severity",
            "attack_affected_assets",
            "split_name",
        ]
        if column in frame.columns
    ]
    predictions = frame[selected_columns].copy()
    predictions = predictions.rename(columns={"scenario_window_id": "scenario_id"})
    predictions["score"] = np.asarray(scores, dtype=float)
    predictions["predicted"] = (predictions["score"] >= float(threshold)).astype(int)
    return predictions


def _summary_row(model_name: str, metrics: dict[str, object], model_info: dict[str, object]) -> dict[str, object]:
    return {
        "model_name": model_name,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "weighted_precision": metrics.get("weighted_precision"),
        "weighted_recall": metrics.get("weighted_recall"),
        "threshold": metrics.get("threshold"),
        "feature_count": len(model_info.get("feature_columns", [])),
        "seq_len": model_info.get("seq_len"),
    }


def _write_diagnostic_tables(full_data: FullRunData, report_root: Path, feature_counts: list[int], window_seconds: int) -> None:
    split_counts = (
        full_data.split_df.groupby(["split_name", "attack_present"], observed=False)
        .size()
        .reset_index(name="window_count")
        .sort_values(["split_name", "attack_present"])
    )
    split_counts.to_csv(report_root / "class_balance_table.csv", index=False)
    scenario_counts = (
        full_data.split_df[full_data.split_df["attack_present"] == 1]
        .groupby("scenario_window_id", observed=False)
        .size()
        .reset_index(name="attack_window_count")
        .rename(columns={"scenario_window_id": "scenario_id"})
    )
    scenario_counts.to_csv(report_root / "attack_window_counts_per_scenario.csv", index=False)
    baseline_window_row = pd.DataFrame(
        [
            {
                "window_seconds": window_seconds,
                "step_seconds": 60,
                "rows": int(len(full_data.split_df)),
                "attack_rows": int(full_data.split_df["attack_present"].sum()),
                "attack_window_count": int(full_data.split_df["attack_present"].sum()),
                "note": "Aligned nominal-grid full-run baseline.",
            }
        ]
    )
    existing_window_path = report_root / "window_sweep_table.csv"
    if existing_window_path.exists():
        existing_window = pd.read_csv(existing_window_path)
        window_sweep = (
            pd.concat([existing_window, baseline_window_row], ignore_index=True, sort=False)
            .drop_duplicates(subset=["window_seconds", "step_seconds"], keep="last")
            .sort_values(["window_seconds", "step_seconds"])
            .reset_index(drop=True)
        )
    else:
        window_sweep = baseline_window_row
    window_sweep.to_csv(report_root / "window_sweep_table.csv", index=False)


def generate_model_plots(
    model_name: str,
    metrics: dict[str, object],
    curves: dict[str, list[float]],
    history: dict[str, list[float]],
    predictions: pd.DataFrame,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    figure_manifest.append({"path": str(_plot_threshold(predictions, float(metrics.get("threshold", 0.5) or 0.5), threshold_path, display_name)), "description": f"Threshold visualization for {display_name}."})
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
    benign = predictions[predictions["attack_present"] == 0]["score"]
    attack = predictions[predictions["attack_present"] == 1]["score"]
    if not benign.empty:
        ax.hist(benign, bins=20, alpha=0.6, label="benign")
    if not attack.empty:
        ax.hist(attack, bins=20, alpha=0.6, label="attack")
    if not benign.empty or not attack.empty:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No prediction scores available", ha="center", va="center")
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
    ordered = predictions.sort_values("window_start_utc").reset_index(drop=True)
    if ordered.empty:
        ax.text(0.5, 0.5, "No prediction scores available", ha="center", va="center")
    else:
        ax.plot(ordered["score"], label="score")
        ax.axhline(threshold, color="red", linestyle="--", label="threshold")
        ax.legend()
    ax.set_title(f"{model_name} threshold visualization")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_model_comparison(summary_df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    if summary_df.empty:
        ax.text(0.5, 0.5, "No model results available", ha="center", va="center")
    else:
        metrics = ["precision", "recall", "f1", "average_precision"]
        x = np.arange(len(summary_df))
        width = 0.18
        for idx, metric in enumerate(metrics):
            ax.bar(x + idx * width, summary_df[metric].fillna(0.0), width=width, label=metric)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(summary_df["model_name"].map(display_model_name), rotation=30)
        ax.legend()
    ax.set_title("Full-Run Model Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_latency_by_scenario(latency_df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    if latency_df.empty:
        ax.text(0.5, 0.5, "No latency values available", ha="center", va="center")
    else:
        pivot = latency_df.pivot_table(index="scenario_id", columns="model_name", values="latency_seconds", aggfunc="mean")
        pivot = pivot.rename(columns=display_model_name)
        pivot.plot(kind="bar", ax=ax)
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Detection Latency by Scenario")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_fusion_ablation(fusion_df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    if fusion_df.empty:
        ax.text(0.5, 0.5, "No fusion rows available", ha="center", va="center")
    else:
        metrics = ["precision", "recall", "f1", "average_precision"]
        x = np.arange(len(fusion_df))
        width = 0.18
        for idx, metric in enumerate(metrics):
            ax.bar(x + idx * width, fusion_df[metric].fillna(0.0), width=width, label=metric)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([item.replace("_", "\n") for item in fusion_df["fusion_mode"]], rotation=0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Fusion Ablation Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_phase1_upgrade_overview(
    summary_df: pd.DataFrame,
    fusion_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    per_scenario_df: pd.DataFrame,
    path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    if summary_df.empty:
        axes[0, 0].text(0.5, 0.5, "No model summary", ha="center", va="center")
    else:
        top_models = summary_df.sort_values("f1", ascending=False).head(5)
        axes[0, 0].barh(top_models["model_name"].map(display_model_name), top_models["f1"].fillna(0.0))
        axes[0, 0].set_title("Top Phase 1 Models by F1")
        axes[0, 0].set_xlim(0.0, 1.05)

    if fusion_df.empty:
        axes[0, 1].text(0.5, 0.5, "No fusion ablation", ha="center", va="center")
    else:
        axes[0, 1].plot(fusion_df["fusion_mode"], fusion_df["f1"], marker="o", label="F1")
        axes[0, 1].plot(fusion_df["fusion_mode"], fusion_df["recall"], marker="o", label="Recall")
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].tick_params(axis="x", rotation=20)
        axes[0, 1].set_title("Fusion Mode Lift")
        axes[0, 1].legend(loc="best", fontsize=8)

    if latency_df.empty:
        axes[1, 0].text(0.5, 0.5, "No latency table", ha="center", va="center")
    else:
        latency_summary = latency_df.groupby("model_name", observed=False)["latency_seconds"].mean().reset_index()
        latency_summary = latency_summary.sort_values("latency_seconds").head(6)
        axes[1, 0].bar(latency_summary["model_name"].map(display_model_name), latency_summary["latency_seconds"].fillna(0.0))
        axes[1, 0].tick_params(axis="x", rotation=30)
        axes[1, 0].set_title("Mean Detection Latency")
        axes[1, 0].set_ylabel("Seconds")

    if per_scenario_df.empty:
        axes[1, 1].text(0.5, 0.5, "No per-scenario metrics", ha="center", va="center")
    else:
        scenario_pivot = per_scenario_df.pivot_table(index="scenario_id", columns="model_name", values="f1", aggfunc="mean").fillna(0.0)
        scenario_pivot = scenario_pivot.loc[:, [column for column in MODEL_PRIORITY if column in scenario_pivot.columns][:5]]
        heatmap = axes[1, 1].imshow(scenario_pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Per-Scenario F1 Heatmap")
        axes[1, 1].set_yticks(np.arange(len(scenario_pivot.index)))
        axes[1, 1].set_yticklabels(scenario_pivot.index, fontsize=7)
        axes[1, 1].set_xticks(np.arange(len(scenario_pivot.columns)))
        axes[1, 1].set_xticklabels([display_model_name(column) for column in scenario_pivot.columns], rotation=30, ha="right", fontsize=8)
        fig.colorbar(heatmap, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_model_report_full(
    report_root: Path,
    summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    per_scenario_df: pd.DataFrame,
    diagnostics: FullRunData,
    fusion_summary_df: pd.DataFrame,
    context_summary_count: int,
    reasoning_summary_count: int,
    context_paths: tuple[Path, Path],
    reasoning_paths: tuple[Path, Path, Path],
) -> None:
    highest_recall_row = summary_df.sort_values(["recall", "precision", "f1"], ascending=False).iloc[0].to_dict() if not summary_df.empty else {}
    best_balanced_row = summary_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0].to_dict() if not summary_df.empty else {}
    summary_md_df = apply_model_display_names(summary_df)
    threshold_md_df = apply_model_display_names(threshold_df)
    per_scenario_md_df = apply_model_display_names(per_scenario_df)
    lines = ["# Full Model Report", ""]
    lines.append("## Root Cause Diagnosis")
    lines.append("- The original smoke pipeline used a chronological split that placed all positive attack windows in training and left validation/test with zero attacks.")
    lines.append("- The original merged-window builder used jittered measured timestamps as window anchors, so clean and attacked windows were misaligned and residual comparisons were invalid.")
    lines.append("- The original smoke models used raw windows without aligned residual features, fixed thresholds, no class weighting, no early stopping, and no calibrated validation sweep.")
    lines.append("")
    lines.append("## Headline Models")
    if highest_recall_row:
        lines.append(f"- highest_recall_model: `{display_model_name(highest_recall_row.get('model_name'))}` with recall `{highest_recall_row.get('recall')}`, precision `{highest_recall_row.get('precision')}`, F1 `{highest_recall_row.get('f1')}`")
    if best_balanced_row:
        lines.append(f"- best_balanced_model: `{display_model_name(best_balanced_row.get('model_name'))}` with precision `{best_balanced_row.get('precision')}`, recall `{best_balanced_row.get('recall')}`, F1 `{best_balanced_row.get('f1')}`, PR-AUC `{best_balanced_row.get('average_precision')}`")
        lines.append("- Publication-oriented recommendation should follow the balanced model unless the use case explicitly prioritizes recall at the expense of false alarms.")
    if not highest_recall_row and not best_balanced_row:
        lines.append("- No model summary rows were available.")
    lines.append("")
    lines.append("## Publication Readiness Note")
    if best_balanced_row:
        lines.append(f"- `{display_model_name(best_balanced_row.get('model_name'))}` is the strongest paper-usable detector in this run because it preserves high recall without collapsing precision.")
        if highest_recall_row and highest_recall_row.get("model_name") != best_balanced_row.get("model_name"):
            lines.append(f"- `{display_model_name(highest_recall_row.get('model_name'))}` remains the highest-recall option, but its false-alarm burden is materially higher.")
    else:
        lines.append("- Publication-readiness could not be determined because no summary rows were available.")
    lines.append("")
    lines.append("## Model Summary Table")
    lines.append(_to_markdown(summary_md_df))
    lines.append("")
    lines.append("## Threshold Sweep Highlights")
    lines.append(_to_markdown(threshold_md_df.sort_values(['val_recall', 'val_precision', 'val_f1'], ascending=False).head(20)))
    lines.append("")
    lines.append("## Top Feature Separability")
    lines.append(_to_markdown(diagnostics.feature_ranking.head(20)))
    lines.append("")
    lines.append("## Per-Scenario Metrics")
    lines.append(_to_markdown(per_scenario_md_df))
    lines.append("")
    lines.append("## Class Balance By Split")
    class_balance_path = report_root / "class_balance_table.csv"
    lines.append(_to_markdown(pd.read_csv(class_balance_path)) if class_balance_path.exists() else "Class balance table unavailable.")
    lines.append("")
    lines.append("## Attack Window Coverage")
    scenario_count_path = report_root / "attack_window_counts_per_scenario.csv"
    lines.append(_to_markdown(pd.read_csv(scenario_count_path)) if scenario_count_path.exists() else "Attack window count table unavailable.")
    lines.append("")
    lines.append("## Detection Latency")
    lines.append("Latency is anchored to detector emission time when present; otherwise offline windowed detectors use `window_end_utc`.")
    lines.append("")
    latency_path = report_root / "detection_latency_analysis_full.csv"
    lines.append(_to_markdown(apply_model_display_names(pd.read_csv(latency_path))) if latency_path.exists() else "Latency table unavailable.")
    lines.append("")
    lines.append("## Window Sweep Check")
    window_path = report_root / "window_sweep_table.csv"
    lines.append(_to_markdown(pd.read_csv(window_path)) if window_path.exists() else "Window sweep table unavailable.")
    lines.append("")
    lines.append("## Context Builder")
    lines.append(
        f"- Structured context summaries were generated for `{context_summary_count}` aligned windows and saved to `{context_paths[0]}`."
    )
    lines.append(
        "- Each summary includes top deviating signals, mapped assets/components, environmental context, feeder context, voltage violations, PV availability-vs-dispatch checks, and BESS SOC consistency checks."
    )
    lines.append("")
    lines.append("## Optional Context-Aware Reasoning")
    lines.append(
        f"- Local prompt-style reasoning outputs were generated for `{reasoning_summary_count}` windows and saved to `{reasoning_paths[0]}`."
    )
    lines.append(
        "- This reasoning layer is optional and does not replace the primary detector. It produces likely anomaly family, explanation text, confidence score, and likely affected assets."
    )
    lines.append("")
    if not fusion_summary_df.empty:
        lines.append("## Fusion Ablation")
        lines.append(
            "- Fusion combines detector anomaly scores with optional reasoning confidence and token-model scores using validation-fitted logistic calibration."
        )
        lines.append(_to_markdown(fusion_summary_df))
        lines.append("")
    lines.append("## Output Paths")
    for relative_path in [
        "residual_windows_full_run.parquet",
        "model_summary_table_full.csv",
        "model_summary_table_full.md",
        "per_scenario_metrics.csv",
        "threshold_sweep_table.csv",
        "feature_separability_table.csv",
        "feature_importance_table.csv",
        "detection_latency_analysis_full.csv",
        "model_comparison_plot_full.png",
        "latency_by_scenario.png",
        "phase1_context_summaries.jsonl",
        "phase1_context_summaries.parquet",
        "phase1_context_reasoning_outputs.jsonl",
        "phase1_context_reasoning_outputs.parquet",
        "fusion_ablation_table.csv",
        "fusion_ablation_plot.png",
        "phase1_upgrade_overview_panel.png",
        "paper_figures_manifest_full.json",
        "paper_tables_manifest_full.json",
    ]:
        lines.append(f"- `{report_root / relative_path}`")
    lines.append("")
    (report_root / "model_report_full.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n|---|\n"
    header = "| " + " | ".join(str(column) for column in df.columns) + " |"
    divider = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = [header, divider]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
