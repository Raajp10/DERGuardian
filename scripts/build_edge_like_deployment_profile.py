"""Build a constrained edge-like offline deployment benchmark summary.

The profile is intentionally not a field deployment claim. It reuses the
existing workstation/constrained benchmark rows for frozen detector packages and
adds a measured CPU timing row for the new LSTM autoencoder extension
checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

from phase1_models.train_lstm_autoencoder import (
    LSTMSequenceAutoencoder,
    SequenceStandardizer,
    _build_sequences,
    _score_model,
)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _rss_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))


def _existing_constrained_rows() -> pd.DataFrame:
    path = ROOT / "artifacts" / "deployment" / "deployment_benchmark_results.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    keep = frame.loc[
        frame["profile"].eq("constrained_single_thread")
        & frame["model_name"].isin(["transformer", "lstm", "threshold_baseline"])
    ].copy()
    keep["profile"] = "constrained_edge_like_offline_single_thread"
    keep["benchmark_source"] = _repo_rel(path)
    keep["edge_like_constraint"] = "single-thread offline profile from existing deployment benchmark"
    return keep


def _lstm_autoencoder_deployment_row() -> pd.DataFrame:
    checkpoint_path = ROOT / "outputs" / "phase1_lstm_autoencoder_extension" / "checkpoints" / "lstm_autoencoder_300s.pt"
    residual_path = ROOT / "outputs" / "window_size_study" / "300s" / "residual_windows.parquet"
    if not checkpoint_path.exists() or not residual_path.exists():
        return pd.DataFrame()

    torch.set_num_threads(1)
    rss_before = _rss_mb()
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_columns = list(loaded["feature_columns"])
    standardizer = SequenceStandardizer(np.asarray(loaded["standardizer_mean"]), np.asarray(loaded["standardizer_std"]))
    config = loaded["config"]
    model = LSTMSequenceAutoencoder(
        input_dim=len(feature_columns),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
    )
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    rss_after_load = _rss_mb()

    frame = pd.read_parquet(residual_path)
    test_frame = frame.loc[frame["split_name"].astype(str).str.lower().eq("test")].copy()
    x, y, _ = _build_sequences(test_frame, feature_columns, standardizer, int(config["seq_len_by_window"]["300s"]), max_sequences=1000)
    start = time.perf_counter()
    scores, inference_seconds = _score_model(model, x, batch_size=128)
    wall_seconds = time.perf_counter() - start
    rss_after_inference = _rss_mb()
    threshold = float(loaded["threshold"])
    y_pred = (scores >= threshold).astype(int) if len(scores) else np.array([], dtype=int)
    tp = int(((y == 1) & (y_pred == 1)).sum()) if len(y) else 0
    fp = int(((y == 0) & (y_pred == 1)).sum()) if len(y) else 0
    fn = int(((y == 1) & (y_pred == 0)).sum()) if len(y) else 0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return pd.DataFrame(
        [
            {
                "profile": "constrained_edge_like_offline_single_thread",
                "model_name": "lstm_autoencoder",
                "window_label": "300s",
                "window_seconds": 300,
                "ready_package_dir": _repo_rel(checkpoint_path.parent),
                "replay_window_paths": _repo_rel(residual_path),
                "replay_source_count": 1,
                "benchmarked_window_count": int(len(x)),
                "scored_window_count": int(len(scores)),
                "load_time_seconds": np.nan,
                "cpu_inference_seconds": float(inference_seconds or wall_seconds),
                "mean_cpu_inference_ms_per_window": float(((inference_seconds or wall_seconds) / len(scores)) * 1000.0) if len(scores) else np.nan,
                "throughput_windows_per_sec": float(len(scores) / (inference_seconds or wall_seconds)) if (inference_seconds or wall_seconds) > 0 else np.nan,
                "rss_baseline_mb": rss_before,
                "rss_after_load_mb": rss_after_load,
                "rss_after_inference_mb": rss_after_inference,
                "rss_peak_mb": max(rss_before, rss_after_load, rss_after_inference),
                "model_package_size_mb": float(checkpoint_path.stat().st_size / (1024 * 1024)),
                "replay_precision": precision,
                "replay_recall": recall,
                "replay_f1": f1,
                "benchmark_f1_reference": 0.133971,
                "benchmark_source": _repo_rel(ROOT / "artifacts" / "extensions" / "phase1_lstm_autoencoder_results.csv"),
                "edge_like_constraint": "single-thread CPU timing of extension checkpoint on canonical 300s test residual sequences",
            }
        ]
    )


def _write_profile_doc(path: Path) -> None:
    lines = [
        "# Constrained Edge-Like Offline Deployment Profile",
        "",
        "This is an offline benchmark profile, not field edge deployment evidence.",
        "",
        "## Actual Hardware Used",
        "",
        f"- Platform: {platform.platform()}",
        f"- Processor string: {platform.processor() or 'not reported by OS'}",
        f"- CPU count visible to Python: {os.cpu_count()}",
        f"- Python: {platform.python_version()}",
        f"- PyTorch: {torch.__version__}",
        "",
        "## Constraint",
        "",
        "The profile uses single-thread CPU settings where measured during this pass or reuses the existing constrained-single-thread deployment rows.",
        "This approximates a lower-resource runtime profile but does not emulate a specific field device and does not validate field deployment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(results: pd.DataFrame, path: Path, csv_path: Path) -> None:
    display = results[
        [
            "model_name",
            "window_label",
            "mean_cpu_inference_ms_per_window",
            "throughput_windows_per_sec",
            "rss_peak_mb",
            "model_package_size_mb",
            "replay_f1",
            "edge_like_constraint",
        ]
    ].copy()
    lines = [
        "# Constrained Edge-Like Deployment Benchmark Report",
        "",
        "This report strengthens the deployment evidence as an offline, constrained profile only.",
        "It does not claim field edge deployment or hardware-in-the-loop validation.",
        "",
        display.to_markdown(index=False),
        "",
        "## Safe Wording",
        "",
        "Use: **constrained edge-like offline benchmark** or **offline lightweight deployment benchmark**.",
        "Do not use: **field edge deployment**.",
        "",
        f"Source CSV: `{_repo_rel(csv_path)}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(results: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = results["model_name"] + "@" + results["window_label"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(labels, results["mean_cpu_inference_ms_per_window"].astype(float))
    axes[0].set_ylabel("ms/window")
    axes[0].set_title("CPU inference latency")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(labels, results["rss_peak_mb"].astype(float))
    axes[1].set_ylabel("RSS peak MB")
    axes[1].set_title("Process memory footprint")
    axes[1].tick_params(axis="x", rotation=35)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    """Write constrained edge-like deployment artifacts."""

    artifact_dir = ROOT / "artifacts" / "deployment"
    docs_dir = ROOT / "docs" / "reports"
    figure_dir = ROOT / "docs" / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = [_existing_constrained_rows(), _lstm_autoencoder_deployment_row()]
    results = pd.concat([row for row in rows if not row.empty], ignore_index=True)
    csv_path = artifact_dir / "deployment_edge_like_results.csv"
    results.to_csv(csv_path, index=False)
    _write_profile_doc(docs_dir / "deployment_edge_like_profile.md")
    _write_report(results, docs_dir / "deployment_edge_like_report.md", csv_path)
    _write_figure(results, figure_dir / "deployment_edge_like_comparison.png")
    print({"deployment_edge_like_rows": int(len(results)), "csv": _repo_rel(csv_path)})


if __name__ == "__main__":
    main()
