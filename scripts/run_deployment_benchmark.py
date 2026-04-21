"""Run the offline lightweight deployment benchmark for DERGuardian.

The benchmark loads frozen detector packages, measures CPU inference latency,
memory, package size, and throughput on replay windows, and writes deployment
reports and figures. It is an offline workstation benchmark only and does not
claim edge-hardware or field deployment evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import psutil
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methodology_alignment_common import file_size_mb, pretty_label, write_markdown
from phase1_models.model_loader import load_phase1_package, run_phase1_inference


RESULTS_PATH = ROOT / "deployment_benchmark_results.csv"
REPORT_PATH = ROOT / "deployment_benchmark_report.md"
READINESS_PATH = ROOT / "deployment_readiness_summary.md"
ENVIRONMENT_MANIFEST_PATH = ROOT / "deployment_environment_manifest.md"
REPRO_COMMAND_PATH = ROOT / "deployment_repro_command.sh"
LATENCY_FIG = ROOT / "deployment_latency_by_model.png"
MEMORY_FIG = ROOT / "deployment_memory_by_model.png"
THROUGHPUT_FIG = ROOT / "deployment_throughput_by_model.png"
TRADEOFF_FIG = ROOT / "deployment_accuracy_latency_tradeoff.png"

SCRIPT_PATH = Path(__file__).resolve()
REPLAY_INPUT_ROOT = ROOT / "outputs" / "window_size_study" / "improved_phase3" / "replay_inputs"


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _hardware_info() -> dict[str, Any]:
    """Collect host hardware details for the deployment environment manifest."""

    cpu_name = platform.processor()
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
            text=True,
        )
        if output.strip():
            cpu_name = output.strip()
    except Exception:
        pass
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_name": cpu_name,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_memory_gb": round(vm.total / (1024.0 ** 3), 2),
    }


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in ["torch", "transformers", "scikit_learn", "pandas", "numpy", "psutil"]:
        module_name = "sklearn" if package_name == "scikit_learn" else package_name
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
        except Exception:
            version = "unavailable"
        versions[package_name] = str(version)
    try:
        import tsfm_public

        versions["tsfm_public"] = getattr(tsfm_public, "__version__", "unknown")
    except Exception:
        versions["tsfm_public"] = "unavailable"
    return versions


def _model_specs() -> list[dict[str, Any]]:
    """Select frozen packages used by the offline deployment benchmark."""

    comparison = pd.read_csv(ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv")
    wanted = [
        ("threshold_baseline", "10s", "workstation_cpu"),
        ("transformer", "60s", "workstation_cpu"),
        ("lstm", "300s", "workstation_cpu"),
        ("threshold_baseline", "10s", "constrained_single_thread"),
        ("transformer", "60s", "constrained_single_thread"),
        ("lstm", "300s", "constrained_single_thread"),
    ]
    specs = []
    for model_name, window_label, profile in wanted:
        row = comparison.loc[(comparison["model_name"] == model_name) & (comparison["window_label"] == window_label)].iloc[0]
        specs.append(
            {
                "model_name": model_name,
                "window_label": window_label,
                "window_seconds": int(row["window_seconds"]),
                "ready_package_dir": str(row["ready_package_dir"]),
                "benchmark_f1": float(row["f1"]),
                "profile": profile,
            }
        )
    return specs


def _windows_paths(window_label: str) -> list[Path]:
    return sorted(REPLAY_INPUT_ROOT.glob(f"*\\{window_label}\\residual_windows.parquet"))


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Measure one model/profile pair in a fresh subprocess."""

    import torch

    if spec["profile"] == "constrained_single_thread":
        # This approximates a constrained CPU profile; it is not a claim that
        # the models ran on real edge DER hardware.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    process = psutil.Process(os.getpid())
    baseline_rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
    load_start = time.perf_counter()
    package = load_phase1_package(spec["ready_package_dir"])
    load_seconds = time.perf_counter() - load_start
    rss_after_load_mb = process.memory_info().rss / (1024.0 * 1024.0)

    window_paths = _windows_paths(spec["window_label"])
    if not window_paths:
        raise FileNotFoundError(f"No replay residual windows found for {spec['window_label']} under {REPLAY_INPUT_ROOT}")
    replay_frames = [pd.read_parquet(path) for path in window_paths]
    windows_df = pd.concat(replay_frames, ignore_index=True)
    if len(windows_df) > 2000:
        benchmark_df = windows_df.head(2000).copy()
    else:
        benchmark_df = windows_df.copy()
    warmup_df = benchmark_df.head(min(len(benchmark_df), 32)).copy()
    if not warmup_df.empty:
        _ = run_phase1_inference(package, warmup_df)

    started = time.perf_counter()
    result = run_phase1_inference(package, benchmark_df)
    elapsed_seconds = time.perf_counter() - started
    rss_after_inference_mb = process.memory_info().rss / (1024.0 * 1024.0)
    peak_rss_mb = max(baseline_rss_mb, rss_after_load_mb, rss_after_inference_mb)

    predictions = result["predictions"].copy()
    if predictions.empty:
        raise RuntimeError(
            f"No scored windows were produced for {spec['model_name']} @ {spec['window_label']}; replay slice did not satisfy sequence requirements."
        )
    if "attack_present" in predictions.columns:
        truth = predictions["attack_present"].astype(int).to_numpy()
    else:
        truth = benchmark_df["attack_present"].astype(int).tail(len(predictions)).to_numpy()
    predicted = predictions["predicted"].astype(int).to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)

    mean_ms_per_window = (elapsed_seconds / max(len(benchmark_df), 1)) * 1000.0
    throughput = len(benchmark_df) / max(elapsed_seconds, 1e-9)

    return {
        "profile": spec["profile"],
        "model_name": spec["model_name"],
        "window_label": spec["window_label"],
        "window_seconds": spec["window_seconds"],
        "ready_package_dir": spec["ready_package_dir"],
        "replay_window_paths": ";".join(str(path) for path in window_paths),
        "replay_source_count": len(window_paths),
        "benchmarked_window_count": int(len(benchmark_df)),
        "scored_window_count": int(len(predictions)),
        "load_time_seconds": round(load_seconds, 6),
        "cpu_inference_seconds": round(elapsed_seconds, 6),
        "mean_cpu_inference_ms_per_window": round(mean_ms_per_window, 6),
        "throughput_windows_per_sec": round(throughput, 6),
        "rss_baseline_mb": round(baseline_rss_mb, 3),
        "rss_after_load_mb": round(rss_after_load_mb, 3),
        "rss_after_inference_mb": round(rss_after_inference_mb, 3),
        "rss_peak_mb": round(peak_rss_mb, 3),
        "model_package_size_mb": round(file_size_mb(Path(spec["ready_package_dir"])), 3),
        "replay_precision": round(float(precision), 6),
        "replay_recall": round(float(recall), 6),
        "replay_f1": round(float(f1), 6),
        "benchmark_f1_reference": round(float(spec["benchmark_f1"]), 6),
    }


def _run_worker_subprocess(spec: dict[str, Any]) -> dict[str, Any]:
    cmd = [sys.executable, str(SCRIPT_PATH), "--worker-spec", json.dumps(spec)]
    output = subprocess.check_output(cmd, text=True, cwd=str(ROOT))
    return json.loads(output)


def _plot_figures(results_df: pd.DataFrame) -> None:
    actual = results_df[results_df["profile"] == "workstation_cpu"].copy()
    order = actual.sort_values("mean_cpu_inference_ms_per_window")["model_name"].tolist()
    labels = [f"{name} ({window})" for name, window in zip(actual["model_name"], actual["window_label"])]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, actual["mean_cpu_inference_ms_per_window"], color="#4c78a8")
    ax.set_ylabel("Mean CPU inference ms / window")
    ax.set_title("Offline deployment latency by model")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(LATENCY_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, actual["rss_peak_mb"], color="#f58518")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Offline deployment memory by model")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(MEMORY_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, actual["throughput_windows_per_sec"], color="#54a24b")
    ax.set_ylabel("Windows / sec")
    ax.set_title("Offline deployment throughput by model")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(THROUGHPUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    scatter = ax.scatter(
        actual["mean_cpu_inference_ms_per_window"],
        actual["replay_f1"],
        s=actual["model_package_size_mb"] * 12 + 40,
        c=["#4c78a8", "#1b9e77", "#e45756"],
        alpha=0.85,
    )
    _ = scatter
    for _, row in actual.iterrows():
        ax.text(
            row["mean_cpu_inference_ms_per_window"] + 0.15,
            row["replay_f1"] + 0.01,
            f"{row['model_name']} ({row['window_label']})",
            fontsize=8,
        )
    ax.set_xlabel("Mean CPU inference ms / window")
    ax.set_ylabel("Replay F1 on sampled canonical replay")
    ax.set_title("Accuracy vs latency tradeoff")
    fig.tight_layout()
    fig.savefig(TRADEOFF_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_reports(results_df: pd.DataFrame, hardware: dict[str, Any]) -> None:
    actual = results_df[results_df["profile"] == "workstation_cpu"].copy()
    constrained = results_df[results_df["profile"] == "constrained_single_thread"].copy()
    fastest = actual.sort_values("mean_cpu_inference_ms_per_window").iloc[0]
    strongest = actual.sort_values("replay_f1", ascending=False).iloc[0]
    smallest = actual.sort_values("model_package_size_mb").iloc[0]

    report_lines = [
        "# Deployment Benchmark Report",
        "",
        "This is an offline lightweight deployment benchmark on the current workspace machine. No real edge hardware was used.",
        "",
        "## Hardware used",
        "",
        f"- Platform: {hardware['platform']}",
        f"- CPU: {hardware['cpu_name']}",
        f"- Physical cores: {hardware['physical_cores']}",
        f"- Logical cores: {hardware['logical_cores']}",
        f"- System memory (GB): {hardware['total_memory_gb']}",
        "",
        "## Method",
        "",
        "- Saved frozen detector packages only",
        "- Offline replay residual windows drawn from the existing heldout, repaired, and human-authored replay inputs already present in the workspace",
        "- Up to 2,000 windows per model/profile for bounded offline timing",
        "- Measured on CPU with separate subprocesses to isolate memory readings",
        "",
        "## Results",
        "",
        "```text",
        results_df.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        f"- Fastest sampled CPU path: `{fastest['model_name']}` at `{fastest['window_label']}` with {fastest['mean_cpu_inference_ms_per_window']:.3f} ms/window.",
        f"- Strongest sampled replay F1 in this offline benchmark: `{strongest['model_name']}` at `{strongest['window_label']}` with F1={strongest['replay_f1']:.3f}.",
        f"- Smallest package on disk: `{smallest['model_name']}` at `{smallest['window_label']}` with {smallest['model_package_size_mb']:.2f} MB.",
        "",
        "The repository now supports an offline lightweight deployment benchmark. It still does not justify a real edge-hardware claim or a field-deployment claim.",
    ]
    write_markdown(REPORT_PATH, "\n".join(report_lines))

    readiness_lines = [
        "# Deployment Readiness Summary",
        "",
        "## What is now supported",
        "",
        "- Frozen model packages can be replay-benchmarked on CPU for latency, memory, package size, and throughput.",
        "- The benchmark includes the canonical winner (`transformer @ 60s`) plus comparison baselines (`threshold_baseline @ 10s`, `lstm @ 300s`).",
        "- A constrained single-thread profile is included as a lightweight approximation, not as a true edge-hardware measurement.",
        "",
        "## What is still not supported as a claim",
        "",
        "- No real edge device was used.",
        "- No field deployment or live DER gateway trial was performed.",
        "- Results are offline replay measurements only.",
        "",
        "## Bottom line",
        "",
        "Safe wording: `offline lightweight deployment benchmark` or `offline replay-oriented deployment feasibility benchmark`.",
    ]
    write_markdown(READINESS_PATH, "\n".join(readiness_lines))


def _write_environment_manifest(results_df: pd.DataFrame, hardware: dict[str, Any], versions: dict[str, str]) -> None:
    lines = [
        "# Deployment Environment Manifest",
        "",
        "## Host machine",
        "",
        f"- Platform: {hardware['platform']}",
        f"- Python: {hardware['python_version']}",
        f"- CPU: {hardware['cpu_name']}",
        f"- Physical cores: {hardware['physical_cores']}",
        f"- Logical cores: {hardware['logical_cores']}",
        f"- System memory (GB): {hardware['total_memory_gb']}",
        "",
        "## Library versions",
        "",
    ]
    for name, version in versions.items():
        lines.append(f"- {name}: {version}")
    lines.extend(
        [
            "",
            "## Benchmark scope",
            "",
            "- This benchmark used offline replay inputs already present in the workspace.",
            "- No edge hardware was used.",
            "- TTM is not included in this deployment benchmark because it currently exists as an extension checkpoint rather than a frozen Phase 1 ready-package runtime contract.",
            "",
            "## Benchmarked frozen packages",
            "",
        ]
    )
    for _, row in results_df[results_df["profile"] == "workstation_cpu"].iterrows():
        lines.append(
            f"- {row['model_name']} @ {row['window_label']}: package=`{row['ready_package_dir']}`, size_mb=`{row['model_package_size_mb']}`"
        )
    write_markdown(ENVIRONMENT_MANIFEST_PATH, "\n".join(lines))


def _write_repro_command() -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Offline lightweight deployment benchmark on the current workspace machine.",
        "python scripts/run_deployment_benchmark.py",
    ]
    REPRO_COMMAND_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entrypoint for worker mode or the full offline benchmark run."""

    parser = argparse.ArgumentParser(description="Run offline deployment-oriented replay benchmarks.")
    parser.add_argument("--worker-spec")
    args = parser.parse_args()

    if args.worker_spec:
        spec = json.loads(args.worker_spec)
        print(json.dumps(_worker(spec)))
        return

    specs = _model_specs()
    rows = [_run_worker_subprocess(spec) for spec in specs]
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH, index=False)

    hardware = _hardware_info()
    versions = _library_versions()
    _plot_figures(results_df)
    _write_reports(results_df, hardware)
    _write_environment_manifest(results_df, hardware, versions)
    _write_repro_command()

    print(
        json.dumps(
            {
                "results": _safe_rel(RESULTS_PATH),
                "report": _safe_rel(REPORT_PATH),
                "readiness": _safe_rel(READINESS_PATH),
                "environment_manifest": _safe_rel(ENVIRONMENT_MANIFEST_PATH),
                "repro_command": _safe_rel(REPRO_COMMAND_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
