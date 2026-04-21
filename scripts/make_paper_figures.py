from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_figure_suite.generate_ieee_results_package import (  # noqa: E402
    MODEL_LABELS,
    MODEL_SEQUENCE,
    _build_metrics_table,
    _discover_prediction_artifacts,
)
from paper_figure_suite.shared_style import configure_matplotlib  # noqa: E402


OUT_DIR = ROOT / "outputs" / "paper_figures"
EXPORT_SUFFIXES = (".png", ".pdf", ".svg")

CORE_TABLE_PATH = ROOT / "outputs" / "ieee_results_package" / "tables" / "table01_detector_comparison.csv"
PER_SCENARIO_METRICS_PATH = ROOT / "outputs" / "reports" / "phase3_zero_day_artifacts" / "zero_day_per_scenario_metrics.csv"
ATTACK_LABELS_PATH = ROOT / "outputs" / "attacked" / "attack_labels.parquet"
ATTACKED_TRACE_PATH = ROOT / "outputs" / "attacked" / "measured_physical_timeseries.parquet"
CLEAN_TRACE_PATH = ROOT / "outputs" / "clean" / "measured_physical_timeseries.parquet"
LLM_VALIDATION_PATH = ROOT / "phase2_llm_benchmark" / "final_results" / "llm_validation_comparison.csv"
LLM_DIVERSITY_PATH = ROOT / "phase2_llm_benchmark" / "final_results" / "llm_diversity_comparison.csv"
DETECTOR_BY_LLM_PATH = ROOT / "phase2_llm_benchmark" / "final_results" / "detector_performance_by_llm.csv"
XAI_SUMMARY_PATH = ROOT / "phase2_llm_benchmark" / "final_results" / "xai_comparison_by_llm_v4.csv"

CORE_MODELS = ["lstm", "gru", "transformer", "autoencoder", "threshold_baseline"]
CURVE_MODELS = ["lstm", "gru", "transformer"]
LLM_ORDER = ["chatgpt", "claude", "gemini", "grok"]
LLM_LABELS = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "grok": "Grok",
}
DETECTOR_COLORS = {
    "lstm": "#0072B2",
    "gru": "#009E73",
    "transformer": "#CC79A7",
    "autoencoder": "#D55E00",
    "threshold_baseline": "#E69F00",
    "isolation_forest": "#4D4D4D",
}
LLM_COLORS = {
    "chatgpt": "#0072B2",
    "claude": "#009E73",
    "gemini": "#E69F00",
    "grok": "#CC79A7",
}
REP_CASE = {
    "scenario_id": "scn_coordinated_pv83_reg4a_disturbance",
    "scenario_title": "Coordinated PV83 Curtailment and Regulator Shift",
    "voltage_signal": "bus_83_v_pu_phase_a",
    "power_signal": "pv_pv83_p_kw",
    "third_signal": "feeder_q_kvar_total",
    "third_title": "Feeder reactive power",
    "third_ylabel": "Reactive power (kvar)",
    "detector": "lstm",
}

USED_FILES: set[Path] = set()


def _record(path: Path) -> Path:
    USED_FILES.add(path.resolve())
    return path


def _style_axes(ax: plt.Axes, ygrid: bool = False) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)
    if ygrid:
        ax.grid(True, axis="y", color="#D0D7DE", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, stem: str, dpi: int = 400) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix in EXPORT_SUFFIXES:
        path = OUT_DIR / f"{stem}{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    return saved


def _load_attack_labels() -> pd.DataFrame:
    labels = pd.read_parquet(_record(ATTACK_LABELS_PATH))
    labels["start_time_utc"] = pd.to_datetime(labels["start_time_utc"], utc=True)
    labels["end_time_utc"] = pd.to_datetime(labels["end_time_utc"], utc=True)
    return labels


def _load_artifacts() -> dict[str, object]:
    return {name: _discover_prediction_artifacts(name) for name in MODEL_SEQUENCE}


def _load_metric_df() -> tuple[pd.DataFrame, dict[str, object]]:
    labels = _load_attack_labels()
    artifacts = _load_artifacts()
    metric_df, _ = _build_metrics_table(artifacts, labels)
    metric_df = metric_df.loc[metric_df["model_name"].isin(CORE_MODELS)].copy()
    return metric_df, artifacts


def _validate_metric_table(metric_df: pd.DataFrame) -> None:
    published = pd.read_csv(_record(CORE_TABLE_PATH))
    published = published.loc[published["Model"].isin(metric_df["Model"])].copy()
    merged = metric_df.merge(published, on="Model", suffixes=("_raw", "_table"), how="inner")
    if len(merged) != len(metric_df):
        raise RuntimeError("Core detector table coverage mismatch between raw predictions and published table.")
    numeric = ["Precision", "Recall", "F1", "AUPRC", "ROC-AUC", "Mean first-detection delay (s)"]
    tolerance = 5e-4
    errors: list[str] = []
    for column in numeric:
        diff = (merged[f"{column}_raw"] - merged[f"{column}_table"]).abs()
        bad = merged.loc[diff > tolerance, ["Model", f"{column}_raw", f"{column}_table"]]
        for _, row in bad.iterrows():
            errors.append(
                f"{row['Model']} {column}: raw={row[f'{column}_raw']:.6f}, table={row[f'{column}_table']:.6f}"
            )
    if errors:
        raise RuntimeError("Metric validation mismatch detected:\n" + "\n".join(errors))


def _false_positive_counts(artifacts: dict[str, object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model_name in CORE_MODELS:
        frame = artifacts[model_name].frame
        counts[model_name] = int(((frame["attack_present"] == 0) & (frame["predicted"] == 1)).sum())
    return counts


def _scenario_spread() -> pd.DataFrame:
    spread = pd.read_csv(_record(PER_SCENARIO_METRICS_PATH))
    spread = spread.loc[spread["model_name"].isin(CORE_MODELS)].copy()
    grouped = (
        spread.groupby("model_name", as_index=False)["f1"]
        .agg(["std", "count"])
        .reset_index()
        .rename(columns={"std": "f1_std", "count": "scenario_count"})
    )
    return grouped


def _event_lines(ax: plt.Axes, attack_minutes: float, detection_minutes: float) -> None:
    ax.axvspan(0.0, attack_minutes, color="#EFE7D6", alpha=0.55, zorder=0)
    ax.axvline(0.0, color="#B22222", linestyle="--", linewidth=1.1)
    ax.axvline(attack_minutes, color="#222222", linestyle="--", linewidth=1.0)
    ax.axvline(detection_minutes, color="#117A65", linestyle="--", linewidth=1.1)


def _load_representative_case(artifacts: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    labels = _load_attack_labels()
    row = labels.loc[labels["scenario_id"] == REP_CASE["scenario_id"]]
    if row.empty:
        raise RuntimeError(f"Representative scenario `{REP_CASE['scenario_id']}` not found in attack labels.")
    case = dict(REP_CASE)
    case.update(row.iloc[0].to_dict())
    case["attack_start_utc"] = pd.to_datetime(case["start_time_utc"], utc=True)
    case["attack_end_utc"] = pd.to_datetime(case["end_time_utc"], utc=True)

    attacked = pd.read_parquet(
        _record(ATTACKED_TRACE_PATH),
        columns=["timestamp_utc", "simulation_index", case["voltage_signal"], case["power_signal"], case["third_signal"]],
    )
    clean = pd.read_parquet(
        _record(CLEAN_TRACE_PATH),
        columns=["simulation_index", case["voltage_signal"], case["power_signal"], case["third_signal"]],
    )
    attacked["timestamp_utc"] = pd.to_datetime(attacked["timestamp_utc"], utc=True)
    window = attacked.loc[
        (attacked["timestamp_utc"] >= case["attack_start_utc"] - pd.Timedelta(minutes=3))
        & (attacked["timestamp_utc"] <= case["attack_end_utc"] + pd.Timedelta(minutes=4)),
        ["timestamp_utc", "simulation_index", case["voltage_signal"], case["power_signal"], case["third_signal"]],
    ].copy()
    traces = window.merge(clean, on="simulation_index", how="left", suffixes=("_attacked", "_clean"))
    traces["minutes_from_attack"] = (traces["timestamp_utc"] - case["attack_start_utc"]).dt.total_seconds() / 60.0

    pred_path = ROOT / "outputs" / "phase3_zero_day" / "zero_day_models" / case["scenario_id"] / case["detector"] / "predictions.parquet"
    results_path = ROOT / "outputs" / "phase3_zero_day" / "zero_day_models" / case["scenario_id"] / case["detector"] / "results.json"
    pred = pd.read_parquet(_record(pred_path))
    results = json.loads(_record(results_path).read_text(encoding="utf-8"))
    threshold = float(results["metrics"]["threshold"])
    pred["window_end_utc"] = pd.to_datetime(pred["window_end_utc"], utc=True)
    pred = pred.loc[pred["split_name"] == "test", ["window_end_utc", "score", "predicted"]].copy()
    pred = pred.loc[
        (pred["window_end_utc"] >= case["attack_start_utc"] - pd.Timedelta(minutes=3))
        & (pred["window_end_utc"] <= case["attack_end_utc"] + pd.Timedelta(minutes=4))
    ].copy()
    pred["threshold"] = threshold
    pred["minutes_from_attack"] = (pred["window_end_utc"] - case["attack_start_utc"]).dt.total_seconds() / 60.0

    first_detection = pred.loc[
        (pred["predicted"] == 1) & (pred["window_end_utc"] >= case["attack_start_utc"]),
        "window_end_utc",
    ]
    if first_detection.empty:
        raise RuntimeError(f"No post-onset detection found for `{case['scenario_id']}`.")
    case["first_detection_utc"] = first_detection.min()
    case["detection_delay_s"] = float((case["first_detection_utc"] - case["attack_start_utc"]).total_seconds())
    return case, traces, pred


def make_benchmark_figure(metric_df: pd.DataFrame, artifacts: dict[str, object]) -> dict[str, object]:
    spread = _scenario_spread()
    fp_counts = _false_positive_counts(artifacts)
    plot_df = metric_df.merge(spread, on="model_name", how="left").sort_values("F1", ascending=False).reset_index(drop=True)

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.1), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()
    x = np.arange(len(plot_df))

    _style_axes(ax_a, ygrid=True)
    ax_a.bar(
        x,
        [fp_counts[str(model)] for model in plot_df["model_name"]],
        color=[DETECTOR_COLORS[str(model)] for model in plot_df["model_name"]],
        edgecolor="white",
        linewidth=0.8,
        width=0.66,
    )
    ax_a.set_ylabel("False alarms")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(plot_df["Model"], rotation=24, ha="right")
    ax_a.set_title("(a) False alarms by model", loc="left", fontsize=10.6)

    _style_axes(ax_b, ygrid=True)
    width = 0.34
    ax_b.bar(x - width / 2, plot_df["Precision"], width=width, color="#7FB3D5", edgecolor="white", linewidth=0.8, label="Precision")
    ax_b.bar(x + width / 2, plot_df["Recall"], width=width, color="#F0B27A", edgecolor="white", linewidth=0.8, label="Recall")
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("Score")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(plot_df["Model"], rotation=24, ha="right")
    ax_b.set_title("(b) Precision and recall", loc="left", fontsize=10.6)
    ax_b.legend(frameon=False, fontsize=8.2, loc="upper right")

    _style_axes(ax_c, ygrid=True)
    ax_c.bar(
        x,
        plot_df["F1"],
        yerr=plot_df["f1_std"].fillna(0.0),
        capsize=2.5,
        color=[DETECTOR_COLORS[str(model)] for model in plot_df["model_name"]],
        edgecolor="white",
        linewidth=0.8,
        width=0.66,
        error_kw={"elinewidth": 0.9, "ecolor": "#444444"},
    )
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_ylabel("F1-score")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(plot_df["Model"], rotation=24, ha="right")
    ax_c.set_title("(c) F1-score with scenario spread", loc="left", fontsize=10.6)

    _style_axes(ax_d, ygrid=True)
    ax_d.bar(
        x,
        plot_df["Mean first-detection delay (s)"],
        color=[DETECTOR_COLORS[str(model)] for model in plot_df["model_name"]],
        edgecolor="white",
        linewidth=0.8,
        width=0.66,
    )
    ax_d.set_ylabel("Detection delay (s)")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(plot_df["Model"], rotation=24, ha="right")
    ax_d.set_title("(d) Detection delay", loc="left", fontsize=10.6)

    fig.suptitle("DERGuardian benchmark performance across core detectors", fontsize=12.2, y=1.01)
    _save_figure(fig, "results")
    _save_figure(fig, "fig03_benchmark_performance")
    plt.close(fig)
    return {
        "false_alarms": fp_counts,
        "ordered_models": plot_df["model_name"].tolist(),
    }


def make_physical_impact_figure(artifacts: dict[str, object]) -> dict[str, object]:
    case, traces, pred = _load_representative_case(artifacts)
    attack_minutes = float((case["attack_end_utc"] - case["attack_start_utc"]).total_seconds() / 60.0)
    detection_minutes = float(case["detection_delay_s"] / 60.0)

    configure_matplotlib()
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.2), sharex=True, constrained_layout=False)
    fig.subplots_adjust(top=0.88, hspace=0.22)
    plotted = [
        (case["voltage_signal"], "Bus 83 voltage magnitude", "Voltage (pu)"),
        (case["power_signal"], "DER active power", "Active power (kW)"),
        (case["third_signal"], case["third_title"], case["third_ylabel"]),
    ]
    panel_labels = ["(a)", "(b)", "(c)"]
    for ax, (signal, title, ylabel), panel in zip(axes[:3], plotted, panel_labels):
        _style_axes(ax, ygrid=False)
        _event_lines(ax, attack_minutes, detection_minutes)
        ax.plot(traces["minutes_from_attack"], traces[f"{signal}_clean"], color="#4C72B0", linewidth=1.25, label="Clean baseline")
        ax.plot(traces["minutes_from_attack"], traces[f"{signal}_attacked"], color="#C44E52", linewidth=1.25, label="Attacked trajectory")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel} {title}", loc="left", fontsize=10.2)

    ax_score = axes[3]
    _style_axes(ax_score, ygrid=False)
    _event_lines(ax_score, attack_minutes, detection_minutes)
    ax_score.plot(pred["minutes_from_attack"], pred["score"], color=DETECTOR_COLORS[case["detector"]], linewidth=1.35, label=f"{MODEL_LABELS[case['detector']]} score")
    ax_score.plot(pred["minutes_from_attack"], pred["threshold"], color="#444444", linestyle="--", linewidth=1.0, label="Threshold")
    ax_score.set_ylabel("Score")
    ax_score.set_xlabel("Minutes relative to attack start")
    ax_score.set_title("(d) Anomaly score and threshold", loc="left", fontsize=10.2)

    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="#4C72B0", lw=1.4, label="Clean baseline"),
            plt.Line2D([0], [0], color="#C44E52", lw=1.4, label="Attacked trajectory"),
            plt.Line2D([0], [0], color="#B22222", lw=1.1, ls="--", label="Attack start"),
            plt.Line2D([0], [0], color="#222222", lw=1.0, ls="--", label="Attack end"),
            plt.Line2D([0], [0], color="#117A65", lw=1.1, ls="--", label="First detection"),
        ],
        frameon=False,
        fontsize=7.6,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle("Representative physical impact and detection timing", fontsize=12.0, y=0.975)
    _save_figure(fig, "fig04_physical_impact_detection")
    plt.close(fig)
    return case


def make_pr_curve_figure(artifacts: dict[str, object], metric_df: pd.DataFrame) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    _style_axes(ax, ygrid=False)
    metric_lookup = metric_df.set_index("model_name")
    for model_name in CURVE_MODELS:
        frame = artifacts[model_name].frame
        y_true = frame["attack_present"].to_numpy()
        y_score = frame["score"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = float(average_precision_score(y_true, y_score))
        if abs(ap - float(metric_lookup.loc[model_name, "AUPRC"])) > 1e-9:
            raise RuntimeError(f"PR validation mismatch for {model_name}.")
        ax.plot(recall, precision, linewidth=2.0, color=DETECTOR_COLORS[model_name], label=f"{MODEL_LABELS[model_name]} (AP={ap:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall characteristics for strongest detectors", loc="left", fontsize=10.8)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")
    _save_figure(fig, "fig05_precision_recall_curves")
    plt.close(fig)


def make_cross_llm_figure() -> pd.DataFrame:
    validation = pd.read_csv(_record(LLM_VALIDATION_PATH)).rename(columns={"model_name": "llm_name"})
    diversity = pd.read_csv(_record(LLM_DIVERSITY_PATH)).rename(columns={"model_name": "llm_name"})
    detector = pd.read_csv(_record(DETECTOR_BY_LLM_PATH))
    best = detector.sort_values(["llm_name", "f1"], ascending=[True, False]).groupby("llm_name", as_index=False).first()
    merged = validation.merge(diversity, on="llm_name", how="inner").merge(best[["llm_name", "detector", "f1"]], on="llm_name", how="inner")
    merged["llm_name"] = pd.Categorical(merged["llm_name"], categories=LLM_ORDER, ordered=True)
    merged = merged.sort_values("llm_name").reset_index(drop=True)

    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    x = np.arange(len(merged))
    bar_colors = [LLM_COLORS[str(name)] for name in merged["llm_name"]]

    panels = [
        ("scenarios_valid", "Validated scenarios", "(a) Valid scenarios per LLM"),
        ("family_count", "Family coverage", "(b) Family coverage per LLM"),
        ("f1", "Best detector F1", "(c) Best detector F1 per LLM"),
    ]
    for ax, (column, ylabel, title) in zip(axes, panels):
        _style_axes(ax, ygrid=True)
        ax.bar(x, merged[column], color=bar_colors, edgecolor="white", linewidth=0.8, width=0.66)
        ax.set_xticks(x)
        ax.set_xticklabels([LLM_LABELS[str(name)] for name in merged["llm_name"]], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10.0)
    axes[2].set_ylim(0.0, 1.0)
    for idx, row in merged.iterrows():
        axes[2].text(idx, float(row["f1"]) + 0.03, MODEL_LABELS[str(row["detector"])], ha="center", va="bottom", fontsize=7.5, color="#333333")

    fig.suptitle("Cross-LLM robustness summary", fontsize=12.0, y=1.05)
    _save_figure(fig, "fig06_cross_llm_robustness")
    plt.close(fig)
    return merged


def make_xai_figure() -> pd.DataFrame:
    summary = pd.read_csv(_record(XAI_SUMMARY_PATH))
    summary["llm_name"] = pd.Categorical(summary["llm_name"], categories=LLM_ORDER, ordered=True)
    summary = summary.sort_values("llm_name").reset_index(drop=True)

    metrics = [
        ("family_accuracy", "Family top-1"),
        ("family_top3_accuracy", "Family top-3"),
        ("asset_accuracy", "Asset attribution"),
        ("evidence_grounding_rate", "Evidence grounding"),
        ("action_relevance", "Action relevance"),
    ]
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.8, 4.2), constrained_layout=True)
    _style_axes(ax, ygrid=True)
    x = np.arange(len(metrics))
    width = 0.18
    for offset, llm_name in zip(np.linspace(-1.5 * width, 1.5 * width, len(LLM_ORDER)), LLM_ORDER):
        row = summary.loc[summary["llm_name"] == llm_name].iloc[0]
        values = [float(row[column]) for column, _ in metrics]
        ax.bar(x + offset, values, width=width, color=LLM_COLORS[llm_name], edgecolor="white", linewidth=0.8, label=LLM_LABELS[llm_name])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=14, ha="right")
    ax.set_title("Audited runtime-only explanation performance", loc="left", fontsize=10.8)
    ax.legend(frameon=False, ncol=4, fontsize=8.0, loc="upper right")
    _save_figure(fig, "fig07_explanation_audit_summary")
    plt.close(fig)
    return summary


def write_readme(benchmark_info: dict[str, object], physical_case: dict[str, object], cross_llm: pd.DataFrame, xai: pd.DataFrame) -> Path:
    ordered_models = [MODEL_LABELS[name] for name in benchmark_info["ordered_models"]]
    lines = [
        "# DERGuardian Paper Figures",
        "",
        "## Generated figures",
        "- `results.(png|pdf|svg)`: core 2x2 detector benchmark summary.",
        "- `fig03_benchmark_performance.(png|pdf|svg)`: same benchmark figure with manuscript numbering.",
        "- `fig04_physical_impact_detection.(png|pdf|svg)`: representative physical impact and detection timing from the real zero-day benchmark.",
        "- `fig05_precision_recall_curves.(png|pdf|svg)`: PR curves for LSTM, GRU, and Transformer.",
        "- `fig06_cross_llm_robustness.(png|pdf|svg)`: validated scenarios, family coverage, and best-detector F1 by LLM source.",
        "- `fig07_explanation_audit_summary.(png|pdf|svg)`: audited runtime-only XAI metrics by LLM source.",
        "",
        "## Source files used",
    ]
    for path in sorted(USED_FILES):
        lines.append(f"- `{path.relative_to(ROOT).as_posix()}`")
    lines.extend(
        [
            "",
            "## Key selections",
            f"- Core benchmark detector order in Fig. 3: {', '.join(ordered_models)}.",
            f"- Representative physical scenario: `{physical_case['scenario_id']}` ({physical_case['scenario_title']}).",
            f"- Physical signals: `{physical_case['voltage_signal']}`, `{physical_case['power_signal']}`, `{physical_case['third_signal']}`.",
            f"- Attack interval: `{physical_case['attack_start_utc'].isoformat()}` to `{physical_case['attack_end_utc'].isoformat()}`.",
            f"- First detection marker: `{physical_case['first_detection_utc'].isoformat()}` from `{MODEL_LABELS[physical_case['detector']]}`.",
            "",
            "## Validation checks",
            "- Recomputed core detector metrics directly from raw zero-day prediction parquet files and verified agreement with `outputs/ieee_results_package/tables/table01_detector_comparison.csv` within 0.0005.",
            "- Recomputed PR curves from the same raw score vectors used in the core detector table.",
            "- Used raw physical trace samples from `outputs/attacked/measured_physical_timeseries.parquet` and `outputs/clean/measured_physical_timeseries.parquet`; no smoothing or synthetic interpolation was applied.",
            "- F1 error bars in Fig. 3(c) use per-scenario spread from `outputs/reports/phase3_zero_day_artifacts/zero_day_per_scenario_metrics.csv` because a complete multi-seed zero-day summary is not available for all five core detectors.",
            "",
            "## Cross-LLM summary used",
            cross_llm.to_markdown(index=False),
            "",
            "## XAI summary used",
            xai.assign(llm_name=xai["llm_name"].map(LLM_LABELS)).to_markdown(index=False),
            "",
        ]
    )
    readme_path = OUT_DIR / "README_figures.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_df, artifacts = _load_metric_df()
    _validate_metric_table(metric_df)
    benchmark_info = make_benchmark_figure(metric_df, artifacts)
    physical_case = make_physical_impact_figure(artifacts)
    make_pr_curve_figure(artifacts, metric_df)
    cross_llm = make_cross_llm_figure()
    xai = make_xai_figure()
    readme_path = write_readme(benchmark_info, physical_case, cross_llm, xai)

    print("Saved figures to:")
    for stem in [
        "results",
        "fig03_benchmark_performance",
        "fig04_physical_impact_detection",
        "fig05_precision_recall_curves",
        "fig06_cross_llm_robustness",
        "fig07_explanation_audit_summary",
    ]:
        print(f"  - outputs/paper_figures/{stem}.png/.pdf/.svg")
    print("Input files used:")
    for path in sorted(USED_FILES):
        print(f"  - {path.relative_to(ROOT).as_posix()}")
    print("Selected physical signals:")
    print(f"  - voltage: {physical_case['voltage_signal']}")
    print(f"  - DER active power: {physical_case['power_signal']}")
    print(f"  - third panel: {physical_case['third_signal']}")
    print(f"  - detection latency (s): {physical_case['detection_delay_s']:.1f}")
    print(f"README: {readme_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
