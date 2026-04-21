"""Repository orchestration script for DERGuardian.

This script runs or rebuilds generate final combined results figure artifacts for audits, figures,
reports, or reproducibility checks. It is release-support code and must preserve
the separation between canonical benchmark, replay, heldout synthetic, and
extension experiment contexts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "outputs" / "window_size_study" / "professor_graph_package"
TABLE_ROOT = PACKAGE_ROOT / "graph_data_tables"

PERFORMANCE_PATH = TABLE_ROOT / "performance_vs_model_all_llm_attacks.csv"
LATENCY_PATH = TABLE_ROOT / "latency_vs_model_all_llm_attacks.csv"
GENERATOR_MATRIX_PATH = TABLE_ROOT / "per_generator_performance_matrix.csv"
TIMELINE_TABLE_PATH = TABLE_ROOT / "timeline_detection_scenario_1.csv"
TIMELINE_SECONDS_SUMMARY_PATH = PACKAGE_ROOT / "timeline_detection_seconds_summary.csv"
SCENARIO_DIFFICULTY_PATH = TABLE_ROOT / "llm_scenario_difficulty_summary.csv"

OUTPUT_PNG = PACKAGE_ROOT / "final_combined_results_figure.png"
OUTPUT_PDF = PACKAGE_ROOT / "final_combined_results_figure.pdf"
OUTPUT_CAPTION = PACKAGE_ROOT / "final_combined_results_figure_caption.md"

MODEL_ORDER = [
    ("threshold_baseline", "10s"),
    ("transformer", "60s"),
    ("lstm", "300s"),
]
MODEL_DISPLAY = {
    ("threshold_baseline", "10s"): "Threshold",
    ("transformer", "60s"): "Transformer",
    ("lstm", "300s"): "LSTM",
}
MODEL_COLORS = {
    ("threshold_baseline", "10s"): "#7a7a7a",
    ("transformer", "60s"): "#355c7d",
    ("lstm", "300s"): "#4f7c5d",
}
GENERATOR_DISPLAY = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "grok": "Grok",
}
TIMELINE_LINESTYLES = {
    ("threshold_baseline", "10s"): ":",
    ("transformer", "60s"): "--",
    ("lstm", "300s"): "-.",
}
TIMELINE_LABELS = {
    ("threshold_baseline", "10s"): "Threshold detect",
    ("transformer", "60s"): "Transformer detect",
    ("lstm", "300s"): "LSTM detect",
}


def _configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.6,
        }
    )


def _parse_range(value: str) -> tuple[float, float]:
    parsed = ast.literal_eval(value)
    return float(parsed[0]), float(parsed[1])


def _choose_seconds_tick_step(span_seconds: float) -> int:
    if span_seconds <= 60:
        return 5
    if span_seconds <= 180:
        return 10
    if span_seconds <= 360:
        return 20
    return 30


def _build_seconds_ticks(x_min: float, x_max: float) -> np.ndarray:
    span = x_max - x_min
    step = _choose_seconds_tick_step(span)
    start = int(np.floor(x_min / step) * step)
    end = int(np.ceil(x_max / step) * step)
    return np.arange(start, end + step, step, dtype=float)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


def _load_data() -> dict[str, pd.DataFrame]:
    return {
        "performance": pd.read_csv(PERFORMANCE_PATH),
        "latency": pd.read_csv(LATENCY_PATH),
        "generator": pd.read_csv(GENERATOR_MATRIX_PATH),
        "timeline": pd.read_csv(TIMELINE_TABLE_PATH),
        "timeline_summary": pd.read_csv(TIMELINE_SECONDS_SUMMARY_PATH),
        "scenario_difficulty": pd.read_csv(
            SCENARIO_DIFFICULTY_PATH,
            parse_dates=["start_time_utc", "end_time_utc"],
        ),
    }


def _plot_panel_a(ax: plt.Axes, performance_df: pd.DataFrame) -> None:
    ordered_rows = []
    for model_name, window_label in MODEL_ORDER:
        row = performance_df.loc[
            (performance_df["model_name"] == model_name)
            & (performance_df["window_label"] == window_label)
        ].iloc[0]
        ordered_rows.append(row)
    ordered = pd.DataFrame(ordered_rows)
    x = np.arange(len(ordered))

    bars = ax.bar(
        x,
        ordered["f1"],
        width=0.56,
        color=[MODEL_COLORS[key] for key in MODEL_ORDER],
        edgecolor="#1f2937",
        linewidth=0.7,
    )
    for bar, value in zip(bars, ordered["f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + 0.025,
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[key].replace(" ", "\n", 1) for key in MODEL_ORDER])
    ax.set_ylim(0.0, 0.98)
    ax.set_ylabel("F1")
    ax.set_title("Heldout Replay F1 by Model")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_panel_b(ax: plt.Axes, latency_df: pd.DataFrame) -> None:
    ordered_rows = []
    for model_name, window_label in MODEL_ORDER:
        row = latency_df.loc[
            (latency_df["model_name"] == model_name)
            & (latency_df["window_label"] == window_label)
        ].iloc[0]
        ordered_rows.append(row)
    ordered = pd.DataFrame(ordered_rows)
    x = np.arange(len(ordered))
    values = ordered["mean_latency_seconds"].astype(float).to_numpy()

    bars = ax.bar(
        x,
        values,
        width=0.56,
        color=[MODEL_COLORS[key] for key in MODEL_ORDER],
        edgecolor="#1f2937",
        linewidth=0.7,
    )
    y_max = float(values.max()) if len(values) else 1.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + y_max * 0.02,
            f"{float(value):.1f} s",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[key].replace(" ", "\n", 1) for key in MODEL_ORDER])
    ax.set_ylabel("Seconds")
    ax.set_title("Mean Detection Latency (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_panel_c(ax: plt.Axes, generator_df: pd.DataFrame) -> None:
    ordered_cols = [f"{model_name}|{window_label}" for model_name, window_label in MODEL_ORDER]
    working = generator_df.copy()
    working["col_key"] = working["model_name"] + "|" + working["window_label"]
    pivot = working.pivot(index="generator_source", columns="col_key", values="f1").reindex(columns=ordered_cols)
    row_means = pivot.mean(axis=1).sort_values()
    pivot = pivot.reindex(index=row_means.index)

    cmap = LinearSegmentedColormap.from_list(
        "combined_heldout_blues",
        ["#f7fbff", "#dbe9f6", "#8fbad8", "#3973ac", "#0b3c73"],
    )
    image = ax.imshow(pivot.values, cmap=cmap, norm=PowerNorm(gamma=0.8, vmin=0.0, vmax=1.0), aspect="auto")
    ax.set_xticks(np.arange(len(ordered_cols)))
    ax.set_xticklabels([MODEL_DISPLAY[key].replace(" ", "\n", 1) for key in MODEL_ORDER], fontsize=8.6)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([GENERATOR_DISPLAY.get(name, name.title()) for name in pivot.index], fontsize=9.5)
    ax.set_title("Heldout F1 by Generator")
    ax.set_xlabel("Frozen Model")
    ax.set_ylabel("Generator Source")
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = float(pivot.iloc[row_idx, col_idx])
            text_color = "white" if value >= 0.72 else "#0f172a"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=9.2, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("F1 Score", fontsize=9)
    colorbar.ax.tick_params(labelsize=8.5)


def _plot_panel_d(
    ax: plt.Axes,
    timeline_df: pd.DataFrame,
    timeline_summary_df: pd.DataFrame,
    scenario_difficulty_df: pd.DataFrame,
) -> None:
    summary_row = timeline_summary_df.loc[
        timeline_summary_df["scenario_id"] == "scn_cmd_delay_pv60_curtailment_release"
    ].iloc[0]
    meta_row = scenario_difficulty_df.loc[
        scenario_difficulty_df["scenario_id"] == "scn_cmd_delay_pv60_curtailment_release"
    ].iloc[0]

    x_min, x_max = _parse_range(summary_row["chosen_zoom_plot_range_seconds"])
    ticks = _build_seconds_ticks(x_min, x_max)
    duration_seconds = float(meta_row["duration_seconds"])

    working = timeline_df.copy()
    working["relative_seconds"] = working["relative_minutes"].astype(float) * 60.0

    ax.axvspan(0.0, min(duration_seconds, x_max), color="#dbe4ee", alpha=0.45, label="_nolegend_", zorder=0)
    ax.plot(
        working["relative_seconds"],
        working["clean_value"],
        color="#303030",
        linewidth=1.9,
        linestyle="--",
        label="Clean",
    )
    ax.plot(
        working["relative_seconds"],
        working["attacked_value"],
        color="#355c7d",
        linewidth=2.2,
        label="Attacked",
    )
    ax.axvline(0.0, color="#8b1e3f", linewidth=2.2, linestyle="-", label="Attack start")

    detections = [
        (("threshold_baseline", "10s"), float(summary_row["threshold_detect_seconds"])),
        (("transformer", "60s"), float(summary_row["transformer_detect_seconds"])),
        (("lstm", "300s"), float(summary_row["lstm_detect_seconds"])),
    ]
    for model_key, latency_seconds in detections:
        if np.isnan(latency_seconds):
            continue
        label = TIMELINE_LABELS[model_key]
        line_color = MODEL_COLORS[model_key]
        line_width = 2.3
        line_alpha = 1.0
        if model_key == ("threshold_baseline", "10s"):
            label = "_nolegend_"
            line_color = "#8a8a8a"
            line_width = 1.8
            line_alpha = 0.7
        ax.axvline(
            latency_seconds,
            color=line_color,
            linewidth=line_width,
            linestyle=TIMELINE_LINESTYLES[model_key],
            label=label,
            alpha=line_alpha,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(ticks)
    ax.set_xlabel("Seconds Relative to Attack Start")
    ax.set_ylabel("feeder_p_kw_total")
    ax.set_title("Representative Detection Timeline")
    ax.grid(axis="y", linestyle="--", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.98,
        "ChatGPT | command-delay scenario",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#374151",
    )

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    ordered_handles = []
    ordered_labels = []
    for label, handle in zip(labels, handles):
        if label in seen:
            continue
        seen.add(label)
        ordered_handles.append(handle)
        ordered_labels.append(label)
    ax.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=7.4,
        columnspacing=0.9,
        handlelength=2.4,
    )


def _write_caption() -> None:
    caption = """# Final Combined Results Figure Caption

**Figure X.** Combined heldout replay summary across the accepted LLM-generated heldout attack bundles from ChatGPT, Claude, Gemini, and Grok. Panels **(A)** through **(C)** compare the threshold baseline with 10-second windows, the transformer with 60-second windows, and the LSTM with 300-second windows. **(A)** Pooled heldout replay F1 by frozen model shows that the LSTM achieves the strongest overall F1, followed by the transformer. **(B)** Mean detection latency in seconds shows that the transformer is the fastest despite not having the highest pooled F1. **(C)** Generator-source by model heatmap shows that heldout replay performance varies across LLM-generated bundles rather than remaining uniform across sources. **(D)** Representative heldout replay timeline for a command-delay scenario on `feeder_p_kw_total` illustrates the attack onset and differing model response times on a concrete physical variable; in this example, the transformer detects within approximately 1 s, while the LSTM responds at approximately 61 s, with the threshold baseline shown as a faint reference line. This figure summarizes heldout replay behavior only and does not claim real-world zero-day robustness.
"""
    OUTPUT_CAPTION.write_text(caption.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    """Run the command-line entrypoint for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    _configure_style()
    data = _load_data()

    fig = plt.figure(figsize=(13.8, 9.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.06], height_ratios=[0.95, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _plot_panel_a(ax_a, data["performance"])
    _plot_panel_b(ax_b, data["latency"])
    _plot_panel_c(ax_c, data["generator"])
    _plot_panel_d(ax_d, data["timeline"], data["timeline_summary"], data["scenario_difficulty"])

    for axis, label in zip([ax_a, ax_b, ax_c, ax_d], ["(A)", "(B)", "(C)", "(D)"]):
        _panel_label(axis, label)

    fig.savefig(OUTPUT_PNG, dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    _write_caption()


if __name__ == "__main__":
    main()
