from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import asdict
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.config import ProjectPaths, load_pipeline_config, save_pipeline_config
from common.dss_runner import extract_inventory, load_profile_specs, simulate_truth
from common.graph_builders import build_validation_graphs
from common.io_utils import write_dataframe, write_json, write_jsonl
from common.metadata_schema import RunManifest, artifact_from_path
from common.noise_models import build_measured_layer, impairment_summaries_to_frame
from common.units_and_channel_dictionary import build_channel_dictionary, write_data_dictionary
from common.weather_load_generators import (
    build_bess_schedule,
    build_load_schedule,
    build_pv_schedule,
    generate_reference_profile_bundle,
    generate_environmental_inputs,
)
from phase1.build_windows import build_merged_windows
from phase1.validate_clean_dataset import validate_clean_layers
from phase2.cyber_log_generator import generate_baseline_cyber_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the clean IEEE 123-bus DER cyber-physical dataset layers.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--simulation-resolution-seconds", type=int, default=None)
    parser.add_argument("--output-format", default=None)
    parser.add_argument("--clean-output-root", default=None)
    parser.add_argument("--windows-output-root", default=None)
    parser.add_argument("--labeled-output-root", default=None)
    parser.add_argument("--reports-output-root", default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if args.duration_hours is not None:
        config.duration_hours = args.duration_hours
    if args.simulation_resolution_seconds is not None:
        config.simulation_resolution_seconds = args.simulation_resolution_seconds
    if args.output_format is not None:
        config.output_format = args.output_format

    clean_output = Path(args.clean_output_root) if args.clean_output_root else ROOT / "outputs" / "clean"
    windows_output = Path(args.windows_output_root) if args.windows_output_root else ROOT / "outputs" / "windows"
    labeled_output = Path(args.labeled_output_root) if args.labeled_output_root else ROOT / "outputs" / "labeled"
    reports_output = Path(args.reports_output_root) if args.reports_output_root else ROOT / "outputs" / "reports"
    if not clean_output.is_absolute():
        clean_output = (ROOT / clean_output).resolve()
    if not windows_output.is_absolute():
        windows_output = (ROOT / windows_output).resolve()
    if not labeled_output.is_absolute():
        labeled_output = (ROOT / labeled_output).resolve()
    if not reports_output.is_absolute():
        reports_output = (ROOT / reports_output).resolve()

    paths = ProjectPaths(
        root=ROOT,
        clean_output=clean_output,
        windows_output=windows_output,
        labeled_output=labeled_output,
        reports_output=reports_output,
        report_file=reports_output / "generated_report.md",
    ).ensure()
    rng = np.random.default_rng(config.random_seed)
    inventory = extract_inventory(config, paths)
    coarse_env, fine_env = generate_environmental_inputs(config, paths, rng)
    load_schedule, load_class_map = build_load_schedule(config, load_profile_specs(inventory), fine_env, paths, rng)
    pv_schedule = build_pv_schedule(config, inventory.pv_assets, fine_env, rng)
    bess_schedule = build_bess_schedule(config, inventory.bess_assets, fine_env, rng)
    reference_profiles = generate_reference_profile_bundle(
        config=config,
        load_specs=load_profile_specs(inventory),
        pv_assets=inventory.pv_assets,
        bess_assets=inventory.bess_assets,
        paths=paths,
        base_seed=config.random_seed + 41,
    )
    truth_df, control_df = simulate_truth(
        config=config,
        inventory=inventory,
        env_df=fine_env,
        load_schedule_df=load_schedule,
        pv_schedule_df=pv_schedule,
        bess_schedule_df=bess_schedule,
        paths=paths,
        scenario_id=config.scenario_id,
        split_id=config.split_id,
    )
    measured_df, impairment_summary = build_measured_layer(truth_df, config, np.random.default_rng(config.random_seed + 11))
    cyber_df = generate_baseline_cyber_events(truth_df, control_df, config, np.random.default_rng(config.random_seed + 29))
    empty_labels = pd.DataFrame(
        columns=[
            "scenario_id",
            "scenario_name",
            "attack_family",
            "severity",
            "start_time_utc",
            "end_time_utc",
            "affected_assets",
            "affected_signals",
            "target_component",
        ]
    )
    windows_df = build_merged_windows(measured_df, cyber_df, empty_labels, config.windows)

    truth_path = paths.clean_output / f"truth_physical_timeseries.{config.output_format}"
    measured_path = paths.clean_output / f"measured_physical_timeseries.{config.output_format}"
    cyber_path = paths.clean_output / f"cyber_events.{config.output_format}"
    windows_path = paths.windows_output / f"merged_windows_clean.{config.output_format}"
    labels_path = paths.labeled_output / "attack_labels_clean.csv"
    impairment_path = paths.clean_output / "measurement_impairments.csv"
    config_snapshot_path = paths.clean_output / "config_snapshot.json"
    control_path = paths.clean_output / f"control_schedule.{config.output_format}"
    env_fine_path = paths.clean_output / f"input_environment_timeseries.{config.output_format}"
    env_coarse_path = paths.clean_output / f"input_environment_coarse.{config.output_format}"
    load_schedule_path = paths.clean_output / f"input_load_schedule.{config.output_format}"
    pv_schedule_path = paths.clean_output / f"input_pv_schedule.{config.output_format}"
    bess_schedule_path = paths.clean_output / f"input_bess_schedule.{config.output_format}"
    load_class_map_path = paths.clean_output / "load_class_map.csv"
    reference_profiles_path = paths.reports_output / f"representative_profiles.{config.output_format}"
    data_dictionary_path = paths.reports_output / "data_dictionary.md"
    validation_json_path = paths.reports_output / "clean_validation_summary.json"
    validation_md_path = paths.reports_output / "clean_validation_summary.md"

    write_dataframe(truth_df, truth_path, config.output_format)
    write_dataframe(measured_df, measured_path, config.output_format)
    write_dataframe(cyber_df, cyber_path, config.output_format)
    write_dataframe(windows_df, windows_path, config.output_format)
    write_dataframe(empty_labels, labels_path, "csv")
    write_dataframe(impairment_summaries_to_frame(impairment_summary), impairment_path, "csv")
    write_dataframe(control_df, control_path, config.output_format)
    write_dataframe(fine_env, env_fine_path, config.output_format)
    write_dataframe(coarse_env, env_coarse_path, config.output_format)
    write_dataframe(load_schedule, load_schedule_path, config.output_format)
    write_dataframe(pv_schedule, pv_schedule_path, config.output_format)
    write_dataframe(bess_schedule, bess_schedule_path, config.output_format)
    write_dataframe(load_class_map, load_class_map_path, "csv")
    write_dataframe(reference_profiles, reference_profiles_path, config.output_format)
    save_pipeline_config(config, config_snapshot_path)
    write_jsonl(cyber_df.to_dict(orient="records"), paths.clean_output / "cyber_events.jsonl")

    dictionary_entries = (
        build_channel_dictionary(truth_df, "truth")
        + build_channel_dictionary(measured_df, "measured")
        + build_channel_dictionary(cyber_df, "cyber")
        + build_channel_dictionary(empty_labels, "labels")
        + build_channel_dictionary(windows_df, "windows")
    )
    write_data_dictionary(dictionary_entries, str(data_dictionary_path))

    validation_checks = validate_clean_layers(truth_df, measured_df, cyber_df)
    _write_validation_summary(validation_checks, validation_json_path, validation_md_path, "Clean Dataset Validation")
    graph_paths = build_validation_graphs(
        truth_df,
        measured_df,
        cyber_df,
        empty_labels,
        paths.reports_output / "graphs",
        impairment_df=impairment_summaries_to_frame(impairment_summary),
        reference_profiles_df=reference_profiles,
    )
    manifest = RunManifest(
        run_id=config.run_id,
        scenario_id=config.scenario_id,
        split_id=config.split_id,
        config=config.to_dict(),
        inventory_summary=inventory.summary(),
        assumptions=[
            "Base feeder topology uses the canonical IEEE 123-bus OpenDSS case.",
            "Environmental and load-shape generation is deterministic under the configured seed.",
            "Measured layer is produced from truth through noise, missingness, latency, and sparse observability.",
            "Phase 2 reuses the saved exogenous and control inputs rather than LLM-generated raw time series.",
        ],
        artifacts=[
            artifact_from_path(truth_path, len(truth_df), len(truth_df.columns), "Physics-consistent clean truth layer."),
            artifact_from_path(measured_path, len(measured_df), len(measured_df.columns), "Measured layer with sensor impairments."),
            artifact_from_path(cyber_path, len(cyber_df), len(cyber_df.columns), "Baseline cyber event stream."),
            artifact_from_path(labels_path, len(empty_labels), len(empty_labels.columns), "Benign clean labels placeholder."),
            artifact_from_path(windows_path, len(windows_df), len(windows_df.columns), "Model-ready merged windows."),
            artifact_from_path(data_dictionary_path, description="Machine-generated data dictionary."),
            artifact_from_path(reference_profiles_path, len(reference_profiles), len(reference_profiles.columns), "Representative weekday/weekend and seasonal profile bundle."),
            artifact_from_path(validation_json_path, description="Structured clean validation summary."),
            artifact_from_path(validation_md_path, description="Markdown clean validation summary."),
        ],
        validation_checks=validation_checks,
        graph_paths=graph_paths,
        applied_scenarios=[],
    )
    write_json(manifest.to_dict(), paths.clean_output / "scenario_manifest.json")
    print(f"Clean dataset written to {paths.clean_output}")


def _write_validation_summary(checks: list, json_path: Path, md_path: Path, title: str) -> None:
    payload = [asdict(check) for check in checks]
    write_json(payload, json_path)
    lines = [f"# {title}", ""]
    for item in checks:
        metric = f" metric={item.metric:.4f}" if item.metric is not None else ""
        threshold = f" threshold={item.threshold}" if item.threshold is not None else ""
        lines.append(f"- {item.name}: {item.status}. {item.detail}{metric}{threshold}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
