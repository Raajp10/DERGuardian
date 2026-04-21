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

from common.config import ProjectPaths, load_pipeline_config
from common.dss_runner import extract_inventory, simulate_truth
from common.graph_builders import build_validation_graphs
from common.io_utils import read_dataframe, write_dataframe, write_json, write_jsonl
from common.metadata_schema import RunManifest, artifact_from_path
from common.noise_models import build_measured_layer
from phase1.build_windows import build_merged_windows
from phase2.compile_injections import build_asset_bounds, compile_scenarios, load_and_validate_scenarios
from phase2.contracts import CANONICAL_PHASE2_EXECUTION_PATH, CANONICAL_TARGET_COMPONENTS
from phase2.cyber_log_generator import generate_baseline_cyber_events, inject_attack_events
from phase2.reporting import write_phase2_summary_reports
from phase2.validate_attacked_dataset import validate_attacked_layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate attacked dataset layers from schema-validated Phase 2 scenario JSON using the canonical JSON-driven execution path.")
    parser.add_argument("--scenarios", default=str(ROOT / "phase2" / "example_scenarios.json"))
    parser.add_argument("--schema", default=str(ROOT / "phase2" / "scenario_schema.json"))
    parser.add_argument("--clean-root", default=str(ROOT / "outputs" / "clean"))
    parser.add_argument("--attacked-output-root", default=None)
    parser.add_argument("--windows-output-root", default=None)
    parser.add_argument("--labeled-output-root", default=None)
    parser.add_argument("--reports-output-root", default=None)
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    schema_path = Path(args.schema)
    clean_root = Path(args.clean_root)
    if not scenarios_path.is_absolute():
        scenarios_path = (ROOT / scenarios_path).resolve()
    if not schema_path.is_absolute():
        schema_path = (ROOT / schema_path).resolve()
    if not clean_root.is_absolute():
        clean_root = (ROOT / clean_root).resolve()

    attacked_output = Path(args.attacked_output_root) if args.attacked_output_root else ROOT / "outputs" / "attacked"
    windows_output = Path(args.windows_output_root) if args.windows_output_root else ROOT / "outputs" / "windows"
    labeled_output = Path(args.labeled_output_root) if args.labeled_output_root else ROOT / "outputs" / "labeled"
    reports_output = Path(args.reports_output_root) if args.reports_output_root else ROOT / "outputs" / "reports"
    if not attacked_output.is_absolute():
        attacked_output = (ROOT / attacked_output).resolve()
    if not windows_output.is_absolute():
        windows_output = (ROOT / windows_output).resolve()
    if not labeled_output.is_absolute():
        labeled_output = (ROOT / labeled_output).resolve()
    if not reports_output.is_absolute():
        reports_output = (ROOT / reports_output).resolve()

    paths = ProjectPaths(
        root=ROOT,
        clean_output=clean_root,
        attacked_output=attacked_output,
        windows_output=windows_output,
        labeled_output=labeled_output,
        reports_output=reports_output,
        report_file=reports_output / "generated_report.md",
    ).ensure()
    config = load_pipeline_config(clean_root / "config_snapshot.json")
    config.run_id = f"{config.run_id}-attacked"
    measured_clean = read_dataframe(clean_root / f"measured_physical_timeseries.{config.output_format}")
    truth_clean = read_dataframe(clean_root / f"truth_physical_timeseries.{config.output_format}")
    env_df = read_dataframe(clean_root / f"input_environment_timeseries.{config.output_format}")
    load_schedule = read_dataframe(clean_root / f"input_load_schedule.{config.output_format}")
    pv_schedule = read_dataframe(clean_root / f"input_pv_schedule.{config.output_format}")
    bess_schedule = read_dataframe(clean_root / f"input_bess_schedule.{config.output_format}")
    cyber_clean = read_dataframe(clean_root / f"cyber_events.{config.output_format}")
    impairment_df = _optional_read(clean_root / "measurement_impairments.csv")
    reference_profiles_df = _optional_read(paths.reports_output / f"representative_profiles.{config.output_format}")

    inventory = extract_inventory(config, paths)
    asset_bounds = build_asset_bounds(config, inventory)
    available_assets = {asset.name for asset in inventory.pv_assets + inventory.bess_assets} | set(inventory.regulators) | set(inventory.capacitors) | set(inventory.switches)
    payload = load_and_validate_scenarios(
        scenario_source=scenarios_path,
        schema_file=schema_path,
        timeline=pd.to_datetime(truth_clean["timestamp_utc"], utc=True),
        available_assets=available_assets,
        available_columns=set(measured_clean.columns),
        asset_bounds=asset_bounds,
    )
    config.scenario_id = str(payload.get("dataset_id", "attacked_bundle"))
    compiled = compile_scenarios(payload, pd.to_datetime(truth_clean["timestamp_utc"], utc=True))
    measurement_actions = compiled["measurement_actions"]
    physical_actions = compiled["physical_actions"]
    labels_df = compiled["labels_df"]
    override_df, override_warnings = _apply_physical_actions(
        compiled["override_df"],
        physical_actions,
        pv_schedule,
        bess_schedule,
    )

    has_physical = len([column for column in override_df.columns if column != "timestamp_utc" and override_df[column].notna().any()]) > 0
    if has_physical:
        attacked_truth, control_df = simulate_truth(
            config=config,
            inventory=inventory,
            env_df=env_df,
            load_schedule_df=load_schedule,
            pv_schedule_df=pv_schedule,
            bess_schedule_df=bess_schedule,
            paths=paths,
            scenario_id=config.scenario_id,
            split_id=config.split_id,
            override_df=override_df,
        )
    else:
        attacked_truth = truth_clean.copy()
        attacked_truth["scenario_id"] = config.scenario_id
        attacked_truth["run_id"] = config.run_id
        control_df = _optional_read(clean_root / f"control_schedule.{config.output_format}")
        if control_df is None:
            control_df = pd.DataFrame({"timestamp_utc": attacked_truth["timestamp_utc"]})

    attacked_measured, _ = build_measured_layer(attacked_truth, config, np.random.default_rng(config.random_seed + 133))
    attacked_measured = _apply_measurement_actions(attacked_measured, measurement_actions)
    cyber_df = generate_baseline_cyber_events(attacked_truth, control_df, config, np.random.default_rng(config.random_seed + 151))
    cyber_df = inject_attack_events(cyber_df, payload["scenarios"], np.random.default_rng(config.random_seed + 173))
    windows_df = build_merged_windows(attacked_measured, cyber_df, labels_df, config.windows)

    truth_path = paths.attacked_output / f"truth_physical_timeseries.{config.output_format}"
    measured_path = paths.attacked_output / f"measured_physical_timeseries.{config.output_format}"
    cyber_path = paths.attacked_output / f"cyber_events.{config.output_format}"
    labels_path = paths.labeled_output / "attack_labels.csv"
    labels_attacked_path = paths.attacked_output / f"attack_labels.{config.output_format}"
    windows_path = paths.windows_output / f"merged_windows_attacked.{config.output_format}"
    windows_attacked_path = paths.attacked_output / f"merged_windows.{config.output_format}"
    override_path = paths.attacked_output / f"compiled_overrides.{config.output_format}"
    compiled_manifest_path = paths.attacked_output / "compiled_manifest.json"
    validation_json_path = paths.reports_output / "attacked_validation_summary.json"
    validation_md_path = paths.reports_output / "attacked_validation_summary.md"

    write_dataframe(attacked_truth, truth_path, config.output_format)
    write_dataframe(attacked_measured, measured_path, config.output_format)
    write_dataframe(cyber_df, cyber_path, config.output_format)
    write_dataframe(labels_df, labels_path, "csv")
    write_dataframe(labels_df, labels_attacked_path, config.output_format)
    write_dataframe(windows_df, windows_path, config.output_format)
    write_dataframe(windows_df, windows_attacked_path, config.output_format)
    write_dataframe(override_df, override_path, config.output_format)
    write_json(measurement_actions, paths.attacked_output / "compiled_measurement_actions.json")
    write_json(physical_actions, paths.attacked_output / "compiled_physical_actions.json")
    write_json(compiled["compiled_manifest"], compiled_manifest_path)
    write_jsonl(cyber_df.to_dict(orient="records"), paths.attacked_output / "cyber_events.jsonl")

    validation_checks = validate_attacked_layers(
        attacked_truth_df=attacked_truth,
        attacked_measured_df=attacked_measured,
        cyber_df=cyber_df,
        labels_df=labels_df,
        baseline_truth_df=truth_clean,
        baseline_measured_df=measured_clean,
        windows_df=windows_df,
    )
    _write_validation_summary(validation_checks, validation_json_path, validation_md_path, "Attacked Dataset Validation")
    summary_artifacts = write_phase2_summary_reports(
        reports_root=paths.reports_output,
        payload=payload,
        labels_df=labels_df,
        windows_df=windows_df,
        validation_checks=validation_checks,
        clean_truth_df=truth_clean,
        attacked_truth_df=attacked_truth,
        clean_measured_df=measured_clean,
        attacked_measured_df=attacked_measured,
        clean_cyber_df=cyber_clean,
        attacked_cyber_df=cyber_df,
    )
    graph_paths = build_validation_graphs(
        attacked_truth,
        attacked_measured,
        cyber_df,
        labels_df,
        paths.reports_output / "graphs_attacked",
        baseline_truth_df=truth_clean,
        baseline_measured_df=measured_clean,
        impairment_df=impairment_df,
        reference_profiles_df=reference_profiles_df,
    )
    manifest = RunManifest(
        run_id=config.run_id,
        scenario_id=config.scenario_id,
        split_id=config.split_id,
        config=config.to_dict(),
        inventory_summary=inventory.summary(),
        assumptions=[
            "Phase 2 attacks are generated strictly from scenario JSON metadata.",
            "The canonical executable Phase 2 path is JSON-driven and does not require an embedded LLM call.",
            "Physical effects are rerun through OpenDSS when scenarios produce control overrides.",
            "Measurement-layer attacks are applied after baseline sensor impairments are synthesized.",
            "Scenario bundles may be provided either as a single JSON file or as a folder of JSON scenario files.",
            f"Canonical supported target components: {', '.join(CANONICAL_TARGET_COMPONENTS)}.",
        ],
        artifacts=[
            artifact_from_path(truth_path, len(attacked_truth), len(attacked_truth.columns), "Attacked truth layer."),
            artifact_from_path(measured_path, len(attacked_measured), len(attacked_measured.columns), "Attacked measured layer."),
            artifact_from_path(cyber_path, len(cyber_df), len(cyber_df.columns), "Cyber layer with injected attack events."),
            artifact_from_path(labels_path, len(labels_df), len(labels_df.columns), "Attack labels."),
            artifact_from_path(labels_attacked_path, len(labels_df), len(labels_df.columns), "Attack labels colocated with attacked dataset outputs."),
            artifact_from_path(windows_path, len(windows_df), len(windows_df.columns), "Merged attacked windows."),
            artifact_from_path(windows_attacked_path, len(windows_df), len(windows_df.columns), "Merged attacked windows colocated with attacked dataset outputs."),
            artifact_from_path(override_path, len(override_df), len(override_df.columns), "Compiled physical override table."),
            artifact_from_path(compiled_manifest_path, description="Compiled scenario manifest with source files and timing."),
            artifact_from_path(validation_json_path, description="Structured attacked validation summary."),
            artifact_from_path(validation_md_path, description="Markdown attacked validation summary."),
            artifact_from_path(summary_artifacts["coverage_summary_json"], description="Phase 2 scenario coverage summary."),
            artifact_from_path(summary_artifacts["coverage_summary_md"], description="Markdown Phase 2 scenario coverage summary."),
            artifact_from_path(summary_artifacts["label_summary_json"], description="Phase 2 label completeness summary."),
            artifact_from_path(summary_artifacts["label_summary_md"], description="Markdown Phase 2 label completeness summary."),
            artifact_from_path(summary_artifacts["effect_summary_json"], description="Per-scenario attacked-versus-clean effect summary."),
            artifact_from_path(summary_artifacts["effect_summary_md"], description="Markdown per-scenario attacked-versus-clean effect summary."),
            artifact_from_path(summary_artifacts["scenario_difficulty_json"], description="Per-scenario difficulty heuristic summary."),
            artifact_from_path(summary_artifacts["scenario_difficulty_md"], description="Markdown per-scenario difficulty heuristic summary."),
        ],
        validation_checks=validation_checks,
        graph_paths=graph_paths,
        applied_scenarios=payload["scenarios"],
        warnings=override_warnings,
    )
    manifest_payload = manifest.to_dict()
    manifest_payload["canonical_phase2_execution_path"] = list(CANONICAL_PHASE2_EXECUTION_PATH)
    write_json(manifest_payload, paths.attacked_output / "scenario_manifest.json")
    print(f"Attacked dataset written to {paths.attacked_output}")


def _optional_read(path: Path) -> pd.DataFrame | None:
    return read_dataframe(path) if path.exists() else None


def _apply_physical_actions(
    override_df: pd.DataFrame,
    physical_actions: list[dict[str, object]],
    pv_schedule_df: pd.DataFrame,
    bess_schedule_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    override = override_df.copy()
    override["timestamp_utc"] = pd.to_datetime(override["timestamp_utc"], utc=True)
    override = override.set_index("timestamp_utc")
    pv_schedule = pv_schedule_df.copy().set_index(pd.to_datetime(pv_schedule_df["timestamp_utc"], utc=True))
    bess_schedule = bess_schedule_df.copy().set_index(pd.to_datetime(bess_schedule_df["timestamp_utc"], utc=True))
    sample_seconds = _sample_seconds(override.index)
    warnings: list[str] = []

    for action in physical_actions:
        column = str(action["override_column"])
        component = str(action["target_component"])
        base_series = None
        if component == "pv" and column in pv_schedule.columns:
            base_series = pv_schedule[column]
        elif component == "bess" and column in bess_schedule.columns:
            base_series = bess_schedule[column]
        if column not in override.columns:
            override[column] = np.nan
        try:
            override[column] = _materialize_action_series(override[column], base_series, action, sample_seconds)
        except Exception as exc:
            warnings.append(f"Scenario {action['scenario_id']} could not be materialized for {column}: {exc}")
    return override.reset_index(), warnings


def _materialize_action_series(
    existing: pd.Series,
    base_series: pd.Series | None,
    action: dict[str, object],
    sample_seconds: int,
) -> pd.Series:
    series = existing.copy()
    timeline = pd.DatetimeIndex(series.index)
    if len(timeline) == 0:
        return series
    start = pd.Timestamp(action["start_time_utc"]).tz_convert("UTC")
    end = pd.Timestamp(action["end_time_utc"]).tz_convert("UTC")
    start_pos = int(timeline.searchsorted(start))
    end_pos = int(timeline.searchsorted(end))
    if end_pos <= start_pos:
        return series
    window_len = end_pos - start_pos
    kind = str(action["injection_type"])
    delay_steps = int(round(float(action.get("delay_seconds", 0) or 0) / max(sample_seconds, 1)))
    offset_steps = int(round(float(action.get("source_offset_seconds", 0) or 0) / max(sample_seconds, 1)))
    text_value = action.get("magnitude_text")
    profile = np.asarray(action.get("profile", []), dtype=float)
    base_window = base_series.iloc[start_pos:end_pos] if base_series is not None else None

    if kind in {"bias", "scale"} and base_window is not None:
        numeric_values = base_window.astype(float).to_numpy()
        values = numeric_values + profile[:window_len] if kind == "bias" else numeric_values * profile[:window_len]
        _assign_slice(series, start_pos, values)
        return series

    if kind in {"freeze", "command_suppression", "dropout"} and base_window is not None:
        hold = base_series.iloc[start_pos - 1] if start_pos > 0 else base_window.iloc[0]
        fill_value = np.nan if kind == "dropout" else hold
        _assign_slice(series, start_pos, [fill_value] * window_len)
        return series

    if kind == "replay" and base_series is not None:
        source_start = max(start_pos - offset_steps, 0)
        values = list(base_series.iloc[source_start : source_start + window_len])
        if values:
            if len(values) < window_len:
                values.extend([values[-1]] * (window_len - len(values)))
            _assign_slice(series, start_pos, values)
        return series

    if kind in {"delay", "command_delay"}:
        target_start = start_pos + delay_steps
        if base_window is not None and (text_value is None or str(text_value).strip() == "") and not profile.size:
            values = list(base_window)
        else:
            values = _override_values_for_window(base_window, text_value, profile, window_len)
        if base_window is not None:
            hold = base_series.iloc[start_pos - 1] if start_pos > 0 else base_window.iloc[0]
            _assign_slice(series, start_pos, [hold] * window_len)
        _assign_slice(series, target_start, values)
        return series

    values = _override_values_for_window(base_window, text_value, profile, window_len)
    _assign_slice(series, start_pos, values)
    return series


def _override_values_for_window(
    base_window: pd.Series | None,
    text_value: object,
    profile: np.ndarray,
    window_len: int,
) -> list[object]:
    if text_value is not None and str(text_value).strip() != "":
        return [text_value] * window_len
    if profile.size:
        return profile[:window_len].tolist()
    if base_window is not None:
        return list(base_window)
    return [np.nan] * window_len


def _assign_slice(series: pd.Series, start_pos: int, values: list[object] | np.ndarray) -> None:
    if start_pos >= len(series):
        return
    value_list = list(values)
    clipped = value_list[: max(len(series) - start_pos, 0)]
    if clipped:
        series.iloc[start_pos : start_pos + len(clipped)] = clipped


def _sample_seconds(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    return max(int((index[1] - index[0]).total_seconds()), 1)


def _apply_measurement_actions(df: pd.DataFrame, actions: list[dict[str, object]]) -> pd.DataFrame:
    attacked = df.copy()
    attacked["timestamp_utc"] = pd.to_datetime(attacked["timestamp_utc"], utc=True)
    sample_seconds = _sample_seconds(pd.DatetimeIndex(attacked["timestamp_utc"]))
    for action in actions:
        column = str(action["target_column"])
        if column not in attacked.columns:
            continue
        start = pd.Timestamp(action["start_time_utc"]).tz_convert("UTC")
        end = pd.Timestamp(action["end_time_utc"]).tz_convert("UTC")
        mask = (attacked["timestamp_utc"] >= start) & (attacked["timestamp_utc"] < end)
        window = attacked.loc[mask, column].copy()
        if window.empty:
            continue
        profile = np.array(action.get("profile", []), dtype=float)
        kind = str(action["injection_type"])
        delay_steps = max(int(round(float(action.get("delay_seconds", 0) or 0) / max(sample_seconds, 1))), 1)
        if kind == "bias":
            attacked.loc[mask, column] = window.to_numpy() + profile[: len(window)]
        elif kind == "scale":
            attacked.loc[mask, column] = window.to_numpy() * profile[: len(window)]
        elif kind in {"freeze", "command_suppression"}:
            attacked.loc[mask, column] = window.iloc[0]
        elif kind == "dropout":
            attacked.loc[mask, column] = np.nan
        elif kind == "replay":
            offset = int(action.get("source_offset_seconds", 0))
            source_start = start - pd.Timedelta(seconds=offset)
            source_end = source_start + (end - start)
            source = attacked[(attacked["timestamp_utc"] >= source_start) & (attacked["timestamp_utc"] < source_end)][column].to_numpy()
            if len(source) >= len(window):
                attacked.loc[mask, column] = source[: len(window)]
        elif kind in {"delay", "command_delay"}:
            attacked.loc[mask, column] = window.shift(delay_steps).bfill().to_numpy()
        elif kind == "command_override":
            attacked.loc[mask, column] = profile[: len(window)] if len(profile) else window
    return attacked


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
