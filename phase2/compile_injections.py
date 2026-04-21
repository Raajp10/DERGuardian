"""Phase 2 scenario and attacked-dataset support for DERGuardian.

This module implements compile injections logic for schema-bound synthetic attack
scenarios, injection compilation, cyber logs, labels, validation, or reporting.
Generated scenarios are heldout synthetic evidence and are not claimed as
real-world zero-day proof.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jsonschema
import numpy as np
import pandas as pd

from common.config import PipelineConfig
from common.dss_runner import CircuitInventory
from common.io_utils import read_dataframe, read_json, write_dataframe, write_json
from phase2.contracts import CANONICAL_TARGET_COMPONENTS, is_supported_target_component


def load_and_validate_scenarios(
    scenario_source: str | Path,
    schema_file: str | Path,
    timeline: pd.DatetimeIndex | None = None,
    available_assets: set[str] | None = None,
    available_columns: set[str] | None = None,
    asset_bounds: dict[str, dict[str, object]] | None = None,
) -> dict:
    """Load a scenario bundle and enforce the canonical executable Phase 2 contract.

    Validation here is intentionally bounded and defensive: schema correctness,
    timeline containment, supported target components, allowed assets/signals,
    and configured asset bounds. This is not a formal physics-proof engine.
    """
    payload = load_scenario_bundle(scenario_source)
    schema = read_json(schema_file)
    jsonschema.validate(payload, schema)
    for scenario in payload["scenarios"]:
        _validate_target_contract(
            scenario=scenario,
            timeline=timeline,
            available_assets=available_assets,
            available_columns=available_columns,
            asset_bounds=asset_bounds,
        )
        for index, target in enumerate(scenario.get("additional_targets", []), start=1):
            synthetic = {
                "scenario_id": f"{scenario['scenario_id']}#additional_target_{index}",
                "target_component": target["target_component"],
                "target_asset": target["target_asset"],
                "target_signal": target["target_signal"],
                "injection_type": target["injection_type"],
                "magnitude_value": target.get("magnitude_value"),
                "magnitude_text": target.get("magnitude_text"),
                "magnitude_units": target.get("magnitude_units"),
                "start_time_utc": scenario.get("start_time_utc"),
                "start_offset_seconds": scenario.get("start_offset_seconds"),
                "duration_seconds": scenario.get("duration_seconds"),
            }
            _validate_target_contract(
                scenario=synthetic,
                timeline=timeline,
                available_assets=available_assets,
                available_columns=available_columns,
                asset_bounds=asset_bounds,
            )
    return payload


def load_scenario_bundle(source: str | Path) -> dict:
    """Load scenario bundle for the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    src = Path(source)
    if src.is_file():
        payload = _normalize_payload(read_json(src), src.stem)
        payload["source_files"] = [str(src)]
        return payload
    if not src.is_dir():
        raise FileNotFoundError(f"Scenario source not found: {src}")
    scenarios: list[dict] = []
    files = sorted(src.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON scenario files were found in {src}")
    for file in files:
        payload = _normalize_payload(read_json(file), file.stem)
        scenarios.extend(payload["scenarios"])
    return {"dataset_id": src.name, "scenarios": scenarios, "source_files": [str(file) for file in files]}


def compile_scenarios(payload: dict, timeline: pd.DatetimeIndex) -> dict[str, object]:
    """Handle compile scenarios within the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    override_df = pd.DataFrame({"timestamp_utc": timeline})
    measurement_actions: list[dict[str, object]] = []
    physical_actions: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    compiled_rows: list[dict[str, object]] = []

    for scenario in payload["scenarios"]:
        targets = [_scenario_target(scenario)] + list(scenario.get("additional_targets", []))
        start = _resolve_start_time(scenario, timeline)
        duration = int(scenario["duration_seconds"])
        end = start + pd.Timedelta(seconds=duration)
        mask = (timeline >= start) & (timeline < end)
        compiled_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "scenario_name": scenario["scenario_name"],
                "attack_family": scenario["category"],
                "severity": scenario["severity"],
                "start_time_utc": start,
                "end_time_utc": end,
                "target_component": scenario["target_component"],
                "target_asset": scenario["target_asset"],
                "target_signal": scenario["target_signal"],
                "observable_signals": scenario["observable_signals"],
                "source_files": payload.get("source_files", []),
            }
        )
        profile = _temporal_profile(
            pattern=scenario["temporal_pattern"],
            magnitude=float(scenario.get("magnitude_value", 0.0) or 0.0),
            steps=int(mask.sum()),
        )
        target_columns: list[str] = []
        for target in targets:
            if not is_supported_target_component(str(target["target_component"])):
                raise ValueError(
                    f"Scenario {scenario['scenario_id']} resolved to unsupported target_component={target['target_component']} during compilation."
                )
            if target["target_component"] == "measured_layer":
                column = _measurement_column_name(target["target_asset"], target["target_signal"])
                target_columns.append(column)
                measurement_actions.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "attack_family": scenario["category"],
                        "severity": scenario["severity"],
                        "target_column": column,
                        "target_component": target["target_component"],
                        "target_asset": target["target_asset"],
                        "target_signal": target["target_signal"],
                        "injection_type": target["injection_type"],
                        "start_time_utc": start,
                        "end_time_utc": end,
                        "profile": profile.tolist(),
                        "magnitude_text": target.get("magnitude_text"),
                        "source_offset_seconds": int(scenario.get("source_offset_seconds", 0)),
                        "delay_seconds": int(scenario.get("delay_seconds", 0)),
                        "expected_effect": scenario["expected_effect"],
                    }
                )
            else:
                column = _override_column_name(target["target_component"], target["target_asset"], target["target_signal"])
                target_columns.append(column)
                physical_action = {
                    "scenario_id": scenario["scenario_id"],
                    "attack_family": scenario["category"],
                    "severity": scenario["severity"],
                    "target_component": target["target_component"],
                    "target_asset": target["target_asset"],
                    "target_signal": target["target_signal"],
                    "override_column": column,
                    "injection_type": target["injection_type"],
                    "start_time_utc": start,
                    "end_time_utc": end,
                    "profile": profile.tolist(),
                    "magnitude_text": target.get("magnitude_text"),
                    "delay_seconds": int(scenario.get("delay_seconds", 0)),
                    "source_offset_seconds": int(scenario.get("source_offset_seconds", 0)),
                }
                physical_actions.append(physical_action)
                if target["target_component"] in {"regulator", "capacitor", "switch"} and target["injection_type"] in {"bias", "scale", "freeze", "delay", "dropout", "command_override", "mode_change"}:
                    if column not in override_df.columns:
                        override_df[column] = np.nan
                    if target["injection_type"] == "delay" and int(scenario.get("delay_seconds", 0)) > 0:
                        delayed_start = start + pd.Timedelta(seconds=int(scenario.get("delay_seconds", 0)))
                        delayed_mask = (timeline >= delayed_start) & (timeline < end + pd.Timedelta(seconds=int(scenario.get("delay_seconds", 0))))
                        override_df.loc[delayed_mask, column] = profile[: int(delayed_mask.sum())]
                    elif target["injection_type"] != "dropout":
                        override_df.loc[mask, column] = profile
        label_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "scenario_name": scenario["scenario_name"],
                "attack_family": scenario["category"],
                "severity": scenario["severity"],
                "start_time_utc": start,
                "end_time_utc": end,
                "affected_assets": sorted(set(target["target_asset"] for target in targets)),
                "affected_signals": sorted(set(target_columns)),
                "target_component": scenario["target_component"],
                "causal_metadata": {
                    "expected_effect": scenario["expected_effect"],
                    "observable_signals": scenario["observable_signals"],
                    "protocol": scenario.get("protocol"),
                    "notes": scenario.get("notes"),
                    "description": scenario.get("description"),
                    "injection_type": scenario.get("injection_type"),
                    "delay_seconds": int(scenario.get("delay_seconds", 0)),
                    "source_offset_seconds": int(scenario.get("source_offset_seconds", 0)),
                },
            }
        )
    return {
        "override_df": override_df,
        "measurement_actions": measurement_actions,
        "physical_actions": physical_actions,
        "labels_df": pd.DataFrame(label_rows),
        "compiled_manifest": {
            "dataset_id": payload.get("dataset_id"),
            "source_files": payload.get("source_files", []),
            "scenario_count": len(compiled_rows),
            "scenarios": compiled_rows,
        },
    }


def build_asset_bounds(config: PipelineConfig, inventory: CircuitInventory) -> dict[str, dict[str, object]]:
    """Build asset bounds for the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    bounds: dict[str, dict[str, object]] = {}
    for asset in config.pv_assets:
        bounds[asset.name] = {
            "component": "pv",
            "signals": {
                "curtailment_frac": {"min": 0.0, "max": 1.0},
                "status_cmd": {"allowed": [0, 1]},
                "mode_cmd": {"allowed_text": ["volt_var", "fixed_pf", "standby", "solar_firming"]},
            },
        }
    for asset in config.bess_assets:
        bounds[asset.name] = {
            "component": "bess",
            "signals": {
                "target_kw": {"min": -(asset.kw_rated or 0.0), "max": asset.kw_rated or 0.0},
                "status_cmd": {"allowed": [0, 1]},
                "mode_cmd": {"allowed_text": ["peak_shaving", "solar_firming", "standby"]},
            },
        }
    for regulator in inventory.regulators:
        bounds[regulator] = {"component": "regulator", "signals": {"vreg": {"min": 110.0, "max": 130.0}}}
    for capacitor in inventory.capacitors:
        bounds[capacitor] = {"component": "capacitor", "signals": {"state": {"allowed": [0, 1]}}}
    for switch in inventory.switches:
        bounds[switch] = {"component": "switch", "signals": {"state": {"allowed": [0, 1]}}}
    return bounds


def _normalize_payload(raw_payload: dict, fallback_dataset_id: str) -> dict:
    if "scenarios" in raw_payload:
        return {"dataset_id": raw_payload.get("dataset_id", fallback_dataset_id), "scenarios": raw_payload["scenarios"]}
    return {"dataset_id": fallback_dataset_id, "scenarios": [raw_payload]}


def _validate_target_contract(
    scenario: dict,
    timeline: pd.DatetimeIndex | None,
    available_assets: set[str] | None,
    available_columns: set[str] | None,
    asset_bounds: dict[str, dict[str, object]] | None,
) -> None:
    component = str(scenario["target_component"])
    if not is_supported_target_component(component):
        raise ValueError(
            f"Scenario {scenario['scenario_id']} uses unsupported target_component={component}. "
            f"Canonical supported components: {', '.join(CANONICAL_TARGET_COMPONENTS)}."
        )
    if timeline is not None:
        start = _resolve_start_time(scenario, timeline)
        end = start + pd.Timedelta(seconds=int(scenario["duration_seconds"]))
        if start < timeline.min() or end > timeline.max():
            raise ValueError(f"Scenario {scenario['scenario_id']} falls outside the clean-dataset timeline.")
    if available_assets is not None and component != "measured_layer" and scenario["target_asset"] not in available_assets:
        raise ValueError(f"Scenario {scenario['scenario_id']} targets unknown asset {scenario['target_asset']}.")
    if available_columns is not None and component == "measured_layer":
        column = _measurement_column_name(scenario["target_asset"], scenario["target_signal"])
        if column not in available_columns and scenario["target_signal"] not in available_columns:
            raise ValueError(f"Scenario {scenario['scenario_id']} targets unknown measured signal {scenario['target_signal']}.")
    _validate_target_bounds(scenario, asset_bounds)


def _validate_target_bounds(scenario: dict, asset_bounds: dict[str, dict[str, object]] | None) -> None:
    if asset_bounds is None or scenario["target_component"] == "measured_layer":
        return
    asset = scenario["target_asset"]
    signal = scenario["target_signal"]
    spec = asset_bounds.get(asset)
    if spec is None:
        return
    signal_spec = spec.get("signals", {}).get(signal)
    if signal_spec is None:
        return
    if "allowed" in signal_spec and scenario.get("magnitude_value") not in signal_spec["allowed"]:
        raise ValueError(f"Scenario {scenario['scenario_id']} uses invalid value {scenario.get('magnitude_value')} for {asset}:{signal}.")
    if "allowed_text" in signal_spec and scenario.get("magnitude_text") not in signal_spec["allowed_text"]:
        raise ValueError(f"Scenario {scenario['scenario_id']} uses invalid text value {scenario.get('magnitude_text')} for {asset}:{signal}.")
    value = scenario.get("magnitude_value")
    if value is not None and "min" in signal_spec and not (float(signal_spec["min"]) <= float(value) <= float(signal_spec["max"])):
        raise ValueError(
            f"Scenario {scenario['scenario_id']} magnitude {value} for {asset}:{signal} is outside bounds [{signal_spec['min']}, {signal_spec['max']}]."
        )


def _scenario_target(scenario: dict) -> dict:
    return {
        "target_component": scenario["target_component"],
        "target_asset": scenario["target_asset"],
        "target_signal": scenario["target_signal"],
        "injection_type": scenario["injection_type"],
        "magnitude_value": scenario.get("magnitude_value"),
        "magnitude_text": scenario.get("magnitude_text"),
        "magnitude_units": scenario["magnitude_units"],
    }


def _resolve_start_time(scenario: dict, timeline: pd.DatetimeIndex) -> pd.Timestamp:
    if scenario.get("start_offset_seconds") is not None:
        return timeline.min() + pd.Timedelta(seconds=int(scenario["start_offset_seconds"]))
    return pd.Timestamp(scenario["start_time_utc"]).tz_convert("UTC")


def _temporal_profile(pattern: str, magnitude: float, steps: int) -> np.ndarray:
    if steps <= 0:
        return np.array([], dtype=float)
    if pattern == "step":
        return np.full(steps, magnitude)
    if pattern == "ramp":
        return np.linspace(0.0, magnitude, steps)
    if pattern == "pulse":
        profile = np.zeros(steps)
        profile[steps // 4 : 3 * steps // 4] = magnitude
        return profile
    if pattern == "staircase":
        return np.repeat(np.linspace(magnitude / 4.0, magnitude, 4), repeats=int(np.ceil(steps / 4)))[:steps]
    if pattern == "sine":
        x = np.linspace(0.0, 2.0 * np.pi, steps)
        return magnitude * np.sin(x)
    if pattern in {"replay", "freeze", "burst"}:
        return np.full(steps, magnitude)
    raise ValueError(f"Unsupported temporal pattern: {pattern}")


def _measurement_column_name(asset: str, signal: str) -> str:
    if signal.startswith(("bus_", "feeder_", "pv_", "bess_", "line_", "regulator_", "capacitor_", "switch_", "breaker_", "relay_")):
        return signal
    if asset.startswith("pv"):
        return f"pv_{asset}_{signal}"
    if asset.startswith("bess"):
        return f"bess_{asset}_{signal}"
    if asset.startswith("bus"):
        return signal if signal.startswith("bus_") else f"bus_{asset.replace('bus', '')}_{signal}"
    return signal


def _override_column_name(component: str, asset: str, signal: str) -> str:
    if component == "pv":
        return f"pv_{asset}_{signal}"
    if component == "bess":
        return f"bess_{asset}_{signal}"
    if component == "regulator":
        return f"regulator_{asset}_{signal}"
    if component == "capacitor":
        return f"capacitor_{asset}_{signal}"
    if component == "switch":
        return f"switch_{asset}_{signal}"
    raise ValueError(f"Unsupported physical target component {component}")


def main() -> None:
    """Run the command-line entrypoint for the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Compile scenario JSON into physical overrides and measurement actions.")
    parser.add_argument("--scenarios", required=True, help="Scenario JSON file or folder containing scenario JSON files.")
    parser.add_argument("--schema", default=str(ROOT / "phase2" / "scenario_schema.json"))
    parser.add_argument("--timeline", required=True, help="Truth-layer file used to infer timeline and available columns.")
    parser.add_argument(
        "--measured-source",
        default=str(ROOT / "outputs" / "clean" / "measured_physical_timeseries.parquet"),
        help="Measured-layer file used to validate measured_layer target columns.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    truth_df = read_dataframe(args.timeline)
    measured_df = read_dataframe(args.measured_source)
    timeline = pd.to_datetime(truth_df["timestamp_utc"], utc=True)
    payload = load_and_validate_scenarios(
        scenario_source=args.scenarios,
        schema_file=args.schema,
        timeline=timeline,
        available_columns=set(measured_df.columns),
    )
    compiled = compile_scenarios(payload, timeline)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(compiled["override_df"], output_dir / "compiled_overrides.parquet", "parquet")
    write_json(compiled["measurement_actions"], output_dir / "compiled_measurement_actions.json")
    write_json(compiled["physical_actions"], output_dir / "compiled_physical_actions.json")
    write_dataframe(compiled["labels_df"], output_dir / "compiled_attack_labels.csv", "csv")
    write_json(compiled["compiled_manifest"], output_dir / "compiled_manifest.json")


if __name__ == "__main__":
    main()
