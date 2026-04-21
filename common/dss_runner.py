"""Shared utility support for DERGuardian.

This module provides dss runner helpers used across the Phase 1 data
pipeline, Phase 2 scenario pipeline, and Phase 3 evaluation/reporting layers.
The functions here are infrastructure code: they prepare paths, metadata,
profiles, graphs, units, or time alignment without changing canonical detector
outputs or benchmark decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import opendssdirect as dss

from common.config import DERAssetSpec, PipelineConfig, ProjectPaths, research_master_dss_path
from common.weather_load_generators import LoadProfileSpec, classify_load


PHASE_LABELS = {1: "phase_a", 2: "phase_b", 3: "phase_c"}


@dataclass(slots=True)
class LoadSpec:
    """Structured object used by the shared DERGuardian utility workflow."""

    name: str
    bus: str
    phases: int
    base_kw: float
    base_kvar: float
    load_class: str


@dataclass(slots=True)
class LineSpec:
    """Structured object used by the shared DERGuardian utility workflow."""

    name: str
    bus1: str
    bus2: str
    phases: int
    is_switch: bool


@dataclass(slots=True)
class CircuitInventory:
    """Structured object used by the shared DERGuardian utility workflow."""

    buses: list[str]
    bus_phase_nodes: dict[str, list[int]]
    loads: list[LoadSpec]
    lines: list[LineSpec]
    regulators: list[str]
    capacitors: list[str]
    switches: list[str]
    pv_assets: list[DERAssetSpec]
    bess_assets: list[DERAssetSpec]

    def summary(self) -> dict[str, int]:
        return {
            "num_buses": len(self.buses),
            "num_loads": len(self.loads),
            "num_lines": len(self.lines),
            "num_regulators": len(self.regulators),
            "num_capacitors": len(self.capacitors),
            "num_switches": len(self.switches),
            "num_pv_assets": len(self.pv_assets),
            "num_bess_assets": len(self.bess_assets),
        }


def compile_research_circuit(paths: ProjectPaths | None = None) -> None:
    """Handle compile research circuit within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    master_path = research_master_dss_path(paths)
    original_cwd = Path.cwd()
    dss.Basic.ClearAll()
    dss.Text.Command(f"compile [{master_path}]")
    dss.Text.Command("set maxcontroliter=30")
    dss.Text.Command("solve")
    os.chdir(original_cwd)


def extract_inventory(config: PipelineConfig, paths: ProjectPaths) -> CircuitInventory:
    """Handle extract inventory within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    compile_research_circuit(paths)
    buses = list(dss.Circuit.AllBusNames())
    bus_phase_nodes: dict[str, list[int]] = {}
    for bus in buses:
        dss.Circuit.SetActiveBus(bus)
        bus_phase_nodes[bus] = [node for node in dss.Bus.Nodes() if node in (1, 2, 3)]

    loads: list[LoadSpec] = []
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)
        bus = dss.CktElement.BusNames()[0].split(".")[0]
        phases = dss.CktElement.NumPhases()
        base_kw = float(dss.Loads.kW())
        base_kvar = float(dss.Loads.kvar())
        loads.append(
            LoadSpec(
                name=name,
                bus=bus,
                phases=phases,
                base_kw=base_kw,
                base_kvar=base_kvar,
                load_class=classify_load(base_kw, phases, bus),
            )
        )

    lines: list[LineSpec] = []
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        lines.append(
            LineSpec(
                name=name,
                bus1=dss.Lines.Bus1().split(".")[0],
                bus2=dss.Lines.Bus2().split(".")[0],
                phases=dss.Lines.Phases(),
                is_switch=name.lower().startswith("sw"),
            )
        )

    return CircuitInventory(
        buses=buses,
        bus_phase_nodes=bus_phase_nodes,
        loads=loads,
        lines=lines,
        regulators=list(dss.RegControls.AllNames()),
        capacitors=list(dss.Capacitors.AllNames()),
        switches=[line.name for line in lines if line.is_switch],
        pv_assets=config.pv_assets,
        bess_assets=config.bess_assets,
    )


def load_profile_specs(inventory: CircuitInventory) -> list[LoadProfileSpec]:
    """Load profile specs for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return [
        LoadProfileSpec(
            name=item.name,
            bus=item.bus,
            base_kw=item.base_kw,
            base_kvar=item.base_kvar,
            phases=item.phases,
            load_class=item.load_class,
        )
        for item in inventory.loads
    ]


def simulate_truth(
    config: PipelineConfig,
    inventory: CircuitInventory,
    env_df: pd.DataFrame,
    load_schedule_df: pd.DataFrame,
    pv_schedule_df: pd.DataFrame,
    bess_schedule_df: pd.DataFrame,
    paths: ProjectPaths,
    scenario_id: str,
    split_id: str,
    override_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Handle simulate truth within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    compile_research_circuit(paths)
    env = env_df.set_index("timestamp_utc")
    load_schedule = load_schedule_df.set_index("timestamp_utc")
    pv_schedule = pv_schedule_df.set_index("timestamp_utc")
    bess_schedule = bess_schedule_df.set_index("timestamp_utc")
    overrides = override_df.set_index("timestamp_utc") if override_df is not None and not override_df.empty else None

    storage_soc = {
        asset.name: float(asset.base_soc if asset.base_soc is not None else 0.5)
        for asset in inventory.bess_assets
    }
    previous_bus_voltage = {asset.bus: 1.0 for asset in inventory.pv_assets + inventory.bess_assets}
    truth_columns: dict[str, list[object]] = {}
    control_columns: dict[str, list[object]] = {}
    dt_hours = config.simulation_resolution_seconds / 3600.0

    for step_index, timestamp in enumerate(env.index):
        row_env = env.loc[timestamp]
        row_load = load_schedule.loc[timestamp]
        row_pv = pv_schedule.loc[timestamp]
        row_bess = bess_schedule.loc[timestamp]
        row_override = overrides.loc[timestamp] if overrides is not None and timestamp in overrides.index else None

        _apply_loads(inventory.loads, row_load)
        control_row: dict[str, object] = {
            "timestamp_utc": timestamp,
            "scenario_id": scenario_id,
            "run_id": config.run_id,
            "split_id": split_id,
        }
        for asset in inventory.pv_assets:
            control_row.update(_dispatch_pv(asset, row_env, row_pv, row_override, previous_bus_voltage.get(asset.bus, 1.0)))
        for asset in inventory.bess_assets:
            bess_control, new_soc = _dispatch_bess(
                asset=asset,
                schedule_row=row_bess,
                override_row=row_override,
                local_voltage_pu=previous_bus_voltage.get(asset.bus, 1.0),
                soc=storage_soc[asset.name],
                config=config,
                dt_hours=dt_hours,
            )
            storage_soc[asset.name] = new_soc
            control_row.update(bess_control)
        _apply_discrete_overrides(row_override, inventory)

        dss.Text.Command("solve")
        truth_row = _collect_truth_row(
            config=config,
            inventory=inventory,
            timestamp=timestamp,
            step_index=step_index,
            scenario_id=scenario_id,
            split_id=split_id,
            env_row=row_env,
            control_row=control_row,
        )
        _append_row_to_buffers(truth_columns, truth_row, step_index)
        _append_row_to_buffers(control_columns, control_row, step_index)
        for asset in inventory.pv_assets + inventory.bess_assets:
            bus_metrics = _bus_node_metrics(asset.bus, inventory.bus_phase_nodes.get(asset.bus, []))
            observed = [value for key, value in bus_metrics.items() if "_v_pu_" in key]
            previous_bus_voltage[asset.bus] = float(np.nanmean(observed)) if observed else 1.0

    truth_df = pd.DataFrame(truth_columns)
    control_df = pd.DataFrame(control_columns)
    _append_derived_features(truth_df, inventory, config)
    return truth_df, control_df


def _append_row_to_buffers(
    buffers: dict[str, list[object]],
    row: dict[str, object],
    completed_rows: int,
) -> None:
    row_keys = set(row)
    for key in list(buffers):
        if key not in row_keys:
            buffers[key].append(None)
    for key, value in row.items():
        if key not in buffers:
            buffers[key] = [None] * completed_rows
        buffers[key].append(value)


def _apply_loads(loads: list[LoadSpec], schedule_row: pd.Series) -> None:
    for load in loads:
        dss.Loads.Name(load.name)
        dss.Loads.kW(float(schedule_row[f"load_{load.name}_p_kw"]))
        dss.Loads.kvar(float(schedule_row[f"load_{load.name}_q_kvar"]))


def _dispatch_pv(
    asset: DERAssetSpec,
    env_row: pd.Series,
    schedule_row: pd.Series,
    override_row: pd.Series | None,
    local_voltage_pu: float,
) -> dict[str, object]:
    available = float(schedule_row[f"pv_{asset.name}_available_kw"])
    status = int(schedule_row[f"pv_{asset.name}_status_cmd"])
    mode = str(schedule_row[f"pv_{asset.name}_mode_cmd"])
    curtailment = float(schedule_row[f"pv_{asset.name}_curtailment_frac"])
    if override_row is not None:
        status = int(_override_value(override_row, f"pv_{asset.name}_status_cmd", status))
        mode = str(_override_value(override_row, f"pv_{asset.name}_mode_cmd", mode))
        curtailment = float(_override_value(override_row, f"pv_{asset.name}_curtailment_frac", curtailment))
    curtailment = float(np.clip(curtailment, 0.0, 1.0))
    p_target = available * status * (1.0 - curtailment)
    q_target = _volt_var_q(local_voltage_pu, asset.kva, p_target, mode)
    p_dispatch, q_dispatch = _enforce_apparent_limit(asset.kva, p_target, q_target)
    pct_pmpp = 100.0 * p_dispatch / max(available, 1e-6) if available > 0.0 else 0.0
    dss.Text.Command(
        f"edit pvsystem.{asset.name} enabled={'yes' if status else 'no'} irradiance=1 "
        f"pmpp={max(available, 0.0):.6f} %Pmpp={np.clip(pct_pmpp, 0.0, 100.0):.6f} "
        f"kvar={q_dispatch:.6f} temperature={float(env_row['env_temperature_c']):.3f}"
    )
    return {
        f"pv_{asset.name}_available_kw": available,
        f"pv_{asset.name}_curtailment_frac": curtailment,
        f"pv_{asset.name}_setpoint_kw": p_dispatch,
        f"pv_{asset.name}_setpoint_q_kvar": q_dispatch,
        f"pv_{asset.name}_status": status,
        f"pv_{asset.name}_mode": mode,
    }


def _dispatch_bess(
    asset: DERAssetSpec,
    schedule_row: pd.Series,
    override_row: pd.Series | None,
    local_voltage_pu: float,
    soc: float,
    config: PipelineConfig,
    dt_hours: float,
) -> tuple[dict[str, object], float]:
    target_kw = float(schedule_row[f"bess_{asset.name}_target_kw"])
    mode = str(schedule_row[f"bess_{asset.name}_mode_cmd"])
    status = int(schedule_row[f"bess_{asset.name}_status_cmd"])
    if override_row is not None:
        target_kw = float(_override_value(override_row, f"bess_{asset.name}_target_kw", target_kw))
        mode = str(_override_value(override_row, f"bess_{asset.name}_mode_cmd", mode))
        status = int(_override_value(override_row, f"bess_{asset.name}_status_cmd", status))
    rated_kw = asset.kw_rated or 0.0
    rated_kwh = asset.kwh_rated or 1.0
    reserve = asset.reserve_soc
    actual_kw = 0.0
    state = "idling"
    if status:
        if target_kw > 0.0 and soc > reserve:
            max_energy_kw = max((soc - reserve) * rated_kwh / dt_hours * config.storage_discharge_efficiency, 0.0)
            actual_kw = min(target_kw, rated_kw, max_energy_kw)
            if actual_kw > 0.0:
                soc -= actual_kw * dt_hours / (rated_kwh * config.storage_discharge_efficiency)
                state = "discharging"
        elif target_kw < 0.0 and soc < 0.98:
            charge_kw = min(abs(target_kw), rated_kw, max((0.98 - soc) * rated_kwh / dt_hours / config.storage_charge_efficiency, 0.0))
            actual_kw = -charge_kw
            if charge_kw > 0.0:
                soc += charge_kw * dt_hours * config.storage_charge_efficiency / rated_kwh
                state = "charging"
    soc = float(np.clip(soc, 0.0, 1.0))
    q_target = _volt_var_q(local_voltage_pu, asset.kva, abs(actual_kw), mode)
    _, q_dispatch = _enforce_apparent_limit(asset.kva, abs(actual_kw), q_target)
    if state == "charging":
        dss.Text.Command(
            f"edit storage.{asset.name} enabled={'yes' if status else 'no'} state=charging "
            f"kW={abs(actual_kw):.6f} kvar={q_dispatch:.6f} %stored={soc * 100.0:.4f}"
        )
    elif state == "discharging":
        dss.Text.Command(
            f"edit storage.{asset.name} enabled={'yes' if status else 'no'} state=discharging "
            f"kW={abs(actual_kw):.6f} kvar={q_dispatch:.6f} %stored={soc * 100.0:.4f}"
        )
    else:
        dss.Text.Command(
            f"edit storage.{asset.name} enabled={'yes' if status else 'no'} state=idling "
            "kW=0 kvar=0 "
            f"%stored={soc * 100.0:.4f}"
        )
    return (
        {
            f"bess_{asset.name}_target_kw": target_kw,
            f"bess_{asset.name}_actual_kw": actual_kw,
            f"bess_{asset.name}_setpoint_q_kvar": q_dispatch,
            f"bess_{asset.name}_soc": soc,
            f"bess_{asset.name}_energy_kwh": soc * rated_kwh,
            f"bess_{asset.name}_status": status,
            f"bess_{asset.name}_mode": mode,
            f"bess_{asset.name}_state": state,
        },
        soc,
    )


def _apply_discrete_overrides(override_row: pd.Series | None, inventory: CircuitInventory) -> None:
    if override_row is None:
        return
    for regulator in inventory.regulators:
        key = f"regulator_{regulator}_vreg"
        if key in override_row.index and pd.notna(override_row[key]):
            dss.Text.Command(f"edit regcontrol.{regulator} vreg={float(override_row[key]):.4f}")
    for capacitor in inventory.capacitors:
        key = f"capacitor_{capacitor}_state"
        if key in override_row.index and pd.notna(override_row[key]):
            dss.Text.Command(f"edit capacitor.{capacitor} states=({int(override_row[key])})")
    for switch in inventory.switches:
        key = f"switch_{switch}_state"
        if key in override_row.index and pd.notna(override_row[key]):
            dss.Text.Command(f"{'close' if int(override_row[key]) else 'open'} line.{switch} 1")


def _collect_truth_row(
    config: PipelineConfig,
    inventory: CircuitInventory,
    timestamp: pd.Timestamp,
    step_index: int,
    scenario_id: str,
    split_id: str,
    env_row: pd.Series,
    control_row: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp_utc": timestamp,
        "simulation_index": step_index,
        "scenario_id": scenario_id,
        "run_id": config.run_id,
        "split_id": split_id,
        "source_layer": "truth",
        "sample_rate_seconds": config.simulation_resolution_seconds,
    }
    row.update(env_row.to_dict())
    row.update(_feeder_metrics(config.feeder_head_line))
    for bus in inventory.buses:
        row.update(_bus_node_metrics(bus, inventory.bus_phase_nodes.get(bus, [])))
    for line in inventory.lines:
        if line.name in config.selected_lines:
            row.update(_line_metrics(line.name))
        row.update(_load_aggregate_metrics(inventory.loads))
        for regulator in inventory.regulators:
            row[f"regulator_{regulator}_tap_pos"] = _regulator_tap_position(regulator)
        for capacitor in inventory.capacitors:
            row[f"capacitor_{capacitor}_state"] = _capacitor_state(capacitor)
        for switch in inventory.switches:
            row[f"switch_{switch}_state"] = _switch_state(switch)
        row["breaker_substation_state"] = _switch_state("sw1") if "sw1" in inventory.switches else 1
        for asset in inventory.pv_assets:
            row.update(_der_terminal_metrics("PVSystem", asset.name, f"pv_{asset.name}"))
            row[f"pv_{asset.name}_available_kw"] = control_row[f"pv_{asset.name}_available_kw"]
            row[f"pv_{asset.name}_curtailment_frac"] = control_row[f"pv_{asset.name}_curtailment_frac"]
            row[f"pv_{asset.name}_mode"] = control_row[f"pv_{asset.name}_mode"]
            row[f"pv_{asset.name}_status"] = control_row[f"pv_{asset.name}_status"]
        for asset in inventory.bess_assets:
            row.update(_der_terminal_metrics("Storage", asset.name, f"bess_{asset.name}"))
            row[f"bess_{asset.name}_soc"] = control_row[f"bess_{asset.name}_soc"]
            row[f"bess_{asset.name}_energy_kwh"] = control_row[f"bess_{asset.name}_energy_kwh"]
            row[f"bess_{asset.name}_reserve_soc"] = asset.reserve_soc
            row[f"bess_{asset.name}_soc_min"] = asset.reserve_soc
            row[f"bess_{asset.name}_soc_max"] = 1.0
            row[f"bess_{asset.name}_available_discharge_kw"] = max(
                (control_row[f"bess_{asset.name}_soc"] - asset.reserve_soc) * float(asset.kwh_rated or 0.0) / (config.simulation_resolution_seconds / 3600.0) * config.storage_discharge_efficiency,
                0.0,
            )
            row[f"bess_{asset.name}_available_charge_kw"] = max(
                (1.0 - control_row[f"bess_{asset.name}_soc"]) * float(asset.kwh_rated or 0.0) / (config.simulation_resolution_seconds / 3600.0) / config.storage_charge_efficiency,
                0.0,
            )
            row[f"bess_{asset.name}_mode"] = control_row[f"bess_{asset.name}_mode"]
            row[f"bess_{asset.name}_status"] = control_row[f"bess_{asset.name}_status"]
            row[f"bess_{asset.name}_state"] = control_row[f"bess_{asset.name}_state"]
        return row


def _feeder_metrics(head_line: str) -> dict[str, float]:
    dss.Circuit.SetActiveElement(f"Line.{head_line}")
    powers = dss.CktElement.Powers()
    currents = dss.CktElement.CurrentsMagAng()
    num_phases = dss.CktElement.NumPhases()
    row: dict[str, float] = {}
    phase_p = []
    phase_q = []
    for idx in range(num_phases):
        label = PHASE_LABELS[idx + 1]
        p = float(powers[2 * idx])
        q = float(powers[2 * idx + 1])
        row[f"feeder_p_kw_{label}"] = p
        row[f"feeder_q_kvar_{label}"] = q
        row[f"feeder_i_a_{label}"] = float(currents[2 * idx])
        row[f"feeder_i_angle_deg_{label}"] = float(currents[2 * idx + 1])
        phase_p.append(p)
        phase_q.append(q)
    row.update({key.replace("bus_150_", "feeder_"): value for key, value in _bus_node_metrics("150", [1, 2, 3]).items()})
    losses = dss.Circuit.Losses()
    row["feeder_p_kw_total"] = float(sum(phase_p))
    row["feeder_q_kvar_total"] = float(sum(phase_q))
    row["feeder_i_a_total"] = float(sum(abs(item) for item in currents[::2][:num_phases]))
    row["feeder_losses_kw_total"] = float(losses[0] / 1000.0)
    row["feeder_losses_kvar_total"] = float(losses[1] / 1000.0)
    return row


def _bus_node_metrics(bus: str, phases: list[int]) -> dict[str, float]:
    dss.Circuit.SetActiveBus(bus)
    pu_values = dss.Bus.PuVoltage()
    node_order = dss.Bus.Nodes()
    phasors: dict[int, complex] = {}
    for idx, node in enumerate(node_order):
        phasors[node] = complex(pu_values[2 * idx], pu_values[2 * idx + 1])
    row: dict[str, float] = {}
    for phase in phases:
        label = PHASE_LABELS[phase]
        phasor = phasors.get(phase)
        row[f"bus_{bus}_v_pu_{label}"] = abs(phasor) if phasor is not None else np.nan
        row[f"bus_{bus}_angle_deg_{label}"] = math.degrees(math.atan2(phasor.imag, phasor.real)) if phasor is not None else np.nan
    return row


def _line_metrics(name: str) -> dict[str, float]:
    dss.Circuit.SetActiveElement(f"Line.{name}")
    powers = dss.CktElement.Powers()
    currents = dss.CktElement.CurrentsMagAng()
    num_phases = dss.CktElement.NumPhases()
    row: dict[str, float] = {}
    total_p_send = 0.0
    total_q_send = 0.0
    total_p_recv = 0.0
    total_q_recv = 0.0
    for idx in range(num_phases):
        label = PHASE_LABELS[idx + 1]
        send_p = float(powers[2 * idx])
        send_q = float(powers[2 * idx + 1])
        recv_p = float(powers[2 * (idx + num_phases)])
        recv_q = float(powers[2 * (idx + num_phases) + 1])
        row[f"line_{name}_p_kw_{label}"] = send_p
        row[f"line_{name}_q_kvar_{label}"] = send_q
        row[f"line_{name}_p_kw_receiving_{label}"] = recv_p
        row[f"line_{name}_q_kvar_receiving_{label}"] = recv_q
        row[f"line_{name}_current_a_{label}"] = float(currents[2 * idx])
        total_p_send += send_p
        total_q_send += send_q
        total_p_recv += recv_p
        total_q_recv += recv_q
    row[f"line_{name}_p_kw_total"] = total_p_send
    row[f"line_{name}_q_kvar_total"] = total_q_send
    row[f"line_{name}_p_kw_receiving_total"] = total_p_recv
    row[f"line_{name}_q_kvar_receiving_total"] = total_q_recv
    return row


def _load_aggregate_metrics(loads: list[LoadSpec]) -> dict[str, float]:
    totals = {"residential": 0.0, "commercial": 0.0, "industrial": 0.0}
    total_p = 0.0
    total_q = 0.0
    for load in loads:
        dss.Loads.Name(load.name)
        kw = float(dss.Loads.kW())
        kvar = float(dss.Loads.kvar())
        total_p += kw
        total_q += kvar
        totals[load.load_class] += kw
    return {
        "load_aggregate_p_kw": total_p,
        "load_aggregate_q_kvar": total_q,
        "load_residential_p_kw": totals["residential"],
        "load_commercial_p_kw": totals["commercial"],
        "load_industrial_p_kw": totals["industrial"],
    }


def _der_terminal_metrics(class_name: str, asset_name: str, prefix: str) -> dict[str, float]:
    dss.Circuit.SetActiveElement(f"{class_name}.{asset_name}")
    powers = dss.CktElement.Powers()
    currents = dss.CktElement.CurrentsMagAng()
    num_phases = dss.CktElement.NumPhases()
    p = -float(sum(powers[2 * idx] for idx in range(num_phases)))
    q = -float(sum(powers[2 * idx + 1] for idx in range(num_phases)))
    bus = dss.CktElement.BusNames()[0].split(".")[0]
    bus_metrics = _bus_node_metrics(bus, [1, 2, 3])
    terminal_vs = [value for key, value in bus_metrics.items() if key.startswith(f"bus_{bus}_v_pu_")]
    terminal_is = [float(currents[2 * idx]) for idx in range(num_phases)]
    return {
        f"{prefix}_p_kw": p,
        f"{prefix}_q_kvar": q,
        f"{prefix}_terminal_v_pu": float(np.nanmean(terminal_vs)) if terminal_vs else np.nan,
        f"{prefix}_terminal_i_a": float(np.nanmean(terminal_is)) if terminal_is else np.nan,
    }


def _regulator_tap_position(name: str) -> float:
    dss.RegControls.Name(name)
    transformer_name = dss.RegControls.Transformer()
    dss.Transformers.Name(transformer_name)
    dss.Transformers.Wdg(2)
    tap = float(dss.Transformers.Tap())
    return round((tap - 1.0) / 0.00625)


def _capacitor_state(name: str) -> int:
    dss.Capacitors.Name(name)
    states = dss.Capacitors.States()
    return int(states[0]) if states else 0


def _switch_state(name: str) -> int:
    dss.Circuit.SetActiveElement(f"Line.{name}")
    return 0 if dss.CktElement.IsOpen(1, 0) else 1


def _override_value(row: pd.Series, column: str, default: object) -> object:
    if column not in row.index or pd.isna(row[column]):
        return default
    return row[column]


def _volt_var_q(local_voltage_pu: float, kva: float, kw: float, mode: str) -> float:
    if mode not in {"volt_var", "solar_firming", "peak_shaving"}:
        return 0.0
    if local_voltage_pu <= 0.97:
        q_pu = min((0.99 - local_voltage_pu) / 0.03, 1.0) * 0.44
    elif local_voltage_pu >= 1.03:
        q_pu = -min((local_voltage_pu - 1.01) / 0.03, 1.0) * 0.44
    else:
        q_pu = 0.0
    q = q_pu * kva
    limit = math.sqrt(max(kva ** 2 - kw ** 2, 0.0))
    return float(np.clip(q, -limit, limit))


def _enforce_apparent_limit(kva: float, p_kw: float, q_kvar: float) -> tuple[float, float]:
    apparent = math.sqrt(max(p_kw ** 2 + q_kvar ** 2, 0.0))
    if apparent <= kva or apparent == 0.0:
        return p_kw, q_kvar
    scale = kva / apparent
    return p_kw * scale, q_kvar * scale


def _append_derived_features(truth_df: pd.DataFrame, inventory: CircuitInventory, config: PipelineConfig) -> None:
    dt_seconds = max(float(config.simulation_resolution_seconds), 1.0)
    truth_df["derived_feeder_power_balance_residual_kw"] = (
        truth_df["feeder_p_kw_total"]
        - (
            truth_df["load_aggregate_p_kw"]
            - sum(truth_df[f"pv_{asset.name}_p_kw"] for asset in inventory.pv_assets)
            - sum(truth_df[f"bess_{asset.name}_p_kw"] for asset in inventory.bess_assets)
            + truth_df["feeder_losses_kw_total"]
        )
    )
    voltage_columns = [column for column in truth_df.columns if column.startswith("bus_") and "_v_pu_" in column]
    truth_df["derived_voltage_violation_count"] = (
        truth_df[voltage_columns].lt(config.nominal_voltage_limits_pu[0]).sum(axis=1)
        + truth_df[voltage_columns].gt(config.nominal_voltage_limits_pu[1]).sum(axis=1)
    )
    truth_df["derived_voltage_violation_flag"] = (truth_df["derived_voltage_violation_count"] > 0).astype(int)
    truth_df["derived_feeder_ramp_kw_per_s"] = truth_df["feeder_p_kw_total"].diff().fillna(0.0) / dt_seconds
    truth_df["derived_feeder_power_ma_60s_kw"] = truth_df["feeder_p_kw_total"].rolling(
        max(int(60 / config.simulation_resolution_seconds), 1), min_periods=1
    ).mean()
    truth_df["relay_substation_alarm_state"] = (
        (truth_df["derived_voltage_violation_count"] > 0) | (truth_df["feeder_i_a_total"] > truth_df["feeder_i_a_total"].quantile(0.99))
    ).astype(int)
    truth_df["relay_substation_trip_state"] = (
        (truth_df["derived_voltage_violation_count"] > max(len(voltage_columns) * 0.1, 1))
        | (truth_df["feeder_i_a_total"] > truth_df["feeder_i_a_total"].quantile(0.999))
    ).astype(int)
    for asset in inventory.pv_assets:
        truth_df[f"derived_pv_{asset.name}_availability_residual_kw"] = truth_df[f"pv_{asset.name}_available_kw"] - truth_df[f"pv_{asset.name}_p_kw"]
        truth_df[f"derived_pv_{asset.name}_ramp_kw_per_s"] = truth_df[f"pv_{asset.name}_p_kw"].diff().fillna(0.0) / dt_seconds
    for asset in inventory.bess_assets:
        truth_df[f"derived_bess_{asset.name}_soc_consistency_residual"] = truth_df[f"bess_{asset.name}_soc"].diff().fillna(0.0)
