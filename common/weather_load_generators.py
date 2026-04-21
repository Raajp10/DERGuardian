"""Shared utility support for DERGuardian.

This module provides weather load generators helpers used across the Phase 1 data
pipeline, Phase 2 scenario pipeline, and Phase 3 evaluation/reporting layers.
The functions here are infrastructure code: they prepare paths, metadata,
profiles, graphs, units, or time alignment without changing canonical detector
outputs or benchmark decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from common.config import DERAssetSpec, PipelineConfig, ProjectPaths


SEASON_MAP = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "fall",
    10: "fall",
    11: "fall",
}


@dataclass(slots=True)
class LoadProfileSpec:
    """Structured object used by the shared DERGuardian utility workflow."""

    name: str
    bus: str
    base_kw: float
    base_kvar: float
    phases: int
    load_class: str


def build_simulation_index(config: PipelineConfig) -> pd.DatetimeIndex:
    """Build simulation index for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    total_steps = int(config.duration_hours * 3600 / config.simulation_resolution_seconds)
    return pd.date_range(
        start=pd.Timestamp(config.start_time_utc),
        periods=total_steps,
        freq=f"{config.simulation_resolution_seconds}s",
        tz="UTC",
    )


def build_shape_index(config: PipelineConfig) -> pd.DatetimeIndex:
    """Build shape index for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    total_steps = int(config.duration_hours * 3600 / config.shape_resolution_seconds)
    return pd.date_range(
        start=pd.Timestamp(config.start_time_utc),
        periods=total_steps,
        freq=f"{config.shape_resolution_seconds}s",
        tz="UTC",
    )


def generate_environmental_inputs(
    config: PipelineConfig,
    paths: ProjectPaths,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate environmental inputs for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    coarse_index = build_shape_index(config)
    sim_index = build_simulation_index(config)
    holidays = USFederalHolidayCalendar().holidays(start=coarse_index.min(), end=coarse_index.max())
    daylight = _solar_envelope(coarse_index, config.latitude_deg)
    cloud_factor = _cloud_factor(coarse_index, paths, rng)
    irradiance = 1000.0 * daylight * cloud_factor
    temperature = _temperature_profile(coarse_index, irradiance, rng)
    wind_speed = _wind_profile(coarse_index, rng)
    humidity = _humidity_profile(coarse_index, temperature, cloud_factor)

    coarse = pd.DataFrame(index=coarse_index)
    coarse["env_irradiance_wm2"] = irradiance
    coarse["env_temperature_c"] = temperature
    coarse["env_wind_speed_mps"] = wind_speed
    coarse["env_humidity_pct"] = humidity
    coarse["env_cloud_index"] = 1.0 - cloud_factor
    coarse["env_day_of_week"] = coarse.index.day_name()
    coarse["env_month"] = coarse.index.month
    coarse["env_is_weekend"] = (coarse.index.dayofweek >= 5).astype(int)
    coarse["env_is_holiday"] = coarse.index.normalize().isin(holidays.normalize()).astype(int)
    coarse["env_season"] = coarse["env_month"].map(SEASON_MAP)
    coarse = coarse.reset_index(names="timestamp_utc")

    reindexed = coarse.set_index("timestamp_utc").reindex(sim_index)
    numeric_columns = [
        "env_irradiance_wm2",
        "env_temperature_c",
        "env_wind_speed_mps",
        "env_humidity_pct",
        "env_cloud_index",
        "env_month",
        "env_is_weekend",
        "env_is_holiday",
    ]
    categorical_columns = ["env_day_of_week", "env_season"]
    numeric = reindexed[numeric_columns].interpolate(method="time", limit_direction="both")
    categorical = reindexed[categorical_columns].ffill().bfill()
    fine = pd.concat([numeric, categorical], axis=1).reset_index(names="timestamp_utc")
    return coarse, fine


def build_load_schedule(
    config: PipelineConfig,
    load_specs: list[LoadProfileSpec],
    env_df: pd.DataFrame,
    paths: ProjectPaths,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build load schedule for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    env = env_df.set_index("timestamp_utc")
    year_ref = _reference_loadshape(paths)
    timestamp_index = env.index
    ref_multiplier = year_ref.reindex(timestamp_index, method="nearest").interpolate(method="time")
    schedule_data: dict[str, np.ndarray] = {}
    class_rows = []
    for spec in load_specs:
        diversity = _smoothed_noise(len(timestamp_index), rng, scale=0.03)
        temp_sensitivity = {"residential": 0.22, "commercial": 0.12, "industrial": 0.06}[spec.load_class]
        daily = _daily_class_profile(timestamp_index, spec.load_class)
        ev = _ev_component(timestamp_index, spec.load_class, rng)
        weekend_multiplier = np.where(env["env_is_weekend"].to_numpy() > 0, 0.94, 1.03)
        temperature_adjustment = 1.0 + temp_sensitivity * ((env["env_temperature_c"].to_numpy() - 22.0) / 30.0)
        multiplier = ref_multiplier.to_numpy() * daily * weekend_multiplier * (1.0 + diversity) * temperature_adjustment + ev
        multiplier = np.clip(multiplier, 0.25, 1.95)
        schedule_data[f"load_{spec.name}_p_kw"] = spec.base_kw * multiplier
        reactive_adjustment = 1.0 + 0.03 * np.sin(np.linspace(0.0, 8.0 * math.pi, len(timestamp_index)))
        schedule_data[f"load_{spec.name}_q_kvar"] = spec.base_kvar * multiplier * reactive_adjustment
        class_rows.append(
            {
                "load_name": spec.name,
                "bus": spec.bus,
                "load_class": spec.load_class,
                "base_kw": spec.base_kw,
                "base_kvar": spec.base_kvar,
                "pf": spec.base_kw / math.sqrt(spec.base_kw ** 2 + spec.base_kvar ** 2) if spec.base_kw or spec.base_kvar else 1.0,
            }
        )
    schedule = pd.DataFrame(schedule_data, index=timestamp_index).reset_index(names="timestamp_utc")
    class_map = pd.DataFrame(class_rows)
    return schedule, class_map


def build_pv_schedule(
    config: PipelineConfig,
    pv_assets: list[DERAssetSpec],
    env_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build pv schedule for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    env = env_df.set_index("timestamp_utc")
    schedule = pd.DataFrame(index=env.index)
    for asset in pv_assets:
        site_factor = 0.96 + 0.08 * rng.random()
        soiling = 0.98 - 0.03 * rng.random()
        temp_derate = np.clip(1.0 - 0.004 * (env["env_temperature_c"].to_numpy() - 25.0), 0.85, 1.03)
        irradiance_pu = env["env_irradiance_wm2"].to_numpy() / 1000.0
        available = np.maximum(0.0, asset.pmpp_kw or 0.0) * irradiance_pu * temp_derate * site_factor * soiling
        planned_curtailment = np.where(
            (env.index.hour >= 12) & (env.index.hour <= 14) & (env["env_irradiance_wm2"].to_numpy() > 650.0),
            0.02,
            0.0,
        )
        schedule[f"pv_{asset.name}_available_kw"] = available
        schedule[f"pv_{asset.name}_curtailment_frac"] = planned_curtailment
        schedule[f"pv_{asset.name}_status_cmd"] = 1
        schedule[f"pv_{asset.name}_mode_cmd"] = asset.control_mode
    return schedule.reset_index(names="timestamp_utc")


def build_bess_schedule(
    config: PipelineConfig,
    bess_assets: list[DERAssetSpec],
    env_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build bess schedule for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    env = env_df.set_index("timestamp_utc")
    schedule = pd.DataFrame(index=env.index)
    for asset in bess_assets:
        daylight = env["env_irradiance_wm2"].to_numpy()
        power = np.zeros(len(env), dtype=float)
        midday_charge = (env.index.hour >= 10) & (env.index.hour <= 14) & (daylight > 450.0)
        evening_discharge = (env.index.hour >= 17) & (env.index.hour <= 21)
        pre_dawn_charge = (env.index.hour >= 2) & (env.index.hour <= 4)
        power[midday_charge] = -(asset.kw_rated or 0.0) * (0.45 + 0.25 * (daylight[midday_charge] / 1000.0))
        power[evening_discharge] = (asset.kw_rated or 0.0) * 0.55
        power[pre_dawn_charge] = np.minimum(power[pre_dawn_charge], -(asset.kw_rated or 0.0) * 0.20)
        power = power + _smoothed_noise(len(env), rng, scale=max((asset.kw_rated or 0.0) * 0.02, 1.0))
        power = np.clip(power, -(asset.kw_rated or 0.0), asset.kw_rated or 0.0)
        mode = np.where(power > 20.0, "peak_shaving", np.where(power < -20.0, "solar_firming", "standby"))
        schedule[f"bess_{asset.name}_target_kw"] = power
        schedule[f"bess_{asset.name}_mode_cmd"] = mode
        schedule[f"bess_{asset.name}_status_cmd"] = 1
    return schedule.reset_index(names="timestamp_utc")


def generate_reference_profile_bundle(
    config: PipelineConfig,
    load_specs: list[LoadProfileSpec],
    pv_assets: list[DERAssetSpec],
    bess_assets: list[DERAssetSpec],
    paths: ProjectPaths,
    base_seed: int,
) -> pd.DataFrame:
    """Generate reference profile bundle for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    cases = [
        ("winter", "weekday", "2025-01-15T00:00:00Z"),
        ("winter", "weekend", "2025-01-18T00:00:00Z"),
        ("spring", "weekday", "2025-04-16T00:00:00Z"),
        ("spring", "weekend", "2025-04-19T00:00:00Z"),
        ("summer", "weekday", "2025-07-16T00:00:00Z"),
        ("summer", "weekend", "2025-07-19T00:00:00Z"),
        ("fall", "weekday", "2025-10-15T00:00:00Z"),
        ("fall", "weekend", "2025-10-18T00:00:00Z"),
    ]
    records: list[pd.DataFrame] = []
    for idx, (season, day_type, start_time) in enumerate(cases):
        case_config = PipelineConfig.from_dict(config.to_dict())
        case_config.start_time_utc = start_time
        case_config.duration_hours = 24
        case_config.shape_resolution_seconds = max(config.shape_resolution_seconds, 900)
        case_config.simulation_resolution_seconds = max(config.simulation_resolution_seconds, 900)
        rng = np.random.default_rng(base_seed + idx + 1)
        _, env_df = generate_environmental_inputs(case_config, paths, rng)
        load_schedule, _ = build_load_schedule(case_config, load_specs, env_df, paths, rng)
        pv_schedule = build_pv_schedule(case_config, pv_assets, env_df, rng)
        bess_schedule = build_bess_schedule(case_config, bess_assets, env_df, rng)
        frame = env_df[["timestamp_utc", "env_irradiance_wm2", "env_temperature_c", "env_cloud_index"]].copy()
        frame["season"] = season
        frame["day_type"] = day_type
        frame["hour_of_day"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.hour + pd.to_datetime(frame["timestamp_utc"], utc=True).dt.minute / 60.0
        frame["load_aggregate_p_kw"] = load_schedule[[column for column in load_schedule.columns if column.endswith("_p_kw")]].sum(axis=1)
        frame["pv_aggregate_potential_kw"] = pv_schedule[[column for column in pv_schedule.columns if column.endswith("_available_kw")]].sum(axis=1)
        frame["bess_target_total_kw"] = bess_schedule[[column for column in bess_schedule.columns if column.endswith("_target_kw")]].sum(axis=1)
        records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def classify_load(base_kw: float, phases: int, bus: str) -> str:
    """Handle classify load within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if phases == 1 and base_kw <= 90.0:
        return "residential"
    if phases >= 3 and base_kw >= 180.0:
        return "industrial"
    if bus.endswith(("0", "5")) or phases >= 2:
        return "commercial"
    return "residential"


def _reference_loadshape(paths: ProjectPaths) -> pd.Series:
    source = paths.opendss_root / "ieee123_base" / "PaperLoadShape.txt"
    values = pd.read_csv(source, header=None).iloc[:, 0].astype(float)
    index = pd.date_range("2025-01-01T00:00:00Z", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values.to_numpy(), index=index)


def _cloud_factor(index: pd.DatetimeIndex, paths: ProjectPaths, rng: np.random.Generator) -> np.ndarray:
    source = paths.opendss_root / "ieee123_base" / "PV5sdata1.csv"
    reference = pd.read_csv(source, header=None).iloc[:, 0].astype(float)
    normalized = (reference - reference.min()) / max(reference.max() - reference.min(), 1e-6)
    source_positions = np.linspace(0, len(normalized) - 1, num=max(len(index), 1))
    tiled = np.interp(source_positions, np.arange(len(normalized)), normalized.to_numpy())
    seasonal_cloudiness = 0.82 + 0.12 * np.cos((index.dayofyear.to_numpy() - 210.0) * 2.0 * math.pi / 365.0)
    disturbances = _smoothed_noise(len(index), rng, scale=0.08)
    factor = np.clip(seasonal_cloudiness * (0.7 + 0.3 * tiled) + disturbances, 0.08, 1.0)
    return factor


def _solar_envelope(index: pd.DatetimeIndex, latitude_deg: float) -> np.ndarray:
    day_of_year = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0 + index.second.to_numpy() / 3600.0
    declination = np.deg2rad(23.45 * np.sin(np.deg2rad((360.0 / 365.0) * (284.0 + day_of_year))))
    latitude = np.deg2rad(latitude_deg)
    hour_angle = np.deg2rad(15.0 * (hour - 12.0))
    sin_altitude = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    return np.clip(sin_altitude, 0.0, 1.0)


def _temperature_profile(index: pd.DatetimeIndex, irradiance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    season = 16.0 + 12.0 * np.sin((index.dayofyear.to_numpy() - 80.0) * 2.0 * math.pi / 365.0)
    diurnal = 7.5 * np.sin((index.hour.to_numpy() - 8.0) * 2.0 * math.pi / 24.0)
    solar_warming = 4.0 * (irradiance / 1000.0)
    return season + diurnal + solar_warming + _smoothed_noise(len(index), rng, scale=0.5)


def _wind_profile(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    base = 3.5 + 1.0 * np.sin(index.hour.to_numpy() * 2.0 * math.pi / 24.0)
    wind = base + _smoothed_noise(len(index), rng, scale=0.8)
    return np.clip(wind, 0.2, 12.0)


def _humidity_profile(index: pd.DatetimeIndex, temperature: np.ndarray, cloud_factor: np.ndarray) -> np.ndarray:
    raw = 70.0 - 0.7 * (temperature - 20.0) + 12.0 * (1.0 - cloud_factor)
    return np.clip(raw, 25.0, 99.0)


def _daily_class_profile(index: pd.DatetimeIndex, load_class: str) -> np.ndarray:
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    if load_class == "residential":
        morning = np.exp(-((hour - 7.0) / 2.3) ** 2)
        evening = 1.3 * np.exp(-((hour - 19.0) / 3.0) ** 2)
        overnight = 0.35 + 0.1 * np.exp(-((hour - 2.0) / 2.0) ** 2)
        return 0.42 + 0.25 * morning + 0.45 * evening + overnight
    if load_class == "commercial":
        workday = np.exp(-((hour - 13.0) / 4.2) ** 2)
        shoulder = 0.35 * np.exp(-((hour - 8.0) / 2.0) ** 2) + 0.28 * np.exp(-((hour - 17.5) / 2.6) ** 2)
        return 0.38 + 0.8 * workday + shoulder
    return 0.85 + 0.08 * np.sin((hour - 6.0) * 2.0 * math.pi / 24.0)


def _ev_component(index: pd.DatetimeIndex, load_class: str, rng: np.random.Generator) -> np.ndarray:
    if load_class != "residential":
        return np.zeros(len(index), dtype=float)
    event_center = 20.0 + rng.normal(0.0, 0.6)
    event_width = 1.7 + abs(rng.normal(0.0, 0.3))
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    return 0.11 * np.exp(-((hour - event_center) / event_width) ** 2)


def _smoothed_noise(length: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    raw = rng.normal(0.0, scale, size=length)
    if length <= 3:
        return raw
    kernel = np.array([0.08, 0.16, 0.28, 0.16, 0.08])
    return np.convolve(raw, kernel / kernel.sum(), mode="same")[:length]
