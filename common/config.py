from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class DERAssetSpec:
    name: str
    kind: str
    bus: str
    phases: int
    kv: float
    kva: float
    pmpp_kw: float | None = None
    kw_rated: float | None = None
    kwh_rated: float | None = None
    base_soc: float | None = None
    reserve_soc: float = 0.2
    control_mode: str = "volt_var"


@dataclass(slots=True)
class MeasurementConfig:
    default_numeric_noise_std: float = 0.002
    default_relative_noise_std: float = 0.005
    default_missing_probability: float = 0.001
    default_stale_probability: float = 0.0008
    default_latency_seconds: int = 2
    default_latency_jitter_seconds: int = 1
    default_clock_drift_ppm: float = 20.0
    sparse_observability_keep_fraction: float = 0.3


@dataclass(slots=True)
class WindowConfig:
    window_seconds: int = 300
    step_seconds: int = 60
    min_attack_overlap_fraction: float = 0.2


@dataclass(slots=True)
class PipelineConfig:
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:10]}")
    scenario_id: str = "clean_baseline"
    split_id: str = "train"
    random_seed: int = 1729
    timezone: str = "UTC"
    start_time_utc: str = "2025-06-01T00:00:00Z"
    duration_hours: int = 24
    shape_resolution_seconds: int = 60
    simulation_resolution_seconds: int = 1
    feeder_head_line: str = "l115"
    critical_buses: list[str] = field(
        default_factory=lambda: ["13", "18", "35", "48", "60", "83", "94", "108", "114", "150"]
    )
    selected_lines: list[str] = field(
        default_factory=lambda: ["l115", "l13", "l19", "l31", "l58", "l67", "sw4", "sw5"]
    )
    selected_switches: list[str] = field(default_factory=lambda: ["sw1", "sw2", "sw3", "sw4", "sw5", "sw6", "sw7", "sw8"])
    pv_assets: list[DERAssetSpec] = field(
        default_factory=lambda: [
            DERAssetSpec(name="pv35", kind="pv", bus="35", phases=3, kv=4.16, kva=550.0, pmpp_kw=500.0),
            DERAssetSpec(name="pv60", kind="pv", bus="60", phases=3, kv=4.16, kva=950.0, pmpp_kw=880.0),
            DERAssetSpec(name="pv83", kind="pv", bus="83", phases=3, kv=4.16, kva=700.0, pmpp_kw=640.0),
        ]
    )
    bess_assets: list[DERAssetSpec] = field(
        default_factory=lambda: [
            DERAssetSpec(
                name="bess48",
                kind="bess",
                bus="48",
                phases=3,
                kv=4.16,
                kva=600.0,
                kw_rated=500.0,
                kwh_rated=2000.0,
                base_soc=0.58,
                reserve_soc=0.20,
            ),
            DERAssetSpec(
                name="bess108",
                kind="bess",
                bus="108",
                phases=3,
                kv=4.16,
                kva=450.0,
                kw_rated=350.0,
                kwh_rated=1400.0,
                base_soc=0.63,
                reserve_soc=0.15,
            ),
        ]
    )
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    latitude_deg: float = 33.955
    longitude_deg: float = -84.55
    storage_charge_efficiency: float = 0.95
    storage_discharge_efficiency: float = 0.94
    nominal_voltage_limits_pu: tuple[float, float] = (0.95, 1.05)
    measured_bus_observability: list[str] = field(
        default_factory=lambda: ["13", "35", "48", "60", "83", "94", "108", "114", "150"]
    )
    cyber_telemetry_interval_seconds: int = 2
    cyber_auth_interval_minutes: int = 30
    output_format: str = "parquet"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        pv_assets = [DERAssetSpec(**item) for item in data.get("pv_assets", [])]
        bess_assets = [DERAssetSpec(**item) for item in data.get("bess_assets", [])]
        measurement = MeasurementConfig(**data.get("measurement", {}))
        windows = WindowConfig(**data.get("windows", {}))
        payload = {**data}
        payload["pv_assets"] = pv_assets
        payload["bess_assets"] = bess_assets
        payload["measurement"] = measurement
        payload["windows"] = windows
        return cls(**payload)


@dataclass(slots=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    opendss_root: Path = field(default_factory=lambda: PROJECT_ROOT / "opendss")
    clean_output: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "clean")
    attacked_output: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "attacked")
    labeled_output: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "labeled")
    windows_output: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "windows")
    reports_output: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "reports")
    report_file: Path = field(default_factory=lambda: PROJECT_ROOT / "reports" / "generated_report.md")

    def ensure(self) -> "ProjectPaths":
        for path in (
            self.clean_output,
            self.attacked_output,
            self.labeled_output,
            self.windows_output,
            self.reports_output,
            self.report_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def default_pipeline_config() -> PipelineConfig:
    return PipelineConfig()


def load_pipeline_config(config_path: str | Path | None = None) -> PipelineConfig:
    if config_path is None:
        return default_pipeline_config()
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return PipelineConfig.from_dict(data)


def save_pipeline_config(config: PipelineConfig, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    return out_path


def research_master_dss_path(paths: ProjectPaths | None = None) -> Path:
    resolved = (paths or ProjectPaths()).opendss_root / "Research_IEEE123_Master.dss"
    return resolved.resolve()
