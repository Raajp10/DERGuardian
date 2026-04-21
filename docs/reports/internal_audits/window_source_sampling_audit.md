# Window Source Sampling Audit

Date: 2026-04-18

## Scope

This audit checks whether the canonical source files used before `phase1/build_windows.py::build_merged_windows(...)` are genuinely pre-window time series, or whether they are already aggregated/windowed artifacts that would invalidate a 5s / 10s / 60s / 300s window-size study.

## Canonical code path checked

- `phase1/build_clean_dataset.py`
  - writes `outputs/clean/truth_physical_timeseries.parquet`
  - writes `outputs/clean/measured_physical_timeseries.parquet`
  - writes `outputs/clean/cyber_events.parquet`
  - then calls `build_merged_windows(measured_df, cyber_df, empty_labels, config.windows)`
- `phase2/generate_attacked_dataset.py`
  - writes `outputs/attacked/truth_physical_timeseries.parquet`
  - writes `outputs/attacked/measured_physical_timeseries.parquet`
  - writes `outputs/attacked/cyber_events.parquet`
  - then calls `build_merged_windows(attacked_measured, cyber_df, labels_df, config.windows)`
- `phase1/build_windows.py`
  - uses measured physical data plus cyber events plus labels
  - reconstructs `analysis_timestamp_utc` from measured `timestamp_utc + simulation_index + sample_rate_seconds`

## Key finding

The canonical physical source files are **1-second pre-window time-series parquet files**, not already-windowed artifacts.

The existing files:

- `outputs/windows/merged_windows_clean.parquet`
- `outputs/attacked/merged_windows.parquet`

are already aggregated 300-second windows with 60-second step size and must **not** be reused as the source for 5s / 10s / 60s studies.

## File-by-file audit

| Path | Rows | Cols | Timestamp used | Inferred interval | Pre-window or aggregated | Valid for 5s/10s/60s/300s? | Notes |
|---|---:|---:|---|---|---|---|---|
| `outputs/clean/truth_physical_timeseries.parquet` | 86,400 | 707 | `timestamp_utc` | exact 1.0 s | Pre-window raw truth time series | Yes as reference; not direct window-builder input | Contains `simulation_index` and `sample_rate_seconds=1`; one row per second over 24 h. |
| `outputs/clean/measured_physical_timeseries.parquet` | 86,400 | 196 | raw `timestamp_utc`; effective builder time is reconstructed `analysis_timestamp_utc` | nominal 1.0 s; observed raw timestamp drift about 1.00002 s | Pre-window measured time series | Yes; this is the canonical clean physical source | `build_merged_windows` reconstructs nominal 1-second timestamps before windowing. |
| `outputs/attacked/truth_physical_timeseries.parquet` | 86,400 | 707 | `timestamp_utc` | exact 1.0 s | Pre-window raw truth time series | Yes as reference; not direct window-builder input | Same 1-second cadence as clean truth. |
| `outputs/attacked/measured_physical_timeseries.parquet` | 86,400 | 196 | raw `timestamp_utc`; effective builder time is reconstructed `analysis_timestamp_utc` | nominal 1.0 s; observed raw timestamp drift about 1.00002 s | Pre-window measured time series | Yes; this is the canonical attacked physical source | Direct attacked input to `build_merged_windows`. |
| `outputs/clean/cyber_events.parquet` | 388,992 | 42 | `timestamp_utc` | irregular event stream; modal delta 0.0 s, max observed 1.0 s | Pre-window event stream | Yes as supplemental window-builder input | Multiple cyber events can share the same second; this is expected and valid. |
| `outputs/attacked/cyber_events.parquet` | 386,740 | 42 | `timestamp_utc` | irregular event stream; modal delta 0.0 s, max observed 1.0 s | Pre-window event stream | Yes as supplemental window-builder input | Includes attack events in addition to baseline telemetry/command/auth/config events. |
| `outputs/clean/input_environment_timeseries.parquet` | 86,400 | 11 | `timestamp_utc` | exact 1.0 s | Pre-window upstream exogenous time series | Upstream-valid, but not direct input to `build_merged_windows` | Supports truth simulation, not direct window construction. |
| `outputs/clean/control_schedule.parquet` | 86,400 | 38 | `timestamp_utc` | exact 1.0 s | Pre-window upstream control trace | Upstream-valid, but not direct input to `build_merged_windows` | Used to generate cyber/control behavior, not direct window construction. |
| `outputs/windows/merged_windows_clean.parquet` | 1,435 | 930 | `window_start_utc` | exact 60.0 s step | Already aggregated windows | No for 5s/10s/60s source; only valid as an already-built 300s artifact | Contains window columns and `__mean/__std/__min/__max/__last` feature suffixes. |
| `outputs/attacked/merged_windows.parquet` | 1,435 | 930 | `window_start_utc` | exact 60.0 s step | Already aggregated windows | No for 5s/10s/60s source; only valid as an already-built 300s artifact | Same aggregation pattern as clean merged windows. |

## Supporting evidence

### Config evidence

`outputs/clean/config_snapshot.json` contains:

- `simulation_resolution_seconds = 1`
- canonical windows default: `window_seconds = 300`, `step_seconds = 60`

This means the pipeline itself was generated at 1-second physical resolution first, then windowed later into 300-second artifacts.

### Timestamp handling evidence

`phase1/build_windows.py` does this for measured data:

- parses `timestamp_utc`
- reconstructs `analysis_timestamp_utc = reconstruct_nominal_timestamps(measured)`

For the measured parquet files, the raw observed clock carries slight drift/noise, but the reconstructed analysis clock is exact 1-second cadence:

- clean measured nominal analysis delta: 1.0 s
- attacked measured nominal analysis delta: 1.0 s

### Windowed-file evidence

The merged window parquet files show both:

- `window_start_utc` / `window_end_utc`
- feature names with window-stat suffixes such as `__mean`, `__std`, `__min`, `__max`, `__last`

Those files are therefore already aggregated feature tables, not raw source traces.

## Raw monitor export check

A search found only:

- `opendss/Monitor_Config.dss`

This file defines optional OpenDSS monitor objects for traceability, but there are **no canonical exported monitor CSV/Parquet files** being used as the active benchmark source.

So the effective canonical pre-window physical source is the saved parquet layers:

- `outputs/clean/truth_physical_timeseries.parquet`
- `outputs/clean/measured_physical_timeseries.parquet`
- `outputs/attacked/truth_physical_timeseries.parquet`
- `outputs/attacked/measured_physical_timeseries.parquet`

## Strict conclusion

The valid source resolution for a window-size study is present in this repository:

- physical source cadence: **1 second**
- cyber source: **irregular event stream aligned by timestamp**
- existing `merged_windows` files: **already aggregated 300-second outputs and not acceptable as the source for smaller windows**

