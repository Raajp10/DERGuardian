# Window Source Decision

Date: 2026-04-18

## Decision

The canonical source for the window-size study is:

- `outputs/clean/measured_physical_timeseries.parquet`
- `outputs/clean/cyber_events.parquet`
- `outputs/attacked/measured_physical_timeseries.parquet`
- `outputs/attacked/cyber_events.parquet`
- `outputs/attacked/attack_labels.parquet`

with measured timestamps interpreted through the canonical builder logic:

- raw column in file: `timestamp_utc`
- effective physical alignment used by `phase1/build_windows.py`: `analysis_timestamp_utc = reconstruct_nominal_timestamps(measured)`

## Why this is the correct source

1. It is the highest-frequency canonical pre-window source available.
   - both clean and attacked measured physical parquet files are 1-second traces with 86,400 rows each
   - `sample_rate_seconds = 1`
   - reconstructed nominal physical cadence is exactly 1 second

2. It is not already windowed.
   - these files do not contain `window_start_utc` / `window_end_utc`
   - these files do not use `__mean/__std/__last` window-stat feature naming

3. It matches the repository’s actual generation path.
   - `phase1/build_clean_dataset.py` creates measured/cyber first, then windows
   - `phase2/generate_attacked_dataset.py` creates attacked measured/cyber first, then windows

4. It is sufficient for all requested window sizes.
   - smallest requested window: 5 seconds
   - source physical sampling interval: 1 second
   - therefore 5s, 10s, 60s, and 300s windows are all valid to rebuild from this source

## Files explicitly rejected as study source

Do **not** use these as the source for the window-size study:

- `outputs/windows/merged_windows_clean.parquet`
- `outputs/attacked/merged_windows.parquet`
- `outputs/windows/merged_windows_attacked.parquet` if present elsewhere in the repo
- any other parquet file that already contains:
  - `window_start_utc`
  - `window_end_utc`
  - window-stat suffixes such as `__mean`, `__std`, `__min`, `__max`, `__last`

Reason:

These are already aggregated 300-second window tables with 60-second step size, so re-windowing them for 5s / 10s / 60s would be invalid.

## Important clarification about the interrupted run

The interrupted `scripts/run_window_size_study.py` execution was reading:

- `outputs/clean/measured_physical_timeseries.parquet`
- `outputs/clean/cyber_events.parquet`
- `outputs/attacked/measured_physical_timeseries.parquet`
- `outputs/attacked/cyber_events.parquet`
- `outputs/attacked/attack_labels.parquet`

It was **not** rebuilding 5s/10s/60s windows from the already aggregated `merged_windows` parquet files.

So the source foundation is valid.

## Canonical rebuild rule going forward

Use:

- clean measured + clean cyber + empty labels for training-side clean windows
- attacked measured + attacked cyber + attack labels for evaluation-side attacked windows

Do not mix in any already-windowed artifact as the source layer.

## Final decision

Proceed with the window-size study from the 1-second measured/cyber source only.

