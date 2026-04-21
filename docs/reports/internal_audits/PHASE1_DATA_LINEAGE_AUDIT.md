# Phase 1 Data Lineage Audit

This audit reconstructs the canonical Phase 1 path from raw clean/attacked data through windowing, residualization, split assignment, and benchmark model packaging.

## Canonical raw sources

- Clean measured physical time series: `outputs/clean/measured_physical_timeseries.parquet`
- Clean cyber event stream: `outputs/clean/cyber_events.parquet`
- Attacked measured physical time series: `outputs/attacked/measured_physical_timeseries.parquet`
- Attacked cyber event stream: `outputs/attacked/cyber_events.parquet`
- Attack labels: `outputs/attacked/attack_labels.parquet`

The run manifest confirms these are pre-window time-series artifacts rather than merged-window artifacts.

## Transformation chain

1. Clean and attacked measured/cyber streams are loaded from `outputs/clean/` and `outputs/attacked/`.
2. `phase1/build_windows.py` reconstructs `analysis_timestamp_utc` from `timestamp_utc + simulation_index * sample_rate_seconds` when available.
3. `build_merged_windows(...)` aggregates physical signals into `mean/std/min/max/last` window statistics and appends cyber event-count features.
4. Window labels are assigned by attack-label overlap using `min_attack_overlap_fraction = 0.2`.
5. `build_aligned_residual_dataframe(...)` inner-joins attacked and clean windows on `window_start_utc` and computes `delta__* = attacked - clean`.
6. `assign_attack_aware_split(...)` applies scenario-aware train/val/test splits with two benign buffer windows around attack windows.
7. Per-model training selects a feature subset from the residual columns, calibrates scores on validation data, and writes a ready package.

## Clean vs attacked separation

- Clean windows are written under `outputs/window_size_study/<window>/data/merged_windows_clean.parquet`.
- Attacked windows are written under `outputs/window_size_study/<window>/data/merged_windows_attacked.parquet`.
- Residual windows are written under `outputs/window_size_study/<window>/residual_windows.parquet`.
- The canonical cached full-run residual artifact is `outputs/reports/model_full_run_artifacts/residual_windows_full_run.parquet` for the older 300 s full-run pipeline.

## Window-size contract

- `5s`: window `5` s, step `1` s.
- `10s`: window `10` s, step `2` s.
- `60s`: window `60` s, step `12` s.
- `300s`: window `300` s, step `60` s.

- Step policy from `run_manifest.json`: 20 percent of window size, rounded to integer seconds, with a floor of 1 second. This preserves the canonical 300s->60s overlap ratio.

## Canonical benchmark winner

- Source-of-truth file: `outputs/window_size_study/final_window_comparison.csv`
- Winner: `transformer` at `60s`
- Benchmark F1: `0.818182`
- Ready package: `outputs/window_size_study/60s/ready_packages/transformer`

## Benchmarked model roster

- Implemented and benchmarked in the canonical window-size study: autoencoder, gru, isolation_forest, llm_baseline, lstm, threshold_baseline, transformer.
- `autoencoder` is an MLP autoencoder, not an LSTM autoencoder.
- `llm_baseline` is a tokenized time-series baseline, not a generative LLM detector.
- `arima` is not implemented in the current repo.
- `ttm` is not part of the frozen canonical benchmark roster, but a separate extension benchmark now exists in `phase1_ttm_results.csv` and `phase1_ttm_eval_report.md`.

## Metadata vs training inputs

- Residual 60 s dataset columns: `932`
- Residual `delta__*` feature columns: `920`
- Selected canonical transformer feature columns: `64`
- Canonical transformer input schema: `aligned_residual`, sequence length `12`, input dimension `64`.

The column-level inventory is saved in `phase1_training_input_inventory.csv`.

## Environment variables

- Raw window datasets include environment features (`env_*`) because `window_dataset_summary.csv` lists them in the merged-window feature columns.
- Residual datasets also contain `delta__env_*` columns (`40` total).
- Selected canonical transformer features use `0` environment residual columns.
- So environment variables are simulated and present in the raw/model-ready window artifacts, but they are not part of the final selected 60 s transformer feature subset.

## Label-generation consistency

- Attack presence is assigned at the window level by overlap with `attack_labels.parquet`.
- Residual windows keep `attack_present`, `attack_family`, `attack_severity`, and `attack_affected_assets` metadata from attacked windows.
- Scenario-aware splitting uses `scenario_window_id` so attacked windows are split by scenario rather than by random row shuffling.

## Diagram accuracy notes

- `Baseline Models ML Models (LSTM, Autoencoder, Isolation Forest)` is incomplete because the benchmark suite also includes transformer, GRU, threshold baseline, and a tokenized LLM-style baseline.
- The diagram label can overread `Autoencoder` as `LSTM AE`, but the actual repository autoencoder is MLP-based.
- `Normal Data Generation OpenDSS Simulation (Voltage, Current, Power, SOC, Temp)` is directionally true, but the actual monitored channels also include feeder losses, control states, cyber counts, and derived residual features.
- `LLM Analysis Context Learning of Normal Behavior` exists as context-summary and reasoning artifacts in the legacy full-run pipeline; it is not part of the frozen canonical benchmark winner selection.

## Canonical vs legacy note

- The older `outputs/reports/model_full_run_artifacts/` pipeline still exists and is useful for lineage/reference.
- The frozen source-of-truth benchmark selection is the newer window-size study under `outputs/window_size_study/`.
