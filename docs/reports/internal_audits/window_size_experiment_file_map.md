# Window Size Experiment File Map

## Canonical Inputs Used

### Clean physical data

- `outputs/clean/truth_physical_timeseries.parquet`
  - Canonical clean truth layer emitted by the Phase 1 clean-data pipeline.
- `outputs/clean/measured_physical_timeseries.parquet`
  - Canonical clean measured layer used for rebuilding clean windows.
- `outputs/clean/cyber_events.parquet`
  - Canonical clean cyber layer used when rebuilding clean windows.

### Attacked physical data

- `outputs/attacked/truth_physical_timeseries.parquet`
  - Canonical attacked truth layer emitted by the Phase 2 attacked-data generator.
- `outputs/attacked/measured_physical_timeseries.parquet`
  - Canonical attacked measured layer used for rebuilding attacked windows.
- `outputs/attacked/cyber_events.parquet`
  - Canonical attacked cyber layer used for rebuilding attacked windows.

### Attack labels

- `outputs/attacked/attack_labels.parquet`
  - Canonical Phase 2 attack labels.
  - This is the label source consumed by the canonical residual builder in `phase1_models/residual_dataset.py`.

### Merged windows

- `outputs/windows/merged_windows_clean.parquet`
  - Canonical clean merged-window artifact.
- `outputs/attacked/merged_windows.parquet`
  - Canonical attacked merged-window artifact used by the Phase 1 residual benchmark.

### Residual dataset

- `outputs/reports/model_full_run_artifacts/residual_windows_full_run.parquet`
  - Canonical persisted aligned-residual full-run benchmark dataset.

### Canonical benchmark code

- `phase1/build_windows.py`
  - Canonical merged-window builder.
- `phase1_models/residual_dataset.py`
  - Canonical aligned-residual constructor and canonical source-path definitions.
- `phase1_models/run_full_evaluation.py`
  - Canonical Phase 1 benchmark suite for:
    - `threshold_baseline`
    - `isolation_forest`
    - `autoencoder`
    - `gru`
    - `lstm`
    - `transformer`
    - `llm_baseline`
- `phase2/generate_attacked_dataset.py`
  - Canonical Phase 2 attacked-data generator.
- `phase2/contracts.py`
  - Canonical Phase 2 execution-path contract.

## Why These Are Canonical

- `phase1_models/residual_dataset.py` explicitly defines the canonical attacked window source as `outputs/attacked/merged_windows.parquet`.
- `phase1_models/run_full_evaluation.py` is the active integrated benchmark driver used by the repo’s current Phase 1 evaluation path.
- `phase2/generate_attacked_dataset.py` is the active JSON-driven attacked-data generator and writes the attacked artifacts later consumed by Phase 1.
- The current repo-level outputs already contain the full canonical clean, attacked, windowed, and residual artifacts with matching paths and schemas.

## Explicitly Ignored

- `outputs/windows/merged_windows_attacked.parquet`
  - Duplicate attacked-window copy written by Phase 2, but not the canonical source referenced by `phase1_models/residual_dataset.py`.
- `outputs/reports/model_full_run_artifacts/tmp_clean_windows_120.parquet`
- `outputs/reports/model_full_run_artifacts/tmp_attacked_windows_120.parquet`
  - Temporary artifacts, not canonical benchmark inputs.
- `phase2/example_scenarios.json`
- `phase2/example_scenario_bundle/*`
  - Example scenario sources, not the canonical currently materialized attacked dataset in `outputs/attacked/`.
- Any removed benchmark sidecars or paper packaging folders
  - Not part of the cleaned canonical workspace anymore.

## Experiment Rule Used In This Study

- Canonical source data stay read-only.
- Rebuilt window artifacts, residual datasets, model runs, summaries, and figures are written only under:
  - `outputs/window_size_study/`
