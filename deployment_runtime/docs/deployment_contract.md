# Deployment Contract

## Alert Packet Contract

The alert packet schema is defined in `deployment_runtime/alert_packet_schema.json`.

Required core fields:

- `alert_id`
- `timestamp_utc`
- `site_id`
- `packet_source`
- `model_name`
- `model_profile`
- `score`
- `threshold`
- `severity`
- `modality`
- `window_start_utc`
- `window_end_utc`
- `top_features`
- `affected_assets`
- `local_context_path`

Optional fields:

- `asset_id`
- `buffer_id`
- `explanation_packet_path`
- `metadata`

## Model Artifact Expectations

### Threshold Baseline

- Artifact directory: `outputs/models_full_run/threshold_baseline`
- Required files:
  - `model.pkl`
  - `results.json`
  - `predictions.parquet`
- Runtime assumptions:
  - `model.pkl` contains `standardizer`, `calibrator`, and `model_info`.
  - `model_info.feature_columns` are aligned residual features.
  - Clean reference windows come from `outputs/windows/merged_windows_clean.parquet`.

### Autoencoder

- Artifact directory: `outputs/phase3_zero_day/latency_window_sweep/w120_s24/autoencoder`
- Required files:
  - `metadata.pkl`
  - `model.pt`
  - `results.json`
  - `predictions.parquet`
- Runtime assumptions:
  - `metadata.pkl` contains `standardizer`, `calibrator`, and `model_info`.
  - `model.pt` contains the serialized lightweight MLP autoencoder weights.
  - Clean reference windows come from `outputs/phase3_zero_day/latency_window_sweep/w120_s24/clean_windows_120s_24s.parquet`.

## Runtime Assumptions

- The prototype is local and file-based. No cloud dependency is required.
- The runtime currently uses clean reference window stores so it can remain compatible with the project's aligned residual feature mode.
- Gateway and control-center layers do not issue control actions.
- Explanation is optional and replay-grounded. The control center first tries to attach a full upstream explanation packet and otherwise falls back to a lighter bridge packet that still reuses the Phase 3 explanation logic.
