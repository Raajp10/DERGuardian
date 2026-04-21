# Phase 1 File Map

## Scope

Phase 1 is the clean-data generation, window construction, detector training/evaluation, and ready-model packaging layer.

## Canonical Entry Points

- `phase1/build_clean_dataset.py`
- `phase1/validate_clean_dataset.py`
- `phase1/build_windows.py`
- `phase1_models/run_full_evaluation.py`
- `phase1_model_loader.py` for packaged-model loading and downstream reuse

## Supporting Files

- `phase1/extract_channels.py`
- `phase1_models/dataset_loader.py`
- `phase1_models/feature_builder.py`
- `phase1_models/metrics.py`
- `phase1_models/model_utils.py`
- `phase1_models/neural_models.py`
- `phase1_models/neural_training.py`
- `phase1_models/ready_package_utils.py`
- `phase1_models/residual_dataset.py`
- `phase1_models/sequence_train_utils.py`
- `phase1_models/thresholds.py`

## Configs And Shared Infrastructure

- `common/config.py`
- `common/dss_runner.py`
- `common/weather_load_generators.py`
- `common/noise_models.py`
- `common/time_alignment.py`
- `common/io_utils.py`
- `common/metadata_schema.py`
- `common/graph_builders.py`
- `common/units_and_channel_dictionary.py`
- `opendss/Research_IEEE123_Master.dss`
- `opendss/DER_Assets.dss`
- `opendss/DER_Controls.dss`
- `opendss/Monitor_Config.dss`
- `opendss/ieee123_base/*`

These are shared infrastructure, but Phase 1 is the first major consumer.

## Generated Outputs

- `outputs/clean/truth_physical_timeseries.parquet`
- `outputs/clean/measured_physical_timeseries.parquet`
- `outputs/clean/cyber_events.parquet`
- `outputs/windows/merged_windows_clean.parquet`
- `outputs/models_full_run/*`
- `outputs/reports/model_full_run_artifacts/*`
- `outputs/phase1_ready_models/*`

## Validations

- `phase1/validate_clean_dataset.py`
- ready-package verification path in `phase1_model_loader.py`

## Reports

- `phase1_models/export_model_report.py`
- `phase1_models/generate_paper_figures.py`
- `phase1_models/PHASE1_UPGRADED_PIPELINE.md`
- generated artifacts under `outputs/reports/model_full_run_artifacts/`

## Paper-Safe Files To Cite

- `phase1/build_clean_dataset.py`
- `phase1/build_windows.py`
- `phase1_models/run_full_evaluation.py`
- `phase1_models/residual_dataset.py`
- `phase1_models/feature_builder.py`
- `phase1_models/neural_models.py`
- `phase1_models/ready_package_utils.py`
- `phase1_model_loader.py`
- `outputs/reports/model_full_run_artifacts/feature_importance_table.csv`
- `outputs/reports/model_full_run_artifacts/phase1_ready_package_verification.json`

## Files To Ignore Or Treat As Secondary

- `phase1_models/evaluate_all_models.py`: older/secondary evaluation entry, superseded by `run_full_evaluation.py`
- `phase1_models/train_gru.py`
- `phase1_models/train_lstm.py`
- `phase1_models/train_transformer.py`
- `phase1_models/train_autoencoder.py`
- `phase1_models/train_isolation_forest.py`
- `phase1_models/train_threshold_baseline.py`
- `phase1_models/train_llm_baseline.py`

Those training scripts are still useful, but they are not the cleanest canonical citation path for the final paper because the integrated evaluation logic now sits in `run_full_evaluation.py`.

## Conservative Canonical Story

1. Build clean feeder data with `phase1/build_clean_dataset.py`.
2. Validate the clean artifact with `phase1/validate_clean_dataset.py`.
3. Convert time series into detector windows with `phase1/build_windows.py`.
4. Train/evaluate/package detectors with `phase1_models/run_full_evaluation.py`.
5. Reuse packaged detectors downstream through `phase1_model_loader.py`.
