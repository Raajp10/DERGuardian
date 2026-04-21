# Configs

Publication cleanup did not move active configs because existing scripts resolve paths relative to the repository root. Add new experiment configs here for future runs.

- `pipeline_config.yaml` includes a `runtime_config` section consumed by `phase1.build_clean_dataset` plus documentation-oriented sections for the three-phase public pipeline contract.
- `model_config.yaml` records model rosters, thresholds, and canonical-vs-extension labels.
- `data_config.yaml` records public sample-data and generated-output path conventions.
