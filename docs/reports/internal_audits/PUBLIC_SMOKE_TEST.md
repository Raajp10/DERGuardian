# Public Smoke Test

## Scope

The smoke test intentionally avoided rerunning heavy canonical experiments or overwriting frozen outputs. It used the public sample parquet, dependency imports, config loading, and Python syntax compilation.

## Commands Tested

- `python -m compileall common phase1 phase1_models phase2 phase3 phase3_explanations deployment_runtime scripts tests -q`
- Import smoke test for:
  - `common.config.load_pipeline_config`
  - `phase1.build_windows.build_merged_windows`
  - `phase1_models.context.phase1_context_builder`
  - `phase1_models.model_loader`
  - `phase3_explanations.build_explanation_packet`
  - `torch`
  - `opendssdirect`
- Sample read:
  - `data/sample_or_manifest_only/small_sample_windows.parquet`

## Result

- Syntax compilation: pass
- Sample parquet read: pass
- Key imports: pass after config fix
- Full heavy experiment rerun: intentionally not performed

## Fix Made During This Pass

The first smoke attempt found that `configs/pipeline_config.yaml` was documentation-oriented while `load_pipeline_config()` only loaded JSON into `PipelineConfig`. This would have broken a first real Phase 1 run from README/orchestrator commands.

Fix:

- `common/config.py` now supports JSON and YAML config files.
- `configs/pipeline_config.yaml` now contains a `runtime_config` section consumed by Phase 1 runtime code while preserving documentation-oriented sections.

This is an execution/packaging fix only. It does not change model outputs, metrics, thresholds, canonical benchmark decisions, or frozen artifacts.

