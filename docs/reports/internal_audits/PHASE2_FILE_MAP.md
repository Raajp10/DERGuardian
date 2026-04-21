# Phase 2 File Map

## Scope

Phase 2 is the attacked-dataset generation layer. It turns validated attack scenarios into physical overrides, cyber events, labels, manifests, and attacked windows.

## Canonical Entry Points

- `phase2/validate_scenarios.py`
- `phase2/generate_attacked_dataset.py`
- `phase2/validate_attacked_dataset.py`

## Supporting Files

- `phase2/contracts.py`
- `phase2/compile_injections.py`
- `phase2/cyber_log_generator.py`
- `phase2/reporting.py`
- `phase2/merge_physical_and_cyber.py`

## Core Configs And Scenario Definitions

- `phase2/research_attack_scenarios.json`
- `phase2/scenario_schema.json`

These are the canonical Phase 2 scenario sources to cite for the final pipeline.

## Generated Outputs

- `outputs/attacked/truth_physical_timeseries.parquet`
- `outputs/attacked/measured_physical_timeseries.parquet`
- `outputs/attacked/cyber_events.parquet`
- `outputs/attacked/attack_labels.parquet`
- `outputs/attacked/scenario_manifest.json`
- `outputs/windows/merged_windows_attacked.parquet`
- Phase 2 reports/manifests emitted by `phase2/reporting.py`

## Validations

- `phase2/validate_scenarios.py`
- `phase2/validate_attacked_dataset.py`

## Reports

- `phase2/reporting.py`
- generated manifests and markdown/JSON outputs under `outputs/attacked/` and report folders

## Paper-Safe Files To Cite

- `phase2/research_attack_scenarios.json`
- `phase2/scenario_schema.json`
- `phase2/validate_scenarios.py`
- `phase2/generate_attacked_dataset.py`
- `phase2/compile_injections.py`
- `phase2/cyber_log_generator.py`
- `phase2/validate_attacked_dataset.py`
- `outputs/attacked/scenario_manifest.json`

## Files To Ignore Or Treat As Legacy/Secondary

- `phase2/example_scenarios.json`
- `phase2/example_scenario_bundle/*`

These are examples, not the canonical research scenario bank.

- `phase2_with_llm/*`

Useful as optional LLM-facing infrastructure, but not the canonical active attacked-dataset pipeline.

- `phase2_llm_benchmark/*`

This is a benchmark and analysis layer around LLM-generated scenarios and XAI evaluation. It is valuable research material, but it is not the core attacked-dataset generator to cite as the main Phase 2 pipeline.

## Conservative Canonical Story

1. Validate the scenario set with `phase2/validate_scenarios.py`.
2. Compile attack actions and overlays via `phase2/compile_injections.py`.
3. Build cyber logs with `phase2/cyber_log_generator.py`.
4. Materialize attacked physical/cyber datasets with `phase2/generate_attacked_dataset.py`.
5. Validate the finished attacked dataset with `phase2/validate_attacked_dataset.py`.

## Shared Infrastructure Worth Mentioning But Not Mixing

- Phase 2 consumes clean artifacts from Phase 1.
- Phase 2 produces the attacked artifacts later used by Phase 1 detectors and Phase 3 evaluation.
- `demo/` and `deployment_runtime/` are consumers of Phase 2 outputs, not part of the canonical Phase 2 generator itself.
