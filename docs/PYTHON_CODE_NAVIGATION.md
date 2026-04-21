# Python Code Navigation

This guide explains how the Python code is organized and which files are most important for a developer or reviewer.

## `common/`

Shared infrastructure used by multiple phases.

- `config.py`: Pipeline configuration dataclasses and config loading/saving.
- `dss_runner.py`: OpenDSS inventory extraction and simulation helpers.
- `io_utils.py`: DataFrame/JSON/JSONL read-write helpers.
- `metadata_schema.py`: Manifest and validation-check structures.
- `noise_models.py`: Measured-layer sensor impairment simulation.
- `time_alignment.py`: Timestamp reconstruction and alignment helpers.
- `units_and_channel_dictionary.py`: Channel dictionary and unit metadata.
- `weather_load_generators.py`: Deterministic weather, load, PV, and BESS schedule generation.

## `phase1/`

Phase 1 clean-data and window-building code.

- `build_clean_dataset.py`: Main clean-data generator for truth, measured, cyber, schedules, windows, manifests, and validation reports.
- `build_windows.py`: Converts measured physical telemetry and cyber events into detector-ready fixed-width windows.
- `extract_channels.py`: Channel extraction helper.
- `validate_clean_dataset.py`: Clean-layer validation checks.

## `phase1_models/`

Detector modeling and benchmark code.

- `run_full_evaluation.py`: Full Phase 1 detector benchmark runner.
- `residual_dataset.py`: Residual/deviation feature construction.
- `feature_builder.py`: Standardization, discretization, and feature transforms.
- `metrics.py`: Binary metrics, curves, latency, and per-scenario metrics.
- `neural_models.py`: MLP autoencoder, GRU, LSTM, transformer, and token baseline architectures.
- `neural_training.py`: Shared training and prediction utilities.
- `model_loader.py`: Loads frozen ready packages and runs inference for replay/deployment-style audits.
- `ready_package_utils.py`: Ready-package export and calibration serialization.
- `train_*.py`: Individual detector training entrypoints.

## `phase1_models/context/`

Support code for context summaries and fusion.

- `phase1_context_builder.py`: Builds structured normal-system context summaries.
- `phase1_llm_reasoner.py`: Produces bounded reasoning/context artifacts used for support analysis.
- `phase1_fusion.py`: Builds fusion inputs and fusion calibration artifacts.

These files support explanation and fusion layers. They do not replace the canonical detector-selection result.

## `phase2/`

Scenario and attacked-dataset pipeline.

- `scenario_schema.json`: Structured scenario contract.
- `research_attack_scenarios.json`: Canonical scenario bundle.
- `validate_scenarios.py`: Schema and physics/safety validation.
- `compile_injections.py`: Converts validated scenarios into physical and measured-layer actions.
- `generate_attacked_dataset.py`: Produces attacked truth/measured/cyber data, labels, attacked windows, manifests, and validation reports.
- `cyber_log_generator.py`: Baseline and attack cyber-event generation.
- `validate_attacked_dataset.py`: Attacked-layer validation against labels and clean baseline.
- `reporting.py`: Phase 2 reports and summaries.

## `phase3/`

Evaluation and analysis helpers.

- `run_zero_day_evaluation.py`: Zero-day-like evaluation helper code.
- `zero_day_splitter.py`: Split helpers for heldout/zero-day-like contexts.
- `zero_day_report.py`: Zero-day-like report generation.
- `error_analysis.py`: Error summaries.
- `run_*_ablation.py`: Modality, FDI, sequence, and latency ablation scripts.
- `build_final_artifacts.py`: Final Phase 3 artifact packaging.

## `phase3_explanations/`

Grounded explanation layer.

- `build_explanation_packet.py`: Joins detector predictions, physical evidence, cyber evidence, and scenario metadata into one packet.
- `classify_attack_family.py`: Bounded attack-family hinting from packet evidence.
- `generate_explanations_llm.py`: LLM-assisted explanation generation from grounded packets.
- `rationale_evaluator.py`: Explanation/rationale scoring.
- `validate_explanations.py`: Schema and content validation.
- `shared.py`: Shared loading, normalization, and safe operator-action helpers.

This layer is post-alert operator support. It is not human-like root-cause analysis.

## `deployment_runtime/`

Offline runtime demonstration and deployment-benchmark support.

- `runtime_common.py`: Shared runtime helpers.
- `load_deployed_models.py`: Loads packaged models for runtime-style inference.
- `stream_window_builder.py`: Builds stream-like windows.
- `edge_runtime.py`, `gateway_runtime.py`, `control_center_runtime.py`: Offline runtime roles used for architecture demonstration.
- `build_alert_packet.py`: Alert packet builder.
- `latency_budget.py`: Latency-budget helper.
- `export_deployment_report.py`: Runtime/deployment report writer.

This code supports offline benchmarking and demonstrations only. It is not evidence of field edge deployment.

## `scripts/`

Release, audit, and experiment orchestration.

- `run_full_pipeline.py`: Recommended release-facing entrypoint.
- `run_window_size_study.py`: Canonical Phase 1 window/model benchmark runner.
- `run_detector_window_zero_day_audit.py`: Detector window coverage and heldout synthetic zero-day-like evaluation audit.
- `run_phase1_ttm_extension.py`: TTM extension experiment runner.
- `run_phase3_lora_extension.py`: Experimental LoRA explanation/classification branch.
- `run_deployment_benchmark.py`: Offline lightweight deployment benchmark.
- `build_xai_and_final_reports.py`: XAI validation and final report generation.
- `final_perfection_verify.py`: Release-layout and claim-safety verifier.

## `tests/`

Smoke and contract tests.

- `test_phase1_code_completion.py`
- `test_phase1_ready_packages.py`
- `test_phase1_upgrade_pipeline.py`
- `test_phase2_remediation.py`
- `test_phase3_pipeline.py`

The tests are intended to catch missing contracts and package regressions without changing scientific outputs.

