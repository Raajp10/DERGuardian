# Phase 3 File Map

## Scope

Phase 3 is the generalization, zero-day evaluation, explanation, and deployment-prototype layer.

## Canonical Entry Points

- `phase3/run_zero_day_evaluation.py`
- `phase3/build_final_artifacts.py`

## Canonical Ablation And Sweep Drivers

- `phase3/run_modality_ablation.py`
- `phase3/run_fdi_ablation.py`
- `phase3/run_latency_window_sweep.py`
- `phase3/run_sequence_model_sweep.py`
- `phase3/error_analysis.py`

These are active supporting experiments, not the single main entry point, but they are part of the canonical Phase 3 study set.

## Supporting Files

- `phase3/experiment_utils.py`
- `phase3/zero_day_splitter.py`
- `phase3/zero_day_report.py`
- `phase3/fdi_feature_builder.py`

## Explanation Layer

- `phase3_explanations/build_explanation_packet.py`
- `phase3_explanations/classify_attack_family.py`
- `phase3_explanations/generate_explanations_llm.py`
- `phase3_explanations/validate_explanations.py`
- `phase3_explanations/rationale_evaluator.py`
- `phase3_explanations/shared.py`
- `phase3_explanations/explanation_schema.json`
- `phase3_explanations/explanation_prompt_system.txt`
- `phase3_explanations/explanation_prompt_user_template.txt`
- `phase3_explanations/docs/*`

These are canonical if the paper discusses grounded XAI or operator-facing explanation outputs.

## Deployment Prototype Files

- `deployment_runtime/demo_replay.py`
- `deployment_runtime/edge_runtime.py`
- `deployment_runtime/gateway_runtime.py`
- `deployment_runtime/control_center_runtime.py`
- `deployment_runtime/load_deployed_models.py`
- `deployment_runtime/stream_window_builder.py`
- `deployment_runtime/local_buffer.py`
- `deployment_runtime/build_alert_packet.py`
- `deployment_runtime/export_deployment_report.py`
- `deployment_runtime/latency_budget.py`
- `deployment_runtime/runtime_common.py`
- `deployment_runtime/config_runtime.json`
- `deployment_runtime/alert_packet_schema.json`

These belong in Phase 3 because they operationalize the detector/explanation stack as a replayable deployment architecture.

## Generated Outputs

- `outputs/phase3_zero_day/*`
- `outputs/reports/phase3_zero_day_artifacts/*`
- `outputs/reports/explanation_artifacts/*`
- `outputs/reports/phase3_error_analysis*`
- `outputs/deployment_runtime/*`
- `outputs/reports/deployment_runtime/*`

## Validations

- `phase3_explanations/validate_explanations.py`
- report-generation and cross-check logic inside `phase3/build_final_artifacts.py`

## Reports

- `phase3/zero_day_report.py`
- `phase3/build_final_artifacts.py`
- `phase3_explanations/docs/explanation_contract.md`
- `phase3_explanations/docs/explanation_usage.md`
- `deployment_runtime/README.md`
- `deployment_runtime/docs/*`

## Paper-Safe Files To Cite

- `phase3/run_zero_day_evaluation.py`
- `phase3/zero_day_splitter.py`
- `phase3/experiment_utils.py`
- `phase3/run_latency_window_sweep.py`
- `phase3/build_final_artifacts.py`
- `phase3_explanations/build_explanation_packet.py`
- `phase3_explanations/classify_attack_family.py`
- `phase3_explanations/generate_explanations_llm.py`
- `phase3_explanations/validate_explanations.py`
- `deployment_runtime/demo_replay.py`
- `deployment_runtime/edge_runtime.py`
- `deployment_runtime/gateway_runtime.py`
- `deployment_runtime/control_center_runtime.py`
- `deployment_runtime/alert_packet_schema.json`

## Files To Ignore Or Treat As Secondary

- `deployment_runtime/examples/*`: illustrative only
- `deployment_runtime/docs/*`: cite as supporting documentation, not as proof of implementation
- `deployment_runtime/__pycache__/*`: generated cache
- `phase3_explanations/examples/*`: examples only

## Conservative Canonical Story

1. Split scenarios for generalization testing with `phase3/zero_day_splitter.py`.
2. Run zero-day evaluation with `phase3/run_zero_day_evaluation.py`.
3. Run focused ablations/sweeps with the Phase 3 drivers listed above.
4. Build grounded explanations through `phase3_explanations/build_explanation_packet.py` and `generate_explanations_llm.py`.
5. Assemble final study artifacts with `phase3/build_final_artifacts.py`.
6. Demonstrate deployment feasibility with `deployment_runtime/demo_replay.py`.
