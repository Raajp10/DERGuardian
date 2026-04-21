# Heldout Bundle Validation Report

This report covers every discovered JSON file under `phase2_llm_benchmark/heldout_llm_response/` and records which file was selected, which duplicate file was ignored, and which scenarios were rejected by canonical validation before Phase 3 evaluation.

## Inventory Snapshot

| generator_source | json_file | dataset_id | scenario_count | validation_status | compile_status | run_status |
| --- | --- | --- | --- | --- | --- | --- |
| chatgpt | phase2_llm_benchmark\heldout_llm_response\chatgpt\new_respnse.json | phase2_zero_day_like_bundle_2026_04_15_c | 10 | validated | reused_existing_phase2_artifacts | completed_saved_transformer_replay |
| chatgpt | phase2_llm_benchmark\heldout_llm_response\chatgpt\response.json | phase2_zero_day_like_bundle_2026_04_15_c | 10 | duplicate_of:new_respnse.json | not_recompiled_duplicate_input | not_run_duplicate_input |
| claude | phase2_llm_benchmark\heldout_llm_response\claude\new_respnse.json | claude_phase2_llm_benchmark_v2 | 10 | validated | reused_existing_phase2_artifacts | completed_saved_transformer_replay |
| claude | phase2_llm_benchmark\heldout_llm_response\claude\response.json | claude_phase2_llm_benchmark_v2 | 10 | duplicate_of:new_respnse.json | not_recompiled_duplicate_input | not_run_duplicate_input |
| gemini | phase2_llm_benchmark\heldout_llm_response\gemini\new_respnse.json | phase2_llm_synthetic_benchmark_002 | 10 | validated | reused_existing_phase2_artifacts | completed_saved_transformer_replay |
| gemini | phase2_llm_benchmark\heldout_llm_response\gemini\response.json | phase2_llm_synthetic_benchmark_002 | 10 | duplicate_of:new_respnse.json | not_recompiled_duplicate_input | not_run_duplicate_input |
| grok | phase2_llm_benchmark\heldout_llm_response\grok\new_respnse.json | phase2_llm_benchmark_grok_synthetic_03 | 10 | validated | reused_existing_phase2_artifacts | completed_saved_transformer_replay |
| grok | phase2_llm_benchmark\heldout_llm_response\grok\response.json | phase2_llm_benchmark_grok_synthetic_03 | 10 | duplicate_of:new_respnse.json | not_recompiled_duplicate_input | not_run_duplicate_input |

## Accepted vs Rejected Counts

| generator_source | dataset_id | submitted | accepted | rejected |
| --- | --- | --- | --- | --- |
| chatgpt | phase2_zero_day_like_bundle_2026_04_15_c | 10 | 8 | 2 |
| claude | claude_phase2_llm_benchmark_v2 | 10 | 7 | 3 |
| gemini | phase2_llm_synthetic_benchmark_002 | 10 | 5 | 5 |
| grok | phase2_llm_benchmark_grok_synthetic_03 | 10 | 9 | 1 |

## chatgpt

- selected bundle: `phase2_llm_benchmark\heldout_llm_response\chatgpt\new_respnse.json`
- duplicate files: `phase2_llm_benchmark\heldout_llm_response\chatgpt\response.json`
- dataset_id: `phase2_zero_day_like_bundle_2026_04_15_c`
- submitted scenarios: 10
- valid scenarios: 8
- rejected scenarios: 2
- validation summary: `phase2_llm_benchmark\new_respnse_result\models\chatgpt\reports\validation_summary.json`
- compilation report: `phase2_llm_benchmark\new_respnse_result\models\chatgpt\compiled_injections\compilation_report.json`

Rejected scenario reasons:
- `scn_cmd_delay_creg4b_vreg_ramp`
  - scn_cmd_delay_creg4b_vreg_ramp: magnitude.value=2.5 is outside bounds [110.0, 130.0] for creg4b:vreg.
  - scn_cmd_delay_creg4b_vreg_ramp: canonical Phase-2 validation failed: Scenario scn_cmd_delay_creg4b_vreg_ramp magnitude 2.5 for creg4b:vreg is outside bounds [110.0, 130.0].
- `scn_der_disconnect_bess108_standby_ramp`
  - scn_der_disconnect_bess108_standby_ramp: bess108:mode_cmd requires magnitude.text from the shared allowed_text list.
  - scn_der_disconnect_bess108_standby_ramp: canonical Phase-2 validation failed: Scenario scn_der_disconnect_bess108_standby_ramp uses invalid text value None for bess108:mode_cmd.

## claude

- selected bundle: `phase2_llm_benchmark\heldout_llm_response\claude\new_respnse.json`
- duplicate files: `phase2_llm_benchmark\heldout_llm_response\claude\response.json`
- dataset_id: `claude_phase2_llm_benchmark_v2`
- submitted scenarios: 10
- valid scenarios: 7
- rejected scenarios: 3
- validation summary: `phase2_llm_benchmark\new_respnse_result\models\claude\reports\validation_summary.json`
- compilation report: `phase2_llm_benchmark\new_respnse_result\models\claude\compiled_injections\compilation_report.json`

Rejected scenario reasons:
- `scn_cmd_delay_creg4b_vreg_step`
  - scn_cmd_delay_creg4b_vreg_step: magnitude.value=3.0 is outside bounds [110.0, 130.0] for creg4b:vreg.
  - scn_cmd_delay_creg4b_vreg_step: canonical Phase-2 validation failed: Scenario scn_cmd_delay_creg4b_vreg_step magnitude 3.0 for creg4b:vreg is outside bounds [110.0, 130.0].
- `scn_der_disconnect_bess108_mode_standby`
  - scn_der_disconnect_bess108_mode_standby: bess108:mode_cmd requires magnitude.text from the shared allowed_text list.
  - scn_der_disconnect_bess108_mode_standby: canonical Phase-2 validation failed: Scenario scn_der_disconnect_bess108_mode_standby uses invalid text value None for bess108:mode_cmd.
- `scn_coordinated_pv83_bess48_evening_ramp_disruption`
  - scn_coordinated_pv83_bess48_evening_ramp_disruption.additional_targets[0]: injection_type=command_suppression is not allowed for family=coordinated_multi_asset.

## gemini

- selected bundle: `phase2_llm_benchmark\heldout_llm_response\gemini\new_respnse.json`
- duplicate files: `phase2_llm_benchmark\heldout_llm_response\gemini\response.json`
- dataset_id: `phase2_llm_synthetic_benchmark_002`
- submitted scenarios: 10
- valid scenarios: 5
- rejected scenarios: 5
- validation summary: `phase2_llm_benchmark\new_respnse_result\models\gemini\reports\validation_summary.json`
- compilation report: `phase2_llm_benchmark\new_respnse_result\models\gemini\compiled_injections\compilation_report.json`

Rejected scenario reasons:
- `scn_02_delay_creg4a`
  - scn_02_delay_creg4a: safety_notes must explicitly state synthetic and simulation-only use.
  - scn_02_delay_creg4a: magnitude.value=60.0 is outside bounds [110.0, 130.0] for creg4a:vreg.
  - scn_02_delay_creg4a: canonical Phase-2 validation failed: Scenario scn_02_delay_creg4a magnitude 60.0 for creg4a:vreg is outside bounds [110.0, 130.0].
- `scn_06_coord_curtail_and_charge`
  - scn_06_coord_curtail_and_charge: safety_notes must explicitly state synthetic and simulation-only use.
- `scn_07_fdi_bus48_dropout`
  - scn_07_fdi_bus48_dropout: safety_notes must explicitly state synthetic and simulation-only use.
- `scn_08_oscillation_sw4`
  - scn_08_oscillation_sw4: safety_notes must explicitly state synthetic and simulation-only use.
- `scn_10_fdi_pv35_voltage_scale`
  - scn_10_fdi_pv35_voltage_scale: safety_notes must explicitly state synthetic and simulation-only use.

## grok

- selected bundle: `phase2_llm_benchmark\heldout_llm_response\grok\new_respnse.json`
- duplicate files: `phase2_llm_benchmark\heldout_llm_response\grok\response.json`
- dataset_id: `phase2_llm_benchmark_grok_synthetic_03`
- submitted scenarios: 10
- valid scenarios: 9
- rejected scenarios: 1
- validation summary: `phase2_llm_benchmark\new_respnse_result\models\grok\reports\validation_summary.json`
- compilation report: `phase2_llm_benchmark\new_respnse_result\models\grok\compiled_injections\compilation_report.json`

Rejected scenario reasons:
- `scn_command_delay_creg3c_vreg`
  - scn_command_delay_creg3c_vreg: magnitude.value=3.0 is outside bounds [110.0, 130.0] for creg3c:vreg.
  - scn_command_delay_creg3c_vreg: canonical Phase-2 validation failed: Scenario scn_command_delay_creg3c_vreg magnitude 3.0 for creg3c:vreg is outside bounds [110.0, 130.0].
