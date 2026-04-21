# Phase 3 Zero-Day Report

## objective

Evaluate the saved canonical Phase 1 winner on unseen heldout attacked bundles generated from different LLM sources without retraining the detector.

## which Phase 1 winner was reused

- saved package: `outputs/window_size_study/best_model_package/ready_package`
- model: `transformer`
- window setup: `60s` windows, `12s` step, `0.2` minimum attack overlap fraction
- detector reuse policy: fixed preprocessing, fixed feature set, fixed calibration, fixed threshold

## heldout bundle inventory

| generator_source | dataset_id | scenario_count | accepted_scenario_count | f1 | precision | recall | mean_latency_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_bundle | ieee123_der_research_attacks_13scenario | 13 | 13 | 0.7443 | 0.8202 | 0.6812 | 4.0000 |
| chatgpt | phase2_zero_day_like_bundle_2026_04_15_c | 10 | 8 | 0.8615 | 0.8924 | 0.8327 | 1.0000 |
| claude | claude_phase2_llm_benchmark_v2 | 10 | 7 | 0.4866 | 0.9286 | 0.3297 | 1.0000 |
| gemini | phase2_llm_synthetic_benchmark_002 | 10 | 5 | 0.5035 | 0.8710 | 0.3541 | 1.0000 |
| grok | phase2_llm_benchmark_grok_synthetic_03 | 10 | 9 | 0.6356 | 0.8533 | 0.5065 | 2.5000 |

## validation summary

- each heldout generator folder contained two identical JSON files: `response.json` and `new_respnse.json`
- the run treated `new_respnse.json` as the selected effective bundle when present and marked `response.json` as a duplicate reference
- Phase 2 validation and compilation were reused only from `phase2_llm_benchmark/new_respnse_result/models/<generator>/...` artifacts that pointed back to the heldout JSON source path
- malformed or rejected scenarios were retained in the validation report and excluded from bundle evaluation only after canonical validation rejected them

## per-generator results

| generator_source | precision | recall | f1 | average_precision | roc_auc | mean_latency_seconds | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_bundle | 0.8202 | 0.6812 | 0.7443 | 0.7243 | 0.8514 | 4.0000 | In-domain canonical attacked bundle reused from existing repository outputs. Metrics below are bundle-level replay results from the saved 60s transformer package, not the Phase 1 test split. Replay also required numerical sanitization for 18 selected features across 160 rows. |
| chatgpt | 0.8924 | 0.8327 | 0.8615 | 0.8309 | 0.9473 | 1.0000 | Reused existing Phase 2 heldout artifacts; accepted 8/10, rejected 2. Non-finite residual values affected 1 selected features across 98 rows; replay used a sanitized copy with non-finite values replaced by 0.0 and extreme values clipped for numerical stability. |
| claude | 0.9286 | 0.3297 | 0.4866 | 0.4510 | 0.8030 | 1.0000 | Reused existing Phase 2 heldout artifacts; accepted 7/10, rejected 3. Non-finite residual values affected 6 selected features across 346 rows; replay used a sanitized copy with non-finite values replaced by 0.0 and extreme values clipped for numerical stability. |
| gemini | 0.8710 | 0.3541 | 0.5035 | 0.4368 | 0.7274 | 1.0000 | Reused existing Phase 2 heldout artifacts; accepted 5/10, rejected 5. Non-finite residual values affected 9 selected features across 55 rows; replay used a sanitized copy with non-finite values replaced by 0.0 and extreme values clipped for numerical stability. |
| grok | 0.8533 | 0.5065 | 0.6356 | 0.6161 | 0.7514 | 2.5000 | Reused existing Phase 2 heldout artifacts; accepted 9/10, rejected 1. Non-finite residual values affected 7 selected features across 55 rows; replay used a sanitized copy with non-finite values replaced by 0.0 and extreme values clipped for numerical stability. |

## per-family results

| generator_source | attack_family | scenario_count | detected_count | recall_like_rate | mean_latency_seconds |
| --- | --- | --- | --- | --- | --- |
| canonical_bundle | command_delay | 2 | 2 | 1.0000 | 1.0000 |
| canonical_bundle | command_suppression | 1 | 1 | 1.0000 | 13.0000 |
| canonical_bundle | coordinated_campaign | 2 | 2 | 1.0000 | 7.0000 |
| canonical_bundle | degradation | 2 | 2 | 1.0000 | 1.0000 |
| canonical_bundle | false_data_injection | 2 | 2 | 1.0000 | 1.0000 |
| canonical_bundle | replay | 1 | 1 | 1.0000 | 1.0000 |
| canonical_bundle | telemetry_corruption | 1 | 0 | 0.0000 |  |
| canonical_bundle | unauthorized_command | 2 | 2 | 1.0000 | 7.0000 |
| chatgpt | command_delay | 1 | 1 | 1.0000 | 1.0000 |
| chatgpt | command_suppression | 1 | 1 | 1.0000 | 1.0000 |
| chatgpt | coordinated_campaign | 1 | 1 | 1.0000 | 1.0000 |
| chatgpt | degradation | 1 | 1 | 1.0000 | 1.0000 |
| chatgpt | false_data_injection | 2 | 2 | 1.0000 | 1.0000 |
| chatgpt | unauthorized_command | 2 | 2 | 1.0000 | 1.0000 |
| claude | command_delay | 1 | 1 | 1.0000 | 1.0000 |
| claude | command_suppression | 1 | 1 | 1.0000 | 1.0000 |
| claude | false_data_injection | 4 | 4 | 1.0000 | 1.0000 |
| claude | unauthorized_command | 1 | 1 | 1.0000 | 1.0000 |
| gemini | command_suppression | 1 | 1 | 1.0000 | 1.0000 |
| gemini | degradation | 2 | 1 | 0.5000 | 1.0000 |
| gemini | false_data_injection | 1 | 1 | 1.0000 | 1.0000 |
| gemini | unauthorized_command | 1 | 1 | 1.0000 | 1.0000 |
| grok | command_delay | 1 | 1 | 1.0000 | 1.0000 |
| grok | command_suppression | 1 | 1 | 1.0000 | 1.0000 |
| grok | coordinated_campaign | 1 | 1 | 1.0000 | 13.0000 |
| grok | degradation | 1 | 1 | 1.0000 | 1.0000 |
| grok | false_data_injection | 3 | 2 | 0.6667 | 1.0000 |
| grok | unauthorized_command | 2 | 2 | 1.0000 | 1.0000 |

## latency interpretation

- mean heldout detection latency ranged from 1.0s to 2.5s across valid heldout bundles.
- best heldout F1 came from `chatgpt`.
- bundle-level latency here is measured from attack start to the first positive detector decision using the saved 60s-window transformer package.

## what can be claimed safely

- the saved 60s transformer generalized beyond the canonical Phase 2 bundle onto multiple unseen heldout generator bundles without retraining
- the heldout results are generator-conditioned bundle replays, not exhaustive real-world zero-day evidence
- the explanation layer should be described as grounded post-alert operator support, not human-level root-cause analysis

## limitations

- heldout bundles were still constrained by the same defensive scenario schema and physical validation rules
- generator bundles differ in accepted scenario count because some submissions were rejected by canonical validation
- canonical bundle metrics in this report are bundle-level replay numbers and should not be confused with the Phase 1 held-out test split used to choose the winning model
- no Ollama heldout bundle was added because Ollama was not available locally during this run

## whether cross-generator evidence supports a zero-day-like claim

The evidence supports only a limited heldout cross-generator generalization claim. It does not yet support a strong zero-day-like claim without stronger consistency or broader heldout coverage.
