# LoRA Improvement Pass

This pass re-evaluates the real LoRA prediction files with explicit family-label normalization and asset-overlap scoring.
It does not turn LoRA into detector evidence; LoRA remains an experimental explanation-side branch.

| model_variant   | split                        |   example_count |   family_accuracy_normalized |   asset_any_overlap_rate |   asset_overlap_jaccard_mean |   confidence_accuracy_normalized |   mean_latency_ms |   median_latency_ms | improvement_pass_action                                                       |
|:----------------|:-----------------------------|----------------:|-----------------------------:|-------------------------:|-----------------------------:|---------------------------------:|------------------:|--------------------:|:------------------------------------------------------------------------------|
| base_zero_shot  | aux_in_domain_holdout        |              16 |                     0        |                        0 |                            0 |                         0        |           506.155 |             486.144 | post-hoc label normalization and asset-overlap audit of real prediction files |
| base_zero_shot  | test_generator_heldout       |              47 |                     0        |                        0 |                            0 |                         0        |           592.288 |             585.559 | post-hoc label normalization and asset-overlap audit of real prediction files |
| base_zero_shot  | validation_generator_heldout |              29 |                     0        |                        0 |                            0 |                         0        |           557.507 |             583.68  | post-hoc label normalization and asset-overlap audit of real prediction files |
| lora_finetuned  | aux_in_domain_holdout        |              16 |                     0.5625   |                        0 |                            0 |                         0.25     |          3401.77  |            3430.61  | post-hoc label normalization and asset-overlap audit of real prediction files |
| lora_finetuned  | test_generator_heldout       |              47 |                     0.234043 |                        0 |                            0 |                         0.425532 |          3507.03  |            3551.45  | post-hoc label normalization and asset-overlap audit of real prediction files |
| lora_finetuned  | validation_generator_heldout |              29 |                     0        |                        0 |                            0 |                         0.172414 |          3507.08  |            3538.3   | post-hoc label normalization and asset-overlap audit of real prediction files |

## Interpretation

Family labels improve only where misspellings or close aliases are normalized. Asset attribution and grounding remain weak, so the branch remains experimental.
Updated CSV: `artifacts/extensions/phase3_lora_updated_results.csv`
