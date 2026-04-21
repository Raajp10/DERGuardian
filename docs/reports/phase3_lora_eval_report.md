# Phase 3 LoRA Experimental Branch

This branch is an extension experiment only. It does not replace the canonical benchmark-selected transformer detector.

- Base model: `google/flan-t5-small`
- Training examples: 63
- Auxiliary in-domain holdout examples: 16
- Validation generator-heldout examples: 29
- Test generator-heldout examples: 47
- Trainable parameters: 344064
- Total parameters in wrapped model: 77305216
- Adapter size on disk (MB): 1.33

## Results

```text
 model_variant                        split  example_count  family_accuracy  asset_accuracy  evidence_grounding_quality  confidence_accuracy  parse_rate  mean_latency_ms  median_latency_ms  p95_latency_ms  adapter_size_mb  model_memory_footprint_mb
base_zero_shot        aux_in_domain_holdout             16         0.000000             0.0                     0.00000             0.000000         1.0       506.154925           486.1442      726.685925              NaN                     293.58
base_zero_shot validation_generator_heldout             29         0.000000             0.0                     0.00000             0.000000         1.0       557.506648           583.6798      617.064360              NaN                     293.58
base_zero_shot       test_generator_heldout             47         0.000000             0.0                     0.00000             0.000000         1.0       592.287572           585.5590      635.421710              NaN                     293.58
lora_finetuned        aux_in_domain_holdout             16         0.562500             0.0                     0.00000             0.250000         1.0      3401.772312          3430.6121     3833.820900             1.33                     294.90
lora_finetuned validation_generator_heldout             29         0.000000             0.0                     0.08046             0.172414         1.0      3507.078428          3538.2970     3730.886540             1.33                     294.90
lora_finetuned       test_generator_heldout             47         0.234043             0.0                     0.00000             0.425532         1.0      3507.033866          3551.4457     3779.755920             1.33                     294.90
```

## Interpretation

- Family attribution and asset prediction should be discussed as bounded explanation-side tasks, not as the main anomaly detector.
- Grounding quality is measured against scenario observable-signal overlap, which is stronger than a free-form text score but still not a human-level reasoning claim.
- The branch remains experimental even if LoRA improves over the untuned base model.

## Training History

```text
 epoch  train_loss  validation_loss
     1    3.194956         2.343492
     2    2.724698         1.886374
     3    2.271530         1.466425
     4    1.847677         1.127908
     5    1.490488         0.913980
```
