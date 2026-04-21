# Phase 3 LoRA Evidence Checklist

## Artifact existence

- `phase3_lora_dataset_manifest.json`: exists=`True`, nonempty=`True`
- `phase3_lora_training_config.yaml`: exists=`True`, nonempty=`True`
- `phase3_lora_results.csv`: exists=`True`, nonempty=`True`
- `phase3_lora_eval_report.md`: exists=`True`, nonempty=`True`
- `phase3_lora_model_card.md`: exists=`True`, nonempty=`True`
- `phase3_lora_run_log.txt`: exists=`True`, nonempty=`True`
- `phase3_lora_family_accuracy.png`: exists=`True`, nonempty=`True`
- `phase3_lora_asset_accuracy.png`: exists=`True`, nonempty=`True`
- `phase3_lora_grounding_comparison.png`: exists=`True`, nonempty=`True`
- `phase3_lora_latency_vs_quality.png`: exists=`True`, nonempty=`True`

## Training evidence

- Adapter directory exists: `True`
- Tokenizer directory exists: `True`
- `training_info.json` exists: `True`
- Training history rows: `5`
- Trainable parameters: `344064`
- Total parameters: `77305216`

## Split evidence

- Train examples: `63`
- Aux in-domain holdout: `16`
- Validation generator holdout: `29`
- Test generator holdout: `47`

## Metric evidence

- Result rows: `6`
- Base model rows: `3`
- LoRA rows: `3`

## Conservative interpretation

- This branch is explanation-side and experimental.
- Weak asset grounding or heldout family performance must be reported directly rather than hidden.
- These outputs do not replace the canonical transformer detector benchmark.
