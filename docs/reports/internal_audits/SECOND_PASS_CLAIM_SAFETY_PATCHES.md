# Second Pass Claim Safety Patches

## Applied patches

- `scripts/build_phase1_lineage_audit.py`: Updated the generator so TTM is described as an implemented extension benchmark rather than as unimplemented.
- `PHASE1_DATA_LINEAGE_AUDIT.md`: Regenerated to remove the stale pre-extension claim that TTM was not implemented.
- `phase1_canonical_input_summary.md`: Regenerated to mark TTM as outside the frozen canonical benchmark roster rather than missing.

## Final markdown reports re-opened

- `PHASE1_DATA_LINEAGE_AUDIT.md`
- `phase1_canonical_input_summary.md`
- `phase1_ttm_eval_report.md`
- `phase2_coverage_summary.md`
- `phase2_diversity_and_quality_report.md`
- `phase3_lora_eval_report.md`
- `xai_final_validation_report.md`
- `deployment_benchmark_report.md`
- `FINAL_DIAGRAM_ALIGNMENT.md`
- `DIAGRAM_SAFE_LABELS.md`
- `COMPLETE_PROJECT_STATUS.md`
- `COMPLETE_RESULTS_AND_DISCUSSION.md`
- `COMPLETE_PUBLICATION_SAFE_CLAIMS.md`
- `FINAL_PROJECT_DECISION.md`
- `FINAL_TRIPLE_VERIFICATION_CHECKLIST.md`
- `VERIFICATION_GAPS.md`

## Claim-safety findings

- TTM remains explicitly labeled as an extension benchmark; no final report promotes it to the canonical winner.
- Transformer remains the frozen canonical benchmark winner in `outputs/window_size_study/final_window_comparison.csv` and in the final narrative reports.
- AOI appears only in cautionary or diagram-audit contexts; it is not claimed as an implemented standalone metric.
- Deployment wording remains restricted to an offline workstation benchmark rather than edge-hardware deployment.

- Reviewed cautionary phrase hits: `36`
- Suspicious phrase hits requiring further patching: `0`

## Remaining suspicious hits

- None after the second-pass wording review.
