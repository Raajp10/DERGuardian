# XAI Strengthening Report

This pass strengthens the validation layer by adding strict grounding, partial-credit, and operator-support scoring to the existing case-level audit.
It does not claim human-like root-cause analysis.

## Updated Overall Metrics

- Cases audited: 155
- Family exact accuracy: 0.774
- Asset accuracy mean: 0.268
- Strict grounding pass rate: 0.148
- Partial grounding pass rate: 0.755
- Strict operator-support pass rate: 0.084
- Unsupported-claim rate: 0.052

## Interpretation

The strongest supported wording is **grounded operator support** or **structured post-alert explanation**.
The audit still shows asset attribution is materially weaker than family attribution, so human-like root-cause analysis remains unsupported.

## Artifacts

- Updated case audit: `artifacts/xai/xai_updated_case_level_audit.csv`
- Updated metrics: `artifacts/xai/xai_updated_metrics.csv`
