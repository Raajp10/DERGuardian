# Final Publication-Safe Claims

## safe to claim now

- A canonical 60s transformer package was selected from the completed Phase 1 window-size study and reused without retraining for heldout evaluation.
- The repository contains a canonical Phase 2 attacked bundle plus multiple heldout cross-generator bundles with explicit validation filtering.
- The saved detector generalized to multiple unseen heldout generator bundles under the same defensive schema and physical constraints.
- The explanation layer can be described as an evidence-grounded post-alert operator-facing aid.

## safe with minor wording changes

- `Zero-day-like cross-generator evaluation` is acceptable when explicitly defined as frozen-model replay on unseen LLM-generated heldout bundles under shared safety constraints.
- `Generalizes across generators` is acceptable if paired with the exact bundle-level metrics and acceptance/rejection counts.
- `Grounded explanation layer` is acceptable if the paper cites the existing heldout XAI grounding summaries and avoids human-level language.

## not safe to claim

- True real-world zero-day robustness.
- Human-like root-cause analysis.
- Complete coverage of all plausible DER cyber-physical attack space.
- Cross-generator robustness independent of the shared scenario schema.

## missing evidence still required

- External non-schema-constrained attack bundles or human-authored scenarios.
- More than one additional independent generator family beyond the current local set if a stronger generalization claim is needed.
- Real deployment latency measurements from the edge/runtime package rather than offline parquet replay only.
