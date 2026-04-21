# Phase 2 Diversity And Quality Report

This quality package is intentionally conservative: final usable coverage is limited to accepted scenarios, while rejected and repaired rows remain visible for auditability.

## Plausibility

```text
         source_bundle  rows  accepted  replay_evaluated  acceptance_rate
Human-authored heldout     6         6                 6              1.0
      Canonical bundle    13        13                13              1.0
       Chatgpt heldout    10         8                 8              0.8
        Claude heldout    10         7                 7              0.7
        Gemini heldout    10         5                 5              0.5
          Grok heldout    10         9                 9              0.9
      Chatgpt repaired    10        10                10              1.0
       Claude repaired    10        10                10              1.0
       Gemini repaired    10        10                10              1.0
         Grok repaired    10        10                10              1.0
```

- Overall acceptance rate: 0.889
- Validated rows with replay evidence: 1.000

## Diversity

- Validated families: 8
- Validated assets: 18
- Validated signals: 29
- Validated source bundles: 10

## Metadata completeness

- Mean completeness over all inventory rows: 0.991
- Mean completeness over validated rows: 1.000
- Required inventory fields audited: ['dataset_id', 'source_bundle', 'generator_source', 'scenario_id', 'attack_family', 'affected_assets', 'affected_signals', 'target_component', 'severity', 'accepted_rejected', 'repair_applied', 'replay_evaluated']

## Balance summary

- Family balance score (normalized entropy): 0.873
- Asset balance score (normalized entropy): 0.864
- Signal balance score (normalized entropy): 0.918

Higher balance scores indicate more even spread across observed categories; they do not imply exhaustive coverage.

## Interpretation

- Plausibility is strongest for the canonical bundle, the accepted heldout bundles, and the repaired bundles that passed the same validator after documented edits.
- Diversity improves materially once the repaired bundles and the human-authored heldout source are included, but some families, assets, and signals remain thinly represented.
- Metadata completeness is high for accepted Phase 2 bundle rows and lower for raw rejected rows where canonical severity labels were never generated.
