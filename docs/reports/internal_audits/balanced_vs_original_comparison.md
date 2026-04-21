# Balanced vs Original Heldout Comparison

## Fairness Improvement

- All repaired heldout generators converged to the same accepted scenario count (10), so the balanced evaluation uses the repaired full bundles without extra subsampling.
- Original accepted scenario counts varied across generators. Repaired/balanced evaluation counts are recorded in `heldout_balanced_inventory.csv`.

## Metric Comparison

```text
generator_source  original_f1  balanced_f1  delta_f1  original_recall  balanced_recall  original_precision  balanced_precision
         chatgpt     0.861538     0.799353 -0.062186         0.832714         0.715942            0.892430            0.904762
          claude     0.486631     0.608424  0.121793         0.329710         0.453488            0.928571            0.924171
          gemini     0.503497     0.628798  0.125301         0.354098         0.496868            0.870968            0.856115
            grok     0.635628     0.682676  0.047048         0.506452         0.580175            0.853261            0.829167
```

## Generator Ranking Shift

- Original transformer replay F1 ranking: ['chatgpt', 'grok', 'gemini', 'claude']
- Balanced repaired transformer replay F1 ranking: ['chatgpt', 'grok', 'gemini', 'claude']

## Claim Strength

- The repaired/balanced evaluation is fairer because it removes acceptance-count asymmetry and documents every repair.
- It does not convert the study into real-world zero-day evidence. The balanced set is still derived from bounded synthetic scenario bundles.
