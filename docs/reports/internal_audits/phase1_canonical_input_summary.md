# Phase 1 Canonical Input Summary

## Canonical winner path

- Winner: `transformer` at `60s` from `outputs/window_size_study/final_window_comparison.csv`.
- Residual training input artifact: `outputs/window_size_study/60s/residual_windows.parquet`.
- Ready package: `outputs/window_size_study/60s/ready_packages/transformer`.
- Model report directory: `outputs/window_size_study/60s/reports/transformer`.

## Input contract

- Residual windows rows: `7195`
- Total residual columns: `932`
- Residual feature columns available before selection: `920`
- Selected feature columns used by the canonical transformer: `64`
- Sequence length: `12`
- Window / step: `60 s / 12 s`

## Benchmarked model set in the canonical study

```text
window_label         model_name  rows
         10s        autoencoder     1
         10s                gru     1
         10s   isolation_forest     1
         10s       llm_baseline     1
         10s               lstm     1
         10s threshold_baseline     1
         10s        transformer     1
        300s        autoencoder     1
        300s                gru     1
        300s   isolation_forest     1
        300s       llm_baseline     1
        300s               lstm     1
        300s threshold_baseline     1
        300s        transformer     1
          5s        autoencoder     1
          5s                gru     1
          5s   isolation_forest     1
          5s       llm_baseline     1
          5s               lstm     1
          5s threshold_baseline     1
          5s        transformer     1
         60s        autoencoder     1
         60s                gru     1
         60s   isolation_forest     1
         60s       llm_baseline     1
         60s               lstm     1
         60s threshold_baseline     1
         60s        transformer     1
```

## Outside the canonical benchmark roster

- `ARIMA`: no training script, result table, or ready package found.
- `TTM`: implemented separately as an extension benchmark; it is not part of the frozen canonical window-size benchmark roster.
