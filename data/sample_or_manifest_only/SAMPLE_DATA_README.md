# Sample Data README

`small_sample_windows.parquet` is a tiny schema-inspection sample extracted from the existing 60s residual-window artifact.

It is **not** a benchmark dataset and must not be used to report model performance. It exists only so a GitHub visitor can inspect the shape of window-level metadata and residual/deviation features without downloading the full generated `outputs/` tree.

Sample file:

- `data/sample_or_manifest_only/small_sample_windows.parquet`

Canonical source of truth remains local/generated:

- `outputs/window_size_study/final_window_comparison.csv`
- `outputs/window_size_study/60s/residual_windows.parquet`
- `outputs/window_size_study/60s/ready_packages/transformer`

To regenerate full data, use:

```bash
python scripts/run_full_pipeline.py
```

The default run skips completed heavy stages when their completion markers already exist.
