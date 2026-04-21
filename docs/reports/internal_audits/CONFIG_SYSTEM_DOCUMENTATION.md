# Config System Documentation

The repository now has a small, explicit config surface under `configs/`.

- `configs/pipeline_config.yaml` records the three-phase orchestration, window sizes, context separation, and report locations.
- `configs/model_config.yaml` records the canonical winner, implemented model roster, blocked model labels, and threshold sources.
- `configs/data_config.yaml` records clean, attacked, window, sample-data, and Git/data-release paths.

These files are documentation-grade configs for reproducibility and orchestration. They do not overwrite the frozen canonical outputs. The canonical benchmark source remains `outputs/window_size_study/final_window_comparison.csv`, and the canonical winner remains `transformer @ 60s`.

Use these configs for new runs or for understanding the pipeline contract. Historical scripts may still carry their own CLI defaults, so treat the configs as the release-facing contract rather than proof that old generated outputs were regenerated.
