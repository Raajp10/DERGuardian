# Final Experiment Notes

- Canonical source data were treated as read-only.
- All rebuilt windows, residual datasets, model runs, figures, and comparison files were written under `outputs/window_size_study/`.
- The stock Phase 1 threshold baseline stores a placeholder inference-time value; this study recomputed and corrected that value after each threshold run.
- Cross-window best-model selection used highest test F1, with recall, precision, and average precision as tie-breaks.
- If any model failed, that failure is recorded directly in the per-window `model_summary.csv` and the stack trace is saved in the corresponding window report folder.
