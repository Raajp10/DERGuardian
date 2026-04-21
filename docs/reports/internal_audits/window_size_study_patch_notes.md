# Window Size Study Patch Notes

Date: 2026-04-18

## Patched file

- `scripts/run_window_size_study.py`

## Why it was patched

The original study driver assumed the source files were valid but did not prove that at runtime. That left room for future accidental misuse, especially if someone pointed the script at already-windowed parquet files or coarse aggregated sources.

## Changes made

1. Added explicit canonical source definitions.
   - clean measured
   - clean cyber
   - attacked measured
   - attacked cyber
   - attack labels

2. Added pre-run source validation.
   - audits the physical source parquet files before model training starts
   - detects whether a source is already windowed or aggregated
   - infers the physical sampling interval from reconstructed nominal measured timestamps

3. Added strict failure behavior.
   - if a measured source is already windowed, the script exits with a clear error
   - if the inferred physical sampling interval is coarser than the smallest requested window, the script exits with a clear error
   - if the interval cannot be inferred, the script exits instead of guessing

4. Added runtime reporting.
   - prints detected source path
   - prints detected physical nominal interval
   - prints observed measured timestamp interval
   - prints whether the source is windowed
   - prints cyber source timing summary

5. Added source audit to the run manifest.
   - `outputs/window_size_study/run_manifest.json` now records the detected source audit and validation rule used for the run

## Result

The driver now refuses to run a 5s / 10s / 60s study on a coarse or already-windowed source.

## What was not changed

- The canonical window-building logic in `phase1/build_windows.py`
- The residual dataset logic
- The model suite logic
- The per-window output structure

No training was resumed in this patch step. The source audit was completed first, per the experiment safety requirement.
