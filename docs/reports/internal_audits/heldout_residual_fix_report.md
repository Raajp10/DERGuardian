# Heldout Residual Fix Report

## Which Features Broke

- True replay-stage NaN/Inf values were not present in the previously flagged saved-transformer heldout features.
- The old Phase 3 audit labeled some features as 'nonfinite' even when `nonfinite_count=0`; those rows were actually finite out-of-distribution values that were clipped by the old 100-sigma guardrail.
- Upstream measured time series still legitimately contain missing values because the canonical measured layer synthesizes sensor latency and missing bursts. Those upstream NaNs are real measurement impairments, not replay-only corruption.

## Why They Broke

- The original replay report conflated two different issues: true NaN/Inf values and finite-value clipping for numerical guardrails.
- A separate upstream hygiene issue did exist in the legacy window builder: raw measured-layer NaNs could propagate into `__last` or all-missing aggregate statistics before residual construction.
- In practice, the selected replay features used by the saved 60s transformer did not carry true NaN/Inf values into inference; the Phase 3 weakness was mainly the misclassification of clipped finite residuals as 'nonfinite'.

## Code Path Changed

- Patched `phase1/build_windows.py` at `<repo>/phase1\build_windows.py` to compute finite-only window statistics and use the last finite observation fallback instead of blindly preserving missing last-sample values.
- The improved replay path does not clip finite residuals by default. It only replaces true NaN/Inf values if they are actually present in the selected package feature columns.

## Whether Replay Still Required Sanitization

- Improved replay rows with fallback non-finite replacement: 0 / 18.
- Improved replay rows with fallback score-stability clipping: 1 / 18.
- Improved replay rows with true replay-stage non-finite features: 0 / 18.
- Improved replay rows with extreme but finite feature shift: 18 / 18 / 18.

## Impact On Metrics

Saved-transformer bundle replay comparison (old clipped run vs improved no-clipping replay):

```text
generator_source   old_f1   new_f1  delta_f1  old_precision  new_precision  old_recall  new_recall
canonical_bundle 0.744279 0.825571  0.081292       0.820175       0.827839    0.681239    0.823315
         chatgpt 0.861538 0.858252 -0.003286       0.892430       0.898374    0.832714    0.821561
          claude 0.486631 0.452736 -0.033895       0.928571       0.722222    0.329710    0.329710
          gemini 0.503497 0.661290  0.157794       0.870968       0.858639    0.354098    0.537705
            grok 0.635628 0.689139  0.053511       0.853261       0.821429    0.506452    0.593548
```

## Upstream Missingness Context

The canonical measured layer intentionally applies latency and missing bursts. Top affected numeric columns:

```json
{
  "clean_measured": {
    "column_count_with_missingness": 180,
    "top_columns": [
      {
        "column": "bess_bess108_status",
        "nonfinite_count": 383
      },
      {
        "column": "bus_150_v_pu_phase_b",
        "nonfinite_count": 373
      },
      {
        "column": "feeder_p_kw_phase_b",
        "nonfinite_count": 373
      },
      {
        "column": "bus_48_v_pu_phase_a",
        "nonfinite_count": 371
      },
      {
        "column": "bus_13_angle_deg_phase_a",
        "nonfinite_count": 370
      },
      {
        "column": "bess_bess48_soc",
        "nonfinite_count": 363
      },
      {
        "column": "feeder_i_angle_deg_phase_b",
        "nonfinite_count": 363
      },
      {
        "column": "bess_bess108_soc",
        "nonfinite_count": 362
      },
      {
        "column": "feeder_i_angle_deg_phase_a",
        "nonfinite_count": 362
      },
      {
        "column": "derived_pv_pv60_availability_residual_kw",
        "nonfinite_count": 360
      },
      {
        "column": "pv_pv60_available_kw",
        "nonfinite_count": 354
      },
      {
        "column": "bess_bess108_energy_kwh",
        "nonfinite_count": 352
      }
    ]
  },
  "attacked_measured": {
    "column_count_with_missingness": 180,
    "top_columns": [
      {
        "column": "feeder_p_kw_total",
        "nonfinite_count": 628
      },
      {
        "column": "feeder_v_pu_phase_b",
        "nonfinite_count": 404
      },
      {
        "column": "bess_bess48_q_kvar",
        "nonfinite_count": 396
      },
      {
        "column": "bus_48_v_pu_phase_a",
        "nonfinite_count": 392
      },
      {
        "column": "bus_94_angle_deg_phase_a",
        "nonfinite_count": 388
      },
      {
        "column": "bus_83_angle_deg_phase_b",
        "nonfinite_count": 380
      },
      {
        "column": "derived_pv_pv35_ramp_kw_per_s",
        "nonfinite_count": 379
      },
      {
        "column": "bus_94_open_angle_deg_phase_a",
        "nonfinite_count": 375
      },
      {
        "column": "bus_13_v_pu_phase_b",
        "nonfinite_count": 368
      },
      {
        "column": "bus_35_v_pu_phase_b",
        "nonfinite_count": 366
      },
      {
        "column": "pv_pv60_terminal_i_a",
        "nonfinite_count": 364
      },
      {
        "column": "feeder_i_a_total",
        "nonfinite_count": 362
      }
    ]
  }
}
```

## Residual Non-finite Audit Table

```text
generator_source                                          feature_name  nonfinite_count                             issue_type                                                                                                                                            suspected_root_cause  fixed_at_source  replay_sanitization_still_needed
canonical_bundle                  delta__pv_pv83_curtailment_frac__max                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle  delta__derived_pv_pv83_availability_residual_kw__max                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle delta__derived_pv_pv83_availability_residual_kw__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                             delta__pv_pv83_p_kw__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                              delta__pv_pv60_p_kw__min                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle  delta__derived_pv_pv60_availability_residual_kw__max                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                  delta__pv_pv60_curtailment_frac__max                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                      delta__pv_pv60_terminal_i_a__min                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle delta__derived_pv_pv83_availability_residual_kw__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                             delta__pv_pv60_p_kw__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                             delta__pv_pv83_p_kw__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle delta__derived_pv_pv60_availability_residual_kw__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                 delta__pv_pv83_curtailment_frac__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                             delta__pv_pv60_p_kw__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                 delta__pv_pv60_curtailment_frac__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                     delta__pv_pv60_terminal_i_a__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle delta__derived_pv_pv60_availability_residual_kw__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
canonical_bundle                     delta__pv_pv60_terminal_i_a__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
         chatgpt                 delta__pv_pv60_curtailment_frac__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          claude                    delta__feeder_losses_kw_total__min                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          claude                  delta__feeder_losses_kvar_total__min                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          claude                   delta__feeder_losses_kw_total__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          claude                 delta__feeder_losses_kvar_total__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          claude                   delta__feeder_losses_kw_total__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          claude                 delta__feeder_losses_kvar_total__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini                              delta__pv_pv60_p_kw__min                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini  delta__derived_pv_pv60_availability_residual_kw__max                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini                      delta__pv_pv60_terminal_i_a__min                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini                             delta__pv_pv60_p_kw__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini delta__derived_pv_pv60_availability_residual_kw__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini                             delta__pv_pv60_p_kw__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini                     delta__pv_pv60_terminal_i_a__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini delta__derived_pv_pv60_availability_residual_kw__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
          gemini                     delta__pv_pv60_terminal_i_a__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
            grok                 delta__pv_pv60_curtailment_frac__last                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
            grok  delta__derived_pv_pv60_availability_residual_kw__max                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
            grok                  delta__pv_pv60_curtailment_frac__max                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
            grok                             delta__pv_pv60_p_kw__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
            grok                 delta__pv_pv60_curtailment_frac__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
            grok delta__derived_pv_pv60_availability_residual_kw__mean                0 finite_clip_misclassified_as_nonfinite The original replay audit counted finite extreme-value clipping under a 100-standard-deviation guardrail, not NaN/Inf values in the selected residual features.            False                             False
```
