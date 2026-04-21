# Phase 2 Coverage Summary

This report uses the master scenario inventory as the single accounting layer for canonical, heldout, repaired, and additional heldout scenarios.

## Final usable coverage

- Validated scenarios: 88
- Rejected scenarios: 11
- Repair-tagged rows: 11
- Replay-evaluated validated scenarios: 88
- Distinct validated families: 8
- Distinct validated assets: 18
- Distinct validated signals: 29

Only accepted/validated scenarios are counted as final usable coverage.

## Source bundle completeness

```text
         source_bundle  total_rows  accepted_rows  rejected_rows  repair_rows  replay_evaluated
Human-authored heldout           6              6              0            0                 6
      Canonical bundle          13             13              0            0                13
       Chatgpt heldout          10              8              2            0                 8
        Claude heldout          10              7              3            0                 7
        Gemini heldout          10              5              5            0                 5
          Grok heldout          10              9              1            0                 9
      Chatgpt repaired          10             10              0            2                10
       Claude repaired          10             10              0            3                10
       Gemini repaired          10             10              0            5                10
         Grok repaired          10             10              0            1                10
```

## Attack-family coverage

```text
       attack_family  all_scenarios  validated_scenarios  rejected_scenarios  repair_rows  generator_coverage_validated  source_bundle_coverage_validated  replay_evaluated_validated
false_data_injection             27                   25                   2            2                             6                                10                          25
unauthorized_command             17                   16                   1            1                             6                                10                          16
       command_delay             17                   13                   4            4                             6                                 9                          13
         degradation             15                   13                   2            2                             6                                 9                          13
 command_suppression             10                   10                   0            0                             6                                10                          10
coordinated_campaign             11                    9                   2            2                             6                                 8                           9
              replay              1                    1                   0            0                             1                                 1                           1
telemetry_corruption              1                    1                   0            0                             1                                 1                           1
```

## Generator/source bundle coverage

```text
                    source_bundle    source_bundle_label generator_source  validated_scenarios  unique_families  unique_assets  unique_signals  repair_rows
                 canonical_bundle       Canonical bundle canonical_bundle                   13                8             10              13            0
         heldout_original_chatgpt        Chatgpt heldout          chatgpt                    8                6              7               7            0
         heldout_repaired_chatgpt       Chatgpt repaired          chatgpt                   10                6              9               9            2
          heldout_original_claude         Claude heldout           claude                    7                4              7               7            0
          heldout_repaired_claude        Claude repaired           claude                   10                6              8              11            3
          heldout_original_gemini         Gemini heldout           gemini                    5                4              5               5            0
          heldout_repaired_gemini        Gemini repaired           gemini                   10                6             10              11            5
            heldout_original_grok           Grok heldout             grok                    9                6              7               8            0
            heldout_repaired_grok          Grok repaired             grok                   10                6              8               9            1
additional_heldout_human_authored Human-authored heldout   human_authored                    6                6              6               6            0
```

## Under-covered categories

- Families with one or fewer validated scenarios: ['replay', 'telemetry_corruption']
- Assets with one or fewer validated scenarios: ['bus48', 'creg3a', 'creg3c', 'sw4']
- Signals with one or fewer validated scenarios: ['bess_bess48_terminal_i_a', 'bus_48_v_pu_phase_a', 'feeder_q_kvar_total', 'pv_pv35_q_kvar', 'pv_pv35_terminal_v_pu', 'pv_pv60_terminal_v_pu', 'pv_pv83_q_kvar', 'pv_pv83_status_cmd', 'regulator_creg3a_vreg', 'regulator_creg3c_vreg', 'switch_sw4_state', 'bess_bess108_mode', 'bess_bess108_p_kw', 'bess_bess108_q_kvar', 'bess_bess108_soc', 'bess_bess48_p_kw', 'bus_108_v_pu_phase_a', 'bus_108_v_pu_phase_b', 'bus_108_v_pu_phase_c', 'bus_150_v_pu_phase_a', 'bus_150_v_pu_phase_b', 'bus_35_v_pu_phase_a', 'bus_48_v_pu_phase_b', 'bus_48_v_pu_phase_c', 'curtailment_frac']

These are repository-relative under-coverage findings based on observed validated scenario counts only. No unobserved categories were fabricated.

## Rejected and repair appendix

- Original heldout rejections retained in inventory: 11
- Repaired rows accepted after validator-guided edits: 11
- Metadata-only difficulty rows (not replay-evaluated): 11

Rejected scenarios remain visible for auditability but are excluded from final usable coverage claims.
