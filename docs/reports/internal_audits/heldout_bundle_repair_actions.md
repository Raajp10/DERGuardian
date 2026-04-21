# Heldout Bundle Repair Actions

This document records only validator-guided repairs to previously rejected heldout scenarios. Raw heldout JSON files remain unchanged.

- Balanced selection note: All repaired heldout generators converged to the same accepted scenario count (10), so the balanced evaluation uses the repaired full bundles without extra subsampling.

## chatgpt

- `scn_cmd_delay_creg4b_vreg_ramp`
  original rejection reason: scn_cmd_delay_creg4b_vreg_ramp: magnitude.value=2.5 is outside bounds [110.0, 130.0] for creg4b:vreg.; scn_cmd_delay_creg4b_vreg_ramp: canonical Phase-2 validation failed: Scenario scn_cmd_delay_creg4b_vreg_ramp magnitude 2.5 for creg4b:vreg is outside bounds [110.0, 130.0].
  repaired: True
  exact repair made: magnitude.value: 2.5 -> 122.5
  accepted after repair: True
- `scn_der_disconnect_bess108_standby_ramp`
  original rejection reason: scn_der_disconnect_bess108_standby_ramp: bess108:mode_cmd requires magnitude.text from the shared allowed_text list.; scn_der_disconnect_bess108_standby_ramp: canonical Phase-2 validation failed: Scenario scn_der_disconnect_bess108_standby_ramp uses invalid text value None for bess108:mode_cmd.
  repaired: True
  exact repair made: magnitude.text: null -> "standby"
  accepted after repair: True

## claude

- `scn_cmd_delay_creg4b_vreg_step`
  original rejection reason: scn_cmd_delay_creg4b_vreg_step: magnitude.value=3.0 is outside bounds [110.0, 130.0] for creg4b:vreg.; scn_cmd_delay_creg4b_vreg_step: canonical Phase-2 validation failed: Scenario scn_cmd_delay_creg4b_vreg_step magnitude 3.0 for creg4b:vreg is outside bounds [110.0, 130.0].
  repaired: True
  exact repair made: magnitude.value: 3.0 -> 117.0
  accepted after repair: True
- `scn_der_disconnect_bess108_mode_standby`
  original rejection reason: scn_der_disconnect_bess108_mode_standby: bess108:mode_cmd requires magnitude.text from the shared allowed_text list.; scn_der_disconnect_bess108_mode_standby: canonical Phase-2 validation failed: Scenario scn_der_disconnect_bess108_mode_standby uses invalid text value None for bess108:mode_cmd.
  repaired: True
  exact repair made: magnitude.text: null -> "standby"
  accepted after repair: True
- `scn_coordinated_pv83_bess48_evening_ramp_disruption`
  original rejection reason: scn_coordinated_pv83_bess48_evening_ramp_disruption.additional_targets[0]: injection_type=command_suppression is not allowed for family=coordinated_multi_asset.
  repaired: True
  exact repair made: additional_targets[0].injection_type: "command_suppression" -> "command_override"
  accepted after repair: True

## gemini

- `scn_02_delay_creg4a`
  original rejection reason: scn_02_delay_creg4a: safety_notes must explicitly state synthetic and simulation-only use.; scn_02_delay_creg4a: magnitude.value=60.0 is outside bounds [110.0, 130.0] for creg4a:vreg.; scn_02_delay_creg4a: canonical Phase-2 validation failed: Scenario scn_02_delay_creg4a magnitude 60.0 for creg4a:vreg is outside bounds [110.0, 130.0].
  repaired: True
  exact repair made: magnitude.value: 60.0 -> 124.0; magnitude.unit: "seconds" -> "volts"; temporal_pattern.ramp_seconds: 0 -> 60; safety_notes: "Synthetic scenario demonstrating communication delays. Intended solely for anomaly detection evaluation." -> "This scenario is synthetic and simulation-only for defensive DER anomaly-detection research."
  accepted after repair: True
- `scn_06_coord_curtail_and_charge`
  original rejection reason: scn_06_coord_curtail_and_charge: safety_notes must explicitly state synthetic and simulation-only use.
  repaired: True
  exact repair made: safety_notes: "Synthetic multi-asset bounding scenario containing no actual exploit workflows." -> "This scenario is synthetic and simulation-only for defensive DER anomaly-detection research."
  accepted after repair: True
- `scn_07_fdi_bus48_dropout`
  original rejection reason: scn_07_fdi_bus48_dropout: safety_notes must explicitly state synthetic and simulation-only use.
  repaired: True
  exact repair made: safety_notes: "Synthetic communication loss representation for detector testing." -> "This scenario is synthetic and simulation-only for defensive DER anomaly-detection research."
  accepted after repair: True
- `scn_08_oscillation_sw4`
  original rejection reason: scn_08_oscillation_sw4: safety_notes must explicitly state synthetic and simulation-only use.
  repaired: True
  exact repair made: safety_notes: "Synthetic control disruption bounded to OpenDSS rules. Not indicative of real-world exploitation paths." -> "This scenario is synthetic and simulation-only for defensive DER anomaly-detection research."
  accepted after repair: True
- `scn_10_fdi_pv35_voltage_scale`
  original rejection reason: scn_10_fdi_pv35_voltage_scale: safety_notes must explicitly state synthetic and simulation-only use.
  repaired: True
  exact repair made: safety_notes: "This is a purely synthetic data manipulation sequence designed to evaluate monitoring systems." -> "This scenario is synthetic and simulation-only for defensive DER anomaly-detection research."
  accepted after repair: True

## grok

- `scn_command_delay_creg3c_vreg`
  original rejection reason: scn_command_delay_creg3c_vreg: magnitude.value=3.0 is outside bounds [110.0, 130.0] for creg3c:vreg.; scn_command_delay_creg3c_vreg: canonical Phase-2 validation failed: Scenario scn_command_delay_creg3c_vreg magnitude 3.0 for creg3c:vreg is outside bounds [110.0, 130.0].
  repaired: True
  exact repair made: magnitude.value: 3.0 -> 117.0
  accepted after repair: True
