# Incident Summary: incident-threshold_baseline-20250601T131001Z

threshold_baseline raised incident incident-threshold_baseline-20250601T131001Z because the anomaly score 1.0000 exceeded the threshold 0.0219. The strongest structured evidence points to creg4a, pv83, bus83, and the current best closed-set classification is `coordinated_campaign`.

## Suspected Family

- Label: `coordinated_campaign`
- Confidence: `high` (1.0)

## Why Flagged

- The detector score exceeded threshold by 0.9781.
- Top discriminative window features were bus 83 angle deg phase a / mean, bus 83 angle deg phase a / last, bus 83 angle deg phase a / min.
- Physical evidence showed bus_83_angle_deg_phase_a moving decrease versus the clean matched window.
- Cyber evidence included attack_event activity on Precursor to scn_coordinated_pv83_reg4a_disturbance.

## Physical Evidence Used

- `bus_83_angle_deg_phase_a` (mean): observed `-1.434485`, baseline `-0.958981`, delta `-0.475504`
- `bus_83_angle_deg_phase_a` (last): observed `-1.483254`, baseline `-1.019199`, delta `-0.464056`
- `bus_83_angle_deg_phase_a` (min): observed `-1.508723`, baseline `-1.080683`, delta `-0.428039`

## Cyber Evidence Used

- `attack_event` via `tls` on `Precursor to scn_coordinated_pv83_reg4a_disturbance` at `2025-06-01T12:59:45Z`
- `attack_event` via `mqtt` on `curtailment_frac` at `2025-06-01T13:00:00Z`
- `auth_failure` via `tls` on `Precursor to scn_coordinated_pv83_reg4a_disturbance` at `2025-06-01T12:59:45Z`

## Defensive Next Steps

- Inspect command logs for creg4a and verify that recent issued setpoints match the approved schedule.
- Compare operator-facing telemetry against local device logs or historian data for the same window.
- Review authentication, protocol source, and actor identity for suspicious writes or campaign-tagged events.
- Verify DER dispatch policy, curtailment policy, or charge-discharge schedule against expected operating conditions.
- Escalate the event as a potentially coordinated anomaly because multiple assets show concurrent disturbance evidence.

## Confidence Note

- Confidence is higher because both cyber and physical evidence point in the same direction.

## Limitations

- Scenario metadata, when present, is for offline evaluation only and should not be treated as live operational truth.
