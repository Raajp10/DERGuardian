"""Phase 2 scenario and attacked-dataset support for DERGuardian.

This module implements contracts logic for schema-bound synthetic attack
scenarios, injection compilation, cyber logs, labels, validation, or reporting.
Generated scenarios are heldout synthetic evidence and are not claimed as
real-world zero-day proof.
"""

from __future__ import annotations

# Canonical executable Phase 2 components. Every target_component accepted by the
# canonical schema must be safely materializable by the JSON-driven compiler and
# attacked-dataset generator.
CANONICAL_TARGET_COMPONENTS: tuple[str, ...] = (
    "pv",
    "bess",
    "regulator",
    "capacitor",
    "switch",
    "measured_layer",
)

# Historical documentation mentioned cyber_layer as a possible target type, but
# the canonical executable pipeline does not materialize it safely. Keep the
# identifier only as a deprecated note for audits and backward-looking docs.
LEGACY_UNSUPPORTED_TARGET_COMPONENTS: tuple[str, ...] = ("cyber_layer",)

CANONICAL_SCENARIO_CATEGORIES: tuple[str, ...] = (
    "false_data_injection",
    "replay",
    "unauthorized_command",
    "telemetry_corruption",
    "command_delay",
    "command_suppression",
    "degradation",
    "coordinated_campaign",
)

CANONICAL_INJECTION_TYPES: tuple[str, ...] = (
    "bias",
    "scale",
    "replay",
    "freeze",
    "delay",
    "dropout",
    "command_override",
    "command_delay",
    "command_suppression",
    "mode_change",
)

CANONICAL_TEMPORAL_PATTERNS: tuple[str, ...] = (
    "step",
    "ramp",
    "pulse",
    "staircase",
    "sine",
    "replay",
    "freeze",
    "burst",
)

CANONICAL_PHASE2_EXECUTION_PATH: tuple[str, ...] = (
    "Validate structured scenario JSON against the canonical executable schema.",
    "Compile measurement actions and physical/control override actions.",
    "Rerun OpenDSS only when scenarios materialize physical overrides.",
    "Synthesize attacked measured and cyber outputs from the compiled actions.",
    "Emit attacked truth, measured, cyber, label, merged-window, and QA artifacts.",
)


def is_supported_target_component(component: str) -> bool:
    """Handle is supported target component within the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return component in CANONICAL_TARGET_COMPONENTS
