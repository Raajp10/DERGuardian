# LLM Usage Clarity

DERGuardian uses LLM-related components only as bounded support layers. They are not the core anomaly detector.

## Phase 2

In Phase 2, LLMs are treated as scenario authoring assistants. Scenario content must pass structured schema validation, physics/safety checks, metadata checks, and coverage/difficulty audits before it is treated as usable evidence.

Safe wording:

- "LLM-assisted structured scenario generation"
- "schema-bound synthetic attack scenario authoring"
- "heldout synthetic generator bundles"

Unsafe wording:

- "LLM proves real-world zero-day robustness"
- "LLM autonomously discovers field attacks"

## Phase 3

In Phase 3, LLM-related tooling is used for explanation generation and bounded classification support after detector alerts. The XAI layer is evaluated through family attribution, asset attribution, evidence grounding, action relevance, and error taxonomy reports.

Safe wording:

- "grounded post-alert operator-facing support"
- "evidence-grounded family attribution"
- "bounded explanation/classification branch"

Unsafe wording:

- "human-like root-cause analysis"
- "human-level reasoning"

## Not Used For Core Detector Training

The canonical detector benchmark is not an LLM benchmark. The canonical winner remains `transformer @ 60s`, selected from the frozen Phase 1 detector benchmark.

The LoRA branch is experimental and weak. It is not detector benchmark evidence and does not replace the canonical detector.
