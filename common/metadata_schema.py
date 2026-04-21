"""Shared utility support for DERGuardian.

This module provides metadata schema helpers used across the Phase 1 data
pipeline, Phase 2 scenario pipeline, and Phase 3 evaluation/reporting layers.
The functions here are infrastructure code: they prepare paths, metadata,
profiles, graphs, units, or time alignment without changing canonical detector
outputs or benchmark decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetArtifact:
    """Structured object used by the shared DERGuardian utility workflow."""

    name: str
    path: str
    rows: int | None = None
    columns: int | None = None
    description: str = ""


@dataclass(slots=True)
class AttackLabel:
    """Structured object used by the shared DERGuardian utility workflow."""

    scenario_id: str
    scenario_name: str
    attack_family: str
    severity: str
    start_time_utc: str
    end_time_utc: str
    affected_assets: list[str]
    affected_signals: list[str]
    target_component: str
    causal_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationCheck:
    """Structured object used by the shared DERGuardian utility workflow."""

    name: str
    status: str
    detail: str
    metric: float | None = None
    threshold: float | None = None


@dataclass(slots=True)
class RunManifest:
    """Structured object used by the shared DERGuardian utility workflow."""

    run_id: str
    scenario_id: str
    split_id: str
    config: dict[str, Any]
    inventory_summary: dict[str, Any]
    assumptions: list[str]
    artifacts: list[DatasetArtifact] = field(default_factory=list)
    validation_checks: list[ValidationCheck] = field(default_factory=list)
    graph_paths: list[str] = field(default_factory=list)
    applied_scenarios: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "split_id": self.split_id,
            "config": self.config,
            "inventory_summary": self.inventory_summary,
            "assumptions": self.assumptions,
            "artifacts": [asdict(item) for item in self.artifacts],
            "validation_checks": [asdict(item) for item in self.validation_checks],
            "graph_paths": self.graph_paths,
            "applied_scenarios": self.applied_scenarios,
            "warnings": self.warnings,
        }


def artifact_from_path(path: str | Path, rows: int | None = None, columns: int | None = None, description: str = "") -> DatasetArtifact:
    """Handle artifact from path within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    resolved = Path(path)
    return DatasetArtifact(name=resolved.name, path=str(resolved), rows=rows, columns=columns, description=description)
