"""Repository orchestration script for DERGuardian.

This script runs or rebuilds methodology alignment common artifacts for audits, figures,
reports, or reproducibility checks. It is release-support code and must preserve
the separation between canonical benchmark, replay, heldout synthetic, and
extension experiment contexts.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE2_BENCH_ROOT = ROOT / "phase2_llm_benchmark"
HELDOUT_ROOT = PHASE2_BENCH_ROOT / "heldout_llm_response"
HELDOUT_RESULT_ROOT = PHASE2_BENCH_ROOT / "new_respnse_result" / "models"
IMPROVED_PHASE3_ROOT = ROOT / "outputs" / "window_size_study" / "improved_phase3"
IMPROVED_EVAL_ROOT = IMPROVED_PHASE3_ROOT / "evaluations"

GENERATOR_SOURCES = ["chatgpt", "claude", "gemini", "grok"]

BENCHMARK_FAMILY_TO_CANONICAL = {
    "false_data_injection": "false_data_injection",
    "command_delay": "command_delay",
    "command_suppression": "command_suppression",
    "DER_disconnect": "degradation",
    "oscillatory_control": "unauthorized_command",
    "coordinated_multi_asset": "coordinated_campaign",
}

SEVERITY_SCORE = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}

FAMILY_PROXIMITY = {
    "false_data_injection": {"command_suppression": 0.20},
    "command_delay": {"command_suppression": 0.60, "oscillatory_control": 0.20, "coordinated_multi_asset": 0.35},
    "command_suppression": {"command_delay": 0.60, "DER_disconnect": 0.50, "false_data_injection": 0.20},
    "DER_disconnect": {"command_suppression": 0.50, "coordinated_multi_asset": 0.35},
    "oscillatory_control": {"coordinated_multi_asset": 0.50, "command_delay": 0.20},
    "coordinated_multi_asset": {"oscillatory_control": 0.50, "DER_disconnect": 0.35, "command_delay": 0.35},
}

ACTION_EVAL_KEYWORDS = {
    "false_data_injection": {"telemetry", "historian", "sensor", "measurement", "cross-check"},
    "command_delay": {"timestamp", "latency", "queue", "acknowledgement", "delay"},
    "command_suppression": {"blocked", "frozen", "setpoint", "acknowledgement", "write path"},
    "DER_disconnect": {"status", "availability", "trip", "disconnect", "mode"},
    "oscillatory_control": {"oscillation", "toggle", "tap", "switching", "control loop"},
    "coordinated_multi_asset": {"correlate", "multiple assets", "cross-asset", "campaign", "escalate"},
}

GENERIC_ACTION_VERBS = {"inspect", "verify", "compare", "review", "monitor", "correlate", "escalate"}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "into",
    "from",
    "while",
    "than",
    "then",
    "when",
    "same",
    "only",
    "more",
    "less",
    "their",
    "there",
    "which",
    "have",
    "will",
    "does",
    "above",
    "below",
    "remain",
    "remains",
    "over",
    "under",
    "would",
    "could",
    "should",
    "after",
    "before",
    "during",
    "within",
    "through",
}


@dataclass(slots=True)
class ReplaySummary:
    """Structured object used by the repository orchestration workflow."""

    source_bundle: str
    scenario_id: str
    generator_source: str
    detected: bool
    latency_seconds: float | None
    first_detection_time: str | None
    max_score: float | None
    mean_score: float | None
    positive_windows: int | None
    evaluation_dir: str


def ensure_dir(path: Path) -> None:
    """Handle ensure dir within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    """Read json for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    """Write json for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return path


def write_markdown(path: Path, text: str) -> Path:
    """Write markdown for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def json_default(value: Any) -> Any:
    """Handle json default within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_sequence(value: Any) -> list[Any]:
    """Handle parse sequence within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("("):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [text]
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return list(parsed)
        return [text]
    if hasattr(value, "tolist"):
        listed = value.tolist()
        return listed if isinstance(listed, list) else [listed]
    return [value]


def pipe_join(values: list[Any]) -> str:
    """Handle pipe join within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    cleaned = [str(item).strip() for item in values if str(item).strip()]
    return "|".join(cleaned)


def safe_float(value: Any) -> float | None:
    """Handle safe float within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def file_size_mb(path: Path) -> float:
    """Handle file size mb within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if path.is_file():
        return path.stat().st_size / (1024.0 * 1024.0)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024.0 * 1024.0)


def normalize_benchmark_family(family: str | None) -> str:
    """Handle normalize benchmark family within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    raw = str(family or "").strip()
    return BENCHMARK_FAMILY_TO_CANONICAL.get(raw, raw)


def normalize_signal(signal: str) -> str:
    """Handle normalize signal within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    text = str(signal).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return text.strip("_")


def tokenize(text: str) -> set[str]:
    """Handle tokenize within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return {token for token in re.findall(r"[a-z0-9_]+", str(text).lower()) if token and token not in STOPWORDS}


def label_score(label: str) -> float:
    """Handle label score within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if label in {"exact", "high"}:
        return 1.0
    if label in {"partial", "nearest_valid_family", "medium"}:
        return 0.5
    return 0.0


def asset_match(truth_assets: set[str], predicted_assets: set[str]) -> str:
    """Handle asset match within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if truth_assets and predicted_assets == truth_assets:
        return "exact"
    if truth_assets & predicted_assets:
        return "partial"
    return "none"


def family_match(truth_family: str, predicted_family: str) -> str:
    """Handle family match within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if predicted_family == truth_family:
        return "exact"
    if predicted_family in FAMILY_PROXIMITY.get(truth_family, {}):
        return "nearest_valid_family"
    return "none"


def evidence_overlap(
    truth_signals: set[str],
    predicted_signals: set[str],
    truth_assets: set[str],
    predicted_assets: set[str],
) -> tuple[float, str]:
    """Handle evidence overlap within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    base = len(truth_signals & predicted_signals) / max(len(truth_signals), 1) if truth_signals else 0.0
    if base >= 0.5:
        return float(base), "high"
    if base > 0.0 or (truth_assets & predicted_assets):
        return float(max(base, 0.25 if truth_assets & predicted_assets else 0.0)), "medium"
    return float(base), "low"


def action_relevance_score(
    truth_family: str,
    truth_assets: set[str],
    truth_effects: str,
    actions: list[str],
) -> tuple[float, str]:
    """Handle action relevance score within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if not actions:
        return 0.0, "low"
    text = " ".join(str(item) for item in actions).lower()
    asset_hit = any(asset.lower() in text for asset in truth_assets)
    keyword_hit = any(keyword.lower() in text for keyword in ACTION_EVAL_KEYWORDS.get(truth_family, set()))
    effect_keywords = {token for token in tokenize(truth_effects) if len(token) > 4}
    effect_hit = len(effect_keywords & tokenize(text)) >= 1 if effect_keywords else False
    verb_hit = any(verb in text for verb in GENERIC_ACTION_VERBS)
    if (asset_hit or effect_hit) and keyword_hit and verb_hit:
        return 1.0, "high"
    if keyword_hit or ((asset_hit or effect_hit) and verb_hit):
        return 0.5, "medium"
    return 0.0, "low"


def partial_alignment_score(truth_family: str, predicted_family: str, grounding_overlap: float) -> float:
    """Handle partial alignment score within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if truth_family == predicted_family:
        return 1.0
    base = FAMILY_PROXIMITY.get(truth_family, {}).get(predicted_family, 0.0)
    if base <= 0.0:
        return 0.0
    overlap = max(min(grounding_overlap, 1.0), 0.0)
    return float(base * (0.5 + 0.5 * overlap))


def selected_heldout_json_path(generator_source: str) -> Path:
    """Handle selected heldout json path within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    folder = HELDOUT_ROOT / generator_source
    candidate = folder / "new_respnse.json"
    if candidate.exists():
        return candidate
    fallback = folder / "response.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No heldout JSON found for {generator_source}: {folder}")


def load_generator_truth_lookup(generator_source: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load generator truth lookup for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    payload = read_json(selected_heldout_json_path(generator_source))
    lookup = {str(item["scenario_id"]): item for item in payload.get("scenarios", [])}
    return payload, lookup


def load_repaired_truth_lookup(generator_source: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load repaired truth lookup for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    path = IMPROVED_PHASE3_ROOT / "repaired_bundles_raw" / generator_source / "repaired_benchmark_bundle.json"
    payload = read_json(path)
    lookup = {str(item["scenario_id"]): item for item in payload.get("scenarios", [])}
    return payload, lookup


def load_human_authored_truth_lookup() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load human authored truth lookup for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    path = IMPROVED_PHASE3_ROOT / "additional_source" / "human_authored" / "benchmark_bundle.json"
    payload = read_json(path)
    lookup = {str(item["scenario_id"]): item for item in payload.get("scenarios", [])}
    return payload, lookup


def benchmark_scenario_metadata(scenario: dict[str, Any]) -> dict[str, Any]:
    """Handle benchmark scenario metadata within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    additional_targets = list(scenario.get("additional_targets", []) or [])
    target_assets = [str(scenario.get("target_asset", "")).strip()]
    target_assets.extend(str(item.get("target_asset", "")).strip() for item in additional_targets)
    target_signals = [str(scenario.get("target_signal", "")).strip()]
    target_signals.extend(str(item.get("target_signal", "")).strip() for item in additional_targets)
    observable_signals = [str(item).strip() for item in scenario.get("observable_signals", []) if str(item).strip()]
    all_signals = []
    for signal in target_signals + observable_signals:
        if signal and signal not in all_signals:
            all_signals.append(signal)
    magnitude = scenario.get("magnitude", {}) or {}
    numeric_magnitude = safe_float(magnitude.get("value"))
    return {
        "scenario_name": str(scenario.get("scenario_name", "")),
        "attack_family_raw": str(scenario.get("attack_family", "")),
        "attack_family": normalize_benchmark_family(scenario.get("attack_family")),
        "target_component": str(scenario.get("target_component", "")),
        "target_asset": str(scenario.get("target_asset", "")),
        "target_signal": str(scenario.get("target_signal", "")),
        "affected_assets": [item for item in target_assets if item],
        "affected_signals": all_signals,
        "severity": "",
        "observable_signals": observable_signals,
        "description": "",
        "expected_effects": [str(item) for item in scenario.get("expected_physical_effects", [])],
        "numeric_magnitude": numeric_magnitude,
        "magnitude_unit": str(magnitude.get("unit", "")),
        "additional_target_count": len(additional_targets),
    }


def phase2_scenario_metadata(scenario: dict[str, Any]) -> dict[str, Any]:
    """Handle phase2 scenario metadata within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    additional_targets = list(scenario.get("additional_targets", []) or [])
    target_assets = [str(scenario.get("target_asset", "")).strip()]
    target_assets.extend(str(item.get("target_asset", "")).strip() for item in additional_targets)
    target_signals = [str(scenario.get("target_signal", "")).strip()]
    target_signals.extend(str(item.get("target_signal", "")).strip() for item in additional_targets)
    observable_signals = [str(item).strip() for item in scenario.get("observable_signals", []) if str(item).strip()]
    all_signals = []
    for signal in target_signals + observable_signals:
        if signal and signal not in all_signals:
            all_signals.append(signal)
    category = str(scenario.get("category", ""))
    numeric_magnitude = safe_float(scenario.get("magnitude_value"))
    return {
        "scenario_name": str(scenario.get("scenario_name", "")),
        "attack_family_raw": category,
        "attack_family": category,
        "target_component": str(scenario.get("target_component", "")),
        "target_asset": str(scenario.get("target_asset", "")),
        "target_signal": str(scenario.get("target_signal", "")),
        "affected_assets": [item for item in target_assets if item],
        "affected_signals": all_signals,
        "severity": str(scenario.get("severity", "")),
        "observable_signals": observable_signals,
        "description": str(scenario.get("description", "")),
        "expected_effects": [str(scenario.get("expected_effect", ""))] if str(scenario.get("expected_effect", "")).strip() else [],
        "numeric_magnitude": numeric_magnitude,
        "magnitude_unit": str(scenario.get("magnitude_units", "")),
        "additional_target_count": len(additional_targets),
    }


def canonical_labels_lookup() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Handle canonical labels lookup within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    labels = pd.read_parquet(ROOT / "outputs" / "attacked" / "attack_labels.parquet").copy()
    labels_lookup: dict[str, dict[str, Any]] = {}
    for _, row in labels.iterrows():
        labels_lookup[str(row["scenario_id"])] = {
            "severity": str(row.get("severity", "")),
            "affected_assets": [str(item) for item in parse_sequence(row.get("affected_assets")) if str(item).strip()],
            "affected_signals": [str(item) for item in parse_sequence(row.get("affected_signals")) if str(item).strip()],
            "target_component": str(row.get("target_component", "")),
            "attack_family": str(row.get("attack_family", "")),
        }
    return labels, labels_lookup


def read_labels_lookup(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Read labels lookup for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    labels = pd.read_parquet(path).copy()
    labels_lookup: dict[str, dict[str, Any]] = {}
    for _, row in labels.iterrows():
        labels_lookup[str(row["scenario_id"])] = {
            "severity": str(row.get("severity", "")),
            "affected_assets": [str(item) for item in parse_sequence(row.get("affected_assets")) if str(item).strip()],
            "affected_signals": [str(item) for item in parse_sequence(row.get("affected_signals")) if str(item).strip()],
            "target_component": str(row.get("target_component", "")),
            "attack_family": str(row.get("attack_family", "")),
        }
    return labels, labels_lookup


def replay_source_bundle_from_eval_dir(name: str) -> tuple[str, str]:
    """Handle replay source bundle from eval dir within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if "__canonical_bundle_replay__" in name:
        return "canonical_bundle", "canonical_bundle"
    if "__existing_heldout_phase2_bundle__" in name:
        generator = name.split("__", 1)[0]
        return f"heldout_original_{generator}", generator
    if "__repaired_heldout_bundle__" in name:
        generator = name.split("__", 1)[0]
        return f"heldout_repaired_{generator}", generator
    if "__human_authored_heldout__" in name:
        return "additional_heldout_human_authored", "human_authored"
    raise ValueError(f"Unsupported evaluation directory name: {name}")


def load_transformer_replay_lookup() -> dict[tuple[str, str], ReplaySummary]:
    """Load transformer replay lookup for the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    lookup: dict[tuple[str, str], ReplaySummary] = {}
    if not IMPROVED_EVAL_ROOT.exists():
        return lookup
    for eval_dir in sorted(path for path in IMPROVED_EVAL_ROOT.iterdir() if path.is_dir()):
        transformer_summary = eval_dir / "60s" / "transformer" / "scenario_summary.csv"
        if not transformer_summary.exists():
            continue
        source_bundle, generator_source = replay_source_bundle_from_eval_dir(eval_dir.name)
        frame = pd.read_csv(transformer_summary)
        for _, row in frame.iterrows():
            score_summary = row.get("score_summary")
            parsed_score = {}
            if isinstance(score_summary, str) and score_summary.strip():
                try:
                    parsed_score = json.loads(score_summary)
                except json.JSONDecodeError:
                    parsed_score = {}
            summary = ReplaySummary(
                source_bundle=source_bundle,
                scenario_id=str(row["scenario_id"]),
                generator_source=generator_source,
                detected=bool(int(row.get("detected", 0))),
                latency_seconds=safe_float(row.get("latency_seconds")),
                first_detection_time=str(row.get("first_detection_time")) if not pd.isna(row.get("first_detection_time")) else None,
                max_score=safe_float(parsed_score.get("max_score")),
                mean_score=safe_float(parsed_score.get("mean_score")),
                positive_windows=int(parsed_score.get("positive_windows")) if safe_float(parsed_score.get("positive_windows")) is not None else None,
                evaluation_dir=str(eval_dir.relative_to(ROOT)),
            )
            lookup[(source_bundle, summary.scenario_id)] = summary
    return lookup


def scenario_truth_lookup(generator_source: str, source_bundle: str) -> dict[str, dict[str, Any]]:
    """Handle scenario truth lookup within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    if source_bundle == "canonical_bundle":
        manifest = read_json(ROOT / "outputs" / "attacked" / "scenario_manifest.json")
        return {str(item["scenario_id"]): item for item in manifest.get("applied_scenarios", [])}
    if source_bundle.startswith("heldout_repaired_"):
        _, lookup = load_repaired_truth_lookup(generator_source)
        return lookup
    if source_bundle == "additional_heldout_human_authored":
        _, lookup = load_human_authored_truth_lookup()
        return lookup
    _, lookup = load_generator_truth_lookup(generator_source)
    return lookup


def inventory_required_fields() -> list[str]:
    """Handle inventory required fields within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return [
        "dataset_id",
        "source_bundle",
        "generator_source",
        "scenario_id",
        "attack_family",
        "affected_assets",
        "affected_signals",
        "target_component",
        "severity",
        "accepted_rejected",
        "repair_applied",
        "replay_evaluated",
    ]


def metadata_completeness_ratio(row: pd.Series) -> float:
    """Handle metadata completeness ratio within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    filled = 0
    for field in inventory_required_fields():
        value = row.get(field)
        if field in {"affected_assets", "affected_signals"}:
            if parse_sequence(value):
                filled += 1
            continue
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            filled += 1
    return filled / len(inventory_required_fields())


def balance_score_from_counts(counts: pd.Series) -> float:
    """Handle balance score from counts within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    series = pd.to_numeric(counts, errors="coerce").fillna(0.0)
    series = series[series > 0]
    if series.empty or len(series) == 1:
        return 1.0
    probabilities = series / series.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / math.log(len(probabilities))


def severity_numeric(value: str | None) -> float | None:
    """Handle severity numeric within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    text = str(value or "").strip().lower()
    if not text:
        return None
    return SEVERITY_SCORE.get(text)


def pretty_label(source_bundle: str) -> str:
    """Handle pretty label within the repository orchestration workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    mapping = {
        "canonical_bundle": "Canonical bundle",
        "additional_heldout_human_authored": "Human-authored heldout",
    }
    if source_bundle in mapping:
        return mapping[source_bundle]
    if source_bundle.startswith("heldout_original_"):
        return f"{source_bundle.replace('heldout_original_', '').title()} heldout"
    if source_bundle.startswith("heldout_repaired_"):
        return f"{source_bundle.replace('heldout_repaired_', '').title()} repaired"
    return source_bundle.replace("_", " ").title()
