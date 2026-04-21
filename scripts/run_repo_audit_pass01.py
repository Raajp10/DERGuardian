from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]

COMPLETE_TREE_PATH = ROOT / "COMPLETE_FILE_TREE_RELEVANT.txt"
EVIDENCE_INDEX_PATH = ROOT / "EVIDENCE_INDEX.csv"
VERIFICATION_AUDIT_PATH = ROOT / "VERIFICATION_AUDIT_MASTER.csv"
VERIFICATION_GAPS_PATH = ROOT / "VERIFICATION_GAPS.md"


RELEVANT_ROOTS = [
    ROOT / "outputs" / "window_size_study",
    ROOT / "outputs" / "attacked",
    ROOT / "outputs" / "clean",
    ROOT / "outputs" / "phase3_lora_extension",
    ROOT / "phase2",
    ROOT / "phase3_explanations",
    ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models",
    ROOT / "scripts",
    ROOT / "reports",
    ROOT / "deployment_runtime",
]

EXCLUDED_DIRS = {"__pycache__", ".git", ".venv", "venv"}
EXCLUDED_SUFFIXES = {".pyc", ".zip"}
REPORT_SUFFIXES = {".md"}
TABLE_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".parquet"}
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg"}

TOP_LEVEL_KEYWORDS = [
    "benchmark",
    "heldout",
    "replay",
    "phase",
    "xai",
    "lora",
    "deployment",
    "diagram",
    "complete",
    "final",
    "publication",
    "window",
    "claim",
    "canonical",
    "improved",
]

REQUIRED_ARTIFACTS = [
    "CANONICAL_ARTIFACT_FREEZE.md",
    "phase2_scenario_master_inventory.csv",
    "phase2_coverage_summary.md",
    "phase2_attack_family_distribution.csv",
    "phase2_asset_signal_coverage.csv",
    "phase2_difficulty_calibration.csv",
    "phase2_diversity_and_quality_report.md",
    "phase2_family_distribution.png",
    "phase2_asset_coverage_heatmap.png",
    "phase2_signal_coverage_heatmap.png",
    "phase2_difficulty_distribution.png",
    "phase2_generator_coverage_comparison.png",
    "phase3_lora_dataset_manifest.json",
    "phase3_lora_training_config.yaml",
    "phase3_lora_results.csv",
    "phase3_lora_eval_report.md",
    "phase3_lora_model_card.md",
    "phase3_lora_family_accuracy.png",
    "phase3_lora_asset_accuracy.png",
    "phase3_lora_grounding_comparison.png",
    "phase3_lora_latency_vs_quality.png",
    "xai_case_level_audit.csv",
    "xai_error_taxonomy.md",
    "xai_qualitative_examples.md",
    "xai_final_validation_report.md",
    "xai_family_vs_asset_accuracy.png",
    "xai_error_taxonomy_breakdown.png",
    "xai_grounding_vs_family_accuracy.png",
    "deployment_benchmark_results.csv",
    "deployment_benchmark_report.md",
    "deployment_readiness_summary.md",
    "deployment_latency_by_model.png",
    "deployment_memory_by_model.png",
    "deployment_throughput_by_model.png",
    "deployment_accuracy_latency_tradeoff.png",
    "FINAL_DIAGRAM_ALIGNMENT.md",
    "DIAGRAM_SAFE_LABELS.md",
    "COMPLETE_PROJECT_STATUS.md",
    "COMPLETE_RESULTS_AND_DISCUSSION.md",
    "COMPLETE_PUBLICATION_SAFE_CLAIMS.md",
    "FINAL_PROJECT_DECISION.md",
]


PHASE2_REQUIRED_FIELDS = [
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


@dataclass
class VerificationResult:
    artifact_path: str
    required: bool
    exists: bool
    nonempty: bool
    structurally_valid: bool
    content_verified: bool
    backed_by_source_data: bool
    numerically_consistent: bool
    pass_fail: str
    exact_issue: str


def rel_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def path_from_text(value: str) -> Path:
    candidate = value.strip().strip("`").strip()
    candidate = candidate.replace("\\", "/")
    return (ROOT / candidate).resolve()


def truthy(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def parse_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        except Exception:
            pass
    return [item.strip() for item in text.split("|") if item.strip()]


def normalized_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    series = pd.Series(values)
    counts = series.value_counts()
    if len(counts) <= 1:
        return 1.0
    probs = counts / counts.sum()
    entropy = float(-(probs * probs.map(math.log)).sum())
    return entropy / math.log(len(counts))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in FIGURE_SUFFIXES:
        return "figure"
    if suffix in REPORT_SUFFIXES:
        return "report"
    if suffix == ".py":
        return "script"
    if suffix in {".yaml", ".yml"}:
        return "config"
    if suffix == ".json":
        return "json"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "table"
    if suffix in {".pt", ".pkl"}:
        return "checkpoint"
    return suffix.lstrip(".") or "file"


def phase_for(path: Path) -> str:
    lower = rel_path(path).lower()
    name = path.name.lower()
    if "phase3_lora" in lower or "phase3_lora" in name:
        return "phase3_lora"
    if "xai" in lower:
        return "phase3_xai"
    if "deployment" in lower:
        return "phase3_deployment"
    if "phase2" in lower or "attacked" in lower:
        return "phase2"
    if "heldout" in lower or "replay" in lower:
        return "phase3_replay"
    if "window_size" in lower or "phase1" in lower or "model_full_run" in lower:
        return "phase1"
    if "diagram" in name or "complete_" in name or name.startswith("final_"):
        return "final_reports"
    return "supporting"


def is_top_level_relevant(path: Path) -> bool:
    if path.parent != ROOT or not path.is_file():
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    lowered = path.name.lower()
    return any(keyword in lowered for keyword in TOP_LEVEL_KEYWORDS)


def collect_relevant_files() -> list[Path]:
    found: dict[str, Path] = {}
    for root in RELEVANT_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            if path.is_dir():
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            found[str(path.resolve())] = path
    for path in ROOT.iterdir():
        if is_top_level_relevant(path):
            found[str(path.resolve())] = path
    for relative in REQUIRED_ARTIFACTS:
        path = ROOT / relative
        found.setdefault(str(path.resolve()), path)
    return sorted(found.values(), key=lambda item: rel_path(item) if item.exists() else item.name)


def write_tree(paths: list[Path]) -> None:
    lines = ["# Relevant Repository Tree", ""]

    def append_tree(base: Path) -> None:
        if not base.exists():
            lines.append(f"[missing] {rel_path(base)}")
            return
        lines.append(rel_path(base))
        for child in sorted(base.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
            if child.name in EXCLUDED_DIRS or child.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            walk(child, 1)

    def walk(path: Path, depth: int) -> None:
        prefix = "  " * depth + "- "
        if path.is_dir():
            lines.append(f"{prefix}{path.name}/")
            for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
                if child.name in EXCLUDED_DIRS or child.suffix.lower() in EXCLUDED_SUFFIXES:
                    continue
                walk(child, depth + 1)
        else:
            size = path.stat().st_size if path.exists() else 0
            lines.append(f"{prefix}{path.name} [{size} bytes]")

    lines.append("## Top-level relevant files")
    top_level = [path for path in paths if path.parent == ROOT]
    for path in top_level:
        if path.exists():
            lines.append(f"- {rel_path(path)} [{path.stat().st_size} bytes]")
        else:
            lines.append(f"- {path.name} [missing]")
    lines.append("")
    lines.append("## Relevant roots")
    for root in RELEVANT_ROOTS:
        append_tree(root)
        lines.append("")
    COMPLETE_TREE_PATH.write_text("\n".join(lines), encoding="utf-8")


def collect_report_texts(paths: list[Path]) -> dict[str, str]:
    report_text = {}
    for path in paths:
        if path.exists() and path.suffix.lower() in REPORT_SUFFIXES:
            report_text[rel_path(path)] = read_text(path)
    return report_text


def build_evidence_index(paths: list[Path]) -> pd.DataFrame:
    report_texts = collect_report_texts(paths)
    rows = []
    for path in paths:
        exists = path.exists()
        nonempty = exists and path.stat().st_size > 0
        artifact_rel = rel_path(path) if exists else path.name
        basename = path.name
        referenced_by = []
        for report_rel, text in report_texts.items():
            if report_rel == artifact_rel:
                continue
            if artifact_rel in text or basename in text:
                referenced_by.append(report_rel)
        notes = []
        if artifact_rel in REQUIRED_ARTIFACTS or path.name in REQUIRED_ARTIFACTS:
            notes.append("required_artifact")
        if "ttm" in artifact_rel.lower():
            notes.append("ttm_related")
        if "canonical" in artifact_rel.lower():
            notes.append("canonical_related")
        rows.append(
            {
                "artifact_path": artifact_rel,
                "artifact_type": artifact_type(path),
                "phase": phase_for(path),
                "exists": exists,
                "nonempty": nonempty,
                "last_modified": path.stat().st_mtime if exists else None,
                "referenced_by_reports": ";".join(sorted(referenced_by)),
                "needs_verification": exists and (path.suffix.lower() in REPORT_SUFFIXES | TABLE_SUFFIXES | FIGURE_SUFFIXES or path.name in REQUIRED_ARTIFACTS),
                "notes": ";".join(notes),
            }
        )
    frame = pd.DataFrame(rows)
    frame["last_modified"] = pd.to_datetime(frame["last_modified"], unit="s", errors="coerce")
    frame.to_csv(EVIDENCE_INDEX_PATH, index=False)
    return frame


def verify_image(path: Path) -> tuple[bool, bool, str]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            stat = ImageStat.Stat(image.convert("L"))
            extrema = stat.extrema[0]
            spread = float(extrema[1] - extrema[0])
            structurally_valid = width >= 200 and height >= 150
            content_verified = spread > 5.0
            issue = ""
            if not structurally_valid:
                issue = f"Image dimensions too small ({width}x{height})."
            elif not content_verified:
                issue = "Image appears low-variance or blank."
            return structurally_valid, content_verified, issue
    except Exception as exc:
        return False, False, f"Image failed to load: {exc}"


def verify_basic_structure(path: Path) -> tuple[bool, str]:
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            if frame.empty:
                return False, "CSV has no data rows."
            if len(frame.columns) == 0:
                return False, "CSV has no columns."
            return True, ""
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
            if frame.empty:
                return False, "Parquet has no data rows."
            return True, ""
        if path.suffix.lower() == ".json":
            json.loads(read_text(path))
            return True, ""
        if path.suffix.lower() in {".yaml", ".yml"}:
            yaml.safe_load(read_text(path))
            return True, ""
        if path.suffix.lower() == ".md":
            text = read_text(path)
            if not text.strip():
                return False, "Markdown file is empty."
            if "#" not in text:
                return False, "Markdown file has no headings."
            return True, ""
        if path.suffix.lower() in FIGURE_SUFFIXES:
            structurally_valid, _, issue = verify_image(path)
            return structurally_valid, issue
        return True, ""
    except Exception as exc:
        return False, f"Failed to parse file: {exc}"


def load_inventory() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "phase2_scenario_master_inventory.csv")
    frame["repair_applied_bool"] = truthy(frame["repair_applied"])
    frame["replay_evaluated_bool"] = truthy(frame["replay_evaluated"])
    frame["replay_detected_bool"] = truthy(frame["replay_detected"])
    return frame


def compute_phase2_asset_signal_coverage(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension_name, column in [("asset", "affected_assets"), ("signal", "affected_signals")]:
        exploded = inventory.assign(_value=inventory[column].apply(parse_sequence)).explode("_value")
        exploded["_value"] = exploded["_value"].astype(str).str.strip()
        exploded = exploded[exploded["_value"] != ""].copy()
        for value, group in exploded.groupby("_value", dropna=False):
            valid_group = group[group["accepted_rejected"] == "accepted"]
            rows.append(
                {
                    "dimension": dimension_name,
                    "name": value,
                    "all_scenarios": int(group["scenario_id"].nunique()),
                    "validated_scenarios": int(valid_group["scenario_id"].nunique()),
                    "rejected_scenarios": int(group.loc[group["accepted_rejected"] == "rejected", "scenario_id"].nunique()),
                    "repair_rows": int(group["repair_applied_bool"].sum()),
                    "generator_coverage_validated": int(valid_group["generator_source"].nunique()),
                    "source_bundle_coverage_validated": int(valid_group["source_bundle"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["dimension", "validated_scenarios", "name"], ascending=[True, False, True]).reset_index(drop=True)


def parse_float(text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}[^0-9\-]*(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def parse_int(text: str, label: str) -> int | None:
    value = parse_float(text, label)
    return int(round(value)) if value is not None else None


def verify_freeze_report(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    referenced_paths = re.findall(r"`([^`]+)`", text)
    existing_paths = []
    for value in referenced_paths:
        if ":" in value or "/" in value or "\\" in value or value.endswith(".md") or value.endswith(".csv") or value.endswith(".json"):
            candidate = path_from_text(value)
            existing_paths.append(candidate.exists())
    content_verified = "transformer" in text and "60s" in text and "No canonical benchmark outputs were overwritten" in text
    backed = all(existing_paths) if existing_paths else False
    issue = ""
    if not content_verified:
        issue = "Freeze report does not clearly preserve the transformer @ 60s canonical winner."
    elif not backed:
        issue = "One or more referenced freeze-report paths do not exist."
    return content_verified, backed, True, issue


def verify_phase2_inventory(path: Path) -> tuple[bool, bool, bool, str]:
    frame = load_inventory()
    missing = [column for column in PHASE2_REQUIRED_FIELDS if column not in frame.columns]
    accepted_values = set(frame["accepted_rejected"].dropna().astype(str).unique())
    bundle_paths_exist = frame["bundle_path"].apply(lambda value: path_from_text(str(value)).exists()).all()
    replay_dirs_exist = frame.loc[frame["replay_evaluated_bool"], "replay_evaluation_dir"].apply(lambda value: path_from_text(str(value)).exists()).all()
    content_verified = not missing and accepted_values <= {"accepted", "rejected"} and len(frame) >= 50
    backed = bundle_paths_exist and replay_dirs_exist
    numerically_consistent = frame["metadata_completeness_ratio"].between(0, 1).all()
    issue_parts = []
    if missing:
        issue_parts.append(f"Missing columns: {missing}")
    if not bundle_paths_exist:
        issue_parts.append("At least one source bundle path is missing.")
    if not replay_dirs_exist:
        issue_parts.append("At least one replay evaluation directory is missing for a replay-evaluated row.")
    if not numerically_consistent:
        issue_parts.append("metadata_completeness_ratio is outside [0,1].")
    return content_verified, backed, numerically_consistent, " ".join(issue_parts)


def verify_phase2_family_distribution(path: Path) -> tuple[bool, bool, bool, str]:
    inventory = load_inventory()
    frame = pd.read_csv(path).sort_values("attack_family").reset_index(drop=True)
    expected_rows = []
    for family, group in inventory.groupby("attack_family", dropna=False):
        valid_group = group[group["accepted_rejected"] == "accepted"]
        expected_rows.append(
            {
                "attack_family": family,
                "all_scenarios": int(len(group)),
                "validated_scenarios": int(len(valid_group)),
                "rejected_scenarios": int((group["accepted_rejected"] == "rejected").sum()),
                "repair_rows": int(group["repair_applied_bool"].sum()),
                "generator_coverage_validated": int(valid_group["generator_source"].nunique()),
                "source_bundle_coverage_validated": int(valid_group["source_bundle"].nunique()),
                "replay_evaluated_validated": int(valid_group["replay_evaluated_bool"].sum()),
            }
        )
    expected = pd.DataFrame(expected_rows).sort_values("attack_family").reset_index(drop=True)
    content_verified = list(frame.columns) == list(expected.columns)
    numerically_consistent = frame.equals(expected)
    issue = "" if numerically_consistent else "Attack family distribution does not match grouped master inventory counts."
    return content_verified, True, numerically_consistent, issue


def verify_phase2_asset_signal(path: Path) -> tuple[bool, bool, bool, str]:
    inventory = load_inventory()
    frame = pd.read_csv(path).sort_values(["dimension", "name"]).reset_index(drop=True)
    expected = compute_phase2_asset_signal_coverage(inventory).sort_values(["dimension", "name"]).reset_index(drop=True)
    content_verified = list(frame.columns) == list(expected.columns)
    numerically_consistent = frame.equals(expected)
    issue = ""
    if not numerically_consistent:
        issue = "Asset/signal coverage table does not match master inventory-derived counts."
    return content_verified, True, numerically_consistent, issue


def verify_phase2_difficulty(path: Path) -> tuple[bool, bool, bool, str]:
    inventory = load_inventory()
    frame = pd.read_csv(path)
    required = {
        "dataset_id",
        "source_bundle",
        "generator_source",
        "scenario_id",
        "difficulty_score",
        "difficulty_bucket",
    }
    missing = sorted(required - set(frame.columns))
    merged = frame.merge(
        inventory[["dataset_id", "source_bundle", "generator_source", "scenario_id", "accepted_rejected", "replay_detected", "replay_latency_seconds", "replay_max_score"]],
        on=["dataset_id", "source_bundle", "generator_source", "scenario_id"],
        how="left",
        suffixes=("", "_inventory"),
    )
    buckets_ok = set(frame["difficulty_bucket"].dropna().unique()) <= {"easy", "moderate", "hard", "very hard"}
    content_verified = not missing and len(frame) == len(inventory) and buckets_ok
    backed = merged["accepted_rejected_inventory"].notna().all()
    latency_match = True
    if "replay_latency_seconds" in frame.columns:
        compared = merged.dropna(subset=["replay_latency_seconds_inventory"])
        if not compared.empty:
            latency_match = (
                (compared["replay_latency_seconds"].fillna(-9999.0) - compared["replay_latency_seconds_inventory"].fillna(-9999.0)).abs() < 1e-9
            ).all()
    numerically_consistent = latency_match and frame["difficulty_score"].notna().all()
    issue_parts = []
    if missing:
        issue_parts.append(f"Missing columns: {missing}")
    if not buckets_ok:
        issue_parts.append("Unexpected difficulty buckets detected.")
    if not backed:
        issue_parts.append("Some difficulty rows do not map back to the master inventory.")
    if not latency_match:
        issue_parts.append("Difficulty replay latency does not match inventory.")
    return content_verified, backed, numerically_consistent, " ".join(issue_parts)


def verify_phase2_coverage_summary(path: Path) -> tuple[bool, bool, bool, str]:
    inventory = load_inventory()
    text = read_text(path)
    validated = int((inventory["accepted_rejected"] == "accepted").sum())
    rejected = int((inventory["accepted_rejected"] == "rejected").sum())
    repairs = int(inventory["repair_applied_bool"].sum())
    replay_validated = int(inventory.loc[inventory["accepted_rejected"] == "accepted", "replay_evaluated_bool"].sum())
    families = int(inventory.loc[inventory["accepted_rejected"] == "accepted", "attack_family"].nunique())
    asset_count = len(
        {
            item
            for value in inventory.loc[inventory["accepted_rejected"] == "accepted", "affected_assets"]
            for item in parse_sequence(value)
        }
    )
    signal_count = len(
        {
            item
            for value in inventory.loc[inventory["accepted_rejected"] == "accepted", "affected_signals"]
            for item in parse_sequence(value)
        }
    )
    checks = {
        "Validated scenarios": validated,
        "Rejected scenarios": rejected,
        "Repair-tagged rows": repairs,
        "Replay-evaluated validated scenarios": replay_validated,
        "Distinct validated families": families,
        "Distinct validated assets": asset_count,
        "Distinct validated signals": signal_count,
    }
    numerically_consistent = all(parse_int(text, label) == value for label, value in checks.items())
    content_verified = "Only accepted/validated scenarios are counted as final usable coverage." in text and "Under-covered categories" in text
    backed = True
    issue = "" if numerically_consistent else "One or more summary counts do not match the master inventory."
    return content_verified, backed, numerically_consistent, issue


def verify_phase2_diversity_report(path: Path) -> tuple[bool, bool, bool, str]:
    inventory = load_inventory()
    accepted = inventory[inventory["accepted_rejected"] == "accepted"].copy()
    text = read_text(path)
    acceptance_rate = round(float((inventory["accepted_rejected"] == "accepted").mean()), 3)
    replay_rate = round(float(accepted["replay_evaluated_bool"].mean()), 3)
    family_balance = round(
        normalized_entropy(list(accepted["attack_family"].dropna().astype(str))),
        3,
    )
    asset_balance = round(
        normalized_entropy([item for value in accepted["affected_assets"] for item in parse_sequence(value)]),
        3,
    )
    signal_balance = round(
        normalized_entropy([item for value in accepted["affected_signals"] for item in parse_sequence(value)]),
        3,
    )
    overall_completeness = round(float(inventory["metadata_completeness_ratio"].mean()), 3)
    accepted_completeness = round(float(accepted["metadata_completeness_ratio"].mean()), 3)
    checks = {
        "Overall acceptance rate": acceptance_rate,
        "Validated rows with replay evidence": replay_rate,
        "Validated families": int(accepted["attack_family"].nunique()),
        "Validated assets": len({item for value in accepted["affected_assets"] for item in parse_sequence(value)}),
        "Validated signals": len({item for value in accepted["affected_signals"] for item in parse_sequence(value)}),
        "Validated source bundles": int(accepted["source_bundle"].nunique()),
        "Mean completeness over all inventory rows": overall_completeness,
        "Mean completeness over validated rows": accepted_completeness,
        "Family balance score (normalized entropy)": family_balance,
        "Asset balance score (normalized entropy)": asset_balance,
        "Signal balance score (normalized entropy)": signal_balance,
    }
    numerically_consistent = True
    for label, value in checks.items():
        parsed = parse_float(text, label)
        if parsed is None or round(parsed, 3) != round(float(value), 3):
            numerically_consistent = False
            break
    content_verified = "This quality package is intentionally conservative" in text and "Higher balance scores indicate more even spread" in text
    backed = True
    issue = "" if numerically_consistent else "Quality report values do not match the inventory-derived metrics."
    return content_verified, backed, numerically_consistent, issue


def verify_lora_manifest(path: Path) -> tuple[bool, bool, bool, str]:
    manifest = json.loads(read_text(path))
    required_keys = {"branch_role", "base_model", "total_examples", "split_counts"}
    content_verified = required_keys <= set(manifest.keys()) and manifest.get("branch_role") == "experimental_extension_only"
    split_total = sum(int(item["examples"]) for item in manifest.get("split_counts", []))
    case_files = sorted((ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models").glob("*/*/cases/*.json"))
    if not case_files:
        case_files = sorted((ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models").glob("*/xai_v4/cases/*.json"))
    backed = manifest.get("total_examples") == split_total == 155
    numerically_consistent = split_total == int(manifest.get("total_examples", 0))
    issue_parts = []
    if not content_verified:
        issue_parts.append("LoRA manifest is missing required keys or branch role.")
    if not numerically_consistent:
        issue_parts.append("Split counts do not sum to total_examples.")
    if not backed:
        issue_parts.append("Manifest totals do not match the expected heldout case count of 155.")
    return content_verified, backed, numerically_consistent, " ".join(issue_parts)


def verify_lora_config(path: Path) -> tuple[bool, bool, bool, str]:
    config = yaml.safe_load(read_text(path))
    required_paths = [
        config.get("base_model") == "google/flan-t5-small",
        config.get("lora", {}).get("rank") == 8,
        config.get("training", {}).get("epochs") == 5,
        config.get("splits", {}).get("validation_generator") == "gemini",
        config.get("splits", {}).get("test_generator") == "grok",
    ]
    content_verified = all(required_paths)
    backed = True
    numerically_consistent = True
    issue = "" if content_verified else "LoRA training config is missing one or more expected settings."
    return content_verified, backed, numerically_consistent, issue


def verify_lora_results(path: Path) -> tuple[bool, bool, bool, str]:
    results = pd.read_csv(path)
    expected_variants = {"base_zero_shot", "lora_finetuned"}
    expected_splits = {"aux_in_domain_holdout", "validation_generator_heldout", "test_generator_heldout"}
    predictions_dir = ROOT / "outputs" / "phase3_lora_extension" / "predictions"
    prediction_files_exist = (predictions_dir / "base_zero_shot_predictions.csv").exists() and (predictions_dir / "lora_finetuned_predictions.csv").exists()
    content_verified = (
        set(results["model_variant"]) == expected_variants
        and set(results["split"]) == expected_splits
        and len(results) == 6
    )
    backed = prediction_files_exist and (ROOT / "outputs" / "phase3_lora_extension" / "training_info.json").exists()
    numerically_consistent = results[["family_accuracy", "asset_accuracy", "evidence_grounding_quality", "confidence_accuracy", "parse_rate"]].apply(lambda col: col.between(0, 1).all()).all()
    issue_parts = []
    if not content_verified:
        issue_parts.append("Unexpected LoRA results rows or split/variant labels.")
    if not backed:
        issue_parts.append("LoRA predictions or training_info backing files are missing.")
    if not numerically_consistent:
        issue_parts.append("LoRA metric values are outside [0,1].")
    return content_verified, backed, numerically_consistent, " ".join(issue_parts)


def verify_lora_eval_report(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    manifest = json.loads(read_text(ROOT / "phase3_lora_dataset_manifest.json"))
    results = pd.read_csv(ROOT / "phase3_lora_results.csv")
    training_info = json.loads(read_text(ROOT / "outputs" / "phase3_lora_extension" / "training_info.json"))
    test_row = results[(results["model_variant"] == "lora_finetuned") & (results["split"] == "test_generator_heldout")].iloc[0]
    numerically_consistent = (
        parse_int(text, "Training examples") == manifest["split_counts"][2]["examples"]
        and parse_int(text, "Auxiliary in-domain holdout examples") == manifest["split_counts"][0]["examples"]
        and parse_int(text, "Validation generator-heldout examples") == manifest["split_counts"][3]["examples"]
        and parse_int(text, "Test generator-heldout examples") == manifest["split_counts"][1]["examples"]
        and parse_int(text, "Trainable parameters") == training_info["trainable_params"]
        and round(parse_float(text, "Adapter size on disk (MB)") or -1.0, 2) == round(float(training_info["adapter_size_mb"]), 2)
        and f"{test_row['family_accuracy']:.6f}" in text
    )
    content_verified = "extension experiment only" in text.lower() and "does not replace the canonical benchmark-selected transformer detector" in text.lower()
    backed = True
    issue = "" if numerically_consistent else "LoRA evaluation report bullets/table do not match manifest/results/training_info."
    return content_verified, backed, numerically_consistent, issue


def verify_lora_model_card(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    manifest = json.loads(read_text(ROOT / "phase3_lora_dataset_manifest.json"))
    content_verified = (
        "Experimental extension branch" in text
        and "`google/flan-t5-small`" in text
        and "Primary anomaly detection" in text
        and "Human-level root-cause analysis claims" in text
    )
    backed = all(generator in text for generator in manifest["training_generators"])
    numerically_consistent = True
    issue = "" if content_verified else "LoRA model card is missing required experimental/limitation wording."
    return content_verified, backed, numerically_consistent, issue


def verify_xai_case_level(path: Path) -> tuple[bool, bool, bool, str]:
    frame = pd.read_csv(path)
    required = {
        "generator_source",
        "detector",
        "scenario_id",
        "target_family",
        "predicted_family",
        "family_exact_match",
        "asset_accuracy",
        "evidence_grounding_overlap",
        "action_relevance_score",
        "partial_alignment_score",
        "primary_error_category",
        "case_path",
    }
    missing = sorted(required - set(frame.columns))
    case_paths_exist = frame["case_path"].apply(lambda value: (ROOT / str(value)).exists()).all()
    content_verified = not missing and len(frame) == 155
    backed = case_paths_exist
    numerically_consistent = frame[["family_exact_match", "family_top3_match", "asset_accuracy", "evidence_grounding_overlap", "action_relevance_score", "partial_alignment_score"]].apply(
        lambda col: col.between(0, 1).all()
    ).all()
    issue_parts = []
    if missing:
        issue_parts.append(f"Missing columns: {missing}")
    if not case_paths_exist:
        issue_parts.append("At least one XAI case_path is missing.")
    if not numerically_consistent:
        issue_parts.append("One or more XAI metrics are outside [0,1].")
    return content_verified, backed, numerically_consistent, " ".join(issue_parts)


def verify_xai_error_taxonomy(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    frame = pd.read_csv(ROOT / "xai_case_level_audit.csv")
    overall = frame["primary_error_category"].value_counts().sort_index()
    numerically_consistent = True
    for category, count in overall.items():
        if not re.search(rf"{re.escape(category)}\s+{count}\b", text):
            numerically_consistent = False
            break
    content_verified = "The taxonomy below is computed from the heldout v4 case packets" in text and "Reading guide" in text
    backed = True
    issue = "" if numerically_consistent else "XAI error taxonomy counts do not match xai_case_level_audit.csv."
    return content_verified, backed, numerically_consistent, issue


def verify_xai_qualitative(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    frame = pd.read_csv(ROOT / "xai_case_level_audit.csv")
    required_examples = [
        ("fully aligned", "scn_01_fdi_feeder_power_ramp"),
        ("correct family / wrong asset", "scn_oscillatory_control_capacitor_c83_state"),
        ("correct family / partial evidence", "scn_03_suppress_bess48_target"),
        ("wrong family / grounded evidence", "scn_command_delay_bess48_target_kw"),
        ("unsupported claim", "scn_coord_pv60_bess48_misaligned_support"),
    ]
    content_verified = True
    for category, scenario_id in required_examples:
        match = frame[(frame["primary_error_category"] == category) & (frame["scenario_id"] == scenario_id)]
        if match.empty or scenario_id not in text:
            content_verified = False
            break
    backed = True
    numerically_consistent = True
    issue = "" if content_verified else "Qualitative XAI examples do not line up with the audited case set."
    return content_verified, backed, numerically_consistent, issue


def verify_xai_final_report(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    frame = pd.read_csv(ROOT / "xai_case_level_audit.csv")
    checks = {
        "Overall family accuracy": round(float(frame["family_exact_match"].mean()), 3),
        "Overall asset accuracy": round(float(frame["asset_accuracy"].mean()), 3),
        "Overall evidence grounding rate": round(float(frame["evidence_grounding_overlap"].mean()), 3),
        "Overall action relevance": round(float(frame["action_relevance_score"].mean()), 3),
        "Unsupported claim rate": round(float(frame["unsupported_claim"].mean()), 3),
    }
    numerically_consistent = True
    for label, value in checks.items():
        parsed = parse_float(text, label)
        if parsed is None or round(parsed, 3) != value:
            numerically_consistent = False
            break
    content_verified = "safe interpretation remains conservative" in text.lower() and "operator-facing support" in text.lower()
    backed = True
    issue = "" if numerically_consistent else "XAI final validation summary values do not match xai_case_level_audit.csv."
    return content_verified, backed, numerically_consistent, issue


def verify_deployment_results(path: Path) -> tuple[bool, bool, bool, str]:
    frame = pd.read_csv(path)
    expected_models = {
        ("workstation_cpu", "threshold_baseline", "10s"),
        ("workstation_cpu", "transformer", "60s"),
        ("workstation_cpu", "lstm", "300s"),
        ("constrained_single_thread", "threshold_baseline", "10s"),
        ("constrained_single_thread", "transformer", "60s"),
        ("constrained_single_thread", "lstm", "300s"),
    }
    actual_models = set(frame[["profile", "model_name", "window_label"]].itertuples(index=False, name=None))
    ready_packages_exist = frame["ready_package_dir"].apply(lambda value: Path(value).exists()).all()
    replay_paths_exist = frame["replay_window_paths"].apply(
        lambda value: all(Path(item).exists() for item in str(value).split(";") if item.strip())
    ).all()
    content_verified = actual_models == expected_models and len(frame) == 6
    backed = ready_packages_exist and replay_paths_exist
    numerically_consistent = (
        (frame["scored_window_count"] > 0).all()
        and (frame["benchmarked_window_count"] >= frame["scored_window_count"]).all()
        and (frame["model_package_size_mb"] > 0).all()
        and frame["replay_f1"].between(0, 1).all()
    )
    issue_parts = []
    if not content_verified:
        issue_parts.append("Unexpected deployment benchmark rows.")
    if not ready_packages_exist:
        issue_parts.append("A ready package path is missing.")
    if not replay_paths_exist:
        issue_parts.append("At least one deployment replay source path is missing.")
    if not numerically_consistent:
        issue_parts.append("Deployment metrics are internally inconsistent.")
    return content_verified, backed, numerically_consistent, " ".join(issue_parts)


def verify_deployment_report(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    results = pd.read_csv(ROOT / "deployment_benchmark_results.csv")
    actual = results[results["profile"] == "workstation_cpu"].copy()
    fastest = actual.sort_values("mean_cpu_inference_ms_per_window").iloc[0]
    strongest = actual.sort_values("replay_f1", ascending=False).iloc[0]
    smallest = actual.sort_values("model_package_size_mb").iloc[0]
    numerically_consistent = (
        f"`{fastest['model_name']}` at `{fastest['window_label']}` with {fastest['mean_cpu_inference_ms_per_window']:.3f} ms/window." in text
        and f"`{strongest['model_name']}` at `{strongest['window_label']}` with F1={strongest['replay_f1']:.3f}." in text
        and f"`{smallest['model_name']}` at `{smallest['window_label']}` with {smallest['model_package_size_mb']:.2f} MB." in text
    )
    content_verified = "offline lightweight deployment benchmark" in text.lower() and "No real edge hardware was used." in text
    backed = True
    issue = "" if numerically_consistent else "Deployment report interpretation does not match deployment_benchmark_results.csv."
    return content_verified, backed, numerically_consistent, issue


def verify_deployment_readiness(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    content_verified = "No real edge device was used." in text and "offline lightweight deployment benchmark" in text.lower()
    backed = True
    numerically_consistent = True
    issue = "" if content_verified else "Deployment readiness summary does not clearly restrict claims to offline benchmarking."
    return content_verified, backed, numerically_consistent, issue


def verify_final_alignment(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    audit_csv = ROOT / "diagram_box_audit.csv"
    audit_exists = audit_csv.exists()
    row_count_ok = False
    if audit_exists:
        audit_frame = pd.read_csv(audit_csv)
        row_count_ok = len(audit_frame) >= 20 and {
            "diagram_box",
            "implementation_status",
            "label_safety",
            "safe_label",
        }.issubset(audit_frame.columns)
    content_verified = (
        "Implementation status" in text
        and "Label safety" in text
        and "Grounded explanation layer for post-alert operator support" in text
        and "Future-work pressure" in text
        and row_count_ok
    )
    backed = audit_exists and (ROOT / "phase3_lora_results.csv").exists() and (ROOT / "deployment_benchmark_results.csv").exists()
    numerically_consistent = True
    issue = ""
    if not content_verified:
        issue = "Alignment table does not fully classify every relevant diagram box or explicitly include future-work labeling."
    return content_verified, backed, numerically_consistent, issue


def verify_safe_labels(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    required_phrases = [
        "Model Training ML Models + Tiny LLM (LoRA)",
        "Human-like Explanation LLM Generates Root Cause Analysis",
        "Deployment Vision Lightweight Edge AI for DER",
    ]
    content_verified = all(phrase in text for phrase in required_phrases) and text.count("|") >= 12
    backed = True
    numerically_consistent = True
    issue = "" if content_verified else "Safe-label patch file does not cover all known overclaiming boxes."
    return content_verified, backed, numerically_consistent, issue


def verify_complete_project_status(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    content_verified = (
        "transformer @ 60s" in text
        and "LoRA" in text
        and "offline" in text.lower()
        and "zero-day robustness" in text
    )
    backed = True
    numerically_consistent = True
    issue = "" if content_verified else "Project status report is missing one or more major status categories or safe-claim caveats."
    return content_verified, backed, numerically_consistent, issue


def verify_complete_results(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    benchmark = pd.read_csv(ROOT / "outputs" / "window_size_study" / "final_window_comparison.csv")
    canonical = benchmark[(benchmark["model_name"] == "transformer") & (benchmark["window_label"] == "60s")].iloc[0]
    heldout = pd.read_csv(ROOT / "heldout_bundle_metrics.csv")
    balanced = pd.read_csv(ROOT / "balanced_heldout_metrics.csv")
    inventory = load_inventory()
    lora_results = pd.read_csv(ROOT / "phase3_lora_results.csv")
    xai_frame = pd.read_csv(ROOT / "xai_case_level_audit.csv")
    deployment = pd.read_csv(ROOT / "deployment_benchmark_results.csv")
    asset_signal = pd.read_csv(ROOT / "phase2_asset_signal_coverage.csv")
    test_row = lora_results[(lora_results["model_variant"] == "lora_finetuned") & (lora_results["split"] == "test_generator_heldout")].iloc[0]
    deployment_actual = deployment[deployment["profile"] == "workstation_cpu"][["model_name", "window_label", "replay_f1"]]
    mean_original = heldout[heldout["generator_source"] != "canonical_bundle"][["precision", "recall", "f1"]].mean()
    mean_balanced = balanced[["precision", "recall", "f1"]].mean()
    validated_asset_count = int(
        asset_signal.loc[(asset_signal["dimension"] == "asset") & (asset_signal["validated_scenarios"] > 0), "name"].nunique()
    )
    validated_signal_count = int(
        asset_signal.loc[(asset_signal["dimension"] == "signal") & (asset_signal["validated_scenarios"] > 0), "name"].nunique()
    )
    checks = [
        f"benchmark F1 `{canonical['f1']:.6f}`" in text,
        f"`{mean_original['precision']:.3f}` / `{mean_original['recall']:.3f}` / `{mean_original['f1']:.3f}`" in text,
        f"`{mean_balanced['precision']:.3f}` / `{mean_balanced['recall']:.3f}` / `{mean_balanced['f1']:.3f}`" in text,
        f"`{len(inventory)}`" in text,
        f"`{int((inventory['accepted_rejected'] == 'accepted').sum())}`" in text,
        f"`{validated_asset_count}`" in text,
        f"`{validated_signal_count}`" in text,
        f"`family_accuracy={test_row['family_accuracy']:.3f}`" in text,
        f"`asset_accuracy={test_row['asset_accuracy']:.3f}`" in text,
        f"`grounding={test_row['evidence_grounding_quality']:.3f}`" in text,
        f"`{xai_frame['family_exact_match'].mean():.3f}`" in text,
        f"`{xai_frame['asset_accuracy'].mean():.3f}`" in text,
        f"`{xai_frame['evidence_grounding_overlap'].mean():.3f}`" in text,
        f"`{xai_frame['action_relevance_score'].mean():.3f}`" in text,
    ]
    checks.append(all(f"{row['model_name']}" in text and f"{row['replay_f1']:.6f}" in text for _, row in deployment_actual.iterrows()))
    numerically_consistent = all(checks)
    content_verified = "canonical benchmark" in text.lower() and "heldout replay" in text.lower() and "deployment benchmark" in text.lower()
    backed = True
    issue = "" if numerically_consistent else "Results/discussion report does not fully match source benchmark/replay/LoRA/XAI/deployment metrics."
    return content_verified, backed, numerically_consistent, issue


def verify_publication_safe_claims(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    content_verified = (
        "Safe now" in text
        and "Safe with wording changes" in text
        and "Not safe" in text
        and "Missing evidence" in text
        and "offline lightweight deployment benchmark" in text
    )
    backed = True
    numerically_consistent = True
    issue = "" if content_verified else "Publication safe-claims report is missing required sections or deployment/LoRA caveats."
    return content_verified, backed, numerically_consistent, issue


def verify_final_project_decision(path: Path) -> tuple[bool, bool, bool, str]:
    text = read_text(path)
    content_verified = (
        "Mostly" in text
        and "Boxes that are truly satisfied" in text
        and "Boxes that remain partial" in text
        and "What can be said tomorrow" in text
        and "What is safe in the paper" in text
        and "offline workstation replay benchmarking" in text
    )
    backed = True
    numerically_consistent = True
    issue = "" if content_verified else "Final project decision does not clearly separate satisfied, partial, and unsafe claims."
    return content_verified, backed, numerically_consistent, issue


def verify_figure(path: Path) -> tuple[bool, bool, bool, str]:
    structurally_valid, content_verified, issue = verify_image(path)
    backed = True
    numerically_consistent = True
    return structurally_valid, backed, numerically_consistent, issue if issue else ""


def verify_required_artifact(path: Path) -> VerificationResult:
    exists = path.exists()
    nonempty = exists and path.stat().st_size > 0
    if not exists:
        return VerificationResult(path.name, True, False, False, False, False, False, False, "fail", "Required artifact is missing.")
    if not nonempty:
        return VerificationResult(rel_path(path), True, True, False, False, False, False, False, "fail", "Required artifact is empty.")

    structurally_valid, structural_issue = verify_basic_structure(path)
    content_verified = False
    backed = False
    numerically_consistent = False
    issue = structural_issue

    if path.name == "CANONICAL_ARTIFACT_FREEZE.md":
        content_verified, backed, numerically_consistent, issue = verify_freeze_report(path)
    elif path.name == "phase2_scenario_master_inventory.csv":
        content_verified, backed, numerically_consistent, issue = verify_phase2_inventory(path)
    elif path.name == "phase2_attack_family_distribution.csv":
        content_verified, backed, numerically_consistent, issue = verify_phase2_family_distribution(path)
    elif path.name == "phase2_asset_signal_coverage.csv":
        content_verified, backed, numerically_consistent, issue = verify_phase2_asset_signal(path)
    elif path.name == "phase2_difficulty_calibration.csv":
        content_verified, backed, numerically_consistent, issue = verify_phase2_difficulty(path)
    elif path.name == "phase2_coverage_summary.md":
        content_verified, backed, numerically_consistent, issue = verify_phase2_coverage_summary(path)
    elif path.name == "phase2_diversity_and_quality_report.md":
        content_verified, backed, numerically_consistent, issue = verify_phase2_diversity_report(path)
    elif path.name in {
        "phase2_family_distribution.png",
        "phase2_asset_coverage_heatmap.png",
        "phase2_signal_coverage_heatmap.png",
        "phase2_difficulty_distribution.png",
        "phase2_generator_coverage_comparison.png",
        "phase3_lora_family_accuracy.png",
        "phase3_lora_asset_accuracy.png",
        "phase3_lora_grounding_comparison.png",
        "phase3_lora_latency_vs_quality.png",
        "xai_family_vs_asset_accuracy.png",
        "xai_error_taxonomy_breakdown.png",
        "xai_grounding_vs_family_accuracy.png",
        "deployment_latency_by_model.png",
        "deployment_memory_by_model.png",
        "deployment_throughput_by_model.png",
        "deployment_accuracy_latency_tradeoff.png",
    }:
        structurally_valid, content_verified, issue = verify_image(path)
        backed = True
        numerically_consistent = True
    elif path.name == "phase3_lora_dataset_manifest.json":
        content_verified, backed, numerically_consistent, issue = verify_lora_manifest(path)
    elif path.name == "phase3_lora_training_config.yaml":
        content_verified, backed, numerically_consistent, issue = verify_lora_config(path)
    elif path.name == "phase3_lora_results.csv":
        content_verified, backed, numerically_consistent, issue = verify_lora_results(path)
    elif path.name == "phase3_lora_eval_report.md":
        content_verified, backed, numerically_consistent, issue = verify_lora_eval_report(path)
    elif path.name == "phase3_lora_model_card.md":
        content_verified, backed, numerically_consistent, issue = verify_lora_model_card(path)
    elif path.name == "xai_case_level_audit.csv":
        content_verified, backed, numerically_consistent, issue = verify_xai_case_level(path)
    elif path.name == "xai_error_taxonomy.md":
        content_verified, backed, numerically_consistent, issue = verify_xai_error_taxonomy(path)
    elif path.name == "xai_qualitative_examples.md":
        content_verified, backed, numerically_consistent, issue = verify_xai_qualitative(path)
    elif path.name == "xai_final_validation_report.md":
        content_verified, backed, numerically_consistent, issue = verify_xai_final_report(path)
    elif path.name == "deployment_benchmark_results.csv":
        content_verified, backed, numerically_consistent, issue = verify_deployment_results(path)
    elif path.name == "deployment_benchmark_report.md":
        content_verified, backed, numerically_consistent, issue = verify_deployment_report(path)
    elif path.name == "deployment_readiness_summary.md":
        content_verified, backed, numerically_consistent, issue = verify_deployment_readiness(path)
    elif path.name == "FINAL_DIAGRAM_ALIGNMENT.md":
        content_verified, backed, numerically_consistent, issue = verify_final_alignment(path)
    elif path.name == "DIAGRAM_SAFE_LABELS.md":
        content_verified, backed, numerically_consistent, issue = verify_safe_labels(path)
    elif path.name == "COMPLETE_PROJECT_STATUS.md":
        content_verified, backed, numerically_consistent, issue = verify_complete_project_status(path)
    elif path.name == "COMPLETE_RESULTS_AND_DISCUSSION.md":
        content_verified, backed, numerically_consistent, issue = verify_complete_results(path)
    elif path.name == "COMPLETE_PUBLICATION_SAFE_CLAIMS.md":
        content_verified, backed, numerically_consistent, issue = verify_publication_safe_claims(path)
    elif path.name == "FINAL_PROJECT_DECISION.md":
        content_verified, backed, numerically_consistent, issue = verify_final_project_decision(path)
    else:
        content_verified = structurally_valid
        backed = structurally_valid
        numerically_consistent = True

    pass_fail = "pass" if all([exists, nonempty, structurally_valid, content_verified, backed, numerically_consistent]) else "fail"
    exact_issue = issue if issue else ""
    return VerificationResult(
        artifact_path=rel_path(path),
        required=True,
        exists=exists,
        nonempty=nonempty,
        structurally_valid=structurally_valid,
        content_verified=content_verified,
        backed_by_source_data=backed,
        numerically_consistent=numerically_consistent,
        pass_fail=pass_fail,
        exact_issue=exact_issue,
    )


def build_verification_audit() -> pd.DataFrame:
    results = [verify_required_artifact(ROOT / artifact) for artifact in REQUIRED_ARTIFACTS]
    frame = pd.DataFrame([result.__dict__ for result in results])
    frame.to_csv(VERIFICATION_AUDIT_PATH, index=False)
    return frame


def detect_ttm_artifacts(paths: list[Path]) -> list[str]:
    hits = []
    for path in paths:
        target = rel_path(path) if path.exists() else path.name
        lowered = target.lower()
        if "ttm" in lowered or "tinytimemixer" in lowered or "tiny_time_mixer" in lowered:
            hits.append(target)
    return sorted(set(hits))


def write_verification_gaps(audit: pd.DataFrame, paths: list[Path]) -> None:
    failed = audit[audit["pass_fail"] != "pass"].copy()
    ttm_hits = detect_ttm_artifacts(paths)
    lines = ["# Verification Gaps", ""]
    if failed.empty and ttm_hits:
        lines.append("No required artifact from Pass 1 failed hard verification.")
    elif failed.empty:
        lines.append("All previously claimed required artifacts passed the current Pass 1 checks, but repo-level gaps still remain.")
    lines.append("")

    def add_section(title: str, rows: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if rows.empty:
            lines.append("- None")
        else:
            for _, row in rows.iterrows():
                lines.append(f"- `{row['artifact_path']}`: {row['exact_issue'] or 'Verification failed.'}")
        lines.append("")

    blocker_mask = failed["exists"].eq(False) | failed["nonempty"].eq(False) | failed["structurally_valid"].eq(False)
    major_mask = (~blocker_mask) & (failed["content_verified"].eq(False) | failed["backed_by_source_data"].eq(False) | failed["numerically_consistent"].eq(False))

    add_section("Blocker", failed[blocker_mask])
    add_section("Major", failed[major_mask])

    minor_rows = []
    if not ttm_hits:
        minor_rows.append("No TTM code, config, checkpoint, or result artifact was found during Pass 0 discovery. Pass 3 will need a fresh extension implementation.")
    asset_signal = pd.read_csv(ROOT / "phase2_asset_signal_coverage.csv")
    if ((asset_signal["validated_scenarios"] + asset_signal["rejected_scenarios"]) > asset_signal["all_scenarios"]).any():
        minor_rows.append(
            "`phase2_asset_signal_coverage.csv` mixes unique-scenario coverage counts with repair row counts; the semantics are defensible but not explicit in the file itself."
        )
    lines.append("## Minor")
    lines.append("")
    if minor_rows:
        for item in minor_rows:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")
    VERIFICATION_GAPS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    relevant_paths = collect_relevant_files()
    write_tree(relevant_paths)
    build_evidence_index(relevant_paths)
    audit = build_verification_audit()
    write_verification_gaps(audit, relevant_paths)
    print(
        json.dumps(
            {
                "tree": rel_path(COMPLETE_TREE_PATH),
                "evidence_index": rel_path(EVIDENCE_INDEX_PATH),
                "verification_audit": rel_path(VERIFICATION_AUDIT_PATH),
                "verification_gaps": rel_path(VERIFICATION_GAPS_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
