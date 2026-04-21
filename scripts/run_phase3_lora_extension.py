from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methodology_alignment_common import (
    GENERATOR_SOURCES,
    ROOT,
    action_relevance_score,
    asset_match,
    evidence_overlap,
    family_match,
    load_generator_truth_lookup,
    normalize_signal,
    partial_alignment_score,
    pretty_label,
    read_json,
    write_json,
    write_markdown,
)


BASE_MODEL_NAME = "google/flan-t5-small"
TRAIN_GENERATORS = ["chatgpt", "claude"]
VALIDATION_GENERATOR = "gemini"
TEST_GENERATOR = "grok"
IN_DOMAIN_HOLDOUT_RATIO = 0.2

EXPERIMENT_ROOT = ROOT / "outputs" / "phase3_lora_extension"
ADAPTER_ROOT = EXPERIMENT_ROOT / "adapter"
TOKENIZER_ROOT = EXPERIMENT_ROOT / "tokenizer"
PREDICTIONS_ROOT = EXPERIMENT_ROOT / "predictions"
TRAINING_INFO_PATH = EXPERIMENT_ROOT / "training_info.json"

ROOT_OUTPUTS = {
    "manifest": ROOT / "phase3_lora_dataset_manifest.json",
    "config": ROOT / "phase3_lora_training_config.yaml",
    "results": ROOT / "phase3_lora_results.csv",
    "report": ROOT / "phase3_lora_eval_report.md",
    "model_card": ROOT / "phase3_lora_model_card.md",
    "run_log": ROOT / "phase3_lora_run_log.txt",
    "evidence_checklist": ROOT / "phase3_lora_evidence_checklist.md",
    "family_fig": ROOT / "phase3_lora_family_accuracy.png",
    "asset_fig": ROOT / "phase3_lora_asset_accuracy.png",
    "grounding_fig": ROOT / "phase3_lora_grounding_comparison.png",
    "latency_fig": ROOT / "phase3_lora_latency_vs_quality.png",
}


@dataclass(slots=True)
class Example:
    example_id: str
    split: str
    generator_source: str
    detector: str
    scenario_id: str
    target_family: str
    target_assets: list[str]
    target_signals: list[str]
    target_confidence: str
    input_text: str
    target_text: str


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _ensure_dependencies() -> None:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing LoRA dependency. Install `peft` before running this script."
        ) from exc


def _family_scores_sorted(packet: dict[str, Any]) -> list[tuple[str, float]]:
    family_scores = packet.get("family_scores", {}) or {}
    items = [(str(key), float(value)) for key, value in family_scores.items()]
    return sorted(items, key=lambda item: item[1], reverse=True)


def _confidence_bucket(packet: dict[str, Any], truth_family: str) -> str:
    ranked = _family_scores_sorted(packet)
    if not ranked:
        return "medium"
    score_map = dict(ranked)
    top_family, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    truth_score = float(score_map.get(truth_family, 0.0))
    if top_family == truth_family and (top_score - second_score) >= 0.25:
        return "high"
    if truth_family in {name for name, _ in ranked[:3]} or truth_score >= 0.20:
        return "medium"
    return "low"


def _packet_signal_refs(packet: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for item in packet.get("grounded_evidence_items", []):
        for signal in item.get("signal_refs", []):
            value = str(signal).strip()
            if value and value not in refs:
                refs.append(value)
    for signal in packet.get("strongest_physical_signals", []):
        value = str(signal).strip()
        if value and value not in refs:
            refs.append(value)
    for signal in packet.get("strongest_cyber_signals", []):
        value = str(signal).strip()
        if value and value not in refs:
            refs.append(value)
    return refs


def _evidence_targets(packet: dict[str, Any], truth_signals: list[str]) -> list[str]:
    packet_signals = {normalize_signal(item): item for item in _packet_signal_refs(packet)}
    selected: list[str] = []
    for signal in truth_signals:
        normalized = normalize_signal(signal)
        if normalized in packet_signals and packet_signals[normalized] not in selected:
            selected.append(packet_signals[normalized])
    if not selected:
        for signal in truth_signals:
            if signal not in selected:
                selected.append(signal)
            if len(selected) >= 3:
                break
    return selected[:3]


def _build_input_text(generator_source: str, detector: str, packet: dict[str, Any]) -> str:
    family_rank = ", ".join(f"{name}:{score:.3f}" for name, score in _family_scores_sorted(packet)[:4])
    asset_scores = packet.get("asset_scores", {}) or {}
    ranked_assets = sorted(((str(key), float(value)) for key, value in asset_scores.items()), key=lambda item: item[1], reverse=True)
    asset_rank = ", ".join(f"{name}:{score:.3f}" for name, score in ranked_assets[:4])
    grounded_items = []
    for item in packet.get("grounded_evidence_items", [])[:4]:
        description = str(item.get("description", "")).strip()
        if description:
            grounded_items.append(description)
    evidence_lines = "\n".join(f"- {line}" for line in grounded_items)
    strongest_physical = ", ".join(str(item) for item in packet.get("strongest_physical_signals", [])[:4])
    strongest_cyber = ", ".join(str(item) for item in packet.get("strongest_cyber_signals", [])[:4])
    return (
        "Task: produce a compact grounded DER anomaly explanation as JSON.\n"
        f"Source generator: {generator_source}\n"
        f"Detector: {detector}\n"
        f"Window start: {packet.get('window_start_utc', '')}\n"
        f"Window end: {packet.get('window_end_utc', '')}\n"
        f"Detector score: {float(packet.get('detector_score', 0.0)):.6f}\n"
        f"Detector threshold: {float(packet.get('detector_threshold', 0.0)):.6f}\n"
        f"Primary asset guess: {packet.get('primary_asset', '')}\n"
        f"Candidate family scores: {family_rank}\n"
        f"Candidate asset scores: {asset_rank}\n"
        f"Strongest physical signals: {strongest_physical}\n"
        f"Strongest cyber signals: {strongest_cyber}\n"
        "Grounded evidence items:\n"
        f"{evidence_lines}\n"
        "Return JSON with keys family, assets, confidence, evidence_signals, operator_summary."
    )


def _build_target_text(packet: dict[str, Any], truth_family: str, truth_assets: list[str], truth_signals: list[str]) -> str:
    evidence_signals = _evidence_targets(packet, truth_signals)
    confidence = _confidence_bucket(packet, truth_family)
    asset_label = ", ".join(truth_assets)
    signal_label = ", ".join(evidence_signals)
    summary = (
        f"Evidence is most consistent with {truth_family} affecting {asset_label}. "
        f"Grounded signals include {signal_label}. "
        f"Detector crossed threshold near {packet.get('onset_time', packet.get('window_end_utc', 'the alert window'))}."
    )
    payload = {
        "family": truth_family,
        "assets": truth_assets,
        "confidence": confidence,
        "evidence_signals": evidence_signals,
        "operator_summary": summary,
    }
    return json.dumps(payload, separators=(",", ":"))


def _holdout_scenarios(train_examples: list[dict[str, Any]]) -> set[tuple[str, str]]:
    grouped = {}
    for item in train_examples:
        key = (item["generator_source"], item["scenario_id"])
        grouped.setdefault(key, item["target_family"])
    keys = sorted(grouped)
    rng = random.Random(42)
    rng.shuffle(keys)
    holdout_count = max(1, int(round(len(keys) * IN_DOMAIN_HOLDOUT_RATIO)))
    return set(keys[:holdout_count])


def _load_case_examples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generator_source in GENERATOR_SOURCES:
        _, truth_lookup = load_generator_truth_lookup(generator_source)
        case_dir = ROOT / "phase2_llm_benchmark" / "new_respnse_result" / "models" / generator_source / "xai_v4" / "cases"
        for case_path in sorted(case_dir.glob("*.json")):
            payload = read_json(case_path)
            packet = payload["packet"]
            comparison = payload["comparison"]
            scenario_id = str(comparison["scenario_id"])
            truth = truth_lookup.get(scenario_id)
            if truth is None:
                continue
            truth_family = str(truth["attack_family"])
            truth_assets = [str(truth["target_asset"])]
            truth_assets.extend(str(item["target_asset"]) for item in truth.get("additional_targets", []) or [])
            truth_signals = [str(item) for item in truth.get("observable_signals", [])]
            input_text = _build_input_text(generator_source, str(payload["packet"]["detector"]), packet)
            target_text = _build_target_text(packet, truth_family, truth_assets, truth_signals)
            rows.append(
                {
                    "example_id": f"{generator_source}::{case_path.stem}",
                    "generator_source": generator_source,
                    "detector": str(packet["detector"]),
                    "scenario_id": scenario_id,
                    "target_family": truth_family,
                    "target_assets": truth_assets,
                    "target_signals": truth_signals,
                    "target_confidence": _confidence_bucket(packet, truth_family),
                    "input_text": input_text,
                    "target_text": target_text,
                }
            )
    return rows


def _split_examples(rows: list[dict[str, Any]]) -> list[Example]:
    train_candidates = [row for row in rows if row["generator_source"] in TRAIN_GENERATORS]
    holdout_keys = _holdout_scenarios(train_candidates)
    examples: list[Example] = []
    for row in rows:
        key = (row["generator_source"], row["scenario_id"])
        if row["generator_source"] == VALIDATION_GENERATOR:
            split = "validation_generator_heldout"
        elif row["generator_source"] == TEST_GENERATOR:
            split = "test_generator_heldout"
        elif key in holdout_keys:
            split = "aux_in_domain_holdout"
        else:
            split = "train"
        examples.append(Example(split=split, **row))
    return examples


def _manifest_payload(examples: list[Example]) -> dict[str, Any]:
    split_counts = (
        pd.DataFrame([asdict(example) for example in examples])
        .groupby("split")
        .agg(
            examples=("example_id", "count"),
            scenarios=("scenario_id", "nunique"),
            generators=("generator_source", lambda series: sorted(set(series))),
            detectors=("detector", lambda series: sorted(set(series))),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "branch_role": "experimental_extension_only",
        "base_model": BASE_MODEL_NAME,
        "training_generators": TRAIN_GENERATORS,
        "validation_generator": VALIDATION_GENERATOR,
        "test_generator": TEST_GENERATOR,
        "auxiliary_in_domain_holdout": {
            "enabled": True,
            "holdout_ratio": IN_DOMAIN_HOLDOUT_RATIO,
            "note": "Holdout is done at generator+scenario granularity inside the training generators to provide an in-domain explanation test set.",
        },
        "total_examples": len(examples),
        "split_counts": split_counts,
        "excluded_sources": [
            {
                "generator_source": "human_authored",
                "reason": "No aligned xai_v4 case packet set existed for the human-authored bundle, so it was excluded from supervised LoRA training and evaluation to avoid fabricating explanation labels.",
            }
        ],
    }


def _write_config_yaml() -> None:
    text = (
        "branch_role: experimental_extension_only\n"
        f"base_model: {BASE_MODEL_NAME}\n"
        "task: grounded_explanation_json\n"
        "tokenizer_max_input_length: 512\n"
        "tokenizer_max_target_length: 128\n"
        "lora:\n"
        "  task_type: SEQ_2_SEQ_LM\n"
        "  rank: 8\n"
        "  alpha: 16\n"
        "  dropout: 0.05\n"
        "  target_modules:\n"
        "    - q\n"
        "    - v\n"
        "training:\n"
        "  epochs: 5\n"
        "  train_batch_size: 4\n"
        "  gradient_accumulation_steps: 2\n"
        "  learning_rate: 0.0008\n"
        "  weight_decay: 0.01\n"
        "  seed: 42\n"
        "splits:\n"
        f"  train_generators: {TRAIN_GENERATORS}\n"
        f"  validation_generator: {VALIDATION_GENERATOR}\n"
        f"  test_generator: {TEST_GENERATOR}\n"
        f"  auxiliary_in_domain_holdout_ratio: {IN_DOMAIN_HOLDOUT_RATIO}\n"
    )
    ROOT_OUTPUTS["config"].write_text(text, encoding="utf-8")


def _batched(items: list[Example], batch_size: int, *, shuffle: bool) -> list[list[Example]]:
    values = list(items)
    if shuffle:
        rng = random.Random(42)
        rng.shuffle(values)
    return [values[idx : idx + batch_size] for idx in range(0, len(values), batch_size)]


def _encode_batch(tokenizer, batch: list[Example], input_max_length: int, target_max_length: int):
    import torch

    inputs = tokenizer(
        [item.input_text for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=input_max_length,
    )
    labels = tokenizer(
        [item.target_text for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=target_max_length,
    )
    label_ids = labels.input_ids
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    inputs["labels"] = label_ids
    return {key: value.to(torch.device("cpu")) for key, value in inputs.items()}


def _train_model(examples: list[Example]) -> tuple[Any, Any, dict[str, Any]]:
    _ensure_dependencies()
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(42)
    random.seed(42)

    train_examples = [item for item in examples if item.split == "train"]
    valid_examples = [item for item in examples if item.split == "validation_generator_heldout"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)

    best_state = None
    best_val_loss = math.inf
    history: list[dict[str, Any]] = []
    grad_accum_steps = 2

    for epoch in range(1, 6):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        step_counter = 0
        for batch in _batched(train_examples, batch_size=4, shuffle=True):
            encoded = _encode_batch(tokenizer, batch, input_max_length=512, target_max_length=128)
            outputs = model(**encoded)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            train_losses.append(float(outputs.loss.detach().cpu().item()))
            step_counter += 1
            if step_counter % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step_counter % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in _batched(valid_examples, batch_size=4, shuffle=False):
                encoded = _encode_batch(tokenizer, batch, input_max_length=512, target_max_length=128)
                outputs = model(**encoded)
                valid_losses.append(float(outputs.loss.detach().cpu().item()))
        epoch_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        epoch_valid_loss = float(np.mean(valid_losses)) if valid_losses else 0.0
        history.append({"epoch": epoch, "train_loss": epoch_train_loss, "validation_loss": epoch_valid_loss})
        if epoch_valid_loss < best_val_loss:
            best_val_loss = epoch_valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_ROOT)
    tokenizer.save_pretrained(TOKENIZER_ROOT)

    trainable_params = 0
    total_params = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_params += count
        if parameter.requires_grad:
            trainable_params += count

    training_info = {
        "history": history,
        "best_validation_loss": best_val_loss,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "adapter_size_mb": sum(path.stat().st_size for path in ADAPTER_ROOT.rglob("*") if path.is_file()) / (1024.0 * 1024.0),
        "model_memory_footprint_mb": model.get_memory_footprint() / (1024.0 * 1024.0),
    }
    write_json(TRAINING_INFO_PATH, training_info)
    return tokenizer, model, training_info


def _parse_generated_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]
    elif '"family"' in candidate:
        candidate = "{" + candidate.strip().strip(",") + "}"
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    family_match_obj = re.search(r'"family"\s*:\s*"([^"]+)"', candidate)
    assets_match_obj = re.search(r'"assets"\s*:\s*(\[[^\]]*\]|"[^"]+")', candidate)
    confidence_match_obj = re.search(r'"confidence"\s*:\s*"([^"]+)"', candidate)
    signals_match_obj = re.search(r'"evidence_signals"\s*:\s*(\[[^\]]*\]|"[^"]+")', candidate)
    lowered = text.lower()
    payload = {
        "family": "",
        "assets": [],
        "confidence": "",
        "evidence_signals": [],
        "operator_summary": text.strip(),
    }
    known_families = {
        "false_data_injection",
        "command_delay",
        "command_suppression",
        "DER_disconnect",
        "oscillatory_control",
        "coordinated_multi_asset",
    }
    if family_match_obj:
        family_candidate = family_match_obj.group(1).strip()
        close = difflib.get_close_matches(family_candidate, list(known_families), n=1, cutoff=0.6)
        payload["family"] = close[0] if close else family_candidate
    for family in known_families:
        if family.lower() in lowered:
            payload["family"] = family
            break
    if assets_match_obj:
        raw_assets = assets_match_obj.group(1).strip()
        if raw_assets.startswith("["):
            try:
                payload["assets"] = [str(item).strip() for item in json.loads(raw_assets) if str(item).strip()]
            except json.JSONDecodeError:
                payload["assets"] = []
        else:
            payload["assets"] = [raw_assets.strip('"').strip()]
    if confidence_match_obj:
        payload["confidence"] = confidence_match_obj.group(1).strip()
    if signals_match_obj:
        raw_signals = signals_match_obj.group(1).strip()
        if raw_signals.startswith("["):
            try:
                payload["evidence_signals"] = [str(item).strip() for item in json.loads(raw_signals) if str(item).strip()]
            except json.JSONDecodeError:
                payload["evidence_signals"] = []
        else:
            payload["evidence_signals"] = [raw_signals.strip('"').strip()]
    return payload


def _normalize_assets(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [piece.strip() for piece in text.split("|") if piece.strip()]
    return [str(value).strip()]


def _normalize_signals(value: Any) -> list[str]:
    return _normalize_assets(value)


def _evaluate_variant(variant_name: str, tokenizer, model, examples: list[Example]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    model.eval()
    rows = []
    prediction_rows = []
    for split_name in ["aux_in_domain_holdout", "validation_generator_heldout", "test_generator_heldout"]:
        split_examples = [item for item in examples if item.split == split_name]
        latencies = []
        family_hits = []
        asset_scores = []
        grounding_scores = []
        confidence_hits = []
        parse_hits = []
        for example in split_examples:
            encoded = tokenizer(
                example.input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=128,
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            parsed = _parse_generated_json(decoded)
            predicted_family = str(parsed.get("family", "")).strip()
            predicted_assets = set(_normalize_assets(parsed.get("assets")))
            predicted_signals = {normalize_signal(item) for item in _normalize_signals(parsed.get("evidence_signals"))}
            truth_assets = set(example.target_assets)
            truth_signals = {normalize_signal(item) for item in example.target_signals}
            family_hits.append(1.0 if predicted_family == example.target_family else 0.0)
            asset_scores.append(
                1.0 if predicted_assets == truth_assets and truth_assets else 0.5 if predicted_assets & truth_assets else 0.0
            )
            grounding_scores.append(evidence_overlap(truth_signals, predicted_signals, truth_assets, predicted_assets)[0])
            confidence_hits.append(1.0 if str(parsed.get("confidence", "")).strip() == example.target_confidence else 0.0)
            parse_hits.append(1.0 if isinstance(parsed, dict) and parsed else 0.0)
            prediction_rows.append(
                {
                    "variant": variant_name,
                    "split": split_name,
                    "example_id": example.example_id,
                    "generator_source": example.generator_source,
                    "detector": example.detector,
                    "scenario_id": example.scenario_id,
                    "target_family": example.target_family,
                    "predicted_family": predicted_family,
                    "target_assets": "|".join(example.target_assets),
                    "predicted_assets": "|".join(sorted(predicted_assets)),
                    "target_signals": "|".join(example.target_signals),
                    "predicted_signals": "|".join(sorted(predicted_signals)),
                    "target_confidence": example.target_confidence,
                    "predicted_confidence": str(parsed.get("confidence", "")).strip(),
                    "latency_ms": round(elapsed_ms, 3),
                    "raw_output": decoded,
                }
            )
        rows.append(
            {
                "model_variant": variant_name,
                "split": split_name,
                "example_count": len(split_examples),
                "family_accuracy": float(np.mean(family_hits)) if family_hits else 0.0,
                "asset_accuracy": float(np.mean(asset_scores)) if asset_scores else 0.0,
                "evidence_grounding_quality": float(np.mean(grounding_scores)) if grounding_scores else 0.0,
                "confidence_accuracy": float(np.mean(confidence_hits)) if confidence_hits else 0.0,
                "parse_rate": float(np.mean(parse_hits)) if parse_hits else 0.0,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def _load_base_model():
    _ensure_dependencies()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    return tokenizer, model


def _load_existing_lora_model() -> tuple[Any, Any, dict[str, Any]]:
    _ensure_dependencies()
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ROOT if TOKENIZER_ROOT.exists() else BASE_MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, ADAPTER_ROOT)
    if TRAINING_INFO_PATH.exists():
        training_info = read_json(TRAINING_INFO_PATH)
        if float(training_info.get("trainable_params", 0) or 0) <= 0:
            training_info = {}
    else:
        training_info = {}
    if not training_info:
        trainable_params = 0
        total_params = 0
        for name, parameter in model.named_parameters():
            count = parameter.numel()
            total_params += count
            if parameter.requires_grad or "lora_" in name:
                trainable_params += count
        training_info = {
            "history": [],
            "best_validation_loss": None,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "adapter_size_mb": sum(path.stat().st_size for path in ADAPTER_ROOT.rglob("*") if path.is_file()) / (1024.0 * 1024.0),
            "model_memory_footprint_mb": model.get_memory_footprint() / (1024.0 * 1024.0),
        }
        write_json(TRAINING_INFO_PATH, training_info)
    return tokenizer, model, training_info


def _plot_results(results_df: pd.DataFrame) -> None:
    split_labels = {
        "aux_in_domain_holdout": "In-domain holdout",
        "validation_generator_heldout": "Validation generator",
        "test_generator_heldout": "Test generator",
    }
    plotting = results_df.copy()
    plotting["split_label"] = plotting["split"].map(split_labels)
    split_order = ["In-domain holdout", "Validation generator", "Test generator"]
    variant_order = ["base_zero_shot", "lora_finetuned"]
    colors = {"base_zero_shot": "#999999", "lora_finetuned": "#1b9e77"}

    for metric, path, title in [
        ("family_accuracy", ROOT_OUTPUTS["family_fig"], "LoRA family accuracy"),
        ("asset_accuracy", ROOT_OUTPUTS["asset_fig"], "LoRA asset accuracy"),
        ("evidence_grounding_quality", ROOT_OUTPUTS["grounding_fig"], "LoRA grounding quality"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(split_order))
        width = 0.28
        for idx, variant in enumerate(variant_order):
            subset = plotting[plotting["model_variant"] == variant].set_index("split_label").reindex(split_order)
            ax.bar(
                x + (idx - 0.5) * width,
                subset[metric].fillna(0.0).to_numpy(),
                width=width,
                label=variant.replace("_", " "),
                color=colors[variant],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(split_order)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    latency_plot = plotting.copy()
    latency_plot["quality_score"] = (
        0.5 * latency_plot["family_accuracy"]
        + 0.25 * latency_plot["asset_accuracy"]
        + 0.25 * latency_plot["evidence_grounding_quality"]
    )
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for variant in variant_order:
        subset = latency_plot[latency_plot["model_variant"] == variant]
        ax.scatter(
            subset["mean_latency_ms"],
            subset["quality_score"],
            s=90,
            label=variant.replace("_", " "),
            color=colors[variant],
        )
        for _, row in subset.iterrows():
            ax.text(row["mean_latency_ms"] + 2, row["quality_score"] + 0.01, row["split_label"], fontsize=8)
    ax.set_xlabel("Mean generation latency (ms)")
    ax.set_ylabel("Composite quality")
    ax.set_title("LoRA latency vs quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT_OUTPUTS["latency_fig"], dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_reports(examples: list[Example], training_info: dict[str, Any], results_df: pd.DataFrame) -> None:
    history_df = pd.DataFrame(training_info.get("history", []))
    report_lines = [
        "# Phase 3 LoRA Experimental Branch",
        "",
        "This branch is an extension experiment only. It does not replace the canonical benchmark-selected transformer detector.",
        "",
        f"- Base model: `{BASE_MODEL_NAME}`",
        f"- Training examples: {sum(1 for item in examples if item.split == 'train')}",
        f"- Auxiliary in-domain holdout examples: {sum(1 for item in examples if item.split == 'aux_in_domain_holdout')}",
        f"- Validation generator-heldout examples: {sum(1 for item in examples if item.split == 'validation_generator_heldout')}",
        f"- Test generator-heldout examples: {sum(1 for item in examples if item.split == 'test_generator_heldout')}",
        f"- Trainable parameters: {training_info['trainable_params']}",
        f"- Total parameters in wrapped model: {training_info['total_params']}",
        f"- Adapter size on disk (MB): {training_info['adapter_size_mb']:.2f}",
        "",
        "## Results",
        "",
        "```text",
        results_df.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- Family attribution and asset prediction should be discussed as bounded explanation-side tasks, not as the main anomaly detector.",
        "- Grounding quality is measured against scenario observable-signal overlap, which is stronger than a free-form text score but still not a human-level reasoning claim.",
        "- The branch remains experimental even if LoRA improves over the untuned base model.",
    ]
    if not history_df.empty:
        report_lines.extend(
            [
                "",
                "## Training History",
                "",
                "```text",
                history_df.to_string(index=False),
                "```",
            ]
        )
    write_markdown(ROOT_OUTPUTS["report"], "\n".join(report_lines))

    model_card_lines = [
        "# Phase 3 LoRA Model Card",
        "",
        "## Status",
        "",
        "Experimental extension branch for bounded family/asset/explanation JSON generation.",
        "",
        "## Base model",
        "",
        f"- `{BASE_MODEL_NAME}`",
        "- LoRA adapter only; canonical detector remains unchanged.",
        "",
        "## Training data",
        "",
        f"- Training generators: {TRAIN_GENERATORS}",
        f"- Validation generator: {VALIDATION_GENERATOR}",
        f"- Test generator: {TEST_GENERATOR}",
        "- Supervision uses structured runtime packets plus ground-truth scenario family/assets/signals derived from heldout benchmark bundles.",
        "",
        "## Intended use",
        "",
        "- Post-alert operator-facing family attribution",
        "- Likely affected asset suggestion",
        "- Grounded signal citation in a compact JSON response",
        "",
        "## Not intended use",
        "",
        "- Primary anomaly detection",
        "- Human-level root-cause analysis claims",
        "- Real-world deployment claims",
        "",
        "## Limitations",
        "",
        "- Trained only on bounded synthetic heldout explanation packets.",
        "- Human-authored heldout bundle was excluded because aligned xai_v4 supervision packets were not present.",
        "- Results should be cited as explanation-side evidence, not benchmark-winning detector evidence.",
    ]
    write_markdown(ROOT_OUTPUTS["model_card"], "\n".join(model_card_lines))


def _write_run_log(
    examples: list[Example],
    training_info: dict[str, Any],
    results_df: pd.DataFrame,
    *,
    start_time_utc: pd.Timestamp,
    end_time_utc: pd.Timestamp,
    run_mode: str,
) -> None:
    lines = [
        "Phase 3 LoRA run log",
        f"run_started_utc: {start_time_utc.isoformat()}",
        f"run_finished_utc: {end_time_utc.isoformat()}",
        f"run_mode: {run_mode}",
        f"base_model: {BASE_MODEL_NAME}",
        f"train_examples: {sum(1 for item in examples if item.split == 'train')}",
        f"aux_in_domain_holdout_examples: {sum(1 for item in examples if item.split == 'aux_in_domain_holdout')}",
        f"validation_generator_examples: {sum(1 for item in examples if item.split == 'validation_generator_heldout')}",
        f"test_generator_examples: {sum(1 for item in examples if item.split == 'test_generator_heldout')}",
        f"adapter_root: {ADAPTER_ROOT}",
        f"tokenizer_root: {TOKENIZER_ROOT}",
        f"training_info_path: {TRAINING_INFO_PATH}",
        f"trainable_params: {training_info.get('trainable_params', '')}",
        f"total_params: {training_info.get('total_params', '')}",
        f"adapter_size_mb: {training_info.get('adapter_size_mb', '')}",
        f"model_memory_footprint_mb: {training_info.get('model_memory_footprint_mb', '')}",
        "",
        "results_snapshot:",
        results_df.to_string(index=False),
    ]
    ROOT_OUTPUTS["run_log"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_evidence_checklist(examples: list[Example], training_info: dict[str, Any], results_df: pd.DataFrame) -> None:
    history = training_info.get("history", []) or []
    checklist_lines = [
        "# Phase 3 LoRA Evidence Checklist",
        "",
        "## Artifact existence",
        "",
    ]
    artifact_order = [
        "manifest",
        "config",
        "results",
        "report",
        "model_card",
        "run_log",
        "family_fig",
        "asset_fig",
        "grounding_fig",
        "latency_fig",
    ]
    for key in artifact_order:
        path = ROOT_OUTPUTS[key]
        exists = path.exists()
        nonempty = exists and path.stat().st_size > 0
        checklist_lines.append(f"- `{path.name}`: exists=`{exists}`, nonempty=`{nonempty}`")
    checklist_lines.extend(
        [
            "",
            "## Training evidence",
            "",
            f"- Adapter directory exists: `{ADAPTER_ROOT.exists()}`",
            f"- Tokenizer directory exists: `{TOKENIZER_ROOT.exists()}`",
            f"- `training_info.json` exists: `{TRAINING_INFO_PATH.exists()}`",
            f"- Training history rows: `{len(history)}`",
            f"- Trainable parameters: `{training_info.get('trainable_params', '')}`",
            f"- Total parameters: `{training_info.get('total_params', '')}`",
            "",
            "## Split evidence",
            "",
            f"- Train examples: `{sum(1 for item in examples if item.split == 'train')}`",
            f"- Aux in-domain holdout: `{sum(1 for item in examples if item.split == 'aux_in_domain_holdout')}`",
            f"- Validation generator holdout: `{sum(1 for item in examples if item.split == 'validation_generator_heldout')}`",
            f"- Test generator holdout: `{sum(1 for item in examples if item.split == 'test_generator_heldout')}`",
            "",
            "## Metric evidence",
            "",
            f"- Result rows: `{len(results_df)}`",
            f"- Base model rows: `{int((results_df['model_variant'] == 'base_zero_shot').sum())}`",
            f"- LoRA rows: `{int((results_df['model_variant'] == 'lora_finetuned').sum())}`",
            "",
            "## Conservative interpretation",
            "",
            "- This branch is explanation-side and experimental.",
            "- Weak asset grounding or heldout family performance must be reported directly rather than hidden.",
            "- These outputs do not replace the canonical transformer detector benchmark.",
        ]
    )
    write_markdown(ROOT_OUTPUTS["evidence_checklist"], "\n".join(checklist_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experimental LoRA tiny-LLM branch.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain the LoRA adapter even if an existing adapter is present.")
    args = parser.parse_args()

    _ensure_dependencies()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    start_time_utc = pd.Timestamp.now("UTC")

    rows = _load_case_examples()
    examples = _split_examples(rows)
    manifest = _manifest_payload(examples)
    write_json(ROOT_OUTPUTS["manifest"], manifest)
    _write_config_yaml()

    base_tokenizer, base_model = _load_base_model()
    base_results_df, base_predictions_df = _evaluate_variant("base_zero_shot", base_tokenizer, base_model, examples)
    base_predictions_df.to_csv(PREDICTIONS_ROOT / "base_zero_shot_predictions.csv", index=False)

    if (not args.force_retrain) and ADAPTER_ROOT.exists() and any(ADAPTER_ROOT.glob("*")) and TOKENIZER_ROOT.exists():
        lora_tokenizer, lora_model, training_info = _load_existing_lora_model()
        run_mode = "reused_existing_adapter"
    else:
        lora_tokenizer, lora_model, training_info = _train_model(examples)
        run_mode = "fresh_retrain"
    lora_results_df, lora_predictions_df = _evaluate_variant("lora_finetuned", lora_tokenizer, lora_model, examples)
    lora_predictions_df.to_csv(PREDICTIONS_ROOT / "lora_finetuned_predictions.csv", index=False)

    results_df = pd.concat([base_results_df, lora_results_df], ignore_index=True)
    results_df["adapter_size_mb"] = np.nan
    results_df.loc[results_df["model_variant"] == "lora_finetuned", "adapter_size_mb"] = round(training_info["adapter_size_mb"], 2)
    results_df["model_memory_footprint_mb"] = np.nan
    results_df.loc[results_df["model_variant"] == "base_zero_shot", "model_memory_footprint_mb"] = round(
        float(base_model.get_memory_footprint()) / (1024.0 * 1024.0), 2
    )
    results_df.loc[results_df["model_variant"] == "lora_finetuned", "model_memory_footprint_mb"] = round(
        training_info["model_memory_footprint_mb"], 2
    )
    results_df.to_csv(ROOT_OUTPUTS["results"], index=False)

    _plot_results(results_df)
    _write_reports(examples, training_info, results_df)
    end_time_utc = pd.Timestamp.now("UTC")
    _write_run_log(examples, training_info, results_df, start_time_utc=start_time_utc, end_time_utc=end_time_utc, run_mode=run_mode)
    _write_evidence_checklist(examples, training_info, results_df)

    print(
        json.dumps(
            {
                "manifest": _safe_rel(ROOT_OUTPUTS["manifest"]),
                "config": _safe_rel(ROOT_OUTPUTS["config"]),
                "results": _safe_rel(ROOT_OUTPUTS["results"]),
                "report": _safe_rel(ROOT_OUTPUTS["report"]),
                "model_card": _safe_rel(ROOT_OUTPUTS["model_card"]),
                "run_log": _safe_rel(ROOT_OUTPUTS["run_log"]),
                "evidence_checklist": _safe_rel(ROOT_OUTPUTS["evidence_checklist"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
