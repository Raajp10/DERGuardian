"""Robust per-epoch checkpointing with best-model tracking and resume support.

This module provides three primitives used by every Phase 1 training entry
point:

  - save_checkpoint(...): writes a per-epoch checkpoint, updates the latest
    pointer, and copies to best.pt when the validation metric improves.
  - load_checkpoint(...): restores model and optimizer state from a checkpoint
    file and returns the saved CheckpointState.
  - find_resume_checkpoint(...): returns the path of the most recent
    checkpoint, or None if none exist.

Old per-epoch checkpoints are automatically pruned to keep_last_n to bound
disk usage. Training history is appended to a JSONL file for easy plotting.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointState:
    """Metadata recorded with each checkpoint."""

    epoch: int
    train_loss: float
    val_loss: float
    val_metric: float
    best_metric: float
    is_best: bool
    timestamp_utc: str
    seed: int
    model_name: str
    config: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    state: CheckpointState,
    checkpoint_dir: Path,
    keep_last_n: int = 3,
) -> Path:
    """Persist a checkpoint and update best/latest pointers.

    Args:
        model: model whose state_dict is saved.
        optimizer: optimizer whose state_dict is saved (optional). Including
            it lets training resume mid-epoch with the same momentum state.
        state: metadata to record alongside the weights.
        checkpoint_dir: directory where checkpoints land. Will be created.
        keep_last_n: maximum number of per-epoch checkpoints to retain.

    Returns:
        The path of the freshly written epoch checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "state": asdict(state),
    }
    epoch_path = checkpoint_dir / f"epoch_{state.epoch:04d}.pt"
    torch.save(payload, epoch_path)

    # Update latest pointer
    latest_path = checkpoint_dir / "latest.pt"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    shutil.copy2(epoch_path, latest_path)

    # Update best pointer if applicable
    if state.is_best:
        best_path = checkpoint_dir / "best.pt"
        shutil.copy2(epoch_path, best_path)
        (checkpoint_dir / "best_metadata.json").write_text(
            json.dumps(asdict(state), indent=2, default=str)
        )

    # Prune old per-epoch checkpoints
    epoch_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    for old in epoch_files[:-keep_last_n]:
        try:
            old.unlink()
        except FileNotFoundError:
            pass

    # Append to training history
    history_path = checkpoint_dir / "training_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(state), default=str) + "\n")

    return epoch_path


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> CheckpointState:
    """Restore model (and optionally optimizer) state from a checkpoint file.

    Args:
        checkpoint_path: path to a `.pt` file written by save_checkpoint.
        model: model into which the state_dict will be loaded in-place.
        optimizer: optimizer into which the state_dict will be loaded.

    Returns:
        The CheckpointState that was recorded with this checkpoint.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return CheckpointState(**payload["state"])


def find_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the most recent checkpoint to resume from, or None."""
    if not checkpoint_dir.exists():
        return None
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    epoch_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    return epoch_files[-1] if epoch_files else None


def utcnow_iso() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()
