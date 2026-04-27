"""Training entry point for the LSTM-Autoencoder detector.

Trains on benign sequences only, calibrates a 99th-percentile threshold on
validation benign reconstruction errors, evaluates on the test split, and
saves per-epoch checkpoints with resume support.

Usage:
    python -m phase1_models.train_lstm_autoencoder --project-root . --seed 1
    python -m phase1_models.train_lstm_autoencoder --project-root . --seed 1 --resume
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase1_models.checkpointing import (
    CheckpointState,
    find_resume_checkpoint,
    load_checkpoint,
    save_checkpoint,
    utcnow_iso,
)
from phase1_models.feature_builder import (
    build_sequence_dataset,
    fit_standardizer,
    select_numeric_feature_columns,
)
from phase1_models.metrics import compute_binary_metrics, compute_curve_payload
from phase1_models.models.lstm_autoencoder import LSTMAutoencoder, lstm_autoencoder_score
from phase1_models.residual_dataset import (
    canonical_residual_artifact_path,
    load_persisted_residual_dataframe,
)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CuDNN for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_lstm_autoencoder(
    project_root: Path,
    seed: int = 1,
    seq_len: int = 8,
    feature_count: int = 128,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 1,
    epochs: int = 24,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    threshold_percentile: float = 0.99,
    resume: bool = False,
) -> dict:
    """Train the LSTM-Autoencoder and write all artifacts.

    The training set is restricted to benign sequences so the model learns to
    reconstruct only nominal behavior. Reconstruction error then serves as the
    anomaly score on validation and test splits.

    Returns:
        A dictionary of final test-set metrics.
    """
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load residual dataset
    residual_path = canonical_residual_artifact_path(project_root)
    residual_df = load_persisted_residual_dataframe(residual_path)
    feature_columns = select_numeric_feature_columns(residual_df, max_features=feature_count)

    # Use the existing split_name column written by run_full_evaluation
    splits = {name: residual_df[residual_df["split_name"] == name].copy() for name in ("train", "val", "test")}
    train_benign = splits["train"][splits["train"]["attack_present"] == 0]
    standardizer = fit_standardizer(train_benign, feature_columns)

    x_train, y_train, _ = build_sequence_dataset(train_benign, feature_columns, seq_len, standardizer)
    x_val, y_val, val_meta = build_sequence_dataset(splits["val"], feature_columns, seq_len, standardizer)
    x_test, y_test, test_meta = build_sequence_dataset(splits["test"], feature_columns, seq_len, standardizer)

    # Build model and optimizer
    model = LSTMAutoencoder(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Resume if requested
    ckpt_dir = (
        project_root / "outputs" / "models_full_run" / "lstm_autoencoder" / f"seed_{seed}" / "checkpoints"
    )
    start_epoch = 0
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    if resume:
        ckpt_path = find_resume_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            state = load_checkpoint(ckpt_path, model, optimizer)
            start_epoch = state.epoch + 1
            best_val_loss = -state.best_metric  # we store -val_loss as val_metric
            print(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    # Training loop
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    stale_epochs = 0
    train_start = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses: list[float] = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_losses.append(float(criterion(recon, batch).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        is_best = val_loss < best_val_loss - 1e-4
        if is_best:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            state=CheckpointState(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metric=-val_loss,
                best_metric=-best_val_loss,
                is_best=is_best,
                timestamp_utc=utcnow_iso(),
                seed=seed,
                model_name="lstm_autoencoder",
                config={
                    "seq_len": seq_len,
                    "feature_count": feature_count,
                    "hidden_dim": hidden_dim,
                    "latent_dim": latent_dim,
                    "num_layers": num_layers,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                },
            ),
            checkpoint_dir=ckpt_dir,
        )

        print(
            f"[lstm_autoencoder seed={seed}] epoch={epoch} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"best={best_val_loss:.5f} stale={stale_epochs}"
        )
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    training_time = time.perf_counter() - train_start
    model.load_state_dict(best_state)

    # Threshold calibration on validation benign reconstruction errors
    val_benign_mask = splits["val"]["attack_present"].to_numpy()[seq_len - 1 :] == 0
    val_scores = lstm_autoencoder_score(model, torch.tensor(x_val, dtype=torch.float32).to(device)).cpu().numpy()
    val_benign_scores = val_scores[: len(val_benign_mask)][val_benign_mask]
    threshold = float(np.quantile(val_benign_scores, threshold_percentile))

    # Test-set evaluation
    inference_start = time.perf_counter()
    test_scores = lstm_autoencoder_score(model, torch.tensor(x_test, dtype=torch.float32).to(device)).cpu().numpy()
    inference_time = time.perf_counter() - inference_start

    metrics = compute_binary_metrics(y_test.astype(int), test_scores, threshold)
    curves = compute_curve_payload(y_test.astype(int), test_scores)

    # Save final artifacts
    results_dir = project_root / "outputs" / "models_full_run" / "lstm_autoencoder" / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model_name": "lstm_autoencoder",
        "seed": seed,
        "metrics": metrics,
        "curves": curves,
        "threshold": threshold,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "feature_count": feature_count,
        "seq_len": seq_len,
    }
    (results_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # Predictions parquet (for downstream per-family aggregation)
    import pandas as pd
    pred_df = test_meta.copy()
    pred_df["score"] = test_scores
    pred_df["predicted"] = (test_scores >= threshold).astype(int)
    pred_df["detector_emission_utc"] = pd.Timestamp.utcnow()
    pred_df.to_parquet(results_dir / "predictions.parquet")

    print(f"\n[lstm_autoencoder seed={seed}] DONE")
    print(f"  test F1={metrics['f1']:.4f}  precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}")
    print(f"  threshold={threshold:.6f}  training_time={training_time:.1f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM-Autoencoder detector.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--feature-count", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threshold-percentile", type=float, default=0.99)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_lstm_autoencoder(
        project_root=Path(args.project_root),
        seed=args.seed,
        seq_len=args.seq_len,
        feature_count=args.feature_count,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        threshold_percentile=args.threshold_percentile,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
