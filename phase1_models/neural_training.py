from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(
    model: nn.Module,
    x_train: np.ndarray,
    x_val: np.ndarray,
    epochs: int = 6,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int | None = None,
    min_delta: float = 1e-4,
) -> tuple[nn.Module, dict[str, list[float]], float]:
    model = model.to(DEVICE)
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                recon = model(batch)
                val_losses.append(float(criterion(recon, batch).item()))
        history["train_loss"].append(float(np.mean(train_losses)))
        mean_val_loss = float(np.mean(val_losses))
        history["val_loss"].append(mean_val_loss)
        if mean_val_loss + min_delta < best_val:
            best_val = mean_val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                break
    training_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    return model.cpu(), history, training_time


def predict_autoencoder_errors(model: nn.Module, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    model = model.to(DEVICE)
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            error = torch.mean((recon - batch) ** 2, dim=1).cpu().numpy()
            errors.append(error)
    return np.concatenate(errors, axis=0) if errors else np.array([], dtype=float)


def train_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 6,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    token_input: bool = False,
    pos_weight: float | None = None,
    patience: int | None = None,
    min_delta: float = 1e-4,
) -> tuple[nn.Module, dict[str, list[float]], float]:
    tensor_dtype = torch.long if token_input else torch.float32
    model = model.to(DEVICE)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=tensor_dtype), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=tensor_dtype), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor(float(pos_weight), device=DEVICE)
    )
    history = {"train_loss": [], "val_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = model(batch_x)
                val_losses.append(float(criterion(logits, batch_y).item()))
        history["train_loss"].append(float(np.mean(train_losses)))
        mean_val_loss = float(np.mean(val_losses))
        history["val_loss"].append(mean_val_loss)
        if mean_val_loss + min_delta < best_val:
            best_val = mean_val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                break
    training_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    return model.cpu(), history, training_time


def predict_classifier_scores(model: nn.Module, x: np.ndarray, batch_size: int = 128, token_input: bool = False) -> np.ndarray:
    tensor = torch.tensor(x, dtype=torch.long if token_input else torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    model = model.to(DEVICE)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.append(probs)
    return np.concatenate(scores, axis=0) if scores else np.array([], dtype=float)
