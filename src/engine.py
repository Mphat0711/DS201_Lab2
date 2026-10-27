"""Train/evaluate engine. Caller supplies loaders, config values and output location."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from .metrics import metric_summary


def _epoch(model: nn.Module, loader, criterion, device: torch.device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, y_true, y_pred = 0.0, [], []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in tqdm(loader, leave=False, desc="train" if is_train else "val"):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(xb)
                loss = criterion(logits, yb)
            if is_train:
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            total_loss += loss.item() * yb.size(0)
            y_true.extend(yb.detach().cpu().tolist())
            y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    summary = metric_summary(y_true, y_pred)
    return {"loss": total_loss / len(loader.dataset), **summary}


def train_with_early_stopping(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: str | Path,
    checkpoint_metadata: dict[str, Any],
    patience: int = 2,
    min_delta: float = 1e-4,
) -> list[dict[str, float]]:
    """Adam-only training; saves best val macro-F1 state to explicit checkpoint path."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history, best_f1, stale_epochs = [], float("-inf"), 0
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_metrics = _epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if val_metrics["f1_macro"] > best_f1 + min_delta:
            best_f1, stale_epochs = val_metrics["f1_macro"], 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "val_f1_macro": best_f1, **checkpoint_metadata}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return history


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval().to(device)
    y_true, y_pred = [], []
    for xb, yb in tqdm(loader, leave=False, desc="eval"):
        logits = model(xb.to(device, non_blocking=True))
        y_true.extend(yb.tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    return y_true, y_pred
