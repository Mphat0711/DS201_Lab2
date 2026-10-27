
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


def train_fixed_epochs(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: str | Path,
    checkpoint_metadata: dict[str, Any],
) -> list[dict[str, float]]:
    """Adam-only fixed-length training; saves the checkpoint with lowest validation loss."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history, best_val_loss = [], float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_metrics = _epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "val_loss": best_val_loss, **checkpoint_metadata}, checkpoint_path)
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
