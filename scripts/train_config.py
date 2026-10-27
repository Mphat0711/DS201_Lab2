"""Train exactly one explicit config. Notebook loops configs and releases VRAM after each call."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, require_paths
from src.data import ManifestImageDataset, eval_transform, make_loader
from src.engine import train_with_early_stopping
from src.models import build_model
from src.reproducibility import release_cuda, set_seed
from src.reporting import save_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    cfg = load_config(args.config)
    paths = require_paths(cfg, "resized_manifest", "class_mapping", "run_dir")
    set_seed(cfg["seed"])
    run_dir = paths["run_dir"]; run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    mapping = json.loads(paths["class_mapping"].read_text(encoding="utf-8"))
    manifest = pd.read_csv(paths["resized_manifest"])
    transform = eval_transform(cfg["data"]["mean"], cfg["data"]["std"])
    train_ds = ManifestImageDataset(manifest, "train", mapping, transform)
    val_ds = ManifestImageDataset(manifest, "val", mapping, transform)
    pin = torch.cuda.is_available()
    train_dl = make_loader(train_ds, cfg["data"]["batch_size"], True, pin)
    val_dl = make_loader(val_ds, cfg["data"]["batch_size"], False, pin)
    model = build_model(cfg["model"]["name"], len(mapping), cfg["data"]["image_size"], cfg["model"].get("freeze_backbone", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = train_with_early_stopping(model, train_dl, val_dl, device, cfg["training"]["epochs"], cfg["training"]["learning_rate"], run_dir / "best.pt", {"class_to_idx": mapping, "config": cfg}, cfg["training"]["patience"], cfg["training"]["min_delta"])
    history_df = pd.DataFrame(history); history_df.to_csv(run_dir / "training_history.csv", index=False)
    save_training_curves(history_df, run_dir / "training_curves.png", cfg["model"]["name"])
    del model, train_dl, val_dl, train_ds, val_ds
    release_cuda()


if __name__ == "__main__":
    main()
