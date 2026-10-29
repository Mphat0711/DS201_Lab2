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
from src.data import ManifestImageDataset, build_transforms, make_loader
from src.engine import set_seed, train_fixed_epochs
from src.model import build_model, layer_summary, n_params, padding_policy
from src.engine import release_cuda
from src.reporting import divergence_epoch, save_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    cfg = load_config(args.config)
    paths = require_paths(cfg, "resized_manifest", "class_mapping", "run_dir", "report_root")
    set_seed(cfg["seed"])
    run_dir = paths["run_dir"]; run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    mapping = json.loads(paths["class_mapping"].read_text(encoding="utf-8"))
    manifest = pd.read_csv(paths["resized_manifest"])
    train_transform, eval_transform = build_transforms(
        cfg["data"]["mean"], cfg["data"]["std"], cfg["data"]["image_size"], cfg["data"].get("augmentation", False)
    )
    train_ds = ManifestImageDataset(manifest, "train", mapping, train_transform)
    val_ds = ManifestImageDataset(manifest, "val", mapping, eval_transform)
    pin = torch.cuda.is_available()
    train_dl = make_loader(train_ds, cfg["data"]["batch_size"], True, pin)
    val_dl = make_loader(val_ds, cfg["data"]["batch_size"], False, pin)
    model = build_model(cfg["model"]["name"], len(mapping), cfg["data"]["image_size"], cfg["model"].get("freeze_backbone", False), cfg["model"].get("pretrained_id"), cfg["model"].get("pool", "avg"))
    print(f"{cfg['model']['name']}: total={n_params(model):,} | trainable={n_params(model, trainable_only=True):,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = train_fixed_epochs(model, train_dl, val_dl, device, cfg["training"]["epochs"], cfg["training"]["learning_rate"], run_dir / "best.pt", {"class_to_idx": mapping, "config": cfg})
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    save_training_curves(history_df, run_dir / "training_curves.png", cfg["model"]["name"])
    best_epoch = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"])
    details = {"model": cfg["model"]["name"], "image_size": cfg["data"]["image_size"], "augmentation": cfg["data"].get("augmentation", False), "model_padding": padding_policy(model), "total_parameters": n_params(model), "trainable_parameters": n_params(model, trainable_only=True), "best_val_loss_epoch": best_epoch, "divergence_epoch": divergence_epoch(history_df), "seconds_per_epoch": float(history_df["seconds"].mean())}
    (run_dir / "model_summary.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    rows = layer_summary(model, cfg["data"]["image_size"])
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "layer_summary.csv", index=False)

    # The run directory keeps reproducibility artifacts; report/ is the final
    # presentation bundle requested by the lab.
    training_report_dir = paths["report_root"] / "02_training" / run_dir.name
    training_report_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(training_report_dir / "training_history.csv", index=False)
    save_training_curves(history_df, training_report_dir / "training_curves.png", cfg["model"]["name"])
    (training_report_dir / "model_summary.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    if rows:
        pd.DataFrame(rows).to_csv(training_report_dir / "layer_summary.csv", index=False)
    pd.DataFrame([details]).to_csv(training_report_dir / "training_summary.csv", index=False)
    del model, train_dl, val_dl, train_ds, val_ds
    release_cuda()


if __name__ == "__main__":
    main()
