"""Evaluate one saved best checkpoint. The notebook invokes this only after all training finishes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, require_paths
from src.data import ManifestImageDataset, eval_transform, make_loader
from src.engine import predict, release_cuda
from src.evaluation_reporting import save_evaluation_artifacts
from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    cfg = load_config(args.config)
    paths = require_paths(cfg, "resized_manifest", "class_mapping")
    mapping = json.loads(paths["class_mapping"].read_text(encoding="utf-8")); target_names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    manifest = pd.read_csv(paths["resized_manifest"])
    dataset = ManifestImageDataset(manifest, "test", mapping, eval_transform(cfg["data"]["mean"], cfg["data"]["std"]))
    loader = make_loader(dataset, cfg["data"]["batch_size"], False, torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["name"], len(mapping), cfg["data"]["image_size"], cfg["model"].get("freeze_backbone", False), cfg["model"].get("pretrained_id"), cfg["model"].get("pool", "avg"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = predict(model, loader, device)
    summary = save_evaluation_artifacts(y_true, y_pred, target_names, args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True); (Path(args.output_dir) / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model, loader, dataset; release_cuda()


if __name__ == "__main__":
    main()
