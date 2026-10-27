
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
from src.labels import build_class_mapping
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--use-materialized", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    required = ("source_root", "split_manifest", "report_root") if not args.use_materialized else ("source_root", "resized_manifest", "report_root")
    paths = require_paths(cfg, *required)
    manifest = pd.read_csv(paths["resized_manifest"] if args.use_materialized else paths["split_manifest"])
    mapping = build_class_mapping(manifest)
    transform = eval_transform(cfg["data"]["mean"], cfg["data"]["std"])
    source_root = None if args.use_materialized else paths["source_root"]
    dataset = ManifestImageDataset(manifest, "train", mapping, transform, source_root=source_root)
    loader = make_loader(dataset, cfg["data"]["batch_size"], shuffle=True, pin_memory=torch.cuda.is_available())
    xb, yb = next(iter(loader))
    model = build_model(cfg["model"]["name"], len(mapping), cfg["data"]["image_size"], cfg["model"].get("freeze_backbone", False), cfg["model"].get("pretrained_id"))
    with torch.no_grad():
        logits = model(xb)
    if logits.shape != (xb.shape[0], len(mapping)):
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    result = {"input_shape": list(xb.shape), "labels_shape": list(yb.shape), "logits_shape": list(logits.shape), "num_classes": len(mapping), "num_workers": 0}
    paths["report_root"].mkdir(parents=True, exist_ok=True)
    (paths["report_root"] / "smoke_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
