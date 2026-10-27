"""Explicitly freeze a split once or materialize one resolution from that split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, require_paths
from src.labels import build_class_mapping
from src.preprocessing import materialize_resized_dataset
from src.reporting import save_split_reports
from src.splits import create_stratified_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = require_paths(cfg, "source_root", "split_manifest", "resized_root", "report_root")
    if not args.create_split and not args.materialize:
        parser.error("select --create-split, --materialize, or both")
    if args.create_split:
        manifest = create_stratified_split(paths["source_root"], paths["split_manifest"], cfg["data"]["val_fraction"], cfg["seed"])
    elif not paths["split_manifest"].is_file():
        raise FileNotFoundError("frozen split manifest is missing; run with --create-split once first")
    else:
        manifest = pd.read_csv(paths["split_manifest"])
    mapping = build_class_mapping(manifest)
    paths["split_manifest"].parent.mkdir(parents=True, exist_ok=True)
    (paths["split_manifest"].parent / "class_to_idx.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    save_split_reports(manifest, mapping, paths["report_root"])
    if args.materialize:
        materialize_resized_dataset(paths["source_root"], paths["split_manifest"], paths["resized_root"], cfg["data"]["image_size"])
    print(f"Split manifest: {paths['split_manifest']}")


if __name__ == "__main__":
    main()
