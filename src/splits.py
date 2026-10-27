"""Create one reproducible stratified train/validation split before experiments."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _records(source_root: Path, source_split: str, target_split: str) -> list[dict[str, str]]:
    split_root = source_root / source_split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing source split: {split_root}")
    rows: list[dict[str, str]] = []
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append(
                    {
                        "source_relative": str(image_path.relative_to(source_root)),
                        "class_name": class_dir.name,
                        "split": target_split,
                    }
                )
    return rows


def create_stratified_split(
    source_root: str | Path,
    output_csv: str | Path,
    val_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Freeze train/val/test membership in one CSV; never copies source images."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    root, destination = Path(source_root), Path(output_csv)
    train_rows = _records(root, "train", "train")
    test_rows = _records(root, "test", "test")
    if not train_rows or not test_rows:
        raise ValueError("both source train and source test must contain images")

    train_df = pd.DataFrame(train_rows)
    train_indices, val_indices = train_test_split(
        train_df.index,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_df["class_name"],
    )
    train_df.loc[val_indices, "split"] = "val"
    frozen = pd.concat([train_df.loc[train_indices], train_df.loc[val_indices], pd.DataFrame(test_rows)])
    frozen = frozen.sort_values(["split", "class_name", "source_relative"]).reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_csv(destination, index=False)
    return frozen
