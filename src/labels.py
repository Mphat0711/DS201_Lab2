"""Class-label mapping independent of PyTorch/data loading."""
from __future__ import annotations

import pandas as pd


def build_class_mapping(manifest: pd.DataFrame) -> dict[str, int]:
    train_classes = sorted(manifest.loc[manifest["split"] == "train", "class_name"].unique())
    return {name: index for index, name in enumerate(train_classes)}
