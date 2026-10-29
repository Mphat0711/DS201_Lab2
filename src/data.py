
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def build_transforms(mean: list[float], std: list[float], image_size: int, augmentation: bool = False) -> tuple[transforms.Compose, transforms.Compose]:
    """Train uses PDF augmentation only when its explicit config flag is enabled."""
    train_steps: list[object] = []
    if augmentation:
        train_steps.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(train_steps), eval_transform(mean, std)


def eval_transform(mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def build_class_mapping(manifest: pd.DataFrame) -> dict[str, int]:
    classes = sorted(manifest.loc[manifest["split"] == "train", "class_name"].unique())
    return {name: index for index, name in enumerate(classes)}


class ManifestImageDataset(Dataset[tuple[object, int]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        class_to_idx: dict[str, int],
        transform: Callable | None,
        source_root: str | Path | None = None,
    ) -> None:
        self.rows = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.source_root = Path(source_root) if source_root is not None else None
        if self.rows.empty:
            raise ValueError(f"split is empty: {split}")
        if "output_path" not in self.rows.columns and self.source_root is None:
            raise ValueError("source_root is required when manifest has no output_path")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[object, int]:
        row = self.rows.iloc[index]
        path = Path(row.output_path) if "output_path" in self.rows.columns else self.source_root / row.source_relative
        with Image.open(path) as image:
            image = image.convert("RGB")
            sample = self.transform(image) if self.transform else image.copy()
        return sample, self.class_to_idx[row.class_name]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    """The project rule fixes num_workers=0 for stable Kaggle/local execution."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )
