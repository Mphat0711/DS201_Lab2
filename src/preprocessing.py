"""Explicit materialization of aspect-preserving square resized datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm


def square_resize(image: Image.Image, side: int, pad_color: tuple[int, int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    canvas_side = max(width, height)
    canvas = Image.new("RGB", (canvas_side, canvas_side), pad_color)
    canvas.paste(image, ((canvas_side - width) // 2, (canvas_side - height) // 2))
    return canvas.resize((side, side), Image.Resampling.LANCZOS)


def materialize_resized_dataset(
    source_root: str | Path,
    split_manifest: str | Path,
    output_root: str | Path,
    image_size: int,
    pad_color: tuple[int, int, int] = (0, 0, 0),
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write resized files according to a frozen manifest and return output manifest."""
    root, manifest_path, destination = Path(source_root), Path(split_manifest), Path(output_root)
    manifest = pd.read_csv(manifest_path)
    required = {"source_relative", "class_name", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifest must contain {sorted(required)}")
    outputs: list[str] = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc=f"resize {image_size}"):
        source = root / row.source_relative
        target = destination / row.split / row.class_name / (Path(row.source_relative).stem + ".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not target.exists():
            with Image.open(source) as image:
                square_resize(image, image_size, pad_color).save(target, "JPEG", quality=95, optimize=True)
        outputs.append(str(target))
    result = manifest.copy()
    result["output_path"] = outputs
    result["image_size"] = image_size
    result.to_csv(destination / "manifest.csv", index=False)
    return result
