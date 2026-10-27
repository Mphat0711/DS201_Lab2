
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def square_resize(image: Image.Image, side: int, pad_color: tuple[int, int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    canvas_side = max(width, height)
    canvas = Image.new("RGB", (canvas_side, canvas_side), pad_color)
    canvas.paste(image, ((canvas_side - width) // 2, (canvas_side - height) // 2))
    return canvas.resize((side, side), Image.Resampling.LANCZOS)


def _resize_one(task: tuple[str, str, int, tuple[int, int, int], bool]) -> str:
    """One picklable resize task for optional preprocessing parallelism."""
    source_name, target_name, image_size, pad_color, overwrite = task
    source, target = Path(source_name), Path(target_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not target.exists():
        with Image.open(source) as image:
            square_resize(image, image_size, pad_color).save(target, "JPEG", quality=95, optimize=True)
    return str(target)


def materialize_resized_dataset(
    source_root: str | Path,
    split_manifest: str | Path,
    output_root: str | Path,
    image_size: int,
    pad_color: tuple[int, int, int] = (0, 0, 0),
    overwrite: bool = False,
    workers: int = 0,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Write resized files according to a frozen manifest and return output manifest."""
    root, manifest_path, destination = Path(source_root), Path(split_manifest), Path(output_root)
    manifest = pd.read_csv(manifest_path)
    required = {"source_relative", "class_name", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifest must contain {sorted(required)}")
    tasks: list[tuple[str, str, int, tuple[int, int, int], bool]] = []
    for row in manifest.itertuples(index=False):
        source = root / row.source_relative
        target = destination / row.split / row.class_name / (Path(row.source_relative).stem + ".jpg")
        tasks.append((str(source), str(target), image_size, pad_color, overwrite))
    if workers < 0:
        raise ValueError("preprocessing workers must be >= 0")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    total = len(tasks)
    if workers == 0:
        outputs = []
        for completed, task in enumerate(tasks, start=1):
            outputs.append(_resize_one(task))
            if completed % progress_every == 0 or completed == total:
                print(f"[resize {image_size}] {completed:,}/{total:,} images ({completed / total:.1%})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = []
            for completed, output in enumerate(pool.map(_resize_one, tasks, chunksize=16), start=1):
                outputs.append(output)
                if completed % progress_every == 0 or completed == total:
                    print(f"[resize {image_size}, workers={workers}] {completed:,}/{total:,} images ({completed / total:.1%})", flush=True)
    result = manifest.copy()
    result["output_path"] = outputs
    result["image_size"] = image_size
    result.to_csv(destination / "manifest.csv", index=False)
    return result
