
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def require_paths(config: dict[str, Any], *names: str) -> dict[str, Path]:
    paths = config.get("paths", {})
    missing = [name for name in names if not paths.get(name)]
    if missing:
        raise KeyError(f"missing paths in config: {', '.join(missing)}")
    return {name: Path(paths[name]) for name in names}
