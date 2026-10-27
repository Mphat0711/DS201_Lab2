"""Create a ZIP from an explicit report directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    args = parser.parse_args()
    report_dir, zip_path = args.report_dir.resolve(), args.zip_path.resolve()
    if not report_dir.is_dir():
        raise FileNotFoundError(report_dir)
    if zip_path.is_relative_to(report_dir):
        raise ValueError("zip-path must be outside report-dir to avoid recursive archives")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    created = Path(shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(report_dir)))
    if created != zip_path:
        created.replace(zip_path)
    print(zip_path)


if __name__ == "__main__":
    main()
