"""Create a ZIP from an explicit report directory."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def write_report_guide(report_dir: Path) -> None:
    """Create the short reading guide and the final table required by the PDF."""
    rows = []
    training_root = report_dir / "02_training"
    evaluation_root = report_dir / "03_evaluation"
    for summary_path in training_root.glob("*/model_summary.json") if training_root.is_dir() else []:
        run_name = summary_path.parent.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics_path = evaluation_root / run_name / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        rows.append({"run_name": run_name, **summary, "accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("f1_macro")})
    if rows:
        tables = report_dir / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("run_name").to_csv(tables / "model_comparison.csv", index=False)
    (report_dir / "README.md").write_text(
        "# Hướng dẫn đọc kết quả\n\n"
        "- `00_eda/`: phân bố nhãn và mapping class-to-index.\n"
        "- `01_preprocessing_split/`: split cố định và thông tin tiền xử lý.\n"
        "- `02_training/<run>/`: `training_curves.png`, history, `model_summary.json` (tham số, padding, augmentation) và `layer_summary.csv` khi bài yêu cầu.\n"
        "- `03_evaluation/<run>/`: bốn metric test, classification report và confusion matrix.\n"
        "- `tables/model_comparison.csv`: bảng tổng hợp resize, tham số, giây/epoch, accuracy và macro-F1.\n\n"
        "`divergence_epoch` là epoch đầu tiên trong hai epoch liên tiếp có `val_loss > train_loss`; dùng để nhận xét overfitting.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    args = parser.parse_args()
    report_dir, zip_path = args.report_dir.resolve(), args.zip_path.resolve()
    if not report_dir.is_dir():
        raise FileNotFoundError(report_dir)
    write_report_guide(report_dir)
    if zip_path.is_relative_to(report_dir):
        raise ValueError("zip-path must be outside report-dir to avoid recursive archives")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    created = Path(shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(report_dir)))
    if created != zip_path:
        created.replace(zip_path)
    print(zip_path)


if __name__ == "__main__":
    main()
