from __future__ import annotations

from pathlib import Path

import pandas as pd


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to create report figures; install requirements.txt") from exc
    return plt


def save_split_reports(manifest: pd.DataFrame, class_to_idx: dict[str, int], report_dir: str | Path) -> None:
    root = Path(report_dir)
    eda_dir, split_dir = root / "00_eda", root / "01_preprocessing_split"
    eda_dir.mkdir(parents=True, exist_ok=True); split_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame({"class_name": list(class_to_idx), "class_idx": list(class_to_idx.values())})
    mapping.to_csv(eda_dir / "class_mapping.csv", index=False)
    distribution = manifest.groupby(["class_name", "split"]).size().unstack(fill_value=0).reindex(list(class_to_idx), fill_value=0)
    distribution.to_csv(eda_dir / "dataset_distribution.csv")
    try:
        plt = _plt()
        distribution.plot(kind="bar", figsize=(16, 6)); plt.ylabel("images"); plt.tight_layout(); plt.savefig(eda_dir / "train_val_test_distribution.png", dpi=160); plt.close()
    except RuntimeError as warning:
        (eda_dir / "FIGURE_NOT_CREATED.txt").write_text(str(warning), encoding="utf-8")
    distribution.to_csv(split_dir / "split_distribution.csv")


def save_training_curves(history: pd.DataFrame, destination: str | Path, title: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for column, label in (("train_loss", "train"), ("val_loss", "val")):
        axes[0].plot(history["epoch"], history[column], label=label)
    for column, label in (("train_f1_macro", "train"), ("val_f1_macro", "val")):
        axes[1].plot(history["epoch"], history[column], label=label)
    for axis, ylabel in zip(axes, ("loss", "macro-F1")):
        axis.set_xlabel("epoch"); axis.set_ylabel(ylabel); axis.legend()
    divergence = divergence_epoch(history)
    if divergence is not None:
        axes[0].axvline(divergence, color="tab:red", linestyle="--", label="loss divergence")
        axes[0].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def divergence_epoch(history: pd.DataFrame) -> int | None:
    """First of two consecutive epochs where validation loss exceeds train loss."""
    separated = history["val_loss"] > history["train_loss"]
    for index in range(len(history) - 1):
        if bool(separated.iloc[index]) and bool(separated.iloc[index + 1]):
            return int(history.iloc[index]["epoch"])
    return None
