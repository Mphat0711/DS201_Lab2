
from __future__ import annotations

from pathlib import Path

import numpy as np

from .engine import confusion, evaluation_report, metric_summary


def save_evaluation_artifacts(y_true: list[int], y_pred: list[int], target_names: list[str], output_dir: str | Path) -> dict[str, float]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    evaluation_report(y_true, y_pred, target_names).to_csv(root / "evaluation_report.csv")
    summary = metric_summary(y_true, y_pred)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        for normalization, suffix in ((None, "count"), ("true", "normalized")):
            matrix = confusion(y_true, y_pred, len(target_names), normalization)
            figure, axis = plt.subplots(figsize=(14, 12))
            sns.heatmap(matrix, cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=axis, fmt=".2f" if normalization else "d", annot=False)
            axis.set_xlabel("predicted"); axis.set_ylabel("true"); figure.tight_layout()
            figure.savefig(root / f"confusion_matrix_{suffix}.png", dpi=160); plt.close(figure)
        raw = confusion(y_true, y_pred, len(target_names), None)
        pairs = []
        for truth, prediction in zip(*np.where(~np.eye(len(target_names), dtype=bool))):
            if raw[truth, prediction]:
                pairs.append((target_names[truth], target_names[prediction], int(raw[truth, prediction])))
        import pandas as pd
        pd.DataFrame(pairs, columns=["true_class", "predicted_class", "count"]).sort_values("count", ascending=False).to_csv(root / "top_confusions.csv", index=False)
    except ModuleNotFoundError:
        (root / "FIGURES_NOT_CREATED.txt").write_text("matplotlib and seaborn are required for confusion-matrix PNGs", encoding="utf-8")
    return summary
