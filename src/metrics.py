
from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def metric_summary(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": precision, "recall_macro": recall, "f1_macro": f1}


def evaluation_report(y_true: Sequence[int], y_pred: Sequence[int], target_names: list[str]) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=list(range(len(target_names))), target_names=target_names, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def confusion(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int, normalize: str | None = None):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize=normalize)
