# src/evaluation/plots.py
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_roc_curve(y_true, y_proba, title: str = "ROC curve"):
    """
    Рисует ROC-кривую + случайный классификатор.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def pr_curve_points(y_true, y_proba) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает точки для PR-кривой.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds


def plot_pr_curve(precision, recall, title: str = "PR curve"):
    """
    Рисует PR-кривую.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_proba_hist(y_proba, bins: int = 30, title: str = "Predicted probabilities"):
    """
    Гистограмма распределения предсказанных вероятностей.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(y_proba, bins=bins)
    plt.xlabel("Predicted p(click)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.show()