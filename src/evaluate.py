"""
Model Evaluation and Diagnostic Visualization Utilities
Author: Vaibhav Mistry

Provides tools for calculating multi-class classification metrics (accuracy,
F1 scores, precision, recall), plotting publication-grade confusion matrices,
learning curves, and architecture comparison charts.
"""

from typing import List, Optional, Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

STAGE_LABELS = ['W', 'N1', 'N2', 'N3', 'R']
ALL_CLASSES = [0, 1, 2, 3, 4]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class indices. Defaults to [0, 1, 2, 3, 4].
        target_names: Class display names. Defaults to ['W', 'N1', 'N2', 'N3', 'R'].

    Returns:
        Dictionary of computed metrics and text classification report.
    """
    if labels is None:
        labels = ALL_CLASSES
    if target_names is None:
        target_names = STAGE_LABELS

    acc = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )

    return {
        'accuracy': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'report': report
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plot and optionally save a normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=ALL_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=STAGE_LABELS)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_learning_curves(
    history: Any,
    title: str = "Model Training Curves",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation accuracy and loss side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy subplot
    axes[0].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2, linestyle='--')
    axes[0].set_title(f"{title} - Accuracy", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Loss subplot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    axes[1].set_title(f"{title} - Loss", fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved learning curves to {save_path}")
    plt.close()
