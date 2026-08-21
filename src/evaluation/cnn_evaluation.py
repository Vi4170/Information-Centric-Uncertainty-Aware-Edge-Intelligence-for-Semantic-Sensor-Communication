"""CNN classifier evaluation framework for CWRU bearing fault classification.

This module is model-agnostic: it accepts predictions and ground-truth labels
from any trained classifier and computes all required evaluation metrics,
confusion matrices, classification reports, and visualizations.

It does NOT implement:
    - CNN architecture or training
    - Novelty detection
    - Uncertainty estimation
    - VoI integration

Future integration points are documented via the y_prob (class probabilities)
and the embedding contract in CNNEvaluationResult.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
NUM_CLASSES: int = 4

FIGURE_DIR: str = os.path.join("results", "figures")
TABLE_DIR: str = os.path.join("results", "tables")


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------

@dataclass
class CNNEvaluationResult:
    """Structured container for all CNN evaluation outputs.

    Attributes:
        accuracy: Overall classification accuracy.
        macro_precision: Macro-averaged precision.
        macro_recall: Macro-averaged recall.
        macro_f1: Macro-averaged F1-score.
        weighted_precision: Weighted-averaged precision.
        weighted_recall: Weighted-averaged recall.
        weighted_f1: Weighted-averaged F1-score.
        per_class_metrics: Dict mapping class name → {precision, recall, f1, support}.
        confusion_matrix: (NUM_CLASSES, NUM_CLASSES) integer array.
        classification_report_df: Full per-class report as a DataFrame.
        y_prob: Optional (N, 4) class probability array for future uncertainty use.

    Note:
        y_prob is preserved for the future Uncertainty Estimation module.
        Learned CNN embeddings (learned_embedding) are NOT included here — they
        will be passed directly to the future Novelty Estimation module.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=int))
    classification_report_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    y_prob: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

def validate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> None:
    """Validate that ground-truth labels and predictions are compatible.

    Args:
        y_true: 1D integer array of true labels.
        y_pred: 1D integer array of predicted labels.
        num_classes: Expected number of classes.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If inputs are empty, mismatched in length, or contain invalid values.
    """
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be a numpy array, got {type(y_true)}")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be a numpy array, got {type(y_pred)}")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true shape {y_true.shape} and y_pred shape {y_pred.shape} must match"
        )
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")


def validate_probabilities(
    y_prob: np.ndarray,
    n_samples: int,
    num_classes: int = NUM_CLASSES,
    tol: float = 1e-2,
) -> None:
    """Validate a class probability matrix from a CNN softmax output.

    Args:
        y_prob: (n_samples, num_classes) probability array.
        n_samples: Expected number of samples.
        num_classes: Expected number of classes (default 4).
        tol: Tolerance for row-sum check (default 1e-2).

    Raises:
        ValueError: For any structural, numerical, or range violation.
    """
    if not isinstance(y_prob, np.ndarray):
        raise TypeError(f"y_prob must be a numpy array, got {type(y_prob)}")
    if y_prob.ndim != 2:
        raise ValueError(
            f"y_prob must be 2D (n_samples, num_classes), got shape {y_prob.shape}"
        )
    if y_prob.shape[0] != n_samples:
        raise ValueError(
            f"y_prob row count ({y_prob.shape[0]}) must match n_samples ({n_samples})"
        )
    if y_prob.shape[1] != num_classes:
        raise ValueError(
            f"y_prob column count ({y_prob.shape[1]}) must match num_classes ({num_classes})"
        )
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or Inf values")
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob values must be in [0, 1]")
    row_sums = y_prob.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        bad_idx = np.where(np.abs(row_sums - 1.0) > tol)[0]
        raise ValueError(
            f"y_prob rows must sum to ~1.0 (within tol={tol}). "
            f"Violating row indices: {bad_idx[:5]} (row sums: {row_sums[bad_idx[:5]]})"
        )


# ---------------------------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = NUM_CLASSES,
) -> CNNEvaluationResult:
    """Compute all evaluation metrics for a classifier's predictions.

    This is the primary entry point for the CNN evaluation framework.

    Args:
        y_true: 1D integer array of ground-truth labels, shape (N,).
        y_pred: 1D integer array of predicted labels, shape (N,).
        class_names: Ordered list of class name strings.
        y_prob: Optional (N, num_classes) softmax probability array.
                Preserved for the future Uncertainty Estimation module.
        num_classes: Number of target classes (default 4).

    Returns:
        CNNEvaluationResult: Structured container with all metrics and arrays.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    validate_predictions(y_true, y_pred, num_classes=num_classes)

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float32)
        validate_probabilities(y_prob, n_samples=len(y_true), num_classes=num_classes)

    labels = list(range(num_classes))

    # Overall metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    w_prec = float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    w_rec = float(recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    w_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    # Per-class metrics
    per_class_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_support = np.array(
        [int((y_true == lbl).sum()) for lbl in labels], dtype=np.int64
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for i, cname in enumerate(class_names[:num_classes]):
        per_class_metrics[cname] = {
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Classification report DataFrame
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names[:num_classes]),
        zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df.rename(columns={"index": "class"}, inplace=True)

    return CNNEvaluationResult(
        accuracy=acc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_f1=macro_f1,
        weighted_precision=w_prec,
        weighted_recall=w_rec,
        weighted_f1=w_f1,
        per_class_metrics=per_class_metrics,
        confusion_matrix=cm,
        classification_report_df=report_df,
        y_prob=y_prob,
    )


# ---------------------------------------------------------------------------
# Confusion Matrix Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
) -> None:
    """Generate and optionally save a confusion matrix heatmap.

    Args:
        cm: (num_classes, num_classes) confusion matrix array.
        class_names: Ordered class name list.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix — CWRU 4-Class Bearing Fault Classification", fontsize=13)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training History Visualization
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Generate training/validation loss and accuracy curves from a history dict.

    Accepts any Python dictionary with the following keys (all optional):
        - "train_loss", "val_loss"
        - "train_accuracy", "val_accuracy"

    This function is framework-agnostic and does not depend on Keras, PyTorch, etc.

    Args:
        history: Dictionary of metric lists keyed by metric name.
        save_path: If provided, save the figure to this path.
    """
    has_loss = "train_loss" in history or "val_loss" in history
    has_acc = "train_accuracy" in history or "val_accuracy" in history

    if not has_loss and not has_acc:
        raise ValueError(
            "history must contain at least one of: 'train_loss', 'val_loss', "
            "'train_accuracy', 'val_accuracy'"
        )

    n_plots = int(has_loss) + int(has_acc)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    if has_loss:
        ax = axes[plot_idx]
        if "train_loss" in history:
            ax.plot(history["train_loss"], label="Train Loss")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Validation Loss", linestyle="--")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Training vs Validation Loss", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plot_idx += 1

    if has_acc:
        ax = axes[plot_idx]
        if "train_accuracy" in history:
            ax.plot(history["train_accuracy"], label="Train Accuracy")
        if "val_accuracy" in history:
            ax.plot(history["val_accuracy"], label="Validation Accuracy", linestyle="--")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title("Training vs Validation Accuracy", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("CNN Training History", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-Class Performance Visualization
# ---------------------------------------------------------------------------

def plot_class_performance(
    per_class_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Generate a grouped bar chart of per-class precision, recall, and F1-score.

    Args:
        per_class_metrics: Dict mapping class name → {precision, recall, f1_score, support}.
        save_path: If provided, save the figure to this path.
    """
    class_names = list(per_class_metrics.keys())
    precisions = [per_class_metrics[c]["precision"] for c in class_names]
    recalls = [per_class_metrics[c]["recall"] for c in class_names]
    f1s = [per_class_metrics[c]["f1_score"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1-Score")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Precision / Recall / F1-Score", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report Export
# ---------------------------------------------------------------------------

def save_classification_report(
    result: CNNEvaluationResult,
    save_path: str = os.path.join(TABLE_DIR, "cnn_classification_report.csv"),
) -> None:
    """Export the classification report DataFrame to CSV.

    Args:
        result: CNNEvaluationResult from evaluate_classifier().
        save_path: Target CSV file path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result.classification_report_df.to_csv(save_path, index=False)
    print(f"Saved classification report to: {save_path}")


def save_evaluation_summary(
    result: CNNEvaluationResult,
    save_path: str = os.path.join(TABLE_DIR, "cnn_evaluation_summary.csv"),
) -> None:
    """Export overall scalar metrics to a CSV summary table.

    Args:
        result: CNNEvaluationResult from evaluate_classifier().
        save_path: Target CSV file path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary = {
        "accuracy": [result.accuracy],
        "macro_precision": [result.macro_precision],
        "macro_recall": [result.macro_recall],
        "macro_f1": [result.macro_f1],
        "weighted_precision": [result.weighted_precision],
        "weighted_recall": [result.weighted_recall],
        "weighted_f1": [result.weighted_f1],
    }
    pd.DataFrame(summary).to_csv(save_path, index=False)
    print(f"Saved evaluation summary to: {save_path}")
