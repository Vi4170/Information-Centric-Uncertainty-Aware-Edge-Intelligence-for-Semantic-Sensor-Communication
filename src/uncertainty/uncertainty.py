"""Baseline Predictive Entropy Uncertainty Estimation module for CNN class probabilities.

Calculates normalized predictive entropy in [0, 1] from 4-class softmax probability distributions.
    - Confident prediction (e.g. [1, 0, 0, 0]) -> Uncertainty Score = 0.0
    - Uniform / ambiguous prediction (e.g. [0.25, 0.25, 0.25, 0.25]) -> Uncertainty Score = 1.0
"""

import os
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.uncertainty.config import (
    FIGURE_DIR,
    NUM_CLASSES,
    PROB_TOLERANCE,
    TABLE_DIR,
    UNCERTAINTY_BY_CLASS_FIG_PATH,
    UNCERTAINTY_DIST_FIG_PATH,
    UNCERTAINTY_SUMMARY_PATH,
)


def validate_probabilities(
    probabilities: np.ndarray,
    num_classes: int = NUM_CLASSES,
    tolerance: float = PROB_TOLERANCE,
) -> np.ndarray:
    """Validate 2D probability array format, bounds, finiteness, and row sum constraints.

    Args:
        probabilities: 2D probability array of shape (N, num_classes).
        num_classes: Expected number of classification classes (default 4).
        tolerance: Absolute tolerance for row sum check (default 1e-2).

    Returns:
        np.ndarray: Validated float32 probability array.

    Raises:
        TypeError: If probabilities is not a numpy array.
        ValueError: For any structural, numerical, range, or row sum violation.
    """
    if not isinstance(probabilities, np.ndarray):
        raise TypeError(f"probabilities must be a numpy array, got {type(probabilities)}")

    probs = np.asarray(probabilities, dtype=np.float32)

    if probs.ndim != 2:
        raise ValueError(f"probabilities must be a 2D array of shape (N, {num_classes}), got shape {probs.shape}")

    if probs.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} probabilities per sample, got {probs.shape[1]}"
        )

    if probs.size == 0:
        raise ValueError("probabilities array cannot be empty")

    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain NaN or Inf values")

    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must be strictly in range [0, 1]")

    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        bad_idx = np.where(np.abs(row_sums - 1.0) > tolerance)[0]
        raise ValueError(
            f"Probability rows must sum to ~1.0 (within tol={tolerance}). "
            f"Violating row indices: {bad_idx[:5]} (sums: {row_sums[bad_idx[:5]]})"
        )

    return probs


def compute_predictive_entropy(
    probabilities: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Calculate normalized Shannon predictive entropy in range [0, 1].

    Formula:
        H(p) = - sum(p_i * log2(p_i)) / log2(num_classes)

    Args:
        probabilities: 2D array of shape (N, num_classes).
        num_classes: Number of classes (default 4).

    Returns:
        np.ndarray: 1D float32 array of normalized Uncertainty Scores in [0, 1].
    """
    probs = validate_probabilities(probabilities, num_classes=num_classes)

    # Safe handling near zero probability to avoid log(0)
    clipped = np.clip(probs, 1e-12, 1.0)

    # Compute raw Shannon entropy using base-2 logarithm
    raw_entropy = -np.sum(clipped * np.log2(clipped), axis=1)

    # Maximum possible entropy for num_classes uniform distribution (log2(4) = 2.0)
    max_entropy = np.log2(float(num_classes))

    normalized = raw_entropy / max_entropy
    clipped_scores = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    if not np.isfinite(clipped_scores).all():
        raise ValueError("Computed uncertainty scores contain NaN or Inf values")

    return clipped_scores


class EntropyUncertaintyEstimator:
    """Estimator class wrapping normalized predictive entropy uncertainty calculation."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes

    def score(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute normalized Uncertainty Scores for input probability matrix."""
        return compute_predictive_entropy(probabilities, num_classes=self.num_classes)


def predict_with_uncertainty(
    model,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuses CNN model interface to output probabilities, predicted classes, and uncertainty scores.

    Args:
        model: Trained Keras CNN model.
        X: Input vibration tensor of shape (N, 2048, 1).

    Returns:
        Tuple of (probabilities, predicted_classes, uncertainty_scores).
    """
    from src.cnn.model import predict_probabilities

    probs = predict_probabilities(model, X)
    probs = validate_probabilities(probs)
    preds = np.argmax(probs, axis=-1).astype(np.int64)
    scores = compute_predictive_entropy(probs)

    return probs, preds, scores


def plot_uncertainty_distribution(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    save_path: str = UNCERTAINTY_DIST_FIG_PATH,
) -> None:
    """Plot histogram / distribution of Uncertainty Scores across dataset splits."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 25)

    ax.hist(train_scores, bins=bins, alpha=0.5, label="Train Scores", color="navy")
    ax.hist(val_scores, bins=bins, alpha=0.5, label="Validation Scores", color="orange")
    ax.hist(test_scores, bins=bins, alpha=0.5, label="Test Scores", color="green")

    ax.set_xlabel("Uncertainty Score [0, 1] (Normalized Entropy)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Uncertainty Score Distribution Across Dataset Splits", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved uncertainty distribution plot to: {save_path}")


def plot_uncertainty_by_class(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    class_names: dict,
    save_path: str = UNCERTAINTY_BY_CLASS_FIG_PATH,
) -> None:
    """Plot boxplot of Uncertainty Scores by health class on the test set."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    data_by_class = []
    labels_list = []

    for label_id, name in class_names.items():
        mask = test_labels == label_id
        if np.sum(mask) > 0:
            data_by_class.append(test_scores[mask])
            labels_list.append(f"Class {label_id}: {name}")

    ax.boxplot(data_by_class, tick_labels=labels_list, patch_artist=True)
    ax.set_ylabel("Uncertainty Score [0, 1]", fontsize=11)
    ax.set_title("Test Set Uncertainty Scores by Bearing Health Condition", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved uncertainty by class plot to: {save_path}")


def run_uncertainty_pipeline(
    model_path: str = "models/cwru_cnn_baseline.keras",
    data_path: str = "data/processed/cwru/cwru_dataset_v1.npz",
    summary_path: str = UNCERTAINTY_SUMMARY_PATH,
    fig_dir: str = FIGURE_DIR,
) -> pd.DataFrame:
    """Run full predictive entropy uncertainty estimation pipeline using trained CNN outputs.

    Returns summary DataFrame with uncertainty statistics per split and per class.
    """
    import keras
    from src.cnn.model import predict_probabilities

    print("=== Executing Uncertainty Estimation Baseline Pipeline ===")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained CNN model not found at '{model_path}'.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed CWRU dataset not found at '{data_path}'.")

    model = keras.models.load_model(model_path)
    data = np.load(data_path)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("Extracting CNN softmax probabilities...")
    prob_train = predict_probabilities(model, X_train)
    prob_val = predict_probabilities(model, X_val)
    prob_test = predict_probabilities(model, X_test)

    scores_train = compute_predictive_entropy(prob_train)
    scores_val = compute_predictive_entropy(prob_val)
    scores_test = compute_predictive_entropy(prob_test)

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_data = [
        {"split": "train", "mean": float(np.mean(scores_train)), "median": float(np.median(scores_train)), "min": float(np.min(scores_train)), "max": float(np.max(scores_train)), "std": float(np.std(scores_train))},
        {"split": "val", "mean": float(np.mean(scores_val)), "median": float(np.median(scores_val)), "min": float(np.min(scores_val)), "max": float(np.max(scores_val)), "std": float(np.std(scores_val))},
        {"split": "test", "mean": float(np.mean(scores_test)), "median": float(np.median(scores_test)), "min": float(np.min(scores_test)), "max": float(np.max(scores_test)), "std": float(np.std(scores_test))},
    ]

    class_names = {0: "Normal", 1: "Ball Fault", 2: "Inner Race Fault", 3: "Outer Race Fault"}
    for label_id, name in class_names.items():
        mask = y_test == label_id
        if np.sum(mask) > 0:
            c_scores = scores_test[mask]
            summary_data.append({
                "split": f"test_class_{label_id}_{name.replace(' ', '_')}",
                "mean": float(np.mean(c_scores)),
                "median": float(np.median(c_scores)),
                "min": float(np.min(c_scores)),
                "max": float(np.max(c_scores)),
                "std": float(np.std(c_scores)),
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved uncertainty summary table to: {summary_path}")

    plot_uncertainty_distribution(scores_train, scores_val, scores_test, save_path=UNCERTAINTY_DIST_FIG_PATH)
    plot_uncertainty_by_class(scores_test, y_test, class_names, save_path=UNCERTAINTY_BY_CLASS_FIG_PATH)

    print("=== Uncertainty Estimation Baseline Pipeline Complete ===")
    return summary_df


if __name__ == "__main__":
    run_uncertainty_pipeline()
