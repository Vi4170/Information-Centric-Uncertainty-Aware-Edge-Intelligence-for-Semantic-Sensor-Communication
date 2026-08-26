"""Baseline distance-based Novelty Detection module for 64-D learned CNN embeddings.

Calculates normalized Novelty Scores in [0, 1] using distance to training set reference centroid.
Strictly prevents data leakage by fitting reference parameters ONLY on training set embeddings.
"""

import os
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.novelty.config import (
    DEFAULT_DISTANCE_METRIC,
    EMBEDDING_DIM,
    FIGURE_DIR,
    NOVELTY_BY_CLASS_FIG_PATH,
    NOVELTY_DIST_FIG_PATH,
    NOVELTY_SUMMARY_PATH,
    RANDOM_SEED,
    REFERENCE_CLASS,
    TABLE_DIR,
)


class DistanceNoveltyDetector:
    """Distance-based Novelty Detector operating on 64-dimensional CNN embeddings.

    Computes Euclidean distance from input embeddings to the training reference centroid
    and maps raw distances to normalized Novelty Scores in [0, 1].

    Attributes:
        metric: Distance metric string (default "euclidean").
        reference_class: Optional class label filter for fitting reference centroid.
        is_fitted: Boolean flag indicating if detector has been fitted.
        reference_centroid: Mean 64-D feature vector learned from training set.
        d_min: Minimum raw distance observed in training reference set.
        d_max: Maximum raw distance observed in training reference set.
    """

    def __init__(
        self,
        metric: str = DEFAULT_DISTANCE_METRIC,
        reference_class: Optional[int] = REFERENCE_CLASS,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        """Initialize DistanceNoveltyDetector.

        Args:
            metric: Distance metric name (default "euclidean").
            reference_class: Optional integer class ID (e.g. 0 for Normal) to anchor reference.
                             If None, all training samples are used.
            embedding_dim: Expected dimensionality of input embeddings (default 64).
        """
        if metric.lower() != "euclidean":
            raise ValueError(f"Unsupported metric '{metric}'. Only 'euclidean' is supported.")

        self.metric = metric.lower()
        self.reference_class = reference_class
        self.embedding_dim = embedding_dim
        self.is_fitted = False

        self.reference_centroid: Optional[np.ndarray] = None
        self.d_min: float = 0.0
        self.d_max: float = 1.0

    def _validate_embeddings(self, embeddings: np.ndarray, name: str = "embeddings") -> np.ndarray:
        """Validate input embedding array dimensions and values."""
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(embeddings)}")

        if embeddings.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array of shape (N, {self.embedding_dim}), got shape {embeddings.shape}"
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"{name} column count ({embeddings.shape[1]}) does not match expected embedding dimension ({self.embedding_dim})"
            )

        if embeddings.size == 0:
            raise ValueError(f"{name} cannot be empty")

        if not np.isfinite(embeddings).all():
            raise ValueError(f"{name} contains NaN or Infinite values")

        return embeddings.astype(np.float32)

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: Optional[np.ndarray] = None,
    ) -> "DistanceNoveltyDetector":
        """Fit reference centroid and distance scaling bounds strictly on training data.

        Data Leakage Prevention:
            Must ONLY be called with training embeddings. Validation and test set
            embeddings must NEVER be passed to fit().

        Args:
            train_embeddings: 2D array of shape (N_train, 64).
            train_labels: Optional 1D integer array of training set class labels.

        Returns:
            Self (fitted detector instance).
        """
        train_emb = self._validate_embeddings(train_embeddings, name="train_embeddings")

        # Select reference samples
        if self.reference_class is not None and train_labels is not None:
            train_lbls = np.asarray(train_labels, dtype=np.int64)
            if train_lbls.shape[0] != train_emb.shape[0]:
                raise ValueError(
                    f"train_labels length ({train_lbls.shape[0]}) does not match train_embeddings ({train_emb.shape[0]})"
                )

            mask = train_lbls == self.reference_class
            if np.sum(mask) > 0:
                ref_samples = train_emb[mask]
            else:
                print(
                    f"Warning: Reference class {self.reference_class} not found in train_labels. "
                    "Using all training embeddings for reference centroid."
                )
                ref_samples = train_emb
        else:
            ref_samples = train_emb

        # Compute reference centroid
        self.reference_centroid = np.mean(ref_samples, axis=0, dtype=np.float32)

        # Compute raw distances for all training samples relative to reference centroid
        raw_distances = np.linalg.norm(train_emb - self.reference_centroid, axis=1)

        self.d_min = float(np.min(raw_distances))
        self.d_max = float(np.max(raw_distances))

        # Handle edge case where all training distances are identical
        if self.d_max <= self.d_min:
            self.d_max = self.d_min + 1e-6

        self.is_fitted = True
        return self

    def compute_raw_distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate unscaled Euclidean distance from embeddings to reference centroid.

        Args:
            embeddings: 2D array of shape (N, 64).

        Returns:
            np.ndarray: 1D array of raw Euclidean distances of shape (N,).
        """
        if not self.is_fitted or self.reference_centroid is None:
            raise RuntimeError("Detector must be fitted before computing distances. Call fit() first.")

        emb = self._validate_embeddings(embeddings, name="embeddings")
        raw_distances = np.linalg.norm(emb - self.reference_centroid, axis=1)
        return raw_distances.astype(np.float32)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute normalized Novelty Scores in [0, 1] for input embeddings.

        Higher score = more novel (further from baseline reference distribution).

        Args:
            embeddings: 2D array of shape (N, 64).

        Returns:
            np.ndarray: 1D float32 array of normalized Novelty Scores in [0, 1].
        """
        raw_distances = self.compute_raw_distance(embeddings)

        # Min-Max normalization using training set bounds
        normalized = (raw_distances - self.d_min) / (self.d_max - self.d_min)

        # Clip strictly to [0.0, 1.0]
        clipped = np.clip(normalized, 0.0, 1.0).astype(np.float32)

        if not np.isfinite(clipped).all():
            raise ValueError("Computed novelty scores contain NaN or Inf values")

        return clipped


def plot_novelty_distribution(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    save_path: str = NOVELTY_DIST_FIG_PATH,
) -> None:
    """Plot distribution histogram / boxplot of Novelty Scores across dataset splits."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 25)

    ax.hist(train_scores, bins=bins, alpha=0.5, label="Train Scores", color="navy")
    ax.hist(val_scores, bins=bins, alpha=0.5, label="Validation Scores", color="orange")
    ax.hist(test_scores, bins=bins, alpha=0.5, label="Test Scores", color="green")

    ax.set_xlabel("Novelty Score [0, 1]", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Novelty Score Distribution Across Dataset Splits", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved novelty distribution plot to: {save_path}")


def plot_novelty_by_class(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    class_names: dict,
    save_path: str = NOVELTY_BY_CLASS_FIG_PATH,
) -> None:
    """Plot boxplot of Novelty Scores by health class on the test set."""
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
    ax.set_ylabel("Novelty Score [0, 1]", fontsize=11)
    ax.set_title("Test Set Novelty Scores by Bearing Health Condition", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved novelty by class plot to: {save_path}")


def run_novelty_pipeline(
    model_path: str = "models/cwru_cnn_baseline.keras",
    data_path: str = "data/processed/cwru/cwru_dataset_v1.npz",
    summary_path: str = NOVELTY_SUMMARY_PATH,
    fig_dir: str = FIGURE_DIR,
) -> pd.DataFrame:
    """Run full novelty estimation pipeline using trained CNN embeddings.

    Returns summary DataFrame with novelty statistics per split and per class.
    """
    import keras
    from src.cnn.model import extract_embeddings

    print("=== Executing Novelty Detection Baseline Pipeline ===")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained CNN model not found at '{model_path}'.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed CWRU dataset not found at '{data_path}'.")

    # Load trained model and dataset
    model = keras.models.load_model(model_path)
    data = np.load(data_path)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Extract 64-D learned embeddings from CNN
    print("Extracting 64-D learned embeddings from CNN model...")
    emb_train = extract_embeddings(model, X_train)
    emb_val = extract_embeddings(model, X_val)
    emb_test = extract_embeddings(model, X_test)

    # Fit detector strictly on training set (Normal class 0 reference)
    detector = DistanceNoveltyDetector(reference_class=0)
    detector.fit(emb_train, y_train)

    # Score each split separately
    scores_train = detector.score(emb_train)
    scores_val = detector.score(emb_val)
    scores_test = detector.score(emb_test)

    # Export summary table
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_data = [
        {"split": "train", "mean_score": float(np.mean(scores_train)), "min_score": float(np.min(scores_train)), "max_score": float(np.max(scores_train)), "std_score": float(np.std(scores_train))},
        {"split": "val", "mean_score": float(np.mean(scores_val)), "min_score": float(np.min(scores_val)), "max_score": float(np.max(scores_val)), "std_score": float(np.std(scores_val))},
        {"split": "test", "mean_score": float(np.mean(scores_test)), "min_score": float(np.min(scores_test)), "max_score": float(np.max(scores_test)), "std_score": float(np.std(scores_test))},
    ]

    class_names = {0: "Normal", 1: "Ball Fault", 2: "Inner Race Fault", 3: "Outer Race Fault"}
    for label_id, name in class_names.items():
        mask = y_test == label_id
        if np.sum(mask) > 0:
            c_scores = scores_test[mask]
            summary_data.append({
                "split": f"test_class_{label_id}_{name.replace(' ', '_')}",
                "mean_score": float(np.mean(c_scores)),
                "min_score": float(np.min(c_scores)),
                "max_score": float(np.max(c_scores)),
                "std_score": float(np.std(c_scores)),
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved novelty summary table to: {summary_path}")

    # Generate diagnostic plots
    plot_novelty_distribution(scores_train, scores_val, scores_test, save_path=NOVELTY_DIST_FIG_PATH)
    plot_novelty_by_class(scores_test, y_test, class_names, save_path=NOVELTY_BY_CLASS_FIG_PATH)

    print("=== Novelty Detection Baseline Pipeline Complete ===")
    return summary_df


if __name__ == "__main__":
    run_novelty_pipeline()
