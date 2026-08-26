"""Normalized entropy-based uncertainty estimation."""

from __future__ import annotations

import numpy as np


def validate_probabilities(
    probabilities: np.ndarray,
    num_classes: int = 4,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Validate and return a probability matrix."""
    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must be a 2D array"
        )

    if probabilities.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} probabilities per observation, "
            f"got {probabilities.shape[1]}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "probabilities contain NaN or Inf values"
        )

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(
            "probabilities must be in [0, 1]"
        )

    row_sums = np.sum(
        probabilities,
        axis=1,
    )

    if not np.allclose(
        row_sums,
        1.0,
        atol=tolerance,
    ):
        raise ValueError(
            "probability rows must sum to approximately 1.0"
        )

    return probabilities


def normalized_entropy(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Calculate normalized predictive entropy in [0, 1].

    0 = completely confident prediction.
    1 = maximum uncertainty (uniform distribution).
    """
    probabilities = validate_probabilities(
        probabilities
    )

    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        clipped * np.log(clipped),
        axis=1,
    )

    max_entropy = np.log(
        probabilities.shape[1]
    )

    scores = entropy / max_entropy

    return np.clip(
        scores,
        0.0,
        1.0,
    ).astype(np.float32)


def predict_with_uncertainty(
    model,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the existing CNN probability interface and return predictions.

    Returns:
        probabilities: (N, 4)
        predicted_classes: (N,)
        uncertainty_scores: (N,)
    """
    from src.cnn.model import predict_probabilities

    probabilities = predict_probabilities(
        model,
        X,
    )

    probabilities = validate_probabilities(
        probabilities
    )

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    uncertainty_scores = normalized_entropy(
        probabilities
    )

    return (
        probabilities,
        predicted_classes,
        uncertainty_scores,
    )