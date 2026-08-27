"""Baseline Task Relevance estimation module for CWRU 4-class bearing fault diagnosis.

Computes a Task Relevance Score R ∈ [0, 1] that quantifies how relevant a sensor
observation is to the current application task (bearing fault monitoring).

Strategies:
    - "class_mapping":        Direct lookup from predicted class ID to a configurable relevance value.
    - "probability_weighted": Weighted sum R = Σ P(class_i) × relevance(class_i) using the full
                              probability distribution over classes.

Higher score = more relevant to the current application task.
Lower score  = less actionable / routine observation.
"""

from typing import Dict, Optional

import numpy as np

from src.relevance.config import (
    CLASS_RELEVANCE_MAP,
    DEFAULT_STRATEGY,
    NUM_CLASSES,
    PROB_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_relevance_map(
    relevance_map: Dict[int, float],
    num_classes: int = NUM_CLASSES,
) -> Dict[int, float]:
    """Validate that the relevance mapping covers all expected classes with valid values.

    Args:
        relevance_map: Dictionary mapping integer class IDs to relevance values in [0, 1].
        num_classes: Expected number of classes (default 4).

    Returns:
        The validated relevance mapping.

    Raises:
        TypeError: If relevance_map is not a dict.
        ValueError: If any class ID is missing or any relevance value is out of range or non-finite.
    """
    if not isinstance(relevance_map, dict):
        raise TypeError(f"relevance_map must be a dict, got {type(relevance_map)}")

    for class_id in range(num_classes):
        if class_id not in relevance_map:
            raise ValueError(
                f"relevance_map is missing class ID {class_id}. "
                f"Expected keys: {list(range(num_classes))}"
            )

    for class_id, value in relevance_map.items():
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Relevance value for class {class_id} must be numeric, got {type(value)}"
            )
        if not np.isfinite(value):
            raise ValueError(
                f"Relevance value for class {class_id} is not finite: {value}"
            )
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"Relevance value for class {class_id} must be in [0, 1], got {value}"
            )

    return relevance_map


def _validate_predicted_class(
    predicted_class: int,
    num_classes: int = NUM_CLASSES,
) -> int:
    """Validate that predicted_class is an integer in [0, num_classes - 1].

    Args:
        predicted_class: Integer class prediction.
        num_classes: Expected number of classes.

    Returns:
        Validated integer class ID.

    Raises:
        TypeError: If predicted_class is not an int.
        ValueError: If predicted_class is outside valid range.
    """
    if not isinstance(predicted_class, (int, np.integer)):
        raise TypeError(
            f"predicted_class must be an integer, got {type(predicted_class)}"
        )

    predicted_class = int(predicted_class)

    if predicted_class < 0 or predicted_class >= num_classes:
        raise ValueError(
            f"predicted_class must be in [0, {num_classes - 1}], got {predicted_class}"
        )

    return predicted_class


def _validate_probabilities(
    probabilities: np.ndarray,
    num_classes: int = NUM_CLASSES,
    tolerance: float = PROB_TOLERANCE,
) -> np.ndarray:
    """Validate a 1D probability vector for a single observation.

    Args:
        probabilities: 1D array of class probabilities.
        num_classes: Expected number of classes (default 4).
        tolerance: Absolute tolerance for sum-to-one check.

    Returns:
        Validated float32 probability vector.

    Raises:
        TypeError: If probabilities is not a numpy array.
        ValueError: For shape, finiteness, non-negativity, or sum violations.
    """
    if not isinstance(probabilities, np.ndarray):
        raise TypeError(
            f"probabilities must be a numpy array, got {type(probabilities)}"
        )

    probs = np.asarray(probabilities, dtype=np.float32)

    if probs.ndim != 1:
        raise ValueError(
            f"probabilities must be a 1D array of length {num_classes}, "
            f"got shape {probs.shape}"
        )

    if probs.shape[0] != num_classes:
        raise ValueError(
            f"Expected {num_classes} probability values, got {probs.shape[0]}"
        )

    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain NaN or Inf values")

    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")

    prob_sum = float(np.sum(probs))
    if abs(prob_sum - 1.0) > tolerance:
        raise ValueError(
            f"probabilities must sum to ~1.0 (within tol={tolerance}), got sum={prob_sum}"
        )

    return probs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def relevance_from_class(
    predicted_class: int,
    relevance_map: Optional[Dict[int, float]] = None,
    strategy: str = DEFAULT_STRATEGY,
) -> float:
    """Compute Task Relevance Score from a predicted class ID using class-mapping strategy.

    Args:
        predicted_class: Integer predicted class ID in [0, NUM_CLASSES - 1].
        relevance_map: Optional override mapping of class ID -> relevance value.
                       Defaults to CLASS_RELEVANCE_MAP from config.
        strategy: Strategy name. Must be "class_mapping" for this function.

    Returns:
        float: Task Relevance Score R in [0, 1].

    Raises:
        TypeError: If predicted_class is not an integer.
        ValueError: If predicted_class is outside valid range, strategy is invalid,
                    or relevance_map is malformed.
    """
    if strategy != "class_mapping":
        raise ValueError(
            f"relevance_from_class only supports strategy='class_mapping', got '{strategy}'"
        )

    if relevance_map is None:
        relevance_map = CLASS_RELEVANCE_MAP

    _validate_relevance_map(relevance_map)
    predicted_class = _validate_predicted_class(predicted_class)

    score = float(relevance_map[predicted_class])

    # Final safety assertion (should never fail given validated map)
    if not np.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError(
            f"Computed relevance score is out of valid range [0, 1]: {score}"
        )

    return score


def relevance_from_probabilities(
    probabilities: np.ndarray,
    relevance_map: Optional[Dict[int, float]] = None,
    strategy: str = "probability_weighted",
) -> float:
    """Compute Task Relevance Score as a probability-weighted sum over class relevances.

    Formula:
        R = Σ P(class_i) × relevance(class_i)  for i in [0, NUM_CLASSES - 1]

    Args:
        probabilities: 1D numpy array of class probabilities with exactly NUM_CLASSES elements.
        relevance_map: Optional override mapping of class ID -> relevance value.
                       Defaults to CLASS_RELEVANCE_MAP from config.
        strategy: Strategy name. Must be "probability_weighted" for this function.

    Returns:
        float: Task Relevance Score R in [0, 1].

    Raises:
        TypeError: If probabilities is not a numpy array.
        ValueError: If probabilities shape, values, or sum are invalid, strategy is invalid,
                    or relevance_map is malformed.
    """
    if strategy != "probability_weighted":
        raise ValueError(
            f"relevance_from_probabilities only supports strategy='probability_weighted', "
            f"got '{strategy}'"
        )

    if relevance_map is None:
        relevance_map = CLASS_RELEVANCE_MAP

    _validate_relevance_map(relevance_map)
    probs = _validate_probabilities(probabilities)

    # Build relevance weight vector in class-index order
    num_classes = len(relevance_map)
    relevance_weights = np.array(
        [relevance_map[i] for i in range(num_classes)], dtype=np.float32
    )

    score = float(np.dot(probs, relevance_weights))

    # Clip to [0, 1] for numerical safety
    score = float(np.clip(score, 0.0, 1.0))

    if not np.isfinite(score):
        raise ValueError(
            f"Computed relevance score is not finite: {score}"
        )

    return score
