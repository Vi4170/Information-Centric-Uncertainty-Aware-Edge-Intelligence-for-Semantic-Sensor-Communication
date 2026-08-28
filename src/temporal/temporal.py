"""Baseline Temporal Importance estimation module for sequential sensor observations.

Computes a Temporal Importance score T ∈ [0, 1] that quantifies how much a
sensor observation has *changed* relative to the immediately preceding
observation in a temporal sequence.

Method (v0.1):
    For a sequence of observations x_1, x_2, …, x_T (each a 1D vector of
    ``observation_size`` elements):

        D_t = mean(|x_t − x_{t−1}|)          # mean absolute difference
        T_t = clip(D_t / temporal_change_scale, 0, 1)

    The first observation has no predecessor, so T_1 = 0 by convention.

Score interpretation:
    0 = temporally stable / little temporal change
    1 = highly significant temporal change (≥ reference scale)

This is a simple, interpretable baseline designed for computationally
constrained edge deployments.  It is NOT a final temporal model.
"""

from typing import Optional, Union

import numpy as np

from src.temporal.config import (
    DEFAULT_TEMPORAL_CHANGE_SCALE,
    EPSILON,
    MIN_OBSERVATIONS,
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_observations(observations: np.ndarray) -> np.ndarray:
    """Validate and return the observations array.

    Args:
        observations: Expected to be a 2D NumPy array of shape
            ``(num_observations, observation_size)`` with all-finite values.

    Returns:
        The validated array (unchanged).

    Raises:
        TypeError:  If *observations* is not a numpy array.
        ValueError: If *observations* is not 2D, is empty, or contains
            non-finite values.
    """
    if not isinstance(observations, np.ndarray):
        raise TypeError(
            f"observations must be a numpy ndarray, got {type(observations)}"
        )

    if observations.ndim != 2:
        raise ValueError(
            f"observations must be a 2D array of shape "
            f"(num_observations, observation_size), got shape {observations.shape}"
        )

    if observations.shape[0] == 0 or observations.shape[1] == 0:
        raise ValueError(
            f"observations must not be empty, got shape {observations.shape}"
        )

    if not np.isfinite(observations).all():
        raise ValueError(
            "observations contain NaN or Inf values; all values must be finite"
        )

    return observations


def _validate_temporal_change_scale(
    temporal_change_scale: float,
) -> float:
    """Validate the normalization reference scale.

    Args:
        temporal_change_scale: Must be a finite positive number.

    Returns:
        The validated scale as a Python float.

    Raises:
        TypeError:  If the scale is not numeric.
        ValueError: If the scale is non-finite or ≤ 0.
    """
    if not isinstance(temporal_change_scale, (int, float)):
        raise TypeError(
            f"temporal_change_scale must be numeric, got {type(temporal_change_scale)}"
        )

    temporal_change_scale = float(temporal_change_scale)

    if not np.isfinite(temporal_change_scale):
        raise ValueError(
            f"temporal_change_scale must be finite, got {temporal_change_scale}"
        )

    if temporal_change_scale <= 0.0:
        raise ValueError(
            f"temporal_change_scale must be > 0, got {temporal_change_scale}"
        )

    return temporal_change_scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_temporal_importance(
    observations: np.ndarray,
    temporal_change_scale: Optional[float] = None,
) -> np.ndarray:
    """Compute per-observation Temporal Importance scores for a sequence.

    For each consecutive pair of observations the mean absolute difference
    is calculated and normalised against a configurable reference scale to
    produce a score in [0, 1].

    Args:
        observations: 2D NumPy array of shape
            ``(num_observations, observation_size)``.
            Each row is one sequential observation / window.
            For scalar sequences use shape ``(N, 1)``.
        temporal_change_scale: Reference change magnitude used for
            normalisation.  Defaults to
            :data:`~src.temporal.config.DEFAULT_TEMPORAL_CHANGE_SCALE`.

    Returns:
        np.ndarray: 1D float64 array of length ``num_observations`` with
        Temporal Importance scores in [0, 1].  The first element is always
        0.0 (no preceding observation).

    Raises:
        TypeError:  If *observations* is not a numpy array or
            *temporal_change_scale* is not numeric.
        ValueError: For shape, finiteness, or scale violations.

    Examples:
        >>> import numpy as np
        >>> from src.temporal.temporal import compute_temporal_importance
        >>> # Constant sequence → all zeros after first observation
        >>> obs = np.ones((5, 4))
        >>> compute_temporal_importance(obs)
        array([0., 0., 0., 0., 0.])
        >>> # Single observation → [0.0]
        >>> compute_temporal_importance(np.array([[1.0, 2.0]]))
        array([0.])
    """
    # --- Validate inputs ---------------------------------------------------
    observations = _validate_observations(observations)

    if temporal_change_scale is None:
        temporal_change_scale = DEFAULT_TEMPORAL_CHANGE_SCALE
    temporal_change_scale = _validate_temporal_change_scale(temporal_change_scale)

    num_observations = observations.shape[0]

    # --- Allocate output ---------------------------------------------------
    scores = np.zeros(num_observations, dtype=np.float64)
    # scores[0] = 0.0 by convention (no preceding observation)

    if num_observations < 2:
        return scores

    # --- Compute mean absolute differences ---------------------------------
    # Vectorised: diff_matrix has shape (num_observations - 1, observation_size)
    diff_matrix = np.abs(
        observations[1:].astype(np.float64) - observations[:-1].astype(np.float64)
    )
    mean_abs_diffs = np.mean(diff_matrix, axis=1)  # shape (num_observations - 1,)

    # --- Normalise to [0, 1] -----------------------------------------------
    scores[1:] = np.clip(mean_abs_diffs / temporal_change_scale, 0.0, 1.0)

    # --- Final safety check ------------------------------------------------
    if not np.isfinite(scores).all():
        raise ValueError(
            "Computed temporal importance scores contain non-finite values"
        )

    return scores
