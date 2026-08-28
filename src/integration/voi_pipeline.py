"""Thin integration pipeline that connects independently computed VoI factors
to the canonical VoI Engine.

This module does NOT implement any VoI mathematics, decision thresholds,
novelty scoring, uncertainty estimation, relevance mapping, temporal
analysis, or communication-cost calculation.  Those responsibilities belong
to their respective modules.

The sole purpose of this layer is to:

1. Accept five pre-computed, normalised factor values.
2. Validate that each is a finite float in [0, 1].
3. Forward them unchanged to the existing canonical
   :class:`~src.voi.voi_engine.VoIEngine`.
4. Return the canonical :class:`~src.voi.voi_engine.VoIResult`.
"""

from typing import Any, Optional

import numpy as np

from src.voi.decision_policy import PolicyThresholds
from src.voi.scoring import VoIWeights
from src.voi.voi_engine import VoIEngine, VoIResult


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate_factor(value: Any, name: str) -> float:
    """Validate that *value* is a finite numeric scalar in [0, 1].

    Args:
        value: The factor value to validate.
        name: Human-readable name for error messages.

    Returns:
        Validated Python float.

    Raises:
        TypeError:  If *value* is not numeric.
        ValueError: If *value* is non-finite or outside [0, 1].
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"{name} must be numeric (int or float), got {type(value)}"
        )

    val = float(value)

    if not np.isfinite(val):
        raise ValueError(f"{name} must be finite, got {val}")

    if val < 0.0 or val > 1.0:
        raise ValueError(
            f"{name} must be in [0, 1], got {val}"
        )

    return val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_voi_pipeline(
    novelty: float,
    uncertainty: float,
    task_relevance: float,
    temporal_importance: float,
    communication_cost: float,
    *,
    timestamp: Optional[Any] = None,
    weights: Optional[VoIWeights] = None,
    thresholds: Optional[PolicyThresholds] = None,
) -> VoIResult:
    """Run the full VoI integration pipeline for a single observation.

    Accepts five independently computed, normalised VoI factor values,
    validates them, and delegates to the canonical
    :class:`~src.voi.voi_engine.VoIEngine` for scoring and decision-making.

    Args:
        novelty: Normalised Novelty score N ∈ [0, 1].
        uncertainty: Normalised Prediction Uncertainty U ∈ [0, 1].
        task_relevance: Normalised Task Relevance R ∈ [0, 1].
        temporal_importance: Normalised Temporal Importance T ∈ [0, 1].
        communication_cost: Normalised Communication Cost C ∈ [0, 1].
        timestamp: Optional observation timestamp (passed through to the
            engine and preserved in the result).
        weights: Optional :class:`~src.voi.scoring.VoIWeights` override.
            Defaults to the canonical equal baseline weights.
        thresholds: Optional :class:`~src.voi.decision_policy.PolicyThresholds`
            override.  Defaults to the canonical provisional thresholds.

    Returns:
        :class:`~src.voi.voi_engine.VoIResult`: The canonical structured
        result containing ``novelty``, ``uncertainty``, ``task_relevance``,
        ``temporal_importance``, ``resource_cost``, ``raw_voi_score``,
        ``voi_score``, ``decision``, ``timestamp``, and ``metadata``.

    Raises:
        TypeError:  If any factor is not numeric.
        ValueError: If any factor is non-finite or outside [0, 1].

    Examples:
        >>> from src.integration.voi_pipeline import run_voi_pipeline
        >>> result = run_voi_pipeline(
        ...     novelty=0.8,
        ...     uncertainty=0.6,
        ...     task_relevance=0.9,
        ...     temporal_importance=0.3,
        ...     communication_cost=0.2,
        ... )
        >>> 0.0 <= result.voi_score <= 1.0
        True
    """
    # --- Validate all five factors -----------------------------------------
    novelty = _validate_factor(novelty, "novelty")
    uncertainty = _validate_factor(uncertainty, "uncertainty")
    task_relevance = _validate_factor(task_relevance, "task_relevance")
    temporal_importance = _validate_factor(temporal_importance, "temporal_importance")
    communication_cost = _validate_factor(communication_cost, "communication_cost")

    # --- Instantiate canonical engine with optional overrides ---------------
    engine = VoIEngine(
        weights=weights,
        thresholds=thresholds,
    )

    # --- Delegate to canonical engine --------------------------------------
    result: VoIResult = engine.compute(
        novelty=novelty,
        uncertainty=uncertainty,
        task_relevance=task_relevance,
        temporal_importance=temporal_importance,
        resource_cost=communication_cost,
        timestamp=timestamp,
    )

    return result
