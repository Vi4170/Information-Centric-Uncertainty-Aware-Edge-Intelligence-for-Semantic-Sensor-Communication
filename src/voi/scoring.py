"""Scoring module for the VoI Engine.

Implements the baseline mathematical Value of Information (VoI) calculation.
Mathematical formulation is kept strictly separate from communication decision policy logic.
"""

from dataclasses import dataclass, field
from typing import Optional
from src.voi.normalization import VoIInputs


@dataclass
class VoIWeights:
    """Weight configuration for the baseline VoI mathematical formulation.

    Version 0.2 calibrated defaults (Task 14), replacing the Version 0.1
    equal-weight baseline (w_N = w_U = w_R = w_T = w_C = 0.20). Derived from
    training/validation-only analysis (docs/voi_integration_analysis.md,
    docs/voi_calibration_report.md): with this CNN's near-zero predictive
    entropy, Uncertainty carried ~0.04% of the VoI signal at 0.20 weight, and
    Communication Cost is a constant under the current fixed-window-size
    scenario (no channel-variability model exists yet) rather than a
    discriminating signal. Both are de-weighted rather than removed, so a
    future better uncertainty estimator or channel model can regain
    influence without a further weight change. The formula itself is
    unchanged; only these default parameter values were recalibrated.
    """

    novelty: float = 0.30
    uncertainty: float = 0.05
    task_relevance: float = 0.35
    temporal_importance: float = 0.20
    resource_cost: float = 0.10

    def validate(self):
        """Validate weight parameters."""
        for field_name in (
            "novelty",
            "uncertainty",
            "task_relevance",
            "temporal_importance",
            "resource_cost",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)):
                raise TypeError(f"Weight '{field_name}' must be numeric, got {type(val)}")


@dataclass
class ScoringResult:
    """Structured result containing raw and clipped mathematical VoI scores."""

    raw_voi_score: float
    voi_score: float
    weights: VoIWeights = field(default_factory=VoIWeights)


def calculate_voi_score(
    inputs: VoIInputs,
    weights: Optional[VoIWeights] = None,
    clip_output: bool = True,
) -> ScoringResult:
    """Calculate the Value of Information score using the baseline V0.1 formula:

    VoI_raw = w_N * N + w_U * U + w_R * R + w_T * T - w_C * C

    Args:
        inputs: Validated VoIInputs instance containing N, U, R, T, C.
        weights: Configurable weight instance. Defaults to equal weights (0.20 each).
        clip_output: If True, clip the final score to [0, 1] for decision policy use,
                     while preserving raw_voi_score.

    Returns:
        ScoringResult: Structure containing raw_voi_score, voi_score, and weights used.
    """
    if weights is None:
        weights = VoIWeights()
    else:
        weights.validate()

    raw_score = (
        weights.novelty * inputs.novelty
        + weights.uncertainty * inputs.uncertainty
        + weights.task_relevance * inputs.task_relevance
        + weights.temporal_importance * inputs.temporal_importance
        - weights.resource_cost * inputs.resource_cost
    )

    if clip_output:
        clipped_score = max(0.0, min(1.0, raw_score))
    else:
        clipped_score = raw_score

    return ScoringResult(
        raw_voi_score=raw_score,
        voi_score=clipped_score,
        weights=weights,
    )
