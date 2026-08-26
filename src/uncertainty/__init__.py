"""Uncertainty estimation baseline module for CNN class probabilities."""

from src.uncertainty.uncertainty import (
    EntropyUncertaintyEstimator,
    compute_predictive_entropy,
    predict_with_uncertainty,
    validate_probabilities,
)

__all__ = [
    "EntropyUncertaintyEstimator",
    "compute_predictive_entropy",
    "predict_with_uncertainty",
    "validate_probabilities",
]
