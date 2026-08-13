"""Input normalization and validation module for the VoI Engine.

Provides utilities to validate numerical inputs, check bounds, perform generic min-max
scaling, and store validated VoI input vectors.
"""

from dataclasses import dataclass
import math
from typing import Any, Union
import numpy as np


def validate_numeric(value: Any, var_name: str) -> float:
    """Validate that the input value is a valid, finite numerical quantity.

    Args:
        value: Input value to validate.
        var_name: Name of the variable for error reporting.

    Returns:
        float: Validated value cast to float.

    Raises:
        TypeError: If value is not an int, float, or numpy scalar.
        ValueError: If value is NaN or infinite.
    """
    if isinstance(value, bool):
        raise TypeError(f"Variable '{var_name}' must be numeric, got boolean: {value}")

    if isinstance(value, (int, float, np.integer, np.floating)):
        val_float = float(value)
        if math.isnan(val_float) or np.isnan(val_float):
            raise ValueError(f"Variable '{var_name}' cannot be NaN")
        if math.isinf(val_float) or np.isinf(val_float):
            raise ValueError(f"Variable '{var_name}' cannot be infinite")
        return val_float

    raise TypeError(
        f"Variable '{var_name}' must be numeric (int or float), got type {type(value).__name__}"
    )


def validate_and_clip_unit_interval(
    value: Any, var_name: str, clip: bool = False
) -> float:
    """Verify that a numerical value lies in the unit interval [0, 1].

    Args:
        value: Input value to validate.
        var_name: Name of the variable for error reporting.
        clip: If True, clip values outside [0, 1] to the interval bounds.
              If False, raise ValueError for values outside [0, 1].

    Returns:
        float: Validated (and optionally clipped) float value in [0, 1].

    Raises:
        TypeError: If value is non-numeric.
        ValueError: If value is NaN/Inf or outside [0, 1] when clip=False.
    """
    val_float = validate_numeric(value, var_name)

    if val_float < 0.0 or val_float > 1.0:
        if clip:
            return max(0.0, min(1.0, val_float))
        raise ValueError(
            f"Variable '{var_name}' value {val_float} is outside the required unit interval [0, 1]"
        )

    return val_float


def normalize_min_max(
    value: Union[int, float, np.ndarray],
    min_val: Union[int, float],
    max_val: Union[int, float],
) -> Union[float, np.ndarray]:
    """Perform generic min-max normalization scaling an input to [0, 1].

    Note: This is a general-purpose scaling utility for future sensor integration.

    Formula: (value - min_val) / (max_val - min_val)

    Args:
        value: Raw value or numpy array of raw values to scale.
        min_val: Known minimum bound of the raw scale.
        max_val: Known maximum bound of the raw scale.

    Returns:
        Scaled value or array in [0, 1].

    Raises:
        ValueError: If max_val is less than or equal to min_val.
    """
    if max_val <= min_val:
        raise ValueError(
            f"max_val ({max_val}) must be strictly greater than min_val ({min_val})"
        )

    return (value - min_val) / (max_val - min_val)


@dataclass
class VoIInputs:
    """Data container for validated normalized VoI input variables.

    All fields represent abstract normalized variables in [0, 1].
    """

    novelty: float
    uncertainty: float
    task_relevance: float
    temporal_importance: float
    resource_cost: float
    clip: bool = False

    def __post_init__(self):
        """Validate all fields upon initialization."""
        self.validate(clip=self.clip)

    def validate(self, clip: bool = False) -> "VoIInputs":
        """Validate and optionally clip all input variables to [0, 1].

        Args:
            clip: Whether to clip values outside [0, 1].

        Returns:
            VoIInputs: Self with validated/clipped fields.
        """
        self.novelty = validate_and_clip_unit_interval(
            self.novelty, "novelty", clip=clip
        )
        self.uncertainty = validate_and_clip_unit_interval(
            self.uncertainty, "uncertainty", clip=clip
        )
        self.task_relevance = validate_and_clip_unit_interval(
            self.task_relevance, "task_relevance", clip=clip
        )
        self.temporal_importance = validate_and_clip_unit_interval(
            self.temporal_importance, "temporal_importance", clip=clip
        )
        self.resource_cost = validate_and_clip_unit_interval(
            self.resource_cost, "resource_cost", clip=clip
        )
        return self
