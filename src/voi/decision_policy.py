"""Decision policy module for the VoI Engine.

Maps calculated Value of Information (VoI) scores to discrete communication actions.
Thresholds are configurable research parameters rather than fixed constants.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DecisionAction(str, Enum):
    """Provisional communication actions for edge intelligence transmission."""

    DISCARD = "DISCARD"
    BUFFER = "BUFFER"
    SUMMARY = "SUMMARY"
    TRANSMIT = "TRANSMIT"


@dataclass
class PolicyThresholds:
    """Configurable decision policy boundary thresholds.

    Default provisional thresholds:
    0.00 <= VoI < 0.25 -> DISCARD
    0.25 <= VoI < 0.50 -> BUFFER
    0.50 <= VoI < 0.70 -> SUMMARY
    0.70 <= VoI <= 1.00 -> TRANSMIT
    """

    discard_max: float = 0.25
    buffer_max: float = 0.50
    summary_max: float = 0.70

    def __post_init__(self):
        """Validate threshold ordering."""
        if not (0.0 <= self.discard_max < self.buffer_max < self.summary_max <= 1.0):
            raise ValueError(
                f"Invalid policy thresholds configuration: must satisfy "
                f"0.0 <= discard_max ({self.discard_max}) < buffer_max ({self.buffer_max}) "
                f"< summary_max ({self.summary_max}) <= 1.0"
            )


def evaluate_decision(
    voi_score: float, thresholds: Optional[PolicyThresholds] = None
) -> DecisionAction:
    """Deterministically map a VoI score to a communication decision action.

    Args:
        voi_score: Numerical VoI score (typically clipped in [0, 1]).
        thresholds: Configurable PolicyThresholds instance. Defaults to provisional thresholds.

    Returns:
        DecisionAction: Selected communication decision action.
    """
    if thresholds is None:
        thresholds = PolicyThresholds()

    if voi_score < thresholds.discard_max:
        return DecisionAction.DISCARD
    elif voi_score < thresholds.buffer_max:
        return DecisionAction.BUFFER
    elif voi_score < thresholds.summary_max:
        return DecisionAction.SUMMARY
    else:
        return DecisionAction.TRANSMIT
