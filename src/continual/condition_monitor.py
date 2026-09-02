"""Condition Monitor (Task 18 -- Continual Learning Design, Phase 2).

Implements ONLY Phase 2 of docs/continual_learning_design.md: a read-only
monitor that watches a rolling window of already-computed pipeline outputs
(novelty scores and predicted classes) for evidence of a SUSTAINED
distribution/operating-condition shift.

This module consumes plain floats/ints -- it never calls the CNN, the
novelty detector, the uncertainty estimator, relevance, temporal
importance, communication cost, or src/voi/ itself, and it never mutates
anything it is given. It has no side effects: it does not retrain, does
not update any centroid/prototype, does not touch the AdaptationBuffer,
and is not connected to the production VoI pipeline.

Per docs/continual_learning_design.md Section 3.1, a genuine "candidate
new operating condition" requires BOTH of the following to hold over a
rolling window, not a single anomalous observation:

    1. A sustained novelty control-chart shift: novelty scores stay above
       reference_mean + k * reference_std for (a configurable fraction of,
       strictly ALL of by default) the current window.
    2. A predicted-class-distribution shift: the current window's predicted
       class distribution diverges from the original reference (training)
       class distribution by more than a threshold, measured via the
       Population Stability Index (PSI).

High novelty alone, one predicted fault, or one unusual window are each
explicitly INSUFFICIENT on their own -- see ConditionShiftStatus.
"""

from __future__ import annotations

import math
import numbers
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional, Tuple

from src.continual.config import (
    DEFAULT_NOVELTY_FRACTION_THRESHOLD,
    DEFAULT_NOVELTY_K,
    DEFAULT_PSI_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    PSI_EPSILON,
)


class ConditionShiftStatus(str, Enum):
    """Current assessment of whether the monitored stream shows a sustained shift.

    Only CANDIDATE_CONDITION_SHIFT indicates both required signals agree.
    This status is a MONITORING signal, not a fault declaration, and not an
    adaptation trigger -- nothing consumes it automatically in this task.
    """

    INSUFFICIENT_HISTORY = "insufficient_history"
    STABLE = "stable"
    NOVELTY_SHIFT_ONLY = "novelty_shift_only"
    CLASS_DISTRIBUTION_SHIFT_ONLY = "class_distribution_shift_only"
    CANDIDATE_CONDITION_SHIFT = "candidate_condition_shift"


@dataclass(frozen=True)
class ConditionMonitorResult:
    """Structured, read-only report of the monitor's current assessment."""

    has_sufficient_history: bool
    window_size_used: int
    window_size_required: int

    novelty_window_mean: Optional[float]
    novelty_reference_mean: float
    novelty_reference_std: float
    novelty_threshold: float
    novelty_fraction_above_threshold: Optional[float]
    novelty_shift_detected: bool

    class_distribution_psi: Optional[float]
    class_distribution_shift_detected: bool
    window_class_counts: Dict[int, int]

    status: ConditionShiftStatus


def _validate_reference_novelty(mean: float, std: float) -> Tuple[float, float]:
    if isinstance(mean, bool) or not isinstance(mean, numbers.Real):
        raise TypeError(f"reference_novelty_mean must be numeric, got {type(mean)}")
    if isinstance(std, bool) or not isinstance(std, numbers.Real):
        raise TypeError(f"reference_novelty_std must be numeric, got {type(std)}")
    mean = float(mean)
    std = float(std)
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError("reference_novelty_mean/std must be finite")
    if std < 0.0:
        raise ValueError(f"reference_novelty_std must be >= 0, got {std}")
    return mean, std


def _validate_reference_class_distribution(
    distribution: Dict[int, float], num_classes: int
) -> Dict[int, float]:
    if not isinstance(distribution, dict):
        raise TypeError(f"reference_class_distribution must be a dict, got {type(distribution)}")

    expected_keys = set(range(num_classes))
    if set(distribution.keys()) != expected_keys:
        raise ValueError(
            f"reference_class_distribution must have exactly keys {sorted(expected_keys)}, "
            f"got {sorted(distribution.keys())}"
        )

    for class_id, prop in distribution.items():
        if isinstance(prop, bool) or not isinstance(prop, numbers.Real):
            raise TypeError(f"reference_class_distribution[{class_id}] must be numeric, got {type(prop)}")
        if not math.isfinite(prop) or prop < 0.0:
            raise ValueError(f"reference_class_distribution[{class_id}] must be finite and >= 0, got {prop}")

    total = sum(distribution.values())
    if not math.isclose(total, 1.0, abs_tol=1e-2):
        raise ValueError(f"reference_class_distribution must sum to ~1.0, got {total}")

    return distribution


def population_stability_index(
    reference_props: Dict[int, float],
    actual_counts: Dict[int, int],
    num_classes: int,
    window_size: int,
    epsilon: float = PSI_EPSILON,
) -> float:
    """Compute the Population Stability Index between a reference and an observed distribution.

    PSI = sum_c (actual_pct[c] - expected_pct[c]) * ln(actual_pct[c] / expected_pct[c])

    Both distributions are epsilon-smoothed to avoid log(0) / division by
    zero when a class has zero occurrences. This is a pure function of its
    inputs -- it holds no state and mutates nothing.

    Args:
        reference_props: Reference (e.g. training-set) class proportions, keyed by class id.
        actual_counts: Observed class counts in the current window, keyed by class id.
        num_classes: Total number of classes.
        window_size: Total number of observations the counts were drawn from.
        epsilon: Smoothing constant.

    Returns:
        float: The PSI value (>= 0.0; 0.0 means identical distributions).
    """
    psi = 0.0
    for class_id in range(num_classes):
        expected_pct = reference_props.get(class_id, 0.0) + epsilon
        actual_pct = (actual_counts.get(class_id, 0) / window_size if window_size > 0 else 0.0) + epsilon
        psi += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)
    return psi


class ConditionMonitor:
    """Read-only, deterministic monitor for sustained novelty/class-distribution shifts.

    Consumes one (novelty score, predicted class) pair at a time via
    observe(). Holds only a bounded rolling window of plain Python scalars
    -- it never stores or mutates a caller-supplied array, and it never
    calls into any other module.
    """

    def __init__(
        self,
        reference_novelty_mean: float,
        reference_novelty_std: float,
        reference_class_distribution: Dict[int, float],
        num_classes: int,
        window_size: int = DEFAULT_WINDOW_SIZE,
        novelty_k: float = DEFAULT_NOVELTY_K,
        novelty_fraction_threshold: float = DEFAULT_NOVELTY_FRACTION_THRESHOLD,
        psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ) -> None:
        """Initialize the monitor with fixed reference statistics.

        Args:
            reference_novelty_mean: Mean novelty score of the original
                reference distribution (e.g. computed once from training
                data by whatever novelty pipeline the caller uses -- this
                module does not compute it itself).
            reference_novelty_std: Std of the same reference distribution.
            reference_class_distribution: Reference (e.g. training-set)
                predicted/true class proportions, keyed by class id
                0..num_classes-1, summing to ~1.0.
            num_classes: Total number of classes.
            window_size: Number of most-recent observations considered.
            novelty_k: Control-chart multiplier for the novelty threshold.
            novelty_fraction_threshold: Fraction of the window that must
                exceed the novelty threshold to flag a sustained shift.
            psi_threshold: PSI value above which a class-distribution shift is flagged.

        Raises:
            TypeError / ValueError: For invalid reference statistics or configuration.
        """
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(f"num_classes must be an integer >= 2, got {num_classes}")
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f"window_size must be a positive integer, got {window_size}")
        if not isinstance(novelty_k, (int, float)) or novelty_k < 0:
            raise ValueError(f"novelty_k must be a non-negative number, got {novelty_k}")
        if not (0.0 < novelty_fraction_threshold <= 1.0):
            raise ValueError(f"novelty_fraction_threshold must be in (0, 1], got {novelty_fraction_threshold}")
        if not (psi_threshold > 0.0):
            raise ValueError(f"psi_threshold must be > 0, got {psi_threshold}")

        self.reference_novelty_mean, self.reference_novelty_std = _validate_reference_novelty(
            reference_novelty_mean, reference_novelty_std
        )
        self.reference_class_distribution = _validate_reference_class_distribution(
            reference_class_distribution, num_classes
        )
        self.num_classes = num_classes
        self.window_size = window_size
        self.novelty_k = float(novelty_k)
        self.novelty_fraction_threshold = float(novelty_fraction_threshold)
        self.psi_threshold = float(psi_threshold)

        self._novelty_window: Deque[float] = deque(maxlen=window_size)
        self._class_window: Deque[int] = deque(maxlen=window_size)

    def reset(self) -> None:
        """Clear all observed history. Reference statistics are unchanged."""
        self._novelty_window.clear()
        self._class_window.clear()

    @property
    def novelty_threshold(self) -> float:
        """Control-chart limit above which a single observation is 'elevated'."""
        return self.reference_novelty_mean + self.novelty_k * self.reference_novelty_std

    def observe(self, novelty: float, predicted_class: int) -> ConditionMonitorResult:
        """Record one new observation and return the monitor's updated assessment.

        Args:
            novelty: Novelty score for this observation. Must be a finite
                float in [0, 1] (the contract src/novelty's score() output
                already satisfies).
            predicted_class: Predicted class id for this observation. Must
                be an integer in [0, num_classes - 1] (the contract
                src/cnn's predict_classes() output already satisfies).

        Returns:
            ConditionMonitorResult: The monitor's assessment after
            incorporating this observation.

        Raises:
            TypeError: If novelty or predicted_class is the wrong type.
            ValueError: If novelty is non-finite/out of [0, 1], or
                predicted_class is out of range.
        """
        if isinstance(novelty, bool) or not isinstance(novelty, numbers.Real):
            # numbers.Real covers plain int/float and numpy floating/integer
            # types (e.g. np.float32, matching src/novelty's score() output),
            # while still explicitly rejecting bool.
            raise TypeError(f"novelty must be numeric, got {type(novelty)}")
        novelty = float(novelty)
        if not math.isfinite(novelty):
            raise ValueError(f"novelty must be finite, got {novelty}")
        if not (0.0 <= novelty <= 1.0):
            raise ValueError(f"novelty must be in [0, 1], got {novelty}")

        if isinstance(predicted_class, bool) or not isinstance(predicted_class, numbers.Integral):
            # numbers.Integral covers both plain int and numpy integer types
            # (e.g. np.int64), matching src/cnn's predict_classes() output,
            # while still explicitly rejecting bool and float.
            raise TypeError(f"predicted_class must be an integer, got {type(predicted_class)}")
        predicted_class = int(predicted_class)
        if not (0 <= predicted_class < self.num_classes):
            raise ValueError(f"predicted_class must be in [0, {self.num_classes - 1}], got {predicted_class}")

        # Store plain Python scalars only -- nothing here can alias or
        # mutate any caller-supplied array.
        self._novelty_window.append(novelty)
        self._class_window.append(predicted_class)

        return self._evaluate()

    def _evaluate(self) -> ConditionMonitorResult:
        window_size_used = len(self._novelty_window)
        has_sufficient_history = window_size_used >= self.window_size

        if not has_sufficient_history:
            return ConditionMonitorResult(
                has_sufficient_history=False,
                window_size_used=window_size_used,
                window_size_required=self.window_size,
                novelty_window_mean=None,
                novelty_reference_mean=self.reference_novelty_mean,
                novelty_reference_std=self.reference_novelty_std,
                novelty_threshold=self.novelty_threshold,
                novelty_fraction_above_threshold=None,
                novelty_shift_detected=False,
                class_distribution_psi=None,
                class_distribution_shift_detected=False,
                window_class_counts=dict(Counter(self._class_window)),
                status=ConditionShiftStatus.INSUFFICIENT_HISTORY,
            )

        novelty_values = list(self._novelty_window)
        window_mean = sum(novelty_values) / len(novelty_values)
        threshold = self.novelty_threshold
        fraction_above = sum(1 for v in novelty_values if v > threshold) / len(novelty_values)
        novelty_shift_detected = fraction_above >= self.novelty_fraction_threshold

        class_counts = dict(Counter(self._class_window))
        psi = population_stability_index(
            self.reference_class_distribution, class_counts, self.num_classes, len(self._class_window)
        )
        class_shift_detected = psi > self.psi_threshold

        if novelty_shift_detected and class_shift_detected:
            status = ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT
        elif novelty_shift_detected:
            status = ConditionShiftStatus.NOVELTY_SHIFT_ONLY
        elif class_shift_detected:
            status = ConditionShiftStatus.CLASS_DISTRIBUTION_SHIFT_ONLY
        else:
            status = ConditionShiftStatus.STABLE

        return ConditionMonitorResult(
            has_sufficient_history=True,
            window_size_used=window_size_used,
            window_size_required=self.window_size,
            novelty_window_mean=window_mean,
            novelty_reference_mean=self.reference_novelty_mean,
            novelty_reference_std=self.reference_novelty_std,
            novelty_threshold=threshold,
            novelty_fraction_above_threshold=fraction_above,
            novelty_shift_detected=novelty_shift_detected,
            class_distribution_psi=psi,
            class_distribution_shift_detected=class_shift_detected,
            window_class_counts=class_counts,
            status=status,
        )
