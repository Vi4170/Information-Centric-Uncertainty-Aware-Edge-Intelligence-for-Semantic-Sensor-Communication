"""Baseline Communication Cost estimation module for semantic sensor communication.

Computes a Communication Cost score C ∈ [0, 1] that quantifies the relative
resource burden of transmitting a sensor observation from an edge device.

Method (v0.1):
    Three normalised cost components are computed and combined via a
    configurable weighted sum:

        S = clip(payload_size / MAX_PAYLOAD_SIZE, 0, 1)
        T = clip(transmission_time / MAX_TRANSMISSION_TIME, 0, 1)
        B = clip(1 − available_bandwidth / REFERENCE_BANDWIDTH, 0, 1)

        C = clip(W_SIZE × S  +  W_TIME × T  +  W_BANDWIDTH × B,  0, 1)

Score interpretation:
    0 = negligible communication-resource cost
    1 = very high communication-resource cost relative to configured limits

This is a simple, interpretable baseline designed for computationally
constrained edge deployments.  It is NOT an FSO channel model and does NOT
simulate atmospheric attenuation, turbulence, BER, or packet loss.
"""

from typing import Optional, Tuple

import numpy as np

from src.communication.config import (
    MAX_PAYLOAD_SIZE,
    MAX_TRANSMISSION_TIME,
    REFERENCE_BANDWIDTH,
    WEIGHT_BANDWIDTH,
    WEIGHT_SIZE,
    WEIGHT_TIME,
    WEIGHT_SUM_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_non_negative_finite(
    value: float,
    name: str,
) -> float:
    """Validate that *value* is a finite non-negative number.

    Args:
        value: The numeric value to validate.
        name: Human-readable parameter name for error messages.

    Returns:
        The validated value as a Python float.

    Raises:
        TypeError:  If *value* is not numeric.
        ValueError: If *value* is non-finite or negative.
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")

    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")

    return value


def _validate_positive_finite(
    value: float,
    name: str,
) -> float:
    """Validate that *value* is a finite positive number (> 0).

    Args:
        value: The numeric value to validate.
        name: Human-readable parameter name for error messages.

    Returns:
        The validated value as a Python float.

    Raises:
        TypeError:  If *value* is not numeric.
        ValueError: If *value* is non-finite or ≤ 0.
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")

    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")

    return value


def _validate_weight(
    value: float,
    name: str,
) -> float:
    """Validate that a single weight is finite and in [0, 1].

    Args:
        value: Weight value.
        name: Human-readable parameter name for error messages.

    Returns:
        The validated weight as a Python float.

    Raises:
        TypeError:  If *value* is not numeric.
        ValueError: If *value* is non-finite or outside [0, 1].
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be numeric, got {type(value)}")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")

    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")

    return value


def _validate_weights(
    w_size: float,
    w_time: float,
    w_bandwidth: float,
    tolerance: float = WEIGHT_SUM_TOLERANCE,
) -> Tuple[float, float, float]:
    """Validate individual weights and verify they sum to 1.0.

    Args:
        w_size: Payload-size weight.
        w_time: Transmission-time weight.
        w_bandwidth: Bandwidth-pressure weight.
        tolerance: Absolute tolerance for the sum-to-one check.

    Returns:
        Tuple of validated (w_size, w_time, w_bandwidth).

    Raises:
        TypeError:  If any weight is not numeric.
        ValueError: If any weight is outside [0, 1] or if they do not sum
            to 1.0 within *tolerance*.
    """
    w_size = _validate_weight(w_size, "weight_size")
    w_time = _validate_weight(w_time, "weight_time")
    w_bandwidth = _validate_weight(w_bandwidth, "weight_bandwidth")

    weight_sum = w_size + w_time + w_bandwidth
    if abs(weight_sum - 1.0) > tolerance:
        raise ValueError(
            f"Weights must sum to 1.0 (within tol={tolerance}), "
            f"got {w_size} + {w_time} + {w_bandwidth} = {weight_sum}"
        )

    return w_size, w_time, w_bandwidth


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_communication_cost(
    payload_size: float,
    transmission_time: float,
    available_bandwidth: float,
    *,
    max_payload_size: Optional[float] = None,
    max_transmission_time: Optional[float] = None,
    reference_bandwidth: Optional[float] = None,
    weight_size: Optional[float] = None,
    weight_time: Optional[float] = None,
    weight_bandwidth: Optional[float] = None,
) -> float:
    """Compute the Communication Cost score for a single observation.

    The score quantifies the relative resource burden of transmitting
    the observation, normalised to [0, 1].

    Args:
        payload_size: Size of the observation payload in bytes (≥ 0).
        transmission_time: Estimated transmission time in seconds (≥ 0).
        available_bandwidth: Currently available bandwidth in bytes/second
            (≥ 0).
        max_payload_size: Reference maximum payload (bytes, > 0).
            Defaults to :data:`~src.communication.config.MAX_PAYLOAD_SIZE`.
        max_transmission_time: Reference maximum transmission time
            (seconds, > 0).  Defaults to
            :data:`~src.communication.config.MAX_TRANSMISSION_TIME`.
        reference_bandwidth: Reference "full capacity" bandwidth
            (bytes/second, > 0).  Defaults to
            :data:`~src.communication.config.REFERENCE_BANDWIDTH`.
        weight_size: Weight for the payload-size component.
            Defaults to :data:`~src.communication.config.WEIGHT_SIZE`.
        weight_time: Weight for the transmission-time component.
            Defaults to :data:`~src.communication.config.WEIGHT_TIME`.
        weight_bandwidth: Weight for the bandwidth-pressure component.
            Defaults to :data:`~src.communication.config.WEIGHT_BANDWIDTH`.

    Returns:
        float: Communication Cost C in [0, 1].

    Raises:
        TypeError:  If any input is not numeric.
        ValueError: If any input violates its constraints (non-finite,
            negative, or invalid reference limits / weights).

    Examples:
        >>> from src.communication.cost import compute_communication_cost
        >>> # Minimal cost — no payload, no delay, full bandwidth
        >>> compute_communication_cost(0, 0, 1_000_000)
        0.0
        >>> # Maximum cost — payload at limit, time at limit, zero bandwidth
        >>> compute_communication_cost(16_384, 1.0, 0)
        1.0
    """
    # --- Apply defaults ----------------------------------------------------
    if max_payload_size is None:
        max_payload_size = MAX_PAYLOAD_SIZE
    if max_transmission_time is None:
        max_transmission_time = MAX_TRANSMISSION_TIME
    if reference_bandwidth is None:
        reference_bandwidth = REFERENCE_BANDWIDTH
    if weight_size is None:
        weight_size = WEIGHT_SIZE
    if weight_time is None:
        weight_time = WEIGHT_TIME
    if weight_bandwidth is None:
        weight_bandwidth = WEIGHT_BANDWIDTH

    # --- Validate inputs ---------------------------------------------------
    payload_size = _validate_non_negative_finite(payload_size, "payload_size")
    transmission_time = _validate_non_negative_finite(
        transmission_time, "transmission_time"
    )
    available_bandwidth = _validate_non_negative_finite(
        available_bandwidth, "available_bandwidth"
    )

    # --- Validate reference limits -----------------------------------------
    max_payload_size = _validate_positive_finite(
        max_payload_size, "max_payload_size"
    )
    max_transmission_time = _validate_positive_finite(
        max_transmission_time, "max_transmission_time"
    )
    reference_bandwidth = _validate_positive_finite(
        reference_bandwidth, "reference_bandwidth"
    )

    # --- Validate weights --------------------------------------------------
    weight_size, weight_time, weight_bandwidth = _validate_weights(
        weight_size, weight_time, weight_bandwidth
    )

    # --- Compute normalised components -------------------------------------
    size_component = float(np.clip(payload_size / max_payload_size, 0.0, 1.0))
    time_component = float(np.clip(transmission_time / max_transmission_time, 0.0, 1.0))
    bandwidth_component = float(
        np.clip(1.0 - available_bandwidth / reference_bandwidth, 0.0, 1.0)
    )

    # --- Weighted combination ----------------------------------------------
    cost = float(np.clip(
        weight_size * size_component
        + weight_time * time_component
        + weight_bandwidth * bandwidth_component,
        0.0,
        1.0,
    ))

    # --- Final safety check ------------------------------------------------
    if not np.isfinite(cost):
        raise ValueError(
            f"Computed communication cost is not finite: {cost}"
        )

    return cost
