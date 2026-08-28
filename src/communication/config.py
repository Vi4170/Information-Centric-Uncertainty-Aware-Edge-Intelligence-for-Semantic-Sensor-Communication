"""Configuration constants for baseline Communication Cost module.

These are initial design parameters for the v0.1 weighted communication-cost
baseline.  They are NOT optimized or experimentally calibrated values.
"""

# ---------------------------------------------------------------------------
# Reference limits — used to normalise raw input values into [0, 1]
# ---------------------------------------------------------------------------

# Maximum payload size (bytes) considered for normalisation.
# Observations exceeding this value are clipped to S = 1.0.
# Default: 16 384 bytes ≈ 2048 float64 samples (one CWRU window).
MAX_PAYLOAD_SIZE: float = 16_384.0

# Maximum transmission time (seconds) considered for normalisation.
# Default: 1.0 s — a generous upper bound for single-observation edge
# transmissions over typical wireless / FSO links.
MAX_TRANSMISSION_TIME: float = 1.0

# Reference bandwidth (bytes/second) representing "full" available capacity.
# When available bandwidth equals this value, the bandwidth-pressure
# component is 0.  As available bandwidth drops toward 0, the component
# approaches 1.
# Default: 1 000 000 B/s ≈ 1 MB/s — a conservative baseline for
# constrained edge / IoT links.
REFERENCE_BANDWIDTH: float = 1_000_000.0

# ---------------------------------------------------------------------------
# Combination weights — must form a valid convex combination (sum to 1.0)
# ---------------------------------------------------------------------------

# Weight for the payload-size component.
WEIGHT_SIZE: float = 0.5

# Weight for the transmission-time component.
WEIGHT_TIME: float = 0.3

# Weight for the bandwidth-pressure component.
WEIGHT_BANDWIDTH: float = 0.2

# Tolerance for checking that weights sum to 1.0.
WEIGHT_SUM_TOLERANCE: float = 1e-6

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# Global random seed for project consistency.
# Communication Cost v0.1 is fully deterministic, but we keep the seed
# configurable so downstream experiments can share a common seed.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

# Small epsilon to guard against floating-point edge cases.
EPSILON: float = 1e-12
