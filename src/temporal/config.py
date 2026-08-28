"""Configuration constants for baseline Temporal Importance module.

These are initial design parameters for the v0.1 temporal-change baseline.
They are NOT optimized or learned values.
"""

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Reference change magnitude used to normalize raw mean-absolute-difference
# values into the [0, 1] Temporal Importance score.
#
# A reasonable starting point for CWRU bearing vibration data:
#   Accelerometer readings typically range ~0.1–1.0 g in magnitude.
#   A "large" per-sample change of ~0.5 g across a 2048-sample window
#   would produce a mean-absolute-difference of roughly 0.5.
#
# This is a baseline default — adjust per deployment / signal domain.
DEFAULT_TEMPORAL_CHANGE_SCALE: float = 0.5

# ---------------------------------------------------------------------------
# Sequence constraints
# ---------------------------------------------------------------------------

# Minimum number of observations required to compute a meaningful sequence.
# A single observation is accepted and returns [0.0] (no previous context).
MIN_OBSERVATIONS: int = 1

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# Global random seed for project consistency.
# Temporal Importance v0.1 is fully deterministic, but we keep the seed
# configurable so downstream experiments can share a common seed.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

# Small epsilon to prevent division-by-zero or floating-point edge cases.
EPSILON: float = 1e-12
