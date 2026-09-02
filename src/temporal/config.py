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
# Version 0.2 calibrated default (Task 14). The original 0.5 assumed a raw
# accelerometer "g" scale that does not match this pipeline's actual
# per-window-pair mean-absolute-difference on the normalized CWRU signal,
# which saturates the [0, 1] score at ~1.0 for ~90%+ of fault windows,
# collapsing all fault severities into one indistinguishable bucket (see
# docs/voi_integration_analysis.md, Task 13). Recalibrated to the 95th
# percentile of mean-absolute-difference observed across ALL training-split
# windows (label-free, train-only — data/processed/cwru, computed via the
# unmodified compute_temporal_importance formula on X_train): ≈1.8. This
# only rescales where scores saturate; it does not change the formula.
DEFAULT_TEMPORAL_CHANGE_SCALE: float = 1.8

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
