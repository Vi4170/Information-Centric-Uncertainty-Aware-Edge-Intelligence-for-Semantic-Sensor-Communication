"""Configuration constants for baseline Task Relevance module."""

NUM_CLASSES: int = 4
RANDOM_SEED: int = 42

# Probability sum tolerance for validation (same scale as uncertainty module)
PROB_TOLERANCE: float = 1e-2

# Default relevance estimation strategy
DEFAULT_STRATEGY: str = "class_mapping"

# Baseline class-to-relevance mapping for CWRU 4-class bearing fault diagnosis.
# These are initial design parameters, NOT optimized or learned values.
# Rationale:
#   0 (Normal):           Low relevance — routine baseline, little actionable information.
#   1 (Inner Race Fault): Maximum relevance — critical fault requiring immediate attention.
#   2 (Ball Fault):       High relevance — significant fault condition.
#   3 (Outer Race Fault): High relevance — significant fault condition.
CLASS_RELEVANCE_MAP: dict = {
    0: 0.10,  # Normal
    1: 1.00,  # Inner Race Fault
    2: 0.90,  # Ball Fault
    3: 0.90,  # Outer Race Fault
}
