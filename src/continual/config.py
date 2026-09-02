"""Configuration constants for the adaptation buffer (Task 17, Phase 1).

These are structural/infrastructure constants -- not model or scoring
parameters -- and are unrelated to src/voi/'s calibration constants.
"""

# Split value that is never permitted to enter the adaptation buffer.
# Matches the CWRU pipeline's own split-column convention (src/cwru_pipeline).
FORBIDDEN_SPLIT: str = "test"

# Splits explicitly permitted to enter the adaptation buffer.
ALLOWED_SPLITS = ("train", "val")

# Where a persisted buffer is stored, kept structurally separate from
# data/processed/<dataset>/ (which holds the evaluation-only test arrays).
DEFAULT_BUFFER_DIR: str = "data/adaptation_buffer"
DEFAULT_BUFFER_FILENAME: str = "adaptation_buffer.json"

RANDOM_SEED: int = 42
