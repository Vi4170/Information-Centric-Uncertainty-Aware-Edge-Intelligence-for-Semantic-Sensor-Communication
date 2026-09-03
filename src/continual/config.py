"""Configuration constants for the continual-learning infrastructure
(Task 17 Phase 1: adaptation buffer; Task 18 Phase 2: condition monitor;
Task 24 Phase 4D: model lifecycle/versioning).

These are structural/monitoring constants -- not model or scoring
parameters -- and are unrelated to src/voi/'s calibration constants.
"""

# ---------------------------------------------------------------------------
# Phase 1: Adaptation Buffer (Task 17)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Phase 2: Condition Monitor (Task 18)
# ---------------------------------------------------------------------------

# Number of most-recent observations required before the monitor will
# report anything other than INSUFFICIENT_HISTORY.
DEFAULT_WINDOW_SIZE: int = 30

# Novelty control-chart multiplier ("k-sigma" convention): an observation is
# considered elevated if novelty > reference_mean + k * reference_std.
DEFAULT_NOVELTY_K: float = 2.0

# Fraction of the current window that must exceed the novelty control-chart
# threshold for a "sustained" novelty shift to be flagged. 1.0 (the default)
# is the strict interpretation from docs/continual_learning_design.md
# Section 3.1: the ENTIRE window must be elevated, not just its average --
# this is what distinguishes a sustained shift from one high-novelty spike
# diluted into an otherwise-normal window.
DEFAULT_NOVELTY_FRACTION_THRESHOLD: float = 1.0

# Population Stability Index threshold for a "significant" predicted-class
# distribution shift. PSI > 0.2 is a standard, widely used drift-monitoring
# convention (values below ~0.1 are considered no significant shift, 0.1-0.2
# a moderate shift, and >0.2 significant) -- not a value tuned against CWRU.
DEFAULT_PSI_THRESHOLD: float = 0.2

# Small constant added to class proportions before computing PSI, to avoid
# log(0)/division-by-zero when a class has zero occurrences in a window.
PSI_EPSILON: float = 1e-4

# ---------------------------------------------------------------------------
# Phase 4D: Model Lifecycle / Versioning (Task 24)
# ---------------------------------------------------------------------------

# Root directory for the versioned CNN model registry. Kept structurally
# separate from the original, unversioned baseline artifact at
# models/cwru_cnn_baseline.keras (which Task 24 does not touch) and
# alongside DEFAULT_BUFFER_DIR's data/ convention for continual-learning
# state.
DEFAULT_MODEL_REGISTRY_DIR: str = "models/continual"

# Within the registry: accepted, versioned models (v1/, v2/, ...).
MODEL_VERSIONS_SUBDIR: str = "versions"

# Within the registry: staged, not-yet-activated candidates, keyed by
# caller-assigned candidate_id, never by version number.
MODEL_CANDIDATES_SUBDIR: str = "candidates"

# The single small pointer file recording which version is active.
ACTIVE_POINTER_FILENAME: str = "active_cnn_pointer.json"
