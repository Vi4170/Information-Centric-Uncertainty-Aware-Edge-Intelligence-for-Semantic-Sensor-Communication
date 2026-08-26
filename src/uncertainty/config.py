"""Configuration constants for baseline uncertainty estimation module."""

import os

NUM_CLASSES: int = 4
PROB_TOLERANCE: float = 1e-2
RANDOM_SEED: int = 42

# Paths
TABLE_DIR: str = os.path.join("results", "tables")
FIGURE_DIR: str = os.path.join("results", "figures")

UNCERTAINTY_SUMMARY_PATH: str = os.path.join(TABLE_DIR, "uncertainty_scores_summary.csv")
UNCERTAINTY_DIST_FIG_PATH: str = os.path.join(FIGURE_DIR, "uncertainty_score_distribution.png")
UNCERTAINTY_BY_CLASS_FIG_PATH: str = os.path.join(FIGURE_DIR, "uncertainty_by_class.png")
