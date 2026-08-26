"""Configuration constants for baseline novelty detection module."""

import os

# Embedding Parameters
EMBEDDING_DIM: int = 64
DEFAULT_DISTANCE_METRIC: str = "euclidean"
REFERENCE_CLASS: int = 0  # Class 0 = Normal baseline condition
RANDOM_SEED: int = 42

# Paths
TABLE_DIR: str = os.path.join("results", "tables")
FIGURE_DIR: str = os.path.join("results", "figures")

NOVELTY_SUMMARY_PATH: str = os.path.join(TABLE_DIR, "novelty_scores_summary.csv")
NOVELTY_DIST_FIG_PATH: str = os.path.join(FIGURE_DIR, "novelty_score_distribution.png")
NOVELTY_BY_CLASS_FIG_PATH: str = os.path.join(FIGURE_DIR, "novelty_by_class.png")
