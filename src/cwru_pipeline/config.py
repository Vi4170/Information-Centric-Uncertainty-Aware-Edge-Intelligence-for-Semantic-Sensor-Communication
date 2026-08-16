"""Configuration constants for CWRU dataset preprocessing and windowing pipeline."""

import os

# Signal and Windowing Parameters
WINDOW_SIZE: int = 2048
STEP_SIZE: int = 2048  # Non-overlapping windows by default
PREFERRED_CHANNEL: str = "DE"
EXPECTED_SAMPLING_RATE: int = 12000  # 12k Hz Drive End baseline

# Baseline 4-Class Classification Mapping
BASELINE_FAULT_CLASSES: dict = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Ball Fault",
    3: "Outer Race Fault",
}

# Recording-level split target ratios
TARGET_SPLIT_RATIOS: dict = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Reproducibility Seed
RANDOM_SEED: int = 42

# Directory Paths
RAW_DATA_DIR: str = os.path.join("data", "raw", "cwru")
PROCESSED_DATA_DIR: str = os.path.join("data", "processed", "cwru")
FIGURE_DIR: str = os.path.join("results", "figures")
