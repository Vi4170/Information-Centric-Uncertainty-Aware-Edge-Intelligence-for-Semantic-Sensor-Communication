"""Configuration constants for CWRU baseline CNN model and training pipeline."""

import os
from typing import Tuple

# Input & Target Parameters
INPUT_SHAPE: Tuple[int, int] = (2048, 1)
NUM_CLASSES: int = 4
EMBEDDING_DIM: int = 64
EMBEDDING_LAYER_NAME: str = "learned_embedding"

# Training Hyperparameters
BATCH_SIZE: int = 32
EPOCHS: int = 25
LEARNING_RATE: float = 0.001
RANDOM_SEED: int = 42

# Paths
DATA_PATH: str = os.path.join("data", "processed", "cwru", "cwru_dataset_v1.npz")
MODEL_DIR: str = "models"
MODEL_PATH: str = os.path.join(MODEL_DIR, "cwru_cnn_baseline.keras")

FIGURE_DIR: str = os.path.join("results", "figures")
TABLE_DIR: str = os.path.join("results", "tables")
TRAINING_HISTORY_PATH: str = os.path.join(TABLE_DIR, "cnn_training_history.csv")
