"""Run the CNN-embedding novelty detection pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras

from src.cnn.config import MODEL_PATH
from src.cnn.model import extract_embeddings
from src.cnn.train import load_cwru_dataset
from src.novelty.distance_novelty import MahalanobisNoveltyDetector


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru"
    / "cwru_dataset_v1.npz"
)

MODEL_FILE = PROJECT_ROOT / MODEL_PATH

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "novelty"
)

CSV_FILE = OUTPUT_DIR / "novelty_scores.csv"

PLOT_FILE = (
    OUTPUT_DIR
    / "novelty_score_distribution.png"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Normal",
    "Inner Race Fault",
    "Ball Fault",
    "Outer Race Fault",
]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CNN EMBEDDING NOVELTY DETECTION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    # --------------------------------------------------------
    # LOAD TRAINED CNN
    # --------------------------------------------------------

    print("\nLoading trained CNN...")

    model = keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )

    print("CNN loaded successfully.")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading CWRU dataset...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_cwru_dataset(
        str(DATA_PATH)
    )

    print(f"Train:      {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    # --------------------------------------------------------
    # EXTRACT EXISTING CNN EMBEDDINGS
    # --------------------------------------------------------

    print("\nExtracting existing 64-D CNN embeddings...")

    train_embeddings = extract_embeddings(
        model,
        X_train,
    )

    val_embeddings = extract_embeddings(
        model,
        X_val,
    )

    test_embeddings = extract_embeddings(
        model,
        X_test,
    )

    print(
        f"Train embeddings:      {train_embeddings.shape}"
    )

    print(
        f"Validation embeddings: {val_embeddings.shape}"
    )

    print(
        f"Test embeddings:       {test_embeddings.shape}"
    )

    # --------------------------------------------------------
    # VERIFY 64-D INTERFACE
    # --------------------------------------------------------

    if train_embeddings.ndim != 2:
        raise ValueError(
            "Training embeddings must be a 2D array."
        )

    if train_embeddings.shape[1] != 64:
        raise ValueError(
            "Expected 64-D embeddings, got "
            f"{train_embeddings.shape[1]}."
        )

    # --------------------------------------------------------
    # FIT REFERENCE DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nFitting novelty reference distribution "
        "using TRAINING data only..."
    )

    detector = MahalanobisNoveltyDetector(
        regularization=1e-5
    )

    detector.fit(
        train_embeddings
    )

    print("Reference distribution fitted.")

    # --------------------------------------------------------
    # CALCULATE SCORES
    # --------------------------------------------------------

    print("\nCalculating novelty scores...")

    train_scores = detector.novelty_score(
        train_embeddings
    )

    val_scores = detector.novelty_score(
        val_embeddings
    )

    test_scores = detector.novelty_score(
        test_embeddings
    )

    # --------------------------------------------------------
    # VALIDATE SCORES
    # --------------------------------------------------------

    for name, scores in [
        ("Training", train_scores),
        ("Validation", val_scores),
        ("Test", test_scores),
    ]:

        if not np.isfinite(scores).all():
            raise ValueError(
                f"{name} novelty scores contain NaN or Inf."
            )

        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError(
                f"{name} novelty scores are outside [0, 1]."
            )

        print(
            f"{name} score range: "
            f"{scores.min():.4f} - "
            f"{scores.max():.4f}"
        )

    # --------------------------------------------------------
    # SAVE TEST CSV
    # --------------------------------------------------------

    test_label_names = [
        CLASS_NAMES[int(label)]
        for label in y_test
    ]

    results = pd.DataFrame(
        {
            "observation_id": np.arange(
                len(test_scores)
            ),

            "true_label": test_label_names,

            "novelty_score": test_scores,
        }
    )

    results.to_csv(
        CSV_FILE,
        index=False,
    )

    print(
        f"\nSaved test novelty scores:"
        f"\n{CSV_FILE}"
    )

    # --------------------------------------------------------
    # CLASS-WISE NOVELTY DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nGenerating class-wise novelty distribution..."
    )

    plt.figure(
        figsize=(10, 6)
    )

    for label_id, class_name in enumerate(CLASS_NAMES):

        class_scores = test_scores[
            y_test == label_id
        ]

        if len(class_scores) == 0:
            continue

        plt.hist(
            class_scores,
            bins=20,
            alpha=0.45,
            label=class_name,
        )

    plt.xlabel(
        "Novelty Score",
        fontsize=12,
    )

    plt.ylabel(
        "Number of Test Observations",
        fontsize=12,
    )

    plt.title(
        "CNN Embedding Novelty-Score Distribution by Class",
        fontsize=14,
    )

    plt.xlim(
        0.0,
        1.0,
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved novelty distribution:"
        f"\n{PLOT_FILE}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NOVELTY DETECTION COMPLETE")
    print("=" * 70)

    print(
        f"Embedding dimension: 64"
    )

    print(
        f"Test observations: "
        f"{len(test_scores)}"
    )

    print(
        f"Mean test novelty: "
        f"{np.mean(test_scores):.4f}"
    )

    print(
        f"Median test novelty: "
        f"{np.median(test_scores):.4f}"
    )

    print(
        f"Minimum test novelty: "
        f"{np.min(test_scores):.4f}"
    )

    print(
        f"Maximum test novelty: "
        f"{np.max(test_scores):.4f}"
    )

    print("\nGenerated:")
    print(CSV_FILE)
    print(PLOT_FILE)


if __name__ == "__main__":
    main()