"""Run the baseline CNN uncertainty estimation pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras

from src.cnn.config import MODEL_PATH
from src.cnn.model import predict_probabilities
from src.cnn.train import load_cwru_dataset

from src.uncertainty.entropy_uncertainty import (
    normalized_entropy,
    validate_probabilities,
)


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
    / "uncertainty"
)

CSV_FILE = (
    OUTPUT_DIR
    / "uncertainty_scores.csv"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "uncertainty_score_distribution.png"
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
    print("CNN ENTROPY-BASED UNCERTAINTY ESTIMATION")
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
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("\nLoading CWRU dataset...")

    (
        _X_train,
        _y_train,
        _X_val,
        _y_val,
        X_test,
        y_test,
    ) = load_cwru_dataset(
        str(DATA_PATH)
    )

    print(
        f"Test data: {X_test.shape}"
    )

    # --------------------------------------------------------
    # EXISTING CNN PROBABILITY INTERFACE
    # --------------------------------------------------------

    print(
        "\nGenerating 4-class CNN probabilities..."
    )

    probabilities = predict_probabilities(
        model,
        X_test,
    )

    print(
        f"Probability matrix: "
        f"{probabilities.shape}"
    )

    # --------------------------------------------------------
    # VALIDATE PROBABILITY MATRIX
    # --------------------------------------------------------

    print(
        "\nValidating probability vectors..."
    )

    validate_probabilities(
        probabilities,
        num_classes=4,
        tolerance=1e-6,
    )

    print(
        "Probability validation passed."
    )

    # Explicit checks required by Task 4

    if probabilities.shape[1] != 4:
        raise ValueError(
            "Every prediction must contain exactly "
            "4 probabilities."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Probability matrix contains NaN or Inf."
        )

    if np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError(
            "Probabilities must be in [0, 1]."
        )

    row_sums = np.sum(
        probabilities,
        axis=1,
    )

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Probability rows do not sum to 1."
        )

    # --------------------------------------------------------
    # PREDICTED CLASS
    # --------------------------------------------------------

    predicted_classes = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    predicted_class_names = [
        CLASS_NAMES[int(index)]
        for index in predicted_classes
    ]

    # --------------------------------------------------------
    # NORMALIZED ENTROPY
    # --------------------------------------------------------

    print(
        "\nCalculating normalized prediction entropy..."
    )

    uncertainty_scores = normalized_entropy(
        probabilities
    )

    # --------------------------------------------------------
    # VALIDATE UNCERTAINTY
    # --------------------------------------------------------

    if not np.isfinite(
        uncertainty_scores
    ).all():
        raise ValueError(
            "Uncertainty scores contain NaN or Inf."
        )

    if np.any(
        uncertainty_scores < 0.0
    ) or np.any(
        uncertainty_scores > 1.0
    ):
        raise ValueError(
            "Uncertainty scores must be in [0, 1]."
        )

    print(
        "Uncertainty validation passed."
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "observation_id": np.arange(
                len(X_test)
            ),

            "predicted_class":
                predicted_class_names,

            "true_label": [
                CLASS_NAMES[int(label)]
                for label in y_test
            ],

            "prob_normal": probabilities[:, 0],

            "prob_ball_fault": probabilities[:, 1],

            "prob_inner_race_fault":
                probabilities[:, 2],

            "prob_outer_race_fault":
                probabilities[:, 3],

            "uncertainty_score":
                uncertainty_scores,
        }
    )

    results.to_csv(
        CSV_FILE,
        index=False,
    )

    print(
        f"\nSaved uncertainty CSV:"
        f"\n{CSV_FILE}"
    )

    # --------------------------------------------------------
    # UNCERTAINTY DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nGenerating uncertainty distribution..."
    )

    maximum_observed = float(
        np.max(
            uncertainty_scores
        )
    )

    # Zoom the x-axis to the observed range while
    # preserving the fact that the underlying score
    # is normalized to [0, 1].
    plot_upper = max(
        0.03,
        maximum_observed * 1.10,
    )

    plot_upper = min(
        plot_upper,
        1.0,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        uncertainty_scores,
        bins=30,
        alpha=0.75,
    )

    plt.xlabel(
        "Normalized Predictive Entropy",
        fontsize=12,
    )

    plt.ylabel(
        "Number of Test Observations",
        fontsize=12,
    )

    plt.title(
        "CNN Uncertainty-Score Distribution",
        fontsize=14,
    )

    plt.xlim(
        0.0,
        plot_upper,
    )

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
        f"Saved uncertainty distribution:"
        f"\n{PLOT_FILE}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("UNCERTAINTY ESTIMATION COMPLETE")
    print("=" * 70)

    print(
        f"Test observations: "
        f"{len(uncertainty_scores)}"
    )

    print(
        f"Mean uncertainty: "
        f"{np.mean(uncertainty_scores):.4f}"
    )

    print(
        f"Median uncertainty: "
        f"{np.median(uncertainty_scores):.4f}"
    )

    print(
        f"Minimum uncertainty: "
        f"{np.min(uncertainty_scores):.4f}"
    )

    print(
        f"Maximum uncertainty: "
        f"{np.max(uncertainty_scores):.4f}"
    )

    print("\nGenerated:")
    print(CSV_FILE)
    print(PLOT_FILE)


if __name__ == "__main__":
    main()