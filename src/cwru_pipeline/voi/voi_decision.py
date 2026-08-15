from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "CWRU"
    / "cwru_splits.npz"
)

EDGE_MODEL_FILE = (
    PROJECT_ROOT
    / "saved_models"
    / "cnn_baseline.keras"
)

CLOUD_MODEL_FILE = (
    PROJECT_ROOT
    / "saved_models"
    / "cloud"
    / "cnn_cloud.keras"
)

EDGE_METADATA_FILE = (
    PROJECT_ROOT
    / "saved_models"
    / "cnn_baseline_metadata.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "voi"
)

RESULTS_FILE = OUTPUT_DIR / "voi_decision_results.npz"
METRICS_FILE = OUTPUT_DIR / "voi_decision_metrics.json"
PLOT_FILE = OUTPUT_DIR / "voi_accuracy_vs_transmission.png"


# ============================================================
# SETTINGS
# ============================================================

THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]


CLASS_NAMES = [
    "Ball Fault",
    "Inner Race Fault",
    "Normal",
    "Outer Race Fault",
]


# ============================================================
# HELPERS
# ============================================================

def calculate_entropy(probabilities):
    """
    Calculate normalized predictive entropy.

    Entropy is normalized by log(number_of_classes),
    giving a value approximately between 0 and 1.
    """

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    entropy = -np.sum(
        probabilities * np.log(probabilities),
        axis=1
    )

    entropy /= np.log(probabilities.shape[1])

    return entropy


def calculate_uncertainty(probabilities):
    """
    Confidence-based uncertainty.

    uncertainty = 1 - maximum class probability
    """

    confidence = np.max(
        probabilities,
        axis=1
    )

    uncertainty = 1.0 - confidence

    return confidence, uncertainty


def normalize_signals(X, mean, std):
    """
    Normalize signals using training-set statistics.
    """

    return (
        (X.astype(np.float32) - mean)
        / std
    ).astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VALUE OF INFORMATION DECISION ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print("\nChecking required files...")

    required_files = [
        DATA_FILE,
        EDGE_MODEL_FILE,
        CLOUD_MODEL_FILE,
        EDGE_METADATA_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

        print(f"Found: {file_path}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    data = np.load(DATA_FILE)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Training data: {X_train.shape}")
    print(f"Test data:     {X_test.shape}")

    # --------------------------------------------------------
    # LABEL MAPPING
    # --------------------------------------------------------

    print("\nConverting labels...")

    label_to_index = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }

    y_test_int = np.asarray(
        [
            label_to_index[label]
            for label in y_test
        ],
        dtype=np.int64
    )

    print("\nClasses:")

    for index, name in enumerate(CLASS_NAMES):

        count = np.sum(
            y_test_int == index
        )

        print(
            f"  {index}: {name} -> {count}"
        )

    # --------------------------------------------------------
    # LOAD NORMALIZATION PARAMETERS
    # --------------------------------------------------------

    print("\nLoading training normalization parameters...")

    metadata = np.load(
        EDGE_METADATA_FILE
    )

    mean = float(
        metadata["mean"]
    )

    std = float(
        metadata["std"]
    )

    print(
        f"Training mean: {mean:.8f}"
    )

    print(
        f"Training std:  {std:.8f}"
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    print("\nNormalizing test signals...")

    X_test_norm = normalize_signals(
        X_test,
        mean,
        std
    )

    X_test_cnn = X_test_norm[..., np.newaxis]

    print(
        f"CNN input shape: {X_test_cnn.shape}"
    )

    # --------------------------------------------------------
    # LOAD EDGE MODEL
    # --------------------------------------------------------

    print("\nLoading edge model...")

    edge_model = tf.keras.models.load_model(
        EDGE_MODEL_FILE
    )

    print("Edge model loaded.")

    # --------------------------------------------------------
    # LOAD CLOUD MODEL
    # --------------------------------------------------------

    print("\nLoading cloud model...")

    cloud_model = tf.keras.models.load_model(
        CLOUD_MODEL_FILE
    )

    print("Cloud model loaded.")

    # --------------------------------------------------------
    # EDGE PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating edge predictions...")

    edge_probabilities = edge_model.predict(
        X_test_cnn,
        verbose=1
    )

    edge_predictions = np.argmax(
        edge_probabilities,
        axis=1
    )

    print(
        f"Edge prediction matrix: "
        f"{edge_probabilities.shape}"
    )

    # --------------------------------------------------------
    # CLOUD PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating cloud predictions...")

    cloud_probabilities = cloud_model.predict(
        X_test_cnn,
        verbose=1
    )

    cloud_predictions = np.argmax(
        cloud_probabilities,
        axis=1
    )

    print(
        f"Cloud prediction matrix: "
        f"{cloud_probabilities.shape}"
    )

    # --------------------------------------------------------
    # EDGE UNCERTAINTY
    # --------------------------------------------------------

    edge_confidence, edge_uncertainty = (
        calculate_uncertainty(
            edge_probabilities
        )
    )

    edge_entropy = calculate_entropy(
        edge_probabilities
    )

    # --------------------------------------------------------
    # BASELINE PERFORMANCE
    # --------------------------------------------------------

    edge_accuracy = accuracy_score(
        y_test_int,
        edge_predictions
    )

    cloud_accuracy = accuracy_score(
        y_test_int,
        cloud_predictions
    )

    print("\n" + "=" * 70)
    print("EDGE / CLOUD BASELINE PERFORMANCE")
    print("=" * 70)

    print(
        f"\nEdge accuracy:  "
        f"{edge_accuracy:.4f}"
    )

    print(
        f"Cloud accuracy: "
        f"{cloud_accuracy:.4f}"
    )

    print(
        f"Cloud improvement: "
        f"{cloud_accuracy - edge_accuracy:+.4f}"
    )

    print(
        f"\nMean edge confidence: "
        f"{np.mean(edge_confidence):.4f}"
    )

    print(
        f"Mean edge uncertainty: "
        f"{np.mean(edge_uncertainty):.4f}"
    )

    print(
        f"Mean edge entropy: "
        f"{np.mean(edge_entropy):.4f}"
    )

    # --------------------------------------------------------
    # VOI DECISION ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VOI EDGE / CLOUD DECISION ANALYSIS")
    print("=" * 70)

    print(
        "\nDecision rule:"
    )

    print(
        "  If edge uncertainty >= threshold:"
    )

    print(
        "      TRANSMIT -> use cloud prediction"
    )

    print(
        "  Otherwise:"
    )

    print(
        "      KEEP LOCAL -> use edge prediction"
    )

    print("\n")

    header = (
        "Threshold | "
        "Transmitted | "
        "Rate | "
        "Final Acc | "
        "Final F1 | "
        "Accuracy Gain"
    )

    print(header)
    print("-" * len(header))

    transmission_rates = []
    final_accuracies = []
    final_f1s = []
    accuracy_gains = []

    threshold_results = {}

    for threshold in THRESHOLDS:

        transmit_mask = (
            edge_uncertainty >= threshold
        )

        final_predictions = (
            edge_predictions.copy()
        )

        final_predictions[
            transmit_mask
        ] = cloud_predictions[
            transmit_mask
        ]

        transmitted = int(
            np.sum(transmit_mask)
        )

        transmission_rate = (
            transmitted
            / len(y_test_int)
        )

        final_accuracy = accuracy_score(
            y_test_int,
            final_predictions
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test_int,
                final_predictions,
                average="weighted",
                zero_division=0
            )
        )

        accuracy_gain = (
            final_accuracy
            - edge_accuracy
        )

        transmission_rates.append(
            transmission_rate
        )

        final_accuracies.append(
            final_accuracy
        )

        final_f1s.append(
            f1
        )

        accuracy_gains.append(
            accuracy_gain
        )

        threshold_results[str(threshold)] = {
            "threshold": threshold,
            "transmitted_samples": transmitted,
            "transmission_rate": transmission_rate,
            "communication_savings": (
                1.0 - transmission_rate
            ),
            "final_accuracy": final_accuracy,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1,
            "accuracy_gain": accuracy_gain,
        }

        print(
            f"{threshold:9.2f} | "
            f"{transmitted:11d} | "
            f"{transmission_rate * 100:5.1f}% | "
            f"{final_accuracy:9.4f} | "
            f"{f1:8.4f} | "
            f"{accuracy_gain:+.4f}"
        )

    # --------------------------------------------------------
    # FIND BEST OPERATING POINTS
    # --------------------------------------------------------

    best_accuracy_index = int(
        np.argmax(final_accuracies)
    )

    best_f1_index = int(
        np.argmax(final_f1s)
    )

    print("\n" + "=" * 70)
    print("BEST VOI OPERATING POINTS")
    print("=" * 70)

    best_accuracy_threshold = (
        THRESHOLDS[
            best_accuracy_index
        ]
    )

    print(
        "\nBest final accuracy:"
    )

    print(
        f"  Threshold: "
        f"{best_accuracy_threshold:.2f}"
    )

    print(
        f"  Accuracy: "
        f"{final_accuracies[best_accuracy_index]:.4f}"
    )

    print(
        f"  Transmission rate: "
        f"{transmission_rates[best_accuracy_index] * 100:.1f}%"
    )

    print(
        f"  Communication savings: "
        f"{(1 - transmission_rates[best_accuracy_index]) * 100:.1f}%"
    )

    best_f1_threshold = (
        THRESHOLDS[
            best_f1_index
        ]
    )

    print(
        "\nBest weighted F1:"
    )

    print(
        f"  Threshold: "
        f"{best_f1_threshold:.2f}"
    )

    print(
        f"  Weighted F1: "
        f"{final_f1s[best_f1_index]:.4f}"
    )

    print(
        f"  Transmission rate: "
        f"{transmission_rates[best_f1_index] * 100:.1f}%"
    )

    # --------------------------------------------------------
    # SAVE NUMPY RESULTS
    # --------------------------------------------------------

    print("\nSaving VoI decision results...")

    np.savez_compressed(
        RESULTS_FILE,

        y_test=y_test_int,

        edge_probabilities=edge_probabilities,
        cloud_probabilities=cloud_probabilities,

        edge_predictions=edge_predictions,
        cloud_predictions=cloud_predictions,

        edge_confidence=edge_confidence,
        edge_uncertainty=edge_uncertainty,
        edge_entropy=edge_entropy,

        thresholds=np.asarray(
            THRESHOLDS,
            dtype=np.float32
        ),

        transmission_rates=np.asarray(
            transmission_rates,
            dtype=np.float32
        ),

        final_accuracies=np.asarray(
            final_accuracies,
            dtype=np.float32
        ),

        final_f1s=np.asarray(
            final_f1s,
            dtype=np.float32
        ),

        accuracy_gains=np.asarray(
            accuracy_gains,
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # SAVE JSON METRICS
    # --------------------------------------------------------

    metrics = {
        "edge_accuracy": float(
            edge_accuracy
        ),

        "cloud_accuracy": float(
            cloud_accuracy
        ),

        "cloud_accuracy_gain": float(
            cloud_accuracy - edge_accuracy
        ),

        "mean_edge_confidence": float(
            np.mean(edge_confidence)
        ),

        "mean_edge_uncertainty": float(
            np.mean(edge_uncertainty)
        ),

        "mean_edge_entropy": float(
            np.mean(edge_entropy)
        ),

        "best_accuracy_threshold": float(
            best_accuracy_threshold
        ),

        "best_accuracy": float(
            final_accuracies[
                best_accuracy_index
            ]
        ),

        "best_accuracy_transmission_rate": float(
            transmission_rates[
                best_accuracy_index
            ]
        ),

        "best_accuracy_communication_savings": float(
            1.0
            - transmission_rates[
                best_accuracy_index
            ]
        ),

        "best_f1_threshold": float(
            best_f1_threshold
        ),

        "best_weighted_f1": float(
            final_f1s[
                best_f1_index
            ]
        ),

        "threshold_results": threshold_results,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    print("\nCreating accuracy/transmission plot...")

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        np.asarray(transmission_rates) * 100,
        np.asarray(final_accuracies) * 100,
        marker="o"
    )

    plt.axhline(
        edge_accuracy * 100,
        linestyle="--",
        label="Edge-only accuracy"
    )

    plt.axhline(
        cloud_accuracy * 100,
        linestyle=":",
        label="Cloud-only accuracy"
    )

    plt.xlabel(
        "Transmission Rate (%)"
    )

    plt.ylabel(
        "Final Accuracy (%)"
    )

    plt.title(
        "VoI Edge-Cloud Accuracy vs Transmission Rate"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VOI DECISION ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"  {RESULTS_FILE}"
    )

    print(
        f"  {METRICS_FILE}"
    )

    print(
        f"  {PLOT_FILE}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()