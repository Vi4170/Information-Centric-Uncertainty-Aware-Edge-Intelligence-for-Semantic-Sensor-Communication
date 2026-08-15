from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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

MODEL_FILE = (
    PROJECT_ROOT
    / "saved_models"
    / "cnn_baseline.keras"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "saved_models"
    / "cnn_baseline_metadata.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "voi"

OUTPUT_FILE = OUTPUT_DIR / "voi_results.npz"

METRICS_FILE = OUTPUT_DIR / "voi_metrics.json"


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = [
    "Ball Fault",
    "Inner Race Fault",
    "Normal",
    "Outer Race Fault",
]

# Transmission thresholds.
# A sample is transmitted when its uncertainty is
# greater than or equal to the threshold.
UNCERTAINTY_THRESHOLDS = [
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


# ============================================================
# HELPERS
# ============================================================

def load_training_normalization():
    """
    Load the normalization parameters saved during
    baseline training.
    """

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    metadata = np.load(METADATA_FILE)

    mean = float(metadata["mean"])
    std = float(metadata["std"])

    print(f"Training mean: {mean:.8f}")
    print(f"Training std:  {std:.8f}")

    return mean, std


def normalize_signals(X, mean, std):
    """
    Apply the exact same normalization used during
    baseline training.
    """

    X = X.astype(np.float32)

    X = (X - mean) / std

    return X


def calculate_entropy(probabilities):
    """
    Calculate normalized predictive entropy.

    Entropy is normalized to [0, 1]:

        H(p) = -sum(p * log(p)) / log(number_of_classes)

    0   = completely confident prediction
    1   = maximum uncertainty
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

    max_entropy = np.log(probabilities.shape[1])

    normalized_entropy = entropy / max_entropy

    return normalized_entropy.astype(np.float32)


def calculate_margin(probabilities):
    """
    Calculate prediction margin.

    Margin = largest probability - second-largest probability.

    Large margin = confident prediction.
    Small margin = uncertain prediction.
    """

    sorted_probabilities = np.sort(
        probabilities,
        axis=1
    )

    margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    return margin.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VALUE OF INFORMATION (VoI) ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print("\nChecking required files...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Model metadata not found:\n{METADATA_FILE}"
        )

    print("Dataset found.")
    print("Model found.")
    print("Metadata found.")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    data = np.load(DATA_FILE)

    X_test = data["X_test"]
    y_test_raw = data["y_test"]

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test_raw.shape}")

    # --------------------------------------------------------
    # CONVERT LABELS
    # --------------------------------------------------------

    print("\nConverting labels...")

    label_to_index = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }

    y_test = np.array(
        [label_to_index[label] for label in y_test_raw],
        dtype=np.int64
    )

    print("Label mapping:")

    for index, name in enumerate(CLASS_NAMES):

        count = np.sum(y_test == index)

        print(
            f"  {index}: {name} -> {count} samples"
        )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    print("\nLoading training normalization parameters...")

    train_mean, train_std = load_training_normalization()

    print("\nNormalizing test signals...")

    X_test = normalize_signals(
        X_test,
        train_mean,
        train_std
    )

    X_test = X_test[..., np.newaxis]

    print(
        f"CNN input shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading baseline CNN...")

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating model predictions...")

    probabilities = model.predict(
        X_test,
        verbose=1
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    print(
        f"\nPrediction matrix shape: "
        f"{probabilities.shape}"
    )

    print(
        f"Prediction vector shape: "
        f"{predictions.shape}"
    )

    # --------------------------------------------------------
    # BASIC CONFIDENCE
    # --------------------------------------------------------

    confidence = np.max(
        probabilities,
        axis=1
    ).astype(np.float32)

    uncertainty = (
        1.0 - confidence
    ).astype(np.float32)

    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    entropy = calculate_entropy(
        probabilities
    )

    # --------------------------------------------------------
    # MARGIN
    # --------------------------------------------------------

    margin = calculate_margin(
        probabilities
    )

    # --------------------------------------------------------
    # BASELINE PERFORMANCE
    # --------------------------------------------------------

    baseline_accuracy = accuracy_score(
        y_test,
        predictions
    )

    baseline_precision, baseline_recall, baseline_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    print("\n" + "=" * 70)
    print("BASELINE PREDICTION PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy:          {baseline_accuracy:.4f}"
    )

    print(
        f"Weighted precision:{baseline_precision:.4f}"
    )

    print(
        f"Weighted recall:   {baseline_recall:.4f}"
    )

    print(
        f"Weighted F1:       {baseline_f1:.4f}"
    )

    print(
        f"\nMean confidence:   {np.mean(confidence):.4f}"
    )

    print(
        f"Mean uncertainty:  {np.mean(uncertainty):.4f}"
    )

    print(
        f"Mean entropy:      {np.mean(entropy):.4f}"
    )

    print(
        f"Mean margin:       {np.mean(margin):.4f}"
    )

    # --------------------------------------------------------
    # VOI / TRANSMISSION ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VOI TRANSMISSION ANALYSIS")
    print("=" * 70)

    print(
        "\nRule:"
    )

    print(
        "Transmit sample if uncertainty >= threshold."
    )

    print(
        "\nThreshold | Transmitted | Rate | "
        "Local Accuracy | Overall Accuracy"
    )

    print("-" * 70)

    results = []

    for threshold in UNCERTAINTY_THRESHOLDS:

        transmit_mask = (
            uncertainty >= threshold
        )

        local_mask = ~transmit_mask

        transmitted_count = int(
            np.sum(transmit_mask)
        )

        transmitted_rate = (
            transmitted_count
            / len(y_test)
        )

        # ----------------------------------------------------
        # Overall prediction
        #
        # In this simulation:
        #
        # transmitted samples:
        #     baseline model prediction
        #
        # local samples:
        #     baseline model prediction
        #
        # This establishes the uncertainty/transmission
        # relationship first. A later edge/cloud model can
        # replace the transmitted path.
        # ----------------------------------------------------

        overall_predictions = predictions.copy()

        overall_accuracy = accuracy_score(
            y_test,
            overall_predictions
        )

        # Accuracy among samples selected for transmission
        if transmitted_count > 0:

            transmitted_accuracy = accuracy_score(
                y_test[transmit_mask],
                predictions[transmit_mask]
            )

        else:

            transmitted_accuracy = np.nan

        # Accuracy among locally retained samples
        local_count = int(
            np.sum(local_mask)
        )

        if local_count > 0:

            local_accuracy = accuracy_score(
                y_test[local_mask],
                predictions[local_mask]
            )

        else:

            local_accuracy = np.nan

        # ----------------------------------------------------
        # Communication reduction
        # ----------------------------------------------------

        communication_reduction = (
            1.0 - transmitted_rate
        )

        result = {
            "threshold": float(threshold),
            "transmitted_samples": transmitted_count,
            "total_samples": int(len(y_test)),
            "transmission_rate": float(transmitted_rate),
            "communication_reduction": float(
                communication_reduction
            ),
            "local_samples": local_count,
            "local_accuracy": (
                float(local_accuracy)
                if not np.isnan(local_accuracy)
                else None
            ),
            "transmitted_accuracy": (
                float(transmitted_accuracy)
                if not np.isnan(transmitted_accuracy)
                else None
            ),
            "overall_accuracy": float(
                overall_accuracy
            ),
        }

        results.append(result)

        local_text = (
            f"{local_accuracy:.4f}"
            if not np.isnan(local_accuracy)
            else "N/A"
        )

        print(
            f"{threshold:9.2f} | "
            f"{transmitted_count:11d} | "
            f"{transmitted_rate:4.1%} | "
            f"{local_text:14} | "
            f"{overall_accuracy:.4f}"
        )

    # --------------------------------------------------------
    # SAVE RAW VOI DATA
    # --------------------------------------------------------

    print("\nSaving VoI data...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FILE,
        y_true=y_test,
        predictions=predictions,
        probabilities=probabilities,
        confidence=confidence,
        uncertainty=uncertainty,
        entropy=entropy,
        margin=margin,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "baseline": {
            "accuracy": float(baseline_accuracy),
            "weighted_precision": float(
                baseline_precision
            ),
            "weighted_recall": float(
                baseline_recall
            ),
            "weighted_f1": float(
                baseline_f1
            ),
        },
        "uncertainty_statistics": {
            "mean_confidence": float(
                np.mean(confidence)
            ),
            "mean_uncertainty": float(
                np.mean(uncertainty)
            ),
            "mean_entropy": float(
                np.mean(entropy)
            ),
            "mean_margin": float(
                np.mean(margin)
            ),
        },
        "voi_results": results,
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
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VOI ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {METRICS_FILE}"
    )

    print("\nImportant:")
    print(
        "This first experiment measures uncertainty-based "
        "sample selection."
    )

    print(
        "The next stage will introduce a separate "
        "local/edge decision path so that transmission "
        "can actually trade communication against "
        "diagnostic performance."
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()