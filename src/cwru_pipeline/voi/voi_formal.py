from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.isotonic import IsotonicRegression


# ============================================================
# FORMAL VALUE OF INFORMATION
# COST-AWARE PER-SAMPLE EXPECTED-UTILITY ANALYSIS
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

CLOUD_METADATA_FILE = (
    PROJECT_ROOT
    / "saved_models"
    / "cloud"
    / "cnn_cloud_metadata.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "voi"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "formal_voi_results.npz"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "formal_voi_metrics.json"
)


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = [
    "Ball Fault",
    "Inner Race Fault",
    "Normal",
    "Outer Race Fault",
]

COMMUNICATION_COSTS = np.asarray(
    [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
    ],
    dtype=np.float32,
)


# ============================================================
# HELPERS
# ============================================================

def load_normalization_parameters(metadata_file):
    """
    Load training normalization parameters.
    """

    metadata = np.load(metadata_file)

    mean = float(metadata["mean"])
    std = float(metadata["std"])

    return mean, std


def normalize_signals(X, mean, std):
    """
    Apply training-set normalization.
    """

    X = X.astype(np.float32)

    return (
        (X - mean) / std
    ).astype(np.float32)


def labels_to_indices(labels):
    """
    Convert string class labels to integer indices.
    """

    label_to_index = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }

    return np.asarray(
        [
            label_to_index[label]
            for label in labels
        ],
        dtype=np.int64,
    )


def calculate_entropy(probabilities):
    """
    Normalized predictive entropy.
    """

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        probabilities * np.log(probabilities),
        axis=1,
    )

    entropy /= np.log(
        probabilities.shape[1]
    )

    return entropy.astype(np.float32)


def calculate_confidence(probabilities):
    """
    Maximum predicted class probability.
    """

    return np.max(
        probabilities,
        axis=1,
    ).astype(np.float32)


def calculate_expected_utility(probabilities):
    """
    For a 0/1 classification utility:

        U(correct) = 1
        U(incorrect) = 0

    The expected utility of choosing the most
    probable class is therefore its probability.
    """

    return calculate_confidence(probabilities)


def evaluate_predictions(y_true, predictions):
    """
    Return standard classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
    }


def fit_utility_calibrator(
    raw_utility,
    correctness,
):
    """
    Learn a validation-set calibration mapping:

        raw confidence -> empirical probability of correctness

    Isotonic regression is used because it is non-parametric
    and preserves monotonicity.
    """

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        raw_utility,
        correctness.astype(np.float32),
    )

    return calibrator


def safe_calibrated_values(
    calibrator,
    values,
):
    """
    Apply calibration and constrain values to [0,1].
    """

    calibrated = calibrator.predict(
        values.astype(np.float64)
    )

    return np.clip(
        np.asarray(calibrated, dtype=np.float32),
        0.0,
        1.0,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "FORMAL VALUE OF INFORMATION ANALYSIS"
    )
    print(
        "PER-SAMPLE EXPECTED-UTILITY FORMULATION"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
        CLOUD_METADATA_FILE,
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

    print("\nLoading dataset...")

    data = np.load(
        DATA_FILE,
        allow_pickle=True,
    )

    X_val = data["X_val"]
    y_val = labels_to_indices(
        data["y_val"]
    )

    X_test = data["X_test"]
    y_test = labels_to_indices(
        data["y_test"]
    )

    print(
        f"Validation data: {X_val.shape}"
    )

    print(
        f"Test data:       {X_test.shape}"
    )

    # --------------------------------------------------------
    # LOAD NORMALIZATION
    # --------------------------------------------------------

    print(
        "\nLoading normalization parameters..."
    )

    edge_mean, edge_std = (
        load_normalization_parameters(
            EDGE_METADATA_FILE
        )
    )

    cloud_mean, cloud_std = (
        load_normalization_parameters(
            CLOUD_METADATA_FILE
        )
    )

    print(
        f"Edge mean:   {edge_mean:.8f}"
    )

    print(
        f"Edge std:    {edge_std:.8f}"
    )

    print(
        f"Cloud mean:  {cloud_mean:.8f}"
    )

    print(
        f"Cloud std:   {cloud_std:.8f}"
    )

    normalization_match = (
        np.isclose(
            edge_mean,
            cloud_mean,
            atol=1e-6,
        )
        and
        np.isclose(
            edge_std,
            cloud_std,
            atol=1e-6,
        )
    )

    if normalization_match:

        print(
            "\nNormalization parameters match."
        )

    else:

        print(
            "\nWARNING:"
        )

        print(
            "Edge and cloud normalization parameters differ."
        )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    print(
        "\nNormalizing validation data..."
    )

    X_val_norm = normalize_signals(
        X_val,
        edge_mean,
        edge_std,
    )

    print(
        "Normalizing test data..."
    )

    X_test_norm = normalize_signals(
        X_test,
        edge_mean,
        edge_std,
    )

    X_val_cnn = (
        X_val_norm[..., np.newaxis]
    )

    X_test_cnn = (
        X_test_norm[..., np.newaxis]
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print(
        "\nLoading edge model..."
    )

    edge_model = tf.keras.models.load_model(
        EDGE_MODEL_FILE,
        compile=False,
    )

    print(
        "Edge model loaded."
    )

    print(
        "\nLoading cloud model..."
    )

    cloud_model = tf.keras.models.load_model(
        CLOUD_MODEL_FILE,
        compile=False,
    )

    print(
        "Cloud model loaded."
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "VALIDATION-SET UTILITY CALIBRATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    print(
        "\nGenerating edge validation predictions..."
    )

    edge_val_probabilities = (
        edge_model.predict(
            X_val_cnn,
            verbose=1,
        )
    )

    edge_val_probabilities = np.asarray(
        edge_val_probabilities,
        dtype=np.float32,
    )

    print(
        "\nGenerating cloud validation predictions..."
    )

    cloud_val_probabilities = (
        cloud_model.predict(
            X_val_cnn,
            verbose=1,
        )
    )

    cloud_val_probabilities = np.asarray(
        cloud_val_probabilities,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    edge_val_predictions = np.argmax(
        edge_val_probabilities,
        axis=1,
    )

    cloud_val_predictions = np.argmax(
        cloud_val_probabilities,
        axis=1,
    )

    # --------------------------------------------------------
    # Validation utilities
    # --------------------------------------------------------

    edge_val_raw_utility = (
        calculate_expected_utility(
            edge_val_probabilities
        )
    )

    cloud_val_raw_utility = (
        calculate_expected_utility(
            cloud_val_probabilities
        )
    )

    edge_val_correct = (
        edge_val_predictions == y_val
    ).astype(np.float32)

    cloud_val_correct = (
        cloud_val_predictions == y_val
    ).astype(np.float32)

    # --------------------------------------------------------
    # Validation baseline
    # --------------------------------------------------------

    edge_val_metrics = evaluate_predictions(
        y_val,
        edge_val_predictions,
    )

    cloud_val_metrics = evaluate_predictions(
        y_val,
        cloud_val_predictions,
    )

    print(
        f"\nValidation edge accuracy:  "
        f"{edge_val_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation cloud accuracy: "
        f"{cloud_val_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation accuracy benefit: "
        f"{cloud_val_metrics['accuracy'] - edge_val_metrics['accuracy']:+.4f}"
    )

    # --------------------------------------------------------
    # Calibrate expected utilities
    # --------------------------------------------------------

    print(
        "\nCalibrating edge expected utility..."
    )

    edge_calibrator = fit_utility_calibrator(
        edge_val_raw_utility,
        edge_val_correct,
    )

    print(
        "Calibrating cloud expected utility..."
    )

    cloud_calibrator = fit_utility_calibrator(
        cloud_val_raw_utility,
        cloud_val_correct,
    )

    calibrated_edge_val_utility = (
        safe_calibrated_values(
            edge_calibrator,
            edge_val_raw_utility,
        )
    )

    calibrated_cloud_val_utility = (
        safe_calibrated_values(
            cloud_calibrator,
            cloud_val_raw_utility,
        )
    )

    # --------------------------------------------------------
    # Validation information benefit
    # --------------------------------------------------------

    validation_sample_benefit = (
        calibrated_cloud_val_utility
        - calibrated_edge_val_utility
    )

    print(
        "\nValidation expected-utility statistics:"
    )

    print(
        f"Mean calibrated edge utility:   "
        f"{np.mean(calibrated_edge_val_utility):.4f}"
    )

    print(
        f"Mean calibrated cloud utility:  "
        f"{np.mean(calibrated_cloud_val_utility):.4f}"
    )

    print(
        f"Mean estimated information benefit: "
        f"{np.mean(validation_sample_benefit):+.4f}"
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "HELD-OUT TEST-SET EVALUATION"
    )
    print("=" * 70)

    print(
        "\nGenerating edge test predictions..."
    )

    edge_test_probabilities = (
        edge_model.predict(
            X_test_cnn,
            verbose=1,
        )
    )

    edge_test_probabilities = np.asarray(
        edge_test_probabilities,
        dtype=np.float32,
    )

    print(
        "\nGenerating cloud test predictions..."
    )

    cloud_test_probabilities = (
        cloud_model.predict(
            X_test_cnn,
            verbose=1,
        )
    )

    cloud_test_probabilities = np.asarray(
        cloud_test_probabilities,
        dtype=np.float32,
    )

    edge_test_predictions = np.argmax(
        edge_test_probabilities,
        axis=1,
    )

    cloud_test_predictions = np.argmax(
        cloud_test_probabilities,
        axis=1,
    )

    edge_test_metrics = evaluate_predictions(
        y_test,
        edge_test_predictions,
    )

    cloud_test_metrics = evaluate_predictions(
        y_test,
        cloud_test_predictions,
    )

    print(
        f"\nTest edge accuracy:  "
        f"{edge_test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test cloud accuracy: "
        f"{cloud_test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test cloud improvement: "
        f"{cloud_test_metrics['accuracy'] - edge_test_metrics['accuracy']:+.4f}"
    )

    # ========================================================
    # TEST UTILITY ESTIMATION
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "PER-SAMPLE EXPECTED-UTILITY ESTIMATION"
    )
    print("=" * 70)

    test_confidence = (
        calculate_confidence(
            edge_test_probabilities
        )
    )

    test_uncertainty = (
        1.0 - test_confidence
    ).astype(np.float32)

    test_entropy = calculate_entropy(
        edge_test_probabilities
    )

    # Raw expected utilities
    raw_edge_test_utility = (
        calculate_expected_utility(
            edge_test_probabilities
        )
    )

    raw_cloud_test_utility = (
        calculate_expected_utility(
            cloud_test_probabilities
        )
    )

    # Calibrate using validation-fitted calibrators
    calibrated_edge_test_utility = (
        safe_calibrated_values(
            edge_calibrator,
            raw_edge_test_utility,
        )
    )

    calibrated_cloud_test_utility = (
        safe_calibrated_values(
            cloud_calibrator,
            raw_cloud_test_utility,
        )
    )

    # --------------------------------------------------------
    # Per-sample information benefit
    # --------------------------------------------------------

    estimated_sample_benefit = (
        calibrated_cloud_test_utility
        - calibrated_edge_test_utility
    )

    print(
        f"\nMean calibrated edge utility:   "
        f"{np.mean(calibrated_edge_test_utility):.4f}"
    )

    print(
        f"Mean calibrated cloud utility:  "
        f"{np.mean(calibrated_cloud_test_utility):.4f}"
    )

    print(
        f"Mean estimated information benefit: "
        f"{np.mean(estimated_sample_benefit):+.4f}"
    )

    print(
        f"Positive-benefit sample rate: "
        f"{np.mean(estimated_sample_benefit > 0) * 100:.2f}%"
    )

    # ========================================================
    # COST-AWARE VoI
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "COST-AWARE FORMAL VoI ANALYSIS"
    )
    print("=" * 70)

    print(
        "\nFormal formulation:"
    )

    print(
        "  EU_edge(x)  = calibrated probability of correct"
    )

    print(
        "                edge decision"
    )

    print(
        "  EU_cloud(x) = calibrated probability of correct"
    )

    print(
        "                cloud decision"
    )

    print(
        "  Benefit(x)  = EU_cloud(x) - EU_edge(x)"
    )

    print(
        "  VoI(x)      = Benefit(x) - Communication Cost"
    )

    print(
        "  Transmit when VoI(x) > 0"
    )

    print(
        "\n"
    )

    transmission_masks = []
    estimated_voi_all = []
    final_predictions_all = []

    cost_results = []

    for communication_cost in COMMUNICATION_COSTS:

        estimated_voi = (
            estimated_sample_benefit
            - communication_cost
        )

        transmit_mask = (
            estimated_voi > 0.0
        )

        final_predictions = (
            edge_test_predictions.copy()
        )

        final_predictions[
            transmit_mask
        ] = cloud_test_predictions[
            transmit_mask
        ]

        transmitted_samples = int(
            np.sum(transmit_mask)
        )

        transmission_rate = (
            transmitted_samples
            / len(y_test)
        )

        communication_savings = (
            1.0 - transmission_rate
        )

        final_metrics = evaluate_predictions(
            y_test,
            final_predictions,
        )

        accuracy_gain = (
            final_metrics["accuracy"]
            - edge_test_metrics["accuracy"]
        )

        mean_estimated_voi = float(
            np.mean(estimated_voi)
        )

        positive_voi_rate = float(
            np.mean(
                estimated_voi > 0.0
            )
        )

        print(
            f"Cost = {communication_cost:.2f}"
        )

        print(
            f"  Transmitted:       "
            f"{transmitted_samples}"
        )

        print(
            f"  Transmission rate: "
            f"{transmission_rate * 100:.2f}%"
        )

        print(
            f"  Communication save:"
            f" {communication_savings * 100:.2f}%"
        )

        print(
            f"  Final accuracy:     "
            f"{final_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"  Accuracy gain:      "
            f"{accuracy_gain * 100:+.2f}%"
        )

        print(
            f"  Weighted F1:        "
            f"{final_metrics['weighted_f1']:.4f}"
        )

        print(
            f"  Mean estimated VoI: "
            f"{mean_estimated_voi:+.4f}"
        )

        print(
            f"  Positive VoI rate:  "
            f"{positive_voi_rate * 100:.2f}%"
        )

        print()

        transmission_masks.append(
            transmit_mask
        )

        estimated_voi_all.append(
            estimated_voi
        )

        final_predictions_all.append(
            final_predictions
        )

        cost_results.append(
            {
                "communication_cost": float(
                    communication_cost
                ),

                "transmitted_samples": (
                    transmitted_samples
                ),

                "transmission_rate": float(
                    transmission_rate
                ),

                "communication_savings": float(
                    communication_savings
                ),

                "final_accuracy": float(
                    final_metrics["accuracy"]
                ),

                "weighted_precision": float(
                    final_metrics["weighted_precision"]
                ),

                "weighted_recall": float(
                    final_metrics["weighted_recall"]
                ),

                "weighted_f1": float(
                    final_metrics["weighted_f1"]
                ),

                "accuracy_gain": float(
                    accuracy_gain
                ),

                "mean_estimated_voi": (
                    mean_estimated_voi
                ),

                "positive_voi_rate": (
                    positive_voi_rate
                ),
            }
        )

    # ========================================================
    # OPERATING POINTS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "FORMAL VoI OPERATING POINTS"
    )
    print("=" * 70)

    best_accuracy = max(
        cost_results,
        key=lambda x: x["final_accuracy"],
    )

    best_f1 = max(
        cost_results,
        key=lambda x: x["weighted_f1"],
    )

    saving_points = [
        x
        for x in cost_results
        if x["communication_savings"] > 0
    ]

    best_accuracy_with_saving = max(
        saving_points,
        key=lambda x: x["final_accuracy"],
    )

    print(
        "\nBest final accuracy:"
    )

    print(
        f"  Communication cost: "
        f"{best_accuracy['communication_cost']:.2f}"
    )

    print(
        f"  Accuracy: "
        f"{best_accuracy['final_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Transmission rate: "
        f"{best_accuracy['transmission_rate'] * 100:.2f}%"
    )

    print(
        f"  Communication savings: "
        f"{best_accuracy['communication_savings'] * 100:.2f}%"
    )

    print(
        "\nBest weighted F1:"
    )

    print(
        f"  Communication cost: "
        f"{best_f1['communication_cost']:.2f}"
    )

    print(
        f"  Weighted F1: "
        f"{best_f1['weighted_f1']:.4f}"
    )

    print(
        f"  Accuracy: "
        f"{best_f1['final_accuracy'] * 100:.2f}%"
    )

    print(
        "\nBest accuracy with communication savings:"
    )

    print(
        f"  Communication cost: "
        f"{best_accuracy_with_saving['communication_cost']:.2f}"
    )

    print(
        f"  Accuracy: "
        f"{best_accuracy_with_saving['final_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Transmission rate: "
        f"{best_accuracy_with_saving['transmission_rate'] * 100:.2f}%"
    )

    print(
        f"  Communication savings: "
        f"{best_accuracy_with_saving['communication_savings'] * 100:.2f}%"
    )

    print(
        f"  Accuracy gain vs edge: "
        f"{best_accuracy_with_saving['accuracy_gain'] * 100:+.2f}%"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print(
        "\nSaving formal VoI results..."
    )

    np.savez_compressed(
        RESULTS_FILE,

        y_test=y_test,

        edge_test_probabilities=(
            edge_test_probabilities
        ),

        cloud_test_probabilities=(
            cloud_test_probabilities
        ),

        edge_test_predictions=(
            edge_test_predictions
        ),

        cloud_test_predictions=(
            cloud_test_predictions
        ),

        test_confidence=test_confidence,

        test_uncertainty=test_uncertainty,

        test_entropy=test_entropy,

        raw_edge_test_utility=(
            raw_edge_test_utility
        ),

        raw_cloud_test_utility=(
            raw_cloud_test_utility
        ),

        calibrated_edge_test_utility=(
            calibrated_edge_test_utility
        ),

        calibrated_cloud_test_utility=(
            calibrated_cloud_test_utility
        ),

        estimated_sample_benefit=(
            estimated_sample_benefit
        ),

        communication_costs=(
            COMMUNICATION_COSTS
        ),

        transmission_masks=np.asarray(
            transmission_masks,
            dtype=bool,
        ),

        estimated_voi=np.asarray(
            estimated_voi_all,
            dtype=np.float32,
        ),

        final_predictions=np.asarray(
            final_predictions_all,
            dtype=np.int64,
        ),
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "formulation": {
            "edge_expected_utility": (
                "calibrated probability that edge decision is correct"
            ),

            "cloud_expected_utility": (
                "calibrated probability that cloud decision is correct"
            ),

            "information_benefit": (
                "EU_cloud - EU_edge"
            ),

            "voi": (
                "information_benefit - communication_cost"
            ),

            "decision_rule": (
                "Transmit when VoI > 0"
            ),

            "utility_definition": {
                "correct_prediction": 1.0,
                "incorrect_prediction": 0.0,
            },
        },

        "validation": {
            "edge_accuracy": (
                edge_val_metrics["accuracy"]
            ),

            "cloud_accuracy": (
                cloud_val_metrics["accuracy"]
            ),

            "accuracy_benefit": (
                cloud_val_metrics["accuracy"]
                - edge_val_metrics["accuracy"]
            ),

            "mean_information_benefit": (
                float(
                    np.mean(
                        validation_sample_benefit
                    )
                )
            ),

            "calibration_method": (
                "Isotonic regression"
            ),
        },

        "test": {
            "edge_accuracy": (
                edge_test_metrics["accuracy"]
            ),

            "cloud_accuracy": (
                cloud_test_metrics["accuracy"]
            ),

            "cloud_accuracy_gain": (
                cloud_test_metrics["accuracy"]
                - edge_test_metrics["accuracy"]
            ),

            "mean_edge_utility": (
                float(
                    np.mean(
                        calibrated_edge_test_utility
                    )
                )
            ),

            "mean_cloud_utility": (
                float(
                    np.mean(
                        calibrated_cloud_test_utility
                    )
                )
            ),

            "mean_information_benefit": (
                float(
                    np.mean(
                        estimated_sample_benefit
                    )
                )
            ),
        },

        "cost_results": cost_results,

        "best_accuracy": best_accuracy,

        "best_weighted_f1": best_f1,

        "best_accuracy_with_communication_saving": (
            best_accuracy_with_saving
        ),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "FORMAL VoI ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print(
        "\nGenerated:"
    )

    print(
        f"  {RESULTS_FILE}"
    )

    print(
        f"  {METRICS_FILE}"
    )

    print(
        "\nImportant:"
    )

    print(
        "The utility calibration was learned only from"
    )

    print(
        "the validation set and the final policy was"
    )

    print(
        "evaluated on the held-out test set."
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()