import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

SPLIT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "CWRU",
    "cwru_splits.npz"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cnn_baseline.keras"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "cnn_baseline_metadata.npz"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "baseline"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "Ball Fault",
    "Inner Race Fault",
    "Normal",
    "Outer Race Fault"
]

LABEL_TO_INDEX = {
    "Ball Fault": 0,
    "Inner Race Fault": 1,
    "Normal": 2,
    "Outer Race Fault": 3
}

INDEX_TO_LABEL = {
    0: "Ball Fault",
    1: "Inner Race Fault",
    2: "Normal",
    3: "Outer Race Fault"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("CWRU BASELINE MODEL EVALUATION")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

required_files = {
    "Dataset": SPLIT_PATH,
    "Model": MODEL_PATH,
    "Metadata": METADATA_PATH
}

for name, path in required_files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    print(f"{name} found.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data = np.load(
    SPLIT_PATH,
    allow_pickle=True
)

print("Available arrays:")

for key in data.files:

    arr = data[key]

    print(
        f"  {key}: "
        f"shape={arr.shape}, "
        f"dtype={arr.dtype}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = data["X_test"].astype(np.float32)
y_test_strings = data["y_test"]


print("\nTest data:")

print(
    f"  X_test shape: {X_test.shape}"
)

print(
    f"  X_test dtype: {X_test.dtype}"
)

print(
    f"  y_test shape: {y_test_strings.shape}"
)

print(
    f"  y_test dtype: {y_test_strings.dtype}"
)


# ============================================================
# VERIFY TEST LABELS
# ============================================================

print("\nTest classes:")

unique_classes, counts = np.unique(
    y_test_strings,
    return_counts=True
)

for class_name, count in zip(
    unique_classes,
    counts
):

    print(
        f"  {class_name}: {count}"
    )


# ============================================================
# CONVERT LABELS
# ============================================================

print("\nConverting labels to integer indices...")

unknown_labels = [
    str(label)
    for label in y_test_strings
    if str(label) not in LABEL_TO_INDEX
]

if unknown_labels:

    raise ValueError(
        f"Unknown labels found: "
        f"{sorted(set(unknown_labels))}"
    )


y_test = np.array(
    [
        LABEL_TO_INDEX[str(label)]
        for label in y_test_strings
    ],
    dtype=np.int64
)


print(
    f"Integer y_test shape: {y_test.shape}"
)

print(
    f"Integer y_test dtype: {y_test.dtype}"
)


# ============================================================
# LABEL MAPPING CHECK
# ============================================================

print("\nLabel mapping:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(
        y_test == i
    )

    print(
        f"  {i}: {class_name} -> {count} samples"
    )


# ============================================================
# LOAD TRAINING NORMALIZATION PARAMETERS
# ============================================================

print("\nLoading training normalization parameters...")

metadata = np.load(
    METADATA_PATH,
    allow_pickle=True
)

mean = float(metadata["mean"])
std = float(metadata["std"])

saved_classes = metadata["classes"]

print(
    f"Training mean: {mean:.8f}"
)

print(
    f"Training std:  {std:.8f}"
)

print(
    "Saved classes:"
)

for i, class_name in enumerate(saved_classes):

    print(
        f"  {i}: {class_name}"
    )


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

saved_classes = [
    str(x)
    for x in saved_classes
]

if saved_classes != CLASS_NAMES:

    raise ValueError(
        "\nClass ordering mismatch!\n"
        f"Model metadata: {saved_classes}\n"
        f"Expected:       {CLASS_NAMES}"
    )

print(
    "\nClass ordering verified."
)


# ============================================================
# NORMALIZE TEST DATA
# ============================================================

print("\nNormalizing test signals...")

print(
    "Using training-set mean/std "
    "(same normalization as training)."
)

X_test = (
    X_test - mean
) / (
    std + 1e-8
)


# ============================================================
# ADD CNN CHANNEL DIMENSION
# ============================================================

X_test = X_test[
    ...,
    np.newaxis
]

print(
    f"CNN test input shape: {X_test.shape}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

print(
    f"Model: {MODEL_PATH}"
)

model = load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# MODEL CHECK
# ============================================================

print("\nModel output information:")

num_outputs = model.output_shape[-1]

print(
    f"Number of output classes: {num_outputs}"
)

if num_outputs != len(CLASS_NAMES):

    raise ValueError(
        "Model output classes do not match "
        "the expected 4 classes."
    )


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test loss:     {test_loss:.4f}"
)

print(
    f"Test accuracy: {test_accuracy:.4f}"
)

print(
    f"Test accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
).astype(np.int64)

print(
    f"Prediction shape: {y_pred.shape}"
)


# ============================================================
# PREDICTED CLASS DISTRIBUTION
# ============================================================

print("\nPredicted classes:")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(
        y_pred == i
    )

    print(
        f"  {i}: {class_name} -> {count} samples"
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    average="macro",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report_text = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report_text)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "Creating confusion matrix..."
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES))
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(9, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    "CWRU Baseline CNN - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved to:"
)

print(
    confusion_matrix_path
)


# ============================================================
# DETAILED REPORT
# ============================================================

report_dict = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model":
        "CWRU Baseline 1D CNN",

    "dataset":
        "CWRU",

    "evaluation_protocol":
        "file_level_split",

    "test_samples":
        int(len(y_test)),

    "test_loss":
        float(test_loss),

    "accuracy":
        float(accuracy),

    "precision_weighted":
        float(precision),

    "recall_weighted":
        float(recall),

    "f1_weighted":
        float(f1),

    "f1_macro":
        float(macro_f1),

    "normalization":
        "training_global_mean_std",

    "training_mean":
        float(mean),

    "training_std":
        float(std),

    "classes":
        CLASS_NAMES,

    "label_mapping":
        LABEL_TO_INDEX,

    "confusion_matrix":
        cm.tolist(),

    "classification_report":
        report_dict
}


metrics_path = os.path.join(
    RESULTS_DIR,
    "baseline_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print(
    "\nMetrics saved to:"
)

print(
    metrics_path
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print(
    "\nSaving predictions..."
)

predictions_path = os.path.join(
    RESULTS_DIR,
    "baseline_predictions.npz"
)

np.savez_compressed(
    predictions_path,

    y_true=y_test,

    y_pred=y_pred,

    y_true_labels=y_test_strings,

    y_pred_labels=np.array(
        [
            INDEX_TO_LABEL[int(i)]
            for i in y_pred
        ]
    ),

    probabilities=probabilities
)

print(
    "Predictions saved to:"
)

print(
    predictions_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BASELINE EVALUATION COMPLETE")
print("=" * 60)

print(
    f"\nAccuracy:       {accuracy:.4f}"
)

print(
    f"Precision:      {precision:.4f}"
)

print(
    f"Recall:         {recall:.4f}"
)

print(
    f"Weighted F1:    {f1:.4f}"
)

print(
    f"Macro F1:       {macro_f1:.4f}"
)

print(
    "\nResults directory:"
)

print(
    RESULTS_DIR
)

print("\nGenerated files:")

print(
    "  baseline_metrics.json"
)

print(
    "  baseline_predictions.npz"
)

print(
    "  confusion_matrix.png"
)

print("\n" + "=" * 60)