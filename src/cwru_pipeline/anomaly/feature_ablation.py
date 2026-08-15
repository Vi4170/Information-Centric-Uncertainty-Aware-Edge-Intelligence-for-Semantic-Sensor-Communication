import os
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

SEED = 42
np.random.seed(SEED)

DATA_DIR = "results/anomaly"
OUTPUT_DIR = "results/anomaly"

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "extracted_features.npz"
)

NORMAL_LABEL = "Normal"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FEATURE ABLATION FOR NOVELTY ESTIMATION")
print("=" * 70)

data = np.load(
    FEATURE_FILE,
    allow_pickle=True
)

F_train = data["F_train"]
F_val = data["F_val"]
F_test = data["F_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]


# ============================================================
# FEATURE GROUPS
# ============================================================

feature_names = [
    "mean",
    "std",
    "variance",
    "rms",
    "peak",
    "peak_to_peak",
    "abs_mean",
    "skewness",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "band_1",
    "band_2",
    "band_3",
    "band_4",
    "band_5",
    "band_6",
    "band_7",
]


time_domain = list(range(0, 12))

frequency_domain = list(range(12, 23))

combined = list(range(0, 23))


groups = {
    "Time-domain only": time_domain,
    "Frequency-domain only": frequency_domain,
    "Time + Frequency": combined,
}


# ============================================================
# LABELS
# ============================================================

def anomaly_labels(y):

    return (
        y != NORMAL_LABEL
    ).astype(np.int32)


y_val_anomaly = anomaly_labels(y_val)
y_test_anomaly = anomaly_labels(y_test)


# ============================================================
# THRESHOLD
# ============================================================

def find_threshold(
    y_true,
    scores
):

    thresholds = np.percentile(
        scores,
        np.linspace(1, 99, 199)
    )

    best_threshold = thresholds[0]
    best_f1 = -1

    for threshold in thresholds:

        pred = (
            scores >= threshold
        ).astype(np.int32)

        _, _, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                pred,
                average="binary",
                zero_division=0
            )
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1)
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    name,
    val_scores,
    test_scores
):

    threshold, validation_f1 = (
        find_threshold(
            y_val_anomaly,
            val_scores
        )
    )

    prediction = (
        test_scores >= threshold
    ).astype(np.int32)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test_anomaly,
            prediction,
            average="binary",
            zero_division=0
        )
    )

    roc_auc = roc_auc_score(
        y_test_anomaly,
        test_scores
    )

    pr_auc = average_precision_score(
        y_test_anomaly,
        test_scores
    )

    cm = confusion_matrix(
        y_test_anomaly,
        prediction
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / max(
        tn + fp,
        1
    )

    return {
        "feature_group": name,
        "features": [
            feature_names[i]
            for i in groups[name]
        ],
        "num_features": len(
            groups[name]
        ),
        "threshold": threshold,
        "validation_f1": validation_f1,
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# RUN ABLATION
# ============================================================

results = []

for group_name, indices in groups.items():

    print("\n" + "=" * 70)
    print(group_name)
    print("=" * 70)

    print(
        "Number of features:",
        len(indices)
    )

    print(
        "Features:",
        ", ".join(
            feature_names[i]
            for i in indices
        )
    )

    train_group = F_train[
        :,
        indices
    ]

    val_group = F_val[
        :,
        indices
    ]

    test_group = F_test[
        :,
        indices
    ]

    normal_mask = (
        y_train == NORMAL_LABEL
    )

    normal_train = train_group[
        normal_mask
    ]

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    normal_train_scaled = (
        scaler.fit_transform(
            normal_train
        )
    )

    val_scaled = (
        scaler.transform(
            val_group
        )
    )

    test_scaled = (
        scaler.transform(
            test_group
        )
    )

    # --------------------------------------------------------
    # ONE-CLASS SVM
    # --------------------------------------------------------

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )

    model.fit(
        normal_train_scaled
    )

    val_scores = (
        -model.decision_function(
            val_scaled
        )
    )

    test_scores = (
        -model.decision_function(
            test_scaled
        )
    )

    result = evaluate(
        group_name,
        val_scores,
        test_scores
    )

    results.append(result)

    print(
        "\nROC-AUC:",
        round(
            result["test_roc_auc"],
            4
        )
    )

    print(
        "PR-AUC:",
        round(
            result["test_pr_auc"],
            4
        )
    )

    print(
        "F1:",
        round(
            result["test_f1"],
            4
        )
    )

    print(
        "FPR:",
        round(
            result["false_positive_rate"],
            4
        )
    )


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "feature_ablation_results.json"
)

with open(
    output_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE ABLATION SUMMARY")
print("=" * 70)

print(
    f"{'FEATURE GROUP':25s}"
    f"{'N':>5s}"
    f"{'ROC-AUC':>10s}"
    f"{'PR-AUC':>10s}"
    f"{'F1':>10s}"
    f"{'FPR':>10s}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['feature_group']:25s}"
        f"{result['num_features']:5d}"
        f"{result['test_roc_auc']:10.4f}"
        f"{result['test_pr_auc']:10.4f}"
        f"{result['test_f1']:10.4f}"
        f"{result['false_positive_rate']:10.4f}"
    )


best = max(
    results,
    key=lambda x: (
        x["test_pr_auc"],
        x["test_f1"],
        x["test_roc_auc"],
        -x["false_positive_rate"]
    )
)

print("\nBEST FEATURE GROUP:")
print(
    best["feature_group"]
)

print(
    "ROC-AUC:",
    round(
        best["test_roc_auc"],
        4
    )
)

print(
    "PR-AUC:",
    round(
        best["test_pr_auc"],
        4
    )
)

print(
    "F1:",
    round(
        best["test_f1"],
        4
    )
)

print(
    "FPR:",
    round(
        best["false_positive_rate"],
        4
    )
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 70)
print("FEATURE ABLATION COMPLETE")
print("=" * 70)