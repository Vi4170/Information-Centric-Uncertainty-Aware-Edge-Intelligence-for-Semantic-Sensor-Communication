import os
import re
import json
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

DATA_PATH = "data/processed/CWRU/cwru_splits.npz"
OUTPUT_DIR = "results/anomaly"
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(DATA_PATH, allow_pickle=True)

X = np.concatenate([
    data["X_train"],
    data["X_val"],
    data["X_test"]
]).astype(np.float32)

y = np.concatenate([
    data["y_train"],
    data["y_val"],
    data["y_test"]
])

files = np.concatenate([
    data["files_train"],
    data["files_val"],
    data["files_test"]
])


# ------------------------------------------------------------
# LOAD CONDITION
# ------------------------------------------------------------

def get_load(filename):
    match = re.search(r"_(\d+HP)(?:_|\.mat)", str(filename))
    return match.group(1) if match else None


loads = np.array([get_load(f) for f in files])

LOADS = ["0HP", "1HP", "2HP", "3HP"]

print("=" * 70)
print("LEAVE-ONE-LOAD-OUT NOVELTY ROBUSTNESS")
print("=" * 70)

print("\nDataset:", X.shape)

# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_features(X):

    features = []

    for signal in X:

        mean = np.mean(signal)
        std = np.std(signal)
        variance = np.var(signal)
        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))
        ptp = np.ptp(signal)
        abs_mean = np.mean(np.abs(signal))

        centered = signal - mean

        skewness = (
            np.mean(centered ** 3) /
            (std ** 3 + 1e-12)
        )

        kurtosis = (
            np.mean(centered ** 4) /
            (std ** 4 + 1e-12)
        )

        crest_factor = peak / (rms + 1e-12)
        impulse_factor = peak / (abs_mean + 1e-12)
        shape_factor = rms / (abs_mean + 1e-12)

        spectrum = np.abs(np.fft.rfft(signal))
        power = spectrum ** 2

        freqs = np.fft.rfftfreq(
            len(signal),
            d=1.0 / 12000
        )

        dominant_frequency = freqs[
            np.argmax(power[1:]) + 1
        ]

        total_power = np.sum(power) + 1e-12

        spectral_centroid = (
            np.sum(freqs * power) /
            total_power
        )

        spectral_bandwidth = np.sqrt(
            np.sum(
                ((freqs - spectral_centroid) ** 2)
                * power
            ) / total_power
        )

        probability = power / total_power

        spectral_entropy = -np.sum(
            probability *
            np.log(probability + 1e-12)
        )

        # 7 frequency bands
        band_edges = np.linspace(
            0,
            len(power),
            8,
            dtype=int
        )

        bands = []

        for i in range(7):
            band_power = np.sum(
                power[
                    band_edges[i]:
                    band_edges[i + 1]
                ]
            )

            bands.append(
                band_power / total_power
            )

        features.append([
            mean,
            std,
            variance,
            rms,
            peak,
            ptp,
            abs_mean,
            skewness,
            kurtosis,
            crest_factor,
            impulse_factor,
            shape_factor,
            dominant_frequency,
            spectral_centroid,
            spectral_bandwidth,
            spectral_entropy,
            *bands
        ])

    return np.asarray(features, dtype=np.float32)


print("\nExtracting features...")
F = extract_features(X)

print("Feature matrix:", F.shape)


# ------------------------------------------------------------
# FREQUENCY FEATURES ONLY
# ------------------------------------------------------------

# indices:
# 12 = dominant frequency
# 13 = centroid
# 14 = bandwidth
# 15 = entropy
# 16-22 = frequency bands

F_FREQ = F[:, 12:23]


# ------------------------------------------------------------
# NORMALIZE USING TRAINING LOADS ONLY
# ------------------------------------------------------------

def standardize(train, test):

    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0) + 1e-8

    return (
        (train - mean) / std,
        (test - mean) / std
    )


# ------------------------------------------------------------
# THRESHOLD
# ------------------------------------------------------------

def find_threshold(y_true, scores):

    thresholds = np.percentile(
        scores,
        np.linspace(1, 99, 199)
    )

    best_threshold = thresholds[0]
    best_f1 = -1

    for threshold in thresholds:

        pred = (
            scores >= threshold
        ).astype(int)

        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            pred,
            average="binary",
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


# ------------------------------------------------------------
# EXPERIMENT
# ------------------------------------------------------------

results = []

for held_out in LOADS:

    print("\n" + "=" * 70)
    print("HELD-OUT LOAD:", held_out)
    print("=" * 70)

    train_mask = loads != held_out
    test_mask = loads == held_out

    X_train_f = F_FREQ[train_mask]
    X_test_f = F_FREQ[test_mask]

    y_train_labels = y[train_mask]
    y_test_labels = y[test_mask]

    # Normal-only training
    normal_mask = y_train_labels == "Normal"

    X_normal = X_train_f[normal_mask]

    y_test_anomaly = (
        y_test_labels != "Normal"
    ).astype(int)

    # --------------------------------------------------------
    # SCALE WITHOUT LEAKAGE
    # --------------------------------------------------------

    X_normal_scaled, X_test_scaled = standardize(
        X_normal,
        X_test_f
    )

    # --------------------------------------------------------
    # ONE-CLASS SVM
    # --------------------------------------------------------

    svm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )

    svm.fit(X_normal_scaled)

    train_scores = -svm.decision_function(
        X_normal_scaled
    )

    test_scores = -svm.decision_function(
        X_test_scaled
    )

    threshold, validation_f1 = find_threshold(
        np.zeros(len(train_scores)),
        train_scores
    )

    # Use normal training distribution to define
    # a conservative threshold.
    threshold = np.percentile(
        train_scores,
        99
    )

    predictions = (
        test_scores >= threshold
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test_anomaly,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    roc = roc_auc_score(
        y_test_anomaly,
        test_scores
    )

    pr = average_precision_score(
        y_test_anomaly,
        test_scores
    )

    cm = confusion_matrix(
        y_test_anomaly,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / max(tn + fp, 1)

    result = {
        "held_out_load": held_out,
        "model": "Frequency One-Class SVM",
        "train_normal_samples": int(len(X_normal)),
        "test_samples": int(len(X_test_f)),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "threshold": float(threshold),
        "normal_score_mean": float(
            np.mean(
                test_scores[
                    y_test_anomaly == 0
                ]
            )
        ),
        "anomaly_score_mean": float(
            np.mean(
                test_scores[
                    y_test_anomaly == 1
                ]
            )
        ),
        "confusion_matrix": cm.tolist()
    }

    results.append(result)

    print("ROC-AUC:", round(roc, 4))
    print("PR-AUC :", round(pr, 4))
    print("F1     :", round(f1, 4))
    print("FPR    :", round(fpr, 4))


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)

print(
    f"{'LOAD':10s}"
    f"{'ROC-AUC':>10s}"
    f"{'PR-AUC':>10s}"
    f"{'F1':>10s}"
    f"{'FPR':>10s}"
)

print("-" * 50)

for r in results:

    print(
        f"{r['held_out_load']:10s}"
        f"{r['roc_auc']:10.4f}"
        f"{r['pr_auc']:10.4f}"
        f"{r['f1']:10.4f}"
        f"{r['false_positive_rate']:10.4f}"
    )


# ------------------------------------------------------------
# OVERALL
# ------------------------------------------------------------

summary = {
    "experiment": "Leave-one-load-out robustness",
    "loads": LOADS,
    "results": results,
    "mean_roc_auc": float(
        np.mean([r["roc_auc"] for r in results])
    ),
    "mean_pr_auc": float(
        np.mean([r["pr_auc"] for r in results])
    ),
    "mean_f1": float(
        np.mean([r["f1"] for r in results])
    ),
    "mean_fpr": float(
        np.mean(
            [r["false_positive_rate"] for r in results]
        )
    )
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "load_robustness_results.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print("\nSaved:")
print(
    "results/anomaly/load_robustness_results.json"
)

print("\n" + "=" * 70)
print("LOAD ROBUSTNESS EXPERIMENT COMPLETE")
print("=" * 70)