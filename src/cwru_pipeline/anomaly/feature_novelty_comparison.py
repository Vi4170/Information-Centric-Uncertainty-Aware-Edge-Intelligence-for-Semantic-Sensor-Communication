import os
import json
import time
import numpy as np

from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

SEED = 42
np.random.seed(SEED)

DATA_PATH = "data/processed/CWRU/cwru_splits.npz"
OUTPUT_DIR = "results/anomaly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NORMAL_LABEL = "Normal"

# Sampling frequency used by the CWRU processed data.
# Features based on relative spectrum remain useful even if
# the exact acquisition rate differs.
FS = 12000.0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE-BASED NOVELTY / ANOMALY ANALYSIS")
print("=" * 70)

data = np.load(DATA_PATH, allow_pickle=True)

X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"]

X_val = data["X_val"].astype(np.float32)
y_val = data["y_val"]

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"]

print("\nDataset:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# LABEL CONVERSION
# ============================================================

def make_anomaly_labels(y):
    return (y != NORMAL_LABEL).astype(np.int32)


y_val_anomaly = make_anomaly_labels(y_val)
y_test_anomaly = make_anomaly_labels(y_test)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def spectral_entropy(power):
    power = np.asarray(power, dtype=np.float64)
    power = power + 1e-12

    probability = power / np.sum(power)

    return -np.sum(
        probability * np.log2(probability + 1e-12)
    )


def extract_features(X, fs=FS):

    features = []

    for signal in X:

        signal = signal.astype(np.float64)

        # ----------------------------------------------------
        # TIME DOMAIN
        # ----------------------------------------------------

        mean_value = np.mean(signal)
        std_value = np.std(signal)
        variance = np.var(signal)

        rms = np.sqrt(
            np.mean(signal ** 2)
        )

        peak = np.max(
            np.abs(signal)
        )

        peak_to_peak = (
            np.max(signal) -
            np.min(signal)
        )

        abs_mean = np.mean(
            np.abs(signal)
        )

        skewness = skew(
            signal,
            bias=False
        )

        kurt = kurtosis(
            signal,
            fisher=False,
            bias=False
        )

        crest_factor = (
            peak / (rms + 1e-12)
        )

        impulse_factor = (
            peak / (abs_mean + 1e-12)
        )

        shape_factor = (
            rms / (abs_mean + 1e-12)
        )

        # ----------------------------------------------------
        # FREQUENCY DOMAIN
        # ----------------------------------------------------

        spectrum = np.fft.rfft(signal)

        magnitude = np.abs(spectrum)

        power = magnitude ** 2

        frequencies = np.fft.rfftfreq(
            len(signal),
            d=1.0 / fs
        )

        total_power = (
            np.sum(power) + 1e-12
        )

        dominant_frequency = frequencies[
            np.argmax(power)
        ]

        spectral_centroid = np.sum(
            frequencies * power
        ) / total_power

        spectral_bandwidth = np.sqrt(
            np.sum(
                ((frequencies - spectral_centroid) ** 2)
                * power
            ) / total_power
        )

        spec_entropy = spectral_entropy(
            power
        )

        # ----------------------------------------------------
        # BAND ENERGY
        # ----------------------------------------------------

        nyquist = fs / 2.0

        band_edges = [
            0,
            0.05 * nyquist,
            0.10 * nyquist,
            0.20 * nyquist,
            0.40 * nyquist,
            0.60 * nyquist,
            0.80 * nyquist,
            nyquist,
        ]

        band_features = []

        for low, high in zip(
            band_edges[:-1],
            band_edges[1:]
        ):

            mask = (
                (frequencies >= low) &
                (frequencies < high)
            )

            band_power = np.sum(
                power[mask]
            )

            band_features.append(
                band_power / total_power
            )

        sample_features = [
            mean_value,
            std_value,
            variance,
            rms,
            peak,
            peak_to_peak,
            abs_mean,
            skewness,
            kurt,
            crest_factor,
            impulse_factor,
            shape_factor,
            dominant_frequency,
            spectral_centroid,
            spectral_bandwidth,
            spec_entropy,
        ]

        sample_features.extend(
            band_features
        )

        features.append(
            sample_features
        )

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# EXTRACT
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING FEATURES")
print("=" * 70)

start = time.perf_counter()

F_train = extract_features(X_train)
F_val = extract_features(X_val)
F_test = extract_features(X_test)

feature_time = time.perf_counter() - start

print("\nFeature extraction complete.")
print("Train features:", F_train.shape)
print("Validation features:", F_val.shape)
print("Test features:", F_test.shape)
print("Feature extraction time:", round(feature_time, 3), "seconds")


# ============================================================
# DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DATA QUALITY")
print("=" * 70)

print(
    "Train NaN:",
    np.isnan(F_train).sum(),
    "Inf:",
    np.isinf(F_train).sum()
)

print(
    "Validation NaN:",
    np.isnan(F_val).sum(),
    "Inf:",
    np.isinf(F_val).sum()
)

print(
    "Test NaN:",
    np.isnan(F_test).sum(),
    "Inf:",
    np.isinf(F_test).sum()
)


# ============================================================
# NORMAL-ONLY TRAINING
# ============================================================

normal_mask = y_train == NORMAL_LABEL

F_normal_train = F_train[
    normal_mask
]

print("\nNormal training samples:")
print(len(F_normal_train))


# ============================================================
# NORMALIZATION
# FIT ONLY ON NORMAL TRAINING DATA
# ============================================================

scaler = StandardScaler()

F_normal_train_scaled = scaler.fit_transform(
    F_normal_train
)

F_val_scaled = scaler.transform(
    F_val
)

F_test_scaled = scaler.transform(
    F_test
)


# ============================================================
# THRESHOLD
# ============================================================

def find_best_threshold(
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

        prediction = (
            scores >= threshold
        ).astype(np.int32)

        _, _, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                prediction,
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
# EVALUATION
# ============================================================

def evaluate(
    name,
    val_scores,
    test_scores
):

    threshold, validation_f1 = (
        find_best_threshold(
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

    normal_scores = test_scores[
        y_test_anomaly == 0
    ]

    anomaly_scores = test_scores[
        y_test_anomaly == 1
    ]

    result = {
        "model": name,
        "threshold": threshold,
        "validation_f1": validation_f1,
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "false_positive_rate": float(fpr),
        "normal_score_mean": float(
            np.mean(normal_scores)
        ),
        "normal_score_std": float(
            np.std(normal_scores)
        ),
        "anomaly_score_mean": float(
            np.mean(anomaly_scores)
        ),
        "anomaly_score_std": float(
            np.std(anomaly_scores)
        ),
        "confusion_matrix": cm.tolist(),
    }

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        "Threshold:",
        round(threshold, 6)
    )

    print(
        "ROC-AUC:",
        round(roc_auc, 4)
    )

    print(
        "PR-AUC:",
        round(pr_auc, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1:",
        round(f1, 4)
    )

    print(
        "False Positive Rate:",
        round(fpr, 4)
    )

    print(
        "Normal score mean:",
        round(np.mean(normal_scores), 6)
    )

    print(
        "Anomaly score mean:",
        round(np.mean(anomaly_scores), 6)
    )

    return result


# ============================================================
# MODEL 1: ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MODEL 1: ISOLATION FOREST")
print("=" * 70)

start = time.perf_counter()

iso = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=SEED,
    n_jobs=-1
)

iso.fit(
    F_normal_train_scaled
)

iso_train_time = (
    time.perf_counter() - start
)

start = time.perf_counter()

val_iso_scores = (
    -iso.decision_function(
        F_val_scaled
    )
)

test_iso_scores = (
    -iso.decision_function(
        F_test_scaled
    )
)

iso_inference_time = (
    time.perf_counter() - start
)

iso_result = evaluate(
    "Feature Isolation Forest",
    val_iso_scores,
    test_iso_scores
)

iso_result[
    "train_time_seconds"
] = iso_train_time

iso_result[
    "inference_time_seconds"
] = iso_inference_time


# ============================================================
# MODEL 2: ONE-CLASS SVM
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MODEL 2: ONE-CLASS SVM")
print("=" * 70)

start = time.perf_counter()

svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

svm.fit(
    F_normal_train_scaled
)

svm_train_time = (
    time.perf_counter() - start
)

start = time.perf_counter()

val_svm_scores = (
    -svm.decision_function(
        F_val_scaled
    )
)

test_svm_scores = (
    -svm.decision_function(
        F_test_scaled
    )
)

svm_inference_time = (
    time.perf_counter() - start
)

svm_result = evaluate(
    "Feature One-Class SVM",
    val_svm_scores,
    test_svm_scores
)

svm_result[
    "train_time_seconds"
] = svm_train_time

svm_result[
    "inference_time_seconds"
] = svm_inference_time


# ============================================================
# SAVE FEATURES
# ============================================================

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "extracted_features.npz"
    ),
    F_train=F_train,
    F_val=F_val,
    F_test=F_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
)


# ============================================================
# SAVE SCALER
# ============================================================

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "feature_scaler.npz"
    ),
    mean=scaler.mean_,
    scale=scaler.scale_,
)


# ============================================================
# RESULTS
# ============================================================

results = [
    iso_result,
    svm_result,
]

with open(
    os.path.join(
        OUTPUT_DIR,
        "feature_novelty_comparison.json"
    ),
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE-BASED NOVELTY COMPARISON")
print("=" * 70)

print(
    f"{'MODEL':28s}"
    f"{'ROC-AUC':>10s}"
    f"{'PR-AUC':>10s}"
    f"{'F1':>10s}"
    f"{'FPR':>10s}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['model']:28s}"
        f"{result['test_roc_auc']:10.4f}"
        f"{result['test_pr_auc']:10.4f}"
        f"{result['test_f1']:10.4f}"
        f"{result['false_positive_rate']:10.4f}"
    )


print("\nResults saved to:")

print(
    os.path.join(
        OUTPUT_DIR,
        "extracted_features.npz"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "feature_novelty_comparison.json"
    )
)

print("\n" + "=" * 70)
print("FEATURE NOVELTY EXPERIMENT COMPLETE")
print("=" * 70)