import os
import json
import time
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/CWRU/cwru_splits.npz"
OUTPUT_DIR = "results/anomaly"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

NORMAL_LABEL = "Normal"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NOVELTY / ANOMALY DETECTION MODEL COMPARISON")
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
# NORMAL-ONLY TRAINING DATA
# ============================================================

normal_train_mask = y_train == NORMAL_LABEL

X_normal_train = X_train[normal_train_mask]

print("\nNormal-only training:")
print("Normal samples:", len(X_normal_train))

if len(X_normal_train) == 0:
    raise RuntimeError("No Normal samples found in training set.")


# ============================================================
# CONVERT LABELS
# 0 = NORMAL
# 1 = ANOMALY / FAULT
# ============================================================

def make_anomaly_labels(y):
    return (y != NORMAL_LABEL).astype(np.int32)


y_val_anomaly = make_anomaly_labels(y_val)
y_test_anomaly = make_anomaly_labels(y_test)


# ============================================================
# HELPER: THRESHOLD SELECTION
# ============================================================

def find_best_threshold(y_true, scores):
    """
    Select threshold using validation F1.

    Higher score = more anomalous.
    """

    thresholds = np.percentile(
        scores,
        np.linspace(1, 99, 199)
    )

    best_threshold = thresholds[0]
    best_f1 = -1

    for threshold in thresholds:

        pred = (scores >= threshold).astype(np.int32)

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


# ============================================================
# HELPER: EVALUATION
# ============================================================

def evaluate_scores(name, val_scores, test_scores):

    val_threshold, val_f1 = find_best_threshold(
        y_val_anomaly,
        val_scores
    )

    test_pred = (
        test_scores >= val_threshold
    ).astype(np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_anomaly,
        test_pred,
        average="binary",
        zero_division=0
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
        test_pred
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / max(fp + tn, 1)

    normal_scores = test_scores[y_test_anomaly == 0]
    anomaly_scores = test_scores[y_test_anomaly == 1]

    result = {
        "model": name,
        "threshold": val_threshold,
        "validation_f1": val_f1,
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "false_positive_rate": float(fpr),
        "normal_score_mean": float(np.mean(normal_scores)),
        "normal_score_std": float(np.std(normal_scores)),
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "anomaly_score_std": float(np.std(anomaly_scores)),
        "confusion_matrix": cm.tolist(),
    }

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print("Threshold:", round(val_threshold, 6))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("False Positive Rate:", round(fpr, 4))

    print(
        "Normal novelty score:",
        round(np.mean(normal_scores), 4)
    )

    print(
        "Anomaly novelty score:",
        round(np.mean(anomaly_scores), 4)
    )

    return result


# ============================================================
# MODEL 1: ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1: ISOLATION FOREST")
print("=" * 70)

start = time.perf_counter()

iso = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=SEED,
    n_jobs=-1
)

iso.fit(X_normal_train)

train_time = time.perf_counter() - start

start = time.perf_counter()

# sklearn gives lower values for anomalies.
# Negate so higher = more anomalous.
val_iso_scores = -iso.decision_function(X_val)
test_iso_scores = -iso.decision_function(X_test)

inference_time = time.perf_counter() - start

iso_result = evaluate_scores(
    "Isolation Forest",
    val_iso_scores,
    test_iso_scores
)

iso_result["train_time_seconds"] = train_time
iso_result["inference_time_seconds"] = inference_time


# ============================================================
# MODEL 2: ONE-CLASS SVM
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2: ONE-CLASS SVM")
print("=" * 70)

start = time.perf_counter()

ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

ocsvm.fit(X_normal_train)

train_time = time.perf_counter() - start

start = time.perf_counter()

# Higher = more anomalous
val_svm_scores = -ocsvm.decision_function(X_val)
test_svm_scores = -ocsvm.decision_function(X_test)

inference_time = time.perf_counter() - start

svm_result = evaluate_scores(
    "One-Class SVM",
    val_svm_scores,
    test_svm_scores
)

svm_result["train_time_seconds"] = train_time
svm_result["inference_time_seconds"] = inference_time


# ============================================================
# MODEL 3: CNN AUTOENCODER
# ============================================================

print("\n" + "=" * 70)
print("MODEL 3: CNN AUTOENCODER")
print("=" * 70)

X_normal_train_cnn = X_normal_train[..., np.newaxis]

X_val_cnn = X_val[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]


def build_autoencoder(input_length):

    inputs = keras.Input(
        shape=(input_length, 1)
    )

    x = layers.Conv1D(
        16,
        7,
        activation="relu",
        padding="same"
    )(inputs)

    x = layers.MaxPooling1D(
        2,
        padding="same"
    )(x)

    x = layers.Conv1D(
        8,
        7,
        activation="relu",
        padding="same"
    )(x)

    encoded = layers.MaxPooling1D(
        2,
        padding="same",
        name="latent"
    )(x)

    x = layers.Conv1D(
        8,
        7,
        activation="relu",
        padding="same"
    )(encoded)

    x = layers.UpSampling1D(2)(x)

    x = layers.Conv1D(
        16,
        7,
        activation="relu",
        padding="same"
    )(x)

    x = layers.UpSampling1D(2)(x)

    outputs = layers.Conv1D(
        1,
        7,
        activation="linear",
        padding="same"
    )(x)

    model = keras.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mse"
    )

    return model


start = time.perf_counter()

autoencoder = build_autoencoder(
    X_normal_train.shape[1]
)

autoencoder.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

autoencoder.fit(
    X_normal_train_cnn,
    X_normal_train_cnn,
    validation_split=0.2,
    epochs=60,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

train_time = time.perf_counter() - start


# ============================================================
# RECONSTRUCTION ERROR = NOVELTY SCORE
# ============================================================

start = time.perf_counter()

val_reconstruction = autoencoder.predict(
    X_val_cnn,
    verbose=0
)

test_reconstruction = autoencoder.predict(
    X_test_cnn,
    verbose=0
)

val_ae_scores = np.mean(
    np.square(
        X_val_cnn - val_reconstruction
    ),
    axis=(1, 2)
)

test_ae_scores = np.mean(
    np.square(
        X_test_cnn - test_reconstruction
    ),
    axis=(1, 2)
)

inference_time = time.perf_counter() - start

ae_result = evaluate_scores(
    "CNN Autoencoder",
    val_ae_scores,
    test_ae_scores
)

ae_result["train_time_seconds"] = train_time
ae_result["inference_time_seconds"] = inference_time


# ============================================================
# SAVE NOVELTY SCORES
# ============================================================

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "novelty_scores.npz"
    ),
    y_val=y_val,
    y_test=y_test,
    y_val_anomaly=y_val_anomaly,
    y_test_anomaly=y_test_anomaly,

    val_isolation_forest=val_iso_scores,
    test_isolation_forest=test_iso_scores,

    val_one_class_svm=val_svm_scores,
    test_one_class_svm=test_svm_scores,

    val_cnn_autoencoder=val_ae_scores,
    test_cnn_autoencoder=test_ae_scores,
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = [
    iso_result,
    svm_result,
    ae_result
]

with open(
    os.path.join(
        OUTPUT_DIR,
        "novelty_model_comparison.json"
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
print("NOVELTY MODEL COMPARISON")
print("=" * 70)

print(
    f"{'MODEL':25s}"
    f"{'ROC-AUC':>10s}"
    f"{'PR-AUC':>10s}"
    f"{'F1':>10s}"
    f"{'FPR':>10s}"
)

print("-" * 70)

for r in results:

    print(
        f"{r['model']:25s}"
        f"{r['test_roc_auc']:10.4f}"
        f"{r['test_pr_auc']:10.4f}"
        f"{r['test_f1']:10.4f}"
        f"{r['false_positive_rate']:10.4f}"
    )


# ============================================================
# SELECT BEST MODEL
# ============================================================

best = max(
    results,
    key=lambda r: (
        r["test_pr_auc"],
        r["test_f1"],
        r["test_roc_auc"]
    )
)

print("\n" + "=" * 70)
print("BEST NOVELTY ESTIMATOR")
print("=" * 70)

print("Model:", best["model"])
print("ROC-AUC:", round(best["test_roc_auc"], 4))
print("PR-AUC:", round(best["test_pr_auc"], 4))
print("F1:", round(best["test_f1"], 4))
print("Threshold:", round(best["threshold"], 6))

print("\nResults saved to:")
print(
    os.path.join(
        OUTPUT_DIR,
        "novelty_model_comparison.json"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "novelty_scores.npz"
    )
)

print("\n" + "=" * 70)
print("NOVELTY DETECTION EXPERIMENT COMPLETE")
print("=" * 70)