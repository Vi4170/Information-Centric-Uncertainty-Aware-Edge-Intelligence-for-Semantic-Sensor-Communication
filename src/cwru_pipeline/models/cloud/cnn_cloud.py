from pathlib import Path

import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, callbacks


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "CWRU"
    / "cwru_splits.npz"
)

MODEL_DIR = PROJECT_ROOT / "saved_models" / "cloud"
RESULTS_DIR = PROJECT_ROOT / "results" / "cloud"

MODEL_FILE = MODEL_DIR / "cnn_cloud.keras"
METADATA_FILE = MODEL_DIR / "cnn_cloud_metadata.npz"

METRICS_FILE = RESULTS_DIR / "cloud_metrics.json"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

EPOCHS = 40
BATCH_SIZE = 64

NUM_CLASSES = 4

CLASS_NAMES = np.array([
    "Ball Fault",
    "Inner Race Fault",
    "Normal",
    "Outer Race Fault"
])


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# NORMALIZATION
# ============================================================

def calculate_training_statistics(X_train):
    """
    Calculate global training-set mean and standard deviation.
    """

    mean = np.mean(X_train, dtype=np.float64)
    std = np.std(X_train, dtype=np.float64)

    if std < 1e-8:
        raise ValueError("Training standard deviation is too small.")

    return float(mean), float(std)


def normalize(X, mean, std):
    """
    Normalize using training-set statistics only.
    """

    X = (X - mean) / std

    return X.astype(np.float32)


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_labels(labels):
    """
    Convert string labels to integer class indices.
    """

    mapping = {
        "Ball Fault": 0,
        "Inner Race Fault": 1,
        "Normal": 2,
        "Outer Race Fault": 3,
    }

    return np.asarray(
        [mapping[str(label)] for label in labels],
        dtype=np.int64
    )


# ============================================================
# CLOUD CNN
# ============================================================

def build_cloud_model():

    model = models.Sequential(
        [
            layers.Input(shape=(2048, 1)),

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            layers.Conv1D(
                64,
                kernel_size=7,
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.MaxPooling1D(
                pool_size=2
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            layers.Conv1D(
                128,
                kernel_size=5,
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.MaxPooling1D(
                pool_size=2
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            layers.Conv1D(
                256,
                kernel_size=5,
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.MaxPooling1D(
                pool_size=2
            ),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            layers.Conv1D(
                256,
                kernel_size=3,
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.MaxPooling1D(
                pool_size=2
            ),

            # ------------------------------------------------
            # Global representation
            # ------------------------------------------------

            layers.GlobalAveragePooling1D(),

            layers.Dense(
                256,
                activation="relu"
            ),

            layers.Dropout(0.30),

            layers.Dense(
                128,
                activation="relu"
            ),

            layers.Dropout(0.20),

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            ),
        ],
        name="cloud_cnn"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CWRU CLOUD / REMOTE CNN")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print("\nChecking dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Dataset found.")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading dataset...")

    data = np.load(DATA_FILE)

    X_train = data["X_train"].astype(np.float32)
    y_train_raw = data["y_train"]

    X_val = data["X_val"].astype(np.float32)
    y_val_raw = data["y_val"]

    X_test = data["X_test"].astype(np.float32)
    y_test_raw = data["y_test"]

    print(f"Training:   {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test:       {X_test.shape}")

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    print("\nConverting labels...")

    y_train = convert_labels(y_train_raw)
    y_val = convert_labels(y_val_raw)
    y_test = convert_labels(y_test_raw)

    print("\nClasses:")

    for index, name in enumerate(CLASS_NAMES):
        print(
            f"  {index}: {name}"
        )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    print("\nCalculating training normalization...")

    train_mean, train_std = calculate_training_statistics(
        X_train
    )

    print(
        f"Training mean: {train_mean:.8f}"
    )

    print(
        f"Training std:  {train_std:.8f}"
    )

    print("\nNormalizing signals...")

    X_train = normalize(
        X_train,
        train_mean,
        train_std
    )

    X_val = normalize(
        X_val,
        train_mean,
        train_std
    )

    X_test = normalize(
        X_test,
        train_mean,
        train_std
    )

    # --------------------------------------------------------
    # CNN INPUT SHAPE
    # --------------------------------------------------------

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(
        f"\nCNN input shape: {X_train.shape}"
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\nBuilding cloud CNN...")

    model = build_cloud_model()

    model.summary()

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODEL_FILE),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nStarting cloud model training...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print("\nLoading best cloud model...")

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nEvaluating cloud model on test set...")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print("\n" + "=" * 70)
    print("CLOUD MODEL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test loss:     {test_loss:.4f}"
    )

    print(
        f"Test accuracy: {test_accuracy:.4f}"
    )

    print(
        f"Test accuracy: {test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("CLOUD CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    np.savez_compressed(
        METADATA_FILE,
        mean=np.array(train_mean),
        std=np.array(train_std),
        classes=CLASS_NAMES
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "model": "cloud_cnn",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "num_parameters": int(
            model.count_params()
        ),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "classes": CLASS_NAMES.tolist(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "training_epochs_completed": len(
            history.history["loss"]
        ),
        "normalization": {
            "mean": train_mean,
            "std": train_std
        }
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
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLOUD MODEL TRAINING COMPLETE")
    print("=" * 70)

    print("\nModel saved to:")
    print(MODEL_FILE)

    print("\nMetadata saved to:")
    print(METADATA_FILE)

    print("\nMetrics saved to:")
    print(METRICS_FILE)

    print("\nCloud model parameters:")
    print(
        f"{model.count_params():,}"
    )

    print("\nCloud test accuracy:")
    print(
        f"{test_accuracy * 100:.2f}%"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()