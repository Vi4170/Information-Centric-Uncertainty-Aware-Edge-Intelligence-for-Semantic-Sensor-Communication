import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Input
)
from tensorflow.keras.callbacks import EarlyStopping


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

MODEL_DIR = PROJECT_ROOT / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "cnn_baseline.keras"


print("=" * 60)
print("CWRU BASELINE 1D CNN")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

data = np.load(DATA_FILE)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

print(f"Training:   {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")

# ============================================================
# NORMALIZE SIGNALS
# ============================================================

print("\nNormalizing signals...")

mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / (std + 1e-8)
X_val = (X_val - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

# CNN expects:
# samples × time × channels

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"CNN input shape: {X_train.shape}")

# ============================================================
# ENCODE LABELS
# ============================================================

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)
y_test_encoded = encoder.transform(y_test)

print("\nClasses:")

for i, name in enumerate(encoder.classes_):
    print(f"  {i}: {name}")

num_classes = len(encoder.classes_)

# ============================================================
# BUILD CNN
# ============================================================

print("\nBuilding CNN...")

model = Sequential([
    Input(shape=(2048, 1)),

    Conv1D(
        filters=32,
        kernel_size=7,
        activation="relu"
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu"
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    GlobalAveragePooling1D(),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train_encoded,

    validation_data=(
        X_val,
        y_val_encoded
    ),

    epochs=30,
    batch_size=64,

    callbacks=[
        early_stopping
    ],

    verbose=1
)

# ============================================================
# TEST
# ============================================================

print("\nEvaluating on test set...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_encoded,
    verbose=0
)

print(f"\nTest loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

y_pred_prob = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

print("\nClassification report:")
print(
    classification_report(
        y_test_encoded,
        y_pred,
        target_names=encoder.classes_
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

# Save normalization parameters and class names
np.savez(
    MODEL_DIR / "cnn_baseline_metadata.npz",
    mean=mean,
    std=std,
    classes=encoder.classes_
)

print("\n" + "=" * 60)
print("BASELINE TRAINING COMPLETE")
print("=" * 60)