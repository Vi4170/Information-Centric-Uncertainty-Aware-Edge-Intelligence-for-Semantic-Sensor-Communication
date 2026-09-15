"""Task 34 -- modality-specific learned representations for the canonical
multisensor pipeline (vibration, motor_current, temperature).

Reuse assessment (vibration/current vs. the existing CWRU encoder):
``src/cnn/model.py::build_baseline_cnn`` already takes ``input_shape`` and
``num_classes`` as genuine, unhardcoded constructor arguments -- its Conv1D
/ GlobalAveragePooling1D / Dense stack is channel-count-agnostic by
construction (a Keras Conv1D layer works over however many input channels
its Input layer declares; only the first layer's parameter count changes).
Paderborn already reused this exact function, unmodified, at a different
sampling rate (64 kHz vs CWRU's 12 kHz) with a different class count (3 vs
4) and it worked (99.1% test accuracy). Extending that same precedent to a
different CHANNEL count (4 for vibration, 3 for motor_current here, still
single-channel-per-call, just more channels) is the same kind of genuine,
verified compatibility -- not an assumption that "it's vibration so it
must fit". Verified directly: no source change to src/cnn/model.py or
src/cnn/config.py was needed or made. Vibration and motor_current each get
their OWN model instance (separate weights, separate architecture calls,
same function) -- they are never combined into one input tensor.

Temperature is treated differently, deliberately: Task 33 already reduced
each temperature window to its real per-channel mean (2 values), because
temperature's raw ADC rate (~25.6 kHz) is an acquisition-hardware artifact
shared with motor_current, not evidence of high-frequency physical
information. Given the representation is already just 2 real numbers,
training a neural network on top of them would add parameters and a
leakage-risk surface with no principled basis, and would directly
contradict "avoid unnecessary complexity". Temperature's Task 34
"representation" is therefore the identity function over Task 33's already
train-only-normalized 2-value output -- no model, no fitting, no
serialization needed for temperature.

Classification target: fault_type (5 classes: Normal, BPFI, BPFO,
Misalignment, Unbalance), the same kind of target CWRU/Paderborn already
use, applied at the window level (every window inherits its source
condition's single fault_type label, consistent with how every other
dataset in this repo assigns condition/file-level labels to windows).

Discovered limitation, reported rather than silently patched: Task 32's
condition-level 70/15/15 split is not stratified by fault_type, and there
are only 3 "Normal" conditions in the whole dataset (one per load level).
With this session's seed, all 3 landed in the train split -- val and test
have ZERO Normal-class observations. This does not prevent training (the
classifier still trains on all 5 classes), but it does mean the Normal
class's representation quality cannot be validated on held-out data here.
This is disclosed in the saved config/report rather than hidden, and the
split itself is NOT modified (that would change Task 32's already-persisted
observation_id -> split mapping for all 56,250 observations, which is out
of scope for a representation-learning task).

src/voi/ is not imported or modified anywhere in this module.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import pandas as pd

from src.cnn.config import BATCH_SIZE, EMBEDDING_DIM, LEARNING_RATE, RANDOM_SEED
from src.cnn.model import build_baseline_cnn, extract_embeddings
from src.cnn.train import set_random_seed
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp

# ---------------------------------------------------------------------------
# Classification target (window-level, inherited from source condition)
# ---------------------------------------------------------------------------

FAULT_TYPE_LABELS: Tuple[str, ...] = ("Normal", "BPFI", "BPFO", "Misalignment", "Unbalance")
FAULT_TYPE_TO_LABEL_ID: Dict[str, int] = {name: i for i, name in enumerate(FAULT_TYPE_LABELS)}
NUM_CLASSES: int = len(FAULT_TYPE_LABELS)

KNOWN_LIMITATION_ZERO_NORMAL_VAL_TEST_SUPPORT: str = (
    "Task 32's condition-level split is unstratified and only 3 Normal "
    "conditions exist in the whole dataset; with this session's seed, all "
    "3 landed in train, leaving 0 Normal-class observations in val/test. "
    "Training proceeds on all 5 classes; Normal-class representation "
    "quality cannot be validated on held-out data. The split itself was "
    "not modified to fix this -- see module docstring."
)

# ---------------------------------------------------------------------------
# Architecture / dimensionality (Task 34 requirement 6)
# ---------------------------------------------------------------------------

VIBRATION_INPUT_SHAPE: Tuple[int, int] = (schema.WINDOW_SIZE_SAMPLES, len(schema.VIBRATION_CHANNEL_ORDER))
VIBRATION_EMBEDDING_DIM: int = EMBEDDING_DIM

# Verified empirically constant (2049) across all 45 real conditions --
# see tests/test_multimodal_representation.py -- not assumed.
MOTOR_CURRENT_WINDOW_SIZE_SAMPLES: int = 2049
MOTOR_CURRENT_INPUT_SHAPE: Tuple[int, int] = (
    MOTOR_CURRENT_WINDOW_SIZE_SAMPLES,
    len(pp.MOTOR_CURRENT_CHANNEL_NAMES),
)
MOTOR_CURRENT_EMBEDDING_DIM: int = EMBEDDING_DIM

TEMPERATURE_REPRESENTATION_DIM: int = len(pp.TEMPERATURE_CHANNEL_NAMES)
TEMPERATURE_ARCHITECTURE: str = (
    "identity passthrough of Task 33's per-window per-channel mean "
    "(train-only normalized); no learned model"
)

REPRESENTATION_EPOCHS: int = 8  # fixed a priori; see module docstring for rationale
REPRESENTATION_BATCH_SIZE: int = BATCH_SIZE

MODEL_DIR: str = os.path.join("models", "multimodal")
VIBRATION_MODEL_PATH: str = os.path.join(MODEL_DIR, "vibration_encoder.keras")
MOTOR_CURRENT_MODEL_PATH: str = os.path.join(MODEL_DIR, "motor_current_encoder.keras")
REPRESENTATION_CONFIG_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_representation_config.json"
)
TRAINING_HISTORY_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_representation_training_history.json"
)


# ---------------------------------------------------------------------------
# Encoder construction (reuses build_baseline_cnn UNCHANGED)
# ---------------------------------------------------------------------------

def build_vibration_encoder() -> keras.Model:
    return build_baseline_cnn(
        input_shape=VIBRATION_INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
        embedding_dim=VIBRATION_EMBEDDING_DIM,
    )


def build_motor_current_encoder() -> keras.Model:
    return build_baseline_cnn(
        input_shape=MOTOR_CURRENT_INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
        embedding_dim=MOTOR_CURRENT_EMBEDDING_DIM,
    )


# ---------------------------------------------------------------------------
# Split-level array assembly (reuses Task 33's per-condition builders + fitted
# train-only normalization; never refits normalization here)
# ---------------------------------------------------------------------------

def assemble_split_arrays(
    modality: str,
    split: str,
    raw_dir: str = schema.RAW_DATA_DIR,
    normalization_params: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[Dict[str, str]]]:
    """Build (X, y, metadata_df, excluded_conditions) for one modality and
    one split, by iterating that split's conditions and reusing Task 33's
    per-condition window builders. y is the window's fault_type label id
    (broadcast from its source condition). Never touches conditions outside
    `split`.

    A condition whose raw file is internally corrupt (see
    preprocessing.CorruptRawFileError -- discovered for real on
    4Nm_BPFO_10.tdms, which has two empty current channels) is excluded
    from the returned arrays and recorded in `excluded_conditions` rather
    than fabricated or silently dropped without a trace.
    """
    if modality not in schema.CANONICAL_MODALITIES:
        raise ValueError(f"Unknown modality '{modality}'. Expected one of {schema.CANONICAL_MODALITIES}")
    if split not in schema.SPLIT_NAMES:
        raise ValueError(f"Unknown split '{split}'. Expected one of {schema.SPLIT_NAMES}")

    if normalization_params is None:
        normalization_params = pp.load_normalization_params()
    norm = normalization_params[modality]

    registry = pp.get_condition_registry(raw_dir)
    split_rows = registry[registry["split"] == split]

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    meta_parts: List[pd.DataFrame] = []
    excluded: List[Dict[str, str]] = []

    for _, row in split_rows.iterrows():
        label_id = FAULT_TYPE_TO_LABEL_ID[row["fault_type"]]
        try:
            if modality == "vibration":
                X, meta = pp.build_vibration_windows_for_condition(row, normalization=norm)
            elif modality == "motor_current":
                X, meta = pp.build_motor_current_windows_for_condition(row, normalization=norm)
            else:  # temperature
                X, meta = pp.build_temperature_features_for_condition(row, normalization=norm)
        except pp.CorruptRawFileError as exc:
            excluded.append({"condition_code": row["condition_code"], "modality": modality, "reason": str(exc)})
            continue

        if len(X) == 0:
            continue
        X_parts.append(X)
        y_parts.append(np.full(len(X), label_id, dtype=np.int64))
        meta_parts.append(meta)

    if not X_parts:
        return (
            np.empty((0,) + (VIBRATION_INPUT_SHAPE if modality == "vibration" else ()), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(),
            excluded,
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    metadata_df = pd.concat(meta_parts, ignore_index=True)
    return X, y, metadata_df, excluded


# ---------------------------------------------------------------------------
# Training (train-only fitting; val used only for monitoring, never for
# gradient updates or architecture selection; test never touched here)
# ---------------------------------------------------------------------------

def _train_encoder(
    modality: str,
    build_fn,
    raw_dir: str = schema.RAW_DATA_DIR,
    epochs: int = REPRESENTATION_EPOCHS,
    batch_size: int = REPRESENTATION_BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[keras.Model, Dict[str, object]]:
    set_random_seed(seed)
    normalization_params = pp.load_normalization_params()

    X_train, y_train, _, excluded_train = assemble_split_arrays(modality, "train", raw_dir, normalization_params)
    X_val, y_val, _, excluded_val = assemble_split_arrays(modality, "val", raw_dir, normalization_params)

    model = build_fn()
    fit_history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    history = {
        "modality": modality,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_train_windows": int(len(X_train)),
        "n_val_windows": int(len(X_val)),
        "train_loss": [float(v) for v in fit_history.history.get("loss", [])],
        "train_accuracy": [float(v) for v in fit_history.history.get("accuracy", [])],
        "val_loss": [float(v) for v in fit_history.history.get("val_loss", [])],
        "val_accuracy": [float(v) for v in fit_history.history.get("val_accuracy", [])],
        "excluded_conditions": excluded_train + excluded_val,
    }
    return model, history


def train_vibration_encoder(
    raw_dir: str = schema.RAW_DATA_DIR,
    epochs: int = REPRESENTATION_EPOCHS,
    seed: int = RANDOM_SEED,
) -> Tuple[keras.Model, Dict[str, object]]:
    return _train_encoder("vibration", build_vibration_encoder, raw_dir, epochs, seed=seed)


def train_motor_current_encoder(
    raw_dir: str = schema.RAW_DATA_DIR,
    epochs: int = REPRESENTATION_EPOCHS,
    seed: int = RANDOM_SEED,
) -> Tuple[keras.Model, Dict[str, object]]:
    return _train_encoder("motor_current", build_motor_current_encoder, raw_dir, epochs, seed=seed)


# ---------------------------------------------------------------------------
# Persistence (deterministic save/load, reusing Keras's own serialization)
# ---------------------------------------------------------------------------

def save_encoder(model: keras.Model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)


def load_encoder(path: str) -> keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved encoder found at '{path}'")
    return keras.models.load_model(path, compile=False)


# ---------------------------------------------------------------------------
# Embedding / representation extraction (inference-time, deterministic)
# ---------------------------------------------------------------------------

def get_vibration_embeddings(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return extract_embeddings(model, X)


def get_motor_current_embeddings(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return extract_embeddings(model, X)


def get_temperature_representation(
    condition_row: pd.Series,
    normalization_params: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Temperature's Task 34 'representation' is Task 33's own per-window
    mean output, train-only normalized -- no model involved."""
    if normalization_params is None:
        normalization_params = pp.load_normalization_params()
    return pp.build_temperature_features_for_condition(
        condition_row, normalization=normalization_params["temperature"]
    )


# ---------------------------------------------------------------------------
# Machine-readable architecture/dimensionality record (requirement 6)
# ---------------------------------------------------------------------------

def build_representation_config() -> Dict[str, object]:
    return {
        "task": 34,
        "title": "Modality-specific learned representations",
        "fault_type_labels": list(FAULT_TYPE_LABELS),
        "num_classes": NUM_CLASSES,
        "known_limitation_zero_normal_val_test_support": KNOWN_LIMITATION_ZERO_NORMAL_VAL_TEST_SUPPORT,
        "vibration": {
            "architecture": "build_baseline_cnn (src/cnn/model.py, unmodified)",
            "reused_existing_cwru_encoder": True,
            "reuse_justification": (
                "input_shape/num_classes are genuine constructor parameters; "
                "Conv1D/GAP/Dense stack is channel-count-agnostic; Paderborn "
                "already validated this reuse pattern at a different sampling "
                "rate and class count."
            ),
            "input_shape": list(VIBRATION_INPUT_SHAPE),
            "embedding_dim": VIBRATION_EMBEDDING_DIM,
            "channels": list(schema.VIBRATION_CHANNEL_ORDER),
        },
        "motor_current": {
            "architecture": "build_baseline_cnn (src/cnn/model.py, unmodified)",
            "reused_existing_cwru_encoder": True,
            "reuse_justification": "same as vibration; separate model instance, own weights, never concatenated with vibration input.",
            "input_shape": list(MOTOR_CURRENT_INPUT_SHAPE),
            "embedding_dim": MOTOR_CURRENT_EMBEDDING_DIM,
            "channels": list(pp.MOTOR_CURRENT_CHANNEL_NAMES),
        },
        "temperature": {
            "architecture": TEMPERATURE_ARCHITECTURE,
            "reused_existing_cwru_encoder": False,
            "reuse_justification": (
                "Task 33 already reduced temperature to 2 real per-channel "
                "means; training a network on 2 numbers would add parameters "
                "and leakage-risk surface with no principled basis, violating "
                "the avoid-unnecessary-complexity requirement."
            ),
            "representation_dim": TEMPERATURE_REPRESENTATION_DIM,
            "channels": list(pp.TEMPERATURE_CHANNEL_NAMES),
        },
        "training": {
            "epochs": REPRESENTATION_EPOCHS,
            "batch_size": REPRESENTATION_BATCH_SIZE,
            "optimizer_learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
            "fit_only_on_split": "train",
            "validation_split_used_for_monitoring_only": "val",
            "test_split_touched_during_training": False,
        },
        "model_paths": {
            "vibration": VIBRATION_MODEL_PATH,
            "motor_current": MOTOR_CURRENT_MODEL_PATH,
        },
    }


def save_representation_config(path: str = REPRESENTATION_CONFIG_JSON_PATH) -> Dict[str, object]:
    config = build_representation_config()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return config


def save_training_history(histories: Dict[str, Dict[str, object]], path: str = TRAINING_HISTORY_JSON_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    save_representation_config()

    vib_model, vib_history = train_vibration_encoder()
    save_encoder(vib_model, VIBRATION_MODEL_PATH)

    cur_model, cur_history = train_motor_current_encoder()
    save_encoder(cur_model, MOTOR_CURRENT_MODEL_PATH)

    save_training_history({"vibration": vib_history, "motor_current": cur_history})

    print("Vibration  -- final train/val accuracy:", vib_history["train_accuracy"][-1], vib_history["val_accuracy"][-1])
    print("Motor current -- final train/val accuracy:", cur_history["train_accuracy"][-1], cur_history["val_accuracy"][-1])
