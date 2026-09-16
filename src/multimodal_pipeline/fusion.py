"""Task 35 -- leakage-safe multimodal fusion of the Task 34 modality
representations (vibration 64-d embedding, motor_current 64-d embedding,
temperature 2-d representation) into one unified machine-state
representation.

Architecture (simplest scientifically defensible choice, per Task 35
requirement 2): concatenate the modality representations in a fixed order,
then a single learned Dense projection ("fused_representation") trained via
the same supervised fault_type-classification objective Task 34 already
uses, so the penultimate layer serves as the fused embedding -- exactly the
same "train a small classifier, keep its penultimate layer" pattern used
throughout this repo (CWRU/Paderborn/Task 34), extended one level up. No
attention/transformer mechanism is used; none is demonstrated to be needed
by anything built so far.

Task 34's modality encoders are treated as authoritative and FROZEN: this
module only calls ``representation.load_encoder`` (never
``build_vibration_encoder``/``build_motor_current_encoder`` for retraining)
and ``representation.get_*_embeddings`` for inference. Only the small
fusion head (one Dense layer + a classification head on top, for training
purposes) has learnable parameters here. Temperature's representation is
Task 33's identity output, unmodified.

Four ablation configurations are supported, all vibration-anchored (Task 35
requirement 4 does not ask for a current-only or temperature-only
configuration):

    vibration_only                 : (vibration,)                    ->  64-d input
    vibration_current               : (vibration, motor_current)      -> 128-d input
    vibration_temperature           : (vibration, temperature)        ->  66-d input
    vibration_current_temperature   : (vibration, motor_current, temperature) -> 130-d input

Every configuration projects to the SAME fused dimension (64, reusing
EMBEDDING_DIM) so a later consumer (Task 36+) can treat any configuration's
output interchangeably. Concatenation always follows MODALITY_ORDER =
(vibration, motor_current, temperature); a configuration simply omits
absent modalities from that fixed order -- never reorders them.

Missing/corrupt current handling: reuses Task 34's ``pp.CorruptRawFileError``,
raised independently per modality (Task 35's corrective audit, see
docs/multimodal_task35_bpfo_temperature_audit.md). Direct inspection of all
45 raw temperature_current .tdms files confirmed the corruption is scoped to
current only: all 9 BPFO conditions (every load x severity) have exactly 2
of their 3 current-phase channels empty, while both temperature channels in
those same files are fully present, correctly lengthed, and physically
plausible. A condition is therefore excluded ONLY from configurations that
require the specific modality whose own channel group failed validation --
motor_current-requiring configurations (vibration_current,
vibration_current_temperature) exclude all 9 BPFO conditions, but
vibration_temperature does not, since temperature alone is genuinely valid
there. vibration_only is unaffected either way, since it never reads that
file. Never fabricated, never zero-filled.

Window-count mismatch handling: Task 33 established that vibration can have
exactly one more window per condition than motor_current/temperature (a
real measured-rate difference, not a bug). Any configuration requiring
motor_current or temperature is built from the INTERSECTION of observation
ids that have every required modality -- the trailing vibration-only window
is silently excluded from THOSE configurations (not fabricated), while
vibration_only itself still includes it.

Class-support caveat carried forward unmodified from Task 34: Task 32's
condition-level split is not stratified by fault_type, and all 3 "Normal"
conditions landed in train under this session's seed -- val/test have zero
Normal-class support. This is recorded explicitly in the saved fusion
config rather than fixed by reassigning conditions (the split is Task 32's,
and reassigning it here would be exactly the kind of "do not reassign
conditions merely to obtain balanced metrics" the task prohibits).

src/voi/ is not imported or modified anywhere in this module.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import pandas as pd
from keras import layers

from src.cnn.config import LEARNING_RATE, RANDOM_SEED
from src.cnn.model import extract_embeddings
from src.cnn.train import set_random_seed
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep

MODEL_VERSION: str = "task35_v1"

MODALITY_ORDER: Tuple[str, ...] = ("vibration", "motor_current", "temperature")

FUSION_CONFIGS: Dict[str, Tuple[str, ...]] = {
    "vibration_only": ("vibration",),
    "vibration_current": ("vibration", "motor_current"),
    "vibration_temperature": ("vibration", "temperature"),
    "vibration_current_temperature": ("vibration", "motor_current", "temperature"),
}

_MODALITY_DIM: Dict[str, int] = {
    "vibration": rep.VIBRATION_EMBEDDING_DIM,
    "motor_current": rep.MOTOR_CURRENT_EMBEDDING_DIM,
    "temperature": rep.TEMPERATURE_REPRESENTATION_DIM,
}

FUSED_DIM: int = rep.VIBRATION_EMBEDDING_DIM  # 64; same output dim for every configuration
FUSION_EPOCHS: int = 8  # fixed a priori, matching Task 34's rationale; not tuned against results
FUSION_BATCH_SIZE: int = rep.REPRESENTATION_BATCH_SIZE

FUSION_MODEL_DIR: str = os.path.join(rep.MODEL_DIR, "fusion")
FUSION_CONFIG_JSON_PATH: str = os.path.join(schema.PROCESSED_DATA_DIR, "multimodal_fusion_config.json")
FUSION_TRAINING_HISTORY_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_fusion_training_history.json"
)


def fusion_model_path(config_name: str) -> str:
    if config_name not in FUSION_CONFIGS:
        raise ValueError(f"Unknown fusion config '{config_name}'. Expected one of {tuple(FUSION_CONFIGS)}")
    return os.path.join(FUSION_MODEL_DIR, f"{config_name}.keras")


def concat_dim_for_config(config_name: str) -> int:
    if config_name not in FUSION_CONFIGS:
        raise ValueError(f"Unknown fusion config '{config_name}'. Expected one of {tuple(FUSION_CONFIGS)}")
    return sum(_MODALITY_DIM[m] for m in FUSION_CONFIGS[config_name])


# ---------------------------------------------------------------------------
# Per-split representation table: reads every raw file in the split exactly
# once, running the FROZEN Task 34 encoders for inference only -- shared
# across all 4 fusion configurations to avoid redundant I/O/inference.
# ---------------------------------------------------------------------------

def build_split_representation_table(
    split: str,
    vibration_model: keras.Model,
    motor_current_model: keras.Model,
    normalization_params: Dict[str, Dict[str, object]],
    raw_dir: str = schema.RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """One row per vibration observation_id in `split` (vibration is the
    modality with the most windows per condition). current_embedding /
    temperature_representation are None where genuinely unavailable --
    never fabricated. Returns (table, excluded_conditions)."""
    if split not in schema.SPLIT_NAMES:
        raise ValueError(f"Unknown split '{split}'. Expected one of {schema.SPLIT_NAMES}")

    registry = pp.get_condition_registry(raw_dir)
    split_rows = registry[registry["split"] == split]

    records: List[dict] = []
    excluded: List[Dict[str, str]] = []

    for _, row in split_rows.iterrows():
        Xv, mv = pp.build_vibration_windows_for_condition(row, normalization=normalization_params["vibration"])
        vib_emb = rep.get_vibration_embeddings(vibration_model, Xv)
        vib_lookup = dict(zip(mv["observation_id"], vib_emb))

        cur_lookup: Dict[str, np.ndarray] = {}
        try:
            Xc, mc = pp.build_motor_current_windows_for_condition(
                row, normalization=normalization_params["motor_current"]
            )
            cur_emb = rep.get_motor_current_embeddings(motor_current_model, Xc)
            cur_lookup = dict(zip(mc["observation_id"], cur_emb))
        except pp.CorruptRawFileError as exc:
            excluded.append({"condition_code": row["condition_code"], "modality": "motor_current", "reason": str(exc)})

        # Temperature shares its raw file with motor_current, but the two
        # are physically independent channel groups -- Task 35's corrective
        # audit (see docs/multimodal_task35_bpfo_temperature_audit.md)
        # confirmed by direct raw-channel inspection that all 9 BPFO
        # conditions have fully valid temperature data despite invalid
        # current data. Temperature's own validity is therefore checked
        # independently here, never inferred from current's outcome.
        temp_lookup: Dict[str, np.ndarray] = {}
        try:
            Xt, mt = pp.build_temperature_features_for_condition(row, normalization=normalization_params["temperature"])
            temp_lookup = dict(zip(mt["observation_id"], Xt))
        except pp.CorruptRawFileError as exc:
            excluded.append({"condition_code": row["condition_code"], "modality": "temperature", "reason": str(exc)})

        for observation_id, vibration_embedding in vib_lookup.items():
            records.append(
                {
                    "observation_id": observation_id,
                    "condition_code": row["condition_code"],
                    "fault_type": row["fault_type"],
                    "split": split,
                    "source_vibration_file": row["vibration_filename"],
                    "source_temperature_current_file": row["temperature_current_filename"],
                    "vibration_embedding": vibration_embedding,
                    "current_embedding": cur_lookup.get(observation_id),
                    "temperature_representation": temp_lookup.get(observation_id),
                    "has_vibration": True,
                    "has_current": observation_id in cur_lookup,
                    "has_temperature": observation_id in temp_lookup,
                }
            )

    return pd.DataFrame(records), excluded


_MODALITY_COLUMN: Dict[str, str] = {
    "vibration": "vibration_embedding",
    "motor_current": "current_embedding",
    "temperature": "temperature_representation",
}
_MODALITY_AVAILABILITY_COLUMN: Dict[str, str] = {
    "vibration": "has_vibration",
    "motor_current": "has_current",
    "temperature": "has_temperature",
}


def assemble_fusion_split_arrays(
    config_name: str,
    table: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Filter `table` (from build_split_representation_table) to rows where
    EVERY modality this configuration requires is genuinely present, then
    concatenate in MODALITY_ORDER. Never substitutes a missing modality
    with a zero vector or another modality's value."""
    if config_name not in FUSION_CONFIGS:
        raise ValueError(f"Unknown fusion config '{config_name}'. Expected one of {tuple(FUSION_CONFIGS)}")

    required = FUSION_CONFIGS[config_name]
    if table.empty:
        return np.empty((0, concat_dim_for_config(config_name)), dtype=np.float32), np.empty((0,), dtype=np.int64), table

    mask = pd.Series(True, index=table.index)
    for modality in required:
        mask &= table[_MODALITY_AVAILABILITY_COLUMN[modality]]
    subset = table[mask].reset_index(drop=True)

    if subset.empty:
        return np.empty((0, concat_dim_for_config(config_name)), dtype=np.float32), np.empty((0,), dtype=np.int64), subset

    parts = [
        np.stack(subset[_MODALITY_COLUMN[modality]].to_numpy()).astype(np.float32)
        for modality in MODALITY_ORDER
        if modality in required
    ]
    X = np.concatenate(parts, axis=1)
    y = subset["fault_type"].map(rep.FAULT_TYPE_TO_LABEL_ID).to_numpy()
    return X, y, subset


# ---------------------------------------------------------------------------
# Fusion head architecture (simple: concat -> Dense projection -> classifier)
# ---------------------------------------------------------------------------

def build_fusion_head(concat_dim: int, fused_dim: int = FUSED_DIM) -> keras.Model:
    inputs = keras.Input(shape=(concat_dim,), name="concatenated_representation")
    fused = layers.Dense(fused_dim, activation="relu", name="fused_representation")(inputs)
    x = layers.Dropout(0.2, name="dropout")(fused)
    outputs = layers.Dense(rep.NUM_CLASSES, activation="softmax", name="output_probabilities")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="fusion_head")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_fused_representation(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return extract_embeddings(model, X, layer_name="fused_representation")


# ---------------------------------------------------------------------------
# Training (frozen modality encoders; train-only fitting; val for monitoring
# only; test never touched)
# ---------------------------------------------------------------------------

def train_all_fusion_heads(
    raw_dir: str = schema.RAW_DATA_DIR,
    epochs: int = FUSION_EPOCHS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, object]]:
    """Loads Task 34's frozen encoders once, builds the train/val
    representation tables once, then trains all 4 fusion configurations
    reusing those tables (no redundant raw-file reads or re-inference per
    configuration)."""
    set_random_seed(seed)

    vibration_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
    motor_current_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
    normalization_params = pp.load_normalization_params()

    train_table, excluded_train = build_split_representation_table(
        "train", vibration_model, motor_current_model, normalization_params, raw_dir
    )
    val_table, excluded_val = build_split_representation_table(
        "val", vibration_model, motor_current_model, normalization_params, raw_dir
    )
    excluded = excluded_train + excluded_val

    results: Dict[str, Dict[str, object]] = {}
    for config_name in FUSION_CONFIGS:
        set_random_seed(seed)
        X_train, y_train, meta_train = assemble_fusion_split_arrays(config_name, train_table)
        X_val, y_val, meta_val = assemble_fusion_split_arrays(config_name, val_table)

        model = build_fusion_head(concat_dim_for_config(config_name))
        fit_history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=epochs,
            batch_size=FUSION_BATCH_SIZE,
            verbose=2,
        )

        train_classes_present = sorted(set(meta_train["fault_type"])) if len(meta_train) else []
        val_classes_present = sorted(set(meta_val["fault_type"])) if len(meta_val) else []
        missing_train_classes = sorted(set(rep.FAULT_TYPE_LABELS) - set(train_classes_present))

        history = {
            "config": config_name,
            "modalities": list(FUSION_CONFIGS[config_name]),
            "concat_dim": concat_dim_for_config(config_name),
            "fused_dim": FUSED_DIM,
            "epochs": epochs,
            "batch_size": FUSION_BATCH_SIZE,
            "n_train_observations": int(len(X_train)),
            "n_val_observations": int(len(X_val)),
            "train_loss": [float(v) for v in fit_history.history.get("loss", [])],
            "train_accuracy": [float(v) for v in fit_history.history.get("accuracy", [])],
            "val_loss": [float(v) for v in fit_history.history.get("val_loss", [])],
            "val_accuracy": [float(v) for v in fit_history.history.get("val_accuracy", [])],
            "train_classes_present": train_classes_present,
            "val_classes_present": val_classes_present,
            "classes_with_zero_training_support": missing_train_classes,
        }
        results[config_name] = {"model": model, "history": history}

        path = fusion_model_path(config_name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model.save(path)

    results["_excluded_conditions"] = excluded  # type: ignore[assignment]
    return results


def load_fusion_head(config_name: str) -> keras.Model:
    path = fusion_model_path(config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained fusion head found at '{path}'")
    return keras.models.load_model(path, compile=False)


# ---------------------------------------------------------------------------
# Class support reporting (Task 35 requirement 7) -- condition-level counts,
# no raw file access needed.
# ---------------------------------------------------------------------------

def class_support_by_split(raw_dir: str = schema.RAW_DATA_DIR) -> Dict[str, Dict[str, int]]:
    registry = pp.get_condition_registry(raw_dir)
    support: Dict[str, Dict[str, int]] = {}
    for split_name in schema.SPLIT_NAMES:
        split_rows = registry[registry["split"] == split_name]
        counts = {fault_type: 0 for fault_type in rep.FAULT_TYPE_LABELS}
        counts.update(split_rows["fault_type"].value_counts().to_dict())
        support[split_name] = {k: int(v) for k, v in counts.items()}
    return support


# ---------------------------------------------------------------------------
# Machine-readable architecture/provenance record (requirement 13)
# ---------------------------------------------------------------------------

def build_fusion_config_record(
    excluded_conditions: Optional[List[Dict[str, str]]] = None,
    per_config_class_coverage: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    return {
        "task": 35,
        "title": "Leakage-safe multimodal fusion",
        "model_version": MODEL_VERSION,
        "modality_order": list(MODALITY_ORDER),
        "modality_embedding_dims": dict(_MODALITY_DIM),
        "fused_dim": FUSED_DIM,
        "configurations": {
            name: {
                "modalities": list(modalities),
                "concat_dim": concat_dim_for_config(name),
                "model_path": fusion_model_path(name),
            }
            for name, modalities in FUSION_CONFIGS.items()
        },
        "modality_encoders_frozen": True,
        "modality_encoder_paths": {
            "vibration": rep.VIBRATION_MODEL_PATH,
            "motor_current": rep.MOTOR_CURRENT_MODEL_PATH,
        },
        "architecture": "concatenation -> Dense(fused_dim, relu, name='fused_representation') -> Dropout(0.2) -> Dense(num_classes, softmax)",
        "training": {
            "epochs": FUSION_EPOCHS,
            "batch_size": FUSION_BATCH_SIZE,
            "optimizer_learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
            "fit_only_on_split": "train",
            "validation_split_used_for_monitoring_only": "val",
            "test_split_touched_during_training": False,
        },
        "class_support_by_split": class_support_by_split(),
        "class_coverage_by_configuration": per_config_class_coverage or {},
        "known_limitation_zero_normal_val_test_support": rep.KNOWN_LIMITATION_ZERO_NORMAL_VAL_TEST_SUPPORT,
        "known_limitation_weak_motor_current_standalone_accuracy": (
            "Task 34 found motor_current's standalone validation accuracy is "
            "near chance (~18%) for this 5-class task. Not modified, tuned, "
            "or worked around here -- fusion configurations that include "
            "motor_current are trained and reported exactly as observed; "
            "whether current contributes complementary information is left "
            "to a later task, not decided here."
        ),
        "known_limitation_bpfo_missing_from_current_configs": (
            "Discovered while implementing Task 35, corrected by Task 35's "
            "corrective audit (docs/multimodal_task35_bpfo_temperature_audit.md): "
            "ALL 9 BPFO conditions (every load x severity) have 2 of their 3 "
            "current-phase channels empty in the raw temperature+current "
            ".tdms file. The initial Task 35 implementation incorrectly "
            "treated this as invalidating BOTH current AND temperature "
            "(they share one raw file), excluding BPFO from every "
            "configuration that used either modality. Direct inspection of "
            "all 45 raw files confirmed temperature's own 2 channels are "
            "fully present, correctly lengthed, and physically plausible in "
            "every one of the 9 BPFO files -- only current is genuinely "
            "invalid. Fusion configurations requiring motor_current "
            "(vibration_current, vibration_current_temperature) therefore "
            "still train and validate with ZERO BPFO examples -- effectively "
            "a 4-class problem for those two configurations, even though the "
            "output layer still has 5 units. vibration_only and "
            "vibration_temperature both retain genuine 5-class coverage, "
            "including BPFO. See class_coverage_by_configuration for the "
            "exact per-configuration class lists; not fabricated, not worked "
            "around, not hidden."
        ),
        "excluded_conditions": excluded_conditions or [],
    }


def save_fusion_config(
    excluded_conditions: Optional[List[Dict[str, str]]] = None,
    per_config_class_coverage: Optional[Dict[str, Dict[str, object]]] = None,
    path: str = FUSION_CONFIG_JSON_PATH,
) -> Dict[str, object]:
    config = build_fusion_config_record(excluded_conditions, per_config_class_coverage)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return config


def save_fusion_training_history(
    results: Dict[str, Dict[str, object]],
    path: str = FUSION_TRAINING_HISTORY_JSON_PATH,
) -> None:
    histories = {name: r["history"] for name, r in results.items() if name in FUSION_CONFIGS}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    results = train_all_fusion_heads()
    excluded = results.pop("_excluded_conditions")  # type: ignore[arg-type]

    per_config_class_coverage = {
        name: {
            "train_classes_present": r["history"]["train_classes_present"],
            "val_classes_present": r["history"]["val_classes_present"],
            "classes_with_zero_training_support": r["history"]["classes_with_zero_training_support"],
        }
        for name, r in results.items()
    }
    save_fusion_config(excluded_conditions=excluded, per_config_class_coverage=per_config_class_coverage)
    save_fusion_training_history(results)

    for name, r in results.items():
        h = r["history"]
        print(
            f"{name}: concat_dim={h['concat_dim']} n_train={h['n_train_observations']} "
            f"n_val={h['n_val_observations']} final_train_acc={h['train_accuracy'][-1]:.4f} "
            f"final_val_acc={(h['val_accuracy'][-1] if h['val_accuracy'] else float('nan')):.4f}"
        )
    print("Excluded conditions:", excluded)
