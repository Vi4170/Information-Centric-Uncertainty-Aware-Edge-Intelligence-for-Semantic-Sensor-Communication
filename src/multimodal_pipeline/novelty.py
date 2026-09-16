"""Task 36 -- multisensor novelty integration.

Wires Task 35's frozen, fused multimodal machine-state representations into
the EXISTING canonical novelty mechanism
(``src/novelty/novelty.py::DistanceNoveltyDetector``), unmodified. This
module invents no new novelty formula: it only builds the input the
existing detector already expects (a 2D array of embeddings) from the
fused representation of each of Task 35's four ablation configurations, and
calls the existing ``fit``/``score`` interface exactly as CWRU, Paderborn,
and IMS already do (see ``src/evaluation/voi_behaviour_analysis.py``,
``src/evaluation/paderborn_voi_behaviour_analysis.py``,
``src/evaluation/ims_temporal_analysis.py``).

Architectural constraint (Task 36 requirement 11): novelty is computed on
the FUSED representation (the ``fused_representation`` Dense layer's
64-d output of Task 35's frozen fusion head), never per-modality
separately and then averaged. Each fusion configuration therefore gets its
OWN independent novelty reference -- configurations are never merged.

Only this module and its own artifacts are new. Untouched by this task:
``src/voi/`` (not imported anywhere below), ``src/novelty/`` (imported and
called exactly as-is, no edits), and every Task 31-35 module/artifact
(``src/multimodal_pipeline/{dataset_audit,observation_schema,preprocessing,
representation,fusion}.py`` are only ever called through their existing
public functions, never modified for this task).

Task 36 is novelty only: uncertainty, task relevance, temporal importance,
communication cost, and the five-factor VoI engine are explicitly not
computed here.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import keras
import numpy as np
import pandas as pd

from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep
from src.novelty.novelty import DistanceNoveltyDetector

# Normal is label 0 in the multimodal fault_type label scheme (Task 34),
# matching src/novelty/config.py's own REFERENCE_CLASS default -- reused,
# not redefined, so the existing detector's "Normal-only reference" meaning
# stays identical across datasets in this repo.
NOVELTY_REFERENCE_CLASS: int = rep.FAULT_TYPE_TO_LABEL_ID["Normal"]

NOVELTY_CONFIG_JSON_PATH: str = os.path.join(schema.PROCESSED_DATA_DIR, "multimodal_novelty_config.json")
NOVELTY_RESULTS_CSV_PATH: str = os.path.join(schema.PROCESSED_DATA_DIR, "multimodal_novelty_results.csv")

_RESULT_COLUMNS: Tuple[str, ...] = (
    "observation_id",
    "condition_code",
    "fault_type",
    "split",
    "fusion_config",
    "has_vibration",
    "has_current",
    "has_temperature",
    "novelty_score",
)


class NoNormalTrainingSupportError(ValueError):
    """Raised when a fusion configuration has zero Normal-class training
    observations to build a normal-only reference from. Never silently
    worked around by manufacturing Normal observations or falling back to
    a different split -- see Task 36 requirement 6."""


# ---------------------------------------------------------------------------
# Leakage verification (explicit, on top of the structural guarantee that
# DistanceNoveltyDetector.fit() is only ever called with train-split arrays
# below)
# ---------------------------------------------------------------------------

def verify_no_observation_id_leakage(
    meta_train: pd.DataFrame,
    meta_val: pd.DataFrame,
    meta_test: pd.DataFrame,
) -> None:
    """Raise if any observation_id appears in more than one split's
    per-configuration metadata table."""
    ids_train = set(meta_train["observation_id"]) if len(meta_train) else set()
    ids_val = set(meta_val["observation_id"]) if len(meta_val) else set()
    ids_test = set(meta_test["observation_id"]) if len(meta_test) else set()

    overlaps = {
        "train_val_overlap": len(ids_train & ids_val),
        "train_test_overlap": len(ids_train & ids_test),
        "val_test_overlap": len(ids_val & ids_test),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Observation id split leakage detected: {overlaps}")


def verify_no_condition_split_leakage(combined_meta: pd.DataFrame) -> None:
    """Raise if any condition_code appears under more than one split value
    -- reuses Task 32's own condition-level split-integrity definition."""
    if combined_meta.empty:
        return
    schema.verify_no_condition_crosses_split(combined_meta)


# ---------------------------------------------------------------------------
# Fused representation extraction (Task 35's frozen fusion head, inference
# only -- never retrained, never fitted here)
# ---------------------------------------------------------------------------

def get_fused_representation_for_split(
    config_name: str,
    table: pd.DataFrame,
    fusion_model: keras.Model,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (X_fused, y, meta) for one fusion configuration and one
    already-split-specific Task 35 representation table. X_fused is the
    64-d ``fused_representation`` layer output of the FROZEN Task 35 fusion
    head -- never the raw concatenated per-modality embedding, and never a
    per-modality score averaged after the fact. Rows this configuration's
    own required modalities are missing/invalid for (per
    ``fusion.assemble_fusion_split_arrays``) are excluded, not fabricated."""
    X_concat, y, meta = fu.assemble_fusion_split_arrays(config_name, table)
    if len(X_concat) == 0:
        return np.empty((0, fu.FUSED_DIM), dtype=np.float32), y, meta
    X_fused = fu.get_fused_representation(fusion_model, X_concat)
    return X_fused, y, meta


# ---------------------------------------------------------------------------
# Per-configuration novelty (fresh reference, train-only fitting)
# ---------------------------------------------------------------------------

def _score_summary(scores: np.ndarray) -> Dict[str, object]:
    if len(scores) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
    }


def run_novelty_for_config(
    config_name: str,
    train_table: pd.DataFrame,
    val_table: pd.DataFrame,
    test_table: pd.DataFrame,
    fusion_model: keras.Model,
) -> Tuple[Dict[str, object], List[dict]]:
    """Fits a FRESH DistanceNoveltyDetector for ONE fusion configuration,
    strictly on that configuration's own train-split fused representations,
    then scores train/val/test. Returns (reference_info, per_observation_rows).

    Never reuses a CWRU/Paderborn/IMS novelty reference, and never merges
    configurations into one reference -- each call is fully independent.
    """
    X_train, y_train, meta_train = get_fused_representation_for_split(config_name, train_table, fusion_model)
    X_val, y_val, meta_val = get_fused_representation_for_split(config_name, val_table, fusion_model)
    X_test, y_test, meta_test = get_fused_representation_for_split(config_name, test_table, fusion_model)

    verify_no_observation_id_leakage(meta_train, meta_val, meta_test)
    combined_meta = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
    verify_no_condition_split_leakage(combined_meta)

    n_normal_train = int(np.sum(y_train == NOVELTY_REFERENCE_CLASS)) if len(y_train) else 0
    if n_normal_train == 0:
        raise NoNormalTrainingSupportError(
            f"Configuration '{config_name}' has zero Normal-class (label "
            f"{NOVELTY_REFERENCE_CLASS}) training observations after "
            "modality-validity filtering -- refusing to fabricate a "
            "normal-only reference or fall back to a different split."
        )

    # Fresh reference: a new DistanceNoveltyDetector instance per
    # configuration, fitted ONLY on this configuration's own train-split
    # fused embeddings -- never CWRU's/Paderborn's/another configuration's
    # reference, and never anything from val/test.
    detector = DistanceNoveltyDetector(reference_class=NOVELTY_REFERENCE_CLASS, embedding_dim=fu.FUSED_DIM)
    detector.fit(X_train, y_train)

    scores_train = detector.score(X_train) if len(X_train) else np.empty((0,), dtype=np.float32)
    scores_val = detector.score(X_val) if len(X_val) else np.empty((0,), dtype=np.float32)
    scores_test = detector.score(X_test) if len(X_test) else np.empty((0,), dtype=np.float32)

    per_observation_rows: List[dict] = []
    for meta, scores in ((meta_train, scores_train), (meta_val, scores_val), (meta_test, scores_test)):
        for i in range(len(meta)):
            row = meta.iloc[i]
            per_observation_rows.append(
                {
                    "observation_id": row["observation_id"],
                    "condition_code": row["condition_code"],
                    "fault_type": row["fault_type"],
                    "split": row["split"],
                    "fusion_config": config_name,
                    "has_vibration": bool(row["has_vibration"]),
                    "has_current": bool(row["has_current"]),
                    "has_temperature": bool(row["has_temperature"]),
                    "novelty_score": float(scores[i]),
                }
            )

    reference_info: Dict[str, object] = {
        "config": config_name,
        "modalities": list(fu.FUSION_CONFIGS[config_name]),
        "fused_dim": fu.FUSED_DIM,
        "reference_class": NOVELTY_REFERENCE_CLASS,
        "reference_class_name": rep.FAULT_TYPE_LABELS[NOVELTY_REFERENCE_CLASS],
        "n_train_total": int(len(X_train)),
        "n_train_normal_reference": n_normal_train,
        "n_val_total": int(len(X_val)),
        "n_test_total": int(len(X_test)),
        "reference_centroid_dim": int(detector.reference_centroid.shape[0]),
        "reference_centroid": detector.reference_centroid.tolist(),
        "training_distance_bounds": {"d_min": detector.d_min, "d_max": detector.d_max},
        "fit_only_on_split": "train",
        "novelty_summary_by_split": {
            "train": _score_summary(scores_train),
            "val": _score_summary(scores_val),
            "test": _score_summary(scores_test),
        },
    }
    return reference_info, per_observation_rows


# ---------------------------------------------------------------------------
# Full experiment driver (all four configurations, independent references)
# ---------------------------------------------------------------------------

def run_multimodal_novelty_experiment(
    raw_dir: str = schema.RAW_DATA_DIR,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Builds the shared per-split representation tables once (reusing Task
    35's own frozen vibration/motor_current encoders and normalization),
    then runs an independent, fresh novelty reference + scoring pass for
    each of the four fusion configurations. Returns (experiment_record,
    results_df)."""
    vibration_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
    motor_current_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
    normalization_params = pp.load_normalization_params()

    train_table, excluded_train = fu.build_split_representation_table(
        "train", vibration_model, motor_current_model, normalization_params, raw_dir
    )
    val_table, excluded_val = fu.build_split_representation_table(
        "val", vibration_model, motor_current_model, normalization_params, raw_dir
    )
    test_table, excluded_test = fu.build_split_representation_table(
        "test", vibration_model, motor_current_model, normalization_params, raw_dir
    )
    excluded_conditions = excluded_train + excluded_val + excluded_test

    per_config_reference: Dict[str, object] = {}
    all_rows: List[dict] = []
    for config_name in fu.FUSION_CONFIGS:
        fusion_model = fu.load_fusion_head(config_name)
        reference_info, rows = run_novelty_for_config(config_name, train_table, val_table, test_table, fusion_model)
        per_config_reference[config_name] = reference_info
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows, columns=list(_RESULT_COLUMNS))

    experiment_record: Dict[str, object] = {
        "task": 36,
        "title": "Multisensor novelty integration",
        "novelty_mechanism": "src.novelty.novelty.DistanceNoveltyDetector (unmodified, canonical)",
        "fused_representation_source": "Task 35 frozen fusion heads' fused_representation layer output",
        "configurations": per_config_reference,
        "excluded_conditions": excluded_conditions,
        "not_computed_in_this_task": [
            "uncertainty", "task_relevance", "temporal_importance", "communication_cost", "voi_score", "decision",
        ],
    }
    return experiment_record, results_df


def save_novelty_config(
    experiment_record: Dict[str, object],
    path: str = NOVELTY_CONFIG_JSON_PATH,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(experiment_record, handle, indent=2, sort_keys=True)


def save_novelty_results(results_df: pd.DataFrame, path: str = NOVELTY_RESULTS_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results_df.to_csv(path, index=False)


if __name__ == "__main__":
    record, df = run_multimodal_novelty_experiment()
    save_novelty_config(record)
    save_novelty_results(df)

    for config_name, info in record["configurations"].items():
        summary = info["novelty_summary_by_split"]
        print(
            f"{config_name}: n_train={info['n_train_total']} "
            f"(normal_reference={info['n_train_normal_reference']}) "
            f"n_val={info['n_val_total']} n_test={info['n_test_total']} "
            f"train_mean={summary['train']['mean']:.4f} "
            f"val_mean={(summary['val']['mean'] if summary['val']['mean'] is not None else float('nan')):.4f} "
            f"test_mean={(summary['test']['mean'] if summary['test']['mean'] is not None else float('nan')):.4f}"
        )
    print("Excluded conditions:", len(record["excluded_conditions"]))
