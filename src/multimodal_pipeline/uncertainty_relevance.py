"""Task 37 -- multisensor uncertainty and task interpretation (relevance).

======================================================================
Part A -- Uncertainty
======================================================================
Reuses the existing canonical entropy-based uncertainty mechanism
(``src/uncertainty/uncertainty.py::compute_predictive_entropy``) completely
unmodified, applied to each Task 35 FROZEN fusion head's own softmax class
probabilities:

    U = H(p) / log(num_classes),   H(p) = -sum(p_i * log(p_i))

exactly the project's existing formula, just called with
``num_classes=5`` -- the same pattern Paderborn already used
(``compute_predictive_entropy(probabilities, num_classes=NUM_CLASSES)``
with its own class count). Uncertainty is computed on the FUSED model's own
predictive output, never novelty, never max-probability confidence, never
an invented disagreement metric, and no Monte Carlo dropout is used (not
required by the existing project). ``compute_predictive_entropy`` has no
fitting stage at all -- it is a pure, stateless function of a probability
row -- so "train/val/test" below only selects which observations get
scored, never any calibration step; nothing is fit on val/test here because
nothing is fit anywhere for this factor.

======================================================================
Part B -- Relevance / task interpretation
======================================================================
``src/relevance/relevance.py``'s public functions hard-validate against the
module-level ``NUM_CLASSES=4`` default with no override parameter exposed
to callers (``_validate_probabilities``, ``_validate_relevance_map`` --
confirmed by reading the module and ``tests/test_relevance.py::test_16``,
which asserts this 4-class-only validation is the intended, tested
contract). A 5-class probability vector is therefore structurally rejected
by the protected module's own public API -- the exact same interface
mismatch already documented and worked around for Paderborn in
``src/evaluation/paderborn_voi_behaviour_analysis.py``. Following that same
precedent: the protected module is left completely untouched, and the
*existing* "class_mapping" strategy it already defines
(``relevance_from_class``: ``relevance_map[predicted_class]``) is
reimplemented locally against a 5-class map -- no new relevance formula is
invented.

That map can only be filled honestly for 3 of the 5 multimodal classes:

    0 Normal        -> 0.10  (reused verbatim from CWRU's CLASS_RELEVANCE_MAP[0])
    1 BPFI           -> 1.00  (ball-pass-frequency INNER-race defect; same
                                physical fault category as CWRU's "Inner Race
                                Fault" -- reused verbatim from CLASS_RELEVANCE_MAP[1])
    2 BPFO           -> 0.90  (ball-pass-frequency OUTER-race defect; same
                                physical fault category as CWRU's "Outer Race
                                Fault" -- reused verbatim from CLASS_RELEVANCE_MAP[3].
                                CWRU's "Ball Fault", index 2, coincidentally
                                shares the same 0.90 value but is NOT the class
                                being reused here -- BPFO is an outer-race
                                defect, not a ball defect.)
    3 Misalignment   -> undefined (None / NaN)
    4 Unbalance      -> undefined (None / NaN)

Misalignment (rotor/shaft misalignment) and Unbalance (rotor mass
imbalance) are genuinely new failure modes with no analogue in CWRU's or
Paderborn's class schemes, and no established relevance value exists
anywhere in this repository for either. Rather than invent one, this
module leaves them explicitly undefined: ``relevance_score`` is NaN for any
observation whose predicted class is Misalignment or Unbalance. This is a
disclosed, permanent scope gap (not a bug), reported in every artifact this
module produces -- never silently interpreted as zero or averaged away.

Relevance is computed on the model's ARGMAX predicted class (the
"class_mapping" strategy), not the probability-weighted strategy CWRU/
Paderborn use elsewhere: the probability-weighted formula
(``R = sum(P(class_i) * relevance(class_i))``) requires a finite value for
every class it sums over, which is impossible here without fabricating
values for Misalignment/Unbalance. Class-mapping is itself an existing,
already-tested strategy in the protected module (not a new invention),
and avoids fabricating a blended number that would misrepresent how much
of the probability mass is actually unpriced.

======================================================================
Scientific safeguards
======================================================================
Novelty ("how different from the learned normal reference"), uncertainty
("how uncertain is the fused predictive model"), and relevance ("how
important is the predicted condition for the task") are kept strictly
separate outputs here -- none is computed from or substituted for another.
High uncertainty is not interpreted as high information value; high
relevance is not interpreted as high novelty; the CNN's own accuracy is
never treated as evidence that its uncertainty estimate is calibrated.
VoI and communication decisions are not computed in this task.

Neither ``src/uncertainty/``, ``src/relevance/``, ``src/novelty/``, nor
``src/voi/`` is imported for modification anywhere in this module --
``src/uncertainty/`` is imported and called exactly as-is; the others are
not imported at all.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import pandas as pd

from src.cnn.model import predict_probabilities
from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep
from src.multimodal_pipeline.novelty import (
    verify_no_condition_split_leakage,
    verify_no_observation_id_leakage,
)
from src.relevance.config import CLASS_RELEVANCE_MAP as CWRU_CLASS_RELEVANCE_MAP
from src.uncertainty.uncertainty import compute_predictive_entropy

# ---------------------------------------------------------------------------
# Part A constants -- verified, not assumed, against Task 34's actual label
# scheme (src/multimodal_pipeline/representation.py)
# ---------------------------------------------------------------------------

EXPECTED_FAULT_TYPE_LABELS: Tuple[str, ...] = ("Normal", "BPFI", "BPFO", "Misalignment", "Unbalance")
if tuple(rep.FAULT_TYPE_LABELS) != EXPECTED_FAULT_TYPE_LABELS:
    raise AssertionError(
        f"src.multimodal_pipeline.representation.FAULT_TYPE_LABELS changed to "
        f"{rep.FAULT_TYPE_LABELS!r}; Task 37's hard-coded relevance mapping below "
        f"assumes {EXPECTED_FAULT_TYPE_LABELS!r} and must be re-derived, not silently reused."
    )

NUM_CLASSES: int = rep.NUM_CLASSES  # 5, verified above


# ---------------------------------------------------------------------------
# Part B: local 5-class relevance map (see module docstring for provenance
# and the disclosed Misalignment/Unbalance gap). NEVER passed into
# src/relevance/relevance.py's own validators -- reimplemented locally
# because that module's public API is hard-coded to 4 classes.
# ---------------------------------------------------------------------------

MULTIMODAL_RELEVANCE_MAP: Dict[int, Optional[float]] = {
    0: CWRU_CLASS_RELEVANCE_MAP[0],  # Normal
    1: CWRU_CLASS_RELEVANCE_MAP[1],  # BPFI <- CWRU "Inner Race Fault"
    2: CWRU_CLASS_RELEVANCE_MAP[3],  # BPFO <- CWRU "Outer Race Fault" (index 3, not 2)
    3: None,  # Misalignment -- undefined, see module docstring
    4: None,  # Unbalance -- undefined, see module docstring
}
UNDEFINED_RELEVANCE_CLASSES: Tuple[str, ...] = tuple(
    EXPECTED_FAULT_TYPE_LABELS[i] for i, v in MULTIMODAL_RELEVANCE_MAP.items() if v is None
)

UNCERTAINTY_RELEVANCE_CONFIG_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_uncertainty_relevance_config.json"
)
UNCERTAINTY_RELEVANCE_RESULTS_CSV_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_uncertainty_relevance_results.csv"
)

_RESULT_COLUMNS: Tuple[str, ...] = (
    "observation_id",
    "condition_code",
    "fault_type",
    "split",
    "fusion_config",
    "has_vibration",
    "has_current",
    "has_temperature",
    "predicted_class",
    "predicted_fault_type",
    "uncertainty_score",
    "relevance_score",
)


def relevance_from_predicted_class(predicted_class: int) -> Optional[float]:
    """The local 5-class 'class_mapping' relevance lookup -- returns None
    (never a fabricated number) for Misalignment/Unbalance."""
    return MULTIMODAL_RELEVANCE_MAP[int(predicted_class)]


def compute_relevance_batch(probabilities: np.ndarray) -> np.ndarray:
    """Deterministic relevance score per observation from the model's own
    argmax predicted class. NaN (never 0.0, never an average) wherever the
    predicted class has no established relevance value."""
    predicted = np.argmax(probabilities, axis=1)
    return np.array(
        [
            (relevance_from_predicted_class(c) if relevance_from_predicted_class(c) is not None else np.nan)
            for c in predicted
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Fused model predictions (Task 35's frozen fusion head, inference only)
# ---------------------------------------------------------------------------

def get_predictions_for_split(
    config_name: str,
    table: pd.DataFrame,
    fusion_model: keras.Model,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (probabilities, y, meta): the FUSED model's own softmax class
    probabilities for every observation this configuration's own modality
    validity rules retain -- never fabricated, never averaged across
    per-modality scores."""
    X_concat, y, meta = fu.assemble_fusion_split_arrays(config_name, table)
    if len(X_concat) == 0:
        return np.empty((0, NUM_CLASSES), dtype=np.float32), y, meta
    probabilities = predict_probabilities(fusion_model, X_concat)
    return probabilities, y, meta


def _class_support(meta: pd.DataFrame) -> Dict[str, int]:
    counts = {name: 0 for name in EXPECTED_FAULT_TYPE_LABELS}
    if len(meta):
        counts.update(meta["fault_type"].value_counts().to_dict())
    return {k: int(v) for k, v in counts.items()}


def _uncertainty_summary(scores: np.ndarray) -> Dict[str, object]:
    if len(scores) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _relevance_summary(scores: np.ndarray) -> Dict[str, object]:
    if len(scores) == 0:
        return {"n": 0, "n_defined": 0, "n_undefined": 0, "mean_defined": None}
    finite_mask = np.isfinite(scores)
    n_defined = int(np.sum(finite_mask))
    return {
        "n": int(len(scores)),
        "n_defined": n_defined,
        "n_undefined": int(len(scores) - n_defined),
        "mean_defined": float(np.mean(scores[finite_mask])) if n_defined > 0 else None,
    }


# ---------------------------------------------------------------------------
# Per-configuration uncertainty + relevance (frozen fusion model, no fitting)
# ---------------------------------------------------------------------------

def run_uncertainty_relevance_for_config(
    config_name: str,
    train_table: pd.DataFrame,
    val_table: pd.DataFrame,
    test_table: pd.DataFrame,
    fusion_model: keras.Model,
) -> Tuple[Dict[str, object], List[dict]]:
    """Scores train/val/test with the FROZEN Task 35 fusion head for one
    configuration. Nothing is fit here -- compute_predictive_entropy is a
    stateless formula and MULTIMODAL_RELEVANCE_MAP is a fixed constant, so
    there is no calibration step that could leak val/test into a fitted
    parameter; the split loop below only selects which observations to
    score."""
    probs_train, y_train, meta_train = get_predictions_for_split(config_name, train_table, fusion_model)
    probs_val, y_val, meta_val = get_predictions_for_split(config_name, val_table, fusion_model)
    probs_test, y_test, meta_test = get_predictions_for_split(config_name, test_table, fusion_model)

    verify_no_observation_id_leakage(meta_train, meta_val, meta_test)
    combined_meta = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
    verify_no_condition_split_leakage(combined_meta)

    per_observation_rows: List[dict] = []
    per_split_info: Dict[str, Dict[str, object]] = {}
    for split_name, probs, meta in (("train", probs_train, meta_train), ("val", probs_val, meta_val), ("test", probs_test, meta_test)):
        if len(probs):
            for p in probs:
                if p.shape[0] != NUM_CLASSES:
                    raise AssertionError(f"Expected {NUM_CLASSES}-d probabilities, got shape {p.shape}")
            if not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-2):
                raise AssertionError("Fused model probabilities do not sum to ~1.0 per row")
            uncertainty = compute_predictive_entropy(probs, num_classes=NUM_CLASSES)
            relevance = compute_relevance_batch(probs)
            predicted = np.argmax(probs, axis=1)
        else:
            uncertainty = np.empty((0,), dtype=np.float32)
            relevance = np.empty((0,), dtype=np.float32)
            predicted = np.empty((0,), dtype=np.int64)

        for i in range(len(meta)):
            row = meta.iloc[i]
            pred_class = int(predicted[i])
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
                    "predicted_class": pred_class,
                    "predicted_fault_type": EXPECTED_FAULT_TYPE_LABELS[pred_class],
                    "uncertainty_score": float(uncertainty[i]),
                    "relevance_score": float(relevance[i]),
                }
            )

        per_split_info[split_name] = {
            "n_observations": int(len(meta)),
            "class_support": _class_support(meta),
            "uncertainty_summary": _uncertainty_summary(uncertainty),
            "relevance_summary": _relevance_summary(relevance),
        }

    reference_info: Dict[str, object] = {
        "config": config_name,
        "modalities": list(fu.FUSION_CONFIGS[config_name]),
        "num_classes": NUM_CLASSES,
        "probability_dim": NUM_CLASSES,
        "fault_type_labels": list(EXPECTED_FAULT_TYPE_LABELS),
        "model_path": fu.fusion_model_path(config_name),
        "model_version": fu.MODEL_VERSION,
        "representation_source": "Task 35 frozen fusion head (fused representation -> softmax classifier), inference only, not retrained",
        "uncertainty_method": "src.uncertainty.uncertainty.compute_predictive_entropy (unmodified, canonical), normalized Shannon entropy over the fused model's own softmax output",
        "uncertainty_fitting_stage": "none -- compute_predictive_entropy is a stateless function of probabilities; nothing is fit on any split",
        "relevance_method": "local 5-class 'class_mapping' lookup on argmax predicted class (src/relevance/relevance.py's own class_mapping strategy, reimplemented locally because its public API hard-validates 4 classes with no override)",
        "relevance_map": {EXPECTED_FAULT_TYPE_LABELS[k]: v for k, v in MULTIMODAL_RELEVANCE_MAP.items()},
        "relevance_undefined_classes": list(UNDEFINED_RELEVANCE_CLASSES),
        "splits": per_split_info,
    }
    return reference_info, per_observation_rows


# ---------------------------------------------------------------------------
# Full experiment driver (all four configurations, independent scoring)
# ---------------------------------------------------------------------------

def run_multimodal_uncertainty_relevance_experiment(
    raw_dir: str = schema.RAW_DATA_DIR,
) -> Tuple[Dict[str, object], pd.DataFrame]:
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

    per_config_info: Dict[str, object] = {}
    all_rows: List[dict] = []
    for config_name in fu.FUSION_CONFIGS:
        fusion_model = fu.load_fusion_head(config_name)
        info, rows = run_uncertainty_relevance_for_config(config_name, train_table, val_table, test_table, fusion_model)
        per_config_info[config_name] = info
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows, columns=list(_RESULT_COLUMNS))

    experiment_record: Dict[str, object] = {
        "task": 37,
        "title": "Multisensor uncertainty and task interpretation",
        "uncertainty_source": "src.uncertainty.uncertainty.compute_predictive_entropy (unmodified, canonical)",
        "relevance_source": "local 5-class reimplementation of src.relevance.relevance's class_mapping strategy (protected module unmodified)",
        "configurations": per_config_info,
        "excluded_conditions": excluded_conditions,
        "not_computed_in_this_task": ["voi_score", "communication_cost", "decision", "temporal_importance"],
    }
    return experiment_record, results_df


def save_uncertainty_relevance_config(
    experiment_record: Dict[str, object],
    path: str = UNCERTAINTY_RELEVANCE_CONFIG_JSON_PATH,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(experiment_record, handle, indent=2, sort_keys=True)


def save_uncertainty_relevance_results(results_df: pd.DataFrame, path: str = UNCERTAINTY_RELEVANCE_RESULTS_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results_df.to_csv(path, index=False)


if __name__ == "__main__":
    record, df = run_multimodal_uncertainty_relevance_experiment()
    save_uncertainty_relevance_config(record)
    save_uncertainty_relevance_results(df)

    for config_name, info in record["configurations"].items():
        print(f"=== {config_name} ===")
        for split_name, split_info in info["splits"].items():
            u = split_info["uncertainty_summary"]
            r = split_info["relevance_summary"]
            print(
                f"  {split_name}: n={split_info['n_observations']} "
                f"class_support={split_info['class_support']} "
                f"uncertainty_mean={u['mean']} "
                f"relevance_defined={r['n_defined']}/{r['n']}"
            )
    print("Excluded conditions:", len(record["excluded_conditions"]))
