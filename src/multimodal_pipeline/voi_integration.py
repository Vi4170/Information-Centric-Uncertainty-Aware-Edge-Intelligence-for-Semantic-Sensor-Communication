"""Task 39 -- complete multimodal VoI integration.

Wires the five independently computed multimodal factors into the EXISTING
canonical VoI Engine, ``src.voi.voi_engine.VoIEngine`` (unmodified),
calling its ``compute_batch`` method exactly as CWRU
(``src/evaluation/voi_behaviour_analysis.py``) and Paderborn
(``src/evaluation/paderborn_voi_behaviour_analysis.py``) already do -- the
established precedent in this repository, not
``src/integration/voi_pipeline.py`` (a second, unused-by-any-dataset
integration layer built in Task 10). No second VoI formula is implemented
here, and ``src/voi/`` is imported only through its existing public API --
never for editing.

======================================================================
Data availability (checked directly, not assumed from a prior session)
======================================================================
This module was implemented in a session where the raw multimodal data
(``data/raw/multimodal/{vibration,temperature_current}``, ~7.6GB, gitignored
and local-only per Task 32) and all required trained models (Task 34
modality encoders, Task 35/37B fusion heads) were verified present on disk.
``multimodal_raw_data_and_models_available()`` performs that same check at
import/run time -- if it ever returns False (e.g. a different machine or a
fresh checkout without the local raw archive), ``__main__`` below refuses to
run and reports the experiment as BLOCKED rather than fabricating a "real"
result. The five factors below are therefore computed from a genuine,
end-to-end run over the real dataset, not synthetic placeholders.

======================================================================
The five factors -- provenance of each
======================================================================
Novelty:
    A FRESH ``src.novelty.novelty.DistanceNoveltyDetector`` per fusion
    configuration, fit strictly on that configuration's own train-split
    fused (64-d) representation, Normal-only reference -- identical
    mechanism and fitting discipline to Task 36
    (``src/multimodal_pipeline/novelty.py``). Recomputed here (not read from
    Task 36's saved CSV) because Task 36 persisted only the scalar
    ``novelty_score``, not the underlying fused embedding this task also
    needs for temporal importance -- recomputing from the frozen, unmodified
    fusion heads is inference only, not a second experiment.

Uncertainty:
    ``src.uncertainty.uncertainty.compute_predictive_entropy`` (unmodified),
    applied to each frozen Task 35/37B fusion head's own softmax output,
    ``num_classes=5`` -- identical to Task 37
    (``src/multimodal_pipeline/uncertainty_relevance.py``).

Task Relevance:
    ``uncertainty_relevance.compute_relevance_batch`` (Task 37's local
    5-class "class_mapping" reimplementation, re-audited and finalized as
    PARTIAL by Task 38, ``src/multimodal_pipeline/relevance.py``), reused
    unchanged. Observations whose predicted class is Misalignment or
    Unbalance receive ``task_relevance = NaN`` -- per Task 38's own
    documented downstream-handling plan, these are NEVER passed into the
    canonical VoI engine (which structurally rejects NaN/out-of-range
    inputs -- see ``src.voi.normalization.validate_numeric``). Instead they
    get ``voi_score = NaN`` and ``decision = "UNDEFINED_RELEVANCE"``, counted
    and reported separately, never silently coerced to 0 or to a real
    decision.

Temporal Importance:
    ``src.temporal.temporal.compute_temporal_importance`` (unmodified),
    applied per ``condition_code`` (conditions never cross splits -- Task
    32's guarantee), ordered by ``window_index`` (joined from Task 32's own
    ``multimodal_observation_index.csv``, not re-parsed from the
    observation_id string). The per-observation vector fed to this formula
    is the SAME fused 64-d ``fused_representation`` Task 36 already
    established as this track's canonical feature space for downstream
    scoring -- deliberately reused rather than introducing a second,
    inconsistent representation (e.g. raw concatenated per-modality
    embeddings) that no other multimodal factor uses.
    ``DEFAULT_TEMPORAL_CHANGE_SCALE`` (1.8) is reused UNCHANGED from its
    CWRU-training-data calibration, exactly as Paderborn already tested this
    parameter's transfer as-is -- not re-derived for this dataset. If it
    saturates or collapses here, that is reported as a finding (see the
    Task 41 behaviour analysis), not silently recalibrated.

Communication Cost:
    ``src.communication.cost.compute_communication_cost`` (unmodified) with
    the SAME nominal constant inputs CWRU and Paderborn already use
    (``MAX_PAYLOAD_SIZE`` bytes at ``REFERENCE_BANDWIDTH``, full available
    bandwidth) -- a fixed baseline cost, identical across all four fusion
    configurations. This is a deliberate choice to avoid inventing a new,
    unauthorized payload-size model for multi-sensor transmission (e.g.
    scaling bytes by concat_dim or channel count): no such model exists
    anywhere in this repository, and Task 40's own instructions explicitly
    say not to introduce FSO/channel modeling yet. Documented here as reused
    precedent, not a new physical model.

======================================================================
Protected modules
======================================================================
``src/voi/``, ``src/novelty/``, ``src/uncertainty/``, ``src/relevance/`` are
imported only through their existing public, unmodified APIs (or, for
``src/voi/``, through the existing ``src/integration/voi_pipeline.py`` thin
wrapper) -- none is edited by this module. ``src/multimodal_pipeline/{fusion,
observation_schema,preprocessing,representation,novelty,
uncertainty_relevance,relevance}.py`` are called through their existing
public functions/constants only.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import keras
import numpy as np
import pandas as pd

from src.cnn.model import predict_probabilities
from src.communication.config import MAX_PAYLOAD_SIZE, REFERENCE_BANDWIDTH
from src.communication.cost import compute_communication_cost
from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep
from src.multimodal_pipeline.novelty import (
    NOVELTY_REFERENCE_CLASS,
    NoNormalTrainingSupportError,
    verify_no_condition_split_leakage,
    verify_no_observation_id_leakage,
)
from src.multimodal_pipeline.uncertainty_relevance import (
    EXPECTED_FAULT_TYPE_LABELS,
    NUM_CLASSES,
    compute_relevance_batch,
)
from src.novelty.novelty import DistanceNoveltyDetector
from src.temporal.temporal import compute_temporal_importance
from src.uncertainty.uncertainty import compute_predictive_entropy
from src.voi.voi_engine import VoIEngine

UNDEFINED_RELEVANCE_DECISION: str = "UNDEFINED_RELEVANCE"

NOMINAL_PAYLOAD_BYTES: float = MAX_PAYLOAD_SIZE
NOMINAL_BANDWIDTH: float = REFERENCE_BANDWIDTH

VOI_INTEGRATION_CONFIG_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_voi_integration_config.json"
)
VOI_INTEGRATION_RESULTS_CSV_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_voi_integration_results.csv"
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
    "novelty",
    "uncertainty",
    "task_relevance",
    "temporal_importance",
    "resource_cost",
    "relevance_defined",
    "raw_voi_score",
    "voi_score",
    "decision",
)


# ---------------------------------------------------------------------------
# Data/model availability guard (Task 39 requirement: stop, don't fabricate)
# ---------------------------------------------------------------------------


def multimodal_raw_data_and_models_available(raw_dir: str = schema.RAW_DATA_DIR) -> bool:
    """True iff raw data AND every trained model this experiment needs exist.

    Used to decide, at run time, whether the REAL experiment can execute --
    never assumed true from a prior session's claim.
    """
    return (
        schema.raw_data_available(raw_dir)
        and os.path.exists(rep.VIBRATION_MODEL_PATH)
        and os.path.exists(rep.MOTOR_CURRENT_MODEL_PATH)
        and all(os.path.exists(fu.fusion_model_path(name)) for name in fu.FUSION_CONFIGS)
    )


# ---------------------------------------------------------------------------
# Communication cost (nominal constant, identical to CWRU/Paderborn)
# ---------------------------------------------------------------------------


def nominal_communication_cost() -> float:
    """Fixed baseline Communication Cost, identical across all configurations
    and identical to the constant CWRU/Paderborn already use. See module
    docstring for why no per-configuration payload-size model is invented."""
    return compute_communication_cost(
        payload_size=NOMINAL_PAYLOAD_BYTES,
        transmission_time=NOMINAL_PAYLOAD_BYTES / NOMINAL_BANDWIDTH,
        available_bandwidth=NOMINAL_BANDWIDTH,
    )


# ---------------------------------------------------------------------------
# Temporal importance (per-condition sequence, fused-representation feature
# space, window_index-ordered via Task 32's own observation index)
# ---------------------------------------------------------------------------


def compute_temporal_batch(
    X_fused: np.ndarray,
    meta: pd.DataFrame,
    observation_index: pd.DataFrame,
) -> np.ndarray:
    """Per-observation Temporal Importance, ordered by window_index within
    each condition_code (conditions never cross splits -- Task 32's
    guarantee, so an entirely per-split grouping is safe)."""
    n = len(meta)
    if n == 0:
        return np.empty((0,), dtype=np.float32)

    window_lookup = observation_index.set_index("observation_id")["window_index"]
    meta = meta.reset_index(drop=True)
    window_idx = meta["observation_id"].map(window_lookup)
    if window_idx.isna().any():
        missing = meta.loc[window_idx.isna(), "observation_id"].tolist()
        raise AssertionError(
            f"observation_id(s) not found in the Task 32 observation index: {missing[:5]}"
        )

    scores = np.zeros(n, dtype=np.float64)
    for _, group in meta.groupby("condition_code"):
        idx = group.index.to_numpy()
        order = np.argsort(window_idx.loc[idx].to_numpy())
        ordered_idx = idx[order]
        scores[ordered_idx] = compute_temporal_importance(X_fused[ordered_idx])
    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-split factor computation (frozen fusion head, inference only)
# ---------------------------------------------------------------------------


def _get_split_arrays(config_name: str, table: pd.DataFrame, fusion_model: keras.Model) -> Dict[str, object]:
    X_concat, y, meta = fu.assemble_fusion_split_arrays(config_name, table)
    if len(X_concat):
        X_fused = fu.get_fused_representation(fusion_model, X_concat)
        probabilities = predict_probabilities(fusion_model, X_concat)
        if probabilities.shape[1] != NUM_CLASSES:
            raise AssertionError(f"Expected {NUM_CLASSES}-d probabilities, got shape {probabilities.shape}")
        if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-2):
            raise AssertionError("Fused model probabilities do not sum to ~1.0 per row")
    else:
        X_fused = np.empty((0, fu.FUSED_DIM), dtype=np.float32)
        probabilities = np.empty((0, NUM_CLASSES), dtype=np.float32)
    return {"X_fused": X_fused, "probabilities": probabilities, "y": y, "meta": meta}


def _decision_counts(decisions: List[str]) -> Dict[str, int]:
    counts = {"DISCARD": 0, "BUFFER": 0, "SUMMARY": 0, "TRANSMIT": 0, UNDEFINED_RELEVANCE_DECISION: 0}
    for d in decisions:
        counts[d] = counts.get(d, 0) + 1
    return counts


def _availability(values: np.ndarray) -> Dict[str, object]:
    n = len(values)
    if n == 0:
        return {"n": 0, "n_available": 0, "pct_available": 0.0}
    n_available = int(np.sum(np.isfinite(values)))
    return {"n": n, "n_available": n_available, "pct_available": round(100.0 * n_available / n, 4)}


def _class_support(meta: pd.DataFrame) -> Dict[str, int]:
    counts = {name: 0 for name in EXPECTED_FAULT_TYPE_LABELS}
    if len(meta):
        counts.update(meta["fault_type"].value_counts().to_dict())
    return {k: int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Per-configuration VoI integration (fresh novelty reference, train-only fit)
# ---------------------------------------------------------------------------


def run_voi_integration_for_config(
    config_name: str,
    train_table: pd.DataFrame,
    val_table: pd.DataFrame,
    test_table: pd.DataFrame,
    fusion_model: keras.Model,
    observation_index: pd.DataFrame,
    engine: VoIEngine,
) -> Tuple[Dict[str, object], List[dict]]:
    split_arrays = {
        "train": _get_split_arrays(config_name, train_table, fusion_model),
        "val": _get_split_arrays(config_name, val_table, fusion_model),
        "test": _get_split_arrays(config_name, test_table, fusion_model),
    }

    verify_no_observation_id_leakage(
        split_arrays["train"]["meta"], split_arrays["val"]["meta"], split_arrays["test"]["meta"]
    )
    combined_meta = pd.concat([v["meta"] for v in split_arrays.values()], ignore_index=True)
    verify_no_condition_split_leakage(combined_meta)

    y_train = split_arrays["train"]["y"]
    n_normal_train = int(np.sum(y_train == NOVELTY_REFERENCE_CLASS)) if len(y_train) else 0
    if n_normal_train == 0:
        raise NoNormalTrainingSupportError(
            f"Configuration '{config_name}' has zero Normal-class (label "
            f"{NOVELTY_REFERENCE_CLASS}) training observations -- refusing "
            "to fabricate a novelty reference or fall back to another split."
        )

    detector = DistanceNoveltyDetector(reference_class=NOVELTY_REFERENCE_CLASS, embedding_dim=fu.FUSED_DIM)
    detector.fit(split_arrays["train"]["X_fused"], y_train)

    resource_cost_value = nominal_communication_cost()

    per_observation_rows: List[dict] = []
    per_split_info: Dict[str, Dict[str, object]] = {}

    for split_name, arrays in split_arrays.items():
        X_fused = arrays["X_fused"]
        probabilities = arrays["probabilities"]
        meta = arrays["meta"]
        n = len(meta)

        if n == 0:
            novelty = uncertainty = relevance = temporal = np.empty((0,), dtype=np.float32)
            predicted = np.empty((0,), dtype=np.int64)
        else:
            novelty = detector.score(X_fused)
            uncertainty = compute_predictive_entropy(probabilities, num_classes=NUM_CLASSES)
            relevance = compute_relevance_batch(probabilities)
            predicted = np.argmax(probabilities, axis=1)
            temporal = compute_temporal_batch(X_fused, meta, observation_index)

        resource_cost = np.full(n, resource_cost_value, dtype=np.float32)
        defined_mask = np.isfinite(relevance)

        records: List[dict] = []
        for i in range(n):
            row = meta.iloc[i]
            pred_class = int(predicted[i])
            rel = float(relevance[i])
            relevance_defined = bool(defined_mask[i])
            records.append(
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
                    "novelty": float(novelty[i]),
                    "uncertainty": float(uncertainty[i]),
                    "task_relevance": rel if relevance_defined else float("nan"),
                    "temporal_importance": float(temporal[i]),
                    "resource_cost": float(resource_cost[i]),
                    "relevance_defined": relevance_defined,
                    # Never fed to the engine when relevance is undefined
                    # (it would raise on NaN); never silently coerced to 0
                    # or to a real decision. Overwritten below only for
                    # relevance-defined rows.
                    "raw_voi_score": float("nan"),
                    "voi_score": float("nan"),
                    "decision": UNDEFINED_RELEVANCE_DECISION,
                }
            )

        defined_indices = np.where(defined_mask)[0]
        if len(defined_indices):
            factors_df = pd.DataFrame(
                {
                    "novelty": novelty[defined_indices].astype(float),
                    "uncertainty": uncertainty[defined_indices].astype(float),
                    "task_relevance": relevance[defined_indices].astype(float),
                    "temporal_importance": temporal[defined_indices].astype(float),
                    "resource_cost": resource_cost[defined_indices].astype(float),
                }
            )
            voi_result_df = engine.compute_batch(factors_df)
            for j, idx in enumerate(defined_indices):
                records[idx]["raw_voi_score"] = float(voi_result_df.iloc[j]["raw_voi_score"])
                records[idx]["voi_score"] = float(voi_result_df.iloc[j]["voi_score"])
                records[idx]["decision"] = str(voi_result_df.iloc[j]["decision"])

        decisions = [r["decision"] for r in records]
        per_observation_rows.extend(records)

        per_split_info[split_name] = {
            "n_observations": n,
            "class_support": _class_support(meta),
            "novelty_availability": _availability(novelty),
            "uncertainty_availability": _availability(uncertainty),
            "temporal_importance_availability": _availability(temporal),
            "communication_cost_availability": _availability(resource_cost),
            "relevance_coverage": {
                "n": n,
                "n_defined": int(np.sum(np.isfinite(relevance))) if n else 0,
                "n_undefined": int(n - np.sum(np.isfinite(relevance))) if n else 0,
                "pct_defined": round(100.0 * np.sum(np.isfinite(relevance)) / n, 4) if n else 0.0,
            },
            "voi_availability": {
                "n_computed": int(sum(1 for d in decisions if d != UNDEFINED_RELEVANCE_DECISION)),
                "n_undefined_relevance": int(sum(1 for d in decisions if d == UNDEFINED_RELEVANCE_DECISION)),
            },
            "decision_counts": _decision_counts(decisions),
        }

    reference_info: Dict[str, object] = {
        "config": config_name,
        "modalities": list(fu.FUSION_CONFIGS[config_name]),
        "fused_dim": fu.FUSED_DIM,
        "novelty_reference_class": NOVELTY_REFERENCE_CLASS,
        "novelty_reference_class_name": rep.FAULT_TYPE_LABELS[NOVELTY_REFERENCE_CLASS],
        "novelty_n_train_normal_reference": n_normal_train,
        "resource_cost_nominal_value": resource_cost_value,
        "splits": per_split_info,
    }
    return reference_info, per_observation_rows


# ---------------------------------------------------------------------------
# Full experiment driver (all four configurations, independent references)
# ---------------------------------------------------------------------------


def run_multimodal_voi_integration_experiment(
    raw_dir: str = schema.RAW_DATA_DIR,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    if not multimodal_raw_data_and_models_available(raw_dir):
        raise FileNotFoundError(
            "Task 39 real multimodal VoI experiment is BLOCKED: required raw "
            "data and/or trained models are not available in this "
            "environment. Refusing to fabricate a result -- see "
            "multimodal_raw_data_and_models_available()."
        )

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

    observation_index = pd.read_csv(
        schema.OBSERVATION_INDEX_CSV_PATH, usecols=["observation_id", "window_index"]
    )

    engine = VoIEngine()  # canonical default (Task 14-calibrated) weights and thresholds, unmodified

    per_config_info: Dict[str, object] = {}
    all_rows: List[dict] = []
    for config_name in fu.FUSION_CONFIGS:
        fusion_model = fu.load_fusion_head(config_name)
        info, rows = run_voi_integration_for_config(
            config_name, train_table, val_table, test_table, fusion_model, observation_index, engine
        )
        per_config_info[config_name] = info
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows, columns=list(_RESULT_COLUMNS))

    from dataclasses import asdict

    experiment_record: Dict[str, object] = {
        "task": 39,
        "title": "Complete multimodal VoI integration",
        "status": "REAL_EXPERIMENT_EXECUTED",
        "voi_engine_source": "src.voi.voi_engine.VoIEngine.compute_batch (unmodified, canonical; same usage pattern as CWRU/Paderborn)",
        "weights": asdict(engine.weights),
        "thresholds": asdict(engine.thresholds),
        "novelty_source": "src.novelty.novelty.DistanceNoveltyDetector (unmodified), fresh per-config Normal-only reference, train-only fit",
        "uncertainty_source": "src.uncertainty.uncertainty.compute_predictive_entropy (unmodified)",
        "relevance_source": "src.multimodal_pipeline.uncertainty_relevance.compute_relevance_batch (Task 37/38, unmodified here)",
        "temporal_importance_source": "src.temporal.temporal.compute_temporal_importance (unmodified), on Task 36's fused_representation feature space, ordered by Task 32 window_index within condition_code",
        "communication_cost_source": "src.communication.cost.compute_communication_cost (unmodified), nominal constant identical to CWRU/Paderborn -- no FSO/channel model introduced",
        "undefined_relevance_handling": (
            "Observations with NaN task_relevance (predicted class Misalignment "
            "or Unbalance) are never passed to the VoI engine. They receive "
            "voi_score=NaN and decision='UNDEFINED_RELEVANCE', counted "
            "separately in voi_availability/decision_counts per split."
        ),
        "configurations": per_config_info,
        "excluded_conditions": excluded_conditions,
        "not_computed_in_this_task": ["fso_channel_simulation"],
    }
    return experiment_record, results_df


def save_voi_integration_config(
    experiment_record: Dict[str, object],
    path: str = VOI_INTEGRATION_CONFIG_JSON_PATH,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(experiment_record, handle, indent=2, sort_keys=True, default=str)


def save_voi_integration_results(
    results_df: pd.DataFrame,
    path: str = VOI_INTEGRATION_RESULTS_CSV_PATH,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results_df.to_csv(path, index=False)


if __name__ == "__main__":
    if not multimodal_raw_data_and_models_available():
        print("=== Task 39 BLOCKED ===")
        print(
            "Required raw multimodal data and/or trained models are not "
            "available in this environment. Refusing to fabricate a 'real' "
            "VoI experiment result. Stopping here."
        )
    else:
        print("=== Executing Task 39: Complete Multimodal VoI Integration ===")
        record, df = run_multimodal_voi_integration_experiment()
        save_voi_integration_config(record)
        save_voi_integration_results(df)

        for config_name, info in record["configurations"].items():
            print(f"=== {config_name} ===")
            for split_name, split_info in info["splits"].items():
                dc = split_info["decision_counts"]
                rc = split_info["relevance_coverage"]
                print(
                    f"  {split_name}: n={split_info['n_observations']} "
                    f"relevance_defined={rc['n_defined']}/{rc['n']} "
                    f"decisions={dc}"
                )
        print("Excluded conditions:", len(record["excluded_conditions"]))
        print(f"Saved config to: {VOI_INTEGRATION_CONFIG_JSON_PATH}")
        print(f"Saved results to: {VOI_INTEGRATION_RESULTS_CSV_PATH}")
