"""Task 37A -- relevance mapping and prediction audit.

Read-only audit module. Never trains, never saves, never modifies any
protected module (src/voi/, src/relevance/relevance.py, src/uncertainty/,
src/novelty/), any Task 35 model artifact, or Task 36/37's own saved
artifacts. This module only READS the already-frozen Task 35 fusion heads
and the already-saved Task 37 results to compute two things:

1. Prediction distributions (true vs. predicted fault_type) per fusion
   configuration and split, read directly from Task 37's own persisted
   results CSV -- no new model inference for this part.

2. A reload-consistency check: for each configuration, the frozen fusion
   head is loaded from disk and run (inference only) over its own
   train-split arrays (reconstructed via fusion.py's existing, unmodified
   functions), and the resulting accuracy is compared against that same
   configuration's OWN persisted Task 35 training-time accuracy
   (data/processed/multimodal/multimodal_fusion_training_history.json).
   A large, unexplained gap between "accuracy measured live during
   training" and "accuracy measured now by loading the saved file" can
   only be explained by the saved .keras file not being the model that
   actually achieved that training-time accuracy -- i.e. the persisted
   model ARTIFACT is corrupted, independent of the fusion architecture,
   the motor_current encoder's own quality, or class imbalance.

======================================================================
Task 37A finding (see the module-level report this produces)
======================================================================
This check surfaced a genuine, pre-existing defect: ``vibration_only`` and
``vibration_temperature``'s saved fusion heads reload to ~24-25% train
accuracy despite their own training_history.json entries reporting
~95.8%/96.5% -- a ~70 percentage-point gap, while ``vibration_current`` and
``vibration_current_temperature`` reload within ~0.2 percentage points of
their own reported accuracy. Git archaeology (see the audit report)
confirms the root cause: ``tests/test_multimodal_fusion.py``'s
``test_30_train_all_fusion_heads_never_requests_test_split`` called
``train_all_fusion_heads(epochs=1)`` on a synthetic 3-row registry WITHOUT
redirecting ``FUSION_MODEL_DIR`` at the time ``vibration_only`` was
originally trained (present, unpatched, in the very first Task 35 commit
``bf2c9f6``) and again during this session's own Task 35 corrective-audit
work (before the bug was caught and fixed in commit ``df49d7e``) --
silently overwriting the real production fusion models with a
severely undertrained 1-epoch/3-observation model each time it ran,
uncaught because nothing previously compared reload-time behavior against
the persisted training_history.json. ``vibration_current`` and
``vibration_current_temperature`` were retrained AFTER the fix (commit
``d1df258``, built on top of the already-patched ``df49d7e``) and are
therefore undamaged. This module does not retrain or repair anything --
Task 37A explicitly prohibits modifying Task 35 models; see the follow-up
task this audit recommends.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.cnn.model import predict_probabilities
from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep
from src.multimodal_pipeline import uncertainty_relevance as ur

RELOAD_CONSISTENCY_TOLERANCE: float = 0.05  # 5 percentage points

AUDIT_JSON_PATH: str = os.path.join(schema.PROCESSED_DATA_DIR, "multimodal_task37a_relevance_prediction_audit.json")

_RELEVANCE_MAPPING_CONCLUSION: str = "B_PARTIAL_MAPPING_ONLY"
_RELEVANCE_MAPPING_RATIONALE: str = (
    "Repository audit (src/relevance/, docs/task_relevance_report.md, "
    "src/multimodal_pipeline/dataset_audit.py) and the externally-cited "
    "companion dataset paper (Jung et al. 2023, Data in Brief 48:109049 -- "
    "the only project-referenced external source for this dataset) were "
    "both searched for a task-importance/criticality ranking covering "
    "Misalignment and Unbalance. Neither establishes one: the paper is a "
    "pure data-acquisition methodology descriptor with no stated fault "
    "priority, and this repository's own CWRU CLASS_RELEVANCE_MAP is "
    "explicitly self-documented (docs/task_relevance_report.md section 6) "
    "as 'initial design parameters... NOT optimized, learned, or "
    "calibrated' -- domain reasoning specific to 4 bearing-race-fault "
    "classes that does not conceptually cover rotor/shaft-level failure "
    "modes at all. Normal/BPFI/BPFO are reused verbatim by genuine "
    "physical fault-category correspondence (same defect location "
    "semantics as CWRU's Inner/Outer race faults); Misalignment and "
    "Unbalance have no defensible source anywhere searched. Conclusion: "
    "PARTIAL mapping only -- the three defined values are preserved "
    "unchanged; the two undefined values are NOT invented."
)


def relevance_mapping_conclusion() -> Dict[str, object]:
    return {
        "conclusion": _RELEVANCE_MAPPING_CONCLUSION,
        "rationale": _RELEVANCE_MAPPING_RATIONALE,
        "relevance_map": {ur.EXPECTED_FAULT_TYPE_LABELS[k]: v for k, v in ur.MULTIMODAL_RELEVANCE_MAP.items()},
        "undefined_relevance_classes": list(ur.UNDEFINED_RELEVANCE_CLASSES),
    }


# ---------------------------------------------------------------------------
# Part D.1 -- prediction distributions (read-only, from Task 37's own saved
# results; no new model inference needed for this half)
# ---------------------------------------------------------------------------

def compute_prediction_distribution(results_df: pd.DataFrame, config_name: str, split: str) -> Dict[str, object]:
    """True vs. predicted fault_type counts for one configuration/split,
    computed directly from Task 37's own persisted per-observation results."""
    sub = results_df[(results_df["fusion_config"] == config_name) & (results_df["split"] == split)]
    return {
        "n": int(len(sub)),
        "true_distribution": {k: int(v) for k, v in sub["fault_type"].value_counts().to_dict().items()},
        "predicted_distribution": {k: int(v) for k, v in sub["predicted_fault_type"].value_counts().to_dict().items()},
    }


def compute_all_prediction_distributions(results_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    return {
        config_name: {split: compute_prediction_distribution(results_df, config_name, split) for split in ("train", "val", "test")}
        for config_name in fu.FUSION_CONFIGS
    }


# ---------------------------------------------------------------------------
# Part D.2 -- reload-consistency check (read-only: loads the frozen Task 35
# fusion head for inference ONLY, never trains or re-saves it)
# ---------------------------------------------------------------------------

def check_reload_consistency(
    config_name: str,
    train_table: pd.DataFrame,
    tolerance: float = RELOAD_CONSISTENCY_TOLERANCE,
) -> Dict[str, object]:
    """Compares this configuration's OWN persisted Task 35 training-time
    accuracy against a fresh, read-only measurement using the currently
    saved fusion head. A large gap means the saved .keras file is not the
    model that produced the recorded training accuracy."""
    X, y, meta = fu.assemble_fusion_split_arrays(config_name, train_table)
    fusion_model = fu.load_fusion_head(config_name)
    probs = predict_probabilities(fusion_model, X)
    preds = np.argmax(probs, axis=1)
    measured_accuracy = float((preds == y).mean()) if len(y) else None

    with open(fu.FUSION_TRAINING_HISTORY_JSON_PATH, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    reported_accuracy = float(history[config_name]["train_accuracy"][-1])

    delta = abs(measured_accuracy - reported_accuracy) if measured_accuracy is not None else None
    return {
        "config": config_name,
        "n_observations_checked": int(len(y)),
        "reported_train_accuracy": reported_accuracy,
        "reload_measured_train_accuracy": measured_accuracy,
        "delta": delta,
        "reload_consistent": (delta is not None and delta <= tolerance),
    }


def check_all_reload_consistency(
    train_table: pd.DataFrame,
    tolerance: float = RELOAD_CONSISTENCY_TOLERANCE,
) -> Dict[str, Dict[str, object]]:
    return {config_name: check_reload_consistency(config_name, train_table, tolerance) for config_name in fu.FUSION_CONFIGS}


# ---------------------------------------------------------------------------
# Full audit driver
# ---------------------------------------------------------------------------

def run_relevance_prediction_audit(raw_dir: str = schema.RAW_DATA_DIR) -> Dict[str, object]:
    results_df = pd.read_csv(ur.UNCERTAINTY_RELEVANCE_RESULTS_CSV_PATH)

    vibration_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
    motor_current_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
    normalization_params = pp.load_normalization_params()
    train_table, _ = fu.build_split_representation_table(
        "train", vibration_model, motor_current_model, normalization_params, raw_dir
    )

    reload_consistency = check_all_reload_consistency(train_table)
    prediction_distributions = compute_all_prediction_distributions(results_df)

    inconsistent_configs = [name for name, info in reload_consistency.items() if not info["reload_consistent"]]

    record: Dict[str, object] = {
        "task": "37A",
        "title": "Relevance mapping and prediction audit",
        "relevance_mapping": relevance_mapping_conclusion(),
        "prediction_distributions": prediction_distributions,
        "reload_consistency": reload_consistency,
        "reload_inconsistent_configs": inconsistent_configs,
        "task_38_readiness": {
            "blocking_issues": (
                ["reload_inconsistent_model_artifacts"] if inconsistent_configs else []
            ) + ["incomplete_relevance_mapping"],
            "recommendation": (
                "Task 38 should NOT proceed on vibration_only or vibration_temperature "
                "until their fusion head artifacts are regenerated and verified "
                "reload-consistent (see reload_consistency above). vibration_current and "
                "vibration_current_temperature reload consistently but have almost no "
                "test-split observations with a defined relevance score, because relevance "
                "is undefined for Misalignment/Unbalance and those are nearly the only true "
                "classes present in their test split -- not a model defect. No configuration "
                "currently has both a fully trustworthy model artifact and complete relevance "
                "coverage at the same time."
            ),
        },
    }
    return record


def save_audit(record: Dict[str, object], path: str = AUDIT_JSON_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    record = run_relevance_prediction_audit()
    save_audit(record)
    print("Relevance mapping conclusion:", record["relevance_mapping"]["conclusion"])
    print("Reload-inconsistent configs:", record["reload_inconsistent_configs"])
    for name, info in record["reload_consistency"].items():
        print(f"  {name}: reported={info['reported_train_accuracy']:.4f} measured={info['reload_measured_train_accuracy']:.4f} consistent={info['reload_consistent']}")
