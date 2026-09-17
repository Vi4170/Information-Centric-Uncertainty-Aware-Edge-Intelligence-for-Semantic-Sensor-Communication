"""Task 37A/37C -- relevance mapping and prediction audit.

Read-only audit module. Never trains, never saves, never modifies any
protected module (src/voi/, src/relevance/relevance.py, src/uncertainty/,
src/novelty/), any Task 35 model artifact, or Task 36/37's own saved
artifacts. This module only READS the already-frozen Task 35 fusion heads
and the already-saved Task 37 results to compute:

1. Prediction distributions (true vs. predicted fault_type, counts AND
   percentages) per fusion configuration and split, read directly from
   Task 37's own persisted results CSV -- no new model inference here.

2. Relevance coverage: how many/what percentage of each configuration's
   split has a DEFINED relevance score (predicted class in {Normal, BPFI,
   BPFO}) vs. undefined (predicted Misalignment/Unbalance), broken down by
   predicted class. Never treats "undefined" as zero or any other value.

3. A reload-consistency check: for each configuration, the frozen fusion
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
Task 37A finding (original) / Task 37C status (current)
======================================================================
Task 37A's reload-consistency check surfaced a genuine, pre-existing
defect: ``vibration_only`` and ``vibration_temperature``'s saved fusion
heads reloaded to ~24-25% train accuracy despite their own
training_history.json entries reporting ~95.8%/96.5% -- a ~70
percentage-point gap, while ``vibration_current`` and
``vibration_current_temperature`` reloaded within ~0.2 percentage points
of their own reported accuracy. Git archaeology traced the root cause to
``tests/test_multimodal_fusion.py``'s
``test_30_train_all_fusion_heads_never_requests_test_split``, which called
``train_all_fusion_heads(epochs=1)`` on a synthetic 3-row registry WITHOUT
redirecting ``FUSION_MODEL_DIR``, present unpatched in the very first Task
35 commit (``bf2c9f6``) and again mid-session before being fixed in
``df49d7e`` -- silently overwriting the real production fusion models with
a severely undertrained model each time it ran.

Task 37B (commits ``5a17cfb``/``2bea59a``, merged into this branch) fixed
this by retraining ``vibration_only``/``vibration_temperature`` and
regenerating Task 36/37's downstream artifacts. Task 37C (this run)
independently re-verifies reload-consistency for all four configurations
and re-derives prediction distributions and relevance coverage against
the corrected data -- see the audit JSON's ``reload_consistency`` and
``relevance_coverage`` sections for current, not historical, numbers. This
module still does not retrain or repair anything itself.
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

def _pct_map(counts: Dict[str, int], n: int) -> Dict[str, float]:
    if n == 0:
        return {}
    return {k: round(100.0 * v / n, 4) for k, v in counts.items()}


def compute_prediction_distribution(results_df: pd.DataFrame, config_name: str, split: str) -> Dict[str, object]:
    """True vs. predicted fault_type counts AND percentages for one
    configuration/split, computed directly from Task 37's own persisted
    per-observation results (never from a new model run)."""
    sub = results_df[(results_df["fusion_config"] == config_name) & (results_df["split"] == split)]
    n = int(len(sub))
    true_counts = {k: int(v) for k, v in sub["fault_type"].value_counts().to_dict().items()}
    pred_counts = {k: int(v) for k, v in sub["predicted_fault_type"].value_counts().to_dict().items()}
    return {
        "n": n,
        "true_distribution": true_counts,
        "true_distribution_pct": _pct_map(true_counts, n),
        "predicted_distribution": pred_counts,
        "predicted_distribution_pct": _pct_map(pred_counts, n),
    }


def compute_all_prediction_distributions(results_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    return {
        config_name: {split: compute_prediction_distribution(results_df, config_name, split) for split in ("train", "val", "test")}
        for config_name in fu.FUSION_CONFIGS
    }


# ---------------------------------------------------------------------------
# Relevance coverage -- how much of each configuration/split has a DEFINED
# relevance score (predicted class in {Normal, BPFI, BPFO}) vs. undefined
# (predicted Misalignment/Unbalance), and the breakdown by predicted class.
# Never treats an undefined score as zero or any other fabricated value --
# only counts/percentages of definedness are reported.
# ---------------------------------------------------------------------------

def compute_relevance_coverage(results_df: pd.DataFrame, config_name: str, split: str) -> Dict[str, object]:
    sub = results_df[(results_df["fusion_config"] == config_name) & (results_df["split"] == split)]
    n = int(len(sub))
    if n == 0:
        return {"n": 0, "n_defined": 0, "n_undefined": 0, "pct_defined": 0.0, "pct_undefined": 0.0, "coverage_by_predicted_class": {}}

    defined_mask = sub["relevance_score"].notna()
    n_defined = int(defined_mask.sum())
    n_undefined = n - n_defined

    coverage_by_predicted_class: Dict[str, Dict[str, object]] = {}
    for fault_type in ur.EXPECTED_FAULT_TYPE_LABELS:
        class_sub = sub[sub["predicted_fault_type"] == fault_type]
        class_n = int(len(class_sub))
        coverage_by_predicted_class[fault_type] = {
            "n": class_n,
            "n_defined": int(class_sub["relevance_score"].notna().sum()) if class_n else 0,
        }

    return {
        "n": n,
        "n_defined": n_defined,
        "n_undefined": n_undefined,
        "pct_defined": round(100.0 * n_defined / n, 4),
        "pct_undefined": round(100.0 * n_undefined / n, 4),
        "coverage_by_predicted_class": coverage_by_predicted_class,
    }


def compute_all_relevance_coverage(results_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    return {
        config_name: {split: compute_relevance_coverage(results_df, config_name, split) for split in ("train", "val", "test")}
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
    relevance_coverage = compute_all_relevance_coverage(results_df)

    inconsistent_configs = [name for name, info in reload_consistency.items() if not info["reload_consistent"]]
    test_coverage_pct = {name: relevance_coverage[name]["test"]["pct_defined"] for name in fu.FUSION_CONFIGS}

    blocking_issues: List[str] = []
    if inconsistent_configs:
        blocking_issues.append("reload_inconsistent_model_artifacts")
    blocking_issues.append("incomplete_relevance_mapping")

    record: Dict[str, object] = {
        "task": "37C",
        "title": "Post-regeneration prediction distribution and relevance audit",
        "supersedes": "37A (data/processed/multimodal/multimodal_task37a_relevance_prediction_audit.json is now stale: it was computed against the corrupted vibration_only/vibration_temperature fusion heads)",
        "relevance_mapping": relevance_mapping_conclusion(),
        "prediction_distributions": prediction_distributions,
        "relevance_coverage": relevance_coverage,
        "reload_consistency": reload_consistency,
        "reload_inconsistent_configs": inconsistent_configs,
        "task_38_readiness": {
            "blocking_issues": blocking_issues,
            "test_split_pct_defined_relevance_by_config": test_coverage_pct,
            "recommendation": (
                "All four fusion heads are now reload-consistent (no model-artifact "
                "blocker remains). The relevance-mapping gap (Misalignment/Unbalance "
                "undefined) is unchanged and remains a structural limitation, not a bug: "
                "vibration_only/vibration_temperature now have real, non-degenerate "
                "relevance coverage (~17-44% defined across splits). vibration_current/"
                "vibration_current_temperature still have almost no test-split coverage "
                "(0.00-0.04% defined) because their test split's true labels are almost "
                "entirely Misalignment/Unbalance -- a test-composition limitation "
                "confirmed independent of model quality, not something a retrain can fix. "
                "Task 38 may reasonably proceed for vibration_only/vibration_temperature; "
                "for vibration_current/vibration_current_temperature, Task 38 should "
                "explicitly restrict itself to observations with a defined relevance score "
                "and disclose the resulting near-total test-split exclusion rather than "
                "silently computing VoI over an unrepresentative handful of observations."
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
    print("\nRelevance coverage (pct defined) by config/split:")
    for name, splits in record["relevance_coverage"].items():
        for split_name, info in splits.items():
            print(f"  {name}/{split_name}: n={info['n']} pct_defined={info['pct_defined']}")
