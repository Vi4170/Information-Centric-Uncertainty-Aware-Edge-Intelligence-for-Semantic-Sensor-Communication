"""Task 42 -- end-to-end edge intelligence validation.

Final pre-communication-channel checkpoint for the multimodal track:

    Sensors -> Preprocessing -> Modality representations -> Fusion ->
    Novelty -> Uncertainty -> Relevance -> Temporal Importance ->
    Communication Cost -> Canonical VoI -> DISCARD/BUFFER/SUMMARY/TRANSMIT

This module does not recompute the pipeline from scratch a sixth time.
Tasks 31-41 already built, tested, and persisted every stage's artifact;
this module's job is to CERTIFY that those artifacts are still mutually
consistent and that the invariants each task individually established
(determinism, train-only fitting, split integrity, no leakage, model
reload consistency, component/VoI/decision availability) still hold
simultaneously across the CURRENTLY persisted state -- the final
acceptance gate before a communication-channel (FSO) experiment is even
attempted. No FSO/channel modeling is introduced here.

Each check below reuses an existing, already-tested function wherever one
exists (``src.multimodal_pipeline.observation_schema``'s own leakage
verifiers, Task 36's ``verify_no_observation_id_leakage``/
``verify_no_condition_split_leakage`` via Task 40's
``communication_decision.verify_no_leakage``, Task 40's own
``verify_decisions_match_canonical_policy``/
``verify_undefined_relevance_never_fabricated``, and Task 37A/37C's
``relevance_prediction_audit.check_all_reload_consistency``) rather than
reimplementing a parallel check. The only new raw-data read this task
performs is rebuilding the TRAIN split's representation table once, needed
by the reused reload-consistency check -- every other check reads only
already-persisted JSON/CSV artifacts.

Every check is wrapped so a single failing check does not prevent the
report from being generated for every other check -- this is meant to be
read as a full pass/fail matrix, not a script that aborts on the first
problem.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd

from src.multimodal_pipeline import communication_decision as cd
from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import novelty as nov
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import relevance as rel38
from src.multimodal_pipeline import relevance_prediction_audit as rpa
from src.multimodal_pipeline import representation as rep
from src.multimodal_pipeline import uncertainty_relevance as ur
from src.multimodal_pipeline import voi_behaviour_analysis as vba
from src.multimodal_pipeline import voi_integration as vi

VALIDATION_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_end_to_end_validation_config.json"
)

REQUIRED_ARTIFACTS: List[str] = [
    schema.OBSERVATION_INDEX_CSV_PATH,
    schema.OBSERVATION_SCHEMA_JSON_PATH,
    fu.FUSION_CONFIG_JSON_PATH,
    fu.FUSION_TRAINING_HISTORY_JSON_PATH,
    nov.NOVELTY_CONFIG_JSON_PATH,
    nov.NOVELTY_RESULTS_CSV_PATH,
    ur.UNCERTAINTY_RELEVANCE_CONFIG_JSON_PATH,
    ur.UNCERTAINTY_RELEVANCE_RESULTS_CSV_PATH,
    rel38.RELEVANCE_AUDIT_JSON_PATH,
    vi.VOI_INTEGRATION_CONFIG_JSON_PATH,
    vi.VOI_INTEGRATION_RESULTS_CSV_PATH,
    cd.DECISION_CONFIG_JSON_PATH,
    cd.DECISION_SUMMARY_CSV_PATH,
    vba.BEHAVIOUR_ANALYSIS_JSON_PATH,
    vba.FACTOR_SUMMARY_CSV_PATH,
    vba.DOMINANCE_CSV_PATH,
]


def _run_check(name: str, fn) -> Dict[str, object]:
    """Runs one check function, catching any exception so a single failure
    never prevents the rest of the report from being generated."""
    try:
        result = fn()
        result.setdefault("check", name)
        result.setdefault("passed", True)
        return result
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: this is an
        # audit gate, every failure mode must be captured and reported, not
        # allowed to crash the whole report.
        return {
            "check": name,
            "passed": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# 1. Deterministic observation IDs
# ---------------------------------------------------------------------------


def verify_deterministic_observation_ids(raw_dir: str = schema.RAW_DATA_DIR) -> Dict[str, object]:
    df1 = schema.build_observation_index(raw_dir)
    df2 = schema.build_observation_index(raw_dir)
    rebuild_matches_itself = df1["observation_id"].tolist() == df2["observation_id"].tolist()

    schema.verify_observation_id_uniqueness(df1)
    schema.verify_split_disjoint(df1)
    schema.verify_no_condition_crosses_split(df1)

    persisted = pd.read_csv(schema.OBSERVATION_INDEX_CSV_PATH, usecols=["observation_id"])
    rebuild_matches_persisted = sorted(df1["observation_id"]) == sorted(persisted["observation_id"])

    passed = bool(rebuild_matches_itself and rebuild_matches_persisted)
    return {
        "passed": passed,
        "n_observations": int(len(df1)),
        "rebuild_matches_itself": bool(rebuild_matches_itself),
        "rebuild_matches_persisted_index": bool(rebuild_matches_persisted),
    }


# ---------------------------------------------------------------------------
# 2. Train-only fitting / reference statistics
# ---------------------------------------------------------------------------


def verify_train_only_fitting() -> Dict[str, object]:
    with open(fu.FUSION_CONFIG_JSON_PATH, "r", encoding="utf-8") as handle:
        fusion_config = json.load(handle)
    training_info = fusion_config["training"]
    fit_only_train = training_info["fit_only_on_split"] == "train"
    test_untouched = training_info["test_split_touched_during_training"] is False

    with open(vi.VOI_INTEGRATION_CONFIG_JSON_PATH, "r", encoding="utf-8") as handle:
        voi_config = json.load(handle)
    novelty_train_support = {
        name: info["novelty_n_train_normal_reference"] for name, info in voi_config["configurations"].items()
    }
    all_positive = all(v > 0 for v in novelty_train_support.values())

    passed = bool(fit_only_train and test_untouched and all_positive)
    return {
        "passed": passed,
        "fusion_head_fit_only_on_split": training_info["fit_only_on_split"],
        "fusion_head_test_split_touched_during_training": training_info["test_split_touched_during_training"],
        "novelty_n_train_normal_reference_by_config": novelty_train_support,
    }


# ---------------------------------------------------------------------------
# 3. Condition-level split integrity / no test leakage
# ---------------------------------------------------------------------------


def verify_split_integrity_and_no_leakage(results_df: pd.DataFrame) -> Dict[str, object]:
    cd.verify_no_leakage(results_df)  # raises on any violation (reuses Task 36's checks, not reimplemented)
    return {
        "passed": True,
        "configurations_checked": sorted(results_df["fusion_config"].unique().tolist()),
        "n_observations_checked": int(len(results_df)),
    }


# ---------------------------------------------------------------------------
# 4. Model reload consistency (the one live raw-data read this task performs)
# ---------------------------------------------------------------------------


def verify_model_reload_consistency(raw_dir: str = schema.RAW_DATA_DIR) -> Dict[str, object]:
    vibration_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
    motor_current_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
    normalization_params = pp.load_normalization_params()
    train_table, _ = fu.build_split_representation_table(
        "train", vibration_model, motor_current_model, normalization_params, raw_dir
    )
    reload_info = rpa.check_all_reload_consistency(train_table)
    all_consistent = all(info["reload_consistent"] for info in reload_info.values())
    return {"passed": bool(all_consistent), "per_configuration": reload_info}


# ---------------------------------------------------------------------------
# 5. Artifact consistency (existence + cross-artifact count agreement)
# ---------------------------------------------------------------------------


def verify_required_artifacts_exist() -> Dict[str, object]:
    missing = [p for p in REQUIRED_ARTIFACTS if not os.path.exists(p)]
    return {"passed": len(missing) == 0, "n_required": len(REQUIRED_ARTIFACTS), "missing": missing}


def verify_cross_artifact_count_consistency(
    results_df: pd.DataFrame, decision_summary_df: pd.DataFrame, factor_summary_df: pd.DataFrame
) -> Dict[str, object]:
    """Confirms Task 39's per-observation table, Task 40's decision summary,
    and Task 41's factor summary agree on how many observations exist per
    (fusion_config, split) -- catches a stale artifact from before a
    re-run, since these three CSVs are otherwise independently loaded."""
    mismatches: List[dict] = []
    for (config_name, split_name), group in results_df.groupby(["fusion_config", "split"]):
        n_task39 = len(group)

        row40 = decision_summary_df[
            (decision_summary_df["fusion_config"] == config_name) & (decision_summary_df["split"] == split_name)
        ]
        n_task40 = int(row40["n"].iloc[0]) if len(row40) else None

        row41 = factor_summary_df[
            (factor_summary_df["fusion_config"] == config_name) & (factor_summary_df["split"] == split_name)
        ]
        n_task41 = int(row41["n"].iloc[0]) if len(row41) else None

        if n_task40 != n_task39 or n_task41 != n_task39:
            mismatches.append(
                {
                    "fusion_config": config_name,
                    "split": split_name,
                    "task39_n": n_task39,
                    "task40_n": n_task40,
                    "task41_n": n_task41,
                }
            )
    return {"passed": len(mismatches) == 0, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# 6. Component / VoI / decision availability
# ---------------------------------------------------------------------------


def verify_component_and_decision_availability(results_df: pd.DataFrame) -> Dict[str, object]:
    always_on_columns = ("novelty", "uncertainty", "temporal_importance", "resource_cost")
    non_finite: Dict[str, int] = {
        col: int((~np.isfinite(results_df[col])).sum()) for col in always_on_columns
    }
    if any(non_finite.values()):
        raise AssertionError(f"Non-finite values found in always-available factor(s): {non_finite}")

    if results_df["decision"].isna().any():
        raise AssertionError("Some observation(s) have no decision at all (null)")

    cd.verify_decisions_match_canonical_policy(results_df)  # reused, not reimplemented
    cd.verify_undefined_relevance_never_fabricated(results_df)  # reused, not reimplemented

    n = len(results_df)
    n_defined = int(results_df["relevance_defined"].sum())
    return {
        "passed": True,
        "n_observations": n,
        "n_relevance_defined": n_defined,
        "n_relevance_undefined": n - n_defined,
        "always_available_factors_checked": list(always_on_columns),
        "always_available_factors_non_finite_counts": non_finite,
    }


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------


def run_task42(raw_dir: str = schema.RAW_DATA_DIR) -> Dict[str, object]:
    results_df = cd.load_voi_integration_results(vi.VOI_INTEGRATION_RESULTS_CSV_PATH)
    decision_summary_df = pd.read_csv(cd.DECISION_SUMMARY_CSV_PATH)
    factor_summary_df = pd.read_csv(vba.FACTOR_SUMMARY_CSV_PATH)

    checks: Dict[str, Dict[str, object]] = {
        "deterministic_observation_ids": _run_check(
            "deterministic_observation_ids", lambda: verify_deterministic_observation_ids(raw_dir)
        ),
        "train_only_fitting": _run_check("train_only_fitting", verify_train_only_fitting),
        "split_integrity_and_no_leakage": _run_check(
            "split_integrity_and_no_leakage", lambda: verify_split_integrity_and_no_leakage(results_df)
        ),
        "model_reload_consistency": _run_check(
            "model_reload_consistency", lambda: verify_model_reload_consistency(raw_dir)
        ),
        "required_artifacts_exist": _run_check("required_artifacts_exist", verify_required_artifacts_exist),
        "cross_artifact_count_consistency": _run_check(
            "cross_artifact_count_consistency",
            lambda: verify_cross_artifact_count_consistency(results_df, decision_summary_df, factor_summary_df),
        ),
        "component_and_decision_availability": _run_check(
            "component_and_decision_availability",
            lambda: verify_component_and_decision_availability(results_df),
        ),
    }

    all_passed = all(c["passed"] for c in checks.values())
    failed_checks = [name for name, c in checks.items() if not c["passed"]]

    report: Dict[str, object] = {
        "task": 42,
        "title": "End-to-end edge intelligence validation",
        "pipeline_stages_validated": [
            "sensors", "preprocessing", "modality_representations", "fusion",
            "novelty", "uncertainty", "relevance", "temporal_importance",
            "communication_cost", "canonical_voi", "decision",
        ],
        "fso_introduced": False,
        "checks": checks,
        "failed_checks": failed_checks,
        "pipeline_ready_for_communication_channel_experiment": bool(all_passed),
    }
    return report


def save_validation_report(report: Dict[str, object], path: str = VALIDATION_JSON_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    print("=== Executing Task 42: End-to-End Edge Intelligence Validation ===")
    report = run_task42()
    save_validation_report(report)

    for name, result in report["checks"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {name}")
        if not result["passed"] and "error" in result:
            print(f"         error: {result['error']}")

    print(f"\nPipeline ready for communication channel experiment: {report['pipeline_ready_for_communication_channel_experiment']}")
    print(f"Saved report to: {VALIDATION_JSON_PATH}")
