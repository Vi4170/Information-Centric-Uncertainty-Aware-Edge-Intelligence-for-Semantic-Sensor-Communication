import json
import os

import pandas as pd

from src.voi.scoring import VoIWeights
from src.voi.decision_policy import PolicyThresholds

TABLE_DIR = "results/tables"
OUTPUT_PATH = os.path.join("results", "cross_dataset", "task30_cross_dataset_analysis.json")

STATUS_MEASURED = "measured"
STATUS_DERIVED = "derived"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_NOT_SCIENTIFICALLY_VALID = "not_scientifically_valid"

STATUS_VOCABULARY = {
    STATUS_MEASURED: "Computed directly from real data using the existing, unmodified module for this factor.",
    STATUS_DERIVED: "Computed from measured quantities via a documented, non-arbitrary transformation (e.g. a correlation of an already-measured score against trajectory position).",
    STATUS_NOT_AVAILABLE: "Not computed in this experiment, for a reason unrelated to scientific validity (e.g. missing raw data, out of scope).",
    STATUS_NOT_SCIENTIFICALLY_VALID: "Deliberately not computed because computing it would require inventing a label or methodology this project's own rules prohibit.",
}


def _canonical_voi_configuration():
    weights = VoIWeights()
    thresholds = PolicyThresholds()
    return {
        "weights": {
            "novelty": weights.novelty,
            "uncertainty": weights.uncertainty,
            "task_relevance": weights.task_relevance,
            "temporal_importance": weights.temporal_importance,
            "resource_cost": weights.resource_cost,
        },
        "thresholds": {
            "discard_max": thresholds.discard_max,
            "buffer_max": thresholds.buffer_max,
            "summary_max": thresholds.summary_max,
        },
    }


def _model_param_count(model_path):
    import keras

    model = keras.models.load_model(model_path, compile=False)
    return int(model.count_params())


def _cwru_evidence():
    cnn_summary = pd.read_csv(os.path.join(TABLE_DIR, "cnn_evaluation_summary.csv")).iloc[0]
    voi_summary = pd.read_csv(os.path.join(TABLE_DIR, "voi_integration_summary.csv"))
    decision_dist = pd.read_csv(os.path.join(TABLE_DIR, "voi_decision_distribution.csv"))
    dominance = pd.read_csv(os.path.join(TABLE_DIR, "voi_factor_dominance.csv"))

    test_row = voi_summary[voi_summary["group"] == "test"].iloc[0]
    test_decision = decision_dist[decision_dist["group"] == "test"].iloc[0]
    test_dominance = dominance[dominance["group"] == "test"].iloc[0]

    novelty_by_class = {
        row["group"]: float(row["novelty_mean"])
        for _, row in voi_summary.iterrows()
        if str(row["group"]).startswith("test_class_")
    }

    continual = None
    continual_path = os.path.join("results", "continual", "task25_cwru_continual_experiment.json")
    if os.path.exists(continual_path):
        with open(continual_path, "r", encoding="utf-8") as f:
            continual_raw = json.load(f)
        continual = {
            "known_condition_name": continual_raw["experiment_configuration"]["known_condition_name"],
            "new_condition_name": continual_raw["experiment_configuration"]["new_condition_name"],
            "new_condition_already_in_trained_cnn_label_space": True,
            "baseline_model_accuracy": continual_raw["post_hoc_test_metrics"]["baseline_model_accuracy"],
            "final_active_model_accuracy": continual_raw["post_hoc_test_metrics"]["final_active_model_accuracy"],
            "admission_decision": continual_raw["admission_result"]["decision"],
            "regression_decision": continual_raw["regression_result"]["decision"],
            "activated": continual_raw["activation_result"]["activated"],
            "leakage_verified": continual_raw["leakage_verification"]["verified"],
            "active_model_untouched": continual_raw["active_model_untouched"],
        }

    return {
        "dataset": "cwru",
        "available_experiment_type": "full canonical VoI pipeline: supervised CNN classification + 5-factor VoI + decision policy",
        "n_observations": {"train": 1508, "val": 406, "test": 406, "total": 2320},
        "cnn_availability": {"status": STATUS_MEASURED, "params": _model_param_count(os.path.join("models", "cwru_cnn_baseline.keras")), "test_accuracy": float(cnn_summary["accuracy"]), "detail": "src/cnn, canonical architecture, 4-class"},
        "classification_availability": {"status": STATUS_MEASURED, "n_classes": 4, "class_names": ["Normal", "Inner Race Fault", "Ball Fault", "Outer Race Fault"]},
        "novelty_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["novelty_mean"]), "test_by_class": novelty_by_class},
        "uncertainty_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["uncertainty_mean"]), "note": "functionally negligible: highly confident CNN produces near-zero entropy"},
        "relevance_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["task_relevance_mean"]), "share_of_positive_voi_contribution_pct": float(test_dominance["task_relevance_share_of_positive_contribution_pct"])},
        "temporal_importance_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["temporal_importance_mean"]), "note": "measures window-to-window signal volatility, not condition drift, since CWRU is a static single-condition dataset"},
        "communication_cost_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["resource_cost_mean"]), "note": "constant by construction: every window is the same size, no channel model exists"},
        "full_voi_availability": {"status": STATUS_MEASURED, "test_mean_voi_score": float(test_row["voi_score_mean"])},
        "communication_decision_availability": {
            "status": STATUS_MEASURED,
            "note": "decision-policy simulation only, not physical transmission",
            "test_decision_distribution_pct": {
                "DISCARD": float(test_decision["DISCARD_pct"]), "BUFFER": float(test_decision["BUFFER_pct"]),
                "SUMMARY": float(test_decision["SUMMARY_pct"]), "TRANSMIT": float(test_decision["TRANSMIT_pct"]),
            },
        },
        "continual_learning_evidence": (
            {"status": STATUS_MEASURED, "caveat": "new_condition_name was already a class in the CNN's trained label space; this validates gating mechanics, not genuine unseen-class learning", **continual}
            if continual else {"status": STATUS_NOT_AVAILABLE}
        ),
        "run_to_failure_evidence": {"status": STATUS_NOT_AVAILABLE, "reason": "CWRU is a static, single-condition dataset with no degradation trajectory"},
        "main_finding": "The calibrated canonical VoI pipeline discriminates cleanly by class: Normal is discarded 100% of the time, Inner Race Fault (highest relevance weight) is the only class reaching TRANSMIT.",
        "main_limitation": "Uncertainty and Communication Cost are structurally non-discriminating for this dataset/model; BUFFER is nearly unused (4/406 test observations); Temporal Importance measures volatility, not drift.",
    }


def _paderborn_evidence():
    cnn_summary = pd.read_csv(os.path.join(TABLE_DIR, "paderborn_cnn_evaluation_summary.csv")).iloc[0]
    voi_summary = pd.read_csv(os.path.join(TABLE_DIR, "paderborn_voi_integration_summary.csv"))
    decision_dist = pd.read_csv(os.path.join(TABLE_DIR, "paderborn_voi_decision_distribution.csv"))
    dominance = pd.read_csv(os.path.join(TABLE_DIR, "paderborn_voi_factor_dominance.csv"))

    with open(os.path.join("data", "processed", "paderborn", "paderborn_experiment_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_row = voi_summary[voi_summary["group"] == "test"].iloc[0]
    test_decision = decision_dist[decision_dist["group"] == "test"].iloc[0]
    test_dominance = dominance[dominance["group"] == "test"].iloc[0]

    novelty_by_class = {
        row["group"]: float(row["novelty_mean"])
        for _, row in voi_summary.iterrows()
        if str(row["group"]).startswith("test_class_")
    }

    return {
        "dataset": "paderborn",
        "available_experiment_type": "full canonical VoI pipeline: supervised CNN classification + 5-factor VoI + decision policy, single operating condition",
        "n_observations": manifest["split_sizes"] | {"total": manifest["total_windows"]},
        "cnn_availability": {"status": STATUS_MEASURED, "params": _model_param_count(os.path.join("models", "paderborn_cnn_baseline.keras")), "test_accuracy": float(cnn_summary["accuracy"])},
        "classification_availability": {"status": STATUS_MEASURED, "n_classes": 3, "class_names": ["Healthy", "Inner Race Fault", "Outer Race Fault"], "excluded_bearing_codes": manifest["n_bearing_codes_excluded"], "included_bearing_codes": manifest["n_bearing_codes_included"]},
        "novelty_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["novelty_mean"]), "test_by_class": novelty_by_class},
        "uncertainty_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["uncertainty_mean"]), "note": "negligible, ~20x higher than CWRU in absolute terms but still tiny; weight 0.05"},
        "relevance_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["task_relevance_mean"]), "share_of_positive_voi_contribution_pct": float(test_dominance["task_relevance_share_of_positive_contribution_pct"]), "note": "values inherited unchanged from CWRU's per-fault-type relevance map"},
        "temporal_importance_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["temporal_importance_mean"]), "note": "uses CWRU-calibrated scale (1.8) unmodified; did not saturate"},
        "communication_cost_availability": {"status": STATUS_MEASURED, "test_mean": float(test_row["resource_cost_mean"]), "note": "constant, identical value to CWRU (same window size)"},
        "full_voi_availability": {"status": STATUS_MEASURED, "test_mean_voi_score": float(test_row["voi_score_mean"])},
        "communication_decision_availability": {
            "status": STATUS_MEASURED,
            "note": "decision-policy simulation only, not physical transmission",
            "test_decision_distribution_pct": {
                "DISCARD": float(test_decision["DISCARD_pct"]), "BUFFER": float(test_decision["BUFFER_pct"]),
                "SUMMARY": float(test_decision["SUMMARY_pct"]), "TRANSMIT": float(test_decision["TRANSMIT_pct"]),
            },
        },
        "continual_learning_evidence": {"status": STATUS_NOT_AVAILABLE, "reason": "no continual-learning experiment was performed on Paderborn"},
        "run_to_failure_evidence": {"status": STATUS_NOT_AVAILABLE, "reason": "Paderborn bearings carry pre-existing, static, induced damage; not a degradation trajectory"},
        "main_finding": "The identical, unmodified canonical CNN+VoI pipeline transfers mechanically to a different dataset, but produces materially different decision behavior: mass shifts from SUMMARY toward BUFFER and TRANSMIT nearly vanishes (1.49% vs CWRU's 14.04%), traceable primarily to ~3.7x lower novelty scores.",
        "main_limitation": "6 of 32 bearing codes excluded (ambiguous/multi-location damage labels); only 1 of 4 operating conditions used; novelty/temporal not numerically comparable to CWRU (different trained embedding spaces).",
    }


def _ims_evidence():
    progression = pd.read_csv(os.path.join(TABLE_DIR, "ims_temporal_progression_summary.csv"))
    correlation = pd.read_csv(os.path.join(TABLE_DIR, "ims_failure_proximity_correlation.csv"))

    with open(os.path.join("data", "processed", "ims", "ims_temporal_experiment_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    per_target_correlation = correlation.to_dict("records")
    n_windows_total = int(correlation["n_windows"].sum())

    return {
        "dataset": "ims",
        "available_experiment_type": "healthy-reference + degradation-progression analysis (no supervised classification, no full VoI)",
        "n_observations": {"n_runs": 4, "n_analysis_targets": len(correlation), "total_windows_analyzed": n_windows_total},
        "cnn_availability": {"status": STATUS_NOT_AVAILABLE, "reason": "deliberately not trained: no legitimate per-window label exists to train against"},
        "classification_availability": {"status": STATUS_NOT_SCIENTIFICALLY_VALID, "reason": "IMS failure information is run-level only; treating it as a per-window label would misrepresent early-run (healthy) windows"},
        "novelty_availability": {"status": STATUS_MEASURED, "note": "computed on raw flattened window signal, not a CNN embedding, since none can be legitimately trained", "correlation_with_trajectory_position": per_target_correlation},
        "uncertainty_availability": {"status": STATUS_NOT_SCIENTIFICALLY_VALID, "reason": manifest["not_computed"]["uncertainty"]},
        "relevance_availability": {"status": STATUS_NOT_SCIENTIFICALLY_VALID, "reason": manifest["not_computed"]["task_relevance"]},
        "temporal_importance_availability": {"status": STATUS_MEASURED, "note": "existing DEFAULT_TEMPORAL_CHANGE_SCALE (1.8) used unmodified, not re-derived for IMS"},
        "communication_cost_availability": {"status": STATUS_MEASURED, "note": "constant, same value as CWRU/Paderborn (identical window size)"},
        "full_voi_availability": {"status": STATUS_NOT_AVAILABLE, "reason": manifest["not_computed"]["voi_score_and_decision"]},
        "communication_decision_availability": {"status": STATUS_NOT_AVAILABLE, "reason": "no valid VoI score exists to feed the decision policy"},
        "continual_learning_evidence": {"status": STATUS_NOT_AVAILABLE, "reason": "not attempted for IMS"},
        "run_to_failure_evidence": {
            "status": STATUS_MEASURED,
            "note": "first genuine chronological run-to-failure structure evaluated in this project, from real per-file timestamps",
            "documented_failure_targets_stronger_than_controls_in": ["1st_test", "2nd_test", "4th_test (undocumented continuation of 3rd_test)"],
            "exception": "3rd_test's own documented window shows a weak/flat trend (corr 0.31 novelty, 0.22 temporal), and its control is mildly negative (-0.21/-0.21) -- not a universal pattern",
        },
        "main_finding": "In 3 of 4 runs, vendor-documented failing bearings show substantially stronger, often near-saturating late-run rises in Novelty and Temporal Importance than same-run controls; 3rd_test is a genuine exception with flat behavior in both its failing bearing and its control.",
        "main_limitation": "No failure-onset timestamp exists for any run (only 'by end of run'); Novelty uses raw signal, not a learned embedding, so it is not numerically comparable to CWRU/Paderborn; same-run controls are not fully independent of rig-wide effects; the pattern is 3-of-4, not universal.",
    }


def _xjtu_evidence():
    with open(os.path.join("data", "processed", "xjtu", "xjtu_dataset_summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)

    unavailable = {"status": STATUS_NOT_AVAILABLE, "reason": "no raw data present in this environment"}
    return {
        "dataset": "xjtu_sy",
        "available_experiment_type": "not executed: structurally evaluated, blocked by missing raw data in this environment",
        "n_observations": {"n_bearings_per_registry": summary["n_bearings_total"], "n_operating_conditions": len(summary["operating_conditions"]), "n_files_expected": summary["n_files_total_expected"], "n_files_actually_readable_here": 0},
        "cnn_availability": unavailable,
        "classification_availability": unavailable,
        "novelty_availability": unavailable,
        "uncertainty_availability": unavailable,
        "relevance_availability": unavailable,
        "temporal_importance_availability": unavailable,
        "communication_cost_availability": unavailable,
        "full_voi_availability": unavailable,
        "communication_decision_availability": unavailable,
        "continual_learning_evidence": unavailable,
        "run_to_failure_evidence": {"status": STATUS_NOT_AVAILABLE, "reason": "would be measurable per structural design (15 trajectories, 3 operating conditions) if raw data existed; currently absent"},
        "main_finding": "Structural comparison shows XJTU-SY would offer more run-to-failure trajectories (15 vs IMS's 4), more operating conditions (3 vs IMS's 1), and more diverse documented fault types than IMS, but no raw data exists in this environment to compute anything.",
        "main_limitation": "Complete absence of raw data in this environment; even the project's own 37 pre-existing tests for this dataset are unexecuted (raw-data-gated, currently skipped).",
    }


def _mimii_evidence():
    with open(os.path.join("data", "processed", "mimii_dg", "mimii_dg_dataset_summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)

    unavailable = {"status": STATUS_NOT_AVAILABLE, "reason": "original MIMII (0 dB subset, 4 machine types) was never integrated; only the structurally different MIMII-DG exists, with no local raw data in this environment either"}
    return {
        "dataset": "mimii",
        "available_experiment_type": "not integrated",
        "n_observations": {"mimii_dg_recordings_per_committed_summary": summary["recordings"]["total"], "mimii_dg_machine_types": summary["dataset_structure"]["machine_types"]},
        "cnn_availability": unavailable,
        "classification_availability": unavailable,
        "novelty_availability": unavailable,
        "uncertainty_availability": unavailable,
        "relevance_availability": unavailable,
        "temporal_importance_availability": unavailable,
        "communication_cost_availability": unavailable,
        "full_voi_availability": unavailable,
        "communication_decision_availability": unavailable,
        "continual_learning_evidence": unavailable,
        "run_to_failure_evidence": unavailable,
        "main_finding": "N/A -- not integrated.",
        "main_limitation": "The only 'MIMII' code present is MIMII-DG, a dataset with a different machine-type taxonomy and no dB/SNR structure than the originally decided scope (original MIMII, 0 dB subset, 4 machine types); no raw data for either variant exists in this environment.",
    }


def _cross_dataset_patterns():
    return [
        {
            "observed": "Task Relevance is the largest single contributor to positive VoI in both full-VoI experiments (CWRU 48.1%, Paderborn 62.3%).",
            "interpretation": "Relevance's hand-authored class map is the most directly task-aligned, deterministic input among the five factors, so it dominates once Novelty's contribution shrinks.",
            "limitation": "Relevance's values are manually assigned, not learned or measured from data, so this dominance partly reflects the fixed weight and map choice rather than an emergent discovery about the underlying signals.",
        },
        {
            "observed": "Novelty differs sharply between CWRU (test mean 0.633) and Paderborn (0.169), a ~3.7x gap, despite the identical formula and fitting procedure.",
            "interpretation": "The same distance-to-reference-centroid formula is sensitive to how well each dataset's independently-trained classifier happens to separate classes in its own embedding space.",
            "limitation": "This does not establish that Paderborn's faults are intrinsically less novel or severe than CWRU's; the comparison is confounded by two different CNN instances trained on two different datasets.",
        },
        {
            "observed": "Uncertainty contributes 0.01%-0.21% of positive VoI across both full-VoI experiments.",
            "interpretation": "Entropy-based uncertainty from a highly confident, high-accuracy CNN is inherently near-zero, and this pattern held across two independently-trained models.",
            "limitation": "Only two data points exist; this does not establish the estimator is inherently uninformative in general, only under these two well-separated-class scenarios.",
        },
        {
            "observed": "The same Temporal Importance formula (consecutive-window mean absolute difference) is used in every experiment, but only in IMS's genuinely time-ordered run-to-failure data does it correlate strongly (0.22-0.85) with trajectory position; in CWRU/Paderborn it operates on static or within-recording sequences.",
            "interpretation": "The formula measures window-to-window signal volatility regardless of dataset; only when the underlying sequence is genuinely temporally structured (IMS) does that volatility plausibly reflect real degradation rather than incidental within-recording variation.",
            "limitation": "Even within IMS, correlation with trajectory position is not the same as correlation with true failure onset, since no onset timestamp is documented for any run.",
        },
        {
            "observed": "Communication Cost is a literal constant (0.505) in every experiment where it was computed (CWRU, Paderborn, IMS).",
            "interpretation": "With no real channel or FSO model, this factor cannot discriminate between observations by design, regardless of dataset.",
            "limitation": "This is a structural property of the current placeholder implementation, not a discovery from the cross-dataset comparison itself; no communication-cost-driven decision variation has been observed anywhere in the project.",
        },
        {
            "observed": "The identical canonical VoI weights and thresholds produce materially different decision-tier distributions on CWRU (SUMMARY-dominant, 70.69%; TRANSMIT 14.04%) versus Paderborn (BUFFER-dominant, 57.46%; TRANSMIT 1.49%).",
            "interpretation": "A fixed weight/threshold configuration calibrated against one dataset's factor distributions does not automatically produce comparable decision behavior on a different dataset's differently-scaled factor distributions, primarily driven by the novelty gap above.",
            "limitation": "This is evidence against assuming the current calibration generalizes across datasets; it is not evidence that the framework is broken, and no re-calibration was attempted or is implied to be needed without further scoped work.",
        },
    ]


def build_evidence_matrix():
    return [_cwru_evidence(), _paderborn_evidence(), _ims_evidence(), _xjtu_evidence(), _mimii_evidence()]


def run_cross_dataset_analysis():
    evidence_matrix = build_evidence_matrix()

    result = {
        "task": 30,
        "title": "Cross-Dataset Evidence and Research Analysis",
        "canonical_voi_configuration": _canonical_voi_configuration(),
        "status_vocabulary": STATUS_VOCABULARY,
        "evidence_matrix": evidence_matrix,
        "cross_dataset_patterns": _cross_dataset_patterns(),
        "communication_argument": {
            "what_is_demonstrated": "Communication decision simulation / policy behavior: given the five VoI factors for an observation, the canonical decision policy deterministically assigns DISCARD, BUFFER, SUMMARY, or TRANSMIT.",
            "what_is_not_demonstrated": "Physical FSO packet transmission or reception; measured bandwidth or energy savings; channel-condition-dependent communication cost (the cost factor is a config-driven constant, not measured telemetry).",
            "evidence_still_missing": [
                "An actual or physically-modeled FSO channel (attenuation, turbulence, BER, packet loss) feeding a non-constant Communication Cost.",
                "A packet transmission/receiver evaluation demonstrating that a TRANSMIT/SUMMARY/BUFFER/DISCARD decision maps to an actual data-volume or timing outcome.",
                "A measured communication-savings metric (e.g. bytes or channel-time saved) comparing the VoI-gated policy against a naive transmit-everything baseline, computed from real or physically modeled transmission, not from the decision-label distribution alone.",
            ],
        },
        "defensible_contribution": (
            "The project has built and validated, end-to-end, on two independently-trained CNN classifiers over two distinct bearing-vibration datasets (CWRU, Paderborn), a working reference implementation of a five-factor Value-of-Information scoring pipeline "
            "(Novelty, Uncertainty, Task Relevance, Temporal Importance, Communication Cost) feeding a deterministic four-tier communication decision policy, with documented, non-arbitrary train/val-only calibration (Task 14) and independent validation (Task 15). "
            "It has additionally demonstrated Novelty- and Temporal-Importance-based degradation-progression tracking on a genuine run-to-failure vibration dataset (IMS), honestly reporting where the framework's other components (Uncertainty, Relevance, full VoI) do not apply. "
            "It has further demonstrated, on CWRU, that a gated model-adaptation architecture (novelty-triggered condition-shift detection, safety/regression gating, head-only adaptation with rehearsal, versioned model registry) operates correctly under a defined test scenario."
        ),
        "claims_that_must_not_be_made": [
            "A universal semantic communication system -- only two datasets received an end-to-end classification-style evaluation.",
            "Proven generalization across all datasets -- directly contradicted by the CWRU-vs-Paderborn decision-distribution difference above.",
            "A production-ready industrial system -- no real-time, real-hardware, or real-channel testing exists anywhere in the project.",
            "Proven energy savings -- Communication Cost is a constant placeholder, never tied to a real energy or bandwidth measurement.",
            "Proven FSO reliability -- no FSO experiment has been run; src/communication is a config-driven arithmetic placeholder by its own docstring.",
            "Genuine unseen-condition continual learning -- Task 25's 'new condition' (Inner Race Fault) was already a class in the CNN's trained label space; only the gating mechanics were validated.",
        ],
        "research_gaps": {
            "critical_before_final_evaluation": [
                "FSO communication experiment (real or physically modeled channel).",
                "Actual packet transmission/receiver evaluation.",
                "Communication-savings measurement tied to real or physically modeled transmission, not just decision-label counts.",
            ],
            "useful_but_optional": [
                "XJTU-SY raw data acquisition (would add 15 trajectories across 3 operating conditions, replicating the IMS finding with more statistical power).",
                "Additional noise/robustness experiments.",
                "Broader continual-learning validation using a genuinely unseen condition, not one already in the CNN's trained label space.",
                "An improved uncertainty estimator (MC Dropout, ensembles, conformal prediction), since entropy-based uncertainty has been near-zero on every model tested so far.",
            ],
        },
        "recommended_next_task": "FSO communication experiment: replace the constant Communication Cost placeholder with a real or physically modeled channel, and measure communication savings against a naive transmit-everything baseline -- the explicit purpose this analysis was scoped to prepare for.",
        "source_artifacts": [
            "results/tables/cnn_evaluation_summary.csv", "results/tables/voi_integration_summary.csv",
            "results/tables/voi_decision_distribution.csv", "results/tables/voi_factor_dominance.csv",
            "results/tables/paderborn_cnn_evaluation_summary.csv", "results/tables/paderborn_voi_integration_summary.csv",
            "results/tables/paderborn_voi_decision_distribution.csv", "results/tables/paderborn_voi_factor_dominance.csv",
            "results/tables/ims_temporal_progression_summary.csv", "results/tables/ims_failure_proximity_correlation.csv",
            "data/processed/ims/ims_temporal_experiment_manifest.json", "data/processed/paderborn/paderborn_experiment_manifest.json",
            "data/processed/xjtu/xjtu_dataset_summary.json", "data/processed/mimii_dg/mimii_dg_dataset_summary.json",
            "results/continual/task25_cwru_continual_experiment.json",
            "docs/voi_calibration_report.md", "docs/calibrated_voi_validation.md",
        ],
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    return result


if __name__ == "__main__":
    run_cross_dataset_analysis()
