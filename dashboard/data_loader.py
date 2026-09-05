"""Read-only accessors over the project's stored result artifacts, for the results dashboard.

Every function here only reads files already produced by the project's pipelines
(or imports the actual configuration constants used by those pipelines). Nothing
here recomputes a research metric or invents a value that is not already stored
on disk. When an artifact does not exist, functions report that explicitly
instead of fabricating a placeholder result.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"

STATUS_AVAILABLE = "available"
STATUS_NOT_PERFORMED = "not_yet_performed"
STATUS_MISSING = "missing"


def _path(*parts) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _exists(*parts) -> bool:
    return _path(*parts).exists()


def _load_json(*parts):
    p = _path(*parts)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(*parts):
    p = _path(*parts)
    if not p.exists():
        return None
    return pd.read_csv(p)


def figure_path(*parts):
    p = _path(*parts)
    return str(p) if p.exists() else None


def get_pipeline_stage_status():
    """Artifact-existence status for every stage in the Dataset -> ... -> Continual Learning chain."""
    return {
        "CWRU preprocessing": _exists("data", "processed", "cwru", "summary.json"),
        "IMS preprocessing": _exists("data", "processed", "ims", "ims_dataset_summary.json"),
        "Paderborn preprocessing": _exists("data", "processed", "paderborn", "paderborn_dataset_summary.json"),
        "CNN training/evaluation": _exists("models", "cwru_cnn_baseline.keras")
        and _exists("results", "tables", "cnn_evaluation_summary.csv"),
        "Novelty estimation": _exists("results", "tables", "novelty_scores_summary.csv"),
        "Uncertainty estimation": _exists("results", "tables", "uncertainty_scores_summary.csv"),
        "Task relevance / temporal / communication cost (formula modules)": _exists(
            "src", "relevance", "relevance.py"
        )
        and _exists("src", "temporal", "temporal.py")
        and _exists("src", "communication", "cost.py"),
        "VoI synthetic diagnostics": _exists("results", "tables", "voi_decision_distribution.csv"),
        "VoI CWRU integration + calibration": _exists("results", "tables", "voi_integration_summary.csv"),
        "Continual-learning experiment (Task 25)": _exists(
            "results", "continual", "task25_cwru_continual_experiment.json"
        ),
    }


def get_cwru_dataset_info():
    summary = _load_json("data", "processed", "cwru", "summary.json")
    if summary is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "experiment_status": "Preprocessing complete; CNN, novelty, uncertainty, and VoI experiments performed.",
        "summary": summary,
        "split_description": (
            "File-level split before windowing (26/7/7 of 40 source .mat files for train/val/test), "
            "so every window in a split comes only from source recordings assigned to that split -- "
            "no window ever crosses a split boundary."
        ),
    }


def get_ims_dataset_info():
    summary = _load_json("data", "processed", "ims", "ims_dataset_summary.json")
    if summary is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "experiment_status": "Dataset integrated -- CNN/novelty/uncertainty/VoI experiments not yet performed.",
        "summary": summary,
        "split_description": summary.get("split_methodology"),
    }


def get_paderborn_dataset_info():
    summary = _load_json("data", "processed", "paderborn", "paderborn_dataset_summary.json")
    if summary is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "experiment_status": "Dataset integrated -- CNN/novelty/uncertainty/VoI experiments not yet performed.",
        "summary": summary,
        "split_description": summary.get("split_methodology"),
    }


def get_canonical_cnn_results():
    report = _load_csv("results", "tables", "cnn_classification_report.csv")
    summary = _load_csv("results", "tables", "cnn_evaluation_summary.csv")
    history = _load_csv("results", "tables", "cnn_training_history.csv")
    if report is None or summary is None:
        return {"status": STATUS_MISSING}

    model_info = {"total_params": None, "layers": None}
    model_path = _path("models", "cwru_cnn_baseline.keras")
    if model_path.exists():
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            from tensorflow import keras

            model = keras.models.load_model(str(model_path), compile=False)
            model_info["total_params"] = int(model.count_params())
            model_info["layers"] = [
                {"name": layer.name, "type": layer.__class__.__name__} for layer in model.layers
            ]
        except Exception:
            model_info["total_params"] = None
            model_info["layers"] = None

    return {
        "status": STATUS_AVAILABLE,
        "model_path": str(model_path) if model_path.exists() else None,
        "model_info": model_info,
        "classification_report": report,
        "evaluation_summary": summary,
        "training_history": history,
        "confusion_matrix_figure": figure_path("results", "figures", "cnn_confusion_matrix.png"),
        "training_curves_figure": figure_path("results", "figures", "cnn_training_curves.png"),
        "class_performance_figure": figure_path("results", "figures", "cnn_class_performance.png"),
        "note": (
            "Trained/evaluated on the canonical CWRU 2048-sample-window dataset "
            "(406 test windows, drawn from 7 held-out source files). 100% test accuracy "
            "reflects the ease of separating these four static operating conditions on a "
            "clean, high-SNR benchmark recording -- not evidence that the model generalizes "
            "to unseen operating conditions, loads, or noise levels."
        ),
    }


def get_legacy_edge_cloud_results():
    """A separate, alternate CWRU CNN + VoI experimental track (src/cwru_pipeline/{models,voi}),
    evaluated on a differently-split 1043-sample test set. Distinct from the canonical
    src/cnn + src/voi pipeline; kept and shown separately so the two are never conflated."""
    baseline = _load_json("results", "cwru_pipeline", "baseline", "baseline_metrics.json")
    cloud = _load_json("results", "cwru_pipeline", "cloud", "cloud_metrics.json")
    final_voi = _load_json("results", "cwru_pipeline", "voi", "final_voi_evaluation.json")
    formal_metrics = _load_json("results", "cwru_pipeline", "voi", "formal_voi_metrics.json")
    formal_results = _load_json("results", "cwru_pipeline", "voi", "formal_voi_results.json")
    if baseline is None or cloud is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "baseline_cnn": baseline,
        "cloud_cnn": cloud,
        "final_voi_evaluation": final_voi,
        "formal_voi_metrics": formal_metrics,
        "formal_voi_results": formal_results,
        "figures": {
            "baseline_confusion_matrix": figure_path("results", "cwru_pipeline", "baseline", "confusion_matrix.png"),
            "formal_voi_pareto": figure_path("results", "cwru_pipeline", "voi", "formal_voi_pareto.png"),
            "formal_voi_transmission_analysis": figure_path(
                "results", "cwru_pipeline", "voi", "formal_voi_transmission_analysis.png"
            ),
            "formal_voi_cost_sensitivity": figure_path(
                "results", "cwru_pipeline", "voi", "formal_voi_cost_sensitivity.png"
            ),
        },
    }


def get_novelty_results():
    from src.novelty import config as novelty_config

    summary = _load_csv("results", "tables", "novelty_scores_summary.csv")
    if summary is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "config": {
            "embedding_dim": novelty_config.EMBEDDING_DIM,
            "distance_metric": novelty_config.DEFAULT_DISTANCE_METRIC,
            "reference_class": novelty_config.REFERENCE_CLASS,
        },
        "summary": summary,
        "distribution_figure": figure_path("results", "figures", "novelty_score_distribution.png"),
        "by_class_figure": figure_path("results", "figures", "novelty_by_class.png"),
        "method": (
            "DistanceNoveltyDetector: min-max-normalized Euclidean distance, in the CNN's "
            "64-D embedding space, from each observation to the centroid of the 'Normal' "
            "(reference) class computed on the train split only."
        ),
    }


def get_uncertainty_results():
    from src.uncertainty import config as uncertainty_config

    summary = _load_csv("results", "tables", "uncertainty_scores_summary.csv")
    if summary is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "config": {
            "num_classes": uncertainty_config.NUM_CLASSES,
            "prob_tolerance": uncertainty_config.PROB_TOLERANCE,
        },
        "summary": summary,
        "distribution_figure": figure_path("results", "figures", "uncertainty_score_distribution.png"),
        "by_class_figure": figure_path("results", "figures", "uncertainty_by_class.png"),
        "method": (
            "Shannon entropy of the CNN's predicted class-probability vector, normalized by "
            "log2(4) so the score lies in [0, 1]. Near-zero across all classes because the "
            "underlying CNN is close to 100% confident almost everywhere on this dataset -- "
            "flagged in the original report as a weak/near-inert signal, later reflected in "
            "the VoI calibration that down-weighted it."
        ),
    }


def get_relevance_results():
    from src.relevance import config as relevance_config

    summary = _load_csv("results", "tables", "relevance_scores_summary.csv")
    return {
        "status": STATUS_AVAILABLE if summary is not None else STATUS_NOT_PERFORMED,
        "config": {
            "strategy": relevance_config.DEFAULT_STRATEGY,
            "class_relevance_map": relevance_config.CLASS_RELEVANCE_MAP,
        },
        "summary": summary,
        "distribution_figure": figure_path("results", "figures", "relevance_score_distribution.png"),
        "method": (
            "Direct lookup of a fixed class-to-relevance value (class_mapping strategy) or a "
            "probability-weighted sum over the CNN's predicted class probabilities "
            "(probability_weighted strategy). The class_relevance_map values are hand-set "
            "design parameters, not learned or optimized from data."
        ),
    }


def get_temporal_results():
    from src.temporal import config as temporal_config

    return {
        "status": STATUS_AVAILABLE,
        "config": {
            "temporal_change_scale": temporal_config.DEFAULT_TEMPORAL_CHANGE_SCALE,
            "min_observations": temporal_config.MIN_OBSERVATIONS,
        },
        "method": (
            "T_t = clip(mean(|x_t - x_(t-1)|) / temporal_change_scale, 0, 1); the first "
            "observation in any sequence is always T=0 (no prior context). The scale constant "
            "was recalibrated from an initial 0.5 to 1.8 (the 95th percentile of train-split "
            "mean-absolute-difference) after the original value saturated the score to ~1.0 "
            "for most fault windows."
        ),
    }


def get_communication_cost_results():
    from src.communication import config as comm_config

    return {
        "status": STATUS_AVAILABLE,
        "config": {
            "weight_size": comm_config.WEIGHT_SIZE,
            "weight_time": comm_config.WEIGHT_TIME,
            "weight_bandwidth": comm_config.WEIGHT_BANDWIDTH,
            "max_payload_size_bytes": comm_config.MAX_PAYLOAD_SIZE,
            "max_transmission_time_s": comm_config.MAX_TRANSMISSION_TIME,
            "reference_bandwidth_bps": comm_config.REFERENCE_BANDWIDTH,
        },
        "method": (
            "C = weight_size * size_component + weight_time * time_component + "
            "weight_bandwidth * bandwidth_component, each component clipped to [0, 1] against "
            "the reference limits above. Weights are explicitly documented as NOT optimized. "
            "In the CWRU VoI integration this cost is effectively constant (~0.505) because "
            "every observation is the same fixed 2048-sample window under one nominal "
            "scenario -- no channel-variability model exists yet."
        ),
    }


def get_voi_engine_config():
    from src.voi.scoring import VoIWeights
    from src.voi.decision_policy import PolicyThresholds

    w = VoIWeights()
    t = PolicyThresholds()
    return {
        "weights": {
            "novelty": w.novelty,
            "uncertainty": w.uncertainty,
            "task_relevance": w.task_relevance,
            "temporal_importance": w.temporal_importance,
            "resource_cost": w.resource_cost,
        },
        "thresholds": {
            "discard_max": t.discard_max,
            "buffer_max": t.buffer_max,
            "summary_max": t.summary_max,
        },
        "formula": "VoI_raw = w_N*N + w_U*U + w_R*R + w_T*T - w_C*C, clipped to [0, 1] for the decision policy",
    }


def get_voi_synthetic_results():
    scenario = _load_csv("results", "tables", "scenario_analysis.csv")
    reachability = _load_csv("results", "tables", "decision_reachability.csv")
    weight_sens = _load_csv("results", "tables", "weight_sensitivity.csv")
    threshold_sens = _load_csv("results", "tables", "threshold_sensitivity.csv")
    v01_validation = _load_csv("results", "tables", "v01_validation_summary.csv")
    clipping = _load_csv("results", "tables", "clipping_analysis.csv")
    cost_sens = _load_csv("results", "tables", "cost_sensitivity.csv")
    summary_stats = _load_csv("results", "tables", "summary_statistics.csv")
    if scenario is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "dataset": "Synthetic (data/synthetic/synthetic_voi_dataset.csv, seed 42, 1000 observations, 6 scenarios)",
        "scenario_analysis": scenario,
        "decision_reachability": reachability,
        "weight_sensitivity": weight_sens,
        "threshold_sensitivity": threshold_sens,
        "v01_validation_summary": v01_validation,
        "clipping_analysis": clipping,
        "cost_sensitivity": cost_sens,
        "summary_statistics": summary_stats,
        "figures": {
            "scenario_decision_distribution": figure_path("results", "figures", "scenario_decision_distribution.png"),
            "weight_sensitivity": figure_path("results", "figures", "weight_sensitivity.png"),
            "voi_score_distribution": figure_path("results", "figures", "voi_score_distribution.png"),
        },
        "note": (
            "Illustrative/diagnostic only -- built from a synthetic input dataset (not real "
            "sensor data), used to validate that the VoI engine's math behaves as designed "
            "before it was ever connected to real CWRU signals."
        ),
    }


def get_voi_cwru_integration_results():
    integration_summary = _load_csv("results", "tables", "voi_integration_summary.csv")
    decision_distribution = _load_csv("results", "tables", "voi_decision_distribution.csv")
    factor_dominance = _load_csv("results", "tables", "voi_factor_dominance.csv")
    per_observation = _load_csv("results", "tables", "voi_integration_per_observation.csv")
    if integration_summary is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "dataset": "CWRU (real, test split of 406 windows unless noted otherwise)",
        "integration_summary": integration_summary,
        "decision_distribution": decision_distribution,
        "factor_dominance": factor_dominance,
        "per_observation_sample": per_observation.head(20) if per_observation is not None else None,
        "n_per_observation_rows": len(per_observation) if per_observation is not None else None,
        "figures": {
            "voi_score_distribution": figure_path("results", "figures", "voi_score_distribution.png"),
            "voi_decision_distribution": figure_path("results", "figures", "voi_decision_distribution.png"),
            "voi_factor_contribution": figure_path("results", "figures", "voi_factor_contribution.png"),
            "voi_score_by_class": figure_path("results", "figures", "voi_score_by_class.png"),
        },
    }


def get_voi_calibration_results():
    decision_cmp = _load_csv("results", "tables", "calibration_validation_decision_comparison.csv")
    dominance_cmp = _load_csv("results", "tables", "calibration_validation_factor_dominance_comparison.csv")
    transmission_cmp = _load_csv("results", "tables", "calibration_validation_transmission_reduction.csv")
    score_cmp = _load_csv("results", "tables", "calibration_validation_voi_score_comparison.csv")
    reproducibility = _load_csv("results", "tables", "calibration_validation_reproducibility.csv")
    if decision_cmp is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "decision_comparison": decision_cmp,
        "factor_dominance_comparison": dominance_cmp,
        "transmission_reduction": transmission_cmp,
        "voi_score_comparison": score_cmp,
        "reproducibility": reproducibility,
        "figures": {
            "before_after_decision": figure_path("results", "figures", "voi_calibration_before_after_decision.png"),
            "before_after_distribution": figure_path(
                "results", "figures", "voi_calibration_before_after_distribution.png"
            ),
        },
        "note": (
            "Task 14 recalibrated VoI weights and the temporal-importance scale using only "
            "train/validation data; Task 15 independently re-derived the 'before' numbers from "
            "raw data on the untouched test split and found them to match exactly "
            "(reproducible, no leakage into calibration)."
        ),
    }


def get_continual_learning_results():
    result = _load_json("results", "continual", "task25_cwru_continual_experiment.json")
    if result is None:
        return {"status": STATUS_MISSING}
    return {
        "status": STATUS_AVAILABLE,
        "raw": result,
        "note": (
            "CWRU has no genuine temporal drift (each recording is one fixed operating "
            "condition), so this experiment demonstrates the continual-learning MECHANISM "
            "(condition monitoring -> gated prototype admission -> safety/regression gate -> "
            "leakage-safe CNN head adaptation -> versioned activation) working end-to-end on "
            "real data, not a genuine novel-condition detection in the field. The design docs "
            "(docs/continual_learning_design.md) explicitly recommend IMS/Paderborn as future "
            "testbeds with real temporal drift; those experiments have not been run yet."
        ),
    }


def get_test_suite_summary():
    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists():
        return {"status": STATUS_MISSING}
    test_files = sorted(p.name for p in tests_dir.glob("test_*.py"))
    return {"status": STATUS_AVAILABLE, "test_files": test_files, "n_test_files": len(test_files)}
