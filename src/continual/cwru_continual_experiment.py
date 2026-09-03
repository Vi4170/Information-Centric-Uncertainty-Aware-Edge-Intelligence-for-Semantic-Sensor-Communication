from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import pandas as pd

from src.cnn.config import EMBEDDING_DIM
from src.cnn.model import extract_embeddings, predict_classes
from src.continual.adaptation_buffer import AdaptationBuffer, AdaptationRecord, LabelStatus, check_no_overlap
from src.continual.admission_controller import AdmissionResult, GatedPrototypeAdmissionController
from src.continual.cnn_head_adaptation import CandidateAdaptationResult, evaluate_candidate_regression, select_rehearsal_samples, train_candidate_head
from src.continual.condition_monitor import ConditionMonitor, ConditionMonitorResult, ConditionShiftStatus
from src.continual.config import RANDOM_SEED
from src.continual.model_registry import ModelRegistry
from src.continual.novelty_reference import NoveltyReference
from src.continual.safety_regression_gate import GateDecision, GateReport, SafetyRegressionGate
from src.cwru_pipeline.config import BASELINE_FAULT_CLASSES
from src.evaluation.cnn_evaluation import evaluate_classifier
from src.novelty.novelty import DistanceNoveltyDetector

KNOWN_CONDITION_CLASS: int = 0
NEW_CONDITION_CLASS: int = 1
REHEARSAL_K_PER_CONDITION: int = 50
HEAD_ADAPTATION_EPOCHS: int = 5
PROBE_SAMPLE_SIZE: int = 5

DATASET_NAME: str = "cwru"
CNN_MODEL_PATH: str = os.path.join("models", "cwru_cnn_baseline.keras")
CWRU_DATA_PATH: str = os.path.join("data", "processed", "cwru", "cwru_dataset_v1.npz")
CWRU_METADATA_PATH: str = os.path.join("data", "processed", "cwru", "cwru_metadata.csv")
RESULTS_JSON_PATH: str = os.path.join("results", "continual", "task25_cwru_continual_experiment.json")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _gate_report_to_dict(report: GateReport) -> Dict[str, Any]:
    return {
        "condition_id": report.condition_id,
        "decision": report.decision.value,
        "reasons": list(report.reasons),
        "safety": {
            "passed": report.safety.passed,
            "n_observations": report.safety.n_observations,
            "sufficient_observation_count": report.safety.sufficient_observation_count,
            "sustained_fraction": report.safety.sustained_fraction,
            "sustained_evidence_ok": report.safety.sustained_evidence_ok,
            "all_splits_permitted": report.safety.all_splits_permitted,
            "confirmed_count": report.safety.confirmed_count,
            "pseudo_count": report.safety.pseudo_count,
            "distinguishable_from_existing": report.safety.distinguishable_from_existing,
            "nearest_existing_prototype_id": report.safety.nearest_existing_prototype_id,
            "nearest_existing_distance": report.safety.nearest_existing_distance,
            "failed_checks": list(report.safety.failed_checks),
        },
        "regression": {
            "evaluated": report.regression.evaluated,
            "valid": report.regression.valid,
            "per_condition_regression": dict(report.regression.per_condition_regression),
            "worst_condition_id": report.regression.worst_condition_id,
            "worst_regression": report.regression.worst_regression,
            "failed_checks": list(report.regression.failed_checks),
        },
    }


def _admission_result_to_dict(admission: AdmissionResult) -> Dict[str, Any]:
    return {
        "condition_id": admission.condition_id,
        "decision": admission.decision.value,
        "prototype_added": admission.prototype_added,
        "prototype_id": admission.prototype_id,
        "reference_version_before": admission.reference_version_before,
        "reference_version_after": admission.reference_version_after,
        "n_candidate_observations": admission.n_candidate_observations,
        "gate_report": _gate_report_to_dict(admission.gate_report),
    }


def _condition_name(class_id: int, class_names: Dict[int, str]) -> str:
    return class_names.get(class_id, str(class_id))


def _present_condition(
    controller: GatedPrototypeAdmissionController,
    condition_id: str,
    indices: Sequence[int],
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    novelty: np.ndarray,
    predicted_class: np.ndarray,
) -> Tuple[List[ConditionMonitorResult], List[AdaptationRecord]]:
    monitor_results: List[ConditionMonitorResult] = []
    records: List[AdaptationRecord] = []
    for i in indices:
        row = meta.iloc[i]
        observation_id = str(row["observation_id"])
        source_recording_id = str(row["file_id"])
        label = int(row["fault_label"])
        window_index = int(row["window_index"])
        monitor_result = controller.observe(
            observation_id=observation_id,
            embedding=embeddings[i],
            novelty=float(novelty[i]),
            predicted_class=int(predicted_class[i]),
            condition_id=condition_id,
            dataset=DATASET_NAME,
            split="train",
            source_recording_id=source_recording_id,
            label_status=LabelStatus.CONFIRMED,
            label=label,
            window_index=window_index,
        )
        monitor_results.append(monitor_result)
        records.append(
            AdaptationRecord(
                observation_id=observation_id,
                dataset=DATASET_NAME,
                split="train",
                source_recording_id=source_recording_id,
                label_status=LabelStatus.CONFIRMED,
                label=label,
                window_index=window_index,
            )
        )
    return monitor_results, records


def _probe_novelty(reference: NoveltyReference, observation_ids: Sequence[str], embeddings: np.ndarray) -> List[Dict[str, Any]]:
    probe = []
    for obs_id, embedding in zip(observation_ids, embeddings):
        prototype, distance = reference.nearest_prototype(embedding)
        probe.append({"observation_id": obs_id, "nearest_prototype_id": prototype.prototype_id, "distance": float(distance)})
    return probe


def run_experiment(
    active_model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_meta: pd.DataFrame,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_meta: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_meta: pd.DataFrame,
    embeddings_train: np.ndarray,
    novelty_train: np.ndarray,
    predicted_train: np.ndarray,
    registry_dir: str,
    known_condition_class: int = KNOWN_CONDITION_CLASS,
    new_condition_class: int = NEW_CONDITION_CLASS,
    seed: int = RANDOM_SEED,
    rehearsal_k_per_condition: int = REHEARSAL_K_PER_CONDITION,
    head_epochs: int = HEAD_ADAPTATION_EPOCHS,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    class_names = class_names or {}
    known_name = _condition_name(known_condition_class, class_names)
    new_name = _condition_name(new_condition_class, class_names)
    num_classes = int(max(int(y_train.max()), int(y_val.max()), int(y_test.max()))) + 1

    known_train_idx = np.flatnonzero(y_train == known_condition_class).tolist()
    new_train_idx = np.flatnonzero(y_train == new_condition_class).tolist()
    known_val_idx = np.flatnonzero(y_val == known_condition_class).tolist()
    new_val_idx = np.flatnonzero(y_val == new_condition_class).tolist()

    embedding_dim = int(embeddings_train.shape[1])
    reference = NoveltyReference(embedding_dim=embedding_dim)
    known_prototype_id = f"known_condition_{known_condition_class}"
    reference.add_prototype(
        prototype_id=known_prototype_id,
        embeddings=embeddings_train[known_train_idx],
        source_dataset=DATASET_NAME,
        source_condition=known_name,
        source_split="train",
    )
    initial_reference_version = reference.version
    initial_prototype_count = len(reference.prototypes)
    b_absent_from_initial_reference = reference.prototype_ids() == (known_prototype_id,)

    reference_novelty_mean = float(np.mean(novelty_train[known_train_idx]))
    reference_novelty_std = float(np.std(novelty_train[known_train_idx]))
    reference_class_distribution = {c: (1.0 if c == known_condition_class else 0.0) for c in range(num_classes)}

    monitor = ConditionMonitor(
        reference_novelty_mean=reference_novelty_mean,
        reference_novelty_std=reference_novelty_std,
        reference_class_distribution=reference_class_distribution,
        num_classes=num_classes,
    )
    buffer = AdaptationBuffer()
    gate = SafetyRegressionGate()
    controller = GatedPrototypeAdmissionController(monitor, buffer, gate, reference)

    known_condition_id = "known_condition_baseline"
    new_condition_id = "candidate_new_condition"

    known_train_recordings = sorted({str(v) for v in train_meta.iloc[known_train_idx]["file_id"]})
    new_train_recordings = sorted({str(v) for v in train_meta.iloc[new_train_idx]["file_id"]})
    known_val_recordings = sorted({str(v) for v in val_meta.iloc[known_val_idx]["file_id"]})
    new_val_recordings = sorted({str(v) for v in val_meta.iloc[new_val_idx]["file_id"]})

    _, known_train_records = _present_condition(
        controller, known_condition_id, known_train_idx, train_meta, embeddings_train, novelty_train, predicted_train
    )
    new_monitor_results, new_train_records = _present_condition(
        controller, new_condition_id, new_train_idx, train_meta, embeddings_train, novelty_train, predicted_train
    )

    status_counts = dict(Counter(r.status.value for r in new_monitor_results))
    sustained_shift_count = status_counts.get(ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT.value, 0)
    sustained_fraction_observed = (sustained_shift_count / len(new_monitor_results)) if new_monitor_results else 0.0

    probe_positions = new_train_idx[:PROBE_SAMPLE_SIZE]
    probe_observation_ids = [str(train_meta.iloc[i]["observation_id"]) for i in probe_positions]
    probe_X = X_train[probe_positions]
    probe_embeddings = extract_embeddings(active_model, probe_X)
    novelty_before_admission = _probe_novelty(reference, probe_observation_ids, probe_embeddings)

    new_prototype_id = f"new_condition_{new_condition_class}"
    admission = controller.attempt_admission(
        condition_id=new_condition_id,
        prototype_id=new_prototype_id,
        source_dataset=DATASET_NAME,
        source_condition=new_name,
    )

    novelty_after_admission = _probe_novelty(reference, probe_observation_ids, probe_embeddings)

    registry = ModelRegistry(registry_dir=registry_dir)
    model_version_before = registry.register_initial_version(
        model=active_model,
        dataset=DATASET_NAME,
        condition_id="baseline",
        novelty_reference_version=initial_reference_version,
    )
    active_weights_before = [w.copy() for w in active_model.get_weights()]

    test_observation_ids = tuple(str(v) for v in test_meta["observation_id"])

    result: Dict[str, Any] = {
        "experiment_configuration": {
            "dataset": DATASET_NAME,
            "known_condition_class": known_condition_class,
            "known_condition_name": known_name,
            "new_condition_class": new_condition_class,
            "new_condition_name": new_name,
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
            "rehearsal_k_per_condition": rehearsal_k_per_condition,
            "head_adaptation_epochs": head_epochs,
            "condition_monitor_window_size": monitor.window_size,
            "condition_monitor_novelty_k": monitor.novelty_k,
            "condition_monitor_novelty_fraction_threshold": monitor.novelty_fraction_threshold,
            "condition_monitor_psi_threshold": monitor.psi_threshold,
            "gate_min_observation_count": gate.config.min_observation_count,
            "gate_min_sustained_fraction": gate.config.min_sustained_fraction,
            "gate_max_acceptable_regression": gate.config.max_acceptable_regression,
            "gate_review_regression_threshold": gate.config.review_regression_threshold,
        },
        "random_seed": seed,
        "selected_recordings": {
            "known_condition_train": known_train_recordings,
            "new_condition_train": new_train_recordings,
            "known_condition_val": known_val_recordings,
            "new_condition_val": new_val_recordings,
        },
        "adaptation_ids": {
            "known_condition_train_observation_ids": [str(train_meta.iloc[i]["observation_id"]) for i in known_train_idx],
            "new_condition_train_observation_ids": [str(train_meta.iloc[i]["observation_id"]) for i in new_train_idx],
        },
        "test_ids": {
            "n_test_observations": len(test_observation_ids),
            "test_observation_ids": list(test_observation_ids),
        },
        "initial_reference_version": initial_reference_version,
        "initial_prototype_count": initial_prototype_count,
        "b_absent_from_initial_reference": b_absent_from_initial_reference,
        "detection_result": {
            "n_observations_presented": len(new_monitor_results),
            "status_counts": status_counts,
            "sustained_fraction_observed": sustained_fraction_observed,
            "reference_novelty_mean": reference_novelty_mean,
            "reference_novelty_std": reference_novelty_std,
            "novelty_threshold": monitor.novelty_threshold,
        },
        "admission_result": _admission_result_to_dict(admission),
        "reference_version_transition": {
            "before": admission.reference_version_before,
            "after": admission.reference_version_after,
        },
        "novelty_before_admission": novelty_before_admission,
        "novelty_after_admission": novelty_after_admission,
        "model_version_before": model_version_before,
    }

    if admission.decision != GateDecision.ACCEPT or not admission.prototype_added:
        result["candidate_result"] = None
        result["regression_result"] = None
        result["activation_result"] = {
            "activated": False,
            "model_version_after": model_version_before,
            "reason": f"prototype_admission_decision_was_{admission.decision.value}_not_accept",
        }
        result["model_version_after"] = model_version_before
        result["cnn_performance_before_after"] = None
        result["post_activation_check"] = None
        result["active_model_untouched"] = True
    else:
        rehearsal_pool_records = known_train_records
        rehearsal_pool_X = X_train[known_train_idx]
        rehearsal_selection = select_rehearsal_samples(
            reference=reference,
            model=active_model,
            pool_records=rehearsal_pool_records,
            pool_X=rehearsal_pool_X,
            k_per_condition=rehearsal_k_per_condition,
        )
        rehearsal_local_indices = list(rehearsal_selection.condition_id_to_indices.get(known_prototype_id, ()))
        rehearsal_records = [rehearsal_pool_records[i] for i in rehearsal_local_indices]
        if rehearsal_local_indices:
            X_rehearsal = rehearsal_pool_X[rehearsal_local_indices]
        else:
            X_rehearsal = np.empty((0,) + X_train.shape[1:], dtype=X_train.dtype)
        y_rehearsal = np.array([r.label for r in rehearsal_records], dtype=y_train.dtype)

        X_new_train = X_train[new_train_idx]
        y_new_train = y_train[new_train_idx]

        train_records_head = tuple(new_train_records) + tuple(rehearsal_records)
        X_train_head = np.concatenate([X_new_train, X_rehearsal], axis=0)
        y_train_head = np.concatenate([y_new_train, y_rehearsal], axis=0)

        val_all_idx = sorted(known_val_idx + new_val_idx)
        val_records: List[AdaptationRecord] = []
        for i in val_all_idx:
            row = val_meta.iloc[i]
            record = buffer.add(
                observation_id=str(row["observation_id"]),
                dataset=DATASET_NAME,
                split="val",
                source_recording_id=str(row["file_id"]),
                label_status=LabelStatus.CONFIRMED,
                label=int(row["fault_label"]),
                window_index=int(row["window_index"]),
            )
            val_records.append(record)
        X_val_head = X_val[val_all_idx]
        y_val_head = y_val[val_all_idx]

        buffer.verify_no_test_leakage(test_observation_ids)
        leakage_overlap = check_no_overlap(buffer.observation_ids(), test_observation_ids)

        result["adaptation_ids"]["rehearsal_observation_ids"] = [r.observation_id for r in rehearsal_records]
        result["adaptation_ids"]["head_training_observation_ids"] = [r.observation_id for r in train_records_head]
        result["adaptation_ids"]["val_observation_ids"] = [r.observation_id for r in val_records]
        result["leakage_verification"] = {
            "verified": True,
            "buffer_test_overlap_count": len(leakage_overlap),
            "method": "AdaptationBuffer.verify_no_test_leakage + check_no_overlap",
        }

        candidate_result = train_candidate_head(
            active_model=active_model,
            gate=gate,
            condition_id=new_condition_id,
            train_records=train_records_head,
            X_train=X_train_head,
            y_train=y_train_head,
            val_records=tuple(val_records),
            X_val=X_val_head,
            y_val=y_val_head,
            condition_monitor_results=tuple(new_monitor_results),
            epochs=head_epochs,
            seed=seed,
            class_names=class_names,
        )

        active_weights_after = [w.copy() for w in active_model.get_weights()]
        active_model_untouched = all(
            np.array_equal(before, after) for before, after in zip(active_weights_before, active_weights_after)
        )

        regression_report = evaluate_candidate_regression(
            gate=gate,
            result=candidate_result,
            train_records=train_records_head,
            condition_monitor_results=tuple(new_monitor_results),
        )

        candidate_id = "candidate_head_adapt_1"
        registry.persist_candidate(
            candidate_id=candidate_id,
            candidate_model=candidate_result.candidate_model,
            parent_version=model_version_before,
            dataset=DATASET_NAME,
            condition_id=new_condition_id,
            adaptation_result=candidate_result,
            novelty_reference_version=reference.version,
        )

        if regression_report.decision == GateDecision.ACCEPT:
            model_version_after = registry.activate_candidate(candidate_id, regression_report)
            activated = True
            activation_reason = "regression_gate_accept"
        else:
            model_version_after = model_version_before
            activated = False
            activation_reason = f"regression_gate_decision_was_{regression_report.decision.value}_not_accept"
            assert registry.get_active_version() == model_version_before

        post_activation_check: Optional[Dict[str, Any]] = None
        if activated:
            new_active_model = registry.load_active_model()
            probe_embeddings_after_activation = extract_embeddings(new_active_model, probe_X)
            embedding_invariant = bool(np.array_equal(probe_embeddings, probe_embeddings_after_activation))
            predicted_after_activation = predict_classes(new_active_model, probe_X).tolist()
            nearest_prototype_after_activation = _probe_novelty(
                reference, probe_observation_ids, probe_embeddings_after_activation
            )
            post_activation_check = {
                "embedding_invariant": embedding_invariant,
                "predicted_classes": [int(v) for v in predicted_after_activation],
                "nearest_prototype": nearest_prototype_after_activation,
            }

        result["candidate_result"] = {
            "candidate_id": candidate_id,
            "condition_id": candidate_result.condition_id,
            "n_training_samples": candidate_result.n_training_samples,
            "n_validation_samples": candidate_result.n_validation_samples,
            "overall_accuracy_active": candidate_result.overall_accuracy_active,
            "overall_accuracy_candidate": candidate_result.overall_accuracy_candidate,
            "per_condition_accuracy_active": dict(candidate_result.per_condition_accuracy_active),
            "per_condition_accuracy_candidate": dict(candidate_result.per_condition_accuracy_candidate),
            "all_backbone_frozen": candidate_result.freeze_report.all_backbone_frozen,
            "backbone_trainable_param_count": candidate_result.freeze_report.backbone_trainable_param_count,
            "head_trainable_param_count": candidate_result.freeze_report.head_trainable_param_count,
        }
        result["regression_result"] = _gate_report_to_dict(regression_report)
        result["activation_result"] = {
            "activated": activated,
            "model_version_after": model_version_after,
            "reason": activation_reason,
        }
        result["model_version_after"] = model_version_after
        result["cnn_performance_before_after"] = {
            "known_condition_name": known_name,
            "new_condition_name": new_name,
            "per_condition_accuracy_active": dict(candidate_result.per_condition_accuracy_active),
            "per_condition_accuracy_candidate": dict(candidate_result.per_condition_accuracy_candidate),
            "overall_accuracy_active": candidate_result.overall_accuracy_active,
            "overall_accuracy_candidate": candidate_result.overall_accuracy_candidate,
        }
        result["post_activation_check"] = post_activation_check
        result["active_model_untouched"] = active_model_untouched

    final_active_model = registry.load_active_model()
    y_test_pred_baseline = predict_classes(active_model, X_test)
    y_test_pred_final = predict_classes(final_active_model, X_test)
    baseline_eval = evaluate_classifier(y_test, y_test_pred_baseline, num_classes=num_classes)
    final_eval = evaluate_classifier(y_test, y_test_pred_final, num_classes=num_classes)

    result["post_hoc_test_metrics"] = {
        "n_test_samples": int(len(y_test)),
        "baseline_model_accuracy": baseline_eval.accuracy,
        "baseline_model_per_class_metrics": baseline_eval.per_class_metrics,
        "final_active_model_accuracy": final_eval.accuracy,
        "final_active_model_per_class_metrics": final_eval.per_class_metrics,
    }

    return _to_jsonable(result)


def run_cwru_experiment(
    model_path: str = CNN_MODEL_PATH,
    data_path: str = CWRU_DATA_PATH,
    metadata_path: str = CWRU_METADATA_PATH,
    registry_dir: Optional[str] = None,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    if registry_dir is None:
        raise ValueError("registry_dir must be supplied explicitly (never defaults to a shared/production path)")

    active_model = keras.models.load_model(model_path)
    data = np.load(data_path)
    meta = pd.read_csv(metadata_path)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    train_meta = meta[meta["split"] == "train"].reset_index(drop=True)
    val_meta = meta[meta["split"] == "val"].reset_index(drop=True)
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)

    embeddings_train = extract_embeddings(active_model, X_train)
    detector = DistanceNoveltyDetector(reference_class=KNOWN_CONDITION_CLASS, embedding_dim=EMBEDDING_DIM)
    detector.fit(embeddings_train, y_train)
    novelty_train = detector.score(embeddings_train)
    predicted_train = predict_classes(active_model, X_train)

    return run_experiment(
        active_model=active_model,
        X_train=X_train,
        y_train=y_train,
        train_meta=train_meta,
        X_val=X_val,
        y_val=y_val,
        val_meta=val_meta,
        X_test=X_test,
        y_test=y_test,
        test_meta=test_meta,
        embeddings_train=embeddings_train,
        novelty_train=novelty_train,
        predicted_train=predicted_train,
        registry_dir=registry_dir,
        seed=seed,
        class_names=BASELINE_FAULT_CLASSES,
    )


def save_result_json(result: Dict[str, Any], path: str = RESULTS_JSON_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory(prefix="cwru_continual_experiment_registry_") as tmp_registry_dir:
        experiment_result = run_cwru_experiment(registry_dir=tmp_registry_dir)
    save_result_json(experiment_result)
    print(f"Saved Task 25 experiment result to: {RESULTS_JSON_PATH}")
