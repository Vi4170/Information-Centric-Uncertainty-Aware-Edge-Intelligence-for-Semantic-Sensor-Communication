"""Continual-learning infrastructure.

Phase 1 (Task 17): adaptation buffer.
Phase 2 (Task 18): read-only condition monitor.
Phase 3 (Task 19): versioned multi-prototype novelty reference.
Phase 4A (Task 20): safety + regression gate.
Phase 4B (Task 21): gated prototype admission controller.
Phase 4C (Task 23): leakage-safe CNN head-only adaptation.
"""

from src.continual.adaptation_buffer import AdaptationBuffer, AdaptationRecord, LabelStatus, check_no_overlap
from src.continual.condition_monitor import ConditionMonitor, ConditionMonitorResult, ConditionShiftStatus, population_stability_index
from src.continual.novelty_reference import NoveltyReference, Prototype
from src.continual.safety_regression_gate import (
    GateDecision,
    GateReport,
    RegressionCheckReport,
    SafetyCheckReport,
    SafetyRegressionGate,
    SafetyRegressionGateConfig,
)
from src.continual.admission_controller import AdmissionResult, GatedPrototypeAdmissionController
from src.continual.cnn_head_adaptation import (
    BACKBONE_LAYER_NAMES,
    HEAD_LAYER_NAMES,
    CandidateAdaptationResult,
    LayerFreezeReport,
    RehearsalSelection,
    clone_model_with_weights,
    evaluate_candidate_regression,
    freeze_backbone,
    select_rehearsal_samples,
    train_candidate_head,
)

__all__ = [
    "AdaptationBuffer",
    "AdaptationRecord",
    "LabelStatus",
    "check_no_overlap",
    "ConditionMonitor",
    "ConditionMonitorResult",
    "ConditionShiftStatus",
    "population_stability_index",
    "NoveltyReference",
    "Prototype",
    "GateDecision",
    "GateReport",
    "RegressionCheckReport",
    "SafetyCheckReport",
    "SafetyRegressionGate",
    "SafetyRegressionGateConfig",
    "AdmissionResult",
    "GatedPrototypeAdmissionController",
    "BACKBONE_LAYER_NAMES",
    "HEAD_LAYER_NAMES",
    "CandidateAdaptationResult",
    "LayerFreezeReport",
    "RehearsalSelection",
    "clone_model_with_weights",
    "evaluate_candidate_regression",
    "freeze_backbone",
    "select_rehearsal_samples",
    "train_candidate_head",
]
