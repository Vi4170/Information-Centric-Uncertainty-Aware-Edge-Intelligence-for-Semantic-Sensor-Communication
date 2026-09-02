"""Continual-learning infrastructure.

Phase 1 (Task 17): adaptation buffer.
Phase 2 (Task 18): read-only condition monitor.
Phase 3 (Task 19): versioned multi-prototype novelty reference.
"""

from src.continual.adaptation_buffer import AdaptationBuffer, AdaptationRecord, LabelStatus, check_no_overlap
from src.continual.condition_monitor import ConditionMonitor, ConditionMonitorResult, ConditionShiftStatus, population_stability_index
from src.continual.novelty_reference import NoveltyReference, Prototype

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
]
