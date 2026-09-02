"""Continual-learning infrastructure. Phase 1 (Task 17): adaptation buffer only."""

from src.continual.adaptation_buffer import AdaptationBuffer, AdaptationRecord, LabelStatus, check_no_overlap

__all__ = ["AdaptationBuffer", "AdaptationRecord", "LabelStatus", "check_no_overlap"]
