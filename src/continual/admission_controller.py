"""Gated Prototype Admission Controller (Task 21 -- Continual Learning
Design, Phase 4B).

The first REAL, controlled continual-learning loop: it wires together the
four independent, previously-unconnected components (Tasks 17-20) into one
pipeline, without duplicating any of their logic.

    observations
         |
    ConditionMonitor.observe()          (Task 18, unmodified)
         |
    candidate evidence
         |
    AdaptationBuffer.add()              (Task 17, unmodified)
         |
    SafetyRegressionGate.evaluate()     (Task 20, unmodified)
         |
       ACCEPT?
        /    \
      NO      YES
      |        |
    STOP   NoveltyReference.add_prototype()   (Task 19, unmodified)
                |
         new reference version

This controller holds references to already-constructed instances of all
four components (dependency injection) -- it builds none of their internal
logic itself. Its own responsibility is strictly: (1) route one observation
to the monitor and, if its provenance is valid, to the buffer; (2) track
which buffered observations/embeddings belong to which candidate
condition; (3) on an explicit `attempt_admission()` call, assemble that
evidence into the shape SafetyRegressionGate expects and invoke it;
(4) only on ACCEPT, aggregate the tracked embeddings into a centroid and
call NoveltyReference.add_prototype(); (5) report what happened.

CRITICAL SAFETY BOUNDARY: `add_prototype()` is called from exactly one
place in this file, guarded by exactly one condition:
`gate_report.decision == GateDecision.ACCEPT`. There is no other path
through this module that reaches it -- not from high novelty, not from a
ConditionMonitor shift alert alone, not from a single observation, not
from test-split data (rejected before the monitor even sees it), and not
by treating a pseudo-label as confirmed (label_status is passed through
to the gate unchanged, exactly as recorded).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.continual.adaptation_buffer import AdaptationBuffer, AdaptationRecord, LabelStatus
from src.continual.condition_monitor import ConditionMonitor, ConditionMonitorResult
from src.continual.config import ALLOWED_SPLITS
from src.continual.novelty_reference import NoveltyReference, Prototype
from src.continual.safety_regression_gate import GateDecision, GateReport, SafetyRegressionGate


def _validate_embedding_vector(embedding: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Minimal input-hygiene check (shape + finiteness) -- not a duplicate of
    NoveltyReference's centroid/distance algorithm, just fail-fast validation
    at the point of entry, matching the pattern already used throughout
    src/continual/.
    """
    if not isinstance(embedding, np.ndarray):
        raise TypeError(f"embedding must be a numpy array, got {type(embedding)}")
    if embedding.ndim != 1 or embedding.shape[0] != embedding_dim:
        raise ValueError(f"embedding must be a 1D array of length {embedding_dim}, got shape {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains NaN or Inf values")
    return embedding


@dataclass(frozen=True)
class AdmissionResult:
    """Auditable outcome of one attempt_admission() call."""

    condition_id: str
    decision: GateDecision
    gate_report: GateReport
    prototype_added: bool
    prototype_id: Optional[str]
    reference_version_before: int
    reference_version_after: int
    n_candidate_observations: int


class GatedPrototypeAdmissionController:
    """Coordinates ConditionMonitor + AdaptationBuffer + SafetyRegressionGate
    + NoveltyReference. Duplicates none of their logic; owns only the
    per-condition bookkeeping needed to assemble candidate evidence.
    """

    def __init__(
        self,
        condition_monitor: ConditionMonitor,
        buffer: AdaptationBuffer,
        gate: SafetyRegressionGate,
        reference: NoveltyReference,
    ) -> None:
        self._condition_monitor = condition_monitor
        self._buffer = buffer
        self._gate = gate
        self._reference = reference
        self._embedding_dim = reference.embedding_dim

        # Per-condition-id bookkeeping -- owned entirely by this controller.
        # AdaptationBuffer itself has no concept of "candidate condition";
        # it only stores individual observation records permanently.
        self._condition_records: Dict[str, List[AdaptationRecord]] = {}
        self._condition_embeddings: Dict[str, List[np.ndarray]] = {}
        self._condition_monitor_results: Dict[str, List[ConditionMonitorResult]] = {}

    @property
    def reference(self) -> NoveltyReference:
        """The NoveltyReference this controller may (on ACCEPT only) add prototypes to."""
        return self._reference

    def candidate_size(self, condition_id: str) -> int:
        """Number of observations currently tracked for a candidate condition."""
        return len(self._condition_records.get(condition_id, ()))

    def observe(
        self,
        observation_id: str,
        embedding: np.ndarray,
        novelty: float,
        predicted_class: int,
        condition_id: str,
        dataset: str,
        split: str,
        source_recording_id: str,
        label_status: LabelStatus,
        label: Optional[Any] = None,
        window_index: Optional[int] = None,
    ) -> ConditionMonitorResult:
        """Route one observation through monitoring and (if eligible) collection.

        Test-split observations are rejected BEFORE anything else happens
        -- not even the ConditionMonitor's rolling window is touched --
        rather than relying solely on AdaptationBuffer's own split check
        further down the call. AdaptationBuffer.add() still performs its
        own independent validation as a second layer of defense.

        Never adds a prototype. Never calls the gate. See attempt_admission().

        Args:
            observation_id, dataset, split, source_recording_id,
                label_status, label, window_index: passed straight through
                to AdaptationBuffer.add() -- see its docstring for the
                exact contract.
            embedding: 1D array of shape (embedding_dim,) -- the CNN
                embedding for this observation (src/cnn's existing
                contract; no new representation is invented here).
            novelty, predicted_class: passed straight through to
                ConditionMonitor.observe() -- see its docstring for the
                exact contract.
            condition_id: caller-assigned label identifying which
                candidate condition this observation's evidence should be
                grouped under for a later attempt_admission() call.

        Returns:
            The ConditionMonitorResult for this observation (informational
            -- the caller decides when to call attempt_admission(), this
            method never triggers it automatically).

        Raises:
            ValueError / TypeError: If the split is not permitted, the
                embedding is malformed, or AdaptationBuffer.add() rejects
                the observation (duplicate id, invalid provenance, etc.)
                -- in which case nothing is retained: the ConditionMonitor
                is not updated, and no candidate bookkeeping changes.
        """
        if str(split).strip().lower() not in ALLOWED_SPLITS:
            raise ValueError(
                f"Rejected: split='{split}' is not permitted to enter the adaptation "
                "path at all. Test-set (or any unrecognized-split) observations must "
                "never reach the Condition Monitor or Adaptation Buffer."
            )

        validated_embedding = _validate_embedding_vector(embedding, self._embedding_dim)

        # Only after provenance is confirmed permitted do we touch the monitor.
        monitor_result = self._condition_monitor.observe(novelty, predicted_class)

        # AdaptationBuffer.add() re-validates independently (duplicate ids,
        # empty fields, label_status/label consistency, split again) --
        # this is intentional defense in depth, not blind trust of the
        # check above.
        record = self._buffer.add(
            observation_id=observation_id,
            dataset=dataset,
            split=split,
            source_recording_id=source_recording_id,
            label_status=label_status,
            label=label,
            window_index=window_index,
        )

        self._condition_records.setdefault(condition_id, []).append(record)
        self._condition_embeddings.setdefault(condition_id, []).append(validated_embedding.copy())
        self._condition_monitor_results.setdefault(condition_id, []).append(monitor_result)

        return monitor_result

    def attempt_admission(
        self,
        condition_id: str,
        prototype_id: str,
        source_dataset: str,
        source_condition: str,
        baseline_metrics: Optional[Dict[str, float]] = None,
        candidate_metrics: Optional[Dict[str, float]] = None,
    ) -> AdmissionResult:
        """Explicit candidate-building + gate-evaluation step.

        This never runs automatically -- the caller decides when enough
        evidence has been gathered for `condition_id` and calls this
        method deliberately. Builds the candidate embedding as the mean
        of every embedding tracked under `condition_id` (deterministic:
        insertion order, same computation NoveltyReference.add_prototype()
        itself already performs on whatever array it's given), evaluates
        the SafetyRegressionGate, and -- ONLY on ACCEPT -- calls
        NoveltyReference.add_prototype(). On REJECT or REVIEW, the
        reference is completely untouched and the tracked evidence for
        `condition_id` is left exactly as it was, so the caller may keep
        observing and try again later.

        Args:
            condition_id: Which tracked candidate to evaluate.
            prototype_id: Unique id for the prototype IF admitted.
            source_dataset / source_condition: Provenance metadata for the
                prototype IF admitted.
            baseline_metrics / candidate_metrics: Optional regression
                evidence, passed straight through to the gate.

        Returns:
            AdmissionResult with the full gate report and whether a
            prototype was actually added.
        """
        records = tuple(self._condition_records.get(condition_id, ()))
        embeddings = self._condition_embeddings.get(condition_id, [])
        monitor_results = tuple(self._condition_monitor_results.get(condition_id, ()))

        candidate_embedding = np.mean(np.stack(embeddings), axis=0) if embeddings else None

        version_before = self._reference.version

        gate_report = self._gate.evaluate(
            condition_id=condition_id,
            observations=records,
            condition_monitor_results=monitor_results,
            candidate_embedding=candidate_embedding,
            existing_reference=self._reference,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
        )

        prototype: Optional[Prototype] = None
        if gate_report.decision == GateDecision.ACCEPT:
            # This is the ONLY call to add_prototype() in this entire
            # module, and it is reachable only through this branch.
            embeddings_array = np.stack(embeddings)
            # Candidate evidence may legitimately mix "train" and "val"
            # (the gate does not require split homogeneity -- only that
            # every split is permitted). NoveltyReference.add_prototype()
            # needs one split value for its provenance metadata; the
            # majority split among the candidate's own records is used,
            # deterministically (ties broken by first-seen order via
            # Counter.most_common()).
            majority_split = Counter(r.split for r in records).most_common(1)[0][0]
            prototype = self._reference.add_prototype(
                prototype_id=prototype_id,
                embeddings=embeddings_array,
                source_dataset=source_dataset,
                source_condition=source_condition,
                source_split=majority_split,
            )
            # This candidate's evidence has now been permanently captured
            # in the reference; clear the controller's own bookkeeping for
            # it so a later observe() call for the same condition_id starts
            # a fresh candidate rather than silently re-submitting consumed
            # evidence. AdaptationBuffer's own permanent log is untouched
            # (it has no remove operation, by design).
            self._condition_records.pop(condition_id, None)
            self._condition_embeddings.pop(condition_id, None)
            self._condition_monitor_results.pop(condition_id, None)

        return AdmissionResult(
            condition_id=condition_id,
            decision=gate_report.decision,
            gate_report=gate_report,
            prototype_added=prototype is not None,
            prototype_id=prototype.prototype_id if prototype is not None else None,
            reference_version_before=version_before,
            reference_version_after=self._reference.version,
            n_candidate_observations=len(records),
        )
