"""Safety + Regression Gate (Task 20 -- Continual Learning Design, Phase 4A).

Implements ONLY the decision infrastructure described in
docs/continual_learning_design.md Sections 4.1 and 4.3: given evidence
about a candidate condition, decide whether a FUTURE controller may
commit an adaptation update (a new prototype, and/or eventually a CNN
fine-tune). This module never commits anything itself.

    candidate evidence
            |
      safety checks   (provenance, split, sustainment, distinguishability, ...)
            |
    regression checks (baseline vs candidate performance, per known condition)
            |
       GateDecision
       REJECT  -> caller must not commit
       REVIEW  -> no automatic update; needs human judgement
       ACCEPT  -> the supplied evidence satisfies current gate criteria;
                  a future controller MAY commit the proposed update.
                  ACCEPT never means an update has already happened.

This module is read-only with respect to every other continual-learning
component: it only calls NoveltyReference's existing, non-mutating query
methods (nearest_prototype / distance_report), never add_prototype(). It
does not touch AdaptationBuffer at all -- it accepts plain AdaptationRecord
objects a caller has already retrieved (e.g. via
AdaptationBuffer.get_by_recording()), so it depends on that module's data
type without needing to import or hold a live buffer. It does not call the
CNN, retrain anything, or touch src/voi/.

Deliberately re-validates provenance/split itself (does not trust that
AdaptationRecord instances it receives came from a properly-validated
AdaptationBuffer): AdaptationRecord itself performs no validation in its
own constructor -- validation lives in AdaptationBuffer.add(), which a
caller could bypass by constructing AdaptationRecord directly. Task-set
protection must not depend on that discipline, so this gate checks split
and label/provenance validity again from scratch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.continual.adaptation_buffer import AdaptationRecord, LabelStatus
from src.continual.condition_monitor import ConditionMonitorResult, ConditionShiftStatus
from src.continual.config import ALLOWED_SPLITS
from src.continual.novelty_reference import NoveltyReference


class GateDecision(str, Enum):
    """Final decision. See module docstring for the precise meaning of ACCEPT."""

    REJECT = "reject"
    REVIEW = "review"
    ACCEPT = "accept"


@dataclass
class SafetyRegressionGateConfig:
    """Configurable thresholds.

    None of these values come from an experimental calibration study --
    they are ENGINEERING DEFAULTS chosen to be conservative and
    auditable, exactly as docs/continual_learning_design.md's own
    provisional-parameter framing already applies to src/voi/'s weights
    and thresholds. Tighten or loosen them once real multi-condition
    adaptation data exists to calibrate against (see
    docs/safety_regression_gate.md).
    """

    # Safety: minimum number of observations backing a candidate condition.
    min_observation_count: int = 30

    # Safety: minimum fraction of supplied ConditionMonitorResult
    # evaluations that must report CANDIDATE_CONDITION_SHIFT for the
    # evidence to count as "sustained" rather than an isolated anomaly.
    min_sustained_fraction: float = 0.8

    # Safety: minimum raw (unnormalized) distance the candidate's
    # representative embedding must have from every existing prototype's
    # centroid to be considered a genuinely distinguishable condition.
    # Default 0.0 is deliberately permissive (only rejects an EXACT
    # duplicate) because Task 19 established that a calibrated distance
    # scale does not yet exist for the multi-prototype reference -- see
    # docs/multi_prototype_novelty.md and docs/safety_regression_gate.md.
    min_distinguishability_distance: float = 0.0

    # Safety: if True, every observation backing the candidate must carry
    # a CONFIRMED label (appropriate when gating a future CNN fine-tune).
    # Default False, since prototype-only candidates may legitimately be
    # backed by recurring PSEUDO-labelled evidence per
    # docs/continual_learning_design.md Section 4.2.
    require_confirmed_labels: bool = False

    # Regression: per-condition performance drop above this is still
    # acceptable (no regression concern).
    review_regression_threshold: float = 0.01

    # Regression: per-condition performance drop above this is rejected outright.
    max_acceptable_regression: float = 0.02


@dataclass(frozen=True)
class SafetyCheckReport:
    """Structured, auditable result of every safety check."""

    passed: bool
    n_observations: int
    sufficient_observation_count: bool

    sustained_fraction: Optional[float]
    sustained_evidence_ok: bool

    all_splits_permitted: bool
    offending_splits: Tuple[str, ...]

    all_provenance_valid: bool
    provenance_failures: Tuple[str, ...]

    confirmed_count: int
    pseudo_count: int
    confirmed_fraction: Optional[float]
    label_status_ok: bool

    distinguishable_from_existing: bool
    nearest_existing_prototype_id: Optional[str]
    nearest_existing_distance: Optional[float]

    failed_checks: Tuple[str, ...]


@dataclass(frozen=True)
class RegressionCheckReport:
    """Structured, auditable result of the regression comparison."""

    evaluated: bool
    valid: bool
    per_condition_regression: Dict[str, float]
    worst_condition_id: Optional[str]
    worst_regression: Optional[float]
    failed_checks: Tuple[str, ...]


@dataclass(frozen=True)
class GateReport:
    """Full, auditable output of one gate evaluation."""

    condition_id: str
    decision: GateDecision
    safety: SafetyCheckReport
    regression: RegressionCheckReport
    reasons: Tuple[str, ...]


def _validate_provenance(record: AdaptationRecord) -> Optional[str]:
    """Re-validate one AdaptationRecord's provenance from scratch.

    Returns a human-readable failure string, or None if valid. Does not
    trust that the record was produced by a properly-validated
    AdaptationBuffer.add()/add_from_dataframe() call.
    """
    if not record.observation_id:
        return f"observation_id is empty ({record!r})"
    if not record.dataset:
        return f"dataset is empty for observation '{record.observation_id}'"
    if not record.source_recording_id:
        return f"source_recording_id is empty for observation '{record.observation_id}'"
    if not isinstance(record.label_status, LabelStatus):
        return f"label_status is not a LabelStatus for observation '{record.observation_id}'"
    if record.label_status is LabelStatus.CONFIRMED and record.label is None:
        return f"observation '{record.observation_id}' claims CONFIRMED status with no label value"
    return None


class SafetyRegressionGate:
    """Deterministic Safety + Regression Gate.

    Evaluates supplied evidence and returns a GateReport. Mutates nothing:
    it holds no state between calls (aside from its own config) and never
    calls any mutating method on any object it is given.
    """

    def __init__(self, config: Optional[SafetyRegressionGateConfig] = None) -> None:
        self.config = config or SafetyRegressionGateConfig()

    def evaluate(
        self,
        condition_id: str,
        observations: Sequence[AdaptationRecord],
        condition_monitor_results: Sequence[ConditionMonitorResult] = (),
        candidate_embedding: Optional[np.ndarray] = None,
        existing_reference: Optional[NoveltyReference] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        candidate_metrics: Optional[Dict[str, float]] = None,
    ) -> GateReport:
        """Evaluate one candidate condition and return a full decision report.

        Args:
            condition_id: Identifier for the candidate condition (for audit only).
            observations: The AdaptationRecord evidence backing this
                candidate (e.g. from AdaptationBuffer.get_by_recording()).
                Never mutated; never used to call anything on
                AdaptationBuffer itself.
            condition_monitor_results: ConditionMonitorResult snapshots
                gathered while observing this candidate (e.g. from
                repeated ConditionMonitor.observe() calls). Required for
                the "sustained, not an isolated anomaly" safety check --
                an empty sequence fails that check (fail-closed).
            candidate_embedding: A representative embedding for the
                candidate condition (e.g. the mean of its observations'
                CNN embeddings), used only to query
                `existing_reference.nearest_prototype()` -- never to
                create a prototype.
            existing_reference: The current NoveltyReference, queried
                read-only via nearest_prototype()/distance_report(). This
                gate NEVER calls add_prototype() on it.
            baseline_metrics / candidate_metrics: Optional per-condition
                performance metrics (e.g. {"normal": 0.98, "fault_a":
                0.95}). If neither is supplied, the regression check is
                skipped and the decision rests on safety alone. If either
                is supplied, both must be supplied, non-empty, share
                identical keys, and contain only finite numeric values --
                otherwise the candidate is rejected.

        Returns:
            GateReport with the full safety/regression breakdown and the
            final GateDecision. Nothing supplied to this method, and
            nothing it references (existing_reference, observations), is
            mutated.
        """
        safety = self._evaluate_safety(observations, condition_monitor_results, candidate_embedding, existing_reference)
        regression = self._evaluate_regression(baseline_metrics, candidate_metrics)

        decision, reasons = self._combine(safety, regression)

        return GateReport(
            condition_id=condition_id,
            decision=decision,
            safety=safety,
            regression=regression,
            reasons=tuple(reasons),
        )

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _evaluate_safety(
        self,
        observations: Sequence[AdaptationRecord],
        condition_monitor_results: Sequence[ConditionMonitorResult],
        candidate_embedding: Optional[np.ndarray],
        existing_reference: Optional[NoveltyReference],
    ) -> SafetyCheckReport:
        cfg = self.config
        failed_checks = []

        # --- sufficient observation count ---
        n_observations = len(observations)
        sufficient_observation_count = n_observations >= cfg.min_observation_count
        if not sufficient_observation_count:
            failed_checks.append("insufficient_observation_count")

        # --- source split permitted / no test-set observations ---
        # Re-validated here from scratch -- never trusts the caller.
        # (ALLOWED_SPLITS never contains FORBIDDEN_SPLIT, so a single
        # case-insensitive membership check also catches "test" in any case.)
        offending_splits = tuple(
            sorted({r.split for r in observations if str(r.split).strip().lower() not in ALLOWED_SPLITS})
        )
        all_splits_permitted = len(offending_splits) == 0
        if not all_splits_permitted:
            failed_checks.append("forbidden_or_unrecognized_split_present")

        # --- provenance validity (re-checked independently, see module docstring) ---
        provenance_failures = tuple(
            failure for failure in (_validate_provenance(r) for r in observations) if failure is not None
        )
        all_provenance_valid = len(provenance_failures) == 0
        if not all_provenance_valid:
            failed_checks.append("invalid_provenance")

        # --- confirmed vs pseudo label accounting (never conflated) ---
        confirmed_count = sum(1 for r in observations if r.label_status is LabelStatus.CONFIRMED)
        pseudo_count = sum(1 for r in observations if r.label_status is LabelStatus.PSEUDO)
        confirmed_fraction = confirmed_count / n_observations if n_observations > 0 else None
        label_status_ok = True
        if cfg.require_confirmed_labels and pseudo_count > 0:
            label_status_ok = False
            failed_checks.append("confirmed_labels_required_but_pseudo_present")

        # --- sustained evidence, not an isolated anomaly (fail-closed if no evidence given) ---
        if len(condition_monitor_results) == 0:
            sustained_fraction = None
            sustained_evidence_ok = False
            failed_checks.append("no_condition_monitor_evidence_supplied")
        else:
            sustained_fraction = sum(
                1 for r in condition_monitor_results if r.status == ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT
            ) / len(condition_monitor_results)
            sustained_evidence_ok = sustained_fraction >= cfg.min_sustained_fraction
            if not sustained_evidence_ok:
                failed_checks.append("insufficient_sustained_shift_evidence")

        # --- distinguishability from existing known prototypes (fail-closed) ---
        nearest_id: Optional[str] = None
        nearest_distance: Optional[float] = None
        if existing_reference is None or existing_reference.version == 0:
            distinguishable = True  # nothing learned yet -- nothing to be confused with
        elif candidate_embedding is None:
            distinguishable = False
            failed_checks.append("no_candidate_embedding_supplied_for_distinguishability_check")
        else:
            nearest_prototype, nearest_distance = existing_reference.nearest_prototype(candidate_embedding)
            nearest_id = nearest_prototype.prototype_id
            distinguishable = nearest_distance >= cfg.min_distinguishability_distance
            if not distinguishable:
                failed_checks.append("not_distinguishable_from_existing_prototype")

        passed = len(failed_checks) == 0

        return SafetyCheckReport(
            passed=passed,
            n_observations=n_observations,
            sufficient_observation_count=sufficient_observation_count,
            sustained_fraction=sustained_fraction,
            sustained_evidence_ok=sustained_evidence_ok,
            all_splits_permitted=all_splits_permitted,
            offending_splits=offending_splits,
            all_provenance_valid=all_provenance_valid,
            provenance_failures=provenance_failures,
            confirmed_count=confirmed_count,
            pseudo_count=pseudo_count,
            confirmed_fraction=confirmed_fraction,
            label_status_ok=label_status_ok,
            distinguishable_from_existing=distinguishable,
            nearest_existing_prototype_id=nearest_id,
            nearest_existing_distance=nearest_distance,
            failed_checks=tuple(failed_checks),
        )

    # ------------------------------------------------------------------
    # Regression checks
    # ------------------------------------------------------------------

    def _evaluate_regression(
        self,
        baseline_metrics: Optional[Dict[str, float]],
        candidate_metrics: Optional[Dict[str, float]],
    ) -> RegressionCheckReport:
        cfg = self.config
        evaluated = baseline_metrics is not None or candidate_metrics is not None

        if not evaluated:
            return RegressionCheckReport(
                evaluated=False,
                valid=True,
                per_condition_regression={},
                worst_condition_id=None,
                worst_regression=None,
                failed_checks=(),
            )

        failed_checks = []

        if not baseline_metrics:
            failed_checks.append("missing_baseline_metrics")
        if not candidate_metrics:
            failed_checks.append("missing_candidate_metrics")

        if failed_checks:
            return RegressionCheckReport(
                evaluated=True,
                valid=False,
                per_condition_regression={},
                worst_condition_id=None,
                worst_regression=None,
                failed_checks=tuple(failed_checks),
            )

        if set(baseline_metrics.keys()) != set(candidate_metrics.keys()):
            return RegressionCheckReport(
                evaluated=True,
                valid=False,
                per_condition_regression={},
                worst_condition_id=None,
                worst_regression=None,
                failed_checks=("baseline_candidate_condition_key_mismatch",),
            )

        for source_name, metrics in (("baseline_metrics", baseline_metrics), ("candidate_metrics", candidate_metrics)):
            for condition, value in metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return RegressionCheckReport(
                        evaluated=True,
                        valid=False,
                        per_condition_regression={},
                        worst_condition_id=None,
                        worst_regression=None,
                        failed_checks=(f"non_numeric_metric:{source_name}:{condition}",),
                    )
                if not math.isfinite(value):
                    return RegressionCheckReport(
                        evaluated=True,
                        valid=False,
                        per_condition_regression={},
                        worst_condition_id=None,
                        worst_regression=None,
                        failed_checks=(f"non_finite_metric:{source_name}:{condition}",),
                    )

        # Positive regression value = performance got worse (baseline - candidate).
        per_condition_regression = {
            condition: float(baseline_metrics[condition] - candidate_metrics[condition])
            for condition in baseline_metrics
        }
        worst_condition_id = max(per_condition_regression, key=lambda c: per_condition_regression[c])
        worst_regression = per_condition_regression[worst_condition_id]

        regression_failed = []
        if worst_regression > cfg.max_acceptable_regression:
            regression_failed.append("excessive_regression")
        elif worst_regression > cfg.review_regression_threshold:
            regression_failed.append("borderline_regression")

        return RegressionCheckReport(
            evaluated=True,
            valid=True,
            per_condition_regression=per_condition_regression,
            worst_condition_id=worst_condition_id,
            worst_regression=worst_regression,
            failed_checks=tuple(regression_failed),
        )

    # ------------------------------------------------------------------
    # Combination
    # ------------------------------------------------------------------

    def _combine(
        self, safety: SafetyCheckReport, regression: RegressionCheckReport
    ) -> Tuple[GateDecision, list]:
        reasons = []

        if not safety.passed:
            reasons.append(f"Safety check(s) failed: {', '.join(safety.failed_checks)}")
            return GateDecision.REJECT, reasons

        reasons.append("All safety checks passed.")

        if regression.evaluated and not regression.valid:
            reasons.append(f"Regression evidence invalid: {', '.join(regression.failed_checks)}")
            return GateDecision.REJECT, reasons

        if not regression.evaluated:
            reasons.append("No regression evidence supplied; decision based on safety alone.")
            return GateDecision.ACCEPT, reasons

        if "excessive_regression" in regression.failed_checks:
            reasons.append(
                f"Excessive regression on '{regression.worst_condition_id}': "
                f"{regression.worst_regression:.4f} > max {self.config.max_acceptable_regression}"
            )
            return GateDecision.REJECT, reasons

        if "borderline_regression" in regression.failed_checks:
            reasons.append(
                f"Borderline regression on '{regression.worst_condition_id}': "
                f"{regression.worst_regression:.4f} (between {self.config.review_regression_threshold} "
                f"and {self.config.max_acceptable_regression})"
            )
            return GateDecision.REVIEW, reasons

        reasons.append("No unacceptable regression detected.")
        return GateDecision.ACCEPT, reasons
