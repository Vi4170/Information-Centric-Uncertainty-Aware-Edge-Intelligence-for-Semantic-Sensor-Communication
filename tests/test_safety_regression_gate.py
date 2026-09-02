"""Unit test suite for the Safety + Regression Gate (Task 20, continual-learning Phase 4A).

Uses small, deterministic synthetic evidence throughout -- no CWRU data,
no CNN training, no real adaptation. AdaptationRecord instances are
constructed directly (bypassing AdaptationBuffer's own validation) in
several tests specifically to prove the gate re-validates provenance
itself rather than trusting the caller.
"""

import unittest

import numpy as np

from src.continual.adaptation_buffer import AdaptationRecord, LabelStatus
from src.continual.condition_monitor import ConditionMonitorResult, ConditionShiftStatus
from src.continual.novelty_reference import NoveltyReference
from src.continual.safety_regression_gate import (
    GateDecision,
    SafetyRegressionGate,
    SafetyRegressionGateConfig,
)


def make_records(n, split="train", label_status=LabelStatus.PSEUDO, label=None, recording_id="rec1", dataset="synthetic"):
    return [
        AdaptationRecord(
            observation_id=f"obs_{i}",
            dataset=dataset,
            split=split,
            source_recording_id=recording_id,
            label_status=label_status,
            label=label,
        )
        for i in range(n)
    ]


def make_cmr(status: ConditionShiftStatus) -> ConditionMonitorResult:
    return ConditionMonitorResult(
        has_sufficient_history=True,
        window_size_used=30,
        window_size_required=30,
        novelty_window_mean=0.5,
        novelty_reference_mean=0.1,
        novelty_reference_std=0.05,
        novelty_threshold=0.2,
        novelty_fraction_above_threshold=1.0,
        novelty_shift_detected=(status == ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT),
        class_distribution_psi=0.5,
        class_distribution_shift_detected=(status == ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT),
        window_class_counts={0: 30},
        status=status,
    )


SUSTAINED_EVIDENCE = tuple(make_cmr(ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT) for _ in range(10))
UNSUSTAINED_EVIDENCE = tuple(
    make_cmr(ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT if i < 2 else ConditionShiftStatus.STABLE) for i in range(5)
)


class TestSafetyRegressionGate(unittest.TestCase):
    """Test suite for SafetyRegressionGate."""

    def test_01_valid_acceptance(self):
        """1. Test that sufficient, sustained, valid evidence with no regression evidence is ACCEPTed."""
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            condition_id="cond_b",
            observations=make_records(30),
            condition_monitor_results=SUSTAINED_EVIDENCE,
        )
        self.assertEqual(report.decision, GateDecision.ACCEPT)
        self.assertTrue(report.safety.passed)
        self.assertFalse(report.regression.evaluated)

    def test_02_insufficient_evidence_rejected(self):
        """2. Test that too few observations results in REJECT (one of the two acceptable outcomes)."""
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            condition_id="cond_b",
            observations=make_records(5),
            condition_monitor_results=SUSTAINED_EVIDENCE,
        )
        self.assertIn(report.decision, (GateDecision.REJECT, GateDecision.REVIEW))
        self.assertFalse(report.safety.sufficient_observation_count)
        self.assertIn("insufficient_observation_count", report.safety.failed_checks)

    def test_03_test_split_rejected(self):
        """3. Test that test-split evidence is rejected outright."""
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            condition_id="cond_b",
            observations=make_records(30, split="test"),
            condition_monitor_results=SUSTAINED_EVIDENCE,
        )
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertFalse(report.safety.all_splits_permitted)
        self.assertIn("test", report.safety.offending_splits)

    def test_04_mixed_provenance_rejected(self):
        """4. Test that a mix of permitted and test-split evidence is rejected wholesale."""
        gate = SafetyRegressionGate()
        mixed = make_records(29, split="train") + make_records(1, split="test")
        report = gate.evaluate(
            condition_id="cond_b",
            observations=mixed,
            condition_monitor_results=SUSTAINED_EVIDENCE,
        )
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertIn("forbidden_or_unrecognized_split_present", report.safety.failed_checks)

    def test_05_missing_invalid_provenance_rejected(self):
        """5. Test that malformed AdaptationRecord fields are caught even though
        AdaptationRecord's own constructor performs no validation -- the gate
        must not rely solely on the caller (or AdaptationBuffer) for this.
        """
        gate = SafetyRegressionGate()
        bad_records = make_records(29) + [
            AdaptationRecord(
                observation_id="",  # empty id -- would be rejected by AdaptationBuffer.add(), but this
                dataset="synthetic",  # record was constructed directly, bypassing that validation.
                split="train",
                source_recording_id="rec1",
                label_status=LabelStatus.PSEUDO,
            )
        ]
        report = gate.evaluate("cond_b", bad_records, SUSTAINED_EVIDENCE)
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertFalse(report.safety.all_provenance_valid)
        self.assertIn("invalid_provenance", report.safety.failed_checks)

        contradictory_records = make_records(29) + [
            AdaptationRecord(
                observation_id="obs_contradiction",
                dataset="synthetic",
                split="train",
                source_recording_id="rec1",
                label_status=LabelStatus.CONFIRMED,
                label=None,  # CONFIRMED with no label value is a contradiction
            )
        ]
        report2 = gate.evaluate("cond_b", contradictory_records, SUSTAINED_EVIDENCE)
        self.assertEqual(report2.decision, GateDecision.REJECT)
        self.assertIn("invalid_provenance", report2.safety.failed_checks)

    def test_06_pseudo_label_evidence_not_silently_confirmed(self):
        """6. Test that pseudo-label evidence is reported distinctly, never conflated with confirmed."""
        gate = SafetyRegressionGate()  # default: require_confirmed_labels=False
        report = gate.evaluate("cond_b", make_records(30, label_status=LabelStatus.PSEUDO), SUSTAINED_EVIDENCE)
        self.assertEqual(report.safety.confirmed_count, 0)
        self.assertEqual(report.safety.pseudo_count, 30)
        self.assertEqual(report.decision, GateDecision.ACCEPT)  # pseudo is legitimate for prototype-only evidence

        strict_gate = SafetyRegressionGate(SafetyRegressionGateConfig(require_confirmed_labels=True))
        strict_report = strict_gate.evaluate("cond_b", make_records(30, label_status=LabelStatus.PSEUDO), SUSTAINED_EVIDENCE)
        self.assertEqual(strict_report.decision, GateDecision.REJECT)
        self.assertIn("confirmed_labels_required_but_pseudo_present", strict_report.safety.failed_checks)

    def test_07_insufficient_sustained_evidence(self):
        """7. Test that isolated/unsustained evidence and no evidence at all are both rejected."""
        gate = SafetyRegressionGate()
        report = gate.evaluate("cond_b", make_records(30), UNSUSTAINED_EVIDENCE)  # 2/5 = 0.4 < 0.8
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertFalse(report.safety.sustained_evidence_ok)
        self.assertAlmostEqual(report.safety.sustained_fraction, 0.4)

        no_evidence_report = gate.evaluate("cond_b", make_records(30), condition_monitor_results=())
        self.assertEqual(no_evidence_report.decision, GateDecision.REJECT)
        self.assertIn("no_condition_monitor_evidence_supplied", no_evidence_report.safety.failed_checks)

    def test_08_distinguishability_from_existing_prototypes(self):
        """8. Test that a candidate too close to an existing prototype is rejected, a distant one is accepted."""
        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("known", np.array([[0.0, 0.0], [0.0, 0.0]]), "synthetic", "known_cond", "train")

        strict_gate = SafetyRegressionGate(SafetyRegressionGateConfig(min_distinguishability_distance=1.0))

        close_report = strict_gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE, candidate_embedding=np.array([0.1, 0.0]), existing_reference=ref
        )
        self.assertEqual(close_report.decision, GateDecision.REJECT)
        self.assertFalse(close_report.safety.distinguishable_from_existing)
        self.assertEqual(close_report.safety.nearest_existing_prototype_id, "known")

        far_report = strict_gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE, candidate_embedding=np.array([50.0, 50.0]), existing_reference=ref
        )
        self.assertEqual(far_report.decision, GateDecision.ACCEPT)
        self.assertTrue(far_report.safety.distinguishable_from_existing)

    def test_09_missing_candidate_embedding_fails_closed(self):
        """9. Test that a reference with known prototypes but no supplied candidate_embedding fails closed."""
        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("known", np.array([[0.0, 0.0], [0.0, 0.0]]), "synthetic", "known_cond", "train")
        gate = SafetyRegressionGate()
        report = gate.evaluate("cond_b", make_records(30), SUSTAINED_EVIDENCE, candidate_embedding=None, existing_reference=ref)
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertIn("no_candidate_embedding_supplied_for_distinguishability_check", report.safety.failed_checks)

    def test_10_empty_or_absent_reference_trivially_distinguishable(self):
        """10. Test that a None reference, or a freshly constructed empty reference, never blocks acceptance."""
        gate = SafetyRegressionGate()
        report_none = gate.evaluate("cond_b", make_records(30), SUSTAINED_EVIDENCE, existing_reference=None)
        self.assertTrue(report_none.safety.distinguishable_from_existing)

        empty_ref = NoveltyReference(embedding_dim=2)
        report_empty = gate.evaluate("cond_b", make_records(30), SUSTAINED_EVIDENCE, existing_reference=empty_ref)
        self.assertTrue(report_empty.safety.distinguishable_from_existing)
        self.assertEqual(report_empty.decision, GateDecision.ACCEPT)

    def test_11_no_regression_accepted(self):
        """11. Test that negligible/no regression is accepted when safety passes."""
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE,
            baseline_metrics={"normal": 0.90, "fault_a": 0.90},
            candidate_metrics={"normal": 0.90, "fault_a": 0.91},
        )
        self.assertEqual(report.decision, GateDecision.ACCEPT)
        self.assertTrue(report.regression.evaluated)
        self.assertTrue(report.regression.valid)

    def test_12_excessive_regression_rejected(self):
        """12. Test that regression beyond the max threshold is rejected."""
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE,
            baseline_metrics={"normal": 0.90, "fault_a": 0.90},
            candidate_metrics={"normal": 0.90, "fault_a": 0.85},  # regression = 0.05 > 0.02 default max
        )
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertIn("excessive_regression", report.regression.failed_checks)
        self.assertEqual(report.regression.worst_condition_id, "fault_a")

    def test_13_borderline_regression_review(self):
        """13. Test that regression between the review and max thresholds yields REVIEW."""
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE,
            baseline_metrics={"normal": 0.900},
            candidate_metrics={"normal": 0.885},  # regression = 0.015, between 0.01 and 0.02 defaults
        )
        self.assertEqual(report.decision, GateDecision.REVIEW)
        self.assertIn("borderline_regression", report.regression.failed_checks)

    def test_14_improvement_accepted(self):
        """14. Test that an improved candidate is accepted."""
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE,
            baseline_metrics={"normal": 0.90, "fault_a": 0.90},
            candidate_metrics={"normal": 0.95, "fault_a": 0.95},
        )
        self.assertEqual(report.decision, GateDecision.ACCEPT)
        self.assertLess(report.regression.worst_regression, 0)

    def test_15_regression_on_previously_known_condition_detected(self):
        """15. Test that a regression hidden in ONE previously known condition is still caught,
        even though other conditions improved (i.e. averaging cannot mask it).
        """
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE,
            baseline_metrics={"normal": 0.99, "fault_a": 0.95, "fault_b": 0.95},
            candidate_metrics={"normal": 0.99, "fault_a": 0.90, "fault_b": 0.97},
        )
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertEqual(report.regression.worst_condition_id, "fault_a")
        self.assertAlmostEqual(report.regression.worst_regression, 0.05, places=6)

    def test_16_missing_baseline_metrics_rejected(self):
        """16. Test that supplying candidate metrics without baseline metrics is not accepted."""
        gate = SafetyRegressionGate()
        report = gate.evaluate("cond_b", make_records(30), SUSTAINED_EVIDENCE, candidate_metrics={"normal": 0.9})
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertIn("missing_baseline_metrics", report.regression.failed_checks)

        report_empty_baseline = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE, baseline_metrics={}, candidate_metrics={"normal": 0.9}
        )
        self.assertEqual(report_empty_baseline.decision, GateDecision.REJECT)

    def test_17_malformed_metrics_rejected(self):
        """17. Test that non-finite and non-numeric metric values are rejected."""
        gate = SafetyRegressionGate()
        nan_report = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE,
            baseline_metrics={"normal": 0.9}, candidate_metrics={"normal": float("nan")},
        )
        self.assertEqual(nan_report.decision, GateDecision.REJECT)
        self.assertTrue(any("non_finite_metric" in f for f in nan_report.regression.failed_checks))

        non_numeric_report = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE,
            baseline_metrics={"normal": 0.9}, candidate_metrics={"normal": "high"},
        )
        self.assertEqual(non_numeric_report.decision, GateDecision.REJECT)
        self.assertTrue(any("non_numeric_metric" in f for f in non_numeric_report.regression.failed_checks))

    def test_18_key_mismatch_rejected(self):
        """18. Test that mismatched baseline/candidate condition keys are rejected."""
        gate = SafetyRegressionGate()
        report = gate.evaluate(
            "cond_b", make_records(30), SUSTAINED_EVIDENCE,
            baseline_metrics={"a": 0.9}, candidate_metrics={"b": 0.9},
        )
        self.assertEqual(report.decision, GateDecision.REJECT)
        self.assertIn("baseline_candidate_condition_key_mismatch", report.regression.failed_checks)

    def test_19_deterministic_results(self):
        """19. Test that two identical evaluate() calls produce an identical report."""
        gate = SafetyRegressionGate()
        kwargs = dict(
            observations=make_records(30),
            condition_monitor_results=SUSTAINED_EVIDENCE,
            baseline_metrics={"normal": 0.9, "fault_a": 0.9},
            candidate_metrics={"normal": 0.9, "fault_a": 0.89},
        )
        report_a = gate.evaluate("cond_b", **kwargs)
        report_b = gate.evaluate("cond_b", **kwargs)
        self.assertEqual(report_a, report_b)

    def test_20_no_mutation_of_supplied_inputs(self):
        """20. Test that the gate never mutates the observations list, embedding array, or metrics dicts."""
        gate = SafetyRegressionGate()
        records = make_records(30)
        records_snapshot = list(records)
        embedding = np.array([1.0, 2.0])
        embedding_snapshot = embedding.copy()
        baseline = {"normal": 0.9}
        candidate = {"normal": 0.9}

        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("known", np.array([[0.0, 0.0], [0.0, 0.0]]), "synthetic", "known_cond", "train")

        gate.evaluate(
            "cond_b", records, SUSTAINED_EVIDENCE,
            candidate_embedding=embedding, existing_reference=ref,
            baseline_metrics=baseline, candidate_metrics=candidate,
        )

        self.assertEqual(records, records_snapshot)
        np.testing.assert_array_equal(embedding, embedding_snapshot)
        self.assertEqual(baseline, {"normal": 0.9})
        self.assertEqual(candidate, {"normal": 0.9})

    def test_21_no_mutation_of_existing_novelty_reference(self):
        """21. Test that evaluate() never calls add_prototype() -- the reference is completely unaffected."""
        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("a", np.array([[0.0, 0.0], [0.0, 0.0]]), "synthetic", "cond_a", "train")
        ref.add_prototype("b", np.array([[5.0, 5.0], [5.0, 5.0]]), "synthetic", "cond_b", "train")

        version_before = ref.version
        ids_before = ref.prototype_ids()
        centroids_before = {p.prototype_id: p.centroid.copy() for p in ref.prototypes}

        gate = SafetyRegressionGate()
        gate.evaluate("cond_c", make_records(30), SUSTAINED_EVIDENCE, candidate_embedding=np.array([100.0, 100.0]), existing_reference=ref)

        self.assertEqual(ref.version, version_before)
        self.assertEqual(ref.prototype_ids(), ids_before)
        for pid, centroid in centroids_before.items():
            np.testing.assert_array_equal(ref.get_prototype(pid).centroid, centroid)


if __name__ == "__main__":
    unittest.main()
