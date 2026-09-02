"""Unit test suite for the Gated Prototype Admission Controller (Task 21,
continual-learning Phase 4B).

Uses small, deterministic synthetic scenarios throughout to prove the
wiring between the four independently-tested components (Tasks 17-20) is
correct and safe. One lightweight test checks compatibility with real
CWRU provenance fields -- it does not claim CWRU demonstrates genuine
continual learning or operating-condition drift, and no threshold in this
module or its tests was tuned against CWRU.
"""

import os
import unittest

import numpy as np
import pandas as pd

from src.continual.adaptation_buffer import AdaptationBuffer, LabelStatus
from src.continual.admission_controller import GatedPrototypeAdmissionController
from src.continual.condition_monitor import ConditionMonitor
from src.continual.novelty_reference import NoveltyReference
from src.continual.safety_regression_gate import GateDecision, SafetyRegressionGate, SafetyRegressionGateConfig

EMBEDDING_DIM = 3
REF_NOVELTY_MEAN = 0.1
REF_NOVELTY_STD = 0.05  # -> novelty_threshold = 0.2
REF_CLASS_DIST = {0: 0.5, 1: 0.5}
NUM_CLASSES = 2
WINDOW_SIZE = 5

# Small deterministic offsets so a "candidate B" cluster isn't a single
# repeated point, without perturbing its mean noticeably.
OFFSETS = [np.array([0.0, 0.0, 0.0]), np.array([0.01, 0.0, 0.0]), np.array([-0.01, 0.0, 0.0])]


def make_components(gate_config=None):
    monitor = ConditionMonitor(
        reference_novelty_mean=REF_NOVELTY_MEAN,
        reference_novelty_std=REF_NOVELTY_STD,
        reference_class_distribution=REF_CLASS_DIST,
        num_classes=NUM_CLASSES,
        window_size=WINDOW_SIZE,
    )
    buffer = AdaptationBuffer()
    gate = SafetyRegressionGate(gate_config or SafetyRegressionGateConfig(min_observation_count=20, min_distinguishability_distance=1.0))
    reference = NoveltyReference(embedding_dim=EMBEDDING_DIM)
    reference.add_prototype(
        "A",
        np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]),
        source_dataset="synthetic",
        source_condition="condition_a",
        source_split="train",
    )
    controller = GatedPrototypeAdmissionController(monitor, buffer, gate, reference)
    return controller, monitor, buffer, gate, reference


def feed_shifted_condition(controller, condition_id, n=25, dataset="synthetic", recording_id="rec_b", split="train", label_status=LabelStatus.PSEUDO, label=None):
    """Feed n observations that will register as a sustained novelty +
    class-distribution shift relative to REF_NOVELTY_MEAN/STD and
    REF_CLASS_DIST, clustered near [10, 10, 10] in embedding space.
    """
    results = []
    for i in range(n):
        embedding = np.array([10.0, 10.0, 10.0]) + OFFSETS[i % len(OFFSETS)]
        result = controller.observe(
            observation_id=f"{condition_id}_obs_{i}",
            embedding=embedding,
            novelty=0.9,
            predicted_class=1,
            condition_id=condition_id,
            dataset=dataset,
            split=split,
            source_recording_id=recording_id,
            label_status=label_status,
            label=label,
        )
        results.append(result)
    return results


def feed_stable_condition(controller, condition_id, n=25, recording_id="rec_stable"):
    """Feed n observations that look exactly like the reference (no shift)."""
    for i in range(n):
        embedding = np.array([0.0, 0.0, 0.0]) + OFFSETS[i % len(OFFSETS)]
        controller.observe(
            observation_id=f"{condition_id}_obs_{i}",
            embedding=embedding,
            novelty=REF_NOVELTY_MEAN,
            predicted_class=i % 2,  # matches the 0.5/0.5 reference distribution
            condition_id=condition_id,
            dataset="synthetic",
            split="train",
            source_recording_id=recording_id,
            label_status=LabelStatus.PSEUDO,
        )


class TestGatedPrototypeAdmissionController(unittest.TestCase):
    """End-to-end and boundary tests for GatedPrototypeAdmissionController."""

    def test_01_end_to_end_accept_prototype_added_and_a_unchanged(self):
        """1. Full positive scenario: known A, new B, sustained shift, gated ACCEPT, B becomes known, A unaffected."""
        controller, monitor, buffer, gate, reference = make_components()
        embedding_a_probe = np.array([0.0, 0.0, 0.0])
        embedding_b_probe = np.array([10.0, 10.0, 10.0])

        proto_before, dist_before = reference.nearest_prototype(embedding_a_probe)
        self.assertEqual(proto_before.prototype_id, "A")

        feed_shifted_condition(controller, "candidate_b", n=25)
        result = controller.attempt_admission(
            condition_id="candidate_b", prototype_id="B", source_dataset="synthetic", source_condition="condition_b"
        )

        self.assertEqual(result.decision, GateDecision.ACCEPT)
        self.assertTrue(result.prototype_added)
        self.assertEqual(result.prototype_id, "B")
        self.assertEqual(result.reference_version_before, 1)
        self.assertEqual(result.reference_version_after, 2)
        self.assertEqual(reference.version, 2)

        # B is now recognized.
        proto_for_b, dist_for_b = reference.nearest_prototype(embedding_b_probe)
        self.assertEqual(proto_for_b.prototype_id, "B")
        self.assertLess(dist_for_b, 0.1)

        # A is completely unaffected.
        proto_for_a, dist_for_a = reference.nearest_prototype(embedding_a_probe)
        self.assertEqual(proto_for_a.prototype_id, "A")
        self.assertEqual(dist_for_a, dist_before)

    def test_02_reject_no_prototype_added(self):
        """2. Test that unsustained (no real shift) evidence is REJECTed and adds no prototype."""
        controller, monitor, buffer, gate, reference = make_components()
        feed_stable_condition(controller, "candidate_x", n=25)

        result = controller.attempt_admission("candidate_x", "X", "synthetic", "condition_x")
        self.assertEqual(result.decision, GateDecision.REJECT)
        self.assertFalse(result.prototype_added)
        self.assertIsNone(result.prototype_id)
        self.assertEqual(reference.version, 1)
        self.assertEqual(result.reference_version_before, result.reference_version_after)

    def test_03_review_no_prototype_added(self):
        """3. Test that borderline regression evidence yields REVIEW and adds no prototype."""
        controller, monitor, buffer, gate, reference = make_components()
        feed_shifted_condition(controller, "candidate_b", n=25)

        result = controller.attempt_admission(
            "candidate_b", "B", "synthetic", "condition_b",
            baseline_metrics={"condition_a": 0.900},
            candidate_metrics={"condition_a": 0.885},  # regression 0.015 -> REVIEW (default thresholds 0.01/0.02)
        )
        self.assertEqual(result.decision, GateDecision.REVIEW)
        self.assertFalse(result.prototype_added)
        self.assertEqual(reference.version, 1)

    def test_04_version_increments_exactly_once(self):
        """4. Test that a successful admission increments version by exactly 1, and does not re-increment."""
        controller, monitor, buffer, gate, reference = make_components()
        feed_shifted_condition(controller, "candidate_b", n=25)
        result = controller.attempt_admission("candidate_b", "B", "synthetic", "condition_b")
        self.assertEqual(result.reference_version_after, 2)

        # The candidate's evidence was consumed on success; a second
        # immediate attempt for the same (now-empty) condition_id must not
        # add another prototype or increment the version again.
        result2 = controller.attempt_admission("candidate_b", "B2", "synthetic", "condition_b")
        self.assertFalse(result2.prototype_added)
        self.assertEqual(reference.version, 2)

    def test_05_existing_prototypes_unchanged_after_admission(self):
        """5. Test that prototype A's centroid is byte-identical before and after B's admission."""
        controller, monitor, buffer, gate, reference = make_components()
        centroid_a_before = reference.get_prototype("A").centroid.copy()

        feed_shifted_condition(controller, "candidate_b", n=25)
        controller.attempt_admission("candidate_b", "B", "synthetic", "condition_b")

        np.testing.assert_array_equal(reference.get_prototype("A").centroid, centroid_a_before)

    def test_06_duplicate_known_condition_not_added(self):
        """6. Test that a candidate too close to an existing prototype is rejected (no unnecessary duplicate)."""
        controller, monitor, buffer, gate, reference = make_components()

        # Looks like a shift (high novelty, diverging class distribution)
        # but its embedding is essentially the SAME as existing prototype A.
        for i in range(25):
            embedding = np.array([0.0, 0.0, 0.0]) + OFFSETS[i % len(OFFSETS)]
            controller.observe(
                observation_id=f"dup_obs_{i}",
                embedding=embedding,
                novelty=0.9,
                predicted_class=1,
                condition_id="candidate_dup",
                dataset="synthetic",
                split="train",
                source_recording_id="rec_dup",
                label_status=LabelStatus.PSEUDO,
            )

        result = controller.attempt_admission("candidate_dup", "A_dup", "synthetic", "condition_a_again")
        self.assertEqual(result.decision, GateDecision.REJECT)
        self.assertFalse(result.prototype_added)
        self.assertFalse(result.gate_report.safety.distinguishable_from_existing)
        self.assertEqual(reference.version, 1)
        self.assertEqual(reference.prototype_ids(), ("A",))

    def test_07_insufficient_candidate_evidence(self):
        """7. Test that too few observations (below the gate's minimum) results in REJECT."""
        controller, monitor, buffer, gate, reference = make_components()
        feed_shifted_condition(controller, "candidate_small", n=3)
        result = controller.attempt_admission("candidate_small", "S", "synthetic", "condition_s")
        self.assertEqual(result.decision, GateDecision.REJECT)
        self.assertFalse(result.prototype_added)
        self.assertEqual(reference.version, 1)

    def test_08_single_observation_never_admitted(self):
        """8. Critical safety boundary: a single observation must never become a prototype."""
        controller, monitor, buffer, gate, reference = make_components()
        controller.observe(
            observation_id="only_one",
            embedding=np.array([10.0, 10.0, 10.0]),
            novelty=0.9,
            predicted_class=1,
            condition_id="candidate_one",
            dataset="synthetic",
            split="train",
            source_recording_id="rec_one",
            label_status=LabelStatus.CONFIRMED,
            label=1,
        )
        result = controller.attempt_admission("candidate_one", "ONE", "synthetic", "condition_one")
        self.assertEqual(result.decision, GateDecision.REJECT)
        self.assertFalse(result.prototype_added)
        self.assertEqual(reference.version, 1)

    def test_09_test_split_rejected_at_observe_time(self):
        """9. Test that test-split observations are rejected before reaching the monitor or buffer at all."""
        controller, monitor, buffer, gate, reference = make_components()
        with self.assertRaises(ValueError):
            controller.observe(
                observation_id="test_obs",
                embedding=np.array([10.0, 10.0, 10.0]),
                novelty=0.9,
                predicted_class=1,
                condition_id="candidate_leak",
                dataset="synthetic",
                split="test",
                source_recording_id="rec_leak",
                label_status=LabelStatus.PSEUDO,
            )
        self.assertEqual(controller.candidate_size("candidate_leak"), 0)
        self.assertEqual(len(buffer), 0)
        self.assertFalse(monitor.observe(REF_NOVELTY_MEAN, 0).has_sufficient_history)  # window still empty (1st obs)

    def test_10_mixed_permitted_and_test_evidence_cannot_form(self):
        """10. Test that a sequence mixing permitted and test-split calls never lets test data enter the candidate."""
        controller, monitor, buffer, gate, reference = make_components()
        feed_shifted_condition(controller, "candidate_mixed", n=10)
        with self.assertRaises(ValueError):
            controller.observe(
                observation_id="mixed_test_obs",
                embedding=np.array([10.0, 10.0, 10.0]),
                novelty=0.9,
                predicted_class=1,
                condition_id="candidate_mixed",
                dataset="synthetic",
                split="test",
                source_recording_id="rec_b",
                label_status=LabelStatus.PSEUDO,
            )
        # The 10 legitimate observations remain tracked; the test-split one never entered.
        self.assertEqual(controller.candidate_size("candidate_mixed"), 10)
        for record in buffer.get_by_recording("rec_b"):
            self.assertNotEqual(record.split, "test")

    def test_11_missing_provenance_rejected(self):
        """11. Test that invalid provenance (empty dataset) is rejected via AdaptationBuffer's own validation."""
        controller, monitor, buffer, gate, reference = make_components()
        with self.assertRaises(ValueError):
            controller.observe(
                observation_id="bad_prov",
                embedding=np.array([10.0, 10.0, 10.0]),
                novelty=0.9,
                predicted_class=1,
                condition_id="candidate_bad",
                dataset="",  # invalid
                split="train",
                source_recording_id="rec_bad",
                label_status=LabelStatus.PSEUDO,
            )
        self.assertEqual(controller.candidate_size("candidate_bad"), 0)

    def test_12_pseudo_vs_confirmed_evidence_distinguishable(self):
        """12. Test that pseudo vs confirmed label accounting is correctly reflected in the gate report."""
        controller, monitor, buffer, gate, reference = make_components()
        feed_shifted_condition(controller, "candidate_pseudo", n=25, label_status=LabelStatus.PSEUDO)
        result_pseudo = controller.attempt_admission("candidate_pseudo", "P", "synthetic", "condition_p")
        self.assertEqual(result_pseudo.gate_report.safety.confirmed_count, 0)
        self.assertEqual(result_pseudo.gate_report.safety.pseudo_count, 25)
        self.assertEqual(result_pseudo.decision, GateDecision.ACCEPT)

        controller2, *_rest = make_components()
        feed_shifted_condition(controller2, "candidate_confirmed", n=25, label_status=LabelStatus.CONFIRMED, label=1, recording_id="rec_c")
        result_confirmed = controller2.attempt_admission("candidate_confirmed", "C", "synthetic", "condition_c")
        self.assertEqual(result_confirmed.gate_report.safety.confirmed_count, 25)
        self.assertEqual(result_confirmed.gate_report.safety.pseudo_count, 0)

    def test_13_deterministic_candidate_centroid_and_result(self):
        """13. Test that two independently built controllers given identical input produce identical outcomes."""
        controller_a, *_ = make_components()
        controller_b, *_ = make_components()

        feed_shifted_condition(controller_a, "candidate_b", n=25)
        feed_shifted_condition(controller_b, "candidate_b", n=25)

        result_a = controller_a.attempt_admission("candidate_b", "B", "synthetic", "condition_b")
        result_b = controller_b.attempt_admission("candidate_b", "B", "synthetic", "condition_b")

        self.assertEqual(result_a, result_b)
        np.testing.assert_array_equal(
            controller_a.reference.get_prototype("B").centroid,
            controller_b.reference.get_prototype("B").centroid,
        )

    def test_14_candidate_condition_consistency_across_ids(self):
        """14. Test that evidence tracked under different condition_ids never cross-contaminates."""
        controller, *_ = make_components()
        feed_shifted_condition(controller, "candidate_b", n=7, recording_id="rec_b")
        feed_shifted_condition(controller, "candidate_c", n=11, recording_id="rec_c")

        self.assertEqual(controller.candidate_size("candidate_b"), 7)
        self.assertEqual(controller.candidate_size("candidate_c"), 11)

        result_b = controller.attempt_admission("candidate_b", "B", "synthetic", "condition_b")
        self.assertEqual(result_b.n_candidate_observations, 7)
        # candidate_c's evidence must be untouched by attempting admission for candidate_b.
        self.assertEqual(controller.candidate_size("candidate_c"), 11)

    def test_15_no_mutation_on_rejected_or_reviewed_candidates(self):
        """15. Test that a REJECT/REVIEW attempt leaves tracked evidence and the reference completely unchanged."""
        controller, monitor, buffer, gate, reference = make_components()
        feed_stable_condition(controller, "candidate_x", n=25)
        size_before = controller.candidate_size("candidate_x")
        version_before = reference.version
        ids_before = reference.prototype_ids()

        result = controller.attempt_admission("candidate_x", "X", "synthetic", "condition_x")
        self.assertEqual(result.decision, GateDecision.REJECT)

        self.assertEqual(controller.candidate_size("candidate_x"), size_before)  # evidence preserved for retry
        self.assertEqual(reference.version, version_before)
        self.assertEqual(reference.prototype_ids(), ids_before)

    def test_16_real_cwru_provenance_compatibility(self):
        """16. Test compatibility with real CWRU provenance fields (dataset id, real file_id, real split values).

        Uses synthetic embeddings/novelty (this test is about provenance
        plumbing, not re-validating the CNN/novelty pipeline, which
        Tasks 13-18 already cover). Does not claim CWRU demonstrates
        genuine continual learning, and no threshold here was tuned
        against CWRU.
        """
        metadata_path = os.path.join("data", "processed", "cwru", "cwru_metadata.csv")
        if not os.path.exists(metadata_path):
            self.skipTest("Processed CWRU metadata not available in this environment")

        meta = pd.read_csv(metadata_path)
        train_rows = meta[meta["split"] == "train"].head(25)
        test_rows = meta[meta["split"] == "test"].head(1)

        controller, monitor, buffer, gate, reference = make_components()
        for i, (_, row) in enumerate(train_rows.iterrows()):
            embedding = np.array([10.0, 10.0, 10.0]) + OFFSETS[i % len(OFFSETS)]
            controller.observe(
                observation_id=str(row["observation_id"]),
                embedding=embedding,
                novelty=0.9,
                predicted_class=1,
                condition_id="cwru_candidate",
                dataset="cwru",
                split="train",
                source_recording_id=str(row["file_id"]),
                label_status=LabelStatus.CONFIRMED,
                label=int(row["fault_label"]),
            )
        self.assertEqual(controller.candidate_size("cwru_candidate"), len(train_rows))

        test_row = test_rows.iloc[0]
        with self.assertRaises(ValueError):
            controller.observe(
                observation_id=str(test_row["observation_id"]),
                embedding=np.array([10.0, 10.0, 10.0]),
                novelty=0.9,
                predicted_class=1,
                condition_id="cwru_candidate",
                dataset="cwru",
                split="test",
                source_recording_id=str(test_row["file_id"]),
                label_status=LabelStatus.CONFIRMED,
                label=int(test_row["fault_label"]),
            )
        self.assertEqual(controller.candidate_size("cwru_candidate"), len(train_rows))  # unchanged by the rejected call


if __name__ == "__main__":
    unittest.main()
