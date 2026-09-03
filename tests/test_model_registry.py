"""Unit test suite for the CNN model lifecycle / versioning registry (Task 24).

Uses one small, module-level synthetic CNN (the real src/cnn architecture)
built once in setUpClass, and a fresh temporary registry directory per
test. No expensive CWRU training is performed.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from src.cnn.model import build_baseline_cnn
from src.continual.adaptation_buffer import AdaptationRecord, LabelStatus
from src.continual.cnn_head_adaptation import clone_model_with_weights, evaluate_candidate_regression, freeze_backbone, train_candidate_head
from src.continual.condition_monitor import ConditionMonitorResult, ConditionShiftStatus
from src.continual.model_registry import ModelRegistry
from src.continual.novelty_reference import NoveltyReference
from src.continual.safety_regression_gate import (
    GateDecision,
    GateReport,
    RegressionCheckReport,
    SafetyCheckReport,
    SafetyRegressionGate,
    SafetyRegressionGateConfig,
)


class _FakeAdaptationResult:
    """Minimal stand-in for CandidateAdaptationResult -- only the fields
    persist_candidate() actually reads."""

    def __init__(self, n_train=10, n_val=5, active_acc=None, candidate_acc=None):
        self.n_training_samples = n_train
        self.n_validation_samples = n_val
        self.per_condition_accuracy_active = active_acc or {"0": 0.90}
        self.per_condition_accuracy_candidate = candidate_acc or {"0": 0.92}


def make_safety(passed=True, failed_checks=()):
    return SafetyCheckReport(
        passed=passed, n_observations=30, sufficient_observation_count=True,
        sustained_fraction=1.0, sustained_evidence_ok=True, all_splits_permitted=True,
        offending_splits=(), all_provenance_valid=True, provenance_failures=(),
        confirmed_count=30, pseudo_count=0, confirmed_fraction=1.0, label_status_ok=True,
        distinguishable_from_existing=True, nearest_existing_prototype_id=None,
        nearest_existing_distance=None, failed_checks=failed_checks,
    )


def make_regression(worst=-0.02, failed_checks=()):
    return RegressionCheckReport(
        evaluated=True, valid=True, per_condition_regression={"0": worst},
        worst_condition_id="0", worst_regression=worst, failed_checks=failed_checks,
    )


def make_gate_report(decision, condition_id="cond_b", reasons=("ok",)):
    return GateReport(
        condition_id=condition_id, decision=decision,
        safety=make_safety(passed=(decision != GateDecision.REJECT)),
        regression=make_regression(),
        reasons=reasons,
    )


class TestModelRegistry(unittest.TestCase):
    """Test suite for src/continual/model_registry.py."""

    @classmethod
    def setUpClass(cls):
        cls.base_model = build_baseline_cnn()

    def _new_registry(self, tmp_dir):
        return ModelRegistry(registry_dir=tmp_dir)

    def _make_frozen_clone(self, source_model=None):
        candidate = clone_model_with_weights(source_model or self.base_model)
        freeze_backbone(candidate)
        candidate.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return candidate

    def _bootstrapped_registry(self, tmp_dir):
        reg = self._new_registry(tmp_dir)
        reg.register_initial_version(self.base_model, dataset="synthetic", condition_id="baseline")
        return reg

    # ------------------------------------------------------------------
    # Registration / loading
    # ------------------------------------------------------------------

    def test_01_initial_model_registration(self):
        """1. Test that register_initial_version() bootstraps version 1 and activates it."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._new_registry(tmp)
            self.assertFalse(reg.has_active_version())
            version = reg.register_initial_version(self.base_model, dataset="synthetic", condition_id="baseline")
            self.assertEqual(version, 1)
            self.assertEqual(reg.get_active_version(), 1)
            self.assertEqual(reg.list_versions(), (1,))

    def test_02_active_model_loading(self):
        """2. Test that load_active_model() returns a working, weight-matching model."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            loaded = reg.load_active_model()
            for w_a, w_b in zip(self.base_model.get_weights(), loaded.get_weights()):
                np.testing.assert_array_equal(w_a, w_b)

    def test_03_double_bootstrap_rejected(self):
        """3. Test that register_initial_version() refuses to run twice."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            with self.assertRaises(ValueError):
                reg.register_initial_version(self.base_model, dataset="synthetic")

    # ------------------------------------------------------------------
    # Candidate staging / isolation
    # ------------------------------------------------------------------

    def test_04_candidate_persistence_and_isolation(self):
        """4. Test that a persisted candidate is separate from, and does not affect, the active model."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()

            reg.persist_candidate("cand1", candidate, parent_version=1, dataset="synthetic", condition_id="cond_b", adaptation_result=_FakeAdaptationResult())

            # Active pointer must still reference v1 -- persisting never activates.
            self.assertEqual(reg.get_active_version(), 1)
            self.assertEqual(reg.list_versions(), (1,))

    def test_05_candidate_independently_reloadable(self):
        """5. Test that load_candidate() returns a model whose weights match what was persisted."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            reloaded, metadata = reg.load_candidate("cand1")
            for w_a, w_b in zip(candidate.get_weights(), reloaded.get_weights()):
                np.testing.assert_array_equal(w_a, w_b)
            self.assertEqual(metadata.candidate_id, "cand1")
            self.assertIsNone(metadata.version)
            self.assertEqual(metadata.parent_version, 1)

    def test_06_duplicate_candidate_id_rejected(self):
        """6. Test that persisting the same candidate_id twice is rejected (staging is not overwritten)."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())
            with self.assertRaises(ValueError):
                reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())

    def test_07_persist_with_unknown_parent_rejected(self):
        """7. Test that persisting a candidate against a nonexistent parent_version is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            with self.assertRaises(ValueError):
                reg.persist_candidate("cand1", candidate, parent_version=99, dataset="synthetic", condition_id="cond_b", adaptation_result=_FakeAdaptationResult())

    # ------------------------------------------------------------------
    # ACCEPT-only activation
    # ------------------------------------------------------------------

    def test_08_accept_activates_candidate(self):
        """8. Test that an ACCEPT decision activates the candidate as the next version."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            new_version = reg.activate_candidate("cand1", make_gate_report(GateDecision.ACCEPT))
            self.assertEqual(new_version, 2)
            self.assertEqual(reg.get_active_version(), 2)
            self.assertEqual(reg.list_versions(), (1, 2))

    def test_09_reject_does_not_activate(self):
        """9. Test that a REJECT decision leaves the active model and version list unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            with self.assertRaises(ValueError):
                reg.activate_candidate("cand1", make_gate_report(GateDecision.REJECT))
            self.assertEqual(reg.get_active_version(), 1)
            self.assertEqual(reg.list_versions(), (1,))

    def test_10_review_does_not_activate(self):
        """10. Test that a REVIEW decision leaves the active model and version list unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            with self.assertRaises(ValueError):
                reg.activate_candidate("cand1", make_gate_report(GateDecision.REVIEW))
            self.assertEqual(reg.get_active_version(), 1)
            self.assertEqual(reg.list_versions(), (1,))

    def test_11_missing_or_malformed_decision_does_not_activate(self):
        """11. Test that a missing/None report or a malformed decision value is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            with self.assertRaises(TypeError):
                reg.activate_candidate("cand1", None)
            with self.assertRaises(TypeError):
                reg.activate_candidate("cand1", "ACCEPT")  # a plain string is not a GateReport

            report = make_gate_report(GateDecision.ACCEPT)
            malformed = GateReport(condition_id=report.condition_id, decision="accept", safety=report.safety, regression=report.regression, reasons=report.reasons)
            with self.assertRaises(TypeError):
                reg.activate_candidate("cand1", malformed)

            self.assertEqual(reg.get_active_version(), 1)

    def test_12_active_model_unchanged_before_activation(self):
        """12. Test that the active model is byte-identical before and after persisting (but not activating) a candidate."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            before = reg.load_active_model()
            weights_before = [w.copy() for w in before.get_weights()]

            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            after = reg.load_active_model()
            for w_before, w_after in zip(weights_before, after.get_weights()):
                np.testing.assert_array_equal(w_before, w_after)

    # ------------------------------------------------------------------
    # Versioning / monotonicity
    # ------------------------------------------------------------------

    def test_13_version_increments_correctly(self):
        """13. Test that successive accepted candidates increment the version by exactly 1 each time."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)

            cand_b = self._make_frozen_clone()
            reg.persist_candidate("cand_b", cand_b, 1, "synthetic", "cond_b", _FakeAdaptationResult())
            v2 = reg.activate_candidate("cand_b", make_gate_report(GateDecision.ACCEPT, condition_id="cond_b"))
            self.assertEqual(v2, 2)

            cand_c = self._make_frozen_clone()
            reg.persist_candidate("cand_c", cand_c, 2, "synthetic", "cond_c", _FakeAdaptationResult())
            v3 = reg.activate_candidate("cand_c", make_gate_report(GateDecision.ACCEPT, condition_id="cond_c"))
            self.assertEqual(v3, 3)
            self.assertEqual(reg.list_versions(), (1, 2, 3))

    def test_14_rejected_candidate_does_not_consume_version(self):
        """14. Test that a REJECTed candidate never occupies a version slot: the next
        ACCEPTed candidate still becomes v2, not v3."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)

            rejected = self._make_frozen_clone()
            reg.persist_candidate("cand_rejected", rejected, 1, "synthetic", "cond_x", _FakeAdaptationResult())
            with self.assertRaises(ValueError):
                reg.activate_candidate("cand_rejected", make_gate_report(GateDecision.REJECT))
            self.assertEqual(reg.list_versions(), (1,))

            accepted = self._make_frozen_clone()
            reg.persist_candidate("cand_accepted", accepted, 1, "synthetic", "cond_y", _FakeAdaptationResult())
            v2 = reg.activate_candidate("cand_accepted", make_gate_report(GateDecision.ACCEPT))
            self.assertEqual(v2, 2)  # not 3
            self.assertEqual(reg.list_versions(), (1, 2))

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def test_15_rollback_restores_previous_version(self):
        """15. Test the v1 -> v2 -> rollback -> v1 sequence."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())
            reg.activate_candidate("cand1", make_gate_report(GateDecision.ACCEPT))
            self.assertEqual(reg.get_active_version(), 2)

            restored = reg.rollback(1)
            self.assertEqual(restored, 1)
            self.assertEqual(reg.get_active_version(), 1)
            # Rollback must never delete anything.
            self.assertEqual(reg.list_versions(), (1, 2))

    def test_16_rollback_rejects_nonexistent_version(self):
        """16. Test that rollback() refuses a target version that was never registered."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            with self.assertRaises(ValueError):
                reg.rollback(99)
            self.assertEqual(reg.get_active_version(), 1)

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def test_17_incompatible_embedding_dim_and_architecture_rejected(self):
        """17. Test that a candidate with a different embedding dimensionality
        (and therefore different layer shapes) is rejected at activation, fail-closed."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            incompatible_model = build_baseline_cnn(embedding_dim=32)
            freeze_backbone_report_model = incompatible_model  # not frozen -- irrelevant to this check
            incompatible_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            reg.persist_candidate("cand_bad_dim", incompatible_model, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            compat = reg.check_compatibility(incompatible_model, parent_version=1)
            self.assertFalse(compat.compatible)
            self.assertFalse(compat.same_embedding_dim)
            self.assertFalse(compat.same_architecture)
            self.assertIn("embedding_dimension_mismatch", compat.reasons)

            with self.assertRaises(ValueError):
                reg.activate_candidate("cand_bad_dim", make_gate_report(GateDecision.ACCEPT))
            self.assertEqual(reg.get_active_version(), 1)  # unchanged
            self.assertEqual(reg.list_versions(), (1,))  # no version consumed

    def test_18_backbone_weight_mismatch_rejected(self):
        """18. Test that a candidate whose backbone weights differ from its parent's
        (i.e. the freeze contract was violated somewhere) is rejected at activation."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)

            differently_initialized = build_baseline_cnn()  # same architecture, different random weights
            freeze_backbone(differently_initialized)
            differently_initialized.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            reg.persist_candidate("cand_bad_backbone", differently_initialized, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            compat = reg.check_compatibility(differently_initialized, parent_version=1)
            self.assertFalse(compat.same_backbone_weights)
            self.assertIn("backbone_weight_mismatch", compat.reasons)

            with self.assertRaises(ValueError):
                reg.activate_candidate("cand_bad_backbone", make_gate_report(GateDecision.ACCEPT))
            self.assertEqual(reg.get_active_version(), 1)
            self.assertEqual(reg.list_versions(), (1,))

    def test_19_compatible_candidate_passes(self):
        """19. Test that a properly frozen clone of the active model IS reported compatible."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            compat = reg.check_compatibility(candidate, parent_version=1)
            self.assertTrue(compat.compatible)
            self.assertEqual(compat.reasons, ())

    # ------------------------------------------------------------------
    # NoveltyReference relationship
    # ------------------------------------------------------------------

    def test_20_novelty_reference_unchanged_by_activation(self):
        """20. Test that activation only RECORDS a novelty_reference_version in metadata --
        it never calls any mutating NoveltyReference method."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            reference = NoveltyReference(embedding_dim=64)
            reference.add_prototype("normal", np.random.default_rng(0).normal(size=(3, 64)), "synthetic", "normal", "train")
            version_before = reference.version
            ids_before = reference.prototype_ids()

            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult(), novelty_reference_version=reference.version)
            reg.activate_candidate("cand1", make_gate_report(GateDecision.ACCEPT))

            self.assertEqual(reference.version, version_before)
            self.assertEqual(reference.prototype_ids(), ids_before)

            meta = reg.get_version_metadata(2)
            self.assertEqual(meta.novelty_reference_version, version_before)

    # ------------------------------------------------------------------
    # Metadata / auditability
    # ------------------------------------------------------------------

    def test_21_metadata_persisted_and_recoverable(self):
        """21. Test that version metadata round-trips exactly through save/load."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate(
                "cand1", candidate, 1, "synthetic", "cond_b",
                _FakeAdaptationResult(n_train=42, n_val=13, active_acc={"0": 0.91}, candidate_acc={"0": 0.93}),
                novelty_reference_version=7,
            )
            reg.activate_candidate("cand1", make_gate_report(GateDecision.ACCEPT, reasons=("all checks passed",)))

            meta = reg.get_version_metadata(2)
            self.assertEqual(meta.version, 2)
            self.assertEqual(meta.parent_version, 1)
            self.assertEqual(meta.candidate_id, "cand1")
            self.assertEqual(meta.n_training_samples, 42)
            self.assertEqual(meta.n_validation_samples, 13)
            self.assertEqual(meta.per_condition_accuracy_candidate, {"0": 0.93})
            self.assertEqual(meta.novelty_reference_version, 7)
            self.assertEqual(meta.gate_decision, "accept")
            self.assertEqual(meta.gate_reasons, ("all checks passed",))

    def test_22_malformed_metadata_file_raises_cleanly(self):
        """22. Test that a corrupted metadata.json on disk raises rather than crashing obscurely."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            meta_path = os.path.join(reg._version_dir(1), "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            with self.assertRaises(json.JSONDecodeError):
                reg.get_version_metadata(1)

    def test_23_missing_version_raises_cleanly(self):
        """23. Test that requesting a nonexistent version's model/metadata raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            with self.assertRaises(FileNotFoundError):
                reg.load_model_version(99)
            with self.assertRaises(FileNotFoundError):
                reg.get_version_metadata(99)
            with self.assertRaises(FileNotFoundError):
                reg.load_candidate("nonexistent")

    # ------------------------------------------------------------------
    # Pointer safety / atomicity
    # ------------------------------------------------------------------

    def test_24_pointer_never_left_as_temp_file(self):
        """24. Test that after a successful activation, no leftover .tmp pointer/version file remains."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            candidate = self._make_frozen_clone()
            reg.persist_candidate("cand1", candidate, 1, "synthetic", "cond_b", _FakeAdaptationResult())
            reg.activate_candidate("cand1", make_gate_report(GateDecision.ACCEPT))

            self.assertFalse(os.path.exists(reg.pointer_path + ".tmp"))
            self.assertFalse(os.path.isdir(reg._version_dir(2) + ".tmp"))
            with open(reg.pointer_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload, {"active_version": 2})

    def test_25_no_leftover_tmp_after_failed_activation(self):
        """25. Test that a failed activation (incompatible candidate) leaves no partial v{n}.tmp directory."""
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            incompatible_model = build_baseline_cnn(embedding_dim=32)
            incompatible_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            reg.persist_candidate("cand_bad", incompatible_model, 1, "synthetic", "cond_b", _FakeAdaptationResult())

            with self.assertRaises(ValueError):
                reg.activate_candidate("cand_bad", make_gate_report(GateDecision.ACCEPT))

            self.assertFalse(os.path.isdir(reg._version_dir(2)))
            self.assertFalse(os.path.isdir(reg._version_dir(2) + ".tmp"))

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_26_deterministic_lifecycle(self):
        """26. Test that two independent registries taken through the identical sequence
        of operations end up with identical, matching metadata and versions."""
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            reg_a = self._bootstrapped_registry(tmp_a)
            reg_b = self._bootstrapped_registry(tmp_b)

            cand_a = self._make_frozen_clone()
            cand_b = self._make_frozen_clone()
            result = _FakeAdaptationResult(n_train=20, n_val=8, active_acc={"0": 0.9}, candidate_acc={"0": 0.95})

            reg_a.persist_candidate("cand1", cand_a, 1, "synthetic", "cond_b", result, novelty_reference_version=3)
            reg_b.persist_candidate("cand1", cand_b, 1, "synthetic", "cond_b", result, novelty_reference_version=3)

            v_a = reg_a.activate_candidate("cand1", make_gate_report(GateDecision.ACCEPT))
            v_b = reg_b.activate_candidate("cand1", make_gate_report(GateDecision.ACCEPT))

            self.assertEqual(v_a, v_b)
            meta_a = reg_a.get_version_metadata(v_a)
            meta_b = reg_b.get_version_metadata(v_b)
            self.assertEqual(meta_a.to_dict(), meta_b.to_dict())

    # ------------------------------------------------------------------
    # Full Task 23 -> Task 24 integration
    # ------------------------------------------------------------------

    def test_27_full_task23_integration(self):
        """27. End-to-end: train_candidate_head() (Task 23, unmodified) produces a
        candidate -> persist -> evaluate_candidate_regression() (Task 23, unmodified)
        -> explicit ACCEPT -> activate_candidate() (Task 24). Confirms the two
        modules compose without any merged/opaque training+activation function.
        """
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._bootstrapped_registry(tmp)
            active = reg.load_active_model()

            rng = np.random.default_rng(50)
            X_train = rng.normal(size=(15, 2048, 1)).astype(np.float32)
            y_train = rng.integers(0, 4, size=15)
            X_val = rng.normal(size=(6, 2048, 1)).astype(np.float32)
            y_val = rng.integers(0, 4, size=6)

            train_records = [
                AdaptationRecord(observation_id=f"t{i}", dataset="synthetic", split="train", source_recording_id="rec1", label_status=LabelStatus.CONFIRMED, label=int(y_train[i]))
                for i in range(15)
            ]
            val_records = [
                AdaptationRecord(observation_id=f"v{i}", dataset="synthetic", split="train", source_recording_id="rec1", label_status=LabelStatus.CONFIRMED, label=int(y_val[i]))
                for i in range(6)
            ]

            def make_cmr(status):
                return ConditionMonitorResult(
                    has_sufficient_history=True, window_size_used=30, window_size_required=30,
                    novelty_window_mean=0.5, novelty_reference_mean=0.1, novelty_reference_std=0.05,
                    novelty_threshold=0.2, novelty_fraction_above_threshold=1.0,
                    novelty_shift_detected=True, class_distribution_psi=0.5,
                    class_distribution_shift_detected=True, window_class_counts={0: 30}, status=status,
                )
            sustained_evidence = tuple(make_cmr(ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT) for _ in range(10))

            gate = SafetyRegressionGate(SafetyRegressionGateConfig(min_observation_count=10, require_confirmed_labels=True))

            # Step 1: train (Task 23, unmodified).
            adaptation_result = train_candidate_head(
                active, gate, "cond_b", train_records, X_train, y_train, val_records, X_val, y_val,
                condition_monitor_results=sustained_evidence, epochs=1,
            )

            # Step 2: persist (Task 24) -- candidate isolated, active untouched.
            reg.persist_candidate("cand_integration", adaptation_result.candidate_model, parent_version=1, dataset="synthetic", condition_id="cond_b", adaptation_result=adaptation_result)
            self.assertEqual(reg.get_active_version(), 1)

            # Step 3: regression evaluation (Task 23, unmodified) -- gate decides.
            gate_report = evaluate_candidate_regression(gate, adaptation_result, train_records, sustained_evidence)

            # Step 4: explicit ACCEPT-gated activation (Task 24). If the tiny,
            # 1-epoch run happened to REJECT/REVIEW on regression grounds,
            # confirm activation correctly refuses rather than forcing it --
            # either outcome is a valid pass for this integration test.
            if gate_report.decision == GateDecision.ACCEPT:
                new_version = reg.activate_candidate("cand_integration", gate_report)
                self.assertEqual(new_version, 2)
                self.assertEqual(reg.get_active_version(), 2)
            else:
                with self.assertRaises(ValueError):
                    reg.activate_candidate("cand_integration", gate_report)
                self.assertEqual(reg.get_active_version(), 1)


if __name__ == "__main__":
    unittest.main()
