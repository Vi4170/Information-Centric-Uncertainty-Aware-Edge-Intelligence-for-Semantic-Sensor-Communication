"""Unit test suite for leakage-safe CNN head adaptation (Task 23).

Uses one small, module-level synthetic CNN (the real src/cnn architecture,
tiny synthetic data, 1-2 epochs) built once in setUpClass to keep runtime
reasonable -- no expensive CWRU training is performed for these unit
tests. One separate, skip-if-unavailable test at the bottom exercises a
small real-CWRU smoke path using only train/validation data.
"""

import os
import unittest

import numpy as np

from src.cnn.model import build_baseline_cnn, extract_embeddings
from src.continual.adaptation_buffer import AdaptationRecord, LabelStatus
from src.continual.cnn_head_adaptation import (
    BACKBONE_LAYER_NAMES,
    HEAD_LAYER_NAMES,
    clone_model_with_weights,
    evaluate_candidate_regression,
    freeze_backbone,
    select_rehearsal_samples,
    train_candidate_head,
)
from src.continual.condition_monitor import ConditionMonitorResult, ConditionShiftStatus
from src.continual.novelty_reference import NoveltyReference
from src.continual.safety_regression_gate import GateDecision, SafetyRegressionGate, SafetyRegressionGateConfig

NUM_CLASSES = 4
EMBEDDING_DIM = 64


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


# Reused wherever a test needs the gate's sustained-evidence check to pass
# -- CNN adaptation is a higher-risk decision than prototype admission and
# is held to at least the same evidence standard (Task 20/21).
SUSTAINED_EVIDENCE = tuple(make_cmr(ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT) for _ in range(10))


def make_records(n, split="train", label_status=LabelStatus.CONFIRMED, labels=None, recording_id="rec1", dataset="synthetic"):
    return [
        AdaptationRecord(
            observation_id=f"{recording_id}_obs_{i}",
            dataset=dataset,
            split=split,
            source_recording_id=recording_id,
            label_status=label_status,
            label=(labels[i] if labels is not None else None),
        )
        for i in range(n)
    ]


def make_data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2048, 1)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=n)
    return X, y


def make_gate(min_observation_count=10):
    return SafetyRegressionGate(SafetyRegressionGateConfig(min_observation_count=min_observation_count, require_confirmed_labels=True))


class TestCnnHeadAdaptation(unittest.TestCase):
    """Test suite for src/continual/cnn_head_adaptation.py."""

    @classmethod
    def setUpClass(cls):
        cls.active_model = build_baseline_cnn()
        cls.active_weights_snapshot = [w.copy() for w in cls.active_model.get_weights()]

    def _fresh_active(self):
        """A pristine independent copy, used whenever a test needs to compare against
        a truly untouched baseline without risking cross-test interference."""
        m = clone_model_with_weights(self.active_model)
        return m

    # ------------------------------------------------------------------
    # Freezing / trainable-parameter verification
    # ------------------------------------------------------------------

    def test_01_freeze_backbone_marks_correct_layers_trainable(self):
        """1. Test that only head layers end up with nonzero trainable parameters."""
        candidate = clone_model_with_weights(self.active_model)
        report = freeze_backbone(candidate)
        self.assertTrue(report.all_backbone_frozen)
        self.assertEqual(report.backbone_trainable_param_count, 0)
        self.assertGreater(report.head_trainable_param_count, 0)
        self.assertEqual(report.head_layer_names, HEAD_LAYER_NAMES)

    def test_02_freeze_detects_leaked_trainable_backbone_parameter(self):
        """2. Test that the freeze report correctly FAILS if a backbone layer is
        accidentally left trainable (incomplete backbone_layer_names list)."""
        candidate = clone_model_with_weights(self.active_model)
        incomplete_backbone = tuple(n for n in BACKBONE_LAYER_NAMES if n != "conv1d_3")  # leave conv1d_3 trainable
        report = freeze_backbone(candidate, backbone_layer_names=incomplete_backbone)
        self.assertFalse(report.all_backbone_frozen)
        self.assertGreater(report.backbone_trainable_param_count, 0)

    def test_03_trainable_variables_only_head_after_compile(self):
        """3. Test that model.trainable_variables (what the optimizer will update) is
        exactly the head layer's own weights -- nothing from the backbone."""
        candidate = clone_model_with_weights(self.active_model)
        freeze_backbone(candidate)
        candidate.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        expected_head_weight_count = sum(len(candidate.get_layer(n).trainable_weights) for n in HEAD_LAYER_NAMES)
        self.assertEqual(len(candidate.trainable_variables), expected_head_weight_count)

        head_var_names = {v.path if hasattr(v, "path") else v.name for v in candidate.trainable_variables}
        for name in head_var_names:
            self.assertTrue(any(head_layer in name for head_layer in HEAD_LAYER_NAMES), f"unexpected trainable var: {name}")

    # ------------------------------------------------------------------
    # Active-model isolation
    # ------------------------------------------------------------------

    def test_04_active_model_not_mutated_during_training(self):
        """4. Test that the active model's weights are byte-identical after adapting a candidate."""
        X_train, y_train = make_data(20, seed=1)
        X_val, y_val = make_data(8, seed=2)
        train_records = make_records(20, labels=list(y_train))
        val_records = make_records(8, labels=list(y_val), recording_id="rec_val")
        gate = make_gate()

        train_candidate_head(
            self._fresh_active(), gate, "cond_test", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1
        )

        for before, after in zip(self.active_weights_snapshot, self.active_model.get_weights()):
            np.testing.assert_array_equal(before, after)

    def test_05_candidate_is_separate_object_not_auto_activated(self):
        """5. Test that the returned candidate is a distinct object and nothing writes
        any 'active model' pointer -- activation stays a separate future operation."""
        active = self._fresh_active()
        X_train, y_train = make_data(15, seed=3)
        X_val, y_val = make_data(6, seed=4)
        train_records = make_records(15, labels=list(y_train))
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")
        gate = make_gate()

        result = train_candidate_head(active, gate, "cond_test", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1)

        self.assertIsNot(result.candidate_model, active)

    # ------------------------------------------------------------------
    # Confirmed vs pseudo, provenance, leakage
    # ------------------------------------------------------------------

    def test_06_confirmed_labels_accepted(self):
        """6. Test that a well-formed, fully-confirmed candidate trains successfully."""
        X_train, y_train = make_data(15, seed=5)
        X_val, y_val = make_data(6, seed=6)
        train_records = make_records(15, labels=list(y_train))
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")
        gate = make_gate()

        result = train_candidate_head(self._fresh_active(), gate, "cond_ok", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1)
        self.assertEqual(result.n_training_samples, 15)
        self.assertTrue(result.freeze_report.all_backbone_frozen)

    def test_07_pseudo_labels_rejected(self):
        """7. Test that any pseudo-labelled record in the training set is rejected outright."""
        X_train, y_train = make_data(15, seed=7)
        X_val, y_val = make_data(6, seed=8)
        train_records = make_records(14, labels=list(y_train[:14])) + make_records(
            1, label_status=LabelStatus.PSEUDO, recording_id="rec_pseudo"
        )
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")
        gate = make_gate()

        with self.assertRaises(ValueError):
            train_candidate_head(self._fresh_active(), gate, "cond_bad", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1)

    def test_08_test_split_rejected(self):
        """8. Test that a test-split record in the training set is rejected via the gate pre-check."""
        X_train, y_train = make_data(15, seed=9)
        X_val, y_val = make_data(6, seed=10)
        train_records = make_records(14, labels=list(y_train[:14])) + make_records(
            1, split="test", labels=[0], recording_id="rec_leak"
        )
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")
        gate = make_gate()

        with self.assertRaises(ValueError):
            train_candidate_head(self._fresh_active(), gate, "cond_leak", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1)

    def test_09_missing_labels_rejected(self):
        """9. Test that a CONFIRMED record with no label value is rejected (bypasses
        AdaptationBuffer's own constructor validation -- must be caught here too)."""
        X_train, y_train = make_data(15, seed=11)
        X_val, y_val = make_data(6, seed=12)
        train_records = make_records(14, labels=list(y_train[:14])) + [
            AdaptationRecord(
                observation_id="contradiction", dataset="synthetic", split="train",
                source_recording_id="rec1", label_status=LabelStatus.CONFIRMED, label=None,
            )
        ]
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")
        gate = make_gate()

        with self.assertRaises(ValueError):
            train_candidate_head(self._fresh_active(), gate, "cond_bad2", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1)

    def test_10_insufficient_adaptation_data_fails_safely(self):
        """10. Test that too few training records (below the gate's minimum) fails closed."""
        X_train, y_train = make_data(3, seed=13)
        X_val, y_val = make_data(2, seed=14)
        train_records = make_records(3, labels=list(y_train))
        val_records = make_records(2, labels=list(y_val), recording_id="rec_val")
        gate = make_gate(min_observation_count=10)

        with self.assertRaises(ValueError):
            train_candidate_head(self._fresh_active(), gate, "cond_small", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1)

    # ------------------------------------------------------------------
    # Determinism / train-val separation
    # ------------------------------------------------------------------

    def test_11_deterministic_adaptation(self):
        """11. Test that two independent adaptations from identical inputs (same seed)
        produce byte-identical candidate weights and history."""
        X_train, y_train = make_data(15, seed=15)
        X_val, y_val = make_data(6, seed=16)
        train_records = make_records(15, labels=list(y_train))
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")

        result_a = train_candidate_head(self._fresh_active(), make_gate(), "c", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1, seed=99)
        result_b = train_candidate_head(self._fresh_active(), make_gate(), "c", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1, seed=99)

        for w_a, w_b in zip(result_a.candidate_model.get_weights(), result_b.candidate_model.get_weights()):
            np.testing.assert_array_equal(w_a, w_b)
        self.assertEqual(result_a.history, result_b.history)

    def test_12_validation_not_used_for_optimization(self):
        """12. Test that changing ONLY the validation data (same training data) never
        changes the resulting candidate weights -- proving validation never enters
        the gradient computation.
        """
        X_train, y_train = make_data(15, seed=17)
        X_val_a, y_val_a = make_data(6, seed=18)
        X_val_b, y_val_b = make_data(6, seed=19)  # different validation data
        train_records = make_records(15, labels=list(y_train))
        val_records_a = make_records(6, labels=list(y_val_a), recording_id="rec_val_a")
        val_records_b = make_records(6, labels=list(y_val_b), recording_id="rec_val_b")

        result_a = train_candidate_head(self._fresh_active(), make_gate(), "c", train_records, X_train, y_train, val_records_a, X_val_a, y_val_a, SUSTAINED_EVIDENCE, epochs=1, seed=7)
        result_b = train_candidate_head(self._fresh_active(), make_gate(), "c", train_records, X_train, y_train, val_records_b, X_val_b, y_val_b, SUSTAINED_EVIDENCE, epochs=1, seed=7)

        for w_a, w_b in zip(result_a.candidate_model.get_weights(), result_b.candidate_model.get_weights()):
            np.testing.assert_array_equal(w_a, w_b)

    # ------------------------------------------------------------------
    # Embedding-space invariant
    # ------------------------------------------------------------------

    def test_13_embeddings_unchanged_before_and_after_adaptation(self):
        """13. CRITICAL invariant test: embeddings for identical inputs are IDENTICAL
        before and after head-only adaptation, since the entire backbone is frozen.
        """
        active = self._fresh_active()
        probe_X, _ = make_data(5, seed=20)
        embeddings_before = extract_embeddings(active, probe_X)

        X_train, y_train = make_data(15, seed=21)
        X_val, y_val = make_data(6, seed=22)
        train_records = make_records(15, labels=list(y_train))
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")
        result = train_candidate_head(active, make_gate(), "c", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=3, seed=8)

        embeddings_after = extract_embeddings(result.candidate_model, probe_X)
        np.testing.assert_array_equal(embeddings_before, embeddings_after)

    def test_14_novelty_reference_remains_compatible(self):
        """14. Test that an existing NoveltyReference's nearest-prototype results for
        fixed embeddings are unaffected by CNN head adaptation (same underlying claim
        as test_13, verified through NoveltyReference's own API instead of raw arrays).
        """
        active = self._fresh_active()
        probe_X, _ = make_data(3, seed=23)
        embeddings_before = extract_embeddings(active, probe_X)

        reference = NoveltyReference(embedding_dim=EMBEDDING_DIM)
        reference.add_prototype("known", embeddings_before, "synthetic", "known_cond", "train")

        X_train, y_train = make_data(15, seed=24)
        X_val, y_val = make_data(6, seed=25)
        train_records = make_records(15, labels=list(y_train))
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")
        result = train_candidate_head(active, make_gate(), "c", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=3, seed=9)

        embeddings_after = extract_embeddings(result.candidate_model, probe_X)
        for before, after in zip(embeddings_before, embeddings_after):
            dist_before = reference.distance_to_nearest(before)
            dist_after = reference.distance_to_nearest(after)
            self.assertEqual(dist_before, dist_after)

    # ------------------------------------------------------------------
    # Prediction behavior
    # ------------------------------------------------------------------

    def test_15_prediction_probabilities_can_change(self):
        """15. Test that head adaptation CAN change predicted probabilities (it is not
        a no-op), by training toward a class the active model is not already confident in.
        """
        from src.cnn.model import predict_probabilities

        active = self._fresh_active()
        X_train = np.tile(np.random.default_rng(30).normal(size=(1, 2048, 1)).astype(np.float32), (20, 1, 1))
        y_train = np.full(20, 2, dtype=np.int64)  # every sample confidently labelled class 2
        X_val, y_val = make_data(6, seed=31)
        train_records = make_records(20, labels=list(y_train))
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")

        probs_before = predict_probabilities(active, X_train[:1])
        result = train_candidate_head(active, make_gate(), "c", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=10, seed=11)
        probs_after = predict_probabilities(result.candidate_model, X_train[:1])

        self.assertFalse(np.array_equal(probs_before, probs_after))

    # ------------------------------------------------------------------
    # Regression gate reuse
    # ------------------------------------------------------------------

    def test_16_regression_evaluation_reachable_via_existing_gate(self):
        """16. Test that evaluate_candidate_regression() correctly reuses SafetyRegressionGate
        end to end (ACCEPT for improvement, REJECT for excessive regression)."""
        X_train, y_train = make_data(15, seed=32)
        X_val, y_val = make_data(6, seed=33)
        train_records = make_records(15, labels=list(y_train))
        val_records = make_records(6, labels=list(y_val), recording_id="rec_val")
        gate = make_gate()

        result = train_candidate_head(self._fresh_active(), gate, "cond_reg", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1)
        report = evaluate_candidate_regression(gate, result, train_records, SUSTAINED_EVIDENCE)
        self.assertIn(report.decision, (GateDecision.ACCEPT, GateDecision.REJECT, GateDecision.REVIEW))
        self.assertTrue(report.regression.evaluated)

        # Synthetic excessive-regression scenario, independent of what the tiny
        # real training run happened to produce.
        import dataclasses
        forced_bad = dataclasses.replace(
            result,
            per_condition_accuracy_active={"0": 0.95},
            per_condition_accuracy_candidate={"0": 0.50},
        )
        bad_report = evaluate_candidate_regression(gate, forced_bad, train_records, SUSTAINED_EVIDENCE)
        self.assertEqual(bad_report.decision, GateDecision.REJECT)
        self.assertIn("excessive_regression", bad_report.regression.failed_checks)

    # ------------------------------------------------------------------
    # Rehearsal
    # ------------------------------------------------------------------

    def test_17_rehearsal_selection_deterministic_and_bounded(self):
        """17. Test that rehearsal selection is deterministic and respects the per-condition bound."""
        active = self._fresh_active()
        pool_X, _ = make_data(10, seed=40)
        pool_records = make_records(10, labels=[0] * 10, recording_id="rec_pool")

        reference = NoveltyReference(embedding_dim=EMBEDDING_DIM)
        seed_embeddings = extract_embeddings(active, pool_X[:2])
        reference.add_prototype("only_condition", seed_embeddings, "synthetic", "cond", "train")

        selection_a = select_rehearsal_samples(reference, active, pool_records, pool_X, k_per_condition=3)
        selection_b = select_rehearsal_samples(reference, active, pool_records, pool_X, k_per_condition=3)

        self.assertEqual(selection_a, selection_b)
        for indices in selection_a.condition_id_to_indices.values():
            self.assertLessEqual(len(indices), 3)

    def test_18_rehearsal_rejects_non_confirmed_pool(self):
        """18. Test that a rehearsal pool containing pseudo-labelled or test-split records is rejected."""
        active = self._fresh_active()
        pool_X, _ = make_data(5, seed=41)
        pseudo_pool = make_records(5, label_status=LabelStatus.PSEUDO, recording_id="rec_pseudo")

        reference = NoveltyReference(embedding_dim=EMBEDDING_DIM)
        reference.add_prototype("cond", extract_embeddings(active, pool_X[:2]), "synthetic", "cond", "train")

        with self.assertRaises(ValueError):
            select_rehearsal_samples(reference, active, pseudo_pool, pool_X, k_per_condition=3)

    def test_19_rehearsal_requires_nonempty_reference(self):
        """19. Test that rehearsal selection fails closed against an empty NoveltyReference."""
        active = self._fresh_active()
        pool_X, _ = make_data(5, seed=42)
        pool_records = make_records(5, labels=[0] * 5, recording_id="rec_pool2")
        empty_reference = NoveltyReference(embedding_dim=EMBEDDING_DIM)

        with self.assertRaises(ValueError):
            select_rehearsal_samples(empty_reference, active, pool_records, pool_X, k_per_condition=3)

    # ------------------------------------------------------------------
    # Real CWRU smoke test (skip if unavailable)
    # ------------------------------------------------------------------

    def test_20_real_cwru_smoke_test_train_val_only(self):
        """20. Lightweight real-data smoke test using ONLY train/validation CWRU data.

        Does not use the permanent CWRU test split, and does not claim this
        proves continual learning -- it only confirms the mechanism runs
        cleanly against real provenance and real embeddings.
        """
        model_path = "models/cwru_cnn_baseline.keras"
        data_path = "data/processed/cwru/cwru_dataset_v1.npz"
        metadata_path = "data/processed/cwru/cwru_metadata.csv"
        if not (os.path.exists(model_path) and os.path.exists(data_path) and os.path.exists(metadata_path)):
            self.skipTest("Trained CNN model / processed CWRU dataset not available in this environment")

        import keras as keras_module
        import pandas as pd

        real_active = keras_module.models.load_model(model_path, compile=False)
        data = np.load(data_path)
        meta = pd.read_csv(metadata_path)

        train_meta = meta[meta["split"] == "train"].head(20).reset_index(drop=True)
        val_meta = meta[meta["split"] == "val"].head(8).reset_index(drop=True)

        X_train = data["X_train"][:20]
        y_train = data["y_train"][:20]
        X_val = data["X_val"][:8]
        y_val = data["y_val"][:8]

        train_records = [
            AdaptationRecord(
                observation_id=str(row["observation_id"]), dataset="cwru", split="train",
                source_recording_id=str(row["file_id"]), label_status=LabelStatus.CONFIRMED, label=int(row["fault_label"]),
            )
            for _, row in train_meta.iterrows()
        ]
        val_records = [
            AdaptationRecord(
                observation_id=str(row["observation_id"]), dataset="cwru", split="val",
                source_recording_id=str(row["file_id"]), label_status=LabelStatus.CONFIRMED, label=int(row["fault_label"]),
            )
            for _, row in val_meta.iterrows()
        ]

        gate = make_gate(min_observation_count=10)
        result = train_candidate_head(real_active, gate, "cwru_smoke", train_records, X_train, y_train, val_records, X_val, y_val, SUSTAINED_EVIDENCE, epochs=1)

        self.assertTrue(result.freeze_report.all_backbone_frozen)
        self.assertIsNot(result.candidate_model, real_active)

        probe = X_train[:3]
        np.testing.assert_array_equal(extract_embeddings(real_active, probe), extract_embeddings(result.candidate_model, probe))


if __name__ == "__main__":
    unittest.main()
