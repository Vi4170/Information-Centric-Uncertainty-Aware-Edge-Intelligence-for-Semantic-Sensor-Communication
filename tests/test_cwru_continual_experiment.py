import tempfile
import unittest

import keras
import numpy as np
import pandas as pd

from src.cnn.model import build_baseline_cnn, extract_embeddings
from src.continual.condition_monitor import ConditionShiftStatus
from src.continual.cwru_continual_experiment import run_experiment
from src.continual.model_registry import ModelRegistry
from src.continual.safety_regression_gate import GateDecision
from src.novelty.novelty import DistanceNoveltyDetector

CLASS_NAMES = {0: "Normal", 1: "Inner Race Fault", 2: "Ball Fault", 3: "Outer Race Fault"}


def _make_signal(n, offset, scale, seed):
    rng = np.random.default_rng(seed)
    return (offset + rng.normal(scale=scale, size=(n, 2048, 1))).astype(np.float32)


def _make_meta(observation_ids, split, file_ids, fault_label, window_indices):
    return pd.DataFrame(
        {
            "observation_id": observation_ids,
            "split": split,
            "window_index": window_indices,
            "file_id": file_ids,
            "fault_label": fault_label,
        }
    )


def _build_dataset(n_a_train=40, n_b_train=150, n_a_val=10, n_b_val=15, n_a_test=5, n_b_test=5, seed=42):
    X_a_train = _make_signal(n_a_train, 0.0, 0.05, seed + 1)
    X_b_train = _make_signal(n_b_train, 5.0, 0.5, seed + 2)
    X_a_val = _make_signal(n_a_val, 0.0, 0.05, seed + 3)
    X_b_val = _make_signal(n_b_val, 5.0, 0.5, seed + 4)
    X_a_test = _make_signal(n_a_test, 0.0, 0.05, seed + 5)
    X_b_test = _make_signal(n_b_test, 5.0, 0.5, seed + 6)

    X_train = np.concatenate([X_a_train, X_b_train])
    y_train = np.concatenate([np.zeros(n_a_train, dtype=np.int64), np.ones(n_b_train, dtype=np.int64)])
    X_val = np.concatenate([X_a_val, X_b_val])
    y_val = np.concatenate([np.zeros(n_a_val, dtype=np.int64), np.ones(n_b_val, dtype=np.int64)])
    X_test = np.concatenate([X_a_test, X_b_test])
    y_test = np.concatenate([np.zeros(n_a_test, dtype=np.int64), np.ones(n_b_test, dtype=np.int64)])

    train_meta = pd.concat(
        [
            _make_meta([f"a_train_{i:04d}" for i in range(n_a_train)], "train", "rec_a_train", 0, list(range(n_a_train))),
            _make_meta([f"b_train_{i:04d}" for i in range(n_b_train)], "train", "rec_b_train", 1, list(range(n_b_train))),
        ],
        ignore_index=True,
    )
    val_meta = pd.concat(
        [
            _make_meta([f"a_val_{i:04d}" for i in range(n_a_val)], "val", "rec_a_val", 0, list(range(n_a_val))),
            _make_meta([f"b_val_{i:04d}" for i in range(n_b_val)], "val", "rec_b_val", 1, list(range(n_b_val))),
        ],
        ignore_index=True,
    )
    test_meta = pd.concat(
        [
            _make_meta([f"a_test_{i:04d}" for i in range(n_a_test)], "test", "rec_a_test", 0, list(range(n_a_test))),
            _make_meta([f"b_test_{i:04d}" for i in range(n_b_test)], "test", "rec_b_test", 1, list(range(n_b_test))),
        ],
        ignore_index=True,
    )

    return X_train, y_train, train_meta, X_val, y_val, val_meta, X_test, y_test, test_meta


def _make_model(seed=42):
    keras.utils.set_random_seed(seed)
    return build_baseline_cnn()


def _compute_novelty(model, X_train, y_train):
    embeddings_train = extract_embeddings(model, X_train)
    detector = DistanceNoveltyDetector(reference_class=0)
    detector.fit(embeddings_train, y_train)
    novelty_train = detector.score(embeddings_train)
    return embeddings_train, novelty_train


class TestCwruContinualExperimentOrchestration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _make_model(seed=42)
        (
            cls.X_train,
            cls.y_train,
            cls.train_meta,
            cls.X_val,
            cls.y_val,
            cls.val_meta,
            cls.X_test,
            cls.y_test,
            cls.test_meta,
        ) = _build_dataset()
        cls.model.fit(cls.X_train, cls.y_train, epochs=8, batch_size=16, verbose=0)
        cls.embeddings_train, cls.novelty_train = _compute_novelty(cls.model, cls.X_train, cls.y_train)
        cls.predicted_train = cls.y_train.copy()

    def _run(self, registry_dir, **overrides):
        kwargs = dict(
            active_model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            train_meta=self.train_meta,
            X_val=self.X_val,
            y_val=self.y_val,
            val_meta=self.val_meta,
            X_test=self.X_test,
            y_test=self.y_test,
            test_meta=self.test_meta,
            embeddings_train=self.embeddings_train,
            novelty_train=self.novelty_train,
            predicted_train=self.predicted_train,
            registry_dir=registry_dir,
            seed=42,
            head_epochs=2,
            class_names=CLASS_NAMES,
        )
        kwargs.update(overrides)
        return run_experiment(**kwargs)

    def test_01_b_absent_from_initial_reference(self):
        with tempfile.TemporaryDirectory() as d:
            result = self._run(d)
        self.assertTrue(result["b_absent_from_initial_reference"])
        self.assertEqual(result["initial_reference_version"], 1)
        self.assertEqual(result["initial_prototype_count"], 1)

    def test_02_actual_admission_controller_path_used_and_accepts(self):
        with tempfile.TemporaryDirectory() as d:
            result = self._run(d)
        self.assertEqual(result["admission_result"]["decision"], GateDecision.ACCEPT.value)
        self.assertTrue(result["admission_result"]["prototype_added"])
        self.assertGreaterEqual(result["admission_result"]["gate_report"]["safety"]["n_observations"], 30)
        status_counts = result["detection_result"]["status_counts"]
        self.assertIn(ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT.value, status_counts)

    def test_03_prototype_version_changes_only_after_accept(self):
        with tempfile.TemporaryDirectory() as d:
            accepted = self._run(d)
        self.assertEqual(accepted["reference_version_transition"], {"before": 1, "after": 2})

        with tempfile.TemporaryDirectory() as d:
            rejected = self._run(
                d,
                train_meta=self.train_meta.iloc[:50].reset_index(drop=True),
                y_train=self.y_train[:50],
                X_train=self.X_train[:50],
                embeddings_train=self.embeddings_train[:50],
                novelty_train=self.novelty_train[:50],
                predicted_train=self.predicted_train[:50],
            )
        self.assertNotEqual(rejected["admission_result"]["decision"], GateDecision.ACCEPT.value)
        self.assertFalse(rejected["admission_result"]["prototype_added"])
        self.assertEqual(rejected["reference_version_transition"], {"before": 1, "after": 1})
        self.assertIsNone(rejected["candidate_result"])
        self.assertFalse(rejected["activation_result"]["activated"])
        self.assertEqual(rejected["model_version_after"], rejected["model_version_before"])

    def test_04_rejected_candidate_does_not_activate(self):
        with tempfile.TemporaryDirectory() as d:
            rejected = self._run(
                d,
                train_meta=self.train_meta.iloc[:50].reset_index(drop=True),
                y_train=self.y_train[:50],
                X_train=self.X_train[:50],
                embeddings_train=self.embeddings_train[:50],
                novelty_train=self.novelty_train[:50],
                predicted_train=self.predicted_train[:50],
            )
            registry = ModelRegistry(registry_dir=d)
            self.assertEqual(registry.get_active_version(), 1)
        self.assertIsNone(rejected["regression_result"])
        self.assertEqual(rejected["activation_result"]["model_version_after"], 1)

    def test_05_accepted_candidate_activates(self):
        with tempfile.TemporaryDirectory() as d:
            result = self._run(d)
            registry = ModelRegistry(registry_dir=d)
            self.assertEqual(registry.get_active_version(), result["model_version_after"])
        self.assertTrue(result["activation_result"]["activated"])
        self.assertEqual(result["model_version_after"], result["model_version_before"] + 1)
        self.assertEqual(result["regression_result"]["decision"], GateDecision.ACCEPT.value)

    def test_06_candidate_does_not_mutate_active_model(self):
        weights_before = [w.copy() for w in self.model.get_weights()]
        with tempfile.TemporaryDirectory() as d:
            result = self._run(d)
        weights_after = self.model.get_weights()
        self.assertTrue(result["active_model_untouched"])
        for before, after in zip(weights_before, weights_after):
            self.assertTrue(np.array_equal(before, after))

    def test_07_no_adaptation_test_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            result = self._run(d)
        self.assertEqual(result["leakage_verification"]["buffer_test_overlap_count"], 0)
        test_ids = set(result["test_ids"]["test_observation_ids"])
        adaptation_ids = (
            set(result["adaptation_ids"]["known_condition_train_observation_ids"])
            | set(result["adaptation_ids"]["new_condition_train_observation_ids"])
            | set(result["adaptation_ids"]["rehearsal_observation_ids"])
            | set(result["adaptation_ids"]["head_training_observation_ids"])
            | set(result["adaptation_ids"]["val_observation_ids"])
        )
        self.assertEqual(test_ids & adaptation_ids, set())

    def test_08_deterministic_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            result = self._run(d)
        observed_order = [entry["observation_id"] for entry in result["novelty_before_admission"]]
        expected_order = [f"b_train_{i:04d}" for i in range(len(observed_order))]
        self.assertEqual(observed_order, expected_order)

    def test_09_same_seed_produces_equivalent_results(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            result_1 = self._run(d1)
            result_2 = self._run(d2)

        self.assertEqual(result_1["admission_result"]["decision"], result_2["admission_result"]["decision"])
        self.assertEqual(result_1["regression_result"]["decision"], result_2["regression_result"]["decision"])
        self.assertEqual(result_1["activation_result"]["activated"], result_2["activation_result"]["activated"])
        self.assertEqual(result_1["model_version_after"], result_2["model_version_after"])
        self.assertEqual(
            result_1["candidate_result"]["per_condition_accuracy_candidate"],
            result_2["candidate_result"]["per_condition_accuracy_candidate"],
        )
        self.assertEqual(
            result_1["candidate_result"]["overall_accuracy_candidate"],
            result_2["candidate_result"]["overall_accuracy_candidate"],
        )


if __name__ == "__main__":
    unittest.main()
