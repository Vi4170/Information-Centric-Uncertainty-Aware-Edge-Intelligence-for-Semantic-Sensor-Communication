"""Unit test suite for the read-only Condition Monitor (Task 18, continual-learning Phase 2).

Uses small, deterministic synthetic sequences to test each accept/reject
and shift-detection rule in isolation, plus one integration check against
real CWRU pipeline outputs (novelty scores from the existing, unmodified
src/novelty detector and predicted classes from the existing, unmodified
CNN) to confirm the monitor works with real data -- without claiming CWRU
demonstrates genuine continual learning.
"""

import math
import os
import unittest

import numpy as np
import pandas as pd

from src.continual.condition_monitor import (
    ConditionMonitor,
    ConditionShiftStatus,
    population_stability_index,
)

# Shared reference configuration for the small synthetic tests.
REF_NOVELTY_MEAN = 0.1
REF_NOVELTY_STD = 0.05  # -> novelty_threshold = 0.1 + 2*0.05 = 0.2 with default k=2.0
REF_CLASS_DIST = {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}
NUM_CLASSES = 4
WINDOW_SIZE = 5

# A balanced class sequence whose proportions exactly match REF_CLASS_DIST
# (2/5=0.4, 1/5=0.2, 1/5=0.2, 1/5=0.2), used to isolate the novelty signal
# from the class-distribution signal in tests that should not trigger it.
BALANCED_CLASSES = [0, 0, 1, 2, 3]


def make_monitor(window_size: int = WINDOW_SIZE) -> ConditionMonitor:
    return ConditionMonitor(
        reference_novelty_mean=REF_NOVELTY_MEAN,
        reference_novelty_std=REF_NOVELTY_STD,
        reference_class_distribution=REF_CLASS_DIST,
        num_classes=NUM_CLASSES,
        window_size=window_size,
    )


class TestConditionMonitor(unittest.TestCase):
    """Test suite for ConditionMonitor's shift-detection logic and safety rules."""

    def test_01_insufficient_history(self):
        """1. Test that fewer than window_size observations report INSUFFICIENT_HISTORY."""
        monitor = make_monitor()
        for i in range(WINDOW_SIZE - 1):
            result = monitor.observe(0.1, 0)
            self.assertFalse(result.has_sufficient_history)
            self.assertEqual(result.status, ConditionShiftStatus.INSUFFICIENT_HISTORY)
            self.assertEqual(result.window_size_used, i + 1)

    def test_02_stable_observations_no_shift(self):
        """2. Test that observations matching the reference in both novelty and class produce STABLE."""
        monitor = make_monitor()
        result = None
        for novelty, cls in zip([REF_NOVELTY_MEAN] * WINDOW_SIZE, BALANCED_CLASSES):
            result = monitor.observe(novelty, cls)

        self.assertTrue(result.has_sufficient_history)
        self.assertFalse(result.novelty_shift_detected)
        self.assertFalse(result.class_distribution_shift_detected)
        self.assertEqual(result.status, ConditionShiftStatus.STABLE)
        self.assertAlmostEqual(result.class_distribution_psi, 0.0, places=6)

    def test_03_isolated_novelty_spike_no_sustained_shift(self):
        """3. Test that a single high-novelty observation among otherwise-normal ones does not trigger a shift.

        The window mean (0.26) exceeds the novelty threshold (0.2), but
        only 1/5 observations individually exceed it -- the strict
        "sustained across the entire window" rule (default fraction
        threshold 1.0) must NOT flag this as a shift.
        """
        monitor = make_monitor()
        novelties = [0.1, 0.1, 0.9, 0.1, 0.1]
        result = None
        for novelty, cls in zip(novelties, BALANCED_CLASSES):
            result = monitor.observe(novelty, cls)

        self.assertGreater(result.novelty_window_mean, result.novelty_threshold)
        self.assertLess(result.novelty_fraction_above_threshold, 1.0)
        self.assertFalse(result.novelty_shift_detected)
        self.assertEqual(result.status, ConditionShiftStatus.STABLE)

    def test_04_sustained_novelty_shift_detected_and_alone_insufficient(self):
        """4. Test that a fully-elevated window triggers a novelty shift, and that alone is not CANDIDATE."""
        monitor = make_monitor()
        result = None
        for novelty, cls in zip([0.9] * WINDOW_SIZE, BALANCED_CLASSES):
            result = monitor.observe(novelty, cls)

        self.assertEqual(result.novelty_fraction_above_threshold, 1.0)
        self.assertTrue(result.novelty_shift_detected)
        self.assertFalse(result.class_distribution_shift_detected)
        self.assertEqual(result.status, ConditionShiftStatus.NOVELTY_SHIFT_ONLY)
        self.assertNotEqual(result.status, ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT)

    def test_05_class_distribution_shift_detected_and_alone_insufficient(self):
        """5. Test that a collapsed class distribution triggers a class shift, and that alone is not CANDIDATE."""
        monitor = make_monitor()
        result = None
        for novelty in [REF_NOVELTY_MEAN] * WINDOW_SIZE:
            result = monitor.observe(novelty, 1)  # every observation predicted as class 1

        self.assertFalse(result.novelty_shift_detected)
        self.assertGreater(result.class_distribution_psi, 0.2)
        self.assertTrue(result.class_distribution_shift_detected)
        self.assertEqual(result.status, ConditionShiftStatus.CLASS_DISTRIBUTION_SHIFT_ONLY)
        self.assertNotEqual(result.status, ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT)

    def test_06_both_signals_trigger_candidate_shift(self):
        """6. Test that sustained novelty AND class-distribution shift together yield CANDIDATE_CONDITION_SHIFT."""
        monitor = make_monitor()
        result = None
        for novelty in [0.9] * WINDOW_SIZE:
            result = monitor.observe(novelty, 1)

        self.assertTrue(result.novelty_shift_detected)
        self.assertTrue(result.class_distribution_shift_detected)
        self.assertEqual(result.status, ConditionShiftStatus.CANDIDATE_CONDITION_SHIFT)

    def test_07_deterministic_and_reproducible(self):
        """7. Test that two independently constructed monitors given identical input produce identical results."""
        sequence = [(0.1, 0), (0.9, 1), (0.9, 1), (0.2, 2), (0.9, 1), (0.05, 0), (0.9, 1)]
        monitor_a = make_monitor()
        monitor_b = make_monitor()

        results_a = [monitor_a.observe(n, c) for n, c in sequence]
        results_b = [monitor_b.observe(n, c) for n, c in sequence]

        self.assertEqual(results_a, results_b)

    def test_08_malformed_novelty_rejected(self):
        """8. Test that non-finite or out-of-range novelty values are rejected."""
        monitor = make_monitor()
        with self.assertRaises(ValueError):
            monitor.observe(float("nan"), 0)
        with self.assertRaises(ValueError):
            monitor.observe(float("inf"), 0)
        with self.assertRaises(ValueError):
            monitor.observe(-0.1, 0)
        with self.assertRaises(ValueError):
            monitor.observe(1.5, 0)
        with self.assertRaises(TypeError):
            monitor.observe("0.5", 0)

    def test_09_malformed_predicted_class_rejected(self):
        """9. Test that out-of-range, non-integer, or boolean predicted_class values are rejected."""
        monitor = make_monitor()
        with self.assertRaises(ValueError):
            monitor.observe(0.1, -1)
        with self.assertRaises(ValueError):
            monitor.observe(0.1, NUM_CLASSES)
        with self.assertRaises(TypeError):
            monitor.observe(0.1, 1.5)
        with self.assertRaises(TypeError):
            monitor.observe(0.1, True)

    def test_10_no_mutation_of_supplied_inputs(self):
        """10. Test that observing a numpy array element does not alias or later reflect array mutation."""
        monitor = make_monitor()
        arr = np.array([0.5], dtype=np.float32)
        monitor.observe(arr[0], np.int64(2))

        arr[0] = 999.0  # mutate the source array AFTER observation

        # The monitor must have copied the scalar value, not kept a reference.
        stored_novelty = list(monitor._novelty_window)[0]
        self.assertAlmostEqual(stored_novelty, 0.5, places=5)
        self.assertNotAlmostEqual(stored_novelty, 999.0, places=1)

    def test_11_invalid_reference_configuration_rejected(self):
        """11. Test that invalid constructor arguments are rejected at construction time."""
        with self.assertRaises(ValueError):
            ConditionMonitor(0.1, -0.05, REF_CLASS_DIST, NUM_CLASSES)  # negative std
        with self.assertRaises(ValueError):
            ConditionMonitor(0.1, 0.05, {0: 0.5, 1: 0.5}, NUM_CLASSES)  # missing keys for num_classes=4
        with self.assertRaises(ValueError):
            ConditionMonitor(0.1, 0.05, {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}, NUM_CLASSES)  # doesn't sum to 1
        with self.assertRaises(ValueError):
            ConditionMonitor(0.1, 0.05, REF_CLASS_DIST, num_classes=1)
        with self.assertRaises(ValueError):
            ConditionMonitor(0.1, 0.05, REF_CLASS_DIST, NUM_CLASSES, window_size=0)
        with self.assertRaises(ValueError):
            ConditionMonitor(0.1, 0.05, REF_CLASS_DIST, NUM_CLASSES, novelty_fraction_threshold=0.0)
        with self.assertRaises(ValueError):
            ConditionMonitor(0.1, 0.05, REF_CLASS_DIST, NUM_CLASSES, psi_threshold=0.0)

    def test_12_reset_clears_history(self):
        """12. Test that reset() clears the rolling window back to INSUFFICIENT_HISTORY."""
        monitor = make_monitor()
        for novelty, cls in zip([0.9] * WINDOW_SIZE, BALANCED_CLASSES):
            result = monitor.observe(novelty, cls)
        self.assertTrue(result.has_sufficient_history)

        monitor.reset()
        result_after_reset = monitor.observe(0.1, 0)
        self.assertFalse(result_after_reset.has_sufficient_history)
        self.assertEqual(result_after_reset.window_size_used, 1)
        self.assertEqual(result_after_reset.status, ConditionShiftStatus.INSUFFICIENT_HISTORY)

    def test_13_population_stability_index_pure_function(self):
        """13. Test the standalone PSI helper directly: identical distributions -> ~0, divergent -> large."""
        reference = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        identical_counts = {0: 5, 1: 5, 2: 5, 3: 5}
        psi_identical = population_stability_index(reference, identical_counts, 4, window_size=20)
        self.assertAlmostEqual(psi_identical, 0.0, places=3)

        divergent_counts = {0: 20, 1: 0, 2: 0, 3: 0}
        psi_divergent = population_stability_index(reference, divergent_counts, 4, window_size=20)
        self.assertGreater(psi_divergent, 1.0)

        # Pure function: must not mutate its inputs.
        self.assertEqual(reference, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        self.assertEqual(divergent_counts, {0: 20, 1: 0, 2: 0, 3: 0})

    def test_14_window_size_used_caps_at_window_size(self):
        """14. Test that window_size_used never exceeds window_size once the buffer is full."""
        monitor = make_monitor()
        for _ in range(WINDOW_SIZE + 10):
            result = monitor.observe(0.1, 0)
        self.assertEqual(result.window_size_used, WINDOW_SIZE)

    def test_15_real_cwru_pipeline_outputs_integration(self):
        """15. Integration test: feed real novelty/predicted-class outputs from the existing pipeline.

        This validates that the monitor works with genuine pipeline
        outputs -- it does NOT claim CWRU demonstrates real operating
        condition drift (see docs/condition_monitor.md's documented
        limitation: CWRU recordings are single fixed-condition throughout).
        Skips gracefully if the trained model/dataset are unavailable.
        """
        model_path = "models/cwru_cnn_baseline.keras"
        data_path = "data/processed/cwru/cwru_dataset_v1.npz"
        metadata_path = "data/processed/cwru/cwru_metadata.csv"
        if not (os.path.exists(model_path) and os.path.exists(data_path) and os.path.exists(metadata_path)):
            self.skipTest("Trained CNN model / processed CWRU dataset not available in this environment")

        import keras
        from src.cnn.model import extract_embeddings, predict_classes
        from src.novelty.novelty import DistanceNoveltyDetector

        model = keras.models.load_model(model_path, compile=False)
        data = np.load(data_path)
        meta = pd.read_csv(metadata_path)

        X_train, y_train = data["X_train"], data["y_train"]
        X_test = data["X_test"]
        meta_train = meta[meta["split"] == "train"].reset_index(drop=True)

        # Reference statistics: derived strictly from TRAINING data, exactly
        # as Task 14's calibration discipline requires -- never from test.
        # Novelty reference is taken from Normal-only training embeddings:
        # this represents the genuine "in-control" baseline (how novel does
        # already-known-normal data look), not the full training set, which
        # is ~90% already-labelled fault data here and would make the
        # control-chart threshold degenerate (see docs/condition_monitor.md).
        train_embeddings = extract_embeddings(model, X_train)
        detector = DistanceNoveltyDetector(reference_class=0)
        detector.fit(train_embeddings, y_train)
        train_novelty = detector.score(train_embeddings)
        normal_mask = y_train == 0
        reference_novelty_mean = float(np.mean(train_novelty[normal_mask]))
        reference_novelty_std = float(np.std(train_novelty[normal_mask]))

        train_class_counts = meta_train["fault_label"].value_counts(normalize=True).to_dict()
        reference_class_distribution = {c: float(train_class_counts.get(c, 0.0)) for c in range(4)}

        monitor = ConditionMonitor(
            reference_novelty_mean=reference_novelty_mean,
            reference_novelty_std=reference_novelty_std,
            reference_class_distribution=reference_class_distribution,
            num_classes=4,
            window_size=30,
        )

        test_embeddings = extract_embeddings(model, X_test)
        test_novelty = detector.score(test_embeddings)
        test_predicted_classes = predict_classes(model, X_test)

        statuses = []
        for novelty, predicted_class in zip(test_novelty, test_predicted_classes):
            result = monitor.observe(float(novelty), int(predicted_class))
            statuses.append(result.status)
            self.assertIsInstance(result.status, ConditionShiftStatus)

        # The first window_size - 1 observations must report insufficient history.
        self.assertTrue(all(s == ConditionShiftStatus.INSUFFICIENT_HISTORY for s in statuses[:29]))
        # The monitor must run cleanly over the entire real test split without error.
        self.assertEqual(len(statuses), len(test_novelty))


if __name__ == "__main__":
    unittest.main()
