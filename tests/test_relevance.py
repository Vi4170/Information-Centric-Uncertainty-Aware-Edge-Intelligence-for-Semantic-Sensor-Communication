"""Unit test suite for baseline Task Relevance module (src/relevance/).

Tests cover class-mapping strategy, probability-weighted strategy, output range
constraints, input validation (class IDs, probability shapes, NaN/Inf, sums),
strategy rejection, reproducibility, and score differentiation across classes.
"""

import unittest

import numpy as np

from src.relevance.config import CLASS_RELEVANCE_MAP, NUM_CLASSES
from src.relevance.relevance import relevance_from_class, relevance_from_probabilities


class TestTaskRelevance(unittest.TestCase):
    """Test suite for Task Relevance module functions."""

    def setUp(self):
        np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Score always in [0, 1]
    # ------------------------------------------------------------------
    def test_01_class_mapping_score_in_range(self):
        """1. Test class-mapping scores are always in [0, 1] for all valid classes."""
        for class_id in range(NUM_CLASSES):
            score = relevance_from_class(class_id)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertTrue(np.isfinite(score))

    # ------------------------------------------------------------------
    # 2. Valid class input works
    # ------------------------------------------------------------------
    def test_02_valid_class_input(self):
        """2. Test each valid class ID returns the expected configured relevance value."""
        for class_id, expected_value in CLASS_RELEVANCE_MAP.items():
            score = relevance_from_class(class_id)
            self.assertAlmostEqual(score, expected_value, places=6)

    # ------------------------------------------------------------------
    # 3. Invalid class IDs rejected
    # ------------------------------------------------------------------
    def test_03_invalid_class_ids_rejected(self):
        """3. Test out-of-range class IDs raise ValueError."""
        with self.assertRaises(ValueError):
            relevance_from_class(-1)

        with self.assertRaises(ValueError):
            relevance_from_class(4)

        with self.assertRaises(ValueError):
            relevance_from_class(100)

    # ------------------------------------------------------------------
    # 4. Invalid types rejected
    # ------------------------------------------------------------------
    def test_04_invalid_types_rejected(self):
        """4. Test non-integer class IDs raise TypeError."""
        with self.assertRaises(TypeError):
            relevance_from_class(1.5)

        with self.assertRaises(TypeError):
            relevance_from_class("1")

        with self.assertRaises(TypeError):
            relevance_from_class(None)

    # ------------------------------------------------------------------
    # 5. Valid probability vectors work
    # ------------------------------------------------------------------
    def test_05_valid_probability_vectors(self):
        """5. Test probability-weighted strategy with valid probability vectors."""
        # One-hot for class 1 (Inner Race Fault) -> R = 1.00
        probs = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        score = relevance_from_probabilities(probs)
        self.assertAlmostEqual(score, 1.0, places=5)

        # One-hot for class 0 (Normal) -> R = 0.10
        probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        score = relevance_from_probabilities(probs)
        self.assertAlmostEqual(score, 0.10, places=5)

        # Uniform distribution -> R = (0.10 + 1.00 + 0.90 + 0.90) / 4 = 0.725
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        score = relevance_from_probabilities(probs)
        expected = 0.25 * 0.10 + 0.25 * 1.00 + 0.25 * 0.90 + 0.25 * 0.90
        self.assertAlmostEqual(score, expected, places=4)

    # ------------------------------------------------------------------
    # 6. Invalid probability shapes rejected
    # ------------------------------------------------------------------
    def test_06_invalid_probability_shapes_rejected(self):
        """6. Test non-1D arrays or wrong-length arrays raise ValueError."""
        # 2D array
        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
            )

        # 3 elements instead of 4
        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([0.5, 0.3, 0.2], dtype=np.float32)
            )

        # 5 elements instead of 4
        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
            )

    # ------------------------------------------------------------------
    # 7. Negative / NaN / Inf probabilities rejected
    # ------------------------------------------------------------------
    def test_07_negative_nan_inf_probabilities_rejected(self):
        """7. Test negative, NaN, and Inf probability values raise ValueError."""
        # Negative
        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([-0.1, 0.5, 0.3, 0.3], dtype=np.float32)
            )

        # NaN
        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([0.25, 0.25, 0.25, np.nan], dtype=np.float32)
            )

        # Inf
        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([0.25, np.inf, 0.25, 0.25], dtype=np.float32)
            )

    # ------------------------------------------------------------------
    # 8. Probabilities not summing to ~1 rejected
    # ------------------------------------------------------------------
    def test_08_probabilities_not_summing_to_one_rejected(self):
        """8. Test probability vectors that do not sum to ~1.0 raise ValueError."""
        # Sum = 0.5
        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([0.1, 0.1, 0.1, 0.2], dtype=np.float32)
            )

        # Sum = 2.0
        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            )

    # ------------------------------------------------------------------
    # 9. Reproducibility: same input gives same output
    # ------------------------------------------------------------------
    def test_09_reproducibility(self):
        """9. Test deterministic execution produces identical scores across calls."""
        score1 = relevance_from_class(2)
        score2 = relevance_from_class(2)
        self.assertEqual(score1, score2)

        probs = np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32)
        score_p1 = relevance_from_probabilities(probs)
        score_p2 = relevance_from_probabilities(probs)
        self.assertEqual(score_p1, score_p2)

    # ------------------------------------------------------------------
    # 10. Different classes produce different relevance scores
    # ------------------------------------------------------------------
    def test_10_different_classes_produce_different_scores(self):
        """10. Test that Normal vs fault classes yield distinct relevance scores."""
        score_normal = relevance_from_class(0)
        score_inner = relevance_from_class(1)
        score_ball = relevance_from_class(2)
        score_outer = relevance_from_class(3)

        # Normal should be distinctly lower than all fault classes
        self.assertLess(score_normal, score_inner)
        self.assertLess(score_normal, score_ball)
        self.assertLess(score_normal, score_outer)

        # Inner Race Fault should be the highest
        self.assertGreaterEqual(score_inner, score_ball)
        self.assertGreaterEqual(score_inner, score_outer)

    # ------------------------------------------------------------------
    # 11. Class-mapping strategy works end-to-end
    # ------------------------------------------------------------------
    def test_11_class_mapping_strategy(self):
        """11. Test class_mapping strategy returns correct values with custom map."""
        custom_map = {0: 0.0, 1: 0.5, 2: 0.7, 3: 1.0}
        for class_id, expected in custom_map.items():
            score = relevance_from_class(
                class_id, relevance_map=custom_map, strategy="class_mapping"
            )
            self.assertAlmostEqual(score, expected, places=6)

    # ------------------------------------------------------------------
    # 12. Probability-weighted strategy works end-to-end
    # ------------------------------------------------------------------
    def test_12_probability_weighted_strategy(self):
        """12. Test probability_weighted strategy computes correct weighted sum."""
        custom_map = {0: 0.0, 1: 0.5, 2: 0.5, 3: 1.0}
        probs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        score = relevance_from_probabilities(
            probs, relevance_map=custom_map, strategy="probability_weighted"
        )

        expected = 0.1 * 0.0 + 0.2 * 0.5 + 0.3 * 0.5 + 0.4 * 1.0
        self.assertAlmostEqual(score, expected, places=4)

    # ------------------------------------------------------------------
    # 13. Invalid strategy rejected
    # ------------------------------------------------------------------
    def test_13_invalid_strategy_rejected(self):
        """13. Test that unsupported strategy names raise ValueError."""
        with self.assertRaises(ValueError):
            relevance_from_class(0, strategy="unsupported_strategy")

        with self.assertRaises(ValueError):
            relevance_from_probabilities(
                np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
                strategy="class_mapping",
            )

        with self.assertRaises(ValueError):
            relevance_from_class(0, strategy="probability_weighted")

    # ------------------------------------------------------------------
    # 14. Output remains in [0, 1] across probability-weighted edge cases
    # ------------------------------------------------------------------
    def test_14_probability_weighted_output_in_range(self):
        """14. Test probability-weighted scores remain in [0, 1] for diverse inputs."""
        # One-hot for each class
        for class_id in range(NUM_CLASSES):
            probs = np.zeros(NUM_CLASSES, dtype=np.float32)
            probs[class_id] = 1.0
            score = relevance_from_probabilities(probs)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertTrue(np.isfinite(score))

        # Various mixed distributions
        test_probs = [
            np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
            np.array([0.01, 0.97, 0.01, 0.01], dtype=np.float32),
            np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        ]
        for probs in test_probs:
            score = relevance_from_probabilities(probs)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertTrue(np.isfinite(score))

    # ------------------------------------------------------------------
    # Additional: Non-numpy input to probability function
    # ------------------------------------------------------------------
    def test_15_non_numpy_probability_input_rejected(self):
        """15. Test that non-numpy probability inputs raise TypeError."""
        with self.assertRaises(TypeError):
            relevance_from_probabilities([0.25, 0.25, 0.25, 0.25])

        with self.assertRaises(TypeError):
            relevance_from_probabilities(0.5)

    # ------------------------------------------------------------------
    # Additional: Invalid relevance map
    # ------------------------------------------------------------------
    def test_16_invalid_relevance_map_rejected(self):
        """16. Test that malformed relevance maps raise appropriate errors."""
        # Missing a class
        incomplete_map = {0: 0.1, 1: 1.0, 2: 0.9}
        with self.assertRaises(ValueError):
            relevance_from_class(0, relevance_map=incomplete_map)

        # Value out of [0, 1]
        out_of_range_map = {0: 0.1, 1: 1.5, 2: 0.9, 3: 0.9}
        with self.assertRaises(ValueError):
            relevance_from_class(0, relevance_map=out_of_range_map)

        # Non-dict
        with self.assertRaises(TypeError):
            relevance_from_class(0, relevance_map=[0.1, 1.0, 0.9, 0.9])

    # ------------------------------------------------------------------
    # Additional: numpy integer class IDs accepted
    # ------------------------------------------------------------------
    def test_17_numpy_integer_class_ids_accepted(self):
        """17. Test that numpy integer types are accepted as class IDs."""
        for np_int_type in [np.int32, np.int64, np.uint8]:
            class_id = np_int_type(1)
            score = relevance_from_class(class_id)
            self.assertAlmostEqual(score, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
