"""Unit test suite for baseline Uncertainty Estimation module (src/uncertainty/).

Tests cover predictive entropy calculations, output shapes, [0, 1] range constraints,
confident vs ambiguous predictions, shape validation, out-of-range/NaN/Inf rejection,
and reproducibility.
"""

import unittest
import numpy as np

from src.uncertainty.config import NUM_CLASSES
from src.uncertainty.uncertainty import (
    EntropyUncertaintyEstimator,
    compute_predictive_entropy,
    validate_probabilities,
)


class TestUncertaintyEstimator(unittest.TestCase):
    """Test suite for EntropyUncertaintyEstimator and compute_predictive_entropy."""

    def setUp(self):
        np.random.seed(42)
        self.estimator = EntropyUncertaintyEstimator(num_classes=NUM_CLASSES)

    def test_01_correct_entropy_calculation(self):
        """1. Test exact entropy values for one-hot (0.0) and uniform (1.0) distributions."""
        one_hot = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        score_one_hot = compute_predictive_entropy(one_hot)[0]
        self.assertAlmostEqual(score_one_hot, 0.0, places=5)

        uniform = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
        score_uniform = compute_predictive_entropy(uniform)[0]
        self.assertAlmostEqual(score_uniform, 1.0, places=5)

    def test_02_output_shape(self):
        """2. Test scoring produces 1D array of shape (N,)."""
        probs = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.9, 0.05, 0.03, 0.02],
        ], dtype=np.float32)

        scores = compute_predictive_entropy(probs)
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(scores.dtype, np.float32)

    def test_03_output_range_zero_to_one(self):
        """3. Test all output uncertainty scores are in [0, 1] and finite."""
        probs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        scores = self.estimator.score(probs)
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_04_confident_prediction_gives_low_uncertainty(self):
        """4. Test that highly confident probabilities give near-zero uncertainty."""
        confident_prob = np.array([[0.99, 0.005, 0.003, 0.002]], dtype=np.float32)
        score = compute_predictive_entropy(confident_prob)[0]

        self.assertLess(score, 0.10)

    def test_05_more_even_distribution_gives_higher_uncertainty(self):
        """5. Test that more ambiguous distributions give strictly higher uncertainty."""
        confident = np.array([[0.90, 0.05, 0.03, 0.02]], dtype=np.float32)
        ambiguous = np.array([[0.40, 0.30, 0.20, 0.10]], dtype=np.float32)

        score_confident = compute_predictive_entropy(confident)[0]
        score_ambiguous = compute_predictive_entropy(ambiguous)[0]

        self.assertGreater(score_ambiguous, score_confident)

    def test_06_invalid_probability_shapes_rejected(self):
        """6. Test non-2D arrays or arrays not having 4 columns raise ValueError."""
        # 1D array
        with self.assertRaises(ValueError):
            validate_probabilities(np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32))

        # 3 classes instead of 4
        with self.assertRaises(ValueError):
            validate_probabilities(np.array([[0.5, 0.3, 0.2]], dtype=np.float32))

        # 5 classes instead of 4
        with self.assertRaises(ValueError):
            validate_probabilities(np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32))

    def test_07_negative_or_out_of_range_probabilities_rejected(self):
        """7. Test negative values or values > 1.0 raise ValueError."""
        negative_prob = np.array([[-0.1, 0.5, 0.3, 0.3]], dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_probabilities(negative_prob)

        greater_than_one = np.array([[1.5, -0.2, 0.0, 0.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_probabilities(greater_than_one)

    def test_08_nan_or_inf_probabilities_rejected(self):
        """8. Test NaN or Inf values raise ValueError."""
        nan_prob = np.array([[0.25, 0.25, 0.25, np.nan]], dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_probabilities(nan_prob)

        inf_prob = np.array([[0.25, np.inf, 0.25, 0.25]], dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_probabilities(inf_prob)

    def test_09_reproducibility_and_determinism(self):
        """9. Test deterministic execution produces identical output across calls."""
        probs = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.3, 0.3, 0.2, 0.2],
        ], dtype=np.float32)

        scores1 = compute_predictive_entropy(probs)
        scores2 = compute_predictive_entropy(probs)

        np.testing.assert_array_equal(scores1, scores2)


if __name__ == "__main__":
    unittest.main()