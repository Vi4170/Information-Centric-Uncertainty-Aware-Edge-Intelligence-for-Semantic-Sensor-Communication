"""Unit test suite for baseline Novelty Detection module (src/novelty/).

Tests cover fitting, scoring, output shapes, [0, 1] range constraints,
reproducibility, input validation, 64-D embedding dimension enforcement,
and training-only reference fitting.
"""

import unittest
import numpy as np

from src.novelty.config import EMBEDDING_DIM
from src.novelty.novelty import DistanceNoveltyDetector


class TestNoveltyDetector(unittest.TestCase):
    """Test suite for DistanceNoveltyDetector."""

    def setUp(self):
        np.random.seed(42)
        self.detector = DistanceNoveltyDetector(embedding_dim=EMBEDDING_DIM)
        self.synthetic_train = np.random.randn(100, EMBEDDING_DIM).astype(np.float32)
        self.synthetic_test = np.random.randn(20, EMBEDDING_DIM).astype(np.float32)

    def test_01_fitting_and_is_fitted_flag(self):
        """1. Test fitting initializes reference centroid and sets is_fitted flag."""
        self.assertFalse(self.detector.is_fitted)

        # Calling score before fit should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.detector.score(self.synthetic_test)

        self.detector.fit(self.synthetic_train)
        self.assertTrue(self.detector.is_fitted)
        self.assertIsNotNone(self.detector.reference_centroid)
        self.assertEqual(self.detector.reference_centroid.shape, (EMBEDDING_DIM,))

    def test_02_output_shape_and_range(self):
        """2. Test scoring produces 1D array of shape (N,) with values in [0, 1]."""
        self.detector.fit(self.synthetic_train)
        scores = self.detector.score(self.synthetic_test)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (20,))
        self.assertEqual(scores.dtype, np.float32)
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_03_reproducibility(self):
        """3. Test deterministic execution yields identical scores across identical inputs."""
        detector1 = DistanceNoveltyDetector().fit(self.synthetic_train)
        scores1 = detector1.score(self.synthetic_test)

        detector2 = DistanceNoveltyDetector().fit(self.synthetic_train)
        scores2 = detector2.score(self.synthetic_test)

        np.testing.assert_array_equal(scores1, scores2)

    def test_04_embedding_dimension_validation(self):
        """4. Test that non-64D embeddings raise ValueError."""
        wrong_dim_train = np.random.randn(50, 32).astype(np.float32)
        with self.assertRaises(ValueError):
            self.detector.fit(wrong_dim_train)

        self.detector.fit(self.synthetic_train)
        wrong_dim_test = np.random.randn(10, 128).astype(np.float32)
        with self.assertRaises(ValueError):
            self.detector.score(wrong_dim_test)

    def test_05_invalid_input_handling(self):
        """5. Test NaN, Inf, non-numpy arrays, and 1D/3D shapes are rejected."""
        # Non-array input
        with self.assertRaises(TypeError):
            self.detector.fit([[1.0] * 64])

        # NaN input
        nan_emb = self.synthetic_train.copy()
        nan_emb[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.detector.fit(nan_emb)

        # Inf input
        inf_emb = self.synthetic_train.copy()
        inf_emb[0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.detector.fit(inf_emb)

        # 1D array
        with self.assertRaises(ValueError):
            self.detector.fit(np.ones(64, dtype=np.float32))

    def test_06_reference_class_filtering(self):
        """6. Test fitting with class label filtering anchors reference to normal class."""
        train_labels = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        detector = DistanceNoveltyDetector(reference_class=0)
        detector.fit(self.synthetic_train, train_labels)

        # Centroid should equal mean of first 50 samples
        expected_centroid = np.mean(self.synthetic_train[:50], axis=0, dtype=np.float32)
        np.testing.assert_allclose(detector.reference_centroid, expected_centroid, atol=1e-5)

    def test_07_higher_distance_yields_higher_novelty(self):
        """7. Test that embeddings further from reference centroid receive higher novelty scores."""
        self.detector.fit(self.synthetic_train)

        near_point = self.detector.reference_centroid.reshape(1, -1)
        far_point = (self.detector.reference_centroid + 10.0).reshape(1, -1)

        near_score = self.detector.score(near_point)[0]
        far_score = self.detector.score(far_point)[0]

        self.assertLess(near_score, far_score)
        self.assertAlmostEqual(near_score, 0.0, places=4)
        self.assertAlmostEqual(far_score, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
