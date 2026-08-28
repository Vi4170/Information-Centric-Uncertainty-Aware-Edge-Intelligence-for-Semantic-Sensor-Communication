"""Unit test suite for baseline Temporal Importance module (src/temporal/).

Tests cover stable / constant / sudden-change sequences, output shape and
range constraints, input validation (type, dimensionality, NaN/Inf, empty),
normalization-scale validation, reproducibility, score differentiation, and
clipping behaviour.
"""

import unittest

import numpy as np

from src.temporal.config import DEFAULT_TEMPORAL_CHANGE_SCALE
from src.temporal.temporal import compute_temporal_importance


class TestTemporalImportance(unittest.TestCase):
    """Test suite for compute_temporal_importance."""

    def setUp(self) -> None:
        np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Stable sequence produces low/zero temporal importance
    # ------------------------------------------------------------------
    def test_01_stable_sequence_low_importance(self) -> None:
        """1. A slowly-varying (nearly constant) sequence produces near-zero scores."""
        base = np.ones((10, 2048), dtype=np.float64) * 0.5
        # Add tiny noise so the signal is *nearly* constant, not exactly
        noise = np.random.normal(0, 1e-8, size=base.shape)
        observations = base + noise

        scores = compute_temporal_importance(observations)

        for s in scores:
            self.assertAlmostEqual(s, 0.0, places=5)

    # ------------------------------------------------------------------
    # 2. Constant sequence produces exactly zero scores
    # ------------------------------------------------------------------
    def test_02_constant_sequence_zero_scores(self) -> None:
        """2. Identical consecutive observations produce all-zero scores."""
        observations = np.ones((8, 100), dtype=np.float64) * 3.14

        scores = compute_temporal_importance(observations)

        np.testing.assert_array_equal(scores, np.zeros(8))

    # ------------------------------------------------------------------
    # 3. Sudden change produces higher score than small change
    # ------------------------------------------------------------------
    def test_03_sudden_vs_small_change(self) -> None:
        """3. A large step between observations yields a higher score than a small step."""
        scale = DEFAULT_TEMPORAL_CHANGE_SCALE
        obs = np.zeros((3, 10), dtype=np.float64)
        obs[1, :] = scale * 0.1   # small change
        obs[2, :] = scale * 0.9   # large change (relative to obs[1])

        scores = compute_temporal_importance(obs, temporal_change_scale=scale)

        self.assertGreater(scores[2], scores[1])

    # ------------------------------------------------------------------
    # 4. Output shape is correct
    # ------------------------------------------------------------------
    def test_04_output_shape(self) -> None:
        """4. Output length matches number of input observations."""
        for n in [1, 2, 5, 20]:
            obs = np.random.randn(n, 50)
            scores = compute_temporal_importance(obs)
            self.assertEqual(scores.shape, (n,))

    # ------------------------------------------------------------------
    # 5. All outputs are within [0, 1]
    # ------------------------------------------------------------------
    def test_05_outputs_in_range(self) -> None:
        """5. All scores lie in [0, 1] for diverse random inputs."""
        for _ in range(10):
            obs = np.random.randn(20, 64) * 10
            scores = compute_temporal_importance(obs, temporal_change_scale=1.0)
            self.assertTrue(np.all(scores >= 0.0))
            self.assertTrue(np.all(scores <= 1.0))
            self.assertTrue(np.isfinite(scores).all())

    # ------------------------------------------------------------------
    # 6. Single observation returns [0.0]
    # ------------------------------------------------------------------
    def test_06_single_observation(self) -> None:
        """6. A single observation has no predecessor → T = 0.0."""
        obs = np.array([[1.0, 2.0, 3.0]])
        scores = compute_temporal_importance(obs)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0], 0.0)

    # ------------------------------------------------------------------
    # 7. Very short sequences work
    # ------------------------------------------------------------------
    def test_07_short_sequences(self) -> None:
        """7. Two-observation sequences produce correct [0.0, T] output."""
        obs = np.array([[0.0, 0.0], [1.0, 1.0]])
        scores = compute_temporal_importance(obs, temporal_change_scale=1.0)
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[1], 1.0, places=6)

    # ------------------------------------------------------------------
    # 8. Empty input is rejected
    # ------------------------------------------------------------------
    def test_08_empty_input_rejected(self) -> None:
        """8. A zero-row array raises ValueError."""
        with self.assertRaises(ValueError):
            compute_temporal_importance(np.empty((0, 10)))

    # ------------------------------------------------------------------
    # 9. Wrong input type is rejected
    # ------------------------------------------------------------------
    def test_09_wrong_type_rejected(self) -> None:
        """9. Non-ndarray inputs raise TypeError."""
        with self.assertRaises(TypeError):
            compute_temporal_importance([[1, 2], [3, 4]])

        with self.assertRaises(TypeError):
            compute_temporal_importance("not an array")

        with self.assertRaises(TypeError):
            compute_temporal_importance(42)

    # ------------------------------------------------------------------
    # 10. Wrong dimensionality is rejected
    # ------------------------------------------------------------------
    def test_10_wrong_dimensionality_rejected(self) -> None:
        """10. 1D and 3D arrays raise ValueError."""
        with self.assertRaises(ValueError):
            compute_temporal_importance(np.array([1.0, 2.0, 3.0]))

        with self.assertRaises(ValueError):
            compute_temporal_importance(np.ones((2, 3, 4)))

    # ------------------------------------------------------------------
    # 11. NaN values are rejected
    # ------------------------------------------------------------------
    def test_11_nan_values_rejected(self) -> None:
        """11. Observations containing NaN raise ValueError."""
        obs = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with self.assertRaises(ValueError):
            compute_temporal_importance(obs)

    # ------------------------------------------------------------------
    # 12. Inf values are rejected
    # ------------------------------------------------------------------
    def test_12_inf_values_rejected(self) -> None:
        """12. Observations containing Inf or -Inf raise ValueError."""
        obs = np.array([[1.0, 2.0], [np.inf, 4.0]])
        with self.assertRaises(ValueError):
            compute_temporal_importance(obs)

        obs_neg = np.array([[1.0, 2.0], [-np.inf, 4.0]])
        with self.assertRaises(ValueError):
            compute_temporal_importance(obs_neg)

    # ------------------------------------------------------------------
    # 13. Invalid normalization scale is rejected
    # ------------------------------------------------------------------
    def test_13_invalid_scale_rejected(self) -> None:
        """13. Non-positive, non-finite, and non-numeric scales raise errors."""
        obs = np.ones((3, 4))

        with self.assertRaises(ValueError):
            compute_temporal_importance(obs, temporal_change_scale=0.0)

        with self.assertRaises(ValueError):
            compute_temporal_importance(obs, temporal_change_scale=-1.0)

        with self.assertRaises(ValueError):
            compute_temporal_importance(obs, temporal_change_scale=np.inf)

        with self.assertRaises(ValueError):
            compute_temporal_importance(obs, temporal_change_scale=np.nan)

        with self.assertRaises(TypeError):
            compute_temporal_importance(obs, temporal_change_scale="bad")

    # ------------------------------------------------------------------
    # 14. Same input produces the same output
    # ------------------------------------------------------------------
    def test_14_reproducibility(self) -> None:
        """14. Deterministic computation — identical inputs give identical outputs."""
        obs = np.random.randn(10, 32)
        scores_a = compute_temporal_importance(obs)
        scores_b = compute_temporal_importance(obs)
        np.testing.assert_array_equal(scores_a, scores_b)

    # ------------------------------------------------------------------
    # 15. Different temporal patterns produce different scores
    # ------------------------------------------------------------------
    def test_15_different_patterns_different_scores(self) -> None:
        """15. Distinct temporal patterns yield distinct score arrays."""
        constant = np.ones((5, 10))
        varying = np.zeros((5, 10))
        varying[1, :] = 1.0
        varying[3, :] = 2.0

        scores_const = compute_temporal_importance(constant)
        scores_vary = compute_temporal_importance(varying)

        self.assertFalse(np.array_equal(scores_const, scores_vary))

    # ------------------------------------------------------------------
    # 16. Scores clipped to [0, 1] for changes larger than scale
    # ------------------------------------------------------------------
    def test_16_clipping_large_changes(self) -> None:
        """16. Changes exceeding the scale saturate at T = 1.0."""
        scale = 0.1
        obs = np.zeros((3, 10))
        obs[1, :] = 10.0   # change = 10.0 >> scale = 0.1
        obs[2, :] = 0.0    # change = 10.0 >> scale

        scores = compute_temporal_importance(obs, temporal_change_scale=scale)

        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 1.0)
        self.assertEqual(scores[2], 1.0)

    # ------------------------------------------------------------------
    # 17. Zero-change consecutive windows produce exactly zero
    # ------------------------------------------------------------------
    def test_17_zero_change_windows_exactly_zero(self) -> None:
        """17. Where two consecutive observations are identical, that score is exactly 0."""
        obs = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],   # same as [0] → T = 0
            [4.0, 5.0, 6.0],   # different → T > 0
            [4.0, 5.0, 6.0],   # same as [2] → T = 0
        ])

        scores = compute_temporal_importance(obs)

        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)
        self.assertGreater(scores[2], 0.0)
        self.assertEqual(scores[3], 0.0)

    # ------------------------------------------------------------------
    # Additional: scalar observations (N, 1) shape
    # ------------------------------------------------------------------
    def test_18_scalar_observations(self) -> None:
        """18. Scalar-observation sequences with shape (N, 1) work correctly."""
        obs = np.array([[0.0], [0.1], [0.2], [10.0]])
        scores = compute_temporal_importance(obs, temporal_change_scale=0.5)

        self.assertEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[1], 0.1 / 0.5, places=6)
        self.assertAlmostEqual(scores[2], 0.1 / 0.5, places=6)
        self.assertEqual(scores[3], 1.0)  # 9.8 / 0.5 >> 1 → clipped

    # ------------------------------------------------------------------
    # Additional: first observation is always 0.0
    # ------------------------------------------------------------------
    def test_19_first_observation_always_zero(self) -> None:
        """19. Regardless of values, the first observation's score is 0.0."""
        for _ in range(5):
            obs = np.random.randn(np.random.randint(1, 50), 16)
            scores = compute_temporal_importance(obs)
            self.assertEqual(scores[0], 0.0)

    # ------------------------------------------------------------------
    # Additional: empty column dimension rejected
    # ------------------------------------------------------------------
    def test_20_empty_column_dimension_rejected(self) -> None:
        """20. An array with 0 columns (observation_size = 0) raises ValueError."""
        with self.assertRaises(ValueError):
            compute_temporal_importance(np.empty((5, 0)))


if __name__ == "__main__":
    unittest.main()
