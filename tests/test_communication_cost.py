"""Unit test suite for baseline Communication Cost module (src/communication/).

Tests cover output range, boundary behaviour, monotonicity with respect to
each input, input validation (type, finiteness, sign), weight validation,
reference-limit validation, reproducibility, clipping, and score
differentiation.
"""

import unittest

import numpy as np

from src.communication.config import (
    MAX_PAYLOAD_SIZE,
    MAX_TRANSMISSION_TIME,
    REFERENCE_BANDWIDTH,
    WEIGHT_BANDWIDTH,
    WEIGHT_SIZE,
    WEIGHT_TIME,
)
from src.communication.cost import compute_communication_cost


class TestCommunicationCost(unittest.TestCase):
    """Test suite for compute_communication_cost."""

    def setUp(self) -> None:
        np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Output is always in [0, 1]
    # ------------------------------------------------------------------
    def test_01_output_in_range(self) -> None:
        """1. Cost score is within [0, 1] for diverse valid inputs."""
        test_cases = [
            (0, 0, REFERENCE_BANDWIDTH),
            (MAX_PAYLOAD_SIZE, MAX_TRANSMISSION_TIME, 0),
            (MAX_PAYLOAD_SIZE / 2, MAX_TRANSMISSION_TIME / 2, REFERENCE_BANDWIDTH / 2),
            (MAX_PAYLOAD_SIZE * 2, MAX_TRANSMISSION_TIME * 3, 0),
            (100, 0.01, REFERENCE_BANDWIDTH * 0.9),
        ]
        for ps, tt, bw in test_cases:
            cost = compute_communication_cost(ps, tt, bw)
            self.assertGreaterEqual(cost, 0.0, msg=f"Failed for ({ps}, {tt}, {bw})")
            self.assertLessEqual(cost, 1.0, msg=f"Failed for ({ps}, {tt}, {bw})")
            self.assertTrue(np.isfinite(cost))

    # ------------------------------------------------------------------
    # 2. Zero / minimal communication cost
    # ------------------------------------------------------------------
    def test_02_zero_cost(self) -> None:
        """2. Zero payload, zero time, full bandwidth → C = 0."""
        cost = compute_communication_cost(0, 0, REFERENCE_BANDWIDTH)
        self.assertEqual(cost, 0.0)

    # ------------------------------------------------------------------
    # 3. Higher payload size gives higher cost
    # ------------------------------------------------------------------
    def test_03_higher_payload_higher_cost(self) -> None:
        """3. Increasing payload size increases cost (all else equal)."""
        bw = REFERENCE_BANDWIDTH
        tt = 0.0
        cost_small = compute_communication_cost(1000, tt, bw)
        cost_large = compute_communication_cost(10_000, tt, bw)
        self.assertGreater(cost_large, cost_small)

    # ------------------------------------------------------------------
    # 4. Higher transmission time gives higher cost
    # ------------------------------------------------------------------
    def test_04_higher_time_higher_cost(self) -> None:
        """4. Increasing transmission time increases cost (all else equal)."""
        bw = REFERENCE_BANDWIDTH
        ps = 0
        cost_short = compute_communication_cost(ps, 0.1, bw)
        cost_long = compute_communication_cost(ps, 0.9, bw)
        self.assertGreater(cost_long, cost_short)

    # ------------------------------------------------------------------
    # 5. Lower available bandwidth gives higher cost
    # ------------------------------------------------------------------
    def test_05_lower_bandwidth_higher_cost(self) -> None:
        """5. Decreasing available bandwidth increases cost (all else equal)."""
        ps = 0
        tt = 0.0
        cost_full = compute_communication_cost(ps, tt, REFERENCE_BANDWIDTH)
        cost_half = compute_communication_cost(ps, tt, REFERENCE_BANDWIDTH * 0.5)
        cost_none = compute_communication_cost(ps, tt, 0)
        self.assertGreater(cost_none, cost_half)
        self.assertGreater(cost_half, cost_full)

    # ------------------------------------------------------------------
    # 6. Valid numeric inputs work
    # ------------------------------------------------------------------
    def test_06_valid_numeric_inputs(self) -> None:
        """6. Integer and float inputs are accepted without error."""
        # Python int
        cost_int = compute_communication_cost(1000, 0, 500_000)
        self.assertIsInstance(cost_int, float)

        # Python float
        cost_float = compute_communication_cost(1000.0, 0.5, 500_000.0)
        self.assertIsInstance(cost_float, float)

        # NumPy numeric types
        cost_np = compute_communication_cost(
            np.float32(1000), np.int64(0), np.float64(500_000)
        )
        self.assertIsInstance(cost_np, float)

    # ------------------------------------------------------------------
    # 7. Negative payload rejected
    # ------------------------------------------------------------------
    def test_07_negative_payload_rejected(self) -> None:
        """7. Negative payload_size raises ValueError."""
        with self.assertRaises(ValueError):
            compute_communication_cost(-1, 0.0, REFERENCE_BANDWIDTH)

    # ------------------------------------------------------------------
    # 8. Negative transmission time rejected
    # ------------------------------------------------------------------
    def test_08_negative_time_rejected(self) -> None:
        """8. Negative transmission_time raises ValueError."""
        with self.assertRaises(ValueError):
            compute_communication_cost(0, -0.1, REFERENCE_BANDWIDTH)

    # ------------------------------------------------------------------
    # 9. Negative bandwidth rejected
    # ------------------------------------------------------------------
    def test_09_negative_bandwidth_rejected(self) -> None:
        """9. Negative available_bandwidth raises ValueError."""
        with self.assertRaises(ValueError):
            compute_communication_cost(0, 0, -100)

    # ------------------------------------------------------------------
    # 10. NaN rejected
    # ------------------------------------------------------------------
    def test_10_nan_rejected(self) -> None:
        """10. NaN values in any input raise ValueError."""
        with self.assertRaises(ValueError):
            compute_communication_cost(np.nan, 0, REFERENCE_BANDWIDTH)

        with self.assertRaises(ValueError):
            compute_communication_cost(0, np.nan, REFERENCE_BANDWIDTH)

        with self.assertRaises(ValueError):
            compute_communication_cost(0, 0, np.nan)

    # ------------------------------------------------------------------
    # 11. Inf rejected
    # ------------------------------------------------------------------
    def test_11_inf_rejected(self) -> None:
        """11. Inf values in any input raise ValueError."""
        with self.assertRaises(ValueError):
            compute_communication_cost(np.inf, 0, REFERENCE_BANDWIDTH)

        with self.assertRaises(ValueError):
            compute_communication_cost(0, np.inf, REFERENCE_BANDWIDTH)

        with self.assertRaises(ValueError):
            compute_communication_cost(0, 0, np.inf)

    # ------------------------------------------------------------------
    # 12. Invalid zero/negative reference limits rejected
    # ------------------------------------------------------------------
    def test_12_invalid_reference_limits_rejected(self) -> None:
        """12. Zero or negative reference limits raise ValueError."""
        with self.assertRaises(ValueError):
            compute_communication_cost(0, 0, REFERENCE_BANDWIDTH, max_payload_size=0)

        with self.assertRaises(ValueError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH, max_payload_size=-1
            )

        with self.assertRaises(ValueError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH, max_transmission_time=0
            )

        with self.assertRaises(ValueError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH, reference_bandwidth=0
            )

        with self.assertRaises(ValueError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH, reference_bandwidth=-500
            )

    # ------------------------------------------------------------------
    # 13. Invalid weights rejected
    # ------------------------------------------------------------------
    def test_13_invalid_weights_rejected(self) -> None:
        """13. Non-numeric, negative, or > 1 weights raise errors."""
        with self.assertRaises(TypeError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH, weight_size="bad"
            )

        with self.assertRaises(ValueError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH, weight_size=-0.1, weight_time=0.6, weight_bandwidth=0.5
            )

        with self.assertRaises(ValueError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH, weight_size=1.5, weight_time=0.0, weight_bandwidth=0.0
            )

    # ------------------------------------------------------------------
    # 14. Weights not summing to 1 rejected
    # ------------------------------------------------------------------
    def test_14_weights_not_summing_to_one_rejected(self) -> None:
        """14. Weights that do not sum to ~1.0 raise ValueError."""
        with self.assertRaises(ValueError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH,
                weight_size=0.5, weight_time=0.5, weight_bandwidth=0.5,
            )

        with self.assertRaises(ValueError):
            compute_communication_cost(
                0, 0, REFERENCE_BANDWIDTH,
                weight_size=0.1, weight_time=0.1, weight_bandwidth=0.1,
            )

    # ------------------------------------------------------------------
    # 15. Reproducibility
    # ------------------------------------------------------------------
    def test_15_reproducibility(self) -> None:
        """15. Same inputs always produce the same output."""
        args = (5000, 0.3, REFERENCE_BANDWIDTH * 0.7)
        cost_a = compute_communication_cost(*args)
        cost_b = compute_communication_cost(*args)
        self.assertEqual(cost_a, cost_b)

    # ------------------------------------------------------------------
    # 16. Large values are clipped and remain in [0, 1]
    # ------------------------------------------------------------------
    def test_16_clipping_large_values(self) -> None:
        """16. Inputs exceeding reference limits saturate components at 1."""
        # payload 10× max, time 10× max, zero bandwidth
        cost = compute_communication_cost(
            MAX_PAYLOAD_SIZE * 10,
            MAX_TRANSMISSION_TIME * 10,
            0,
        )
        self.assertEqual(cost, 1.0)

    # ------------------------------------------------------------------
    # 17. Boundary values behave correctly
    # ------------------------------------------------------------------
    def test_17_boundary_values(self) -> None:
        """17. Exact boundary inputs produce expected component values."""
        # Payload exactly at max, others at zero-cost
        cost_s = compute_communication_cost(
            MAX_PAYLOAD_SIZE, 0, REFERENCE_BANDWIDTH,
        )
        self.assertAlmostEqual(cost_s, WEIGHT_SIZE * 1.0, places=6)

        # Time exactly at max, others at zero-cost
        cost_t = compute_communication_cost(
            0, MAX_TRANSMISSION_TIME, REFERENCE_BANDWIDTH,
        )
        self.assertAlmostEqual(cost_t, WEIGHT_TIME * 1.0, places=6)

        # Bandwidth exactly zero, others at zero-cost
        cost_b = compute_communication_cost(0, 0, 0)
        self.assertAlmostEqual(cost_b, WEIGHT_BANDWIDTH * 1.0, places=6)

    # ------------------------------------------------------------------
    # 18. Different communication conditions produce different costs
    # ------------------------------------------------------------------
    def test_18_different_conditions_different_costs(self) -> None:
        """18. Distinct operating conditions yield distinct cost scores."""
        cost_low = compute_communication_cost(100, 0.01, REFERENCE_BANDWIDTH)
        cost_mid = compute_communication_cost(
            MAX_PAYLOAD_SIZE / 2, 0.5, REFERENCE_BANDWIDTH / 2
        )
        cost_high = compute_communication_cost(MAX_PAYLOAD_SIZE, 1.0, 0)

        self.assertLess(cost_low, cost_mid)
        self.assertLess(cost_mid, cost_high)

    # ------------------------------------------------------------------
    # Additional: non-numeric inputs rejected
    # ------------------------------------------------------------------
    def test_19_non_numeric_inputs_rejected(self) -> None:
        """19. String, None, and list inputs raise TypeError."""
        with self.assertRaises(TypeError):
            compute_communication_cost("100", 0, REFERENCE_BANDWIDTH)

        with self.assertRaises(TypeError):
            compute_communication_cost(100, None, REFERENCE_BANDWIDTH)

        with self.assertRaises(TypeError):
            compute_communication_cost(100, 0, [REFERENCE_BANDWIDTH])

    # ------------------------------------------------------------------
    # Additional: max-cost scenario
    # ------------------------------------------------------------------
    def test_20_max_cost_scenario(self) -> None:
        """20. Payload=max, time=max, bandwidth=0 → C = 1.0."""
        cost = compute_communication_cost(MAX_PAYLOAD_SIZE, MAX_TRANSMISSION_TIME, 0)
        self.assertEqual(cost, 1.0)

    # ------------------------------------------------------------------
    # Additional: bandwidth exceeding reference is clipped to B = 0
    # ------------------------------------------------------------------
    def test_21_excess_bandwidth_clipped(self) -> None:
        """21. Bandwidth above the reference yields B = 0 (no pressure)."""
        cost_ref = compute_communication_cost(1000, 0.1, REFERENCE_BANDWIDTH)
        cost_excess = compute_communication_cost(1000, 0.1, REFERENCE_BANDWIDTH * 5)
        # Bandwidth component is 0 in both cases (clipped)
        self.assertEqual(cost_ref, cost_excess)

    # ------------------------------------------------------------------
    # Additional: custom weights accepted
    # ------------------------------------------------------------------
    def test_22_custom_weights(self) -> None:
        """22. Custom weights that sum to 1 are accepted and used."""
        # All weight on size
        cost = compute_communication_cost(
            MAX_PAYLOAD_SIZE, MAX_TRANSMISSION_TIME, 0,
            weight_size=1.0, weight_time=0.0, weight_bandwidth=0.0,
        )
        self.assertAlmostEqual(cost, 1.0, places=6)

        # All weight on bandwidth
        cost_bw = compute_communication_cost(
            MAX_PAYLOAD_SIZE, MAX_TRANSMISSION_TIME, REFERENCE_BANDWIDTH,
            weight_size=0.0, weight_time=0.0, weight_bandwidth=1.0,
        )
        self.assertAlmostEqual(cost_bw, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
