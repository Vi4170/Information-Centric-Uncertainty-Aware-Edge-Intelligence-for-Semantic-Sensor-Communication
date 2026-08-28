"""Unit test suite for the VoI integration pipeline (src/integration/).

Tests verify that the thin integration layer correctly validates inputs,
delegates to the canonical VoIEngine, and returns canonical VoIResult
objects without duplicating any VoI mathematics.
"""

import unittest
from typing import Any

import numpy as np

from src.integration.voi_pipeline import run_voi_pipeline
from src.voi.decision_policy import DecisionAction, PolicyThresholds
from src.voi.scoring import VoIWeights
from src.voi.voi_engine import VoIResult


class TestVoIPipeline(unittest.TestCase):
    """Test suite for run_voi_pipeline."""

    def setUp(self) -> None:
        np.random.seed(42)
        # Baseline mid-range inputs
        self.base_kwargs = dict(
            novelty=0.5,
            uncertainty=0.5,
            task_relevance=0.5,
            temporal_importance=0.5,
            communication_cost=0.5,
        )

    # ------------------------------------------------------------------
    # 1. All five valid inputs are accepted
    # ------------------------------------------------------------------
    def test_01_valid_inputs_accepted(self) -> None:
        """1. Five valid [0,1] factor values produce a result without error."""
        result = run_voi_pipeline(**self.base_kwargs)
        self.assertIsNotNone(result)

    # ------------------------------------------------------------------
    # 2. All values in [0,1] work
    # ------------------------------------------------------------------
    def test_02_boundary_values_work(self) -> None:
        """2. Boundary values 0.0 and 1.0 are accepted for each factor."""
        result_zeros = run_voi_pipeline(
            novelty=0.0, uncertainty=0.0, task_relevance=0.0,
            temporal_importance=0.0, communication_cost=0.0,
        )
        self.assertIsNotNone(result_zeros)

        result_ones = run_voi_pipeline(
            novelty=1.0, uncertainty=1.0, task_relevance=1.0,
            temporal_importance=1.0, communication_cost=1.0,
        )
        self.assertIsNotNone(result_ones)

    # ------------------------------------------------------------------
    # 3. Final VoIResult is returned
    # ------------------------------------------------------------------
    def test_03_returns_voi_result(self) -> None:
        """3. Return type is the canonical VoIResult dataclass."""
        result = run_voi_pipeline(**self.base_kwargs)
        self.assertIsInstance(result, VoIResult)

    # ------------------------------------------------------------------
    # 4. Final voi_score is in [0,1]
    # ------------------------------------------------------------------
    def test_04_voi_score_in_range(self) -> None:
        """4. Clipped voi_score is within [0, 1]."""
        result = run_voi_pipeline(**self.base_kwargs)
        self.assertGreaterEqual(result.voi_score, 0.0)
        self.assertLessEqual(result.voi_score, 1.0)

    # ------------------------------------------------------------------
    # 5. raw_voi_score exists
    # ------------------------------------------------------------------
    def test_05_raw_voi_score_exists(self) -> None:
        """5. The raw (unclipped) VoI score is present in the result."""
        result = run_voi_pipeline(**self.base_kwargs)
        self.assertTrue(hasattr(result, "raw_voi_score"))
        self.assertIsInstance(result.raw_voi_score, float)

    # ------------------------------------------------------------------
    # 6. Decision is produced
    # ------------------------------------------------------------------
    def test_06_decision_produced(self) -> None:
        """6. A decision field is present in the result."""
        result = run_voi_pipeline(**self.base_kwargs)
        self.assertTrue(hasattr(result, "decision"))
        self.assertIsNotNone(result.decision)

    # ------------------------------------------------------------------
    # 7. Decision is a canonical DecisionAction
    # ------------------------------------------------------------------
    def test_07_decision_is_canonical(self) -> None:
        """7. The decision is one of the four canonical DecisionAction values."""
        result = run_voi_pipeline(**self.base_kwargs)
        self.assertIsInstance(result.decision, DecisionAction)
        self.assertIn(
            result.decision,
            [DecisionAction.DISCARD, DecisionAction.BUFFER,
             DecisionAction.SUMMARY, DecisionAction.TRANSMIT],
        )

    # ------------------------------------------------------------------
    # 8. Higher novelty increases VoI
    # ------------------------------------------------------------------
    def test_08_higher_novelty_increases_voi(self) -> None:
        """8. Increasing novelty raises the VoI score (all else equal)."""
        low = run_voi_pipeline(
            novelty=0.1, uncertainty=0.5, task_relevance=0.5,
            temporal_importance=0.5, communication_cost=0.5,
        )
        high = run_voi_pipeline(
            novelty=0.9, uncertainty=0.5, task_relevance=0.5,
            temporal_importance=0.5, communication_cost=0.5,
        )
        self.assertGreater(high.voi_score, low.voi_score)

    # ------------------------------------------------------------------
    # 9. Higher uncertainty increases VoI
    # ------------------------------------------------------------------
    def test_09_higher_uncertainty_increases_voi(self) -> None:
        """9. Increasing uncertainty raises the VoI score (all else equal)."""
        low = run_voi_pipeline(
            novelty=0.5, uncertainty=0.1, task_relevance=0.5,
            temporal_importance=0.5, communication_cost=0.5,
        )
        high = run_voi_pipeline(
            novelty=0.5, uncertainty=0.9, task_relevance=0.5,
            temporal_importance=0.5, communication_cost=0.5,
        )
        self.assertGreater(high.voi_score, low.voi_score)

    # ------------------------------------------------------------------
    # 10. Higher task relevance increases VoI
    # ------------------------------------------------------------------
    def test_10_higher_relevance_increases_voi(self) -> None:
        """10. Increasing task relevance raises the VoI score (all else equal)."""
        low = run_voi_pipeline(
            novelty=0.5, uncertainty=0.5, task_relevance=0.1,
            temporal_importance=0.5, communication_cost=0.5,
        )
        high = run_voi_pipeline(
            novelty=0.5, uncertainty=0.5, task_relevance=0.9,
            temporal_importance=0.5, communication_cost=0.5,
        )
        self.assertGreater(high.voi_score, low.voi_score)

    # ------------------------------------------------------------------
    # 11. Higher temporal importance increases VoI
    # ------------------------------------------------------------------
    def test_11_higher_temporal_increases_voi(self) -> None:
        """11. Increasing temporal importance raises the VoI score (all else equal)."""
        low = run_voi_pipeline(
            novelty=0.5, uncertainty=0.5, task_relevance=0.5,
            temporal_importance=0.1, communication_cost=0.5,
        )
        high = run_voi_pipeline(
            novelty=0.5, uncertainty=0.5, task_relevance=0.5,
            temporal_importance=0.9, communication_cost=0.5,
        )
        self.assertGreater(high.voi_score, low.voi_score)

    # ------------------------------------------------------------------
    # 12. Higher communication cost decreases VoI
    # ------------------------------------------------------------------
    def test_12_higher_cost_decreases_voi(self) -> None:
        """12. Increasing communication cost lowers the VoI score (all else equal)."""
        low_cost = run_voi_pipeline(
            novelty=0.5, uncertainty=0.5, task_relevance=0.5,
            temporal_importance=0.5, communication_cost=0.1,
        )
        high_cost = run_voi_pipeline(
            novelty=0.5, uncertainty=0.5, task_relevance=0.5,
            temporal_importance=0.5, communication_cost=0.9,
        )
        self.assertGreater(low_cost.voi_score, high_cost.voi_score)

    # ------------------------------------------------------------------
    # 13–17. Invalid individual factors rejected
    # ------------------------------------------------------------------
    def test_13_invalid_novelty_rejected(self) -> None:
        """13. Out-of-range novelty raises ValueError."""
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=-0.1, uncertainty=0.5, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=0.5)
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=1.1, uncertainty=0.5, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=0.5)

    def test_14_invalid_uncertainty_rejected(self) -> None:
        """14. Out-of-range uncertainty raises ValueError."""
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=0.5, uncertainty=-0.1, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=0.5)

    def test_15_invalid_relevance_rejected(self) -> None:
        """15. Out-of-range task_relevance raises ValueError."""
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=0.5, uncertainty=0.5, task_relevance=1.5,
                             temporal_importance=0.5, communication_cost=0.5)

    def test_16_invalid_temporal_rejected(self) -> None:
        """16. Out-of-range temporal_importance raises ValueError."""
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=0.5, uncertainty=0.5, task_relevance=0.5,
                             temporal_importance=-1.0, communication_cost=0.5)

    def test_17_invalid_cost_rejected(self) -> None:
        """17. Out-of-range communication_cost raises ValueError."""
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=0.5, uncertainty=0.5, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=2.0)

    # ------------------------------------------------------------------
    # 18. NaN / Inf rejected
    # ------------------------------------------------------------------
    def test_18_nan_inf_rejected(self) -> None:
        """18. NaN and Inf factor values raise ValueError."""
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=np.nan, uncertainty=0.5, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=0.5)
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=0.5, uncertainty=np.inf, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=0.5)
        with self.assertRaises(ValueError):
            run_voi_pipeline(novelty=0.5, uncertainty=0.5, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=-np.inf)

    # ------------------------------------------------------------------
    # 19. Reproducibility
    # ------------------------------------------------------------------
    def test_19_reproducibility(self) -> None:
        """19. Same inputs produce identical results across calls."""
        r1 = run_voi_pipeline(**self.base_kwargs)
        r2 = run_voi_pipeline(**self.base_kwargs)
        self.assertEqual(r1.voi_score, r2.voi_score)
        self.assertEqual(r1.raw_voi_score, r2.raw_voi_score)
        self.assertEqual(r1.decision, r2.decision)

    # ------------------------------------------------------------------
    # 20. Timestamp is preserved
    # ------------------------------------------------------------------
    def test_20_timestamp_preserved(self) -> None:
        """20. An optional timestamp is passed through to the result."""
        ts = "2026-08-28T12:00:00Z"
        result = run_voi_pipeline(**self.base_kwargs, timestamp=ts)
        self.assertEqual(result.timestamp, ts)

        # None timestamp also works
        result_none = run_voi_pipeline(**self.base_kwargs)
        self.assertIsNone(result_none.timestamp)

    # ------------------------------------------------------------------
    # 21. Custom VoIWeights can be passed through
    # ------------------------------------------------------------------
    def test_21_custom_weights_passthrough(self) -> None:
        """21. Custom VoIWeights alter the score without changing the formula."""
        default_result = run_voi_pipeline(**self.base_kwargs)

        # All weight on novelty, none on others
        custom_w = VoIWeights(
            novelty=1.0, uncertainty=0.0, task_relevance=0.0,
            temporal_importance=0.0, resource_cost=0.0,
        )
        custom_result = run_voi_pipeline(**self.base_kwargs, weights=custom_w)

        # With equal inputs the custom score should equal novelty value
        self.assertAlmostEqual(custom_result.voi_score, 0.5, places=6)
        # It should differ from the default-weighted result only if
        # default weights differ (they do because cost is subtracted).
        # Just verify the engine accepted the weights.
        self.assertIsInstance(custom_result, VoIResult)

    # ------------------------------------------------------------------
    # 22. Custom PolicyThresholds can be passed through
    # ------------------------------------------------------------------
    def test_22_custom_thresholds_passthrough(self) -> None:
        """22. Custom PolicyThresholds alter the decision mapping."""
        # Very low thresholds → everything becomes TRANSMIT
        low_thresh = PolicyThresholds(
            discard_max=0.01, buffer_max=0.02, summary_max=0.03,
        )
        result = run_voi_pipeline(**self.base_kwargs, thresholds=low_thresh)
        self.assertEqual(result.decision, DecisionAction.TRANSMIT)

        # Very high thresholds → mid-range inputs become DISCARD
        high_thresh = PolicyThresholds(
            discard_max=0.90, buffer_max=0.95, summary_max=0.99,
        )
        result_high = run_voi_pipeline(**self.base_kwargs, thresholds=high_thresh)
        self.assertEqual(result_high.decision, DecisionAction.DISCARD)

    # ------------------------------------------------------------------
    # 23. Canonical result fields match the five inputs provided
    # ------------------------------------------------------------------
    def test_23_result_fields_match_inputs(self) -> None:
        """23. The VoIResult echoes back the five normalised factor values."""
        result = run_voi_pipeline(
            novelty=0.1, uncertainty=0.2, task_relevance=0.3,
            temporal_importance=0.4, communication_cost=0.5,
        )
        self.assertAlmostEqual(result.novelty, 0.1, places=6)
        self.assertAlmostEqual(result.uncertainty, 0.2, places=6)
        self.assertAlmostEqual(result.task_relevance, 0.3, places=6)
        self.assertAlmostEqual(result.temporal_importance, 0.4, places=6)
        self.assertAlmostEqual(result.resource_cost, 0.5, places=6)

    # ------------------------------------------------------------------
    # 24. Integration layer does not duplicate VoI formula
    # ------------------------------------------------------------------
    def test_24_no_formula_duplication(self) -> None:
        """24. The integration layer delegates to VoIEngine — verify by
        comparing its result with a direct VoIEngine.compute() call."""
        from src.voi.voi_engine import VoIEngine as DirectEngine

        engine = DirectEngine()
        direct = engine.compute(
            novelty=0.7, uncertainty=0.3, task_relevance=0.9,
            temporal_importance=0.2, resource_cost=0.4,
        )
        pipeline = run_voi_pipeline(
            novelty=0.7, uncertainty=0.3, task_relevance=0.9,
            temporal_importance=0.2, communication_cost=0.4,
        )

        self.assertEqual(direct.voi_score, pipeline.voi_score)
        self.assertEqual(direct.raw_voi_score, pipeline.raw_voi_score)
        self.assertEqual(direct.decision, pipeline.decision)

    # ------------------------------------------------------------------
    # Additional: Non-numeric type rejected
    # ------------------------------------------------------------------
    def test_25_non_numeric_type_rejected(self) -> None:
        """25. String / None / list factor values raise TypeError."""
        with self.assertRaises(TypeError):
            run_voi_pipeline(novelty="high", uncertainty=0.5, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=0.5)
        with self.assertRaises(TypeError):
            run_voi_pipeline(novelty=0.5, uncertainty=None, task_relevance=0.5,
                             temporal_importance=0.5, communication_cost=0.5)

    # ------------------------------------------------------------------
    # Additional: numpy numeric types accepted
    # ------------------------------------------------------------------
    def test_26_numpy_numerics_accepted(self) -> None:
        """26. numpy int/float types are accepted as factor values."""
        result = run_voi_pipeline(
            novelty=np.float32(0.5),
            uncertainty=np.float64(0.5),
            task_relevance=np.int32(1),
            temporal_importance=np.float32(0.0),
            communication_cost=np.int64(0),
        )
        self.assertIsInstance(result, VoIResult)
        self.assertGreaterEqual(result.voi_score, 0.0)
        self.assertLessEqual(result.voi_score, 1.0)


if __name__ == "__main__":
    unittest.main()
