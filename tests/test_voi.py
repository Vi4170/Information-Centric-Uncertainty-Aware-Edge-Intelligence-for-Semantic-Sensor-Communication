"""Unit test suite for the Value of Information (VoI) Engine Version 0.1 prototype."""

import math
import unittest
import numpy as np
import pandas as pd

from src.voi.normalization import (
    VoIInputs,
    normalize_min_max,
    validate_and_clip_unit_interval,
    validate_numeric,
)
from src.voi.scoring import VoIWeights, calculate_voi_score
from src.voi.decision_policy import DecisionAction, PolicyThresholds, evaluate_decision
from src.voi.voi_engine import VoIEngine
from src.data_generation.synthetic_generator import generate_synthetic_dataset
from src.evaluation.diagnostics import run_decision_reachability_analysis


class TestVoIEngine(unittest.TestCase):
    """Test suite covering normalization, scoring, decision policy, and VoIEngine orchestrator."""

    def setUp(self):
        """Set up test environment."""
        self.engine = VoIEngine()

    def test_01_all_low_inputs_produce_low_voi(self):
        """1. Test that all-low inputs produce low VoI score and DISCARD decision."""
        res = self.engine.compute(
            novelty=0.05,
            uncertainty=0.05,
            task_relevance=0.05,
            temporal_importance=0.05,
            resource_cost=0.05,
        )
        self.assertLess(res.voi_score, 0.25)
        self.assertEqual(res.decision, DecisionAction.DISCARD)

    def test_02_increasing_novelty_monotonicity(self):
        """2. Test that increasing novelty (all else fixed) does not decrease VoI."""
        base_kwargs = {
            "uncertainty": 0.3,
            "task_relevance": 0.5,
            "temporal_importance": 0.4,
            "resource_cost": 0.2,
        }
        res_low = self.engine.compute(novelty=0.1, **base_kwargs)
        res_high = self.engine.compute(novelty=0.9, **base_kwargs)

        self.assertGreaterEqual(res_high.raw_voi_score, res_low.raw_voi_score)
        self.assertGreaterEqual(res_high.voi_score, res_low.voi_score)

    def test_03_increasing_uncertainty_monotonicity(self):
        """3. Test that increasing uncertainty (all else fixed) does not decrease VoI."""
        base_kwargs = {
            "novelty": 0.4,
            "task_relevance": 0.5,
            "temporal_importance": 0.4,
            "resource_cost": 0.2,
        }
        res_low = self.engine.compute(uncertainty=0.1, **base_kwargs)
        res_high = self.engine.compute(uncertainty=0.9, **base_kwargs)

        self.assertGreaterEqual(res_high.raw_voi_score, res_low.raw_voi_score)
        self.assertGreaterEqual(res_high.voi_score, res_low.voi_score)

    def test_04_increasing_task_relevance_monotonicity(self):
        """4. Test that increasing task relevance (all else fixed) does not decrease VoI."""
        base_kwargs = {
            "novelty": 0.4,
            "uncertainty": 0.3,
            "temporal_importance": 0.4,
            "resource_cost": 0.2,
        }
        res_low = self.engine.compute(task_relevance=0.1, **base_kwargs)
        res_high = self.engine.compute(task_relevance=0.9, **base_kwargs)

        self.assertGreaterEqual(res_high.raw_voi_score, res_low.raw_voi_score)
        self.assertGreaterEqual(res_high.voi_score, res_low.voi_score)

    def test_05_increasing_temporal_importance_monotonicity(self):
        """5. Test that increasing temporal importance (all else fixed) does not decrease VoI."""
        base_kwargs = {
            "novelty": 0.4,
            "uncertainty": 0.3,
            "task_relevance": 0.5,
            "resource_cost": 0.2,
        }
        res_low = self.engine.compute(temporal_importance=0.1, **base_kwargs)
        res_high = self.engine.compute(temporal_importance=0.9, **base_kwargs)

        self.assertGreaterEqual(res_high.raw_voi_score, res_low.raw_voi_score)
        self.assertGreaterEqual(res_high.voi_score, res_low.voi_score)

    def test_06_increasing_resource_cost_penalty(self):
        """6. Test that increasing resource cost (all else fixed) does not increase VoI."""
        base_kwargs = {
            "novelty": 0.8,
            "uncertainty": 0.5,
            "task_relevance": 0.8,
            "temporal_importance": 0.8,
        }
        res_low_cost = self.engine.compute(resource_cost=0.1, **base_kwargs)
        res_high_cost = self.engine.compute(resource_cost=0.9, **base_kwargs)

        self.assertLessEqual(res_high_cost.raw_voi_score, res_low_cost.raw_voi_score)
        self.assertLessEqual(res_high_cost.voi_score, res_low_cost.voi_score)

    def test_07_high_novelty_low_relevance_not_max_voi(self):
        """7. Test that high novelty + low relevance does not automatically yield max VoI."""
        res = self.engine.compute(
            novelty=0.95,
            uncertainty=0.2,
            task_relevance=0.05,
            temporal_importance=0.2,
            resource_cost=0.2,
        )
        self.assertLess(res.voi_score, 0.70)
        self.assertNotEqual(res.decision, DecisionAction.TRANSMIT)

    def test_08_decision_threshold_mapping(self):
        """8. Test threshold mapping across DISCARD, BUFFER, SUMMARY, TRANSMIT boundaries."""
        thresholds = PolicyThresholds(discard_max=0.25, buffer_max=0.50, summary_max=0.70)

        self.assertEqual(evaluate_decision(0.10, thresholds), DecisionAction.DISCARD)
        self.assertEqual(evaluate_decision(0.24, thresholds), DecisionAction.DISCARD)
        self.assertEqual(evaluate_decision(0.25, thresholds), DecisionAction.BUFFER)
        self.assertEqual(evaluate_decision(0.49, thresholds), DecisionAction.BUFFER)
        self.assertEqual(evaluate_decision(0.50, thresholds), DecisionAction.SUMMARY)
        self.assertEqual(evaluate_decision(0.69, thresholds), DecisionAction.SUMMARY)
        self.assertEqual(evaluate_decision(0.70, thresholds), DecisionAction.TRANSMIT)
        self.assertEqual(evaluate_decision(0.95, thresholds), DecisionAction.TRANSMIT)

    def test_09_boundary_values_0_and_1(self):
        """9. Test boundary value handling for 0 and 1 inputs."""
        res_min = self.engine.compute(0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(res_min.raw_voi_score, 0.0)
        self.assertEqual(res_min.voi_score, 0.0)

        res_max = self.engine.compute(1.0, 1.0, 1.0, 1.0, 0.0)
        # Equal weights 0.2: 4*0.2 - 0 = 0.8
        self.assertAlmostEqual(res_max.raw_voi_score, 0.8, places=5)
        self.assertAlmostEqual(res_max.voi_score, 0.8, places=5)

    def test_10_invalid_and_out_of_bounds_inputs(self):
        """10. Test rejection of NaN, Inf, non-numeric, and out-of-bounds inputs."""
        # Non-numeric string
        with self.assertRaises(TypeError):
            self.engine.compute("0.5", 0.5, 0.5, 0.5, 0.5)

        # NaN
        with self.assertRaises(ValueError):
            self.engine.compute(math.nan, 0.5, 0.5, 0.5, 0.5)

        # Infinity
        with self.assertRaises(ValueError):
            self.engine.compute(math.inf, 0.5, 0.5, 0.5, 0.5)

        # Out of bounds when clip=False
        with self.assertRaises(ValueError):
            self.engine.compute(1.5, 0.5, 0.5, 0.5, 0.5)

        with self.assertRaises(ValueError):
            self.engine.compute(-0.2, 0.5, 0.5, 0.5, 0.5)

    def test_11_clipping_behavior(self):
        """11. Test explicit clipping behavior (raw vs clipped score)."""
        inputs = VoIInputs(
            novelty=0.0,
            uncertainty=0.0,
            task_relevance=0.0,
            temporal_importance=0.0,
            resource_cost=1.0,
        )
        res = calculate_voi_score(inputs, clip_output=True)
        self.assertAlmostEqual(res.raw_voi_score, -0.20, places=5)
        self.assertEqual(res.voi_score, 0.0)

        clipping_engine = VoIEngine(clip_inputs=True)
        res_clipped_in = clipping_engine.compute(1.5, -0.2, 0.5, 0.5, 0.5)
        self.assertEqual(res_clipped_in.novelty, 1.0)
        self.assertEqual(res_clipped_in.uncertainty, 0.0)

    def test_12_batch_processing(self):
        """12. Test batch processing DataFrame integration."""
        df_synthetic = generate_synthetic_dataset(output_path=None, num_observations=100, seed=42)
        res_df = self.engine.compute_batch(df_synthetic)

        self.assertEqual(len(res_df), 100)
        self.assertIn("raw_voi_score", res_df.columns)
        self.assertIn("voi_score", res_df.columns)
        self.assertIn("decision", res_df.columns)

        single_res = self.engine.compute(
            novelty=df_synthetic.iloc[0]["novelty"],
            uncertainty=df_synthetic.iloc[0]["uncertainty"],
            task_relevance=df_synthetic.iloc[0]["task_relevance"],
            temporal_importance=df_synthetic.iloc[0]["temporal_importance"],
            resource_cost=df_synthetic.iloc[0]["resource_cost"],
            timestamp=df_synthetic.iloc[0]["timestamp"],
        )
        self.assertAlmostEqual(single_res.voi_score, res_df.iloc[0]["voi_score"], places=5)
        self.assertEqual(single_res.decision.value, res_df.iloc[0]["decision"])

    def test_13_synthetic_reproducibility(self):
        """13. Test that repeated synthetic data generation with seed 42 is identical."""
        df1 = generate_synthetic_dataset(output_path=None, num_observations=100, seed=42)
        df2 = generate_synthetic_dataset(output_path=None, num_observations=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_14_min_max_normalization_utility(self):
        """14. Test generic min-max normalization helper."""
        self.assertAlmostEqual(normalize_min_max(50, 0, 100), 0.5, places=5)
        self.assertAlmostEqual(normalize_min_max(10, 10, 20), 0.0, places=5)
        self.assertAlmostEqual(normalize_min_max(20, 10, 20), 1.0, places=5)
        with self.assertRaises(ValueError):
            normalize_min_max(5, 10, 10)

    def test_15_decision_reachability_analysis(self):
        """15. Test decision reachability analysis calculation."""
        df_synthetic = generate_synthetic_dataset(output_path=None, num_observations=1000, seed=42)
        res_df = self.engine.compute_batch(df_synthetic)
        reachability_df = run_decision_reachability_analysis(res_df, fig_dir="scratch", table_dir="scratch")

        max_clipped_val = reachability_df.loc[reachability_df["Metric"] == "Maximum clipped VoI", "Value"].values[0]
        n_capable = reachability_df.loc[reachability_df["Metric"] == "Observations capable of reaching TRANSMIT", "Value"].values[0]

        self.assertLess(max_clipped_val, 0.70)
        self.assertEqual(n_capable, 0)


if __name__ == "__main__":
    unittest.main()
