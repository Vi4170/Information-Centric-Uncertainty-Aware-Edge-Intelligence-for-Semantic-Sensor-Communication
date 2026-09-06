import os
import unittest

import numpy as np

from src.ims_pipeline.preprocessing import RAW_DATA_DIR, RUN_SUBDIR
from src.evaluation.ims_temporal_analysis import (
    ANALYSIS_TARGETS,
    NOT_COMPUTED_COMPONENTS,
    analyze_target,
    build_failure_proximity_correlation,
    build_progression_summary,
)
from src.voi.scoring import VoIWeights
from src.voi.decision_policy import PolicyThresholds

_SMALL_TARGET = {"run_id": "2nd_test", "bearing_id": 2, "channel_index": 0, "role": "control",
                 "failure_note": "No documented failure for this bearing in this run."}

_RAW_DATA_AVAILABLE = os.path.isdir(os.path.join(RAW_DATA_DIR, RUN_SUBDIR["2nd_test"]))


class TestAnalysisTargetsDefinition(unittest.TestCase):
    def test_every_target_has_required_fields(self):
        for target in ANALYSIS_TARGETS:
            self.assertIn("run_id", target)
            self.assertIn("bearing_id", target)
            self.assertIn("channel_index", target)
            self.assertIn("role", target)
            self.assertIn("failure_note", target)
            self.assertIn(target["role"], {"documented_failure", "control", "undocumented_continuation"})

    def test_no_target_duplicated(self):
        keys = [(t["run_id"], t["bearing_id"], t["channel_index"]) for t in ANALYSIS_TARGETS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_not_computed_components_documented(self):
        self.assertIn("uncertainty", NOT_COMPUTED_COMPONENTS)
        self.assertIn("task_relevance", NOT_COMPUTED_COMPONENTS)
        self.assertIn("voi_score_and_decision", NOT_COMPUTED_COMPONENTS)
        for reason in NOT_COMPUTED_COMPONENTS.values():
            self.assertIsInstance(reason, str)
            self.assertGreater(len(reason), 0)


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "IMS 2nd_test raw data not available")
class TestAnalyzeTargetOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_target(_SMALL_TARGET)

    def test_chronological_ordering_preserved(self):
        indices = self.result["chronological_order_index"].to_numpy()
        self.assertTrue(np.all(np.diff(indices) >= 0))

    def test_no_duplicate_observations(self):
        self.assertTrue(self.result["observation_id"].is_unique)

    def test_split_values_valid_and_ordered(self):
        self.assertTrue(set(self.result["split"].unique()).issubset({"initial", "adaptation", "test"}))
        first_test_pos = np.flatnonzero((self.result["split"] == "test").to_numpy())[0]
        last_initial_pos = np.flatnonzero((self.result["split"] == "initial").to_numpy())[-1]
        self.assertLess(last_initial_pos, first_test_pos)

    def test_normalization_uses_initial_split_only(self):
        from src.ims_pipeline.preprocessing import fit_initial_normalization, load_stream_windows

        X, meta = load_stream_windows(_SMALL_TARGET["run_id"], _SMALL_TARGET["bearing_id"], _SMALL_TARGET["channel_index"])
        expected_mean, expected_std = fit_initial_normalization(X, meta)
        self.assertAlmostEqual(float(self.result["normalization_mean"].iloc[0]), expected_mean, places=5)
        self.assertAlmostEqual(float(self.result["normalization_std"].iloc[0]), expected_std, places=5)

    def test_no_window_level_label_is_fabricated(self):
        self.assertEqual(self.result["label_available"].unique().tolist(), [False])
        self.assertTrue(self.result["condition_label"].isna().all())
        self.assertEqual(self.result["failure_note"].nunique(), 1)

    def test_novelty_and_temporal_in_valid_range(self):
        self.assertTrue(((self.result["novelty"] >= 0.0) & (self.result["novelty"] <= 1.0)).all())
        self.assertTrue(
            ((self.result["temporal_importance"] >= 0.0) & (self.result["temporal_importance"] <= 1.0)).all()
        )

    def test_first_temporal_importance_is_zero(self):
        self.assertEqual(float(self.result["temporal_importance"].iloc[0]), 0.0)

    def test_trajectory_position_bounds(self):
        self.assertAlmostEqual(float(self.result["trajectory_position"].iloc[0]), 0.0, places=6)
        self.assertAlmostEqual(float(self.result["trajectory_position"].iloc[-1]), 1.0, places=6)

    def test_resource_cost_constant(self):
        self.assertEqual(self.result["resource_cost"].nunique(), 1)

    def test_deterministic_repeat(self):
        result2 = analyze_target(_SMALL_TARGET)
        np.testing.assert_array_equal(
            self.result["novelty"].to_numpy(), result2["novelty"].to_numpy()
        )
        np.testing.assert_array_equal(
            self.result["temporal_importance"].to_numpy(), result2["temporal_importance"].to_numpy()
        )

    def test_result_table_builders(self):
        summary_df = build_progression_summary([self.result])
        self.assertGreater(len(summary_df), 0)
        for col in ("novelty_mean", "temporal_importance_mean", "resource_cost_mean"):
            self.assertIn(col, summary_df.columns)

        correlation_df = build_failure_proximity_correlation([self.result])
        self.assertEqual(len(correlation_df), 1)
        self.assertIn("corr_trajectory_position_novelty", correlation_df.columns)
        self.assertIn("corr_trajectory_position_temporal_importance", correlation_df.columns)


class TestVoiProductionConfigUntouched(unittest.TestCase):
    def test_weights_unchanged(self):
        weights = VoIWeights()
        self.assertEqual(weights.novelty, 0.30)
        self.assertEqual(weights.uncertainty, 0.05)
        self.assertEqual(weights.task_relevance, 0.35)
        self.assertEqual(weights.temporal_importance, 0.20)
        self.assertEqual(weights.resource_cost, 0.10)

    def test_thresholds_unchanged(self):
        thresholds = PolicyThresholds()
        self.assertEqual(thresholds.discard_max, 0.25)
        self.assertEqual(thresholds.buffer_max, 0.50)
        self.assertEqual(thresholds.summary_max, 0.70)


if __name__ == "__main__":
    unittest.main()
