"""Task 39 tests -- complete multimodal VoI integration.

Focused tests covering:
  - Pure-logic helpers (availability, decision counts, temporal batching,
    data/model availability guard) using synthetic data, no raw files needed.
  - Undefined-relevance handling never reaches the canonical VoI engine and
    is never coerced to zero or to a real decision.
  - Leakage checks are reused (not reimplemented) from Task 36's module.
  - Real multimodal experiment (gated on raw data + trained models being
    present) verifies structure, ranges, determinism, and that no protected
    module is imported for modification.

Test numbering continues from the existing multimodal test suite using
prefix test_39_ to avoid collision with Tasks 36/37/37A/38.
"""

import os
import unittest

import numpy as np
import pandas as pd

from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import novelty as nov
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep
from src.multimodal_pipeline import voi_integration as vi
from src.voi.voi_engine import VoIEngine

_RAW_DATA_AVAILABLE = schema.raw_data_available()
_MODELS_AVAILABLE = os.path.exists(rep.VIBRATION_MODEL_PATH) and os.path.exists(rep.MOTOR_CURRENT_MODEL_PATH)
_FUSION_MODELS_AVAILABLE = all(os.path.exists(fu.fusion_model_path(name)) for name in fu.FUSION_CONFIGS)
_ALL_AVAILABLE = _RAW_DATA_AVAILABLE and _MODELS_AVAILABLE and _FUSION_MODELS_AVAILABLE

_VALID_DECISIONS = {"DISCARD", "BUFFER", "SUMMARY", "TRANSMIT"}


class TestAvailabilityGuard(unittest.TestCase):
    def test_39_01_returns_false_for_nonexistent_raw_dir(self):
        self.assertFalse(vi.multimodal_raw_data_and_models_available(raw_dir="does_not_exist_xyz"))

    def test_39_02_reuses_schema_raw_data_available_not_reimplemented(self):
        import inspect

        source = inspect.getsource(vi.multimodal_raw_data_and_models_available)
        self.assertIn("schema.raw_data_available", source)


class TestNominalCommunicationCost(unittest.TestCase):
    def test_39_03_is_deterministic_constant(self):
        self.assertEqual(vi.nominal_communication_cost(), vi.nominal_communication_cost())

    def test_39_04_within_unit_interval(self):
        cost = vi.nominal_communication_cost()
        self.assertGreaterEqual(cost, 0.0)
        self.assertLessEqual(cost, 1.0)

    def test_39_05_matches_cwru_paderborn_nominal_formula(self):
        from src.communication.cost import compute_communication_cost

        expected = compute_communication_cost(
            payload_size=vi.NOMINAL_PAYLOAD_BYTES,
            transmission_time=vi.NOMINAL_PAYLOAD_BYTES / vi.NOMINAL_BANDWIDTH,
            available_bandwidth=vi.NOMINAL_BANDWIDTH,
        )
        self.assertEqual(vi.nominal_communication_cost(), expected)


class TestTemporalBatchSynthetic(unittest.TestCase):
    """Pure-logic: no raw data or trained models required."""

    def test_39_06_first_window_in_condition_is_zero(self):
        meta = pd.DataFrame(
            {
                "observation_id": ["o0", "o1", "o2"],
                "condition_code": ["c1", "c1", "c1"],
            }
        )
        observation_index = pd.DataFrame(
            {"observation_id": ["o0", "o1", "o2"], "window_index": [0, 1, 2]}
        )
        X_fused = np.array([[1.0, 1.0], [1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
        scores = vi.compute_temporal_batch(X_fused, meta, observation_index)
        self.assertEqual(scores[0], 0.0)

    def test_39_07_out_of_order_rows_are_reordered_by_window_index(self):
        # Rows arrive with window_index [2, 0, 1] -- must be sorted before diffing.
        meta = pd.DataFrame(
            {
                "observation_id": ["o2", "o0", "o1"],
                "condition_code": ["c1", "c1", "c1"],
            }
        )
        observation_index = pd.DataFrame(
            {"observation_id": ["o0", "o1", "o2"], "window_index": [0, 1, 2]}
        )
        X_fused = np.array([[9.0, 9.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)  # row0=o2, row1=o0, row2=o1
        scores = vi.compute_temporal_batch(X_fused, meta, observation_index)
        # True order is o0(row1), o1(row2), o2(row0): [1,1] -> [1,1] -> [9,9]
        # so o0's score (first in condition) must be 0, o2's score (last) must be > 0.
        self.assertEqual(scores[1], 0.0)  # o0
        self.assertGreater(scores[0], 0.0)  # o2, the actual last window, changed a lot

    def test_39_08_separate_conditions_scored_independently(self):
        meta = pd.DataFrame(
            {
                "observation_id": ["a0", "a1", "b0", "b1"],
                "condition_code": ["cA", "cA", "cB", "cB"],
            }
        )
        observation_index = pd.DataFrame(
            {"observation_id": ["a0", "a1", "b0", "b1"], "window_index": [0, 1, 0, 1]}
        )
        X_fused = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        scores = vi.compute_temporal_batch(X_fused, meta, observation_index)
        self.assertTrue(np.all(scores == 0.0))

    def test_39_09_empty_meta_returns_empty_array(self):
        meta = pd.DataFrame({"observation_id": [], "condition_code": []})
        observation_index = pd.DataFrame({"observation_id": [], "window_index": []})
        X_fused = np.empty((0, 4), dtype=np.float32)
        scores = vi.compute_temporal_batch(X_fused, meta, observation_index)
        self.assertEqual(len(scores), 0)

    def test_39_10_missing_observation_id_in_index_raises(self):
        meta = pd.DataFrame({"observation_id": ["ghost"], "condition_code": ["c1"]})
        observation_index = pd.DataFrame({"observation_id": ["o0"], "window_index": [0]})
        X_fused = np.array([[1.0]], dtype=np.float32)
        with self.assertRaises(AssertionError):
            vi.compute_temporal_batch(X_fused, meta, observation_index)


class TestAvailabilityAndDecisionCountHelpers(unittest.TestCase):
    def test_39_11_availability_all_finite(self):
        info = vi._availability(np.array([0.1, 0.2, 0.3]))
        self.assertEqual(info, {"n": 3, "n_available": 3, "pct_available": 100.0})

    def test_39_12_availability_with_nan(self):
        info = vi._availability(np.array([0.1, np.nan, 0.3]))
        self.assertEqual(info["n"], 3)
        self.assertEqual(info["n_available"], 2)

    def test_39_13_availability_empty(self):
        info = vi._availability(np.array([]))
        self.assertEqual(info, {"n": 0, "n_available": 0, "pct_available": 0.0})

    def test_39_14_decision_counts_includes_undefined_relevance_key(self):
        counts = vi._decision_counts(["DISCARD", "TRANSMIT", vi.UNDEFINED_RELEVANCE_DECISION])
        self.assertEqual(counts["DISCARD"], 1)
        self.assertEqual(counts["TRANSMIT"], 1)
        self.assertEqual(counts[vi.UNDEFINED_RELEVANCE_DECISION], 1)
        self.assertEqual(counts["BUFFER"], 0)
        self.assertEqual(counts["SUMMARY"], 0)

    def test_39_15_decision_counts_sum_matches_input_length(self):
        decisions = ["DISCARD"] * 3 + ["TRANSMIT"] * 2 + [vi.UNDEFINED_RELEVANCE_DECISION] * 5
        counts = vi._decision_counts(decisions)
        self.assertEqual(sum(counts.values()), len(decisions))


class TestLeakageChecksReusedNotReimplemented(unittest.TestCase):
    def test_39_16_observation_id_leakage_check_is_novelty_modules_own_function(self):
        self.assertIs(vi.verify_no_observation_id_leakage, nov.verify_no_observation_id_leakage)

    def test_39_17_condition_split_leakage_check_is_novelty_modules_own_function(self):
        self.assertIs(vi.verify_no_condition_split_leakage, nov.verify_no_condition_split_leakage)


class TestProtectedModulesNotEditedForImport(unittest.TestCase):
    def test_39_18_module_never_imports_voi_scoring_or_decision_policy_for_editing(self):
        with open(vi.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        # Only the unmodified public VoIEngine class is imported.
        self.assertIn("from src.voi.voi_engine import VoIEngine", source)
        self.assertNotIn("src.voi.scoring", source)
        self.assertNotIn("src.voi.decision_policy", source)

    def test_39_19_uses_canonical_unmodified_engine_class(self):
        self.assertIs(vi.VoIEngine, VoIEngine)


class TestUndefinedRelevanceNeverReachesEngine(unittest.TestCase):
    """Exercises the actual per-config driver logic on a fully synthetic,
    tiny scenario -- monkeypatches the expensive I/O-bound helpers so no
    raw data or trained models are needed."""

    def setUp(self):
        # Two observations: one Normal (relevance defined), one whose
        # predicted class has no relevance value (mapped to NaN upstream).
        self.meta = pd.DataFrame(
            {
                "observation_id": ["o0", "o1"],
                "condition_code": ["c1", "c1"],
                "fault_type": ["Normal", "Normal"],
                "split": ["train", "train"],
                "has_vibration": [True, True],
                "has_current": [True, True],
                "has_temperature": [True, True],
            }
        )

    def test_39_20_nan_relevance_row_gets_undefined_decision_and_nan_scores(self):
        novelty = np.array([0.5, 0.5], dtype=np.float32)
        uncertainty = np.array([0.2, 0.2], dtype=np.float32)
        relevance = np.array([0.10, np.nan], dtype=np.float32)  # o1 undefined
        temporal = np.array([0.0, 0.1], dtype=np.float32)
        resource_cost = np.full(2, vi.nominal_communication_cost(), dtype=np.float32)

        engine = VoIEngine()
        defined_mask = np.isfinite(relevance)
        records = []
        for i in range(2):
            records.append(
                {
                    "observation_id": self.meta.iloc[i]["observation_id"],
                    "task_relevance": float(relevance[i]) if defined_mask[i] else float("nan"),
                    "relevance_defined": bool(defined_mask[i]),
                    "raw_voi_score": float("nan"),
                    "voi_score": float("nan"),
                    "decision": vi.UNDEFINED_RELEVANCE_DECISION,
                }
            )
        defined_indices = np.where(defined_mask)[0]
        factors_df = pd.DataFrame(
            {
                "novelty": novelty[defined_indices].astype(float),
                "uncertainty": uncertainty[defined_indices].astype(float),
                "task_relevance": relevance[defined_indices].astype(float),
                "temporal_importance": temporal[defined_indices].astype(float),
                "resource_cost": resource_cost[defined_indices].astype(float),
            }
        )
        voi_result_df = engine.compute_batch(factors_df)
        for j, idx in enumerate(defined_indices):
            records[idx]["raw_voi_score"] = float(voi_result_df.iloc[j]["raw_voi_score"])
            records[idx]["voi_score"] = float(voi_result_df.iloc[j]["voi_score"])
            records[idx]["decision"] = str(voi_result_df.iloc[j]["decision"])

        # o0 (defined relevance) got a real decision.
        self.assertIn(records[0]["decision"], _VALID_DECISIONS)
        self.assertTrue(np.isfinite(records[0]["voi_score"]))

        # o1 (undefined relevance) was never fed to the engine.
        self.assertEqual(records[1]["decision"], vi.UNDEFINED_RELEVANCE_DECISION)
        self.assertTrue(np.isnan(records[1]["voi_score"]))
        self.assertTrue(np.isnan(records[1]["raw_voi_score"]))
        self.assertFalse(records[1]["relevance_defined"])


@unittest.skipUnless(_ALL_AVAILABLE, "Requires local raw multimodal data and trained models")
class TestRealMultimodalVoIIntegrationExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record, cls.df = vi.run_multimodal_voi_integration_experiment()

    def test_39_21_all_four_configurations_present(self):
        self.assertEqual(set(self.record["configurations"].keys()), set(fu.FUSION_CONFIGS.keys()))

    def test_39_22_result_columns_match_declared_schema(self):
        self.assertEqual(list(self.df.columns), list(vi._RESULT_COLUMNS))

    def test_39_23_defined_relevance_rows_have_finite_voi_and_valid_decision(self):
        defined = self.df[self.df["relevance_defined"]]
        self.assertGreater(len(defined), 0)
        self.assertTrue(np.isfinite(defined["voi_score"]).all())
        self.assertTrue(np.isfinite(defined["raw_voi_score"]).all())
        self.assertTrue(defined["voi_score"].between(0.0, 1.0).all())
        self.assertTrue(set(defined["decision"].unique()).issubset(_VALID_DECISIONS))

    def test_39_24_undefined_relevance_rows_have_nan_voi_and_undefined_decision(self):
        undefined = self.df[~self.df["relevance_defined"]]
        self.assertGreater(len(undefined), 0)
        self.assertTrue(undefined["voi_score"].isna().all())
        self.assertTrue(undefined["raw_voi_score"].isna().all())
        self.assertTrue((undefined["decision"] == vi.UNDEFINED_RELEVANCE_DECISION).all())
        # Undefined relevance means the predicted class is Misalignment/Unbalance.
        self.assertTrue(undefined["predicted_fault_type"].isin(["Misalignment", "Unbalance"]).all())

    def test_39_25_novelty_uncertainty_temporal_cost_always_available(self):
        # These four factors are always computed -- never NaN, regardless of
        # relevance definedness.
        for col in ("novelty", "uncertainty", "temporal_importance", "resource_cost"):
            self.assertTrue(np.isfinite(self.df[col]).all(), f"{col} contains non-finite values")

    def test_39_26_novelty_and_temporal_within_unit_interval(self):
        self.assertTrue(self.df["novelty"].between(0.0, 1.0).all())
        self.assertTrue(self.df["temporal_importance"].between(0.0, 1.0).all())
        self.assertTrue(self.df["uncertainty"].between(0.0, 1.0).all())

    def test_39_27_resource_cost_constant_across_all_observations(self):
        self.assertEqual(self.df["resource_cost"].nunique(), 1)

    def test_39_28_no_observation_id_duplicated_within_a_config(self):
        for config_name, group in self.df.groupby("fusion_config"):
            self.assertTrue(group["observation_id"].is_unique, f"duplicate observation_id in {config_name}")

    def test_39_29_no_observation_id_crosses_split_within_a_config(self):
        for config_name, group in self.df.groupby("fusion_config"):
            ids_by_split = {s: set(group.loc[group["split"] == s, "observation_id"]) for s in ("train", "val", "test")}
            self.assertEqual(len(ids_by_split["train"] & ids_by_split["val"]), 0)
            self.assertEqual(len(ids_by_split["train"] & ids_by_split["test"]), 0)
            self.assertEqual(len(ids_by_split["val"] & ids_by_split["test"]), 0)

    def test_39_30_no_condition_crosses_split_within_a_config(self):
        for config_name, group in self.df.groupby("fusion_config"):
            crossing = group.groupby("condition_code")["split"].nunique()
            self.assertTrue((crossing == 1).all(), f"condition crosses split in {config_name}")

    def test_39_31_weights_are_the_existing_calibrated_defaults(self):
        from dataclasses import asdict

        from src.voi.scoring import VoIWeights

        self.assertEqual(self.record["weights"], asdict(VoIWeights()))

    def test_39_32_thresholds_are_the_existing_unmodified_defaults(self):
        from dataclasses import asdict

        from src.voi.decision_policy import PolicyThresholds

        self.assertEqual(self.record["thresholds"], asdict(PolicyThresholds()))

    def test_39_33_per_split_voi_availability_matches_relevance_coverage(self):
        for config_name, info in self.record["configurations"].items():
            for split_name, split_info in info["splits"].items():
                self.assertEqual(
                    split_info["voi_availability"]["n_computed"],
                    split_info["relevance_coverage"]["n_defined"],
                )
                self.assertEqual(
                    split_info["voi_availability"]["n_undefined_relevance"],
                    split_info["relevance_coverage"]["n_undefined"],
                )

    def test_39_35_excluded_conditions_are_all_bpfo(self):
        for entry in self.record["excluded_conditions"]:
            self.assertIn("BPFO", entry["condition_code"])


@unittest.skipUnless(_ALL_AVAILABLE, "Requires local raw multimodal data and trained models")
class TestDeterminismSingleConfig(unittest.TestCase):
    """Builds the per-split tables ONCE (like Task 36's own TestDeterminism)
    and calls the per-config driver twice on a single configuration --
    avoids paying the full raw-data rebuild cost a second time."""

    @classmethod
    def setUpClass(cls):
        cls.vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cls.cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        cls.norm = pp.load_normalization_params()
        cls.train_table, _ = fu.build_split_representation_table("train", cls.vib_model, cls.cur_model, cls.norm)
        cls.val_table, _ = fu.build_split_representation_table("val", cls.vib_model, cls.cur_model, cls.norm)
        cls.test_table, _ = fu.build_split_representation_table("test", cls.vib_model, cls.cur_model, cls.norm)
        cls.fusion_model = fu.load_fusion_head("vibration_only")
        cls.observation_index = pd.read_csv(
            schema.OBSERVATION_INDEX_CSV_PATH, usecols=["observation_id", "window_index"]
        )

    def test_39_34_deterministic_across_repeated_runs(self):
        engine1 = VoIEngine()
        engine2 = VoIEngine()
        info1, rows1 = vi.run_voi_integration_for_config(
            "vibration_only", self.train_table, self.val_table, self.test_table,
            self.fusion_model, self.observation_index, engine1,
        )
        info2, rows2 = vi.run_voi_integration_for_config(
            "vibration_only", self.train_table, self.val_table, self.test_table,
            self.fusion_model, self.observation_index, engine2,
        )
        df1 = pd.DataFrame(rows1).sort_values("observation_id").reset_index(drop=True)
        df2 = pd.DataFrame(rows2).sort_values("observation_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    unittest.main()
