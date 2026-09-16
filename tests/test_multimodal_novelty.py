import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import novelty as nov
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep
from src.novelty.novelty import DistanceNoveltyDetector

_RAW_DATA_AVAILABLE = schema.raw_data_available()
_MODELS_AVAILABLE = os.path.exists(rep.VIBRATION_MODEL_PATH) and os.path.exists(rep.MOTOR_CURRENT_MODEL_PATH)
_FUSION_MODELS_AVAILABLE = all(os.path.exists(fu.fusion_model_path(name)) for name in fu.FUSION_CONFIGS)
_ALL_AVAILABLE = _RAW_DATA_AVAILABLE and _MODELS_AVAILABLE and _FUSION_MODELS_AVAILABLE
_CORRUPT_CONDITION = "4Nm_BPFO_10"


class TestConstants(unittest.TestCase):
    def test_01_reference_class_is_normal_label(self):
        self.assertEqual(nov.NOVELTY_REFERENCE_CLASS, rep.FAULT_TYPE_TO_LABEL_ID["Normal"])
        self.assertEqual(nov.NOVELTY_REFERENCE_CLASS, 0)

    def test_02_uses_canonical_unmodified_detector(self):
        # Task 36 must reuse the existing mechanism, not invent a new one.
        import inspect

        self.assertIs(
            inspect.unwrap(nov.run_novelty_for_config).__globals__["DistanceNoveltyDetector"],
            DistanceNoveltyDetector,
        )

    def test_03_module_never_imports_voi(self):
        with open(nov.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("src.voi", source)
        self.assertNotIn("from src import voi", source)

    def test_04_result_columns_include_full_traceability(self):
        required = {
            "observation_id", "condition_code", "fault_type", "split", "fusion_config",
            "has_vibration", "has_current", "has_temperature", "novelty_score",
        }
        self.assertTrue(required.issubset(set(nov._RESULT_COLUMNS)))


class TestLeakageChecksSynthetic(unittest.TestCase):
    """Pure-logic leakage checks -- no raw data or models required."""

    def test_05_observation_id_overlap_across_splits_raises(self):
        meta_train = pd.DataFrame({"observation_id": ["a", "b"]})
        meta_val = pd.DataFrame({"observation_id": ["b", "c"]})  # "b" leaked into val
        meta_test = pd.DataFrame({"observation_id": ["d"]})
        with self.assertRaises(AssertionError):
            nov.verify_no_observation_id_leakage(meta_train, meta_val, meta_test)

    def test_06_disjoint_observation_ids_pass(self):
        meta_train = pd.DataFrame({"observation_id": ["a", "b"]})
        meta_val = pd.DataFrame({"observation_id": ["c"]})
        meta_test = pd.DataFrame({"observation_id": ["d"]})
        nov.verify_no_observation_id_leakage(meta_train, meta_val, meta_test)  # must not raise

    def test_07_empty_metadata_frames_pass(self):
        empty = pd.DataFrame({"observation_id": []})
        nov.verify_no_observation_id_leakage(empty, empty, empty)  # must not raise

    def test_08_condition_crossing_split_raises(self):
        combined = pd.DataFrame(
            {
                "condition_code": ["c1", "c1", "c2"],
                "split": ["train", "val", "test"],  # c1 appears in both train and val
            }
        )
        with self.assertRaises(AssertionError):
            nov.verify_no_condition_split_leakage(combined)

    def test_09_conditions_each_in_one_split_pass(self):
        combined = pd.DataFrame(
            {
                "condition_code": ["c1", "c1", "c2", "c3"],
                "split": ["train", "train", "val", "test"],
            }
        )
        nov.verify_no_condition_split_leakage(combined)  # must not raise

    def test_10_empty_combined_metadata_passes(self):
        nov.verify_no_condition_split_leakage(pd.DataFrame(columns=["condition_code", "split"]))


class TestNoNormalTrainingSupportGuard(unittest.TestCase):
    """Verifies the explicit Normal-support guard without needing real raw
    data or trained models -- get_fused_representation_for_split is faked."""

    def _fake_get(self, config_name, table, fusion_model):
        if table == "train":
            X = np.random.randn(5, fu.FUSED_DIM).astype(np.float32)
            y = np.array([1, 1, 2, 2, 3], dtype=np.int64)  # zero Normal (label 0)
            meta = pd.DataFrame(
                {
                    "observation_id": [f"train_{i}" for i in range(5)],
                    "condition_code": ["cA"] * 5,
                    "fault_type": ["BPFI"] * 2 + ["BPFO"] * 2 + ["Misalignment"],
                    "split": ["train"] * 5,
                    "has_vibration": [True] * 5,
                    "has_current": [True] * 5,
                    "has_temperature": [True] * 5,
                }
            )
            return X, y, meta
        empty_meta = pd.DataFrame(
            columns=["observation_id", "condition_code", "fault_type", "split", "has_vibration", "has_current", "has_temperature"]
        )
        return np.empty((0, fu.FUSED_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64), empty_meta

    def test_11_raises_when_zero_normal_training_observations(self):
        with patch.object(nov, "get_fused_representation_for_split", side_effect=self._fake_get):
            with self.assertRaises(nov.NoNormalTrainingSupportError):
                nov.run_novelty_for_config("vibration_only", "train", "val", "test", fusion_model=None)


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34 encoders, or Task 35 fusion heads not present")
class TestRealMultimodalNoveltyExperiment(unittest.TestCase):
    """Runs the real Task 36 experiment once (all four configurations) and
    reuses the result across assertions for efficiency."""

    @classmethod
    def setUpClass(cls):
        cls.experiment_record, cls.results_df = nov.run_multimodal_novelty_experiment()

    def test_12_all_four_configurations_present(self):
        self.assertEqual(set(self.experiment_record["configurations"].keys()), set(fu.FUSION_CONFIGS.keys()))
        self.assertEqual(set(self.results_df["fusion_config"]), set(fu.FUSION_CONFIGS.keys()))

    def test_13_reference_centroid_dim_is_fused_dim_for_every_config(self):
        for name, info in self.experiment_record["configurations"].items():
            self.assertEqual(info["reference_centroid_dim"], fu.FUSED_DIM)
            self.assertEqual(len(info["reference_centroid"]), fu.FUSED_DIM)

    def test_14_normal_reference_support_is_positive_for_every_config(self):
        for name, info in self.experiment_record["configurations"].items():
            self.assertGreater(info["n_train_normal_reference"], 0, f"{name} has zero Normal training support")

    def test_15_fit_only_on_train_declared_for_every_config(self):
        for name, info in self.experiment_record["configurations"].items():
            self.assertEqual(info["fit_only_on_split"], "train")

    def test_16_bpfo_retained_for_vibration_only_and_vibration_temperature(self):
        for config_name in ("vibration_only", "vibration_temperature"):
            sub = self.results_df[self.results_df["fusion_config"] == config_name]
            self.assertTrue((sub["fault_type"] == "BPFO").any(), f"{config_name} lost BPFO")

    def test_17_bpfo_excluded_from_current_requiring_configs(self):
        for config_name in ("vibration_current", "vibration_current_temperature"):
            sub = self.results_df[self.results_df["fusion_config"] == config_name]
            self.assertFalse((sub["fault_type"] == "BPFO").any(), f"{config_name} unexpectedly retained BPFO")

    def test_18_no_observation_id_crosses_split_per_config(self):
        for config_name in fu.FUSION_CONFIGS:
            sub = self.results_df[self.results_df["fusion_config"] == config_name]
            counts = sub.groupby("observation_id")["split"].nunique()
            self.assertTrue((counts <= 1).all(), f"{config_name} has an observation_id spanning multiple splits")

    def test_19_no_condition_crosses_split_per_config(self):
        for config_name in fu.FUSION_CONFIGS:
            sub = self.results_df[self.results_df["fusion_config"] == config_name]
            counts = sub.groupby("condition_code")["split"].nunique()
            self.assertTrue((counts <= 1).all(), f"{config_name} has a condition crossing splits")

    def test_20_scores_are_within_unit_interval(self):
        self.assertTrue(self.results_df["novelty_score"].between(0.0, 1.0).all())

    def test_21_excluded_conditions_all_bpfo(self):
        codes = {e["condition_code"] for e in self.experiment_record["excluded_conditions"]}
        for code in codes:
            self.assertIn("BPFO", code)

    def test_22_not_computed_list_excludes_voi_and_other_factors(self):
        not_computed = set(self.experiment_record["not_computed_in_this_task"])
        for factor in ("uncertainty", "task_relevance", "temporal_importance", "communication_cost", "voi_score"):
            self.assertIn(factor, not_computed)

    def test_23_save_and_reload_round_trip_uses_temp_dir_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "novelty_config.json")
            results_path = os.path.join(tmp, "novelty_results.csv")
            nov.save_novelty_config(self.experiment_record, path=config_path)
            nov.save_novelty_results(self.results_df, path=results_path)
            self.assertTrue(os.path.exists(config_path))
            self.assertTrue(os.path.exists(results_path))

            import json

            with open(config_path, "r", encoding="utf-8") as f:
                reloaded = json.load(f)
            self.assertEqual(reloaded["task"], 36)
            reloaded_df = pd.read_csv(results_path)
            self.assertEqual(len(reloaded_df), len(self.results_df))


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34 encoders, or Task 35 fusion heads not present")
class TestFusedRepresentationUsage(unittest.TestCase):
    """Confirms novelty operates on the FUSED representation, never the raw
    concatenated per-modality embedding and never a per-sensor average."""

    @classmethod
    def setUpClass(cls):
        cls.vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cls.cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        cls.norm = pp.load_normalization_params()
        cls.val_table, _ = fu.build_split_representation_table("val", cls.vib_model, cls.cur_model, cls.norm)

    def test_24_fused_dim_not_concat_dim_for_multi_modality_config(self):
        fusion_model = fu.load_fusion_head("vibration_current")
        X_fused, y, meta = nov.get_fused_representation_for_split("vibration_current", self.val_table, fusion_model)
        self.assertEqual(X_fused.shape[1], fu.FUSED_DIM)
        self.assertNotEqual(X_fused.shape[1], fu.concat_dim_for_config("vibration_current"))

    def test_25_fused_representation_matches_fusion_head_layer_output(self):
        fusion_model = fu.load_fusion_head("vibration_temperature")
        X_concat, y, meta = fu.assemble_fusion_split_arrays("vibration_temperature", self.val_table)
        expected = fu.get_fused_representation(fusion_model, X_concat)
        X_fused, _, _ = nov.get_fused_representation_for_split("vibration_temperature", self.val_table, fusion_model)
        np.testing.assert_array_equal(X_fused, expected)

    def test_26_empty_table_returns_empty_fused_array_with_correct_width(self):
        fusion_model = fu.load_fusion_head("vibration_only")
        empty_table = self.val_table.iloc[0:0]
        X_fused, y, meta = nov.get_fused_representation_for_split("vibration_only", empty_table, fusion_model)
        self.assertEqual(X_fused.shape, (0, fu.FUSED_DIM))


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34 encoders, or Task 35 fusion heads not present")
class TestDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cls.cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        cls.norm = pp.load_normalization_params()
        cls.train_table, _ = fu.build_split_representation_table("train", cls.vib_model, cls.cur_model, cls.norm)
        cls.val_table, _ = fu.build_split_representation_table("val", cls.vib_model, cls.cur_model, cls.norm)
        cls.test_table, _ = fu.build_split_representation_table("test", cls.vib_model, cls.cur_model, cls.norm)
        cls.fusion_model = fu.load_fusion_head("vibration_only")

    def test_27_reference_and_scores_are_deterministic_across_runs(self):
        info1, rows1 = nov.run_novelty_for_config("vibration_only", self.train_table, self.val_table, self.test_table, self.fusion_model)
        info2, rows2 = nov.run_novelty_for_config("vibration_only", self.train_table, self.val_table, self.test_table, self.fusion_model)
        self.assertEqual(info1["reference_centroid"], info2["reference_centroid"])
        self.assertEqual(info1["training_distance_bounds"], info2["training_distance_bounds"])
        scores1 = [r["novelty_score"] for r in rows1]
        scores2 = [r["novelty_score"] for r in rows2]
        self.assertEqual(scores1, scores2)


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34 encoders, or Task 35 fusion heads not present")
class TestFitOnlyUsesTrainSplit(unittest.TestCase):
    def test_28_detector_fit_called_only_with_train_sized_arrays(self):
        vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        norm = pp.load_normalization_params()
        train_table, _ = fu.build_split_representation_table("train", vib_model, cur_model, norm)
        val_table, _ = fu.build_split_representation_table("val", vib_model, cur_model, norm)
        test_table, _ = fu.build_split_representation_table("test", vib_model, cur_model, norm)
        fusion_model = fu.load_fusion_head("vibration_only")

        real_fit = DistanceNoveltyDetector.fit
        fit_call_sizes = []

        def spy_fit(self, train_embeddings, train_labels=None):
            fit_call_sizes.append(len(train_embeddings))
            return real_fit(self, train_embeddings, train_labels)

        with patch.object(DistanceNoveltyDetector, "fit", spy_fit):
            info, _ = nov.run_novelty_for_config("vibration_only", train_table, val_table, test_table, fusion_model)

        self.assertEqual(len(fit_call_sizes), 1)
        self.assertEqual(fit_call_sizes[0], info["n_train_total"])
        self.assertNotEqual(fit_call_sizes[0], info["n_val_total"])
        self.assertNotEqual(fit_call_sizes[0], info["n_test_total"])


if __name__ == "__main__":
    unittest.main()
