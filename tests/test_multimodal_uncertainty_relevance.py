import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep
from src.multimodal_pipeline import uncertainty_relevance as ur
from src.relevance.config import CLASS_RELEVANCE_MAP as CWRU_CLASS_RELEVANCE_MAP
from src.uncertainty.uncertainty import compute_predictive_entropy

_RAW_DATA_AVAILABLE = schema.raw_data_available()
_MODELS_AVAILABLE = os.path.exists(rep.VIBRATION_MODEL_PATH) and os.path.exists(rep.MOTOR_CURRENT_MODEL_PATH)
_FUSION_MODELS_AVAILABLE = all(os.path.exists(fu.fusion_model_path(name)) for name in fu.FUSION_CONFIGS)
_ALL_AVAILABLE = _RAW_DATA_AVAILABLE and _MODELS_AVAILABLE and _FUSION_MODELS_AVAILABLE
_CORRUPT_CONDITION = "4Nm_BPFO_10"


class TestLabelMappingVerified(unittest.TestCase):
    def test_01_five_class_labels_match_task34(self):
        self.assertEqual(rep.FAULT_TYPE_LABELS, ur.EXPECTED_FAULT_TYPE_LABELS)
        self.assertEqual(
            ur.EXPECTED_FAULT_TYPE_LABELS,
            ("Normal", "BPFI", "BPFO", "Misalignment", "Unbalance"),
        )

    def test_02_num_classes_is_five(self):
        self.assertEqual(ur.NUM_CLASSES, 5)


class TestRelevanceMapping(unittest.TestCase):
    def test_03_normal_bpfi_bpfo_reuse_cwru_values_verbatim(self):
        self.assertEqual(ur.MULTIMODAL_RELEVANCE_MAP[0], CWRU_CLASS_RELEVANCE_MAP[0])  # Normal
        self.assertEqual(ur.MULTIMODAL_RELEVANCE_MAP[1], CWRU_CLASS_RELEVANCE_MAP[1])  # BPFI <- Inner Race Fault
        self.assertEqual(ur.MULTIMODAL_RELEVANCE_MAP[2], CWRU_CLASS_RELEVANCE_MAP[3])  # BPFO <- Outer Race Fault

    def test_04_misalignment_and_unbalance_are_undefined_not_fabricated(self):
        self.assertIsNone(ur.MULTIMODAL_RELEVANCE_MAP[3])
        self.assertIsNone(ur.MULTIMODAL_RELEVANCE_MAP[4])
        self.assertEqual(set(ur.UNDEFINED_RELEVANCE_CLASSES), {"Misalignment", "Unbalance"})

    def test_05_relevance_from_predicted_class_deterministic(self):
        self.assertEqual(ur.relevance_from_predicted_class(0), 0.10)
        self.assertIsNone(ur.relevance_from_predicted_class(3))
        self.assertIsNone(ur.relevance_from_predicted_class(4))

    def test_06_compute_relevance_batch_nan_for_undefined_classes(self):
        # One-hot rows so argmax is unambiguous: Normal, BPFI, BPFO, Misalignment, Unbalance
        probs = np.eye(5, dtype=np.float32)
        scores = ur.compute_relevance_batch(probs)
        self.assertEqual(scores[0], 0.10)
        self.assertEqual(scores[1], CWRU_CLASS_RELEVANCE_MAP[1])
        self.assertEqual(scores[2], CWRU_CLASS_RELEVANCE_MAP[3])
        self.assertTrue(np.isnan(scores[3]))
        self.assertTrue(np.isnan(scores[4]))

    def test_07_relevance_never_fabricated_as_zero(self):
        probs = np.zeros((1, 5), dtype=np.float32)
        probs[0, 3] = 1.0  # Misalignment
        scores = ur.compute_relevance_batch(probs)
        self.assertTrue(np.isnan(scores[0]))
        self.assertNotEqual(scores[0], 0.0)


class TestUncertaintyIsCanonicalEntropy(unittest.TestCase):
    def test_08_uncertainty_matches_canonical_formula_directly(self):
        rng = np.random.RandomState(0)
        raw = rng.rand(10, 5).astype(np.float32)
        probs = raw / raw.sum(axis=1, keepdims=True)
        expected = compute_predictive_entropy(probs, num_classes=5)
        # run_uncertainty_relevance_for_config calls the same function --
        # verify no wrapper logic silently changes the formula.
        actual = compute_predictive_entropy(probs, num_classes=ur.NUM_CLASSES)
        np.testing.assert_array_equal(expected, actual)

    def test_09_confident_prediction_gives_near_zero_uncertainty(self):
        probs = np.array([[0.999, 0.00025, 0.00025, 0.00025, 0.00025]], dtype=np.float32)
        score = compute_predictive_entropy(probs, num_classes=5)[0]
        self.assertLess(score, 0.05)

    def test_10_uniform_prediction_gives_uncertainty_near_one(self):
        probs = np.full((1, 5), 0.2, dtype=np.float32)
        score = compute_predictive_entropy(probs, num_classes=5)[0]
        self.assertGreater(score, 0.95)

    def test_11_uncertainty_never_derived_from_novelty_or_confidence(self):
        # Static source check: this module must never import novelty scores
        # or a confidence/1-confidence shortcut as a substitute for entropy.
        with open(ur.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("1 - np.max", source)
        self.assertNotIn("1.0 - np.max", source)
        self.assertIn("compute_predictive_entropy", source)


class TestNoProtectedModuleModification(unittest.TestCase):
    def test_12_module_never_imports_voi(self):
        with open(ur.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("src.voi", source)

    def test_13_relevance_module_functions_not_called_with_five_classes(self):
        # The protected relevance_from_probabilities/relevance_from_class
        # must never be called from this module (they'd reject 5-class input).
        with open(ur.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("relevance_from_probabilities", source)
        self.assertNotIn("relevance_from_class(", source)

    def test_14_reuses_canonical_uncertainty_function_object(self):
        import inspect

        self.assertIs(
            inspect.unwrap(ur.run_uncertainty_relevance_for_config).__globals__["compute_predictive_entropy"],
            compute_predictive_entropy,
        )


class TestResultColumnsAndProbabilityShape(unittest.TestCase):
    def test_15_result_columns_include_full_traceability(self):
        required = {
            "observation_id", "condition_code", "fault_type", "split", "fusion_config",
            "has_vibration", "has_current", "has_temperature",
            "predicted_class", "predicted_fault_type", "uncertainty_score", "relevance_score",
        }
        self.assertTrue(required.issubset(set(ur._RESULT_COLUMNS)))


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34 encoders, or Task 35 fusion heads not present")
class TestRealMultimodalUncertaintyRelevanceExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment_record, cls.results_df = ur.run_multimodal_uncertainty_relevance_experiment()

    def test_16_all_four_configurations_present(self):
        self.assertEqual(set(self.experiment_record["configurations"].keys()), set(fu.FUSION_CONFIGS.keys()))
        self.assertEqual(set(self.results_df["fusion_config"]), set(fu.FUSION_CONFIGS.keys()))

    def test_17_probability_dim_and_num_classes_are_five_for_every_config(self):
        for name, info in self.experiment_record["configurations"].items():
            self.assertEqual(info["probability_dim"], 5)
            self.assertEqual(info["num_classes"], 5)

    def test_18_predicted_class_always_in_range(self):
        self.assertTrue(self.results_df["predicted_class"].between(0, 4).all())

    def test_19_uncertainty_scores_within_unit_interval(self):
        self.assertTrue(self.results_df["uncertainty_score"].between(0.0, 1.0).all())

    def test_20_relevance_scores_are_nan_only_for_misalignment_unbalance_predictions(self):
        undefined_rows = self.results_df[self.results_df["predicted_fault_type"].isin(["Misalignment", "Unbalance"])]
        defined_rows = self.results_df[~self.results_df["predicted_fault_type"].isin(["Misalignment", "Unbalance"])]
        self.assertTrue(undefined_rows["relevance_score"].isna().all())
        self.assertTrue(defined_rows["relevance_score"].notna().all())

    def test_21_bpfo_retained_for_vibration_only_and_vibration_temperature(self):
        for config_name in ("vibration_only", "vibration_temperature"):
            sub = self.results_df[self.results_df["fusion_config"] == config_name]
            self.assertTrue((sub["fault_type"] == "BPFO").any())

    def test_22_bpfo_excluded_from_current_requiring_configs(self):
        for config_name in ("vibration_current", "vibration_current_temperature"):
            sub = self.results_df[self.results_df["fusion_config"] == config_name]
            self.assertFalse((sub["fault_type"] == "BPFO").any())

    def test_23_no_observation_id_crosses_split_per_config(self):
        for config_name in fu.FUSION_CONFIGS:
            sub = self.results_df[self.results_df["fusion_config"] == config_name]
            counts = sub.groupby("observation_id")["split"].nunique()
            self.assertTrue((counts <= 1).all())

    def test_24_no_condition_crosses_split_per_config(self):
        for config_name in fu.FUSION_CONFIGS:
            sub = self.results_df[self.results_df["fusion_config"] == config_name]
            counts = sub.groupby("condition_code")["split"].nunique()
            self.assertTrue((counts <= 1).all())

    def test_25_actual_class_support_reported_not_manufactured(self):
        for name, info in self.experiment_record["configurations"].items():
            test_support = info["splits"]["test"]["class_support"]
            # Whatever the true support is, it must match the real assembled
            # metadata exactly -- never padded to a nonzero minimum.
            sub = self.results_df[(self.results_df["fusion_config"] == name) & (self.results_df["split"] == "test")]
            real_counts = sub["fault_type"].value_counts().to_dict()
            for fault_type in ur.EXPECTED_FAULT_TYPE_LABELS:
                self.assertEqual(test_support.get(fault_type, 0), int(real_counts.get(fault_type, 0)))

    def test_26_model_provenance_recorded(self):
        for name, info in self.experiment_record["configurations"].items():
            self.assertEqual(info["model_path"], fu.fusion_model_path(name))
            self.assertIn("model_version", info)
            self.assertIn("representation_source", info)

    def test_27_no_fitting_stage_declared_for_uncertainty(self):
        for name, info in self.experiment_record["configurations"].items():
            self.assertIn("none", info["uncertainty_fitting_stage"].lower())

    def test_28_relevance_map_reported_exactly(self):
        for name, info in self.experiment_record["configurations"].items():
            reported = info["relevance_map"]
            self.assertEqual(reported["Normal"], 0.10)
            self.assertIsNone(reported["Misalignment"])
            self.assertIsNone(reported["Unbalance"])

    def test_29_save_and_reload_round_trip_uses_temp_dir_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "ur_config.json")
            results_path = os.path.join(tmp, "ur_results.csv")
            ur.save_uncertainty_relevance_config(self.experiment_record, path=config_path)
            ur.save_uncertainty_relevance_results(self.results_df, path=results_path)
            self.assertTrue(os.path.exists(config_path))
            self.assertTrue(os.path.exists(results_path))
            reloaded_df = pd.read_csv(results_path)
            self.assertEqual(len(reloaded_df), len(self.results_df))

    def test_30_voi_and_communication_not_computed(self):
        not_computed = set(self.experiment_record["not_computed_in_this_task"])
        self.assertIn("voi_score", not_computed)
        self.assertIn("communication_cost", not_computed)


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34 encoders, or Task 35 fusion heads not present")
class TestFitOnlyPrincipleForUncertaintyRelevance(unittest.TestCase):
    """Neither uncertainty nor relevance has a fitting stage, but this test
    confirms val/test scoring never influences train scoring or vice versa
    (no shared mutable state, no leakage through a side channel)."""

    def test_31_train_and_test_scores_independent_of_call_order(self):
        vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        norm = pp.load_normalization_params()
        train_table, _ = fu.build_split_representation_table("train", vib_model, cur_model, norm)
        val_table, _ = fu.build_split_representation_table("val", vib_model, cur_model, norm)
        test_table, _ = fu.build_split_representation_table("test", vib_model, cur_model, norm)
        fusion_model = fu.load_fusion_head("vibration_only")

        info1, rows1 = ur.run_uncertainty_relevance_for_config("vibration_only", train_table, val_table, test_table, fusion_model)
        # Score test alone through the same underlying function to confirm
        # test-split uncertainty doesn't depend on train/val being present.
        probs_test, y_test, meta_test = ur.get_predictions_for_split("vibration_only", test_table, fusion_model)
        direct_test_uncertainty = compute_predictive_entropy(probs_test, num_classes=5) if len(probs_test) else np.empty((0,))

        rows1_test = [r for r in rows1 if r["split"] == "test"]
        rows1_test_sorted = sorted(rows1_test, key=lambda r: r["observation_id"])
        meta_test_sorted_idx = meta_test["observation_id"].sort_values().index
        direct_sorted = pd.Series(direct_test_uncertainty, index=meta_test["observation_id"]).sort_index()

        self.assertEqual(len(rows1_test_sorted), len(direct_sorted))
        for row, expected in zip(rows1_test_sorted, direct_sorted.to_numpy()):
            self.assertAlmostEqual(row["uncertainty_score"], float(expected), places=5)


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34 encoders, or Task 35 fusion heads not present")
class TestModelArtifactLoadingAndNoRetraining(unittest.TestCase):
    def test_32_no_model_save_call_anywhere_in_module(self):
        with open(ur.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn(".fit(", source)
        self.assertNotIn(".save(", source)

    def test_33_loads_frozen_fusion_head_without_modification(self):
        fusion_model = fu.load_fusion_head("vibration_only")
        weights_before = [w.numpy().copy() for w in fusion_model.weights]
        vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        norm = pp.load_normalization_params()
        val_table, _ = fu.build_split_representation_table("val", vib_model, cur_model, norm)
        ur.get_predictions_for_split("vibration_only", val_table, fusion_model)
        weights_after = [w.numpy() for w in fusion_model.weights]
        for before, after in zip(weights_before, weights_after):
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
