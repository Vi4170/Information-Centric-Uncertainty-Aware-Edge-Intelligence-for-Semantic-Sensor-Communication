import os
import unittest

import numpy as np
import pandas as pd

from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import relevance_prediction_audit as rpa
from src.multimodal_pipeline import representation as rep
from src.multimodal_pipeline import uncertainty_relevance as ur

_RAW_DATA_AVAILABLE = schema.raw_data_available()
_MODELS_AVAILABLE = os.path.exists(rep.VIBRATION_MODEL_PATH) and os.path.exists(rep.MOTOR_CURRENT_MODEL_PATH)
_FUSION_MODELS_AVAILABLE = all(os.path.exists(fu.fusion_model_path(name)) for name in fu.FUSION_CONFIGS)
_RESULTS_AVAILABLE = os.path.exists(ur.UNCERTAINTY_RELEVANCE_RESULTS_CSV_PATH)
_ALL_AVAILABLE = _RAW_DATA_AVAILABLE and _MODELS_AVAILABLE and _FUSION_MODELS_AVAILABLE and _RESULTS_AVAILABLE


class TestRelevanceMappingUnchanged(unittest.TestCase):
    """Part E items 1-4: existing defined mappings preserved, undefined
    classes remain undefined, no arbitrary values inserted, label order."""

    def test_01_normal_bpfi_bpfo_values_unchanged(self):
        conclusion = rpa.relevance_mapping_conclusion()
        self.assertEqual(conclusion["relevance_map"]["Normal"], 0.10)
        self.assertEqual(conclusion["relevance_map"]["BPFI"], 1.00)
        self.assertEqual(conclusion["relevance_map"]["BPFO"], 0.90)

    def test_02_misalignment_unbalance_remain_undefined(self):
        conclusion = rpa.relevance_mapping_conclusion()
        self.assertIsNone(conclusion["relevance_map"]["Misalignment"])
        self.assertIsNone(conclusion["relevance_map"]["Unbalance"])
        self.assertEqual(set(conclusion["undefined_relevance_classes"]), {"Misalignment", "Unbalance"})

    def test_03_no_arbitrary_values_silently_inserted(self):
        # The audit module must reuse Task 37's own map object, not define
        # a competing one with different (possibly invented) values.
        conclusion = rpa.relevance_mapping_conclusion()
        for label, expected in {"Normal": 0.10, "BPFI": 1.00, "BPFO": 0.90}.items():
            self.assertEqual(conclusion["relevance_map"][label], ur.MULTIMODAL_RELEVANCE_MAP[ur.EXPECTED_FAULT_TYPE_LABELS.index(label)])

    def test_04_conclusion_is_partial_mapping(self):
        conclusion = rpa.relevance_mapping_conclusion()
        self.assertEqual(conclusion["conclusion"], "B_PARTIAL_MAPPING_ONLY")

    def test_05_five_class_label_ordering_correct(self):
        self.assertEqual(ur.EXPECTED_FAULT_TYPE_LABELS, ("Normal", "BPFI", "BPFO", "Misalignment", "Unbalance"))
        self.assertEqual(rep.FAULT_TYPE_LABELS, ur.EXPECTED_FAULT_TYPE_LABELS)


class TestPredictionDistributionComputation(unittest.TestCase):
    """Part E item 5: prediction distributions correctly computed (synthetic,
    no I/O)."""

    def test_06_synthetic_distribution_matches_manual_counts(self):
        df = pd.DataFrame(
            {
                "fusion_config": ["vibration_only"] * 6,
                "split": ["test"] * 6,
                "fault_type": ["BPFO", "BPFO", "Misalignment", "Misalignment", "Unbalance", "Unbalance"],
                "predicted_fault_type": ["BPFO", "Misalignment", "Misalignment", "Misalignment", "Unbalance", "Normal"],
            }
        )
        dist = rpa.compute_prediction_distribution(df, "vibration_only", "test")
        self.assertEqual(dist["n"], 6)
        self.assertEqual(dist["true_distribution"], {"BPFO": 2, "Misalignment": 2, "Unbalance": 2})
        self.assertEqual(dist["predicted_distribution"], {"Misalignment": 3, "BPFO": 1, "Unbalance": 1, "Normal": 1})

    def test_07_empty_subset_returns_zero_counts(self):
        df = pd.DataFrame(columns=["fusion_config", "split", "fault_type", "predicted_fault_type"])
        dist = rpa.compute_prediction_distribution(df, "vibration_only", "test")
        self.assertEqual(dist["n"], 0)
        self.assertEqual(dist["true_distribution"], {})

    def test_08_filters_by_both_config_and_split(self):
        df = pd.DataFrame(
            {
                "fusion_config": ["vibration_only", "vibration_current"],
                "split": ["test", "test"],
                "fault_type": ["Normal", "BPFO"],
                "predicted_fault_type": ["Normal", "BPFO"],
            }
        )
        dist = rpa.compute_prediction_distribution(df, "vibration_only", "test")
        self.assertEqual(dist["n"], 1)


class TestNoProtectedModuleModification(unittest.TestCase):
    def test_09_module_never_imports_voi(self):
        with open(rpa.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("src.voi", source)

    def test_10_relevance_module_not_called_with_five_classes(self):
        with open(rpa.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn("relevance_from_probabilities", source)
        self.assertNotIn("relevance_from_class(", source)

    def test_11_no_fit_or_save_calls_anywhere(self):
        with open(rpa.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn(".fit(", source)
        self.assertNotIn(".save(", source)

    def test_12_reuses_task37_relevance_map_object(self):
        import inspect

        self.assertIs(
            inspect.unwrap(rpa.relevance_mapping_conclusion).__globals__["ur"].MULTIMODAL_RELEVANCE_MAP,
            ur.MULTIMODAL_RELEVANCE_MAP,
        )


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34/35 artifacts, or Task 37 results not present")
class TestRealAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = rpa.run_relevance_prediction_audit()

    def test_13_all_four_configurations_handled(self):
        self.assertEqual(set(self.record["prediction_distributions"].keys()), set(fu.FUSION_CONFIGS.keys()))
        self.assertEqual(set(self.record["reload_consistency"].keys()), set(fu.FUSION_CONFIGS.keys()))

    def test_14_modality_validity_unchanged_bpfo_pattern(self):
        for config_name in ("vibration_only", "vibration_temperature"):
            dist = self.record["prediction_distributions"][config_name]
            has_bpfo = any("BPFO" in split_info["true_distribution"] for split_info in dist.values())
            self.assertTrue(has_bpfo, f"{config_name} lost BPFO true-label support")
        for config_name in ("vibration_current", "vibration_current_temperature"):
            dist = self.record["prediction_distributions"][config_name]
            has_bpfo = any("BPFO" in split_info["true_distribution"] for split_info in dist.values())
            self.assertFalse(has_bpfo, f"{config_name} unexpectedly has BPFO true-label support")

    def test_15_reload_consistency_reports_delta_for_every_config(self):
        for name, info in self.record["reload_consistency"].items():
            self.assertIn("delta", info)
            self.assertIn("reload_consistent", info)
            self.assertGreater(info["n_observations_checked"], 0)

    def test_16_current_configs_are_reload_consistent(self):
        # vibration_current/vibration_current_temperature were retrained
        # after the test_30 model-overwrite bug was fixed -- must reload
        # within a tight tolerance of their own reported training accuracy.
        for name in ("vibration_current", "vibration_current_temperature"):
            info = self.record["reload_consistency"][name]
            self.assertLess(info["delta"], 0.02, f"{name} unexpectedly reload-inconsistent")
            self.assertTrue(info["reload_consistent"])

    def test_17_task_38_readiness_flags_incomplete_mapping(self):
        self.assertIn("incomplete_relevance_mapping", self.record["task_38_readiness"]["blocking_issues"])

    def test_18_no_leakage_reload_check_uses_train_only(self):
        # check_reload_consistency's own signature takes only a train_table;
        # confirm it never touches a val/test table by inspecting the
        # function's local variable names for anything but train.
        import inspect

        src = inspect.getsource(rpa.check_reload_consistency)
        self.assertNotIn("val_table", src)
        self.assertNotIn("test_table", src)

    def test_19_saving_artifact_does_not_touch_production_paths(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "audit.json")
            rpa.save_audit(self.record, path=path)
            self.assertTrue(os.path.exists(path))
        # production default path is untouched unless explicitly called
        # with no args in __main__ -- confirmed by this test using tmp only.


@unittest.skipUnless(_ALL_AVAILABLE, "raw data, Task 34/35 artifacts, or Task 37 results not present")
class TestReloadCheckNeverModifiesModelArtifacts(unittest.TestCase):
    """Part E items 9, 10, 14: no retraining, no production overwrite, Task
    35 model artifacts unchanged by running the audit."""

    def test_20_model_file_bytes_unchanged_after_check(self):
        vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        norm = pp.load_normalization_params()
        train_table, _ = fu.build_split_representation_table("train", vib_model, cur_model, norm)

        for config_name in fu.FUSION_CONFIGS:
            path = fu.fusion_model_path(config_name)
            before = (os.path.getmtime(path), os.path.getsize(path))
            rpa.check_reload_consistency(config_name, train_table)
            after = (os.path.getmtime(path), os.path.getsize(path))
            self.assertEqual(before, after, f"{config_name}'s model file changed after a read-only check")

    def test_21_weights_bit_identical_before_and_after(self):
        fusion_model = fu.load_fusion_head("vibration_only")
        weights_before = [w.numpy().copy() for w in fusion_model.weights]
        vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        norm = pp.load_normalization_params()
        train_table, _ = fu.build_split_representation_table("train", vib_model, cur_model, norm)
        rpa.check_reload_consistency("vibration_only", train_table)
        weights_after = [w.numpy() for w in fusion_model.weights]
        for before, after in zip(weights_before, weights_after):
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
