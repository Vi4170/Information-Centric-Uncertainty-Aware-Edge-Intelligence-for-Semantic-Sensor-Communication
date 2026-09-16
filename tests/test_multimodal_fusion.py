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

_RAW_DATA_AVAILABLE = schema.raw_data_available()
_MODELS_AVAILABLE = os.path.exists(rep.VIBRATION_MODEL_PATH) and os.path.exists(rep.MOTOR_CURRENT_MODEL_PATH)
_SMALL_CONDITION = "0Nm_BPFI_03"
_CORRUPT_CONDITION = "4Nm_BPFO_10"


class TestConfigDefinitions(unittest.TestCase):
    def test_01_four_configurations_defined(self):
        self.assertEqual(
            set(fu.FUSION_CONFIGS.keys()),
            {"vibration_only", "vibration_current", "vibration_temperature", "vibration_current_temperature"},
        )

    def test_02_acoustic_never_appears_in_any_configuration(self):
        for modalities in fu.FUSION_CONFIGS.values():
            self.assertNotIn("acoustic", modalities)

    def test_03_every_configuration_includes_vibration(self):
        for modalities in fu.FUSION_CONFIGS.values():
            self.assertIn("vibration", modalities)

    def test_04_modality_order_is_fixed_vibration_current_temperature(self):
        self.assertEqual(fu.MODALITY_ORDER, ("vibration", "motor_current", "temperature"))

    def test_05_configurations_respect_fixed_modality_order(self):
        for modalities in fu.FUSION_CONFIGS.values():
            ordered = tuple(m for m in fu.MODALITY_ORDER if m in modalities)
            self.assertEqual(modalities, ordered)


class TestDimensions(unittest.TestCase):
    def test_06_vibration_only_concat_dim(self):
        self.assertEqual(fu.concat_dim_for_config("vibration_only"), 64)

    def test_07_vibration_current_concat_dim(self):
        self.assertEqual(fu.concat_dim_for_config("vibration_current"), 128)

    def test_08_vibration_temperature_concat_dim(self):
        self.assertEqual(fu.concat_dim_for_config("vibration_temperature"), 66)

    def test_09_vibration_current_temperature_concat_dim(self):
        self.assertEqual(fu.concat_dim_for_config("vibration_current_temperature"), 130)

    def test_10_temperature_not_forced_to_64(self):
        self.assertEqual(fu._MODALITY_DIM["temperature"], 2)
        self.assertNotEqual(fu._MODALITY_DIM["temperature"], fu.FUSED_DIM)

    def test_11_fused_dim_same_across_all_configs(self):
        self.assertEqual(fu.FUSED_DIM, rep.VIBRATION_EMBEDDING_DIM)

    def test_12_unknown_config_rejected_by_concat_dim(self):
        with self.assertRaises(ValueError):
            fu.concat_dim_for_config("not_a_config")

    def test_13_unknown_config_rejected_by_model_path(self):
        with self.assertRaises(ValueError):
            fu.fusion_model_path("not_a_config")


class TestFusionHeadArchitecture(unittest.TestCase):
    def test_14_output_shapes_for_each_concat_dim(self):
        for config_name in fu.FUSION_CONFIGS:
            dim = fu.concat_dim_for_config(config_name)
            model = fu.build_fusion_head(dim)
            x = np.random.randn(4, dim).astype("float32")
            probs = model.predict(x, verbose=0)
            self.assertEqual(probs.shape, (4, rep.NUM_CLASSES))
            fused = fu.get_fused_representation(model, x)
            self.assertEqual(fused.shape, (4, fu.FUSED_DIM))

    def test_15_deterministic_inference(self):
        model = fu.build_fusion_head(128)
        x = np.random.randn(6, 128).astype("float32")
        p1 = model.predict(x, verbose=0)
        p2 = model.predict(x, verbose=0)
        np.testing.assert_array_equal(p1, p2)

    def test_16_deterministic_fused_representation(self):
        model = fu.build_fusion_head(66)
        x = np.random.randn(6, 66).astype("float32")
        e1 = fu.get_fused_representation(model, x)
        e2 = fu.get_fused_representation(model, x)
        np.testing.assert_array_equal(e1, e2)

    def test_17_no_concatenation_of_raw_signals_in_architecture(self):
        # Fusion head's only input is the already-concatenated embedding
        # vector -- never a (window_size, channels)-shaped raw tensor.
        model = fu.build_fusion_head(130)
        self.assertEqual(len(model.input_shape), 2)


class TestModelSaveLoad(unittest.TestCase):
    def test_18_save_load_roundtrip_preserves_predictions(self):
        model = fu.build_fusion_head(64)
        x = np.random.randn(5, 64).astype("float32")
        before = model.predict(x, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fusion.keras")
            model.save(path)
            import keras

            loaded = keras.models.load_model(path, compile=False)
            after = loaded.predict(x, verbose=0)
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_19_load_fusion_head_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(fu, "FUSION_MODEL_DIR", tmp):
                with self.assertRaises(FileNotFoundError):
                    fu.load_fusion_head("vibration_only")


@unittest.skipUnless(_RAW_DATA_AVAILABLE and _MODELS_AVAILABLE, "raw data or Task 34 models not present")
class TestRepresentationTableAndAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cls.cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        cls.norm = pp.load_normalization_params()

    def test_20_table_built_only_from_requested_split(self):
        table, _ = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        self.assertTrue((table["split"] == "val").all())

    def test_21_vibration_only_uses_all_windows(self):
        table, _ = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        X, y, meta = fu.assemble_fusion_split_arrays("vibration_only", table)
        self.assertEqual(len(X), len(table))

    def test_22_corrupt_condition_excluded_from_current_config_not_fabricated(self):
        table, excluded = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        self.assertTrue(any(e["condition_code"] == _CORRUPT_CONDITION for e in excluded))
        X, y, meta = fu.assemble_fusion_split_arrays("vibration_current", table)
        self.assertNotIn(_CORRUPT_CONDITION, set(meta["condition_code"]))

    def test_23_corrupt_condition_still_present_in_vibration_only(self):
        table, _ = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        X, y, meta = fu.assemble_fusion_split_arrays("vibration_only", table)
        self.assertIn(_CORRUPT_CONDITION, set(meta["condition_code"]))

    def test_23b_corrupt_current_condition_retained_in_temperature_config(self):
        # Task 35 corrective audit regression test: invalid current must NOT
        # remove an otherwise-valid temperature observation sharing the same
        # raw file. 4Nm_BPFO_10 has genuinely valid temperature channels
        # (confirmed by direct raw-channel inspection of all 45 files) even
        # though its current channels are genuinely invalid.
        table, excluded = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        self.assertFalse(any(e["condition_code"] == _CORRUPT_CONDITION and e["modality"] == "temperature" for e in excluded))
        X, y, meta = fu.assemble_fusion_split_arrays("vibration_temperature", table)
        self.assertIn(_CORRUPT_CONDITION, set(meta["condition_code"]))

    def test_23c_corrupt_current_condition_still_excluded_from_current_temperature_config(self):
        # The stricter configuration requires motor_current too, so it must
        # still exclude the condition -- current itself is genuinely invalid.
        table, _ = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        X, y, meta = fu.assemble_fusion_split_arrays("vibration_current_temperature", table)
        self.assertNotIn(_CORRUPT_CONDITION, set(meta["condition_code"]))

    def test_24_no_zero_filled_substitution_for_missing_current(self):
        table, _ = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        corrupt_rows = table[table["condition_code"] == _CORRUPT_CONDITION]
        # current_embedding must be None (missing), never an array of zeros.
        self.assertTrue(corrupt_rows["current_embedding"].isna().all() or all(
            v is None for v in corrupt_rows["current_embedding"]
        ))

    def test_25_provenance_columns_preserved_through_assembly(self):
        table, _ = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        _, _, meta = fu.assemble_fusion_split_arrays("vibration_temperature", table)
        required = {
            "observation_id", "condition_code", "split",
            "source_vibration_file", "source_temperature_current_file",
        }
        self.assertTrue(required.issubset(set(meta.columns)))

    def test_26_observation_ids_shared_across_configs_are_identical(self):
        table, _ = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        _, _, meta_vc = fu.assemble_fusion_split_arrays("vibration_current", table)
        _, _, meta_vct = fu.assemble_fusion_split_arrays("vibration_current_temperature", table)
        self.assertEqual(set(meta_vc["observation_id"]), set(meta_vct["observation_id"]))

    def test_27_concat_columns_are_in_fixed_modality_order(self):
        table, _ = fu.build_split_representation_table("val", self.vib_model, self.cur_model, self.norm)
        X, y, meta = fu.assemble_fusion_split_arrays("vibration_current", table)
        first_vib = np.stack(meta["vibration_embedding"].to_numpy())[0]
        first_cur = np.stack(meta["current_embedding"].to_numpy())[0]
        np.testing.assert_allclose(X[0, :64], first_vib, rtol=1e-5)
        np.testing.assert_allclose(X[0, 64:128], first_cur, rtol=1e-5)


class TestNoSilentModalitySubstitution(unittest.TestCase):
    def test_28_missing_modality_column_never_backfilled_with_zeros(self):
        table = pd.DataFrame(
            {
                "observation_id": ["a", "b"],
                "condition_code": ["c1", "c1"],
                "fault_type": ["Normal", "Normal"],
                "split": ["train", "train"],
                "vibration_embedding": [np.ones(64), np.ones(64)],
                "current_embedding": [np.ones(64), None],
                "temperature_representation": [np.ones(2), np.ones(2)],
                "has_vibration": [True, True],
                "has_current": [True, False],
                "has_temperature": [True, True],
            }
        )
        X, y, meta = fu.assemble_fusion_split_arrays("vibration_current", table)
        self.assertEqual(len(X), 1)
        self.assertEqual(meta.iloc[0]["observation_id"], "a")

    def test_29_empty_table_returns_empty_arrays_with_correct_width(self):
        empty = pd.DataFrame(
            columns=[
                "observation_id", "condition_code", "fault_type", "split",
                "vibration_embedding", "current_embedding", "temperature_representation",
                "has_vibration", "has_current", "has_temperature",
            ]
        )
        X, y, meta = fu.assemble_fusion_split_arrays("vibration_current_temperature", empty)
        self.assertEqual(X.shape, (0, 130))
        self.assertEqual(len(y), 0)


class TestTrainOnlyFittingAndLeakage(unittest.TestCase):
    @unittest.skipUnless(_RAW_DATA_AVAILABLE and _MODELS_AVAILABLE, "raw data or Task 34 models not present")
    def test_30_train_all_fusion_heads_never_requests_test_split(self):
        vib_model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        cur_model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        norm = pp.load_normalization_params()
        real_registry = pp.get_condition_registry()

        row_train = real_registry[real_registry["condition_code"] == "0Nm_BPFI_03"].iloc[0].copy()
        row_train["split"] = "train"
        row_val = real_registry[real_registry["condition_code"] == "0Nm_Misalign_01"].iloc[0].copy()
        row_val["split"] = "val"
        row_test = real_registry[real_registry["condition_code"] == "0Nm_Unbalance_1169mg"].iloc[0].copy()
        row_test["split"] = "test"
        fake_registry = pd.DataFrame([row_train, row_val, row_test])

        requested_splits = []
        real_build = fu.build_split_representation_table

        def spy(split, *args, **kwargs):
            requested_splits.append(split)
            return real_build(split, *args, **kwargs)

        # train_all_fusion_heads() saves each config to fusion_model_path(),
        # which reads the module-level FUSION_MODEL_DIR -- redirect it to a
        # scratch directory so this test (which trains throwaway 1-epoch
        # models on a synthetic 3-row registry) can never overwrite the real
        # production fusion models on disk.
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            pp, "get_condition_registry", return_value=fake_registry
        ), patch.object(fu, "build_split_representation_table", side_effect=spy), patch.object(
            fu, "FUSION_MODEL_DIR", tmp
        ):
            fu.train_all_fusion_heads(epochs=1)

        self.assertIn("train", requested_splits)
        self.assertIn("val", requested_splits)
        self.assertNotIn("test", requested_splits)

    def test_31_assemble_fusion_split_arrays_rejects_unknown_config(self):
        with self.assertRaises(ValueError):
            fu.assemble_fusion_split_arrays("bad_config", pd.DataFrame())


class TestClassSupportReporting(unittest.TestCase):
    @unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
    def test_32_class_support_by_split_covers_all_splits_and_classes(self):
        support = fu.class_support_by_split()
        self.assertEqual(set(support.keys()), set(schema.SPLIT_NAMES))
        for split_counts in support.values():
            self.assertEqual(set(split_counts.keys()), set(rep.FAULT_TYPE_LABELS))

    @unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
    def test_33_normal_has_zero_val_test_support_disclosed_not_hidden(self):
        support = fu.class_support_by_split()
        self.assertEqual(support["val"]["Normal"], 0)
        self.assertEqual(support["test"]["Normal"], 0)
        self.assertGreater(support["train"]["Normal"], 0)


@unittest.skipUnless(os.path.exists(fu.FUSION_CONFIG_JSON_PATH), "fusion config has not been generated yet")
class TestSavedFusionConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        with open(fu.FUSION_CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            cls.config = json.load(f)

    def test_34_task_and_modality_order_present(self):
        self.assertEqual(self.config["task"], 35)
        self.assertEqual(self.config["modality_order"], ["vibration", "motor_current", "temperature"])

    def test_35_all_four_configurations_recorded(self):
        self.assertEqual(set(self.config["configurations"].keys()), set(fu.FUSION_CONFIGS.keys()))

    def test_36_modality_encoders_reported_frozen(self):
        self.assertTrue(self.config["modality_encoders_frozen"])

    def test_37_test_split_reported_untouched_during_training(self):
        self.assertFalse(self.config["training"]["test_split_touched_during_training"])

    def test_38_bpfo_limitation_disclosed(self):
        key = "known_limitation_bpfo_missing_from_current_configs"
        self.assertIn(key, self.config)
        self.assertIn("BPFO", self.config[key])

    def test_39_class_coverage_shows_vibration_only_has_all_five_classes(self):
        coverage = self.config["class_coverage_by_configuration"]["vibration_only"]
        self.assertEqual(coverage["classes_with_zero_training_support"], [])

    def test_40_class_coverage_shows_bpfo_missing_only_for_current_requiring_configs(self):
        for name in ("vibration_current", "vibration_current_temperature"):
            coverage = self.config["class_coverage_by_configuration"][name]
            self.assertIn("BPFO", coverage["classes_with_zero_training_support"])

    def test_40b_class_coverage_shows_bpfo_retained_for_vibration_temperature(self):
        # Corrective-audit regression: temperature alone is genuinely valid
        # for BPFO, so vibration_temperature must NOT lose BPFO the way the
        # current-requiring configurations correctly do.
        coverage = self.config["class_coverage_by_configuration"]["vibration_temperature"]
        self.assertNotIn("BPFO", coverage["classes_with_zero_training_support"])

    def test_41_excluded_conditions_all_bpfo(self):
        excluded_codes = {e["condition_code"] for e in self.config["excluded_conditions"]}
        for code in excluded_codes:
            self.assertIn("BPFO", code)


@unittest.skipUnless(
    os.path.exists(fu.FUSION_TRAINING_HISTORY_JSON_PATH), "fusion training history has not been generated yet"
)
class TestSavedFusionHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        with open(fu.FUSION_TRAINING_HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
            cls.history = json.load(f)

    def test_42_all_four_configs_present(self):
        self.assertEqual(set(self.history.keys()), set(fu.FUSION_CONFIGS.keys()))

    def test_43_accuracy_history_length_matches_epochs(self):
        for h in self.history.values():
            self.assertEqual(len(h["train_accuracy"]), h["epochs"])

    def test_44_vibration_current_has_fewer_observations_than_vibration_only(self):
        n_only = self.history["vibration_only"]["n_train_observations"]
        n_current = self.history["vibration_current"]["n_train_observations"]
        self.assertLess(n_current, n_only)


class TestSavedFusionModelsMatchDeclaredDims(unittest.TestCase):
    def test_45_every_trained_fusion_head_matches_its_declared_concat_dim(self):
        checked = 0
        for config_name in fu.FUSION_CONFIGS:
            path = fu.fusion_model_path(config_name)
            if not os.path.exists(path):
                continue
            model = fu.load_fusion_head(config_name)
            self.assertEqual(model.input_shape[1], fu.concat_dim_for_config(config_name))
            self.assertEqual(model.output_shape[-1], rep.NUM_CLASSES)
            checked += 1
        if checked == 0:
            self.skipTest("no fusion heads trained yet")


class TestSavedFusionModelVibrationOnly(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(fu.fusion_model_path("vibration_only")), "not trained yet")
    def test_46_vibration_only_model_dims(self):
        model = fu.load_fusion_head("vibration_only")
        self.assertEqual(model.input_shape[1], 64)


class TestSavedFusionModelVibrationCurrentTemperature(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(fu.fusion_model_path("vibration_current_temperature")), "not trained yet")
    def test_47_vibration_current_temperature_model_dims(self):
        model = fu.load_fusion_head("vibration_current_temperature")
        self.assertEqual(model.input_shape[1], 130)


if __name__ == "__main__":
    unittest.main()
