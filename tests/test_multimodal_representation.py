import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp
from src.multimodal_pipeline import representation as rep

_RAW_DATA_AVAILABLE = schema.raw_data_available()
_SMALL_CONDITION = "0Nm_BPFI_03"
_CORRUPT_CONDITION = "4Nm_BPFO_10"


class TestArchitectureConstants(unittest.TestCase):
    def test_01_vibration_input_shape_has_four_channels(self):
        self.assertEqual(rep.VIBRATION_INPUT_SHAPE, (schema.WINDOW_SIZE_SAMPLES, 4))

    def test_02_motor_current_input_shape_has_three_channels(self):
        self.assertEqual(rep.MOTOR_CURRENT_INPUT_SHAPE, (rep.MOTOR_CURRENT_WINDOW_SIZE_SAMPLES, 3))

    def test_03_embedding_dims_reuse_project_convention(self):
        from src.cnn.config import EMBEDDING_DIM

        self.assertEqual(rep.VIBRATION_EMBEDDING_DIM, EMBEDDING_DIM)
        self.assertEqual(rep.MOTOR_CURRENT_EMBEDDING_DIM, EMBEDDING_DIM)

    def test_04_temperature_representation_dim_is_two(self):
        self.assertEqual(rep.TEMPERATURE_REPRESENTATION_DIM, 2)

    def test_05_num_classes_matches_fault_type_labels(self):
        self.assertEqual(rep.NUM_CLASSES, len(rep.FAULT_TYPE_LABELS))
        self.assertEqual(rep.NUM_CLASSES, 5)

    def test_06_no_model_path_defined_for_temperature(self):
        self.assertFalse(hasattr(rep, "TEMPERATURE_MODEL_PATH"))


class TestEncoderConstructionAndForwardPass(unittest.TestCase):
    def test_07_vibration_encoder_output_shapes(self):
        model = rep.build_vibration_encoder()
        x = np.random.randn(3, *rep.VIBRATION_INPUT_SHAPE).astype("float32")
        probs = model.predict(x, verbose=0)
        self.assertEqual(probs.shape, (3, rep.NUM_CLASSES))
        emb = rep.get_vibration_embeddings(model, x)
        self.assertEqual(emb.shape, (3, rep.VIBRATION_EMBEDDING_DIM))

    def test_08_motor_current_encoder_output_shapes(self):
        model = rep.build_motor_current_encoder()
        x = np.random.randn(3, *rep.MOTOR_CURRENT_INPUT_SHAPE).astype("float32")
        probs = model.predict(x, verbose=0)
        self.assertEqual(probs.shape, (3, rep.NUM_CLASSES))
        emb = rep.get_motor_current_embeddings(model, x)
        self.assertEqual(emb.shape, (3, rep.MOTOR_CURRENT_EMBEDDING_DIM))

    def test_09_vibration_and_current_models_are_separate_objects(self):
        vib = rep.build_vibration_encoder()
        cur = rep.build_motor_current_encoder()
        self.assertIsNot(vib, cur)
        self.assertNotEqual(vib.input_shape, cur.input_shape)

    def test_10_deterministic_inference_same_input_same_output(self):
        model = rep.build_vibration_encoder()
        x = np.random.randn(5, *rep.VIBRATION_INPUT_SHAPE).astype("float32")
        p1 = model.predict(x, verbose=0)
        p2 = model.predict(x, verbose=0)
        np.testing.assert_array_equal(p1, p2)

    def test_11_embedding_extraction_deterministic(self):
        model = rep.build_motor_current_encoder()
        x = np.random.randn(5, *rep.MOTOR_CURRENT_INPUT_SHAPE).astype("float32")
        e1 = rep.get_motor_current_embeddings(model, x)
        e2 = rep.get_motor_current_embeddings(model, x)
        np.testing.assert_array_equal(e1, e2)


class TestModelSaveLoad(unittest.TestCase):
    def test_12_vibration_save_load_roundtrip_preserves_predictions(self):
        model = rep.build_vibration_encoder()
        x = np.random.randn(4, *rep.VIBRATION_INPUT_SHAPE).astype("float32")
        before = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vib.keras")
            rep.save_encoder(model, path)
            self.assertTrue(os.path.exists(path))
            loaded = rep.load_encoder(path)
            after = loaded.predict(x, verbose=0)

        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_13_motor_current_save_load_roundtrip_preserves_embeddings(self):
        model = rep.build_motor_current_encoder()
        x = np.random.randn(4, *rep.MOTOR_CURRENT_INPUT_SHAPE).astype("float32")
        before = rep.get_motor_current_embeddings(model, x)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cur.keras")
            rep.save_encoder(model, path)
            loaded = rep.load_encoder(path)
            after = rep.get_motor_current_embeddings(loaded, x)

        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_14_load_encoder_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            rep.load_encoder("models/multimodal/does_not_exist.keras")


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestTemperatureRepresentation(unittest.TestCase):
    def test_15_is_identity_over_task33_output(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        norm = pp.load_normalization_params()
        X_direct, meta_direct = pp.build_temperature_features_for_condition(row, normalization=norm["temperature"])
        X_via_rep, meta_via_rep = rep.get_temperature_representation(row, normalization_params=norm)
        np.testing.assert_array_equal(X_direct, X_via_rep)
        self.assertEqual(len(meta_direct), len(meta_via_rep))

    def test_16_output_dim_matches_declared_constant(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        X, _ = rep.get_temperature_representation(row)
        self.assertEqual(X.shape[1], rep.TEMPERATURE_REPRESENTATION_DIM)

    def test_17_no_high_frequency_fabrication(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        X, meta = rep.get_temperature_representation(row)
        # Representation is 2 numbers per window, never the raw window length.
        self.assertNotIn(pp.TEMPERATURE_CHANNEL_NAMES[0], "")  # sanity: constant exists
        self.assertEqual(X.ndim, 2)
        self.assertLess(X.shape[1], 100)


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestChannelPreservationAndProvenance(unittest.TestCase):
    def test_18_vibration_metadata_preserves_four_channel_names(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        _, _, meta, _ = rep.assemble_split_arrays("vibration", meta_split := row["split"])
        self.assertEqual(len(meta.iloc[0]["channel_names"]), 4)

    def test_19_motor_current_metadata_preserves_three_channel_names(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        _, _, meta, _ = rep.assemble_split_arrays("motor_current", row["split"])
        self.assertEqual(len(meta.iloc[0]["channel_names"]), 3)

    def test_20_assembled_metadata_has_required_provenance_columns(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        _, _, meta, _ = rep.assemble_split_arrays("vibration", row["split"])
        required = {"observation_id", "condition_code", "source_recording_id", "split", "modality"}
        self.assertTrue(required.issubset(set(meta.columns)))

    def test_21_labels_align_with_fault_type_of_source_condition(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        X, y, meta, _ = rep.assemble_split_arrays("vibration", row["split"])
        subset = meta[meta["condition_code"] == _SMALL_CONDITION]
        expected_label = rep.FAULT_TYPE_TO_LABEL_ID[row["fault_type"]]
        rows_idx = subset.index
        self.assertTrue((y[rows_idx] == expected_label).all())


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestCorruptFileHandling(unittest.TestCase):
    """4Nm_BPFO_10.tdms genuinely has two empty current channels (found while
    implementing Task 34) -- verifies it is excluded, not fabricated."""

    def test_22_motor_current_excludes_corrupt_condition(self):
        row = pp.get_condition_row(_CORRUPT_CONDITION)
        _, _, meta, excluded = rep.assemble_split_arrays("motor_current", row["split"])
        self.assertTrue(any(e["condition_code"] == _CORRUPT_CONDITION for e in excluded))
        self.assertNotIn(_CORRUPT_CONDITION, set(meta.get("condition_code", [])))

    def test_23_temperature_excludes_corrupt_condition(self):
        row = pp.get_condition_row(_CORRUPT_CONDITION)
        _, _, meta, excluded = rep.assemble_split_arrays("temperature", row["split"])
        self.assertTrue(any(e["condition_code"] == _CORRUPT_CONDITION for e in excluded))

    def test_24_vibration_is_unaffected_by_the_corrupt_tdms_file(self):
        row = pp.get_condition_row(_CORRUPT_CONDITION)
        _, _, meta, excluded = rep.assemble_split_arrays("vibration", row["split"])
        self.assertEqual(excluded, [])
        self.assertIn(_CORRUPT_CONDITION, set(meta["condition_code"]))

    def test_25_corrupt_file_raises_specific_exception_type(self):
        row = pp.get_condition_row(_CORRUPT_CONDITION)
        with self.assertRaises(pp.CorruptRawFileError):
            pp._read_temperature_current_tdms(row["temperature_current_path"])

    def test_26_corrupt_file_error_is_a_value_error_subclass(self):
        self.assertTrue(issubclass(pp.CorruptRawFileError, ValueError))


class TestTrainOnlyFittingAndLeakage(unittest.TestCase):
    @unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
    def test_27_assemble_split_arrays_never_mixes_splits(self):
        real_registry = pp.get_condition_registry()
        row_a = real_registry[real_registry["condition_code"] == "0Nm_BPFI_03"].iloc[0].copy()
        row_b = real_registry[real_registry["condition_code"] == "0Nm_BPFO_10"].iloc[0].copy()
        row_a["split"] = "train"
        row_b["split"] = "test"
        fake_registry = pd.DataFrame([row_a, row_b])

        with patch.object(pp, "get_condition_registry", return_value=fake_registry):
            _, _, meta_train, _ = rep.assemble_split_arrays("vibration", "train")
            _, _, meta_test, _ = rep.assemble_split_arrays("vibration", "test")

        self.assertEqual(set(meta_train["condition_code"]), {"0Nm_BPFI_03"})
        self.assertEqual(set(meta_test["condition_code"]), {"0Nm_BPFO_10"})

    @unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
    def test_28_train_encoder_never_touches_test_split(self):
        real_registry = pp.get_condition_registry()
        row_train = real_registry[real_registry["condition_code"] == "0Nm_BPFI_03"].iloc[0].copy()
        row_train["split"] = "train"
        row_val = real_registry[real_registry["condition_code"] == "0Nm_BPFO_10"].iloc[0].copy()
        row_val["split"] = "val"
        row_test = real_registry[real_registry["condition_code"] == "0Nm_Misalign_01"].iloc[0].copy()
        row_test["split"] = "test"
        fake_registry = pd.DataFrame([row_train, row_val, row_test])

        requested_splits = []
        real_assemble = rep.assemble_split_arrays

        def spy(modality, split, *args, **kwargs):
            requested_splits.append(split)
            return real_assemble(modality, split, *args, **kwargs)

        with patch.object(pp, "get_condition_registry", return_value=fake_registry), patch.object(
            rep, "assemble_split_arrays", side_effect=spy
        ):
            rep.train_vibration_encoder(epochs=1)

        self.assertIn("train", requested_splits)
        self.assertIn("val", requested_splits)
        self.assertNotIn("test", requested_splits)

    def test_29_unknown_modality_rejected(self):
        with self.assertRaises(ValueError):
            rep.assemble_split_arrays("acoustic", "train")

    def test_30_unknown_split_rejected(self):
        with self.assertRaises(ValueError):
            rep.assemble_split_arrays("vibration", "not_a_split")


@unittest.skipUnless(
    os.path.exists(rep.REPRESENTATION_CONFIG_JSON_PATH), "representation config has not been generated yet"
)
class TestSavedRepresentationConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        with open(rep.REPRESENTATION_CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            cls.config = json.load(f)

    def test_31_task_and_modality_sections_present(self):
        self.assertEqual(self.config["task"], 34)
        for modality in ("vibration", "motor_current", "temperature"):
            self.assertIn(modality, self.config)

    def test_32_vibration_and_current_report_reused_cwru_encoder(self):
        self.assertTrue(self.config["vibration"]["reused_existing_cwru_encoder"])
        self.assertTrue(self.config["motor_current"]["reused_existing_cwru_encoder"])

    def test_33_temperature_reports_no_reuse_and_no_model(self):
        self.assertFalse(self.config["temperature"]["reused_existing_cwru_encoder"])

    def test_34_known_limitation_is_disclosed(self):
        note = self.config["known_limitation_zero_normal_val_test_support"]
        self.assertIsInstance(note, str)
        self.assertGreater(len(note), 20)

    def test_35_training_never_touches_test_split(self):
        self.assertFalse(self.config["training"]["test_split_touched_during_training"])


@unittest.skipUnless(
    os.path.exists(rep.TRAINING_HISTORY_JSON_PATH), "training history has not been generated yet"
)
class TestSavedTrainingHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        with open(rep.TRAINING_HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
            cls.history = json.load(f)

    def test_36_both_modalities_present(self):
        self.assertEqual(set(self.history.keys()), {"vibration", "motor_current"})

    def test_37_history_lengths_match_epochs(self):
        for modality, h in self.history.items():
            self.assertEqual(len(h["train_accuracy"]), h["epochs"])

    def test_38_no_negative_or_nan_losses(self):
        for h in self.history.values():
            for loss in h["train_loss"] + h["val_loss"]:
                self.assertTrue(np.isfinite(loss))
                self.assertGreaterEqual(loss, 0.0)


@unittest.skipUnless(os.path.exists(rep.VIBRATION_MODEL_PATH), "vibration encoder has not been trained yet")
class TestSavedVibrationModel(unittest.TestCase):
    def test_39_loads_and_matches_declared_shape(self):
        model = rep.load_encoder(rep.VIBRATION_MODEL_PATH)
        self.assertEqual(tuple(model.input_shape[1:]), rep.VIBRATION_INPUT_SHAPE)
        self.assertEqual(model.output_shape[-1], rep.NUM_CLASSES)


@unittest.skipUnless(os.path.exists(rep.MOTOR_CURRENT_MODEL_PATH), "motor_current encoder has not been trained yet")
class TestSavedMotorCurrentModel(unittest.TestCase):
    def test_40_loads_and_matches_declared_shape(self):
        model = rep.load_encoder(rep.MOTOR_CURRENT_MODEL_PATH)
        self.assertEqual(tuple(model.input_shape[1:]), rep.MOTOR_CURRENT_INPUT_SHAPE)
        self.assertEqual(model.output_shape[-1], rep.NUM_CLASSES)


if __name__ == "__main__":
    unittest.main()
