import os
import unittest

import numpy as np
import pandas as pd

from src.ims_pipeline.preprocessing import verify_observation_id_uniqueness
from src.paderborn_pipeline.preprocessing import (
    BEARING_TYPE,
    MODALITY_CHANNELS,
    OPERATING_CONDITIONS,
    PADERBORN_BEARING_REGISTRY,
    RAW_DATA_DIR,
    WINDOW_SIZE,
    _derive_damage_category,
    apply_normalization,
    build_measurement_metadata,
    discover_measurement_files,
    fit_train_normalization,
    list_bearing_codes,
    load_stream_windows,
    read_paderborn_mat_file,
    split_for_measurement,
    verify_no_measurement_crosses_split,
    verify_split_disjoint,
    window_channel_signal,
)

_RAW_DATA_AVAILABLE = all(
    os.path.isdir(os.path.join(RAW_DATA_DIR, code)) for code in ("K001", "KA01", "KB23", "KI04")
)


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "Extracted Paderborn raw data not present under data/raw/paderborn/")
class TestPaderbornPreprocessing(unittest.TestCase):
    def test_01_dataset_discovery_and_deterministic_ordering(self):
        for bearing_code in ("K001", "KA01"):
            for operating_condition in OPERATING_CONDITIONS:
                files_a = discover_measurement_files(bearing_code, operating_condition)
                files_b = discover_measurement_files(bearing_code, operating_condition)
                self.assertEqual(files_a, files_b)
                self.assertEqual(len(files_a), 20)
                numbers = [n for n, _ in files_a]
                self.assertEqual(numbers, sorted(numbers))
                self.assertEqual(set(numbers), set(range(1, 21)))
        with self.assertRaises(ValueError):
            discover_measurement_files("K001", "NOT_A_CONDITION")

    def test_02_deterministic_observation_ids(self):
        meta_a = build_measurement_metadata("KA01", "N15_M07_F10", "vibration", 0)
        meta_b = build_measurement_metadata("KA01", "N15_M07_F10", "vibration", 0)
        self.assertTrue(meta_a.equals(meta_b))

        X_a, wmeta_a = load_stream_windows("KA01", "N15_M07_F10", "vibration", 0, measurement_numbers=[1, 2])
        X_b, wmeta_b = load_stream_windows("KA01", "N15_M07_F10", "vibration", 0, measurement_numbers=[1, 2])
        self.assertEqual(wmeta_a["observation_id"].tolist(), wmeta_b["observation_id"].tolist())
        expected_prefix = "paderborn_KA01_N15_M07_F10_m01_vibration_1_w"
        self.assertTrue(wmeta_a["observation_id"].iloc[0].startswith(expected_prefix))

    def test_03_correct_parsing_of_source_matlab_files(self):
        path = os.path.join(RAW_DATA_DIR, "K001", "N15_M07_F10_K001_1.mat")
        channels = read_paderborn_mat_file(path)
        self.assertIn("vibration_1", channels)
        self.assertIn("phase_current_1", channels)
        self.assertIn("phase_current_2", channels)
        for name in ("vibration_1", "phase_current_1", "phase_current_2"):
            self.assertEqual(channels[name].ndim, 1)
            self.assertGreater(channels[name].size, 200000)
            self.assertTrue(np.isfinite(channels[name]).all())
        with self.assertRaises(FileNotFoundError):
            read_paderborn_mat_file(os.path.join(RAW_DATA_DIR, "K001", "does_not_exist.mat"))

    def test_04_bearing_identifier_extraction(self):
        for bearing_code in ("K001", "KA01", "KB23"):
            meta = build_measurement_metadata(bearing_code, "N15_M07_F10", "vibration", 0)
            self.assertEqual(set(meta["bearing_code"]), {bearing_code})
            self.assertEqual(set(meta["bearing_type"]), {BEARING_TYPE})
        self.assertEqual(len(list_bearing_codes()), 32)
        self.assertEqual(len(list_bearing_codes("healthy")), 6)
        self.assertEqual(len(list_bearing_codes("damaged")), 26)

    def test_05_operating_condition_metadata(self):
        for operating_condition in OPERATING_CONDITIONS:
            meta = build_measurement_metadata("KA01", operating_condition, "vibration", 0)
            self.assertEqual(set(meta["operating_condition"]), {operating_condition})

    def test_06_health_damage_metadata_preserved(self):
        healthy_meta = build_measurement_metadata("K001", "N15_M07_F10", "vibration", 0)
        self.assertEqual(set(healthy_meta["health_state"]), {"healthy"})
        self.assertTrue((healthy_meta["damage_modes"].apply(len) == 0).all())

        damaged_meta = build_measurement_metadata("KA01", "N15_M07_F10", "vibration", 0)
        self.assertEqual(set(damaged_meta["health_state"]), {"damaged"})
        self.assertTrue((damaged_meta["damage_modes"].apply(tuple) == ("artificial",)).all())
        self.assertTrue((damaged_meta["damage_components"].apply(tuple) == ("OR",)).all())

        multi_damage_meta = build_measurement_metadata("KB23", "N15_M07_F10", "vibration", 0)
        self.assertEqual(multi_damage_meta["damage_modes"].iloc[0], ("fatigue", "fatigue", "fatigue"))
        self.assertEqual(multi_damage_meta["damage_components"].iloc[0], ("IR", "IR", "OR"))

    def test_07_artificial_real_damage_category(self):
        self.assertEqual(_derive_damage_category(PADERBORN_BEARING_REGISTRY["K001"]["damage_modes"]), "n/a")
        self.assertEqual(_derive_damage_category(PADERBORN_BEARING_REGISTRY["KA01"]["damage_modes"]), "artificial")
        self.assertEqual(_derive_damage_category(PADERBORN_BEARING_REGISTRY["KB23"]["damage_modes"]), "real")
        self.assertEqual(_derive_damage_category(PADERBORN_BEARING_REGISTRY["KI04"]["damage_modes"]), "real")
        self.assertEqual(_derive_damage_category(PADERBORN_BEARING_REGISTRY["KA08"]["damage_modes"]), "artificial")
        with self.assertRaises(ValueError):
            _derive_damage_category(("artificial", "fatigue"))

    def test_08_modality_metadata_preserved(self):
        vib_meta = build_measurement_metadata("K001", "N15_M07_F10", "vibration", 0)
        self.assertEqual(set(vib_meta["modality"]), {"vibration"})
        self.assertEqual(set(vib_meta["channel_name"]), {"vibration_1"})

        current_meta_0 = build_measurement_metadata("K001", "N15_M07_F10", "motor_current", 0)
        current_meta_1 = build_measurement_metadata("K001", "N15_M07_F10", "motor_current", 1)
        self.assertEqual(set(current_meta_0["channel_name"]), {"phase_current_1"})
        self.assertEqual(set(current_meta_1["channel_name"]), {"phase_current_2"})
        self.assertEqual(MODALITY_CHANNELS["vibration"], ("vibration_1",))
        self.assertEqual(MODALITY_CHANNELS["motor_current"], ("phase_current_1", "phase_current_2"))
        with self.assertRaises(ValueError):
            build_measurement_metadata("K001", "N15_M07_F10", "not_a_modality", 0)
        with self.assertRaises(ValueError):
            build_measurement_metadata("K001", "N15_M07_F10", "motor_current", 5)

    def test_09_window_generation_correctness(self):
        X, meta = load_stream_windows("K001", "N15_M07_F10", "vibration", 0, measurement_numbers=[1])
        path = os.path.join(RAW_DATA_DIR, "K001", "N15_M07_F10_K001_1.mat")
        raw_signal = read_paderborn_mat_file(path)["vibration_1"]
        expected_windows = window_channel_signal(raw_signal, window_size=WINDOW_SIZE, step_size=WINDOW_SIZE)
        self.assertEqual(X.shape, expected_windows.shape)
        self.assertTrue(np.array_equal(X, expected_windows))
        self.assertEqual(list(meta["window_index"]), list(range(X.shape[0])))
        self.assertTrue(np.isfinite(X).all())

    def test_10_no_measurement_crosses_split_boundary(self):
        _, meta = load_stream_windows("KA01", "N15_M07_F10", "vibration", 0, measurement_numbers=[1, 5, 7, 8, 11])
        self.assertTrue(verify_no_measurement_crosses_split(meta))
        corrupted = meta.copy()
        corrupted.loc[corrupted.index[0], "split"] = "val" if corrupted.loc[corrupted.index[0], "split"] != "val" else "test"
        with self.assertRaises(AssertionError):
            verify_no_measurement_crosses_split(corrupted)

    def test_11_no_duplicate_observations(self):
        _, meta_a = load_stream_windows("K001", "N15_M07_F10", "vibration", 0, measurement_numbers=[1, 2])
        _, meta_b = load_stream_windows("K002", "N15_M07_F10", "vibration", 0, measurement_numbers=[1, 2])
        combined = pd.concat([meta_a, meta_b], ignore_index=True)
        self.assertTrue(verify_observation_id_uniqueness(combined))
        duplicated = pd.concat([meta_a, meta_a], ignore_index=True)
        with self.assertRaises(AssertionError):
            verify_observation_id_uniqueness(duplicated)

    def test_12_normalization_uses_train_reference_data_only(self):
        train_numbers = [n for n in range(1, 21) if split_for_measurement(n) == "train"]
        X, meta = load_stream_windows("KA01", "N15_M07_F10", "vibration", 0, measurement_numbers=train_numbers[:3])
        mean, std = fit_train_normalization(X, meta)
        expected_mean = float(np.mean(X))
        expected_std = float(np.std(X))
        self.assertAlmostEqual(mean, expected_mean, places=6)
        self.assertAlmostEqual(std, expected_std, places=6)

        test_numbers = [n for n in range(1, 21) if split_for_measurement(n) == "test"]
        _, test_only_meta = load_stream_windows("KA01", "N15_M07_F10", "vibration", 0, measurement_numbers=test_numbers[:1])
        with self.assertRaises(ValueError):
            fit_train_normalization(np.zeros((len(test_only_meta), WINDOW_SIZE, 1), dtype=np.float32), test_only_meta)

        normalized = apply_normalization(X, mean, std)
        self.assertEqual(normalized.shape, X.shape)
        self.assertAlmostEqual(float(np.mean(normalized)), 0.0, places=4)

    def test_13_test_data_cannot_influence_normalization(self):
        train_numbers = [n for n in range(1, 21) if split_for_measurement(n) == "train"]
        test_numbers = [n for n in range(1, 21) if split_for_measurement(n) == "test"]

        X_train_only, meta_train_only = load_stream_windows(
            "KA01", "N15_M07_F10", "vibration", 0, measurement_numbers=train_numbers[:2]
        )
        mean_train_only, std_train_only = fit_train_normalization(X_train_only, meta_train_only)

        X_with_test, meta_with_test = load_stream_windows(
            "KA01", "N15_M07_F10", "vibration", 0, measurement_numbers=train_numbers[:2] + test_numbers[:2]
        )
        mean_with_test, std_with_test = fit_train_normalization(X_with_test, meta_with_test)

        self.assertAlmostEqual(mean_train_only, mean_with_test, places=6)
        self.assertAlmostEqual(std_train_only, std_with_test, places=6)

    def test_14_repeat_preprocessing_produces_equivalent_representation(self):
        X_a, meta_a = load_stream_windows("KB23", "N09_M07_F10", "vibration", 0, measurement_numbers=[1, 2, 3])
        X_b, meta_b = load_stream_windows("KB23", "N09_M07_F10", "vibration", 0, measurement_numbers=[1, 2, 3])
        self.assertTrue(np.array_equal(X_a, X_b))
        self.assertTrue(meta_a.equals(meta_b))

    def test_15_split_disjoint_across_full_condition(self):
        _, meta = load_stream_windows("K003", "N15_M07_F10", "vibration", 0)
        overlaps = verify_split_disjoint(meta)
        self.assertEqual(overlaps["train_val_overlap"], 0)
        self.assertEqual(overlaps["train_test_overlap"], 0)
        self.assertEqual(overlaps["val_test_overlap"], 0)
        self.assertTrue(verify_no_measurement_crosses_split(meta))
        self.assertEqual(set(meta["split"]), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
