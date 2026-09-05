import os
import unittest

import numpy as np
import pandas as pd

from src.ims_pipeline.preprocessing import verify_observation_id_uniqueness
from src.xjtu_pipeline.preprocessing import (
    BEARING_SPLIT_ASSIGNMENT,
    BEARING_TYPE,
    CHANNEL_NAMES,
    MODALITY_CHANNELS,
    OPERATING_CONDITIONS,
    RAW_DATA_DIR,
    SAMPLES_PER_FILE,
    SAMPLING_RATE_HZ,
    WINDOW_SIZE,
    WINDOWS_PER_FILE,
    XJTU_BEARING_REGISTRY,
    apply_normalization,
    build_bearing_metadata,
    discover_bearing_files,
    fit_train_normalization,
    list_bearing_ids,
    load_stream_windows,
    read_xjtu_csv,
    split_for_bearing,
    verify_no_bearing_crosses_split,
    verify_split_disjoint,
    window_channel_signal,
)


# Check if at least one bearing from each condition is available
_RAW_DATA_AVAILABLE = all(
    os.path.isdir(os.path.join(RAW_DATA_DIR, cond, f"Bearing{i}_1"))
    for i, cond in enumerate(OPERATING_CONDITIONS, 1)
)


class TestXjtuConstants(unittest.TestCase):
    """Tests that do NOT require the raw data to be present."""

    def test_01_bearing_registry_has_15_entries(self):
        self.assertEqual(len(XJTU_BEARING_REGISTRY), 15)

    def test_02_bearing_ids_per_condition(self):
        for condition in OPERATING_CONDITIONS:
            bearings = list_bearing_ids(condition)
            self.assertEqual(len(bearings), 5, f"Condition {condition} should have 5 bearings")

    def test_03_operating_conditions_count(self):
        self.assertEqual(len(OPERATING_CONDITIONS), 3)

    def test_04_channel_names(self):
        self.assertEqual(len(CHANNEL_NAMES), 2)
        self.assertEqual(CHANNEL_NAMES[0], "Horizontal_vibration_signals")
        self.assertEqual(CHANNEL_NAMES[1], "Vertical_vibration_signals")

    def test_05_sampling_rate_confirmed(self):
        self.assertEqual(SAMPLING_RATE_HZ, 25600)

    def test_06_samples_per_file(self):
        self.assertEqual(SAMPLES_PER_FILE, 32768)

    def test_07_windows_per_file(self):
        self.assertEqual(WINDOWS_PER_FILE, 16)
        self.assertEqual(SAMPLES_PER_FILE % WINDOW_SIZE, 0)

    def test_08_split_assignment_is_deterministic(self):
        from src.xjtu_pipeline.preprocessing import _compute_bearing_split_assignment
        a = _compute_bearing_split_assignment()
        b = _compute_bearing_split_assignment()
        self.assertEqual(a, b)

    def test_09_split_assignment_has_correct_counts(self):
        for condition in OPERATING_CONDITIONS:
            cond_num = OPERATING_CONDITIONS.index(condition) + 1
            bearings = [f"Bearing{cond_num}_{i}" for i in range(1, 6)]
            splits = [BEARING_SPLIT_ASSIGNMENT[b] for b in bearings]
            self.assertEqual(splits.count("train"), 3, f"{condition} should have 3 train bearings")
            self.assertEqual(splits.count("val"), 1, f"{condition} should have 1 val bearing")
            self.assertEqual(splits.count("test"), 1, f"{condition} should have 1 test bearing")

    def test_10_all_bearings_assigned(self):
        self.assertEqual(set(BEARING_SPLIT_ASSIGNMENT.keys()), set(XJTU_BEARING_REGISTRY.keys()))

    def test_11_total_split_counts(self):
        all_splits = list(BEARING_SPLIT_ASSIGNMENT.values())
        self.assertEqual(all_splits.count("train"), 9)
        self.assertEqual(all_splits.count("val"), 3)
        self.assertEqual(all_splits.count("test"), 3)

    def test_12_split_for_bearing_rejects_unknown(self):
        with self.assertRaises(ValueError):
            split_for_bearing("BearingX_99")

    def test_13_list_bearing_ids_rejects_unknown_condition(self):
        with self.assertRaises(ValueError):
            list_bearing_ids("NOT_A_CONDITION")

    def test_14_fault_elements_present_for_all_bearings(self):
        for bearing_id, reg in XJTU_BEARING_REGISTRY.items():
            self.assertIn("fault_element", reg, f"Missing fault_element for {bearing_id}")
            self.assertIsInstance(reg["fault_element"], str)
            self.assertGreater(len(reg["fault_element"]), 0)

    def test_15_bearing_type_is_correct(self):
        self.assertEqual(BEARING_TYPE, "LDK UER204")

    def test_16_modality_channels(self):
        self.assertEqual(MODALITY_CHANNELS["vibration"], CHANNEL_NAMES)
        self.assertEqual(len(MODALITY_CHANNELS), 1)

    def test_17_file_counts_match_registry(self):
        """Verify that the registry's documented file counts match the author PDF."""
        expected = {
            "Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158,
            "Bearing1_4": 122, "Bearing1_5": 52,
            "Bearing2_1": 491, "Bearing2_2": 161, "Bearing2_3": 533,
            "Bearing2_4": 42, "Bearing2_5": 339,
            "Bearing3_1": 2538, "Bearing3_2": 2496, "Bearing3_3": 371,
            "Bearing3_4": 1515, "Bearing3_5": 114,
        }
        for bearing_id, n in expected.items():
            self.assertEqual(XJTU_BEARING_REGISTRY[bearing_id]["n_files"], n, f"{bearing_id}")


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "Extracted XJTU-SY raw data not present under data/raw/xjtu/")
class TestXjtuPreprocessing(unittest.TestCase):
    """Tests that require extracted raw CSV data."""

    def test_01_dataset_discovery_and_deterministic_ordering(self):
        for bearing_id in ("Bearing1_1", "Bearing2_1", "Bearing3_1"):
            files_a = discover_bearing_files(bearing_id)
            files_b = discover_bearing_files(bearing_id)
            self.assertEqual(files_a, files_b)
            numbers = [int(f.replace(".csv", "")) for f in files_a]
            self.assertEqual(numbers, sorted(numbers))
            self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

    def test_02_file_counts_match_registry_on_disk(self):
        for bearing_id in XJTU_BEARING_REGISTRY:
            files = discover_bearing_files(bearing_id)
            expected = XJTU_BEARING_REGISTRY[bearing_id]["n_files"]
            self.assertEqual(len(files), expected, f"{bearing_id}: expected {expected} files, found {len(files)}")

    def test_03_csv_file_shape_and_header(self):
        path = os.path.join(RAW_DATA_DIR, "35Hz12kN", "Bearing1_1", "1.csv")
        data = read_xjtu_csv(path)
        self.assertEqual(data.shape, (SAMPLES_PER_FILE, len(CHANNEL_NAMES)))
        self.assertEqual(data.dtype, np.float32)
        self.assertTrue(np.isfinite(data).all())

    def test_04_csv_across_conditions(self):
        """Verify CSV format is consistent across all operating conditions."""
        for bearing_id in ("Bearing1_1", "Bearing2_1", "Bearing3_1"):
            files = discover_bearing_files(bearing_id)
            path = os.path.join(RAW_DATA_DIR,
                                XJTU_BEARING_REGISTRY[bearing_id]["operating_condition"],
                                bearing_id, files[0])
            data = read_xjtu_csv(path)
            self.assertEqual(data.shape, (SAMPLES_PER_FILE, 2))

    def test_05_deterministic_observation_ids(self):
        meta_a = build_bearing_metadata("Bearing1_1", 0)
        meta_b = build_bearing_metadata("Bearing1_1", 0)
        self.assertTrue(meta_a.equals(meta_b))

    def test_06_metadata_correctness(self):
        meta = build_bearing_metadata("Bearing1_1", 0)
        n_files = len(discover_bearing_files("Bearing1_1"))
        expected_rows = n_files * WINDOWS_PER_FILE
        self.assertEqual(len(meta), expected_rows)
        self.assertEqual(set(meta["bearing_id"]), {"Bearing1_1"})
        self.assertEqual(set(meta["channel_name"]), {"Horizontal_vibration_signals"})
        self.assertEqual(set(meta["operating_condition"]), {"35Hz12kN"})
        self.assertEqual(set(meta["fault_element"]), {"Outer race"})
        self.assertEqual(set(meta["sampling_rate_hz"]), {SAMPLING_RATE_HZ})
        # All windows from one bearing have the same split
        self.assertEqual(meta["split"].nunique(), 1)

    def test_07_channel_separation(self):
        """Channels 0 and 1 produce different observation IDs and can be loaded separately."""
        meta_h = build_bearing_metadata("Bearing1_1", 0)
        meta_v = build_bearing_metadata("Bearing1_1", 1)
        self.assertEqual(set(meta_h["channel_name"]), {"Horizontal_vibration_signals"})
        self.assertEqual(set(meta_v["channel_name"]), {"Vertical_vibration_signals"})
        # No overlapping observation IDs
        combined = pd.concat([meta_h, meta_v], ignore_index=True)
        self.assertTrue(combined["observation_id"].is_unique)

    def test_08_bearing_metadata_preserved(self):
        for bearing_id in ("Bearing1_4", "Bearing2_3", "Bearing3_2"):
            meta = build_bearing_metadata(bearing_id, 0)
            expected_fault = XJTU_BEARING_REGISTRY[bearing_id]["fault_element"]
            self.assertEqual(set(meta["fault_element"]), {expected_fault})
            self.assertEqual(set(meta["bearing_type"]), {BEARING_TYPE})

    def test_09_window_generation_correctness(self):
        X, meta = load_stream_windows("Bearing2_4", 0, file_indices=[0])
        self.assertEqual(X.shape, (WINDOWS_PER_FILE, WINDOW_SIZE, 1))
        self.assertEqual(len(meta), WINDOWS_PER_FILE)
        self.assertEqual(list(meta["window_index"]), list(range(WINDOWS_PER_FILE)))
        self.assertTrue(np.isfinite(X).all())

    def test_10_window_data_matches_raw_csv(self):
        """Verify window content matches the raw CSV data."""
        path = os.path.join(RAW_DATA_DIR, "35Hz12kN", "Bearing1_1", "1.csv")
        raw = read_xjtu_csv(path)
        X, _ = load_stream_windows("Bearing1_1", 0, file_indices=[0])

        # Compare first window to first 2048 samples of horizontal channel
        expected_first_window = raw[:WINDOW_SIZE, 0].reshape(WINDOW_SIZE, 1)
        np.testing.assert_array_equal(X[0], expected_first_window)

        # Compare second window
        expected_second_window = raw[WINDOW_SIZE:2*WINDOW_SIZE, 0].reshape(WINDOW_SIZE, 1)
        np.testing.assert_array_equal(X[1], expected_second_window)

    def test_11_no_bearing_crosses_split_boundary(self):
        # Load metadata for multiple bearings from same condition
        frames = []
        for bearing_id in list_bearing_ids("35Hz12kN"):
            meta = build_bearing_metadata(bearing_id, 0)
            frames.append(meta)
        combined = pd.concat(frames, ignore_index=True)
        self.assertTrue(verify_no_bearing_crosses_split(combined))

    def test_12_split_disjoint(self):
        frames = []
        for bearing_id in list_bearing_ids("35Hz12kN"):
            meta = build_bearing_metadata(bearing_id, 0)
            frames.append(meta)
        combined = pd.concat(frames, ignore_index=True)
        overlaps = verify_split_disjoint(combined)
        self.assertEqual(overlaps["train_val_overlap"], 0)
        self.assertEqual(overlaps["train_test_overlap"], 0)
        self.assertEqual(overlaps["val_test_overlap"], 0)
        self.assertEqual(set(combined["split"]), {"train", "val", "test"})

    def test_13_no_duplicate_observations_across_bearings(self):
        frames = []
        for bearing_id in ("Bearing1_1", "Bearing2_1", "Bearing3_1"):
            for ch in range(len(CHANNEL_NAMES)):
                meta = build_bearing_metadata(bearing_id, ch)
                frames.append(meta)
        combined = pd.concat(frames, ignore_index=True)
        self.assertTrue(verify_observation_id_uniqueness(combined))

    def test_14_normalization_uses_train_data_only(self):
        train_bearings = [b for b, s in BEARING_SPLIT_ASSIGNMENT.items() if s == "train"]
        test_bearings = [b for b, s in BEARING_SPLIT_ASSIGNMENT.items() if s == "test"]

        # Load a few files from a train bearing
        X_train, meta_train = load_stream_windows(train_bearings[0], 0, file_indices=[0, 1])
        mean, std = fit_train_normalization(X_train, meta_train)
        expected_mean = float(np.mean(X_train))
        expected_std = float(np.std(X_train))
        self.assertAlmostEqual(mean, expected_mean, places=6)
        self.assertAlmostEqual(std, expected_std, places=6)

        # Test-only data should fail
        X_test, meta_test = load_stream_windows(test_bearings[0], 0, file_indices=[0])
        with self.assertRaises(ValueError):
            fit_train_normalization(X_test, meta_test)

    def test_15_test_data_cannot_influence_normalization(self):
        train_bearings = [b for b, s in BEARING_SPLIT_ASSIGNMENT.items() if s == "train"]
        test_bearings = [b for b, s in BEARING_SPLIT_ASSIGNMENT.items() if s == "test"]

        X_train, meta_train = load_stream_windows(train_bearings[0], 0, file_indices=[0])
        mean_train_only, std_train_only = fit_train_normalization(X_train, meta_train)

        # Add test data and recompute — stats should be identical
        X_test, meta_test = load_stream_windows(test_bearings[0], 0, file_indices=[0])
        X_combined = np.concatenate([X_train, X_test], axis=0)
        meta_combined = pd.concat([meta_train, meta_test], ignore_index=True)
        mean_with_test, std_with_test = fit_train_normalization(X_combined, meta_combined)

        self.assertAlmostEqual(mean_train_only, mean_with_test, places=6)
        self.assertAlmostEqual(std_train_only, std_with_test, places=6)

    def test_16_normalization_application(self):
        train_bearings = [b for b, s in BEARING_SPLIT_ASSIGNMENT.items() if s == "train"]
        X, meta = load_stream_windows(train_bearings[0], 0, file_indices=[0, 1])
        mean, std = fit_train_normalization(X, meta)
        normalized = apply_normalization(X, mean, std)
        self.assertEqual(normalized.shape, X.shape)
        self.assertAlmostEqual(float(np.mean(normalized)), 0.0, places=4)

    def test_17_repeat_preprocessing_deterministic(self):
        X_a, meta_a = load_stream_windows("Bearing1_1", 0, file_indices=[0, 1, 2])
        X_b, meta_b = load_stream_windows("Bearing1_1", 0, file_indices=[0, 1, 2])
        self.assertTrue(np.array_equal(X_a, X_b))
        self.assertTrue(meta_a.equals(meta_b))

    def test_18_vertical_channel_independent(self):
        X_h, meta_h = load_stream_windows("Bearing1_1", 0, file_indices=[0])
        X_v, meta_v = load_stream_windows("Bearing1_1", 1, file_indices=[0])
        # Shapes should match but data differs
        self.assertEqual(X_h.shape, X_v.shape)
        self.assertFalse(np.array_equal(X_h, X_v))
        # Observation IDs differ
        self.assertEqual(len(set(meta_h["observation_id"]) & set(meta_v["observation_id"])), 0)

    def test_19_operating_condition_metadata(self):
        for condition in OPERATING_CONDITIONS:
            bearings = list_bearing_ids(condition)
            meta = build_bearing_metadata(bearings[0], 0)
            self.assertEqual(set(meta["operating_condition"]), {condition})

    def test_20_file_indices_selection(self):
        """Verify that file_indices correctly restricts loaded files."""
        X_all, meta_all = load_stream_windows("Bearing2_4", 0)  # 42 files
        X_partial, meta_partial = load_stream_windows("Bearing2_4", 0, file_indices=[0, 1])
        self.assertEqual(X_partial.shape[0], 2 * WINDOWS_PER_FILE)
        self.assertEqual(X_all.shape[0], 42 * WINDOWS_PER_FILE)
        # Partial data should be subset of full data
        np.testing.assert_array_equal(X_all[:2*WINDOWS_PER_FILE], X_partial)


if __name__ == "__main__":
    unittest.main()
