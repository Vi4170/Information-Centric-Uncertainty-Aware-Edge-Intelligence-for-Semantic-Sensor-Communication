import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import preprocessing as pp

_RAW_DATA_AVAILABLE = schema.raw_data_available()

# Smallest real recordings (60s) -- used wherever a full-file read is needed,
# to keep the suite fast.
_SMALL_CONDITION = "0Nm_BPFI_03"
_SMALL_CONDITION_2 = "0Nm_BPFO_10"


class TestNRealWindowsSynthetic(unittest.TestCase):
    """No raw data required -- pure boundary-condition checks."""

    def test_01_exact_multiple_gives_expected_count(self):
        self.assertEqual(pp._n_real_windows(4096, 2048), 2)

    def test_02_remainder_is_dropped_not_padded(self):
        self.assertEqual(pp._n_real_windows(4097, 2048), 2)
        self.assertEqual(pp._n_real_windows(6143, 2048), 2)

    def test_03_too_short_returns_zero(self):
        self.assertEqual(pp._n_real_windows(100, 2048), 0)

    def test_04_matches_window_channel_signal_count(self):
        from src.ims_pipeline.preprocessing import window_channel_signal

        signal = np.zeros(10000, dtype=np.float32)
        windows = window_channel_signal(signal, window_size=2048, step_size=2048)
        self.assertEqual(windows.shape[0], pp._n_real_windows(10000, 2048))


class TestReaderErrorHandling(unittest.TestCase):
    def test_05_missing_vibration_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            pp._read_vibration_mat("data/raw/multimodal/vibration/does_not_exist.mat")

    def test_06_missing_tdms_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            pp._read_temperature_current_tdms("data/raw/multimodal/temperature_current/does_not_exist.tdms")

    def test_07_unknown_modality_raises(self):
        with self.assertRaises(ValueError):
            pp.fit_modality_normalization("acoustic")


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestRawFileImmutability(unittest.TestCase):
    """Confirms reading a raw file never modifies it on disk."""

    def test_08_vibration_file_bytes_unchanged_after_read(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        path = row["vibration_path"]
        before = os.path.getmtime(path), os.path.getsize(path)
        pp._read_vibration_mat(path)
        after = os.path.getmtime(path), os.path.getsize(path)
        self.assertEqual(before, after)

    def test_09_tdms_file_bytes_unchanged_after_read(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        path = row["temperature_current_path"]
        before = os.path.getmtime(path), os.path.getsize(path)
        pp._read_temperature_current_tdms(path)
        after = os.path.getmtime(path), os.path.getsize(path)
        self.assertEqual(before, after)


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestModalitySpecificReaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = pp.get_condition_row(_SMALL_CONDITION)

    def test_10_vibration_channels_present_and_correct_dtype(self):
        channels = pp._read_vibration_mat(self.row["vibration_path"])
        self.assertEqual(set(channels.keys()), set(schema.VIBRATION_CHANNEL_ORDER))
        for arr in channels.values():
            self.assertEqual(arr.dtype, np.float32)
            self.assertEqual(arr.ndim, 1)

    def test_11_temperature_current_channels_present(self):
        channels = pp._read_temperature_current_tdms(self.row["temperature_current_path"])
        self.assertEqual(
            set(channels.keys()),
            set(pp.TEMPERATURE_CHANNEL_NAMES) | set(pp.MOTOR_CURRENT_CHANNEL_NAMES),
        )

    def test_12_temperature_and_current_channels_are_same_length(self):
        channels = pp._read_temperature_current_tdms(self.row["temperature_current_path"])
        lengths = {len(arr) for arr in channels.values()}
        self.assertEqual(len(lengths), 1, "all 5 TDMS channels should share one sample grid")

    def test_13_measured_tc_rate_is_close_to_nominal_but_not_identical_to_vibration(self):
        rate = pp._measured_temperature_current_rate_hz(self.row["temperature_current_path"])
        self.assertAlmostEqual(rate, 25608.2, delta=5.0)
        self.assertNotEqual(rate, schema.VIBRATION_MEASURED_SAMPLING_RATE_HZ)


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestVibrationPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = pp.get_condition_row(_SMALL_CONDITION)
        cls.X, cls.meta = pp.build_vibration_windows_for_condition(cls.row)

    def test_14_shape_is_window_size_by_four_channels(self):
        self.assertEqual(self.X.shape[1], schema.WINDOW_SIZE_SAMPLES)
        self.assertEqual(self.X.shape[2], 4)
        self.assertEqual(self.X.shape[0], len(self.meta))

    def test_15_metadata_has_required_traceability_columns(self):
        required = {
            "observation_id", "source_session_id", "condition_code", "modality",
            "split", "source_recording_id", "sampling_rate_hz", "window_size_samples",
        }
        self.assertTrue(required.issubset(set(self.meta.columns)))

    def test_16_modality_column_is_vibration(self):
        self.assertTrue((self.meta["modality"] == "vibration").all())

    def test_17_channel_order_matches_schema(self):
        self.assertEqual(self.meta.iloc[0]["channel_names"], list(schema.VIBRATION_CHANNEL_ORDER))

    def test_18_sampling_rate_is_exact_measured_value(self):
        self.assertTrue((self.meta["sampling_rate_hz"] == 25600.0).all())

    def test_19_unnormalized_values_are_not_centered(self):
        # Raw acceleration in g should not already have zero mean/unit std.
        self.assertGreater(np.abs(self.X.mean()), 1e-6)


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestMotorCurrentPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = pp.get_condition_row(_SMALL_CONDITION)
        cls.X, cls.meta = pp.build_motor_current_windows_for_condition(cls.row)

    def test_20_shape_has_three_phase_channels(self):
        self.assertEqual(self.X.shape[2], 3)
        self.assertEqual(self.X.shape[0], len(self.meta))

    def test_21_not_concatenated_with_vibration_channel_dim(self):
        self.assertNotEqual(self.X.shape[2], 4)

    def test_22_phase_channel_names_preserved(self):
        self.assertEqual(self.meta.iloc[0]["channel_names"], list(pp.MOTOR_CURRENT_CHANNEL_NAMES))

    def test_23_own_measured_sampling_rate_used_not_vibrations(self):
        rate = self.meta.iloc[0]["sampling_rate_hz"]
        self.assertNotEqual(rate, schema.VIBRATION_MEASURED_SAMPLING_RATE_HZ)
        self.assertAlmostEqual(rate, 25608.2, delta=5.0)

    def test_24_window_size_differs_from_vibration_but_same_physical_duration(self):
        self.assertNotEqual(self.meta.iloc[0]["window_size_samples"], schema.WINDOW_SIZE_SAMPLES)
        duration = self.meta.iloc[0]["window_size_samples"] / self.meta.iloc[0]["sampling_rate_hz"]
        self.assertAlmostEqual(duration, schema.WINDOW_DURATION_SECONDS, places=3)

    def test_25_source_recording_is_the_tdms_file_not_the_mat_file(self):
        self.assertTrue(self.meta.iloc[0]["source_recording_id"].endswith(".tdms"))


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestTemperaturePreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = pp.get_condition_row(_SMALL_CONDITION)
        cls.X, cls.meta = pp.build_temperature_features_for_condition(cls.row)

    def test_26_output_is_compact_not_a_raw_high_rate_trace(self):
        # One value per channel per window, NOT (n_windows, window_size, 2).
        self.assertEqual(self.X.ndim, 2)
        self.assertEqual(self.X.shape[1], len(pp.TEMPERATURE_CHANNEL_NAMES))

    def test_27_summary_statistic_is_documented_as_mean(self):
        self.assertTrue((self.meta["summary_statistic"] == "mean").all())

    def test_28_physical_time_interval_matches_common_window_duration(self):
        self.assertTrue((self.meta["physical_time_interval_seconds"] == schema.WINDOW_DURATION_SECONDS).all())

    def test_29_value_matches_manual_mean_of_real_samples(self):
        channels = pp._read_temperature_current_tdms(self.row["temperature_current_path"])
        window_size = self.meta.iloc[0]["n_raw_samples_averaged"]
        expected_first_window_mean = channels[pp.TEMPERATURE_CHANNEL_NAMES[0]][:window_size].mean()
        self.assertAlmostEqual(float(self.X[0, 0]), float(expected_first_window_mean), places=3)

    def test_30_no_interpolation_only_real_measured_samples_used(self):
        # n_raw_samples_averaged must be a real, positive count of actual samples.
        self.assertTrue((self.meta["n_raw_samples_averaged"] > 0).all())


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestMissingModalityHandling(unittest.TestCase):
    """The empirically-confirmed case: vibration can have one more valid
    window than temperature/motor_current for the same condition."""

    @classmethod
    def setUpClass(cls):
        cls.row = pp.get_condition_row("0Nm_Normal")
        cls.availability = pp.condition_window_availability(cls.row)

    def test_31_vibration_has_more_or_equal_windows_than_temperature_current(self):
        self.assertGreaterEqual(self.availability["n_vibration"], self.availability["n_temperature_current"])

    def test_32_vibration_only_indices_are_never_fabricated_for_other_modalities(self):
        vib_x, vib_meta = pp.build_vibration_windows_for_condition(self.row)
        tc_x, tc_meta = pp.build_motor_current_windows_for_condition(self.row)
        vib_ids = set(vib_meta["observation_id"])
        tc_ids = set(tc_meta["observation_id"])
        # tc must never claim an observation vibration doesn't also have windows for
        self.assertTrue(tc_ids.issubset(vib_ids))
        # and vibration may legitimately have some ids tc does not
        missing_for_tc = vib_ids - tc_ids
        self.assertEqual(len(missing_for_tc), len(self.availability["vibration_only_window_indices"]))

    def test_33_no_padded_or_zero_filled_extra_window_in_temperature_current(self):
        tc_x, tc_meta = pp.build_motor_current_windows_for_condition(self.row)
        self.assertEqual(tc_x.shape[0], self.availability["n_temperature_current"])


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestModalitySpecificCorruptionHandling(unittest.TestCase):
    """Task 35 corrective audit: direct inspection of all 45 raw
    temperature_current .tdms files confirmed all 9 BPFO conditions have 2
    of 3 current-phase channels empty, while their 2 temperature channels
    are fully present, correctly lengthed, and physically plausible in
    every one of those same 9 files. Corruption in one modality's own
    channel group must never be inferred to also invalidate the other
    modality sharing the same raw file. See
    docs/multimodal_task35_bpfo_temperature_audit.md."""

    @classmethod
    def setUpClass(cls):
        cls.row = pp.get_condition_row(_SMALL_CONDITION_2)  # 0Nm_BPFO_10

    def test_41_temperature_only_read_succeeds_for_bpfo(self):
        channels = pp._read_temperature_current_tdms(
            self.row["temperature_current_path"], require_temperature=True, require_current=False
        )
        for name in pp.TEMPERATURE_CHANNEL_NAMES:
            self.assertGreater(len(channels[name]), 0)

    def test_42_current_only_read_raises_for_bpfo(self):
        with self.assertRaises(pp.CorruptRawFileError):
            pp._read_temperature_current_tdms(
                self.row["temperature_current_path"], require_temperature=False, require_current=True
            )

    def test_43_default_read_still_raises_when_both_required(self):
        with self.assertRaises(pp.CorruptRawFileError):
            pp._read_temperature_current_tdms(self.row["temperature_current_path"])

    def test_44_build_temperature_features_succeeds_for_bpfo(self):
        X, meta = pp.build_temperature_features_for_condition(self.row)
        self.assertGreater(len(X), 0)
        self.assertEqual(X.shape[1], len(pp.TEMPERATURE_CHANNEL_NAMES))

    def test_45_build_motor_current_windows_still_raises_for_bpfo(self):
        with self.assertRaises(pp.CorruptRawFileError):
            pp.build_motor_current_windows_for_condition(self.row)

    def test_46_condition_window_availability_succeeds_despite_corrupt_current(self):
        availability = pp.condition_window_availability(self.row)
        self.assertGreater(availability["n_temperature_current"], 0)

    def test_47_fit_modality_normalization_motor_current_excludes_only_bpfo_train_conditions(self):
        result = pp.fit_modality_normalization("motor_current")
        excluded_codes = {e["condition_code"] for e in result["excluded_conditions"]}
        self.assertTrue(len(excluded_codes) > 0)
        self.assertTrue(all("BPFO" in code for code in excluded_codes))

    def test_48_fit_modality_normalization_temperature_excludes_nothing_for_bpfo(self):
        result = pp.fit_modality_normalization("temperature")
        self.assertEqual(result["excluded_conditions"], [])

    def test_49_fit_modality_normalization_vibration_excludes_nothing(self):
        result = pp.fit_modality_normalization("vibration")
        self.assertEqual(result["excluded_conditions"], [])


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestObservationIdConsistency(unittest.TestCase):
    def test_34_shared_window_indices_have_identical_ids_across_modalities(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        _, vib_meta = pp.build_vibration_windows_for_condition(row)
        _, cur_meta = pp.build_motor_current_windows_for_condition(row)
        _, temp_meta = pp.build_temperature_features_for_condition(row)
        common_index = min(len(vib_meta), len(cur_meta), len(temp_meta)) - 1
        self.assertEqual(
            vib_meta.iloc[common_index]["observation_id"],
            cur_meta.iloc[common_index]["observation_id"],
        )
        self.assertEqual(
            cur_meta.iloc[common_index]["observation_id"],
            temp_meta.iloc[common_index]["observation_id"],
        )

    def test_35_ids_match_schema_module_deterministic_formula(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        _, vib_meta = pp.build_vibration_windows_for_condition(row)
        expected = schema._observation_id(row["condition_code"], 0)
        self.assertEqual(vib_meta.iloc[0]["observation_id"], expected)


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestDeterministicProcessing(unittest.TestCase):
    def test_36_vibration_windows_identical_across_two_calls(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        X1, _ = pp.build_vibration_windows_for_condition(row)
        X2, _ = pp.build_vibration_windows_for_condition(row)
        np.testing.assert_array_equal(X1, X2)

    def test_37_temperature_features_identical_across_two_calls(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        X1, _ = pp.build_temperature_features_for_condition(row)
        X2, _ = pp.build_temperature_features_for_condition(row)
        np.testing.assert_array_equal(X1, X2)

    def test_38_condition_registry_split_assignment_is_deterministic(self):
        r1 = pp.get_condition_registry()
        r2 = pp.get_condition_registry()
        pd.testing.assert_series_equal(
            r1.set_index("condition_code")["split"],
            r2.set_index("condition_code")["split"],
        )


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestSplitIntegrity(unittest.TestCase):
    def test_39_every_condition_has_exactly_one_split(self):
        registry = pp.get_condition_registry()
        self.assertFalse(registry["split"].isna().any())
        self.assertTrue(registry["split"].isin(schema.SPLIT_NAMES).all())

    def test_40_split_counts_match_task32_ratios_roughly(self):
        registry = pp.get_condition_registry()
        counts = registry["split"].value_counts()
        self.assertGreater(counts.get("train", 0), counts.get("val", 0))
        self.assertGreater(counts.get("train", 0), counts.get("test", 0))

    def test_41_a_given_condition_windows_are_all_from_the_same_split(self):
        row = pp.get_condition_row(_SMALL_CONDITION)
        _, meta = pp.build_vibration_windows_for_condition(row)
        self.assertEqual(meta["split"].nunique(), 1)


class TestNormalizationTrainOnlyLeakage(unittest.TestCase):
    """Uses mocked condition registries pointing at real small files, so
    exactly which files get included/excluded can be verified precisely."""

    @unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
    def test_42_only_train_split_files_are_read_for_fitting(self):
        real_registry = pp.get_condition_registry()
        train_row = real_registry[real_registry["condition_code"] == _SMALL_CONDITION].iloc[0].copy()
        other_row = real_registry[real_registry["condition_code"] == _SMALL_CONDITION_2].iloc[0].copy()
        train_row["split"] = "train"
        other_row["split"] = "test"
        fake_registry = pd.DataFrame([train_row, other_row])

        with patch.object(pp, "get_condition_registry", return_value=fake_registry):
            result = pp.fit_modality_normalization("vibration")

        self.assertEqual(result["n_train_conditions"], 1)

    @unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
    def test_43_result_changes_if_the_excluded_file_actually_differs(self):
        real_registry = pp.get_condition_registry()
        row_a = real_registry[real_registry["condition_code"] == _SMALL_CONDITION].iloc[0].copy()
        row_b = real_registry[real_registry["condition_code"] == _SMALL_CONDITION_2].iloc[0].copy()

        registry_a_only = pd.DataFrame([{**row_a.to_dict(), "split": "train"}])
        registry_b_only = pd.DataFrame([{**row_b.to_dict(), "split": "train"}])

        with patch.object(pp, "get_condition_registry", return_value=registry_a_only):
            result_a = pp.fit_modality_normalization("vibration")
        with patch.object(pp, "get_condition_registry", return_value=registry_b_only):
            result_b = pp.fit_modality_normalization("vibration")

        # Different physical recordings (bearing fault + severity differ) should not
        # coincidentally produce bit-identical statistics.
        self.assertNotEqual(result_a["mean"], result_b["mean"])

    def test_44_fit_raises_when_no_train_conditions_present(self):
        empty_registry = pd.DataFrame(
            columns=["condition_code", "split", "vibration_path", "temperature_current_path"]
        )
        with patch.object(pp, "get_condition_registry", return_value=empty_registry):
            with self.assertRaises(ValueError):
                pp.fit_modality_normalization("vibration")


@unittest.skipUnless(
    os.path.exists(pp.NORMALIZATION_PARAMS_JSON_PATH), "normalization params have not been generated yet"
)
class TestSavedNormalizationArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = pp.load_normalization_params()

    def test_45_all_three_modalities_present(self):
        self.assertEqual(set(self.params.keys()), set(schema.CANONICAL_MODALITIES))

    def test_46_each_entry_has_mean_std_and_was_fit_on_train_only(self):
        for modality, entry in self.params.items():
            self.assertIn("mean", entry)
            self.assertIn("std", entry)
            self.assertGreater(entry["std"], 0)
            self.assertEqual(entry["fit_only_on_split"], "train")

    def test_47_std_values_are_finite_and_positive(self):
        for entry in self.params.values():
            self.assertTrue(np.isfinite(entry["mean"]))
            self.assertTrue(np.isfinite(entry["std"]))
            self.assertGreater(entry["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
