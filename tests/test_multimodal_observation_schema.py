import inspect
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.multimodal_pipeline import observation_schema as obs
from src.cnn.config import INPUT_SHAPE

_RAW_DATA_AVAILABLE = obs.raw_data_available()


def _make_fake_raw_tree(root, conditions):
    """Build a minimal fake data/raw/multimodal/ tree (empty files -- this
    module never reads signal content, only checks for file existence)."""
    vib_dir = os.path.join(root, obs.VIBRATION_SUBDIR)
    tc_dir = os.path.join(root, obs.TEMPERATURE_CURRENT_SUBDIR)
    os.makedirs(vib_dir, exist_ok=True)
    os.makedirs(tc_dir, exist_ok=True)
    for cond, has_vib, has_tc in conditions:
        if has_vib:
            open(os.path.join(vib_dir, f"{cond}.mat"), "w").close()
        if has_tc:
            open(os.path.join(tc_dir, f"{cond}.tdms"), "w").close()
    return root


class TestNoProtectedModuleTouched(unittest.TestCase):
    def test_01_module_does_not_import_voi(self):
        source = inspect.getsource(obs)
        self.assertNotIn("src.voi", source)
        self.assertNotIn("import voi", source)

    def test_02_module_does_not_define_fusion_or_cnn_symbols(self):
        forbidden = ("fuse", "compute_voi", "build_cnn", "train_model", "VoIEngine")
        exported = dir(obs)
        for name in forbidden:
            self.assertFalse(
                any(name.lower() in symbol.lower() for symbol in exported),
                f"unexpected fusion/CNN/VoI symbol matching '{name}'",
            )


class TestConstantsAndWindowDefinition(unittest.TestCase):
    def test_03_canonical_modalities_excludes_acoustic(self):
        self.assertEqual(set(obs.CANONICAL_MODALITIES), {"vibration", "temperature", "motor_current"})
        self.assertNotIn("acoustic", obs.CANONICAL_MODALITIES)

    def test_04_window_size_reuses_project_cnn_convention(self):
        self.assertEqual(obs.WINDOW_SIZE_SAMPLES, INPUT_SHAPE[0])

    def test_05_window_duration_matches_nominal_rate(self):
        self.assertAlmostEqual(
            obs.WINDOW_DURATION_SECONDS,
            obs.WINDOW_SIZE_SAMPLES / obs.NOMINAL_SAMPLING_RATE_HZ,
        )

    def test_06_acoustic_window_size_preserves_common_physical_duration(self):
        acoustic_duration = obs.ACOUSTIC_WINDOW_SIZE_SAMPLES / obs.ACOUSTIC_SAMPLING_RATE_HZ
        self.assertAlmostEqual(acoustic_duration, obs.WINDOW_DURATION_SECONDS, places=4)

    def test_07_condition_duration_registry_reflects_measured_non_uniform_normal(self):
        self.assertEqual(obs.CONDITION_DURATION_SECONDS[("Normal", 0)], 300.0)
        self.assertEqual(obs.CONDITION_DURATION_SECONDS[("Normal", 2)], 120.0)
        self.assertEqual(obs.CONDITION_DURATION_SECONDS[("Normal", 4)], 120.0)
        self.assertEqual(obs.CONDITION_DURATION_SECONDS[("BPFI", 0)], 60.0)
        self.assertEqual(obs.CONDITION_DURATION_SECONDS[("Misalignment", 4)], 120.0)

    def test_08_vibration_channel_order_source_is_flagged_unverified(self):
        self.assertIn("not self-describing", obs.VIBRATION_CHANNEL_ORDER_SOURCE.lower())


class TestAlignmentStrategy(unittest.TestCase):
    def test_09_common_temporal_reference_is_elapsed_not_absolute(self):
        ref = obs.ALIGNMENT_STRATEGY["common_temporal_reference"].lower()
        self.assertIn("elapsed", ref)
        self.assertIn("not absolute wall-clock time", ref)

    def test_10_rejects_absolute_clock_alignment_with_evidence(self):
        reason = obs.ALIGNMENT_STRATEGY["why_not_absolute_clock"]
        self.assertIn("varies across sessions", reason)

    def test_11_rejects_shared_sample_index_with_evidence(self):
        reason = obs.ALIGNMENT_STRATEGY["why_not_shared_sample_index"]
        self.assertIn("differ slightly", reason)

    def test_12_acoustic_alignment_not_applicable(self):
        self.assertIn("excluded", obs.ALIGNMENT_STRATEGY["acoustic"].lower())


class TestConditionCodeAndFilenameHandling(unittest.TestCase):
    def test_13_condition_code_normal_has_no_severity_suffix(self):
        self.assertEqual(obs.condition_code(0, "Normal", ""), "0Nm_Normal")

    def test_14_condition_code_fault_includes_severity(self):
        self.assertEqual(obs.condition_code(2, "BPFI", "10"), "2Nm_BPFI_10")

    def test_15_misalignment_uses_correct_filename_token(self):
        self.assertEqual(obs.FAULT_TYPE_FILENAME_TOKEN["Misalignment"], "Misalign")

    def test_16_unbalance_typo_alternate_registered(self):
        self.assertIn("Unbalalnce", obs.VIBRATION_FAULT_TOKEN_ALTERNATES["Unbalance"])

    def test_17_parse_condition_filename_stem_normal(self):
        self.assertEqual(obs._parse_condition_filename_stem("0Nm_Normal"), (0, "Normal", ""))

    def test_18_parse_condition_filename_stem_fault(self):
        self.assertEqual(obs._parse_condition_filename_stem("4Nm_BPFO_30"), (4, "BPFO", "30"))

    def test_19_parse_condition_filename_stem_unparseable_returns_none(self):
        self.assertIsNone(obs._parse_condition_filename_stem("not_a_valid_name"))


class TestObservationIdDeterminism(unittest.TestCase):
    def test_20_observation_id_is_pure_function_of_inputs(self):
        a = obs._observation_id("0Nm_BPFI_03", 5)
        b = obs._observation_id("0Nm_BPFI_03", 5)
        self.assertEqual(a, b)

    def test_21_observation_id_encodes_condition_and_window(self):
        oid = obs._observation_id("2Nm_Unbalance_1751mg", 12)
        self.assertIn("2Nm_Unbalance_1751mg", oid)
        self.assertIn("w0012", oid)

    def test_22_observation_id_differs_by_window_index(self):
        a = obs._observation_id("0Nm_Normal", 0)
        b = obs._observation_id("0Nm_Normal", 1)
        self.assertNotEqual(a, b)


class TestSplitAssignmentDeterminism(unittest.TestCase):
    def test_23_split_assignment_is_deterministic(self):
        codes = [f"cond_{i}" for i in range(20)]
        a = obs._compute_condition_split_assignment(codes)
        b = obs._compute_condition_split_assignment(codes)
        self.assertEqual(a, b)

    def test_24_split_assignment_covers_every_condition_exactly_once(self):
        codes = [f"cond_{i}" for i in range(45)]
        assignment = obs._compute_condition_split_assignment(codes)
        self.assertEqual(set(assignment.keys()), set(codes))
        self.assertTrue(set(assignment.values()).issubset(set(obs.SPLIT_NAMES)))

    def test_25_split_assignment_roughly_matches_ratios(self):
        codes = [f"cond_{i}" for i in range(45)]
        assignment = obs._compute_condition_split_assignment(codes)
        counts = pd.Series(list(assignment.values())).value_counts()
        self.assertGreater(counts.get("train", 0), counts.get("val", 0))
        self.assertGreater(counts.get("train", 0), counts.get("test", 0))


class TestMissingModalityHandlingSynthetic(unittest.TestCase):
    """Uses a small synthetic raw-data tree so these tests never depend on
    (or risk touching) the real, large downloaded dataset."""

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix="multimodal_fake_raw_")

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_26_condition_missing_temperature_current_is_excluded(self):
        _make_fake_raw_tree(
            self.tmp_root,
            [("0Nm_Normal", True, False), ("0Nm_BPFI_03", True, True)],
        )
        df = obs.discover_conditions(self.tmp_root)
        row = df[df["condition_code"] == "0Nm_Normal"].iloc[0]
        self.assertFalse(row["meets_minimum_modality_requirement"])
        self.assertTrue(pd.isna(row["temperature_current_path"]))

    def test_27_condition_missing_vibration_is_excluded(self):
        _make_fake_raw_tree(
            self.tmp_root,
            [("0Nm_Normal", False, True), ("0Nm_BPFI_03", True, True)],
        )
        df = obs.discover_conditions(self.tmp_root)
        row = df[df["condition_code"] == "0Nm_Normal"].iloc[0]
        self.assertFalse(row["meets_minimum_modality_requirement"])
        self.assertTrue(pd.isna(row["vibration_path"]))

    def test_28_complete_condition_meets_requirement(self):
        _make_fake_raw_tree(self.tmp_root, [("0Nm_Normal", True, True)])
        df = obs.discover_conditions(self.tmp_root)
        row = df[df["condition_code"] == "0Nm_Normal"].iloc[0]
        self.assertTrue(row["meets_minimum_modality_requirement"])

    def test_29_incomplete_condition_produces_zero_observations(self):
        _make_fake_raw_tree(
            self.tmp_root,
            [("0Nm_Normal", True, False), ("0Nm_BPFI_03", True, True)],
        )
        index_df = obs.build_observation_index(self.tmp_root)
        self.assertEqual(len(index_df[index_df["condition_code"] == "0Nm_Normal"]), 0)
        self.assertGreater(len(index_df[index_df["condition_code"] == "0Nm_BPFI_03"]), 0)

    def test_30_raw_data_available_false_on_empty_directory(self):
        empty_root = tempfile.mkdtemp(prefix="multimodal_empty_")
        try:
            self.assertFalse(obs.raw_data_available(empty_root))
        finally:
            shutil.rmtree(empty_root, ignore_errors=True)

    def test_31_discover_conditions_raises_when_directories_entirely_absent(self):
        missing_root = os.path.join(self.tmp_root, "does_not_exist")
        with self.assertRaises(FileNotFoundError):
            obs.discover_conditions(missing_root)


class TestLeakagePreventionInjected(unittest.TestCase):
    """Deliberately corrupted DataFrames to prove the verification functions
    actually catch violations, without needing the real dataset."""

    def test_32_verify_no_condition_crosses_split_catches_violation(self):
        df = pd.DataFrame(
            {
                "condition_code": ["a", "a"],
                "split": ["train", "test"],
                "observation_id": ["a_w0", "a_w1"],
            }
        )
        with self.assertRaises(AssertionError):
            obs.verify_no_condition_crosses_split(df)

    def test_33_verify_split_disjoint_catches_duplicate_id_across_splits(self):
        df = pd.DataFrame(
            {
                "observation_id": ["x_w0", "x_w0"],
                "split": ["train", "test"],
            }
        )
        with self.assertRaises(AssertionError):
            obs.verify_split_disjoint(df)

    def test_34_verify_observation_id_uniqueness_catches_duplicates(self):
        df = pd.DataFrame({"observation_id": ["a_w0", "a_w0", "b_w0"]})
        with self.assertRaises(AssertionError):
            obs.verify_observation_id_uniqueness(df)

    def test_35_clean_dataframe_passes_all_three_checks(self):
        df = pd.DataFrame(
            {
                "observation_id": ["a_w0", "a_w1", "b_w0"],
                "condition_code": ["a", "a", "b"],
                "split": ["train", "train", "val"],
            }
        )
        self.assertTrue(obs.verify_observation_id_uniqueness(df))
        self.assertEqual(sum(obs.verify_split_disjoint(df).values()), 0)
        self.assertTrue(obs.verify_no_condition_crosses_split(df))


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "raw multimodal data not present in this environment")
class TestRealRawDataIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conditions = obs.discover_conditions()
        cls.index_df = obs.build_observation_index()

    def test_36_all_45_conditions_discovered(self):
        self.assertEqual(len(self.conditions), 45)

    def test_37_all_45_conditions_meet_minimum_modality_requirement(self):
        self.assertEqual(self.conditions["meets_minimum_modality_requirement"].sum(), 45)

    def test_38_five_acoustic_files_found_but_never_included(self):
        acoustic_present = self.conditions["acoustic_path"].notna().sum()
        self.assertEqual(acoustic_present, 5)
        self.assertFalse((self.index_df["acoustic_included"]).any())

    def test_39_typo_filename_preserved_for_2nm_unbalance(self):
        rows = self.conditions[(self.conditions["load_nm"] == 2) & (self.conditions["fault_type"] == "Unbalance")]
        self.assertEqual(len(rows), 5)
        self.assertTrue(rows["vibration_filename"].str.contains("Unbalalnce").all())

    def test_40_vibration_source_files_exist_on_disk(self):
        sample = self.conditions[self.conditions["meets_minimum_modality_requirement"]].head(10)
        for path in sample["vibration_path"]:
            self.assertTrue(os.path.isfile(path), path)

    def test_41_temperature_current_source_files_exist_on_disk(self):
        sample = self.conditions[self.conditions["meets_minimum_modality_requirement"]].head(10)
        for path in sample["temperature_current_path"]:
            self.assertTrue(os.path.isfile(path), path)

    def test_42_observation_ids_are_globally_unique(self):
        obs.verify_observation_id_uniqueness(self.index_df)

    def test_43_no_condition_crosses_split(self):
        obs.verify_no_condition_crosses_split(self.index_df)

    def test_44_splits_are_disjoint(self):
        overlaps = obs.verify_split_disjoint(self.index_df)
        self.assertEqual(sum(overlaps.values()), 0)

    def test_45_every_observation_has_both_required_source_files(self):
        self.assertTrue(self.index_df["vibration_source_file"].notna().all())
        self.assertTrue(self.index_df["temperature_current_source_file"].notna().all())

    def test_46_window_duration_constant_across_all_observations(self):
        self.assertTrue((self.index_df["window_duration_seconds"] == obs.WINDOW_DURATION_SECONDS).all())

    def test_47_vibration_and_temperature_current_sample_ranges_diverge_over_time(self):
        """Confirms the measured rate mismatch is actually reflected: later
        windows in a long recording should show vibration/temp-current
        sample-range start indices drifting apart, not staying identical."""
        long_condition = self.index_df[self.index_df["condition_code"] == "0Nm_Normal"]
        self.assertGreater(len(long_condition), 100)
        late_window = long_condition[long_condition["window_index"] == 100].iloc[0]
        vib_start = late_window["vibration_sample_range"][0]
        tc_start = late_window["temperature_current_sample_range"][0]
        self.assertNotEqual(vib_start, tc_start)

    def test_48_observation_count_matches_duration_over_window_duration(self):
        normal_0nm = self.index_df[self.index_df["condition_code"] == "0Nm_Normal"]
        expected = int(300.0 // obs.WINDOW_DURATION_SECONDS)
        self.assertEqual(len(normal_0nm), expected)

    def test_49_split_counts_sum_to_total_observations(self):
        counts = self.index_df["split"].value_counts()
        self.assertEqual(counts.sum(), len(self.index_df))
        self.assertEqual(set(counts.index), set(obs.SPLIT_NAMES))


class TestSavedArtifacts(unittest.TestCase):
    @unittest.skipUnless(
        os.path.exists(obs.OBSERVATION_SCHEMA_JSON_PATH), "observation schema has not been generated yet"
    )
    def test_50_schema_json_has_expected_top_level_fields(self):
        import json

        with open(obs.OBSERVATION_SCHEMA_JSON_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertEqual(schema["task"], 32)
        self.assertIn("alignment_strategy", schema)
        self.assertIn("canonical_modalities", schema)
        self.assertNotIn("acoustic", schema["canonical_modalities"])


if __name__ == "__main__":
    unittest.main()
