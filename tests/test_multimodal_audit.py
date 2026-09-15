import json
import os
import unittest

from src.multimodal_pipeline.dataset_audit import (
    ALL_MODALITIES,
    AUDIT_JSON_PATH,
    AUDIT_STATUS_CONFIRMED,
    AUDIT_STATUS_EXCLUDED,
    AUDIT_STATUS_NOT_DOCUMENTED,
    FAULT_TYPES,
    LOAD_LEVELS_NM,
    MODALITY_REGISTRY,
    N_ACOUSTIC_FILES_PART1_SOURCE_STATED,
    N_CONDITIONS_PART1,
    N_TEMPERATURE_CURRENT_FILES_PART1_EXPECTED,
    N_TEMPERATURE_CURRENT_FILES_PART1_SOURCE_STATED,
    N_VIBRATION_FILES_PART1_EXPECTED,
    N_VIBRATION_FILES_PART1_SOURCE_STATED,
    OFFICIAL_DOI,
    RAW_DATA_DIR,
    SEVERITY_LEVELS,
    VALID_AUDIT_STATUSES,
    build_dataset_audit,
    raw_data_root_available,
)

AUDIT_QUESTION_KEYS = [
    "1_available_modalities",
    "2_raw_file_formats_and_directory_structure",
    "3_sampling_rates",
    "4_timestamp_availability_and_resolution",
    "5_recording_session_experiment_identifiers",
    "6_machine_operating_load_conditions",
    "7_health_fault_labels_and_granularity",
    "8_modalities_from_same_physical_experiment",
    "9_cross_modal_synchronization_legitimacy",
    "10_modalities_safely_combinable",
    "11_exclusions",
    "12_missing_data_and_alignment_issues",
    "13_dataset_level_leakage_risks",
    "14_source_provenance_metadata",
    "15_sufficiency_for_task_32",
]


class TestModalityRegistryStructure(unittest.TestCase):
    """Structural checks that require no raw data."""

    def test_01_four_modalities_registered(self):
        self.assertEqual(set(ALL_MODALITIES), {"vibration", "acoustic", "temperature", "motor_current"})

    def test_02_every_modality_has_a_sampling_rate_status(self):
        for modality, info in MODALITY_REGISTRY.items():
            self.assertIn("sampling_rate_status", info, modality)
            for part, status in info["sampling_rate_status"].items():
                self.assertIn(status, VALID_AUDIT_STATUSES, f"{modality}.{part}")

    def test_03_temperature_and_motor_current_share_a_raw_file(self):
        self.assertEqual(MODALITY_REGISTRY["temperature"]["shares_raw_file_with"], "motor_current")

    def test_04_acoustic_not_present_in_part2_speed(self):
        self.assertFalse(MODALITY_REGISTRY["acoustic"]["present_in_part2_speed"])

    def test_05_temperature_not_present_in_part2_speed(self):
        self.assertFalse(MODALITY_REGISTRY["temperature"]["present_in_part2_speed"])

    def test_06_vibration_and_motor_current_present_in_both_parts(self):
        self.assertTrue(MODALITY_REGISTRY["vibration"]["present_in_part1_load"])
        self.assertTrue(MODALITY_REGISTRY["vibration"]["present_in_part2_speed"])
        self.assertTrue(MODALITY_REGISTRY["motor_current"]["present_in_part1_load"])
        self.assertTrue(MODALITY_REGISTRY["motor_current"]["present_in_part2_speed"])

    def test_07_motor_current_sampling_rate_differs_between_parts(self):
        rates = MODALITY_REGISTRY["motor_current"]["sampling_rate_hz"]
        self.assertNotEqual(rates["part1_load"], rates["part2_speed"])
        self.assertEqual(rates["part1_load"], 25600)
        self.assertEqual(rates["part2_speed"], 100000)


class TestSeverityAndFileCountConsistency(unittest.TestCase):
    """Verify the derived file-count math matches the source-stated totals."""

    def test_08_five_fault_types(self):
        self.assertEqual(len(FAULT_TYPES), 5)
        self.assertIn("Normal", FAULT_TYPES)

    def test_09_normal_has_no_severity_levels(self):
        self.assertEqual(SEVERITY_LEVELS["Normal"], ())

    def test_10_unbalance_has_five_severities_others_have_three(self):
        self.assertEqual(len(SEVERITY_LEVELS["Unbalance"]), 5)
        for fault in ("BPFI", "BPFO", "Misalignment"):
            self.assertEqual(len(SEVERITY_LEVELS[fault]), 3)

    def test_11_three_load_levels(self):
        self.assertEqual(LOAD_LEVELS_NM, (0, 2, 4))

    def test_12_n_conditions_part1_is_15(self):
        # Normal(1) + BPFI(3) + BPFO(3) + Misalignment(3) + Unbalance(5) = 15
        self.assertEqual(N_CONDITIONS_PART1, 15)

    def test_13_derived_vibration_count_matches_source_stated(self):
        self.assertEqual(N_VIBRATION_FILES_PART1_EXPECTED, 45)
        self.assertEqual(N_VIBRATION_FILES_PART1_EXPECTED, N_VIBRATION_FILES_PART1_SOURCE_STATED)

    def test_14_derived_temperature_current_count_matches_source_stated(self):
        self.assertEqual(N_TEMPERATURE_CURRENT_FILES_PART1_EXPECTED, 45)
        self.assertEqual(
            N_TEMPERATURE_CURRENT_FILES_PART1_EXPECTED,
            N_TEMPERATURE_CURRENT_FILES_PART1_SOURCE_STATED,
        )

    def test_15_acoustic_file_count_is_the_small_documented_subset(self):
        self.assertEqual(N_ACOUSTIC_FILES_PART1_SOURCE_STATED, 5)
        self.assertLess(N_ACOUSTIC_FILES_PART1_SOURCE_STATED, N_VIBRATION_FILES_PART1_SOURCE_STATED)


class TestRawDataAbsence(unittest.TestCase):
    def test_16_raw_data_directory_does_not_exist_in_this_environment(self):
        self.assertFalse(os.path.isdir(RAW_DATA_DIR))

    def test_17_raw_data_root_available_returns_false(self):
        self.assertFalse(raw_data_root_available())


class TestAuditDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = build_dataset_audit()

    def test_18_task_and_title_fields(self):
        self.assertEqual(self.audit["task"], 31)
        self.assertIn("multimodal", self.audit["title"].lower() + self.audit["dataset"])

    def test_19_all_15_audit_questions_present(self):
        for key in AUDIT_QUESTION_KEYS:
            self.assertIn(key, self.audit, f"missing audit key {key}")

    def test_20_raw_data_available_flag_matches_environment(self):
        self.assertEqual(self.audit["raw_data_available_in_environment"], raw_data_root_available())
        self.assertFalse(self.audit["raw_data_available_in_environment"])

    def test_21_doi_is_the_locked_dataset(self):
        self.assertEqual(self.audit["14_source_provenance_metadata"]["doi"], OFFICIAL_DOI)
        self.assertEqual(OFFICIAL_DOI, "10.17632/ztmf3m7h5x.6")

    def test_22_sample_level_synchronization_is_not_documented_not_confirmed(self):
        sync = self.audit["9_cross_modal_synchronization_legitimacy"]
        self.assertEqual(
            sync["sample_level_synchronization"]["status"],
            AUDIT_STATUS_NOT_DOCUMENTED,
        )
        self.assertNotEqual(
            sync["sample_level_synchronization"]["status"],
            AUDIT_STATUS_CONFIRMED,
        )

    def test_23_session_level_correspondence_is_confirmed_for_the_core_triple(self):
        sync = self.audit["9_cross_modal_synchronization_legitimacy"]
        session = sync["session_level_correspondence"]
        self.assertEqual(session["status"], AUDIT_STATUS_CONFIRMED)
        self.assertEqual(set(session["modalities"]), {"vibration", "temperature", "motor_current"})

    def test_24_acoustic_synchronization_is_excluded(self):
        sync = self.audit["9_cross_modal_synchronization_legitimacy"]
        self.assertEqual(sync["acoustic_synchronization"]["status"], AUDIT_STATUS_EXCLUDED)

    def test_25_part2_speed_is_excluded_from_primary_fusion_scope(self):
        combinable = self.audit["10_modalities_safely_combinable"]
        self.assertEqual(combinable["part2_speed"]["status"], AUDIT_STATUS_EXCLUDED)

    def test_26_primary_combinable_triple_is_vibration_temperature_current(self):
        combinable = self.audit["10_modalities_safely_combinable"]
        primary = combinable["part1_load_primary_triple"]
        self.assertEqual(set(primary["modalities"]), {"vibration", "temperature", "motor_current"})
        self.assertNotIn("acoustic", primary["modalities"])

    def test_27_exclusions_list_is_non_empty_and_mentions_acoustic_and_part2(self):
        exclusions_text = " ".join(self.audit["11_exclusions"]).lower()
        self.assertGreater(len(self.audit["11_exclusions"]), 0)
        self.assertIn("acoustic", exclusions_text)
        self.assertIn("part2_speed", exclusions_text)

    def test_28_sufficiency_conclusion_is_conditional_not_unconditional(self):
        sufficiency = self.audit["15_sufficiency_for_task_32"]
        self.assertEqual(sufficiency["conclusion"], "conditionally_sufficient")
        self.assertTrue(sufficiency["blocking_for_task_32"])

    def test_29_internal_consistency_checks_pass(self):
        checks = self.audit["internal_consistency_checks"]
        self.assertTrue(checks["n_vibration_files_match"])
        self.assertTrue(checks["n_temperature_current_files_match"])

    def test_30_no_fabricated_acoustic_condition_identities(self):
        """The audit must not claim to know which fault/severity acoustic covers."""
        modality_detail = self.audit["1_available_modalities"]["detail"]["acoustic"]
        coverage_text = modality_detail["coverage"]
        # Only Normal is a specifically named condition for acoustic; the
        # source does not say which 2 fault types/severities are covered.
        for fault in ("BPFI", "BPFO", "Misalignment", "Unbalance"):
            self.assertNotIn(fault, coverage_text)

    def test_31_status_vocabulary_matches_valid_statuses(self):
        self.assertEqual(set(self.audit["status_vocabulary"].keys()), VALID_AUDIT_STATUSES)

    def test_32_audit_is_json_serializable(self):
        serialized = json.dumps(self.audit, default=str)
        self.assertGreater(len(serialized), 0)


class TestSavedAuditArtifact(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(AUDIT_JSON_PATH), "audit artifact has not been generated yet")
    def test_33_saved_artifact_matches_module_output(self):
        with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["task"], 31)
        self.assertEqual(saved["dataset"], "multimodal")
        for key in AUDIT_QUESTION_KEYS:
            self.assertIn(key, saved)


if __name__ == "__main__":
    unittest.main()
