"""Task 38 tests -- multimodal relevance evidence audit and 5-class map.

Focused tests covering:
  - 5-class map values and provenance
  - Undefined classes are None / NaN, never fabricated
  - Coverage utilities work on synthetic data
  - Artifact builder produces correct structure
  - Protected modules unchanged

Test numbering continues from the existing multimodal test suite
(test_multimodal_uncertainty_relevance.py uses test_01..test_30,
test_multimodal_relevance_prediction_audit.py uses test_38a_01...).
These tests use prefix test_38_ to avoid any collision.
"""

import json
import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.multimodal_pipeline import relevance as rel38
from src.multimodal_pipeline.uncertainty_relevance import (
    EXPECTED_FAULT_TYPE_LABELS,
    MULTIMODAL_RELEVANCE_MAP,
    UNDEFINED_RELEVANCE_CLASSES,
)
from src.relevance.config import CLASS_RELEVANCE_MAP as CWRU_CLASS_RELEVANCE_MAP


class TestTask38MapConsistencyWithTask37(unittest.TestCase):
    """rel38 re-exports uncertainty_relevance's map -- must be identical."""

    def test_38_01_five_class_map_identical_to_task37_map(self):
        self.assertEqual(rel38.FIVE_CLASS_RELEVANCE_MAP, MULTIMODAL_RELEVANCE_MAP)

    def test_38_02_expected_labels_identical_to_task37(self):
        self.assertEqual(
            tuple(EXPECTED_FAULT_TYPE_LABELS),
            ("Normal", "BPFI", "BPFO", "Misalignment", "Unbalance"),
        )

    def test_38_03_undefined_classes_identical_to_task37(self):
        self.assertEqual(set(rel38.UNDEFINED_CLASSES), set(UNDEFINED_RELEVANCE_CLASSES))
        self.assertEqual(rel38.UNDEFINED_CLASSES, UNDEFINED_RELEVANCE_CLASSES)


class TestTask38DefinedValues(unittest.TestCase):
    """Defined values match CWRU CLASS_RELEVANCE_MAP by fault-category correspondence."""

    def test_38_04_normal_reuses_cwru_normal(self):
        self.assertAlmostEqual(
            rel38.FIVE_CLASS_RELEVANCE_MAP[0], CWRU_CLASS_RELEVANCE_MAP[0], places=6
        )
        self.assertAlmostEqual(rel38.FIVE_CLASS_RELEVANCE_MAP[0], 0.10, places=6)

    def test_38_05_bpfi_reuses_cwru_inner_race(self):
        self.assertAlmostEqual(
            rel38.FIVE_CLASS_RELEVANCE_MAP[1], CWRU_CLASS_RELEVANCE_MAP[1], places=6
        )
        self.assertAlmostEqual(rel38.FIVE_CLASS_RELEVANCE_MAP[1], 1.00, places=6)

    def test_38_06_bpfo_reuses_cwru_outer_race_index3_not_ball_index2(self):
        # Must be CWRU[3] (Outer Race Fault), not CWRU[2] (Ball Fault).
        # Both are 0.90 numerically but the provenance matters.
        self.assertAlmostEqual(
            rel38.FIVE_CLASS_RELEVANCE_MAP[2], CWRU_CLASS_RELEVANCE_MAP[3], places=6
        )
        self.assertAlmostEqual(rel38.FIVE_CLASS_RELEVANCE_MAP[2], 0.90, places=6)

    def test_38_07_all_defined_values_in_unit_interval(self):
        for class_id, value in rel38.FIVE_CLASS_RELEVANCE_MAP.items():
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_38_08_defined_classes_are_normal_bpfi_bpfo(self):
        self.assertEqual(set(rel38.DEFINED_CLASSES), {"Normal", "BPFI", "BPFO"})


class TestTask38UndefinedNotFabricated(unittest.TestCase):
    """Undefined classes are None -- never zero, mean, or any fabricated number."""

    def test_38_09_misalignment_is_none(self):
        self.assertIsNone(rel38.FIVE_CLASS_RELEVANCE_MAP[3])

    def test_38_10_unbalance_is_none(self):
        self.assertIsNone(rel38.FIVE_CLASS_RELEVANCE_MAP[4])

    def test_38_11_undefined_classes_exactly_misalignment_unbalance(self):
        self.assertEqual(set(rel38.UNDEFINED_CLASSES), {"Misalignment", "Unbalance"})

    def test_38_12_undefined_not_zero(self):
        self.assertNotEqual(rel38.FIVE_CLASS_RELEVANCE_MAP[3], 0.0)
        self.assertNotEqual(rel38.FIVE_CLASS_RELEVANCE_MAP[4], 0.0)

    def test_38_13_undefined_not_mean_of_defined(self):
        defined_values = [v for v in rel38.FIVE_CLASS_RELEVANCE_MAP.values() if v is not None]
        mean_defined = sum(defined_values) / len(defined_values)
        self.assertNotEqual(rel38.FIVE_CLASS_RELEVANCE_MAP[3], mean_defined)
        self.assertNotEqual(rel38.FIVE_CLASS_RELEVANCE_MAP[4], mean_defined)


class TestTask38MappingStatus(unittest.TestCase):
    """Mapping status is B_PARTIAL_MAPPING_ONLY."""

    def test_38_14_mapping_status_is_partial(self):
        self.assertEqual(rel38.MAPPING_STATUS, "B_PARTIAL_MAPPING_ONLY")

    def test_38_15_conclusion_mentions_undefined(self):
        self.assertIn("undefined", rel38.MAPPING_CONCLUSION.lower())
        self.assertIn("partial", rel38.MAPPING_CONCLUSION.lower())


class TestTask38PerClassEvidence(unittest.TestCase):
    """Per-class evidence dictionary covers all 5 classes with required fields."""

    def test_38_16_evidence_covers_all_five_classes(self):
        self.assertEqual(
            set(rel38.CLASS_EVIDENCE.keys()),
            {"Normal", "BPFI", "BPFO", "Misalignment", "Unbalance"},
        )

    def test_38_17_defined_classes_have_status_defined(self):
        for name in ("Normal", "BPFI", "BPFO"):
            self.assertEqual(rel38.CLASS_EVIDENCE[name]["status"], "defined")

    def test_38_18_undefined_classes_have_status_undefined(self):
        for name in ("Misalignment", "Unbalance"):
            self.assertEqual(rel38.CLASS_EVIDENCE[name]["status"], "undefined")

    def test_38_19_undefined_classes_have_none_value_in_evidence(self):
        for name in ("Misalignment", "Unbalance"):
            self.assertIsNone(rel38.CLASS_EVIDENCE[name]["relevance_value"])

    def test_38_20_all_evidence_has_why_not_fabricated_key(self):
        for name, entry in rel38.CLASS_EVIDENCE.items():
            self.assertIn("why_not_fabricated", entry, msg=f"Missing for {name}")


class TestTask38HelperFunctions(unittest.TestCase):
    """relevance_is_defined and relevance_value_for_class work correctly."""

    def test_38_21_relevance_is_defined_true_for_normal_bpfi_bpfo(self):
        for fault_type in ("Normal", "BPFI", "BPFO"):
            self.assertTrue(rel38.relevance_is_defined(fault_type))

    def test_38_22_relevance_is_defined_false_for_misalignment_unbalance(self):
        for fault_type in ("Misalignment", "Unbalance"):
            self.assertFalse(rel38.relevance_is_defined(fault_type))

    def test_38_23_relevance_is_defined_raises_on_unknown(self):
        with self.assertRaises(ValueError):
            rel38.relevance_is_defined("BallFault")

    def test_38_24_relevance_value_for_class_returns_none_for_undefined(self):
        self.assertIsNone(rel38.relevance_value_for_class(3))
        self.assertIsNone(rel38.relevance_value_for_class(4))

    def test_38_25_relevance_value_for_class_returns_float_for_defined(self):
        val = rel38.relevance_value_for_class(0)
        self.assertIsNotNone(val)
        self.assertIsInstance(val, float)

    def test_38_26_relevance_value_for_class_raises_on_out_of_range(self):
        with self.assertRaises((ValueError, KeyError)):
            rel38.relevance_value_for_class(99)


class TestTask38CoverageUtility(unittest.TestCase):
    """compute_relevance_coverage_summary returns correct counts on synthetic data."""

    def _make_fake_results_df(self) -> pd.DataFrame:
        """3 rows defined (Normal/BPFI/BPFO), 2 rows undefined (Misalignment/Unbalance)
        for vibration_only/train."""
        rows = []
        fault_types_predicted = ["Normal", "BPFI", "BPFO", "Misalignment", "Unbalance"]
        relevance_values = [0.10, 1.00, 0.90, float("nan"), float("nan")]
        for ft, rv in zip(fault_types_predicted, relevance_values):
            rows.append(
                {
                    "observation_id": f"obs_{ft}",
                    "condition_code": "0Nm_Normal",
                    "fault_type": ft,
                    "split": "train",
                    "fusion_config": "vibration_only",
                    "has_vibration": True,
                    "has_current": True,
                    "has_temperature": True,
                    "predicted_class": fault_types_predicted.index(ft),
                    "predicted_fault_type": ft,
                    "uncertainty_score": 0.1,
                    "relevance_score": rv,
                }
            )
        return pd.DataFrame(rows)

    def test_38_27_coverage_counts_correct(self):
        df = self._make_fake_results_df()
        coverage = rel38.compute_relevance_coverage_summary(df)
        vo = coverage["vibration_only"]["train"]
        self.assertEqual(vo["n"], 5)
        self.assertEqual(vo["n_defined"], 3)
        self.assertEqual(vo["n_undefined"], 2)

    def test_38_28_coverage_percentages_correct(self):
        df = self._make_fake_results_df()
        coverage = rel38.compute_relevance_coverage_summary(df)
        vo = coverage["vibration_only"]["train"]
        self.assertAlmostEqual(vo["pct_defined"], 60.0, places=2)
        self.assertAlmostEqual(vo["pct_undefined"], 40.0, places=2)

    def test_38_29_coverage_empty_split_returns_zeros(self):
        df = self._make_fake_results_df()
        coverage = rel38.compute_relevance_coverage_summary(df)
        # val and test are empty for all configs in synthetic data
        for split in ("val", "test"):
            vo = coverage["vibration_only"][split]
            self.assertEqual(vo["n"], 0)
            self.assertEqual(vo["n_defined"], 0)


class TestTask38ArtifactBuilder(unittest.TestCase):
    """build_relevance_audit_record returns correct structure (no CSV needed)."""

    def test_38_30_artifact_has_required_keys(self):
        record = rel38.build_relevance_audit_record(uncertainty_relevance_csv_path="/nonexistent/path.csv")
        required_keys = [
            "task",
            "title",
            "mapping_status",
            "mapping_conclusion",
            "five_class_map",
            "defined_classes",
            "undefined_classes",
            "per_class_evidence",
            "downstream_handling_of_undefined",
            "protected_modules_unchanged",
        ]
        for key in required_keys:
            self.assertIn(key, record, msg=f"Missing key: {key}")

    def test_38_31_artifact_task_is_38(self):
        record = rel38.build_relevance_audit_record(uncertainty_relevance_csv_path="/nonexistent")
        self.assertEqual(str(record["task"]), "38")

    def test_38_32_artifact_mapping_status_is_partial(self):
        record = rel38.build_relevance_audit_record(uncertainty_relevance_csv_path="/nonexistent")
        self.assertEqual(record["mapping_status"], "B_PARTIAL_MAPPING_ONLY")

    def test_38_33_artifact_five_class_map_has_none_for_misalignment(self):
        record = rel38.build_relevance_audit_record(uncertainty_relevance_csv_path="/nonexistent")
        self.assertIsNone(record["five_class_map"]["Misalignment"])
        self.assertIsNone(record["five_class_map"]["Unbalance"])

    def test_38_34_artifact_save_load_roundtrip(self):
        record = rel38.build_relevance_audit_record(uncertainty_relevance_csv_path="/nonexistent")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "audit.json")
            rel38.save_relevance_audit(record, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "r") as f:
                loaded = json.load(f)
        self.assertEqual(loaded["mapping_status"], "B_PARTIAL_MAPPING_ONLY")
        # None is serialized as null in JSON, loaded as None
        self.assertIsNone(loaded["five_class_map"]["Misalignment"])


class TestTask38ProtectedModulesNotImportedForModification(unittest.TestCase):
    """Verify that rel38 doesn't import from protected modules for modification."""

    def test_38_35_relevance_module_does_not_call_protected_relevance_validate(self):
        # The protected src/relevance/relevance.py's _validate_relevance_map requires
        # NUM_CLASSES=4. rel38 must NOT call it (it would reject our 5-class map).
        import src.multimodal_pipeline.relevance as mod
        # Inspect imports at module level -- it should NOT import from src.relevance.relevance
        import inspect
        source = inspect.getsource(mod)
        self.assertNotIn("from src.relevance.relevance import", source)
        self.assertNotIn("import src.relevance.relevance", source)


if __name__ == "__main__":
    import unittest
    unittest.main()
