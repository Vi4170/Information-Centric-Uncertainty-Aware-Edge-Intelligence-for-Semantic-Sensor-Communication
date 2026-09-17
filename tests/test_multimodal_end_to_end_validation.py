"""Task 42 tests -- end-to-end edge intelligence validation.

Focused tests covering:
  - Each check function's pure logic on synthetic data (no raw data/models
    needed) -- both the pass and the fail path for each check.
  - The `_run_check` wrapper never lets one failing check crash the report.
  - Real-data integration tests (gated on all artifacts being present)
    confirm the whole pipeline currently certifies as consistent.

Test numbering uses prefix test_42_ to avoid collision with other tasks.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.multimodal_pipeline import end_to_end_validation as e2e
from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline import representation as rep
from src.multimodal_pipeline import voi_integration as vi

_RAW_DATA_AVAILABLE = schema.raw_data_available()
_MODELS_AVAILABLE = os.path.exists(rep.VIBRATION_MODEL_PATH) and os.path.exists(rep.MOTOR_CURRENT_MODEL_PATH)
_FUSION_MODELS_AVAILABLE = all(os.path.exists(fu.fusion_model_path(name)) for name in fu.FUSION_CONFIGS)
_ARTIFACTS_AVAILABLE = all(os.path.exists(p) for p in e2e.REQUIRED_ARTIFACTS)
_ALL_AVAILABLE = _RAW_DATA_AVAILABLE and _MODELS_AVAILABLE and _FUSION_MODELS_AVAILABLE and _ARTIFACTS_AVAILABLE

_COLUMNS = (
    "observation_id", "condition_code", "fault_type", "split", "fusion_config",
    "has_vibration", "has_current", "has_temperature",
    "predicted_class", "predicted_fault_type",
    "novelty", "uncertainty", "task_relevance", "temporal_importance", "resource_cost",
    "relevance_defined", "raw_voi_score", "voi_score", "decision",
)


def _row(**kwargs):
    defaults = {
        "observation_id": "o0", "condition_code": "c1", "fault_type": "Normal",
        "split": "train", "fusion_config": "vibration_only",
        "has_vibration": True, "has_current": True, "has_temperature": True,
        "predicted_class": 0, "predicted_fault_type": "Normal",
        "novelty": 0.1, "uncertainty": 0.1, "task_relevance": 0.10,
        "temporal_importance": 0.0, "resource_cost": 0.5049,
        "relevance_defined": True, "raw_voi_score": 0.2, "voi_score": 0.2, "decision": "DISCARD",
    }
    defaults.update(kwargs)
    return defaults


def _make_df(rows):
    return pd.DataFrame(rows, columns=list(_COLUMNS))


class TestRunCheckWrapper(unittest.TestCase):
    def test_42_01_successful_check_marked_passed(self):
        result = e2e._run_check("dummy", lambda: {"passed": True, "value": 1})
        self.assertTrue(result["passed"])
        self.assertEqual(result["check"], "dummy")

    def test_42_02_raising_check_is_caught_not_propagated(self):
        def bad():
            raise AssertionError("synthetic failure")

        result = e2e._run_check("dummy", bad)
        self.assertFalse(result["passed"])
        self.assertIn("synthetic failure", result["error"])
        self.assertIn("check", result)

    def test_42_03_missing_passed_key_defaults_to_true(self):
        result = e2e._run_check("dummy", lambda: {"value": 1})
        self.assertTrue(result["passed"])


class TestSplitIntegrityAndLeakage(unittest.TestCase):
    def test_42_04_clean_data_passes(self):
        df = _make_df(
            [
                _row(observation_id="a", condition_code="c1", split="train"),
                _row(observation_id="b", condition_code="c2", split="val"),
            ]
        )
        result = e2e.verify_split_integrity_and_no_leakage(df)
        self.assertTrue(result["passed"])

    def test_42_05_leaked_observation_id_is_caught(self):
        df = _make_df(
            [
                _row(observation_id="dup", split="train"),
                _row(observation_id="dup", split="val"),
            ]
        )
        with self.assertRaises(AssertionError):
            e2e.verify_split_integrity_and_no_leakage(df)

    def test_42_06_via_run_check_wrapper_reports_failed_not_raised(self):
        df = _make_df(
            [
                _row(observation_id="dup", split="train"),
                _row(observation_id="dup", split="val"),
            ]
        )
        result = e2e._run_check(
            "split_integrity_and_no_leakage", lambda: e2e.verify_split_integrity_and_no_leakage(df)
        )
        self.assertFalse(result["passed"])


class TestComponentAndDecisionAvailability(unittest.TestCase):
    def test_42_07_clean_defined_and_undefined_rows_pass(self):
        df = _make_df(
            [
                _row(observation_id="a"),
                _row(observation_id="b", relevance_defined=False, task_relevance=float("nan"),
                     voi_score=float("nan"), raw_voi_score=float("nan"),
                     predicted_fault_type="Unbalance", fault_type="Unbalance",
                     decision=vi.UNDEFINED_RELEVANCE_DECISION),
            ]
        )
        result = e2e.verify_component_and_decision_availability(df)
        self.assertTrue(result["passed"])
        self.assertEqual(result["n_relevance_defined"], 1)
        self.assertEqual(result["n_relevance_undefined"], 1)

    def test_42_08_non_finite_novelty_is_caught(self):
        df = _make_df([_row(observation_id="a", novelty=float("nan"))])
        with self.assertRaises(AssertionError):
            e2e.verify_component_and_decision_availability(df)

    def test_42_09_null_decision_is_caught(self):
        df = _make_df([_row(observation_id="a", decision=None)])
        with self.assertRaises(AssertionError):
            e2e.verify_component_and_decision_availability(df)

    def test_42_10_fabricated_zero_voi_for_undefined_relevance_is_caught(self):
        df = _make_df(
            [_row(observation_id="a", relevance_defined=False, task_relevance=float("nan"),
                  voi_score=0.0, raw_voi_score=0.0, predicted_fault_type="Unbalance",
                  fault_type="Unbalance", decision=vi.UNDEFINED_RELEVANCE_DECISION)]
        )
        with self.assertRaises(AssertionError):
            e2e.verify_component_and_decision_availability(df)


class TestRequiredArtifactsExist(unittest.TestCase):
    def test_42_11_reports_missing_paths_explicitly(self):
        original = list(e2e.REQUIRED_ARTIFACTS)
        try:
            e2e.REQUIRED_ARTIFACTS[:] = ["does_not_exist_xyz.json"]
            result = e2e.verify_required_artifacts_exist()
            self.assertFalse(result["passed"])
            self.assertIn("does_not_exist_xyz.json", result["missing"])
        finally:
            e2e.REQUIRED_ARTIFACTS[:] = original

    def test_42_12_passes_when_all_paths_exist(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            original = list(e2e.REQUIRED_ARTIFACTS)
            e2e.REQUIRED_ARTIFACTS[:] = [tmp_path]
            result = e2e.verify_required_artifacts_exist()
            self.assertTrue(result["passed"])
            e2e.REQUIRED_ARTIFACTS[:] = original
        finally:
            os.unlink(tmp_path)


class TestCrossArtifactCountConsistency(unittest.TestCase):
    def test_42_13_matching_counts_pass(self):
        results_df = _make_df(
            [_row(observation_id="a", split="train"), _row(observation_id="b", split="train")]
        )
        decision_summary_df = pd.DataFrame(
            [{"fusion_config": "vibration_only", "split": "train", "n": 2}]
        )
        factor_summary_df = pd.DataFrame(
            [{"fusion_config": "vibration_only", "split": "train", "n": 2}]
        )
        result = e2e.verify_cross_artifact_count_consistency(results_df, decision_summary_df, factor_summary_df)
        self.assertTrue(result["passed"])

    def test_42_14_mismatched_count_is_caught(self):
        results_df = _make_df(
            [_row(observation_id="a", split="train"), _row(observation_id="b", split="train")]
        )
        decision_summary_df = pd.DataFrame(
            [{"fusion_config": "vibration_only", "split": "train", "n": 999}]  # stale
        )
        factor_summary_df = pd.DataFrame(
            [{"fusion_config": "vibration_only", "split": "train", "n": 2}]
        )
        result = e2e.verify_cross_artifact_count_consistency(results_df, decision_summary_df, factor_summary_df)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["mismatches"]), 1)
        self.assertEqual(result["mismatches"][0]["task40_n"], 999)


class TestDeterministicObservationIds(unittest.TestCase):
    @unittest.skipUnless(_RAW_DATA_AVAILABLE, "Requires local raw multimodal data")
    def test_42_15_rebuild_is_stable_and_matches_persisted_index(self):
        result = e2e.verify_deterministic_observation_ids()
        self.assertTrue(result["passed"])
        self.assertTrue(result["rebuild_matches_itself"])
        self.assertTrue(result["rebuild_matches_persisted_index"])


@unittest.skipUnless(_ALL_AVAILABLE, "Requires local raw multimodal data, trained models, and all prior task artifacts")
class TestRealTask42Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = e2e.run_task42()

    def test_42_16_all_seven_checks_present(self):
        expected = {
            "deterministic_observation_ids", "train_only_fitting",
            "split_integrity_and_no_leakage", "model_reload_consistency",
            "required_artifacts_exist", "cross_artifact_count_consistency",
            "component_and_decision_availability",
        }
        self.assertEqual(set(self.report["checks"].keys()), expected)

    def test_42_17_pipeline_is_currently_certified_ready(self):
        self.assertEqual(self.report["failed_checks"], [])
        self.assertTrue(self.report["pipeline_ready_for_communication_channel_experiment"])

    def test_42_18_fso_not_introduced(self):
        self.assertFalse(self.report["fso_introduced"])

    def test_42_19_model_reload_consistency_covers_all_four_configs(self):
        per_config = self.report["checks"]["model_reload_consistency"]["per_configuration"]
        self.assertEqual(set(per_config.keys()), set(fu.FUSION_CONFIGS.keys()))
        for name, info in per_config.items():
            self.assertTrue(info["reload_consistent"], f"{name} failed reload consistency")

    def test_42_20_required_artifacts_all_present(self):
        self.assertEqual(self.report["checks"]["required_artifacts_exist"]["missing"], [])

    def test_42_21_report_is_json_serializable(self):
        # Must round-trip cleanly since this is what gets persisted.
        json.dumps(self.report, default=str)

    def test_42_22_deterministic_across_repeated_runs(self):
        report2 = e2e.run_task42()
        self.assertEqual(
            json.dumps(self.report, sort_keys=True, default=str),
            json.dumps(report2, sort_keys=True, default=str),
        )


if __name__ == "__main__":
    unittest.main()
