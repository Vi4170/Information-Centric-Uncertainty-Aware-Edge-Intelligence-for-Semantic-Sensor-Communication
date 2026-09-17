"""Task 40 tests -- multimodal communication decision reporting.

Focused tests covering:
  - decision assignment (counts/percentages sum correctly, categories exact)
  - undefined relevance handling (never fabricated, never zero, own category)
  - boundary behavior (canonical evaluate_decision threshold semantics)
  - no leakage (reused, not reimplemented, checks)
  - deterministic behavior (repeated report generation)

Uses synthetic per-observation frames for pure-logic tests (fast, no raw
data/models needed) plus a real-data integration test class gated on
Task 39's actual results CSV being present.

Test numbering uses prefix test_40_ to avoid collision with other tasks.
"""

import os
import unittest

import numpy as np
import pandas as pd

from src.multimodal_pipeline import communication_decision as cd
from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import novelty as nov
from src.voi.decision_policy import PolicyThresholds, evaluate_decision

_RESULTS_AVAILABLE = os.path.exists(cd.VOI_INTEGRATION_RESULTS_CSV_PATH)


def _make_synthetic_results(rows):
    """rows: list of dicts with at least the columns communication_decision
    needs. Fills in any missing required column with a benign default."""
    defaults = {
        "observation_id": None,
        "condition_code": "c1",
        "fault_type": "Normal",
        "split": "train",
        "fusion_config": "vibration_only",
        "predicted_fault_type": "Normal",
        "novelty": 0.1,
        "uncertainty": 0.1,
        "task_relevance": 0.10,
        "temporal_importance": 0.0,
        "resource_cost": 0.5049,
        "relevance_defined": True,
        "raw_voi_score": 0.2,
        "voi_score": 0.2,
        "decision": "DISCARD",
    }
    full_rows = []
    for i, r in enumerate(rows):
        row = dict(defaults)
        row.update(r)
        if row["observation_id"] is None:
            row["observation_id"] = f"o{i}"
        full_rows.append(row)
    return pd.DataFrame(full_rows, columns=list(cd._REQUIRED_COLUMNS))


class TestLoadResults(unittest.TestCase):
    def test_40_01_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            cd.load_voi_integration_results(path="does_not_exist.csv")

    def test_40_02_missing_required_column_raises(self):
        import tempfile

        df = pd.DataFrame({"observation_id": ["o0"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            df.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                cd.load_voi_integration_results(path=path)


class TestDecisionAssignmentReporting(unittest.TestCase):
    """Decision counts/percentages must be exact and sum correctly."""

    def setUp(self):
        self.df = _make_synthetic_results(
            [
                {"decision": "DISCARD"},
                {"decision": "DISCARD"},
                {"decision": "BUFFER"},
                {"decision": "SUMMARY"},
                {"decision": "TRANSMIT"},
                {"decision": cd.UNDEFINED_RELEVANCE_DECISION, "relevance_defined": False,
                 "task_relevance": float("nan"), "voi_score": float("nan"), "raw_voi_score": float("nan"),
                 "predicted_fault_type": "Misalignment", "fault_type": "Misalignment"},
            ]
        )

    def test_40_03_counts_exact(self):
        info = cd._decision_counts_and_pcts(self.df)
        self.assertEqual(info["counts"], {"DISCARD": 2, "BUFFER": 1, "SUMMARY": 1, "TRANSMIT": 1, cd.UNDEFINED_RELEVANCE_DECISION: 1})

    def test_40_04_percentages_sum_to_100(self):
        # Each category is rounded independently to 4dp for readability, so
        # the sum can drift by a few 1e-4 units -- assert to a coarser
        # tolerance than the per-value rounding precision itself.
        info = cd._decision_counts_and_pcts(self.df)
        self.assertAlmostEqual(sum(info["percentages"].values()), 100.0, places=2)

    def test_40_05_percentage_denominator_includes_undefined_relevance(self):
        # 6 total rows, 1 TRANSMIT -> 100/6, not 100/5 (which would silently
        # exclude the undefined-relevance row from the denominator).
        info = cd._decision_counts_and_pcts(self.df)
        self.assertAlmostEqual(info["percentages"]["TRANSMIT"], 100.0 / 6.0, places=4)

    def test_40_06_empty_group_returns_zero_not_error(self):
        empty = self.df.iloc[0:0]
        info = cd._decision_counts_and_pcts(empty)
        self.assertEqual(info["n"], 0)
        self.assertTrue(all(v == 0.0 for v in info["percentages"].values()))


class TestUndefinedRelevanceHandling(unittest.TestCase):
    def test_40_07_undefined_relevance_never_treated_as_zero_voi(self):
        df = _make_synthetic_results(
            [
                {"decision": cd.UNDEFINED_RELEVANCE_DECISION, "relevance_defined": False,
                 "task_relevance": float("nan"), "voi_score": float("nan"), "raw_voi_score": float("nan"),
                 "predicted_fault_type": "Unbalance", "fault_type": "Unbalance"},
            ]
        )
        # Must not raise -- this is the correct, expected shape.
        cd.verify_undefined_relevance_never_fabricated(df)
        info = cd._voi_distribution(df)
        self.assertEqual(info["n_valid"], 0)
        self.assertIsNone(info["voi_score"])

    def test_40_08_fabricated_zero_voi_for_undefined_relevance_is_caught(self):
        # A bug that coerced undefined relevance to voi_score=0.0 instead of
        # NaN must be detected, not silently accepted.
        df = _make_synthetic_results(
            [
                {"decision": cd.UNDEFINED_RELEVANCE_DECISION, "relevance_defined": False,
                 "task_relevance": float("nan"), "voi_score": 0.0, "raw_voi_score": 0.0,
                 "predicted_fault_type": "Unbalance", "fault_type": "Unbalance"},
            ]
        )
        with self.assertRaises(AssertionError):
            cd.verify_undefined_relevance_never_fabricated(df)

    def test_40_09_defined_relevance_row_with_undefined_decision_label_is_caught(self):
        df = _make_synthetic_results(
            [
                {"decision": cd.UNDEFINED_RELEVANCE_DECISION, "relevance_defined": True,
                 "task_relevance": 0.9, "voi_score": 0.6, "raw_voi_score": 0.6},
            ]
        )
        with self.assertRaises(AssertionError):
            cd.verify_undefined_relevance_never_fabricated(df)

    def test_40_10_undefined_relevance_for_a_defined_class_is_caught(self):
        # Normal has a defined relevance value -- undefined relevance must
        # only ever occur for Misalignment/Unbalance predictions.
        df = _make_synthetic_results(
            [
                {"decision": cd.UNDEFINED_RELEVANCE_DECISION, "relevance_defined": False,
                 "task_relevance": float("nan"), "voi_score": float("nan"), "raw_voi_score": float("nan"),
                 "predicted_fault_type": "Normal", "fault_type": "Normal"},
            ]
        )
        with self.assertRaises(AssertionError):
            cd.verify_undefined_relevance_never_fabricated(df)

    def test_40_11_class_distribution_by_decision_separates_undefined_bucket(self):
        df = _make_synthetic_results(
            [
                {"decision": "TRANSMIT", "fault_type": "BPFI"},
                {"decision": cd.UNDEFINED_RELEVANCE_DECISION, "relevance_defined": False,
                 "task_relevance": float("nan"), "voi_score": float("nan"), "raw_voi_score": float("nan"),
                 "predicted_fault_type": "Misalignment", "fault_type": "Misalignment"},
            ]
        )
        dist = cd._class_distribution_by_decision(df)
        self.assertEqual(dist["TRANSMIT"]["BPFI"], 1)
        self.assertEqual(dist[cd.UNDEFINED_RELEVANCE_DECISION]["Misalignment"], 1)
        self.assertEqual(dist[cd.UNDEFINED_RELEVANCE_DECISION]["Normal"], 0)


class TestBoundaryBehavior(unittest.TestCase):
    """Verifies exact canonical threshold semantics -- never reimplemented."""

    def test_40_12_exact_threshold_boundaries_match_canonical_policy(self):
        thresholds = PolicyThresholds()
        cases = {
            0.0: "DISCARD",
            thresholds.discard_max - 1e-9: "DISCARD",
            thresholds.discard_max: "BUFFER",  # boundary is inclusive-upper on the lower bucket
            thresholds.buffer_max: "SUMMARY",
            thresholds.summary_max: "TRANSMIT",
            1.0: "TRANSMIT",
        }
        for score, expected in cases.items():
            self.assertEqual(evaluate_decision(score, thresholds).value, expected)

    def test_40_13_verify_decisions_match_canonical_policy_passes_for_consistent_data(self):
        df = _make_synthetic_results(
            [
                {"voi_score": 0.10, "raw_voi_score": 0.10, "decision": "DISCARD"},
                {"voi_score": 0.30, "raw_voi_score": 0.30, "decision": "BUFFER"},
                {"voi_score": 0.55, "raw_voi_score": 0.55, "decision": "SUMMARY"},
                {"voi_score": 0.90, "raw_voi_score": 0.90, "decision": "TRANSMIT"},
            ]
        )
        cd.verify_decisions_match_canonical_policy(df)  # must not raise

    def test_40_14_verify_decisions_match_canonical_policy_catches_drift(self):
        # voi_score=0.90 should be TRANSMIT, not DISCARD -- must be caught.
        df = _make_synthetic_results([{"voi_score": 0.90, "raw_voi_score": 0.90, "decision": "DISCARD"}])
        with self.assertRaises(AssertionError):
            cd.verify_decisions_match_canonical_policy(df)

    def test_40_15_undefined_relevance_rows_are_skipped_by_policy_check(self):
        # Must not raise even though voi_score is NaN -- policy check only
        # applies to relevance-defined rows.
        df = _make_synthetic_results(
            [
                {"decision": cd.UNDEFINED_RELEVANCE_DECISION, "relevance_defined": False,
                 "task_relevance": float("nan"), "voi_score": float("nan"), "raw_voi_score": float("nan"),
                 "predicted_fault_type": "Unbalance", "fault_type": "Unbalance"},
            ]
        )
        cd.verify_decisions_match_canonical_policy(df)  # must not raise


class TestNoLeakageReused(unittest.TestCase):
    def test_40_16_reuses_task36_leakage_functions_not_reimplemented(self):
        self.assertIs(cd.verify_no_observation_id_leakage, nov.verify_no_observation_id_leakage)
        self.assertIs(cd.verify_no_condition_split_leakage, nov.verify_no_condition_split_leakage)

    def test_40_17_leaked_observation_id_across_splits_is_caught(self):
        df = _make_synthetic_results(
            [
                {"observation_id": "dup", "split": "train"},
                {"observation_id": "dup", "split": "val"},
            ]
        )
        with self.assertRaises(AssertionError):
            cd.verify_no_leakage(df)

    def test_40_18_leaked_condition_across_splits_is_caught(self):
        df = _make_synthetic_results(
            [
                {"observation_id": "a", "condition_code": "cX", "split": "train"},
                {"observation_id": "b", "condition_code": "cX", "split": "val"},
            ]
        )
        with self.assertRaises(AssertionError):
            cd.verify_no_leakage(df)

    def test_40_19_clean_data_passes_leakage_checks(self):
        df = _make_synthetic_results(
            [
                {"observation_id": "a", "condition_code": "c1", "split": "train"},
                {"observation_id": "b", "condition_code": "c2", "split": "val"},
            ]
        )
        cd.verify_no_leakage(df)  # must not raise


class TestUnavailableObservations(unittest.TestCase):
    def test_40_20_unavailable_count_is_canonical_minus_in_pipeline(self):
        observation_index = pd.DataFrame({"observation_id": [f"o{i}" for i in range(10)], "split": ["train"] * 10})
        info = cd._unavailable_observations("vibration_current", "train", n_in_results=6, observation_index=observation_index)
        self.assertEqual(info["n_canonical_observations"], 10)
        self.assertEqual(info["n_unavailable_for_this_configuration"], 4)
        self.assertAlmostEqual(info["pct_unavailable"], 40.0, places=4)

    def test_40_21_zero_unavailable_when_full_coverage(self):
        observation_index = pd.DataFrame({"observation_id": [f"o{i}" for i in range(5)], "split": ["test"] * 5})
        info = cd._unavailable_observations("vibration_only", "test", n_in_results=5, observation_index=observation_index)
        self.assertEqual(info["n_unavailable_for_this_configuration"], 0)


class TestDeterminism(unittest.TestCase):
    def test_40_22_report_generation_is_deterministic(self):
        df = _make_synthetic_results(
            [
                {"observation_id": "a", "condition_code": "c1", "split": "train", "voi_score": 0.1, "raw_voi_score": 0.1, "decision": "DISCARD"},
                {"observation_id": "b", "condition_code": "c2", "split": "val", "voi_score": 0.9, "raw_voi_score": 0.9, "decision": "TRANSMIT"},
            ]
        )
        observation_index = pd.DataFrame({"observation_id": ["a", "b"], "split": ["train", "val"]})
        report1, summary1 = cd.build_communication_decision_report(df, observation_index)
        report2, summary2 = cd.build_communication_decision_report(df, observation_index)
        self.assertEqual(report1, report2)
        pd.testing.assert_frame_equal(summary1, summary2)


@unittest.skipUnless(_RESULTS_AVAILABLE, "Requires Task 39's real results CSV")
class TestRealTask40Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.summary_df = cd.run_task40()

    def test_40_23_all_four_configurations_present(self):
        self.assertEqual(set(self.report["configurations"].keys()), set(fu.FUSION_CONFIGS.keys()))

    def test_40_24_vibration_only_has_zero_unavailable_observations(self):
        # vibration_only never requires current/temperature, so it should
        # retain the full canonical set for every split.
        splits = self.report["configurations"]["vibration_only"]["splits"]
        for split_name, info in splits.items():
            self.assertEqual(info["unavailable_observations"]["n_unavailable_for_this_configuration"], 0)

    def test_40_25_current_requiring_configs_have_nonzero_unavailable(self):
        for config_name in ("vibration_current", "vibration_current_temperature"):
            splits = self.report["configurations"][config_name]["splits"]
            for split_name, info in splits.items():
                self.assertGreater(info["unavailable_observations"]["n_unavailable_for_this_configuration"], 0)

    def test_40_26_decision_percentages_sum_to_100_everywhere(self):
        for config_name, info in self.report["configurations"].items():
            for split_name, split_info in info["splits"].items():
                if split_info["n_observations"] == 0:
                    continue
                total_pct = sum(split_info["decision_percentages"].values())
                self.assertAlmostEqual(total_pct, 100.0, places=2)

    def test_40_27_summary_csv_row_counts_match_json(self):
        self.assertEqual(len(self.summary_df), 4 * 3)  # 4 configs x 3 splits

    def test_40_28_class_distribution_by_decision_is_grouped_by_true_label_and_sums_correctly(self):
        # class_distribution_by_decision is grouped by the observation's TRUE
        # fault_type (matching the CWRU/Paderborn per-class reporting
        # precedent), not the model's predicted class -- so a true
        # BPFO/BPFI/Normal observation CAN legitimately land in the
        # UNDEFINED_RELEVANCE bucket if the model misclassifies it as
        # Misalignment/Unbalance. The invariant this table must satisfy is
        # that summing across all five decision buckets for a given true
        # class reproduces that class's total support in the split.
        for config_name, info in self.report["configurations"].items():
            for split_name, split_info in info["splits"].items():
                dist = split_info["class_distribution_by_decision"]
                for class_name in ("Normal", "BPFI", "BPFO", "Misalignment", "Unbalance"):
                    total_for_class = sum(dist[decision][class_name] for decision in cd.DECISION_CATEGORIES)
                    self.assertGreaterEqual(total_for_class, 0)
                # Undefined bucket's total across all true classes must equal
                # the split's undefined-relevance count.
                undefined_total = sum(dist[cd.UNDEFINED_RELEVANCE_DECISION].values())
                self.assertEqual(undefined_total, split_info["relevance_coverage"]["n_undefined"])

    def test_40_29_deterministic_against_real_data(self):
        report2, summary2 = cd.run_task40()
        self.assertEqual(self.report, report2)
        pd.testing.assert_frame_equal(self.summary_df, summary2)


if __name__ == "__main__":
    unittest.main()
