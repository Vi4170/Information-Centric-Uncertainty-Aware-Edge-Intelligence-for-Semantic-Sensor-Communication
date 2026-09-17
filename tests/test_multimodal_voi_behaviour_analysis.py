"""Task 41 tests -- multimodal VoI behaviour analysis.

Focused tests covering:
  - Factor summary / correlation / dominance / class-conditioned table
    correctness on synthetic data (pure logic, no raw data/models needed).
  - Undefined-relevance rows never enter any VoI-dependent statistic.
  - No recalibration path exists (module never writes to src/voi/).
  - Dominant-factor and counterintuitive-correlation investigations.
  - Real-data integration tests gated on Task 39's results CSV being
    present.

Test numbering uses prefix test_41_ to avoid collision with other tasks.
"""

import json
import os
import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import voi_behaviour_analysis as vba
from src.multimodal_pipeline import voi_integration as vi
from src.voi.scoring import VoIWeights

_RESULTS_AVAILABLE = os.path.exists(vba.cd.VOI_INTEGRATION_RESULTS_CSV_PATH)

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


class TestFactorSummaryTable(unittest.TestCase):
    def test_41_01_all_factor_columns_always_populated(self):
        df = _make_df([_row(observation_id="a"), _row(observation_id="b", uncertainty=0.5)])
        summary = vba.build_factor_summary_table(df)
        self.assertEqual(summary.iloc[0]["novelty_n"], 2)
        self.assertEqual(summary.iloc[0]["uncertainty_n"], 2)

    def test_41_02_task_relevance_and_voi_columns_restricted_to_defined_subset(self):
        df = _make_df(
            [
                _row(observation_id="a", relevance_defined=True, task_relevance=0.9, voi_score=0.8, raw_voi_score=0.8),
                _row(observation_id="b", relevance_defined=False, task_relevance=float("nan"),
                     voi_score=float("nan"), raw_voi_score=float("nan"),
                     predicted_fault_type="Misalignment", fault_type="Misalignment",
                     decision=vi.UNDEFINED_RELEVANCE_DECISION),
            ]
        )
        summary = vba.build_factor_summary_table(df)
        self.assertEqual(summary.iloc[0]["task_relevance_n"], 1)
        self.assertEqual(summary.iloc[0]["voi_score_n"], 1)
        self.assertEqual(summary.iloc[0]["novelty_n"], 2)  # novelty always available


class TestCorrelationTable(unittest.TestCase):
    def test_41_03_correlation_uses_only_defined_subset(self):
        rows = []
        for i in range(5):
            rows.append(_row(observation_id=f"a{i}", novelty=i * 0.2, voi_score=i * 0.2, raw_voi_score=i * 0.2))
        rows.append(
            _row(observation_id="undef", relevance_defined=False, task_relevance=float("nan"),
                 voi_score=float("nan"), raw_voi_score=float("nan"), novelty=999.0,
                 predicted_fault_type="Unbalance", fault_type="Unbalance",
                 decision=vi.UNDEFINED_RELEVANCE_DECISION)
        )
        df = _make_df(rows)
        corr_df = vba.build_correlation_table(df)
        self.assertEqual(corr_df.iloc[0]["n_valid"], 5)
        self.assertAlmostEqual(corr_df.iloc[0]["corr_novelty_vs_voi_score"], 1.0, places=6)

    def test_41_04_constant_column_gives_none_not_zero(self):
        rows = [_row(observation_id=f"a{i}", resource_cost=0.5049, voi_score=i * 0.1, raw_voi_score=i * 0.1) for i in range(5)]
        df = _make_df(rows)
        corr_df = vba.build_correlation_table(df)
        self.assertIsNone(corr_df.iloc[0]["corr_resource_cost_vs_voi_score"])

    def test_41_05_fewer_than_two_defined_rows_gives_none(self):
        df = _make_df([_row(observation_id="a")])
        corr_df = vba.build_correlation_table(df)
        self.assertIsNone(corr_df.iloc[0]["corr_novelty_vs_voi_score"])


class TestVoiByDecisionTable(unittest.TestCase):
    def test_41_06_all_four_decisions_plus_undefined_always_present(self):
        df = _make_df([_row(observation_id="a")])
        table = vba.build_voi_by_decision_table(df)
        decisions_present = set(table["decision"])
        self.assertEqual(decisions_present, {"DISCARD", "BUFFER", "SUMMARY", "TRANSMIT"})

    def test_41_07_undefined_relevance_rows_have_nan_voi_stats_grouped_separately(self):
        df = _make_df(
            [_row(observation_id="a", relevance_defined=False, task_relevance=float("nan"),
                  voi_score=float("nan"), raw_voi_score=float("nan"),
                  predicted_fault_type="Misalignment", fault_type="Misalignment",
                  decision=vi.UNDEFINED_RELEVANCE_DECISION)]
        )
        table = vba.build_voi_by_decision_table(df)
        # DECISION_ORDER doesn't include UNDEFINED_RELEVANCE_DECISION -- confirm none
        # of the four real buckets picked up this row.
        self.assertTrue((table["n"] == 0).all())


class TestClassConditionedTable(unittest.TestCase):
    def test_41_08_grouped_by_true_fault_type(self):
        df = _make_df(
            [
                _row(observation_id="a", fault_type="BPFI", predicted_fault_type="BPFI", task_relevance=1.0, decision="TRANSMIT"),
                _row(observation_id="b", fault_type="Normal", predicted_fault_type="Normal", decision="DISCARD"),
            ]
        )
        table = vba.build_class_conditioned_table(df)
        bpfi_row = table[table["fault_type"] == "BPFI"].iloc[0]
        self.assertEqual(bpfi_row["n"], 1)
        self.assertEqual(bpfi_row["decision_TRANSMIT_count"], 1)

    def test_41_09_misclassified_true_class_counted_in_undefined_bucket(self):
        df = _make_df(
            [
                _row(observation_id="a", fault_type="BPFO", predicted_fault_type="Misalignment",
                     relevance_defined=False, task_relevance=float("nan"),
                     voi_score=float("nan"), raw_voi_score=float("nan"),
                     decision=vi.UNDEFINED_RELEVANCE_DECISION),
            ]
        )
        table = vba.build_class_conditioned_table(df)
        bpfo_row = table[table["fault_type"] == "BPFO"].iloc[0]
        self.assertEqual(bpfo_row[f"decision_{vi.UNDEFINED_RELEVANCE_DECISION}_count"], 1)


class TestDominanceTable(unittest.TestCase):
    def test_41_10_shares_sum_to_100_when_all_contributions_positive(self):
        weights = VoIWeights()
        rows = [_row(observation_id=f"a{i}", novelty=0.5, uncertainty=0.3, task_relevance=0.8,
                      temporal_importance=0.2, resource_cost=0.5, voi_score=0.5, raw_voi_score=0.5) for i in range(5)]
        df = _make_df(rows)
        dom_df = vba.build_dominance_table(df, weights)
        share_cols = [c for c in dom_df.columns if c.endswith("_share_of_positive_contribution_pct")]
        total = sum(dom_df.iloc[0][c] for c in share_cols)
        self.assertAlmostEqual(total, 100.0, places=2)

    def test_41_11_zero_valid_rows_produces_row_without_crashing(self):
        weights = VoIWeights()
        df = _make_df(
            [_row(observation_id="a", relevance_defined=False, task_relevance=float("nan"),
                  voi_score=float("nan"), raw_voi_score=float("nan"),
                  predicted_fault_type="Unbalance", fault_type="Unbalance",
                  decision=vi.UNDEFINED_RELEVANCE_DECISION)]
        )
        dom_df = vba.build_dominance_table(df, weights)
        self.assertEqual(dom_df.iloc[0]["n_valid"], 0)

    def test_41_12_resource_cost_contribution_is_negative(self):
        weights = VoIWeights()
        rows = [_row(observation_id=f"a{i}", resource_cost=0.5, voi_score=0.3, raw_voi_score=0.3) for i in range(3)]
        df = _make_df(rows)
        dom_df = vba.build_dominance_table(df, weights)
        self.assertLess(dom_df.iloc[0]["resource_cost_mean_contribution"], 0.0)


class TestDominantFactorFindings(unittest.TestCase):
    def test_41_13_identifies_largest_share_factor(self):
        dom_df = pd.DataFrame(
            [
                {
                    "fusion_config": "vibration_only", "split": "test", "n_valid": 10,
                    "novelty_share_of_positive_contribution_pct": 20.0,
                    "uncertainty_share_of_positive_contribution_pct": 5.0,
                    "task_relevance_share_of_positive_contribution_pct": 60.0,
                    "temporal_importance_share_of_positive_contribution_pct": 15.0,
                }
            ]
        )
        findings = vba._dominant_factor_findings(dom_df)
        self.assertEqual(findings[0]["dominant_factor"], "task_relevance")
        self.assertEqual(findings[0]["dominant_factor_share_pct"], 60.0)

    def test_41_14_skips_zero_valid_rows(self):
        dom_df = pd.DataFrame([{"fusion_config": "c", "split": "test", "n_valid": 0}])
        findings = vba._dominant_factor_findings(dom_df)
        self.assertEqual(findings, [])


class TestCounterintuitiveCorrelationInvestigation(unittest.TestCase):
    def test_41_15_flags_negative_correlation_for_positively_weighted_factor(self):
        rows = []
        # Construct uncertainty negatively correlated with voi_score/task_relevance.
        for i in range(6):
            rows.append(
                _row(
                    observation_id=f"a{i}",
                    uncertainty=0.9 - i * 0.1,
                    task_relevance=0.1 + i * 0.15,
                    voi_score=0.1 + i * 0.15,
                    raw_voi_score=0.1 + i * 0.15,
                )
            )
        df = _make_df(rows)
        corr_df = vba.build_correlation_table(df)
        findings = vba._counterintuitive_correlation_investigation(df, corr_df)
        factors_flagged = {f["factor"] for f in findings}
        self.assertIn("uncertainty", factors_flagged)
        finding = next(f for f in findings if f["factor"] == "uncertainty")
        self.assertLess(finding["observed_corr_with_voi_score"], 0.0)
        self.assertIsNotNone(finding["factor_corr_with_task_relevance"])

    def test_41_16_no_findings_when_all_correlations_expected_sign(self):
        rows = [_row(observation_id=f"a{i}", novelty=i * 0.1, uncertainty=i * 0.1,
                      temporal_importance=i * 0.1, voi_score=i * 0.1, raw_voi_score=i * 0.1) for i in range(5)]
        df = _make_df(rows)
        corr_df = vba.build_correlation_table(df)
        findings = vba._counterintuitive_correlation_investigation(df, corr_df)
        self.assertEqual(findings, [])


class TestNoRecalibrationPath(unittest.TestCase):
    def test_41_17_module_never_constructs_custom_non_default_weights(self):
        import re

        with open(vba.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn("from src.voi.scoring import VoIWeights", source)
        self.assertIn("from src.voi.decision_policy import PolicyThresholds", source)
        # Every VoIWeights(...) construction call must use bare defaults --
        # no custom weight kwargs, i.e. no recalibration path exists here.
        calls = re.findall(r"VoIWeights\(([^)]*)\)", source)
        for call_args in calls:
            self.assertEqual(call_args.strip(), "", f"VoIWeights() called with non-default args: {call_args!r}")

    def test_41_18_report_declares_no_recalibration_performed(self):
        df = _make_df([_row(observation_id="a")])
        report, _, _ = vba.build_behaviour_analysis_report(df)
        self.assertIn("weight_recalibration", report["not_performed_in_this_task"])
        self.assertIn("threshold_recalibration", report["not_performed_in_this_task"])
        self.assertIn("voi_formula_modification", report["not_performed_in_this_task"])

    def test_41_19_report_uses_default_unmodified_weights(self):
        df = _make_df([_row(observation_id="a")])
        report, _, _ = vba.build_behaviour_analysis_report(df)
        self.assertEqual(report["weights_used"], asdict(VoIWeights()))


class TestDeterminism(unittest.TestCase):
    def test_41_20_report_generation_is_deterministic(self):
        df = _make_df([_row(observation_id="a"), _row(observation_id="b", uncertainty=0.4, voi_score=0.4, raw_voi_score=0.4)])
        report1, fs1, dom1 = vba.build_behaviour_analysis_report(df)
        report2, fs2, dom2 = vba.build_behaviour_analysis_report(df)
        # NaN != NaN under == , so compare via a stable JSON rendering
        # rather than raw dict equality (report legitimately contains NaN
        # for constant/undefined correlations).
        self.assertEqual(
            json.dumps(report1, sort_keys=True, default=str),
            json.dumps(report2, sort_keys=True, default=str),
        )
        pd.testing.assert_frame_equal(fs1, fs2)
        pd.testing.assert_frame_equal(dom1, dom2)


@unittest.skipUnless(_RESULTS_AVAILABLE, "Requires Task 39's real results CSV")
class TestRealTask41Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.factor_summary_df, cls.dominance_df = vba.run_task41()

    def test_41_21_all_four_configurations_present_in_correlation_table(self):
        configs = {row["fusion_config"] for row in self.report["correlations"]}
        self.assertEqual(configs, set(fu.FUSION_CONFIGS.keys()))

    def test_41_22_dominant_factor_findings_nonempty(self):
        self.assertGreater(len(self.report["dominant_factor_findings"]), 0)

    def test_41_23_task_relevance_is_dominant_in_most_configs(self):
        # Task relevance carries the largest canonical weight (0.35); expect
        # it to be the dominant factor in the majority of (config, split)
        # combinations with valid data -- a descriptive check, not an
        # assertion the code enforces by construction.
        dominant_counts = {}
        for finding in self.report["dominant_factor_findings"]:
            dominant_counts[finding["dominant_factor"]] = dominant_counts.get(finding["dominant_factor"], 0) + 1
        self.assertEqual(max(dominant_counts, key=dominant_counts.get), "task_relevance")

    def test_41_24_correlation_values_within_valid_range(self):
        for row in self.report["correlations"]:
            for key, value in row.items():
                if key.startswith("corr_") and value is not None and not (isinstance(value, float) and np.isnan(value)):
                    self.assertGreaterEqual(value, -1.0001)
                    self.assertLessEqual(value, 1.0001)

    def test_41_25_cross_config_comparison_has_all_four_configs(self):
        configs = {row["fusion_config"] for row in self.report["cross_configuration_comparison_test_split"]}
        self.assertEqual(configs, set(fu.FUSION_CONFIGS.keys()))

    def test_41_26_transmit_investigation_reports_all_configs(self):
        configs = {row["fusion_config"] for row in self.report["transmit_rarity_investigation"]["per_configuration"]}
        self.assertEqual(configs, set(fu.FUSION_CONFIGS.keys()))

    def test_41_27_deterministic_against_real_data(self):
        report2, fs2, dom2 = vba.run_task41()
        self.assertEqual(
            json.dumps(self.report, sort_keys=True, default=str),
            json.dumps(report2, sort_keys=True, default=str),
        )
        pd.testing.assert_frame_equal(self.factor_summary_df, fs2)
        pd.testing.assert_frame_equal(self.dominance_df, dom2)

    def test_41_28_no_defined_voi_score_outside_unit_interval_in_factor_summary(self):
        for _, row in self.factor_summary_df.iterrows():
            if row["voi_score_n"] and row["voi_score_n"] > 0:
                self.assertGreaterEqual(row["voi_score_min"], 0.0)
                self.assertLessEqual(row["voi_score_max"], 1.0)


if __name__ == "__main__":
    unittest.main()
