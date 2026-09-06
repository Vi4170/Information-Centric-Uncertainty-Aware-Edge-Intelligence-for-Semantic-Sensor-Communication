import json
import os
import unittest

from src.evaluation.cross_dataset_analysis import (
    OUTPUT_PATH,
    STATUS_DERIVED,
    STATUS_MEASURED,
    STATUS_NOT_AVAILABLE,
    STATUS_NOT_SCIENTIFICALLY_VALID,
    build_evidence_matrix,
)
from src.voi.scoring import VoIWeights
from src.voi.decision_policy import PolicyThresholds

VALID_STATUSES = {STATUS_MEASURED, STATUS_DERIVED, STATUS_NOT_AVAILABLE, STATUS_NOT_SCIENTIFICALLY_VALID}

REQUIRED_FIELDS = [
    "dataset", "available_experiment_type", "n_observations",
    "cnn_availability", "classification_availability", "novelty_availability",
    "uncertainty_availability", "relevance_availability", "temporal_importance_availability",
    "communication_cost_availability", "full_voi_availability", "communication_decision_availability",
    "continual_learning_evidence", "run_to_failure_evidence", "main_finding", "main_limitation",
]

AVAILABILITY_FIELDS = [
    "cnn_availability", "classification_availability", "novelty_availability",
    "uncertainty_availability", "relevance_availability", "temporal_importance_availability",
    "communication_cost_availability", "full_voi_availability", "communication_decision_availability",
    "continual_learning_evidence", "run_to_failure_evidence",
]


class TestEvidenceMatrixStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = build_evidence_matrix()
        cls.by_dataset = {row["dataset"]: row for row in cls.matrix}

    def test_five_datasets_present(self):
        self.assertEqual(
            set(self.by_dataset.keys()),
            {"cwru", "paderborn", "ims", "xjtu_sy", "mimii"},
        )

    def test_every_row_has_required_fields(self):
        for row in self.matrix:
            for field in REQUIRED_FIELDS:
                self.assertIn(field, row, f"{row.get('dataset')} missing field '{field}'")

    def test_every_availability_field_has_valid_status(self):
        for row in self.matrix:
            for field in AVAILABILITY_FIELDS:
                value = row[field]
                self.assertIn("status", value, f"{row['dataset']}.{field} missing 'status'")
                self.assertIn(value["status"], VALID_STATUSES, f"{row['dataset']}.{field} has invalid status '{value['status']}'")

    def test_main_finding_and_limitation_are_non_empty_strings(self):
        for row in self.matrix:
            self.assertIsInstance(row["main_finding"], str)
            self.assertGreater(len(row["main_finding"]), 0)
            self.assertIsInstance(row["main_limitation"], str)
            self.assertGreater(len(row["main_limitation"]), 0)


class TestNoUnsupportedClaimsEncodedAsMeasured(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.by_dataset = {row["dataset"]: row for row in build_evidence_matrix()}

    def test_ims_uncertainty_and_relevance_not_measured(self):
        ims = self.by_dataset["ims"]
        self.assertEqual(ims["uncertainty_availability"]["status"], STATUS_NOT_SCIENTIFICALLY_VALID)
        self.assertEqual(ims["relevance_availability"]["status"], STATUS_NOT_SCIENTIFICALLY_VALID)

    def test_ims_full_voi_not_measured(self):
        ims = self.by_dataset["ims"]
        self.assertEqual(ims["full_voi_availability"]["status"], STATUS_NOT_AVAILABLE)
        self.assertEqual(ims["communication_decision_availability"]["status"], STATUS_NOT_AVAILABLE)

    def test_ims_classification_not_scientifically_valid(self):
        self.assertEqual(
            self.by_dataset["ims"]["classification_availability"]["status"],
            STATUS_NOT_SCIENTIFICALLY_VALID,
        )

    def test_xjtu_and_mimii_have_no_measured_fields(self):
        for dataset_name in ("xjtu_sy", "mimii"):
            row = self.by_dataset[dataset_name]
            for field in AVAILABILITY_FIELDS:
                self.assertNotEqual(
                    row[field]["status"], STATUS_MEASURED,
                    f"{dataset_name}.{field} incorrectly marked measured",
                )

    def test_cwru_continual_learning_caveat_present_and_not_unseen_class(self):
        continual = self.by_dataset["cwru"]["continual_learning_evidence"]
        if continual["status"] == STATUS_MEASURED:
            self.assertIn("caveat", continual)
            self.assertTrue(continual["new_condition_already_in_trained_cnn_label_space"])

    def test_static_datasets_have_no_run_to_failure_evidence(self):
        for dataset_name in ("cwru", "paderborn"):
            self.assertEqual(
                self.by_dataset[dataset_name]["run_to_failure_evidence"]["status"],
                STATUS_NOT_AVAILABLE,
            )

    def test_ims_run_to_failure_is_measured(self):
        self.assertEqual(
            self.by_dataset["ims"]["run_to_failure_evidence"]["status"],
            STATUS_MEASURED,
        )


class TestCrossDatasetConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = build_evidence_matrix()

    def test_full_voi_measured_implies_all_five_factors_measured(self):
        for row in self.matrix:
            if row["full_voi_availability"]["status"] == STATUS_MEASURED:
                for field in (
                    "novelty_availability", "uncertainty_availability", "relevance_availability",
                    "temporal_importance_availability", "communication_cost_availability",
                ):
                    self.assertEqual(
                        row[field]["status"], STATUS_MEASURED,
                        f"{row['dataset']}: full_voi measured but {field} is not",
                    )

    def test_communication_decision_measured_implies_full_voi_measured(self):
        for row in self.matrix:
            if row["communication_decision_availability"]["status"] == STATUS_MEASURED:
                self.assertEqual(row["full_voi_availability"]["status"], STATUS_MEASURED)

    def test_cwru_and_paderborn_are_full_voi_experiments(self):
        by_dataset = {row["dataset"]: row for row in self.matrix}
        for dataset_name in ("cwru", "paderborn"):
            self.assertEqual(by_dataset[dataset_name]["full_voi_availability"]["status"], STATUS_MEASURED)


class TestCanonicalVoiConfigurationUnchanged(unittest.TestCase):
    def test_weights_and_thresholds_reflect_live_production_config(self):
        from src.evaluation.cross_dataset_analysis import _canonical_voi_configuration

        config = _canonical_voi_configuration()
        weights = VoIWeights()
        thresholds = PolicyThresholds()

        self.assertEqual(config["weights"]["novelty"], weights.novelty)
        self.assertEqual(config["weights"]["uncertainty"], weights.uncertainty)
        self.assertEqual(config["weights"]["task_relevance"], weights.task_relevance)
        self.assertEqual(config["weights"]["temporal_importance"], weights.temporal_importance)
        self.assertEqual(config["weights"]["resource_cost"], weights.resource_cost)
        self.assertEqual(config["thresholds"]["discard_max"], thresholds.discard_max)
        self.assertEqual(config["thresholds"]["buffer_max"], thresholds.buffer_max)
        self.assertEqual(config["thresholds"]["summary_max"], thresholds.summary_max)

    def test_configuration_matches_task_specified_values(self):
        weights = VoIWeights()
        self.assertEqual(weights.novelty, 0.30)
        self.assertEqual(weights.uncertainty, 0.05)
        self.assertEqual(weights.task_relevance, 0.35)
        self.assertEqual(weights.temporal_importance, 0.20)
        self.assertEqual(weights.resource_cost, 0.10)


class TestOutputArtifact(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(OUTPUT_PATH), "cross-dataset analysis has not been generated yet")
    def test_output_file_matches_module_output(self):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["task"], 30)
        self.assertEqual(len(saved["evidence_matrix"]), 5)
        self.assertIn("claims_that_must_not_be_made", saved)
        self.assertIn("research_gaps", saved)
        self.assertGreater(len(saved["claims_that_must_not_be_made"]), 0)
        self.assertGreater(len(saved["cross_dataset_patterns"]), 0)


if __name__ == "__main__":
    unittest.main()
