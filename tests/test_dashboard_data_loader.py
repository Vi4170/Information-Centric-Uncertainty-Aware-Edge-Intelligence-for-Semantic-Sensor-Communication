import unittest

from dashboard import data_loader as dl


class TestDashboardDataLoader(unittest.TestCase):
    def test_load_json_missing_file_returns_none(self):
        self.assertIsNone(dl._load_json("nonexistent_directory", "nonexistent_file.json"))

    def test_load_csv_missing_file_returns_none(self):
        self.assertIsNone(dl._load_csv("nonexistent_directory", "nonexistent_file.csv"))

    def test_figure_path_missing_file_returns_none(self):
        self.assertIsNone(dl.figure_path("nonexistent_directory", "nonexistent_figure.png"))

    def test_pipeline_stage_status_keys_and_types(self):
        status = dl.get_pipeline_stage_status()
        self.assertGreater(len(status), 0)
        for key, value in status.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, bool)

    def test_cwru_dataset_info_available(self):
        info = dl.get_cwru_dataset_info()
        self.assertEqual(info["status"], dl.STATUS_AVAILABLE)
        self.assertEqual(info["summary"]["window_size"], 2048)
        self.assertIn("experiments performed", info["experiment_status"])

    def test_ims_dataset_info_reports_no_experiment_performed(self):
        info = dl.get_ims_dataset_info()
        self.assertEqual(info["status"], dl.STATUS_AVAILABLE)
        self.assertIn("not yet performed", info["experiment_status"])
        self.assertIn("1st_test", info["summary"]["runs"])

    def test_paderborn_dataset_info_reports_no_experiment_performed(self):
        info = dl.get_paderborn_dataset_info()
        self.assertEqual(info["status"], dl.STATUS_AVAILABLE)
        self.assertIn("not yet performed", info["experiment_status"])
        self.assertEqual(info["summary"]["n_healthy_states"], 6)
        self.assertEqual(info["summary"]["n_damaged_states"], 26)

    def test_canonical_cnn_results_match_stored_artifact(self):
        result = dl.get_canonical_cnn_results()
        self.assertEqual(result["status"], dl.STATUS_AVAILABLE)
        self.assertAlmostEqual(float(result["evaluation_summary"].iloc[0]["accuracy"]), 1.0)

    def test_legacy_edge_cloud_results_distinct_from_canonical(self):
        result = dl.get_legacy_edge_cloud_results()
        self.assertEqual(result["status"], dl.STATUS_AVAILABLE)
        self.assertLess(result["baseline_cnn"]["accuracy"], 1.0)
        self.assertEqual(result["baseline_cnn"]["test_samples"], 1043)

    def test_novelty_results_config_matches_source_module(self):
        from src.novelty.config import EMBEDDING_DIM

        result = dl.get_novelty_results()
        self.assertEqual(result["status"], dl.STATUS_AVAILABLE)
        self.assertEqual(result["config"]["embedding_dim"], EMBEDDING_DIM)

    def test_relevance_class_map_matches_source_module(self):
        from src.relevance.config import CLASS_RELEVANCE_MAP

        result = dl.get_relevance_results()
        self.assertEqual(result["config"]["class_relevance_map"], CLASS_RELEVANCE_MAP)

    def test_voi_engine_config_matches_source_defaults(self):
        cfg = dl.get_voi_engine_config()
        self.assertAlmostEqual(cfg["weights"]["novelty"], 0.30)
        self.assertAlmostEqual(cfg["weights"]["uncertainty"], 0.05)
        self.assertAlmostEqual(cfg["weights"]["task_relevance"], 0.35)
        self.assertAlmostEqual(cfg["thresholds"]["discard_max"], 0.25)
        self.assertAlmostEqual(cfg["thresholds"]["summary_max"], 0.70)

    def test_continual_learning_results_structure(self):
        result = dl.get_continual_learning_results()
        self.assertEqual(result["status"], dl.STATUS_AVAILABLE)
        self.assertIn("admission_result", result["raw"])
        self.assertIn("detection_result", result["raw"])

    def test_test_suite_summary_lists_test_files(self):
        summary = dl.get_test_suite_summary()
        self.assertEqual(summary["status"], dl.STATUS_AVAILABLE)
        self.assertIn("test_voi.py", summary["test_files"])


if __name__ == "__main__":
    unittest.main()
