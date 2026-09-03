import os
import unittest
from datetime import datetime

import numpy as np

from src.ims_pipeline.preprocessing import (
    RUN_IDS,
    WINDOW_SIZE,
    WINDOWS_PER_FILE,
    RAW_DATA_DIR,
    apply_normalization,
    assign_split_for_index,
    build_stream_metadata,
    compute_chronological_split_boundaries,
    discover_all_runs_summary,
    discover_run_files,
    fit_initial_normalization,
    load_stream_windows,
    parse_ims_filename_timestamp,
    verify_chronological_split_order,
    verify_no_test_leakage_into_normalization,
    verify_observation_id_uniqueness,
    verify_split_disjoint,
)

_RAW_DATA_AVAILABLE = all(
    os.path.isdir(os.path.join(RAW_DATA_DIR, sub))
    for sub in ("1st_test", "2nd_test", os.path.join("4th_test", "txt"))
)


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "Extracted IMS raw data not present under data/raw/ims/")
class TestImsPreprocessing(unittest.TestCase):
    def test_01_deterministic_chronological_ordering(self):
        for run_id in RUN_IDS:
            files_a = discover_run_files(run_id)
            files_b = discover_run_files(run_id)
            self.assertEqual(files_a, files_b)
            timestamps = [parse_ims_filename_timestamp(f) for f in files_a]
            self.assertTrue(all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1)))

    def test_02_chronological_split_correctness(self):
        boundaries = compute_chronological_split_boundaries(984)
        self.assertEqual(boundaries["initial"], (0, 196))
        self.assertEqual(boundaries["adaptation"], (196, 788))
        self.assertEqual(boundaries["test"], (788, 984))
        self.assertEqual(assign_split_for_index(0, boundaries), "initial")
        self.assertEqual(assign_split_for_index(195, boundaries), "initial")
        self.assertEqual(assign_split_for_index(196, boundaries), "adaptation")
        self.assertEqual(assign_split_for_index(787, boundaries), "adaptation")
        self.assertEqual(assign_split_for_index(788, boundaries), "test")
        self.assertEqual(assign_split_for_index(983, boundaries), "test")
        with self.assertRaises(ValueError):
            compute_chronological_split_boundaries(2)

    def test_03_metadata_correctness_and_expected_window_generation(self):
        metadata = build_stream_metadata("2nd_test", 1, 0)
        n_files = len(discover_run_files("2nd_test"))
        self.assertEqual(len(metadata), n_files * WINDOWS_PER_FILE)
        self.assertEqual(set(metadata["run_id"]), {"2nd_test"})
        self.assertEqual(set(metadata["bearing_id"]), {1})
        self.assertEqual(set(metadata["channel_index"]), {0})
        self.assertEqual(set(metadata["label_available"]), {False})
        self.assertTrue((metadata["condition_label"].isna()).all())
        self.assertEqual(metadata.loc[0, "chronological_order_index"], 0)
        self.assertEqual(metadata.loc[0, "window_index"], 0)
        first_file_rows = metadata[metadata["chronological_order_index"] == 0]
        self.assertEqual(sorted(first_file_rows["window_index"].tolist()), list(range(WINDOWS_PER_FILE)))

    def test_04_repeat_run_produces_equivalent_metadata(self):
        meta_a = build_stream_metadata("2nd_test", 2, 0)
        meta_b = build_stream_metadata("2nd_test", 2, 0)
        self.assertTrue(meta_a.equals(meta_b))

    def test_05_no_train_adaptation_test_overlap(self):
        metadata = build_stream_metadata("2nd_test", 1, 0)
        overlaps = verify_split_disjoint(metadata)
        self.assertEqual(overlaps["initial_adaptation_overlap"], 0)
        self.assertEqual(overlaps["initial_test_overlap"], 0)
        self.assertEqual(overlaps["adaptation_test_overlap"], 0)
        self.assertTrue(verify_chronological_split_order(metadata))
        self.assertTrue(verify_observation_id_uniqueness(metadata))

    def test_06_permanent_test_isolation_and_no_future_to_past_leakage(self):
        metadata = build_stream_metadata("1st_test", 4, 0)
        n_files = len(discover_run_files("1st_test"))
        test_rows = metadata[metadata["split"] == "test"]
        initial_rows = metadata[metadata["split"] == "initial"]
        self.assertTrue((test_rows["chronological_order_index"] >= initial_rows["chronological_order_index"].max()).all())
        self.assertEqual(int(test_rows["chronological_order_index"].max()), n_files - 1)
        self.assertEqual(verify_no_test_leakage_into_normalization(metadata), 0)

    def test_07_deterministic_preprocessing_and_expected_window_shape(self):
        X_a, meta_a = load_stream_windows("2nd_test", 3, 0, file_indices=[0, 1, 5, 500])
        X_b, meta_b = load_stream_windows("2nd_test", 3, 0, file_indices=[0, 1, 5, 500])
        self.assertTrue(np.array_equal(X_a, X_b))
        self.assertTrue(meta_a.equals(meta_b))
        self.assertEqual(X_a.shape, (4 * WINDOWS_PER_FILE, WINDOW_SIZE, 1))
        self.assertTrue(np.isfinite(X_a).all())

    def test_08_normalization_fitted_only_from_initial_data(self):
        boundaries = compute_chronological_split_boundaries(len(discover_run_files("2nd_test")))
        initial_lo, initial_hi = boundaries["initial"]
        test_lo, test_hi = boundaries["test"]
        file_indices = [initial_lo, initial_lo + 1, initial_hi - 1, test_lo, test_hi - 1]
        X, metadata = load_stream_windows("2nd_test", 1, 0, file_indices=file_indices)

        mean, std = fit_initial_normalization(X, metadata)

        initial_mask = (metadata["split"] == "initial").to_numpy()
        expected_mean = float(np.mean(X[initial_mask]))
        expected_std = float(np.std(X[initial_mask]))
        self.assertAlmostEqual(mean, expected_mean, places=6)
        self.assertAlmostEqual(std, expected_std, places=6)

        test_only_metadata = metadata[metadata["split"] == "test"]
        with self.assertRaises(ValueError):
            fit_initial_normalization(X, test_only_metadata.reset_index(drop=True))

        normalized = apply_normalization(X, mean, std)
        self.assertEqual(normalized.shape, X.shape)
        self.assertAlmostEqual(float(np.mean(normalized[initial_mask])), 0.0, places=4)

    def test_09_no_future_observations_used_for_normalization_statistics(self):
        boundaries = compute_chronological_split_boundaries(len(discover_run_files("1st_test")))
        initial_lo, initial_hi = boundaries["initial"]
        file_indices = list(range(initial_lo, min(initial_lo + 5, initial_hi)))
        X_initial_only, meta_initial_only = load_stream_windows("1st_test", 3, 0, file_indices=file_indices)
        mean_initial_only, std_initial_only = fit_initial_normalization(X_initial_only, meta_initial_only)

        file_indices_with_future = file_indices + [boundaries["test"][0], boundaries["test"][1] - 1]
        X_with_future, meta_with_future = load_stream_windows("1st_test", 3, 0, file_indices=file_indices_with_future)
        mean_with_future, std_with_future = fit_initial_normalization(X_with_future, meta_with_future)

        self.assertAlmostEqual(mean_initial_only, mean_with_future, places=6)
        self.assertAlmostEqual(std_initial_only, std_with_future, places=6)

    def test_10_discover_all_runs_summary_is_deterministic_and_leakage_safe(self):
        summary_a = discover_all_runs_summary()
        summary_b = discover_all_runs_summary()
        self.assertEqual(summary_a, summary_b)
        for run_id in RUN_IDS:
            run_summary = summary_a["runs"][run_id]
            boundaries = run_summary["split_file_boundaries"]
            self.assertLessEqual(boundaries["initial"][1], boundaries["adaptation"][0])
            self.assertLessEqual(boundaries["adaptation"][1], boundaries["test"][0])
            self.assertEqual(boundaries["test"][1], run_summary["n_files"])

    def test_11_run_isolation_across_streams(self):
        meta_1 = build_stream_metadata("1st_test", 1, 0)
        meta_2 = build_stream_metadata("2nd_test", 1, 0)
        combined_ids = set(meta_1["observation_id"]) | set(meta_2["observation_id"])
        self.assertEqual(len(combined_ids), len(meta_1) + len(meta_2))
        self.assertTrue((meta_1["run_id"] == "1st_test").all())
        self.assertTrue((meta_2["run_id"] == "2nd_test").all())

    def test_12_documented_set1_and_set2_file_counts(self):
        self.assertEqual(len(discover_run_files("1st_test")), 2156)
        self.assertEqual(len(discover_run_files("2nd_test")), 984)

    def test_13_corrected_set3_interpretation_matches_documented_boundary(self):
        files = discover_run_files("3rd_test")
        self.assertEqual(len(files), 4448)
        self.assertEqual(parse_ims_filename_timestamp(files[0]), datetime(2004, 3, 4, 9, 27, 46))
        self.assertEqual(parse_ims_filename_timestamp(files[-1]), datetime(2004, 4, 4, 19, 1, 57))

    def test_14_extra_files_represented_separately_as_4th_test(self):
        extension_files = discover_run_files("4th_test")
        self.assertEqual(len(extension_files), 1876)
        self.assertEqual(parse_ims_filename_timestamp(extension_files[0]), datetime(2004, 4, 4, 19, 11, 57))
        self.assertEqual(parse_ims_filename_timestamp(extension_files[-1]), datetime(2004, 4, 18, 2, 42, 55))

        set3_files = discover_run_files("3rd_test")
        self.assertEqual(set(set3_files) & set(extension_files), set())
        self.assertEqual(len(set3_files) + len(extension_files), 6324)

        gap_minutes = (
            parse_ims_filename_timestamp(extension_files[0]) - parse_ims_filename_timestamp(set3_files[-1])
        ).total_seconds() / 60
        self.assertAlmostEqual(gap_minutes, 10.0, places=6)

    def test_15_run_failure_descriptions_are_run_level_and_distinct(self):
        summary = discover_all_runs_summary()
        set3_description = summary["runs"]["3rd_test"]["failure_description"]
        extension_description = summary["runs"]["4th_test"]["failure_description"]
        self.assertIn("outer race failure in bearing 3", set3_description)
        self.assertNotIn("outer race failure in bearing 3", extension_description)
        self.assertIn("4,448", set3_description)

    def test_16_provenance_preserved_exactly_in_generated_metadata(self):
        metadata = build_stream_metadata("3rd_test", 3, 0)
        raw_files = discover_run_files("3rd_test")
        self.assertEqual(sorted(metadata["source_file"].unique()), sorted(raw_files))
        for _, row in metadata[metadata["chronological_order_index"] == 0].iterrows():
            self.assertEqual(row["source_file"], raw_files[0])
            self.assertEqual(row["file_timestamp"], parse_ims_filename_timestamp(raw_files[0]).isoformat())
        for _, row in metadata[metadata["chronological_order_index"] == len(raw_files) - 1].iterrows():
            self.assertEqual(row["source_file"], raw_files[-1])
            self.assertEqual(row["file_timestamp"], parse_ims_filename_timestamp(raw_files[-1]).isoformat())

    def test_17_ims_not_modeled_as_uniformly_sampled(self):
        files = discover_run_files("1st_test")
        timestamps = [parse_ims_filename_timestamp(f) for f in files]
        gaps = {round((timestamps[i + 1] - timestamps[i]).total_seconds() / 60) for i in range(50)}
        self.assertIn(5, gaps)
        self.assertIn(10, gaps)


if __name__ == "__main__":
    unittest.main()
