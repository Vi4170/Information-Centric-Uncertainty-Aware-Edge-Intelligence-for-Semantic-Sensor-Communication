"""Unit test suite for the CWRU Bearing Dataset preprocessing & windowing pipeline."""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.io import savemat

from src.cwru_pipeline.config import WINDOW_SIZE
from src.cwru_pipeline.dataset import CWRUFileMetadata, load_cwru_mat_file
from src.cwru_pipeline.preprocessing import (
    create_leakage_safe_split,
    fit_train_normalization,
    normalize_signal,
    run_cwru_preprocessing_pipeline,
    validate_signal,
    window_signal,
)


class TestCWRUPreprocessing(unittest.TestCase):
    """Test suite covering CWRU loader, signal validation, leakage prevention, indexing, and reproducibility."""

    def test_01_signal_validation(self):
        """1. Test signal validation accepts valid signals and rejects invalid signals."""
        valid_sig = np.random.randn(5000).astype(np.float32)
        res = validate_signal(valid_sig, "test_file")
        self.assertEqual(len(res), 5000)

        # Empty array
        with self.assertRaises(ValueError):
            validate_signal(np.array([]), "empty_file")

        # NaN
        nan_sig = valid_sig.copy()
        nan_sig[100] = np.nan
        with self.assertRaises(ValueError):
            validate_signal(nan_sig, "nan_file")

        # Inf
        inf_sig = valid_sig.copy()
        inf_sig[100] = np.inf
        with self.assertRaises(ValueError):
            validate_signal(inf_sig, "inf_file")

        # Short signal < 2048
        short_sig = np.random.randn(1000).astype(np.float32)
        with self.assertRaises(ValueError):
            validate_signal(short_sig, "short_file")

    def test_02_sampling_rate_and_de_channel_extraction(self):
        """2. Test DE vibration channel extraction and metadata parsing from .mat file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mat_path = os.path.join(tmp_dir, "105.mat")
            de_data = np.random.randn(4000).astype(np.float64)
            savemat(mat_path, {"X105_DE_time": de_data.reshape(-1, 1), "X105RPM": 1797})

            sig, meta = load_cwru_mat_file(mat_path)
            self.assertEqual(len(sig), 4000)
            self.assertEqual(meta.file_id, 105)
            self.assertEqual(meta.fault_label, 1)
            self.assertEqual(meta.fault_type, "Inner Race Fault")
            self.assertEqual(meta.sensor_location, "DE")
            self.assertEqual(meta.sampling_rate, 12000)

    def test_03_window_content_indexing(self):
        """3. Test exact window content indexing for known ramp signal [0, 1, 2, ...]."""
        ramp_sig = np.arange(5000, dtype=np.float32)
        meta = CWRUFileMetadata(
            file_id=105,
            source_file="105.mat",
            fault_label=1,
            fault_type="Inner Race Fault",
            fault_size=0.007,
            load_hp=0,
            rpm=1797,
            sampling_rate=12000,
        )

        windows, meta_list, disc = window_signal(ramp_sig, meta, split_name="train")

        self.assertEqual(windows.shape, (2, 2048, 1))
        # Window 0 should contain samples 0:2048
        np.testing.assert_array_equal(windows[0, :, 0], np.arange(0, 2048, dtype=np.float32))
        # Window 1 should contain samples 2048:4096
        np.testing.assert_array_equal(windows[1, :, 0], np.arange(2048, 4096, dtype=np.float32))

    def test_04_incomplete_tail_window_discarding(self):
        """4. Test incomplete final window sample discarding calculation."""
        # 5000 samples -> 2 windows of 2048 = 4096 samples used. Discarded = 5000 - 4096 = 904
        sig = np.random.randn(5000).astype(np.float32)
        meta = CWRUFileMetadata(105, "105.mat", 1, "IR", 0.007, 0, 1797, 12000)

        windows, meta_list, disc = window_signal(sig, meta, split_name="val")
        self.assertEqual(len(windows), 2)
        self.assertEqual(disc, 904)

    def test_05_split_integrity_and_unique_observation_ids(self):
        """5. Test that every window has a unique observation ID and metadata matches split."""
        sig = np.random.randn(4096).astype(np.float32)
        meta = CWRUFileMetadata(97, "97.mat", 0, "Normal", 0.0, 0, 1797, 12000)

        windows, meta_list, disc = window_signal(sig, meta, split_name="test")
        obs_ids = [m["observation_id"] for m in meta_list]

        self.assertEqual(len(obs_ids), len(set(obs_ids)))
        self.assertEqual(meta_list[0]["split"], "test")
        self.assertEqual(meta_list[0]["source_file"], "97.mat")

    def test_06_train_only_normalization_no_leakage(self):
        """6. Test that validation and test samples DO NOT influence train mean and std."""
        train_sig = np.array([10.0, 20.0, 30.0] * 1000, dtype=np.float32)
        val_sig = np.array([1000.0, 2000.0, 3000.0] * 1000, dtype=np.float32)

        train_meta = CWRUFileMetadata(97, "97.mat", 0, "Normal", 0.0, 0, 1797, 12000)
        val_meta = CWRUFileMetadata(98, "98.mat", 0, "Normal", 0.0, 1, 1772, 12000)

        train_recs = [(train_sig, train_meta)]
        val_recs = [(val_sig, val_meta)]

        # Fit normalization on train only
        t_mean, t_std = fit_train_normalization(train_recs)

        self.assertAlmostEqual(t_mean, 20.0, places=4)

        # Normalize val using train statistics
        norm_val = normalize_signal(val_sig, t_mean, t_std)
        # Val mean should NOT equal 0 because it was normalized with train statistics!
        self.assertNotAlmostEqual(float(np.mean(norm_val)), 0.0, places=1)

    def test_07_group_level_leakage_prevention_split(self):
        """7. Test that recording/group-level split prevents file overlap across splits."""
        records = []
        # Create 12 distinct file recordings across 4 classes
        for file_id in [97, 98, 99, 105, 106, 107, 118, 119, 120, 130, 131, 132]:
            sig = np.random.randn(5000).astype(np.float32)
            meta = load_cwru_mat_file
            # Create mock metadata
            if file_id in [97, 98, 99]:
                meta = CWRUFileMetadata(file_id, f"{file_id}.mat", 0, "Normal", 0.0, 0, 1797, 12000)
            elif file_id in [105, 106, 107]:
                meta = CWRUFileMetadata(file_id, f"{file_id}.mat", 1, "IR", 0.007, 0, 1797, 12000)
            elif file_id in [118, 119, 120]:
                meta = CWRUFileMetadata(file_id, f"{file_id}.mat", 2, "B", 0.007, 0, 1797, 12000)
            else:
                meta = CWRUFileMetadata(file_id, f"{file_id}.mat", 3, "OR", 0.007, 0, 1797, 12000)
            records.append((sig, meta))

        train, val, test = create_leakage_safe_split(records, seed=42)

        train_files = {m.source_file for _, m in train}
        val_files = {m.source_file for _, m in val}
        test_files = {m.source_file for _, m in test}

        # Assert zero overlap between file splits!
        self.assertEqual(len(train_files.intersection(val_files)), 0)
        self.assertEqual(len(train_files.intersection(test_files)), 0)
        self.assertEqual(len(val_files.intersection(test_files)), 0)

    def test_08_reproducibility(self):
        """8. Test that running pipeline with seed 42 produces reproducible results."""
        with tempfile.TemporaryDirectory() as tmp_raw, tempfile.TemporaryDirectory() as tmp_proc:
            # Create synthetic raw .mat files
            for file_id in [97, 98, 99, 105, 106, 107, 118, 119, 120, 130, 131, 132]:
                mat_path = os.path.join(tmp_raw, f"{file_id}.mat")
                sig_data = np.sin(np.linspace(0, 100, 5000)).astype(np.float64)
                savemat(mat_path, {f"X{file_id:03d}_DE_time": sig_data.reshape(-1, 1)})

            sum1 = run_cwru_preprocessing_pipeline(
                raw_dir=tmp_raw, processed_dir=tmp_proc, seed=42
            )
            sum2 = run_cwru_preprocessing_pipeline(
                raw_dir=tmp_raw, processed_dir=tmp_proc, seed=42
            )

            self.assertEqual(sum1["train_files"], sum2["train_files"])
            self.assertEqual(sum1["val_files"], sum2["val_files"])
            self.assertEqual(sum1["test_files"], sum2["test_files"])
            self.assertAlmostEqual(sum1["train_mean"], sum2["train_mean"], places=6)


if __name__ == "__main__":
    unittest.main()
