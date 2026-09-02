"""Unit test suite for the adaptation buffer (Task 17, continual-learning Phase 1).

Tests cover accept/reject rules, confirmed-vs-pseudo label handling,
recording/dataset identity and split preservation, batch-ingest atomicity,
determinism, JSON round-tripping, and -- most importantly -- test-set
leakage prevention, including an integration check against the real CWRU
metadata.
"""

import os
import tempfile
import unittest

import pandas as pd

from src.continual.adaptation_buffer import AdaptationBuffer, LabelStatus, check_no_overlap


class TestAdaptationBuffer(unittest.TestCase):
    """Test suite for AdaptationBuffer accept/reject and leakage-safety behavior."""

    def test_01_add_train_observation_succeeds(self):
        """1. Test a train-split observation is accepted."""
        buf = AdaptationBuffer()
        record = buf.add(
            observation_id="obs_1",
            dataset="cwru",
            split="train",
            source_recording_id="97",
            label_status=LabelStatus.PSEUDO,
        )
        self.assertEqual(len(buf), 1)
        self.assertIn("obs_1", buf)
        self.assertEqual(record.split, "train")

    def test_02_add_val_observation_succeeds(self):
        """2. Test a val-split observation is accepted."""
        buf = AdaptationBuffer()
        buf.add(
            observation_id="obs_2",
            dataset="cwru",
            split="val",
            source_recording_id="98",
            label_status=LabelStatus.PSEUDO,
        )
        self.assertEqual(len(buf), 1)

    def test_03_test_split_rejected(self):
        """3. Test that a test-split observation is explicitly rejected, not silently dropped."""
        buf = AdaptationBuffer()
        with self.assertRaises(ValueError):
            buf.add(
                observation_id="obs_3",
                dataset="cwru",
                split="test",
                source_recording_id="99",
                label_status=LabelStatus.PSEUDO,
            )
        self.assertEqual(len(buf), 0)

    def test_04_unrecognized_split_rejected(self):
        """4. Test that an unrecognized split string is rejected."""
        buf = AdaptationBuffer()
        with self.assertRaises(ValueError):
            buf.add(
                observation_id="obs_4",
                dataset="cwru",
                split="holdout",
                source_recording_id="99",
                label_status=LabelStatus.PSEUDO,
            )
        self.assertEqual(len(buf), 0)

    def test_05_duplicate_observation_id_rejected(self):
        """5. Test that inserting the same observation_id twice is rejected, not overwritten."""
        buf = AdaptationBuffer()
        buf.add(observation_id="dup", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.PSEUDO)
        with self.assertRaises(ValueError):
            buf.add(observation_id="dup", dataset="cwru", split="val", source_recording_id="2", label_status=LabelStatus.PSEUDO)
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf.to_dataframe().iloc[0]["split"], "train")

    def test_06_confirmed_requires_label(self):
        """6. Test that label_status=CONFIRMED without a label is rejected."""
        buf = AdaptationBuffer()
        with self.assertRaises(ValueError):
            buf.add(
                observation_id="obs_6",
                dataset="cwru",
                split="train",
                source_recording_id="1",
                label_status=LabelStatus.CONFIRMED,
                label=None,
            )

    def test_07_pseudo_label_optional(self):
        """7. Test that PSEUDO status does not require a label."""
        buf = AdaptationBuffer()
        record = buf.add(
            observation_id="obs_7",
            dataset="cwru",
            split="train",
            source_recording_id="1",
            label_status=LabelStatus.PSEUDO,
        )
        self.assertIsNone(record.label)

    def test_08_confirmed_vs_pseudo_distinguishable(self):
        """8. Test that confirmed and pseudo counts are tracked separately."""
        buf = AdaptationBuffer()
        buf.add(observation_id="c1", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.CONFIRMED, label=0)
        buf.add(observation_id="c2", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.CONFIRMED, label=1)
        buf.add(observation_id="p1", dataset="cwru", split="val", source_recording_id="2", label_status=LabelStatus.PSEUDO)
        self.assertEqual(buf.confirmed_count, 2)
        self.assertEqual(buf.pseudo_count, 1)
        self.assertEqual(len(buf), 3)

    def test_09_source_recording_identity_preserved(self):
        """9. Test that records can be retrieved by their source recording id."""
        buf = AdaptationBuffer()
        buf.add(observation_id="a", dataset="cwru", split="train", source_recording_id="rec_1", label_status=LabelStatus.PSEUDO)
        buf.add(observation_id="b", dataset="cwru", split="train", source_recording_id="rec_1", label_status=LabelStatus.PSEUDO)
        buf.add(observation_id="c", dataset="cwru", split="train", source_recording_id="rec_2", label_status=LabelStatus.PSEUDO)

        rec_1_records = buf.get_by_recording("rec_1")
        self.assertEqual(len(rec_1_records), 2)
        self.assertEqual({r.observation_id for r in rec_1_records}, {"a", "b"})
        self.assertEqual(len(buf.get_by_recording("rec_2")), 1)
        self.assertEqual(len(buf.get_by_recording("nonexistent")), 0)

    def test_10_split_information_preserved(self):
        """10. Test that each record's own split value is preserved exactly."""
        buf = AdaptationBuffer()
        buf.add(observation_id="a", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.PSEUDO)
        buf.add(observation_id="b", dataset="cwru", split="val", source_recording_id="1", label_status=LabelStatus.PSEUDO)
        df = buf.to_dataframe().set_index("observation_id")
        self.assertEqual(df.loc["a", "split"], "train")
        self.assertEqual(df.loc["b", "split"], "val")

    def test_11_batch_add_rejects_test_rows_atomically(self):
        """11. Test that a batch containing any test row is entirely rejected."""
        buf = AdaptationBuffer()
        df = pd.DataFrame(
            {
                "observation_id": ["w1", "w2", "w3"],
                "split": ["train", "test", "train"],
                "file_id": ["1", "1", "1"],
            }
        )
        with self.assertRaises(ValueError):
            buf.add_from_dataframe(df, dataset="cwru", label_status=LabelStatus.PSEUDO)
        self.assertEqual(len(buf), 0, "No rows should be added when the batch contains any test-split row")

    def test_12_batch_add_rejects_duplicate_ids_within_batch(self):
        """12. Test that a batch with duplicate observation_ids is rejected wholesale."""
        buf = AdaptationBuffer()
        df = pd.DataFrame(
            {
                "observation_id": ["w1", "w1"],
                "split": ["train", "train"],
                "file_id": ["1", "1"],
            }
        )
        with self.assertRaises(ValueError):
            buf.add_from_dataframe(df, dataset="cwru", label_status=LabelStatus.PSEUDO)
        self.assertEqual(len(buf), 0)

    def test_13_batch_add_rejects_ids_already_present(self):
        """13. Test that a batch cannot re-add an observation_id already in the buffer."""
        buf = AdaptationBuffer()
        buf.add(observation_id="w1", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.PSEUDO)
        df = pd.DataFrame({"observation_id": ["w1"], "split": ["train"], "file_id": ["1"]})
        with self.assertRaises(ValueError):
            buf.add_from_dataframe(df, dataset="cwru", label_status=LabelStatus.PSEUDO)
        self.assertEqual(len(buf), 1)

    def test_14_batch_add_valid_rows_succeed(self):
        """14. Test that a clean batch (train/val only, no duplicates) is fully ingested."""
        buf = AdaptationBuffer()
        df = pd.DataFrame(
            {
                "observation_id": ["w1", "w2", "w3"],
                "split": ["train", "train", "val"],
                "file_id": ["1", "1", "2"],
                "window_index": [0, 1, 0],
                "fault_label": [0, 1, 0],
            }
        )
        records = buf.add_from_dataframe(
            df, dataset="cwru", label_status=LabelStatus.CONFIRMED, label_column="fault_label"
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(len(buf), 3)
        self.assertEqual(buf.confirmed_count, 3)

    def test_15_deterministic_ordering_and_rebuild(self):
        """15. Test insertion-order determinism and identical rebuilds from identical input."""
        df = pd.DataFrame(
            {
                "observation_id": ["w1", "w2", "w3"],
                "split": ["train", "val", "train"],
                "file_id": ["1", "2", "1"],
            }
        )

        buf_a = AdaptationBuffer()
        buf_a.add_from_dataframe(df, dataset="cwru", label_status=LabelStatus.PSEUDO)
        buf_b = AdaptationBuffer()
        buf_b.add_from_dataframe(df, dataset="cwru", label_status=LabelStatus.PSEUDO)

        self.assertEqual(list(buf_a.to_dataframe()["observation_id"]), ["w1", "w2", "w3"])
        pd.testing.assert_frame_equal(buf_a.to_dataframe(), buf_b.to_dataframe())

    def test_16_verify_no_test_leakage_passes_when_disjoint(self):
        """16. Test that the leakage check passes silently when there is no overlap."""
        buf = AdaptationBuffer()
        buf.add(observation_id="w1", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.PSEUDO)
        buf.verify_no_test_leakage(["w99", "w100"])  # should not raise

    def test_17_verify_no_test_leakage_raises_on_overlap(self):
        """17. Test that the leakage check raises if an id also appears in the supplied test set.

        This is an independent safety net (checks against an externally
        supplied authoritative test-id list) beyond the per-insert split
        rejection covered by test_03.
        """
        buf = AdaptationBuffer()
        buf.add(observation_id="w1", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.PSEUDO)
        with self.assertRaises(AssertionError):
            buf.verify_no_test_leakage(["w1", "w2"])

    def test_18_check_no_overlap_pure_function(self):
        """18. Test the standalone check_no_overlap helper."""
        self.assertEqual(check_no_overlap({"a", "b"}, {"b", "c"}), {"b"})
        self.assertEqual(check_no_overlap({"a"}, {"b"}), set())

    def test_19_save_and_load_json_roundtrip(self):
        """19. Test that a buffer survives a save/load round trip unchanged."""
        buf = AdaptationBuffer()
        buf.add(observation_id="w1", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.CONFIRMED, label=2, window_index=5, extra={"novelty": 0.5})
        buf.add(observation_id="w2", dataset="cwru", split="val", source_recording_id="2", label_status=LabelStatus.PSEUDO)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "buffer.json")
            buf.save_json(path)
            self.assertTrue(os.path.exists(path))

            reloaded = AdaptationBuffer.load_json(path)
            pd.testing.assert_frame_equal(buf.to_dataframe(), reloaded.to_dataframe())

    def test_20_empty_identifier_fields_rejected(self):
        """20. Test that empty observation_id/dataset/source_recording_id are rejected."""
        buf = AdaptationBuffer()
        with self.assertRaises(ValueError):
            buf.add(observation_id="", dataset="cwru", split="train", source_recording_id="1", label_status=LabelStatus.PSEUDO)
        with self.assertRaises(ValueError):
            buf.add(observation_id="w1", dataset="", split="train", source_recording_id="1", label_status=LabelStatus.PSEUDO)
        with self.assertRaises(ValueError):
            buf.add(observation_id="w1", dataset="cwru", split="train", source_recording_id="", label_status=LabelStatus.PSEUDO)

    def test_21_real_cwru_metadata_integration(self):
        """21. Integration test: ingest real CWRU train+val metadata and verify zero test leakage.

        Skips gracefully if the processed CWRU dataset has not been
        generated in this environment (matches the skip style already used
        by other pipeline-dependent tests in this repo).
        """
        metadata_path = os.path.join("data", "processed", "cwru", "cwru_metadata.csv")
        if not os.path.exists(metadata_path):
            self.skipTest("Processed CWRU metadata not available in this environment")

        meta = pd.read_csv(metadata_path)
        train_val_meta = meta[meta["split"].isin(["train", "val"])].reset_index(drop=True)
        test_ids = set(meta.loc[meta["split"] == "test", "observation_id"])

        buf = AdaptationBuffer()
        buf.add_from_dataframe(
            train_val_meta,
            dataset="cwru",
            label_status=LabelStatus.CONFIRMED,
            label_column="fault_label",
        )

        self.assertEqual(len(buf), len(train_val_meta))
        buf.verify_no_test_leakage(test_ids)  # should not raise

        # Spot-check recording identity is preserved for a real recording.
        sample_file_id = str(train_val_meta.iloc[0]["file_id"])
        records = buf.get_by_recording(sample_file_id)
        self.assertGreater(len(records), 0)
        self.assertTrue(all(r.source_recording_id == sample_file_id for r in records))


if __name__ == "__main__":
    unittest.main()
