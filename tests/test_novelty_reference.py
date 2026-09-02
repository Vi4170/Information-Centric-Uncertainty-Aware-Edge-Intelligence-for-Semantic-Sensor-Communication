"""Unit test suite for the versioned multi-prototype novelty reference
(Task 19, continual-learning Phase 3).

Uses small deterministic synthetic embeddings throughout. Covers
nearest-prototype selection, append-only behaviour, versioning, every
rejection rule, metadata preservation, serialization, determinism, and a
small synthetic "condition A / condition B" scenario demonstrating the
append-only mechanism the design doc describes -- this is an
infrastructure demonstration, not an experimental continual-learning
result.
"""

import os
import tempfile
import unittest

import numpy as np

from src.continual.novelty_reference import NoveltyReference, Prototype


def cluster(center, offsets):
    """Build a small (N, dim) embedding batch: center + each offset row."""
    center = np.array(center, dtype=np.float64)
    return np.array([center + np.array(o, dtype=np.float64) for o in offsets], dtype=np.float64)


class TestNoveltyReference(unittest.TestCase):
    """Test suite for NoveltyReference / Prototype."""

    def test_01_single_prototype_nearest_distance(self):
        """1. Test nearest-prototype distance with only one prototype present."""
        ref = NoveltyReference(embedding_dim=3)
        ref.add_prototype(
            "normal",
            cluster([0, 0, 0], [[0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0]]),
            source_dataset="synthetic",
            source_condition="normal",
            source_split="train",
        )
        prototype, distance = ref.nearest_prototype(np.array([0.05, 0.0, 0.0]))
        self.assertEqual(prototype.prototype_id, "normal")
        self.assertAlmostEqual(distance, 0.05, places=6)

    def test_02_multiple_prototype_nearest_distance_and_selection(self):
        """2. Test that the CLOSEST of several prototypes is correctly selected."""
        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("a", cluster([0, 0], [[0, 0], [0, 0]]), "synthetic", "cond_a", "train")
        ref.add_prototype("b", cluster([10, 0], [[0, 0], [0, 0]]), "synthetic", "cond_b", "train")
        ref.add_prototype("c", cluster([0, 10], [[0, 0], [0, 0]]), "synthetic", "cond_c", "train")

        prototype, distance = ref.nearest_prototype(np.array([9.5, 0.2]))
        self.assertEqual(prototype.prototype_id, "b")
        self.assertAlmostEqual(distance, np.linalg.norm([0.5, 0.2]), places=6)

        report = ref.distance_report(np.array([9.5, 0.2]))
        self.assertEqual(set(report.keys()), {"a", "b", "c"})

    def test_03_append_only_behavior(self):
        """3. Test that adding a new prototype never modifies an existing one."""
        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "synthetic", "cond_a", "train")
        a_before = ref.get_prototype("a")
        centroid_before = a_before.centroid.copy()

        ref.add_prototype("b", cluster([100, 100], [[0, 0], [1, 0]]), "synthetic", "cond_b", "train")

        a_after = ref.get_prototype("a")
        np.testing.assert_array_equal(a_after.centroid, centroid_before)
        self.assertEqual(a_after.n_source_embeddings, a_before.n_source_embeddings)
        self.assertEqual(len(ref.prototypes), 2)
        self.assertEqual(ref.prototype_ids(), ("a", "b"))

    def test_04_version_increment(self):
        """4. Test that version increments by exactly 1 per successful add_prototype() call."""
        ref = NoveltyReference(embedding_dim=2)
        self.assertEqual(ref.version, 0)
        ref.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "synthetic", "cond_a", "train")
        self.assertEqual(ref.version, 1)
        ref.add_prototype("b", cluster([5, 5], [[0, 0], [1, 0]]), "synthetic", "cond_b", "train")
        self.assertEqual(ref.version, 2)
        ref.add_prototype("c", cluster([9, 9], [[0, 0], [1, 0]]), "synthetic", "cond_c", "train")
        self.assertEqual(ref.version, 3)
        self.assertEqual([p.version_added for p in ref.prototypes], [1, 2, 3])

    def test_05_duplicate_id_rejected(self):
        """5. Test that re-using an existing prototype_id is rejected, not overwritten."""
        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "synthetic", "cond_a", "train")
        with self.assertRaises(ValueError):
            ref.add_prototype("a", cluster([9, 9], [[0, 0], [1, 0]]), "synthetic", "cond_a_v2", "train")
        self.assertEqual(ref.version, 1)
        self.assertEqual(ref.get_prototype("a").source_condition, "cond_a")

    def test_06_dimension_validation(self):
        """6. Test that mismatched embedding dimensionality is rejected at creation and lookup."""
        ref = NoveltyReference(embedding_dim=3)
        with self.assertRaises(ValueError):
            ref.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "synthetic", "cond_a", "train")  # dim=2, expects 3

        ref.add_prototype("a", cluster([0, 0, 0], [[0, 0, 0], [1, 0, 0]]), "synthetic", "cond_a", "train")
        with self.assertRaises(ValueError):
            ref.nearest_prototype(np.array([0.0, 0.0]))  # wrong length

    def test_07_non_finite_input_rejected(self):
        """7. Test that NaN/Inf embeddings are rejected both at prototype creation and at lookup."""
        ref = NoveltyReference(embedding_dim=2)
        bad_embeddings = np.array([[0.0, 0.0], [np.nan, 0.0]])
        with self.assertRaises(ValueError):
            ref.add_prototype("a", bad_embeddings, "synthetic", "cond_a", "train")

        ref.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "synthetic", "cond_a", "train")
        with self.assertRaises(ValueError):
            ref.nearest_prototype(np.array([np.inf, 0.0]))

    def test_08_invalid_prototype_creation_rejected(self):
        """8. Test rejection of empty embedding arrays, non-array input, and empty identifier fields."""
        ref = NoveltyReference(embedding_dim=2)
        with self.assertRaises(ValueError):
            ref.add_prototype("a", np.zeros((0, 2)), "synthetic", "cond_a", "train")  # empty
        with self.assertRaises(TypeError):
            ref.add_prototype("a", [[0.0, 0.0]], "synthetic", "cond_a", "train")  # not ndarray
        with self.assertRaises(ValueError):
            ref.add_prototype("", cluster([0, 0], [[0, 0]]), "synthetic", "cond_a", "train")
        with self.assertRaises(ValueError):
            ref.add_prototype("a", cluster([0, 0], [[0, 0]]), "", "cond_a", "train")
        with self.assertRaises(ValueError):
            ref.add_prototype("a", cluster([0, 0], [[0, 0]]), "synthetic", "", "train")

    def test_09_test_split_protection(self):
        """9. Test that a prototype cannot be created from test-split observations through this API."""
        ref = NoveltyReference(embedding_dim=2)
        with self.assertRaises(ValueError):
            ref.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "synthetic", "cond_a", source_split="test")
        self.assertEqual(ref.version, 0)
        self.assertEqual(len(ref.prototypes), 0)

        with self.assertRaises(ValueError):
            ref.add_prototype("a", cluster([0, 0], [[0, 0]]), "synthetic", "cond_a", source_split="holdout")

    def test_10_metadata_preservation(self):
        """10. Test that all provenance metadata is preserved exactly, including validation-split provenance."""
        ref = NoveltyReference(embedding_dim=2)
        prototype = ref.add_prototype(
            "a",
            cluster([0, 0], [[0, 0], [1, 0]]),
            source_dataset="cwru",
            source_condition="normal_operation",
            source_split="val",
            extra={"note": "from Task 19 unit test"},
        )
        self.assertEqual(prototype.source_dataset, "cwru")
        self.assertEqual(prototype.source_condition, "normal_operation")
        self.assertEqual(prototype.source_split, "val")  # validation provenance not silently discarded
        self.assertEqual(prototype.n_source_embeddings, 2)
        self.assertEqual(prototype.extra, {"note": "from Task 19 unit test"})

    def test_11_serialization_roundtrip(self):
        """11. Test that save_json/load_json preserves the reference exactly, including history."""
        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "cwru", "cond_a", "train", extra={"k": 1})
        ref.add_prototype("b", cluster([5, 5], [[0, 0], [-1, 0]]), "cwru", "cond_b", "val")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "reference.json")
            ref.save_json(path)
            reloaded = NoveltyReference.load_json(path)

        self.assertEqual(reloaded.version, ref.version)
        self.assertEqual(reloaded.embedding_dim, ref.embedding_dim)
        self.assertEqual(reloaded.prototype_ids(), ref.prototype_ids())
        self.assertEqual(reloaded.history, ref.history)
        for pid in ref.prototype_ids():
            np.testing.assert_array_equal(reloaded.get_prototype(pid).centroid, ref.get_prototype(pid).centroid)
            self.assertEqual(reloaded.get_prototype(pid).to_dict(), ref.get_prototype(pid).to_dict())

    def test_12_deterministic_results(self):
        """12. Test that two independently built references from identical input behave identically."""
        def build():
            r = NoveltyReference(embedding_dim=2)
            r.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "cwru", "cond_a", "train")
            r.add_prototype("b", cluster([5, 5], [[0, 0], [-1, 0]]), "cwru", "cond_b", "train")
            return r

        ref_a = build()
        ref_b = build()

        probe = np.array([2.5, 2.5])
        proto_a, dist_a = ref_a.nearest_prototype(probe)
        proto_b, dist_b = ref_b.nearest_prototype(probe)
        self.assertEqual(proto_a.prototype_id, proto_b.prototype_id)
        self.assertEqual(dist_a, dist_b)
        self.assertEqual(ref_a.to_dict(), ref_b.to_dict())

    def test_13_previously_learned_prototypes_unchanged_after_multiple_updates(self):
        """13. Test that prototype A's centroid/distance behaviour is unchanged after several later additions."""
        ref = NoveltyReference(embedding_dim=2)
        ref.add_prototype("a", cluster([0, 0], [[0, 0], [1, 0]]), "cwru", "cond_a", "train")
        probe = np.array([0.5, 0.0])
        _, distance_before = ref.nearest_prototype(probe)

        for i in range(5):
            ref.add_prototype(f"extra_{i}", cluster([100 + i, 100 + i], [[0, 0], [1, 0]]), "cwru", f"cond_{i}", "train")

        prototype_after, distance_after = ref.nearest_prototype(probe)
        self.assertEqual(prototype_after.prototype_id, "a")
        self.assertEqual(distance_after, distance_before)
        self.assertEqual(ref.version, 6)

    def test_14_empty_reference_lookup_rejected(self):
        """14. Test that querying a reference with zero prototypes is explicitly rejected."""
        ref = NoveltyReference(embedding_dim=2)
        with self.assertRaises(ValueError):
            ref.nearest_prototype(np.array([0.0, 0.0]))
        with self.assertRaises(ValueError):
            ref.distance_report(np.array([0.0, 0.0]))

    def test_15_distance_report_contains_every_prototype(self):
        """15. Test that distance_report reports a distance for every known prototype, not just the nearest."""
        ref = NoveltyReference(embedding_dim=1)
        ref.add_prototype("a", cluster([0], [[0], [0]]), "synthetic", "cond_a", "train")
        ref.add_prototype("b", cluster([10], [[0], [0]]), "synthetic", "cond_b", "train")
        report = ref.distance_report(np.array([5.0]))
        self.assertEqual(report, {"a": 5.0, "b": 5.0})

    def test_16_synthetic_condition_a_then_b_scenario(self):
        """16. Infrastructure demonstration (not an experimental result): a condition initially
        far from all known prototypes becomes 'known' once explicitly validated and added, while
        the original condition's own behaviour is completely unaffected.
        """
        ref = NoveltyReference(embedding_dim=3)
        ref.add_prototype(
            "condition_a",
            cluster([0, 0, 0], [[0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0]]),
            source_dataset="synthetic",
            source_condition="condition_a",
            source_split="train",
        )

        obs_a = np.array([0.05, 0.0, 0.0])
        obs_b = np.array([10.0, 10.0, 10.0])

        # Before B is known: B looks far from everything we know (novel).
        proto_for_a_before, dist_a_before = ref.nearest_prototype(obs_a)
        proto_for_b_before, dist_b_before = ref.nearest_prototype(obs_b)
        self.assertEqual(proto_for_a_before.prototype_id, "condition_a")
        self.assertLess(dist_a_before, 0.2)
        self.assertEqual(proto_for_b_before.prototype_id, "condition_a")  # only prototype that exists
        self.assertGreater(dist_b_before, 15.0)  # far away -- looks novel

        # Condition B is now explicitly validated and added (a future Safety
        # Gate's job in the full architecture; here just a direct, explicit call).
        ref.add_prototype(
            "condition_b",
            cluster([10, 10, 10], [[0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0]]),
            source_dataset="synthetic",
            source_condition="condition_b",
            source_split="train",
        )

        # After B is known: B is now recognized; A is completely unaffected.
        proto_for_b_after, dist_b_after = ref.nearest_prototype(obs_b)
        proto_for_a_after, dist_a_after = ref.nearest_prototype(obs_a)

        self.assertEqual(proto_for_b_after.prototype_id, "condition_b")
        self.assertLess(dist_b_after, 0.2)  # no longer looks novel
        self.assertEqual(proto_for_a_after.prototype_id, "condition_a")
        self.assertEqual(dist_a_after, dist_a_before)  # A's behaviour is byte-identical to before


if __name__ == "__main__":
    unittest.main()
