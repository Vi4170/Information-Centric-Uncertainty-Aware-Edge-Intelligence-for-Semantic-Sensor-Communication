import os
import unittest

import numpy as np

from src.paderborn_pipeline.preprocessing import PADERBORN_BEARING_REGISTRY, RAW_DATA_DIR
from src.paderborn_pipeline.classification_task import (
    BEARING_CLASS_LABELS,
    CLASS_NAMES,
    EXCLUDED_BEARING_CODES,
    EXPERIMENT_CHANNEL_INDEX,
    EXPERIMENT_MODALITY,
    EXPERIMENT_OPERATING_CONDITION,
    build_classification_dataset,
)
from src.paderborn_pipeline.preprocessing import (
    fit_train_normalization,
    apply_normalization,
    verify_no_measurement_crosses_split,
    verify_split_disjoint,
)
from src.ims_pipeline.preprocessing import verify_observation_id_uniqueness
from src.cnn.model import build_baseline_cnn, extract_embeddings, predict_probabilities
from src.voi.scoring import VoIWeights
from src.voi.decision_policy import PolicyThresholds
from src.evaluation.paderborn_voi_behaviour_analysis import PADERBORN_RELEVANCE_MAP, _relevance_from_probabilities

_RAW_DATA_AVAILABLE = all(
    os.path.isdir(os.path.join(RAW_DATA_DIR, code)) for code in ("K001", "KI01")
)


class TestPaderbornBearingTaxonomy(unittest.TestCase):
    def test_taxonomy_covers_full_registry_without_overlap(self):
        all_codes = set(PADERBORN_BEARING_REGISTRY.keys())
        included = set(BEARING_CLASS_LABELS.keys())
        excluded = set(EXCLUDED_BEARING_CODES.keys())
        self.assertEqual(included & excluded, set())
        self.assertEqual(included | excluded, all_codes)

    def test_class_counts(self):
        counts = {0: 0, 1: 0, 2: 0}
        for class_id in BEARING_CLASS_LABELS.values():
            counts[class_id] += 1
        self.assertEqual(counts[0], 6)
        self.assertEqual(counts[1], 9)
        self.assertEqual(counts[2], 11)
        self.assertEqual(len(BEARING_CLASS_LABELS), 26)
        self.assertEqual(len(EXCLUDED_BEARING_CODES), 6)

    def test_excluded_codes_have_non_empty_reasons(self):
        for code, reason in EXCLUDED_BEARING_CODES.items():
            self.assertIsInstance(reason, str)
            self.assertGreater(len(reason), 0)

    def test_class_names_cover_all_labels(self):
        used_labels = set(BEARING_CLASS_LABELS.values())
        self.assertEqual(used_labels, set(CLASS_NAMES.keys()))


@unittest.skipUnless(_RAW_DATA_AVAILABLE, "Paderborn raw data (K001, KI01) not available")
class TestPaderbornDatasetAssemblySmallSubset(unittest.TestCase):
    SUBSET = {"K001": 0, "KI01": 1}

    def test_assembly_produces_correct_shapes_and_labels(self):
        X, meta = build_classification_dataset(bearing_class_labels=self.SUBSET)
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[1], 2048)
        self.assertEqual(X.shape[2], 1)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(len(X), len(meta))
        self.assertTrue(set(meta["bearing_code"].unique()).issubset(set(self.SUBSET.keys())))
        self.assertTrue(set(meta["fault_label"].unique()).issubset({0, 1}))
        self.assertTrue((meta.loc[meta["bearing_code"] == "K001", "fault_label"] == 0).all())
        self.assertTrue((meta.loc[meta["bearing_code"] == "KI01", "fault_label"] == 1).all())
        self.assertTrue(set(meta["operating_condition"].unique()) == {EXPERIMENT_OPERATING_CONDITION})
        self.assertTrue(set(meta["modality"].unique()) == {EXPERIMENT_MODALITY})
        self.assertTrue(set(meta["channel_index"].unique()) == {EXPERIMENT_CHANNEL_INDEX})
        self.assertTrue(set(meta["split"].unique()).issubset({"train", "val", "test"}))

    def test_no_leakage(self):
        _, meta = build_classification_dataset(bearing_class_labels=self.SUBSET)
        self.assertTrue(verify_observation_id_uniqueness(meta))
        overlaps = verify_split_disjoint(meta)
        self.assertEqual(sum(overlaps.values()), 0)
        self.assertTrue(verify_no_measurement_crosses_split(meta))

    def test_normalization_fits_on_train_only(self):
        X, meta = build_classification_dataset(bearing_class_labels=self.SUBSET)
        mean, std = fit_train_normalization(X, meta)
        train_mask = (meta["split"] == "train").to_numpy()
        manual_mean = float(np.mean(X[train_mask]))
        manual_std = float(np.std(X[train_mask]))
        self.assertAlmostEqual(mean, manual_mean, places=5)
        self.assertAlmostEqual(std, manual_std, places=5)

        normalized = apply_normalization(X, mean, std)
        self.assertAlmostEqual(float(np.mean(normalized[train_mask])), 0.0, places=3)

    def test_deterministic_repeat_assembly(self):
        X1, meta1 = build_classification_dataset(bearing_class_labels=self.SUBSET)
        X2, meta2 = build_classification_dataset(bearing_class_labels=self.SUBSET)
        np.testing.assert_array_equal(X1, X2)
        self.assertTrue(meta1["observation_id"].equals(meta2["observation_id"]))
        self.assertTrue(meta1["split"].equals(meta2["split"]))


class TestPaderbornCnnCompatibility(unittest.TestCase):
    def test_builds_with_three_classes(self):
        model = build_baseline_cnn(num_classes=len(CLASS_NAMES))
        output_layer = model.get_layer("output_probabilities")
        self.assertEqual(output_layer.units, len(CLASS_NAMES))

    def test_forward_pass_shapes(self):
        model = build_baseline_cnn(num_classes=len(CLASS_NAMES))
        X_dummy = np.random.randn(4, 2048, 1).astype(np.float32)
        probs = predict_probabilities(model, X_dummy)
        self.assertEqual(probs.shape, (4, len(CLASS_NAMES)))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-4)

        embeddings = extract_embeddings(model, X_dummy)
        self.assertEqual(embeddings.shape, (4, 64))


class TestPaderbornRelevanceFormula(unittest.TestCase):
    def test_relevance_map_inherits_cwru_values(self):
        self.assertEqual(PADERBORN_RELEVANCE_MAP, {0: 0.10, 1: 1.00, 2: 0.90})

    def test_one_hot_matches_map_value(self):
        for class_id, expected in PADERBORN_RELEVANCE_MAP.items():
            probs = np.zeros(3, dtype=np.float32)
            probs[class_id] = 1.0
            score = _relevance_from_probabilities(probs, PADERBORN_RELEVANCE_MAP)
            self.assertAlmostEqual(score, expected, places=5)

    def test_uniform_distribution_matches_weighted_average(self):
        probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        score = _relevance_from_probabilities(probs, PADERBORN_RELEVANCE_MAP)
        expected = sum(PADERBORN_RELEVANCE_MAP.values()) / 3
        self.assertAlmostEqual(score, expected, places=4)


class TestVoiProductionConfigUnchanged(unittest.TestCase):
    def test_weights_unchanged(self):
        weights = VoIWeights()
        self.assertEqual(weights.novelty, 0.30)
        self.assertEqual(weights.uncertainty, 0.05)
        self.assertEqual(weights.task_relevance, 0.35)
        self.assertEqual(weights.temporal_importance, 0.20)
        self.assertEqual(weights.resource_cost, 0.10)

    def test_thresholds_unchanged(self):
        thresholds = PolicyThresholds()
        self.assertEqual(thresholds.discard_max, 0.25)
        self.assertEqual(thresholds.buffer_max, 0.50)
        self.assertEqual(thresholds.summary_max, 0.70)


if __name__ == "__main__":
    unittest.main()
