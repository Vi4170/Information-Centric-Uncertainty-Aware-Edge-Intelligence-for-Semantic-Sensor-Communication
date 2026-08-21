"""Unit test suite for baseline 1D CNN model and training interface (src/cnn/).

Fast and focused tests: validates model architecture, shapes, probability constraints,
embedding extraction, micro-training, and reproducibility without long full runs.
"""

import os
import tempfile
import unittest
import numpy as np

from src.cnn.config import EMBEDDING_DIM, INPUT_SHAPE, NUM_CLASSES
from src.cnn.model import (
    build_baseline_cnn,
    extract_embeddings,
    predict_classes,
    predict_probabilities,
)
from src.cnn.train import set_random_seed, train_cnn


class TestCNNModel(unittest.TestCase):
    """Test suite for the baseline 1D CNN architecture and training module."""

    def setUp(self):
        set_random_seed(42)
        self.model = build_baseline_cnn()

    def test_01_model_input_and_output_shapes(self):
        """1. Verify model accepts (N, 2048, 1) and outputs (N, 4)."""
        X_dummy = np.random.randn(8, *INPUT_SHAPE).astype(np.float32)
        y_prob = predict_probabilities(self.model, X_dummy)
        y_pred = predict_classes(self.model, X_dummy)

        self.assertEqual(y_prob.shape, (8, NUM_CLASSES))
        self.assertEqual(y_pred.shape, (8,))

    def test_02_probability_properties(self):
        """2. Verify output probabilities are in [0, 1] and each row sums to ~1.0."""
        X_dummy = np.random.randn(10, *INPUT_SHAPE).astype(np.float32)
        y_prob = predict_probabilities(self.model, X_dummy)

        self.assertTrue(np.all(y_prob >= 0.0))
        self.assertTrue(np.all(y_prob <= 1.0))
        self.assertTrue(np.all(np.isfinite(y_prob)))
        row_sums = y_prob.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_03_forward_pass_execution(self):
        """3. Verify model performs forward pass on single and batch inputs."""
        single_sample = np.random.randn(1, *INPUT_SHAPE).astype(np.float32)
        prob_single = predict_probabilities(self.model, single_sample)
        self.assertEqual(prob_single.shape, (1, NUM_CLASSES))

        batch_samples = np.random.randn(16, *INPUT_SHAPE).astype(np.float32)
        prob_batch = predict_probabilities(self.model, batch_samples)
        self.assertEqual(prob_batch.shape, (16, NUM_CLASSES))

    def test_04_embedding_extraction(self):
        """4. Verify penultimate embedding extraction produces shape (N, 64)."""
        X_dummy = np.random.randn(12, *INPUT_SHAPE).astype(np.float32)
        embeddings = extract_embeddings(self.model, X_dummy)

        self.assertEqual(embeddings.shape, (12, EMBEDDING_DIM))
        self.assertTrue(np.all(np.isfinite(embeddings)))

    def test_05_micro_training_run(self):
        """5. Verify model can train on a tiny subset for 2 epochs without error."""
        X_micro_train = np.random.randn(16, *INPUT_SHAPE).astype(np.float32)
        y_micro_train = np.random.choice(NUM_CLASSES, size=16).astype(np.int64)
        X_micro_val = np.random.randn(8, *INPUT_SHAPE).astype(np.float32)
        y_micro_val = np.random.choice(NUM_CLASSES, size=8).astype(np.int64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_cnn.keras")
            history_path = os.path.join(tmp_dir, "history.csv")
            fig_dir = os.path.join(tmp_dir, "figures")

            trained_model, history = train_cnn(
                X_train=X_micro_train,
                y_train=y_micro_train,
                X_val=X_micro_val,
                y_val=y_micro_val,
                epochs=2,
                batch_size=8,
                model_path=model_path,
                history_csv_path=history_path,
                fig_dir=fig_dir,
            )

            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(history_path))
            self.assertIn("train_loss", history)
            self.assertIn("val_loss", history)
            self.assertEqual(len(history["train_loss"]), 2)

    def test_06_reproducibility(self):
        """6. Verify initial predictions are identical when random seed is reused."""
        X_dummy = np.random.randn(5, *INPUT_SHAPE).astype(np.float32)

        set_random_seed(42)
        model_a = build_baseline_cnn()
        preds_a = predict_probabilities(model_a, X_dummy)

        set_random_seed(42)
        model_b = build_baseline_cnn()
        preds_b = predict_probabilities(model_b, X_dummy)

        np.testing.assert_allclose(preds_a, preds_b, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
