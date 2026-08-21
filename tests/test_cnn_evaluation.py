"""Unit test suite for the CNN evaluation framework (src/evaluation/cnn_evaluation.py).

Tests are model-agnostic: all predictions are controlled synthetic arrays.
No actual CNN is trained here. These tests validate only the evaluation framework itself.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation.cnn_evaluation import (
    CLASS_NAMES,
    NUM_CLASSES,
    CNNEvaluationResult,
    evaluate_classifier,
    plot_class_performance,
    plot_confusion_matrix,
    plot_training_history,
    save_classification_report,
    save_evaluation_summary,
    validate_predictions,
    validate_probabilities,
)


class TestCNNEvaluation(unittest.TestCase):
    """Test suite for the model-agnostic CNN evaluation framework."""

    # ------------------------------------------------------------------
    # Test 1: Perfect predictions
    # ------------------------------------------------------------------
    def test_01_perfect_predictions(self):
        """1. All predictions correct → accuracy must equal 1.0."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        y_pred = y_true.copy()
        result = evaluate_classifier(y_true, y_pred)

        self.assertAlmostEqual(result.accuracy, 1.0, places=6)
        self.assertAlmostEqual(result.macro_f1, 1.0, places=6)
        self.assertAlmostEqual(result.weighted_f1, 1.0, places=6)

        for class_name in CLASS_NAMES:
            self.assertAlmostEqual(result.per_class_metrics[class_name]["precision"], 1.0, places=6)
            self.assertAlmostEqual(result.per_class_metrics[class_name]["recall"], 1.0, places=6)
            self.assertAlmostEqual(result.per_class_metrics[class_name]["f1_score"], 1.0, places=6)

    # ------------------------------------------------------------------
    # Test 2: Completely incorrect predictions
    # ------------------------------------------------------------------
    def test_02_completely_incorrect_predictions(self):
        """2. All predictions wrong → accuracy must equal 0.0."""
        y_true = np.array([0, 0, 0, 0], dtype=np.int64)
        y_pred = np.array([1, 2, 3, 1], dtype=np.int64)
        result = evaluate_classifier(y_true, y_pred)

        self.assertAlmostEqual(result.accuracy, 0.0, places=6)
        # Class 0 recall should be 0
        self.assertAlmostEqual(result.per_class_metrics["Normal"]["recall"], 0.0, places=6)

    # ------------------------------------------------------------------
    # Test 3: Confusion matrix shape and values
    # ------------------------------------------------------------------
    def test_03_confusion_matrix_shape_and_counts(self):
        """3. Confusion matrix must be (4, 4) with correct cell counts."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 3], dtype=np.int64)
        result = evaluate_classifier(y_true, y_pred)

        cm = result.confusion_matrix
        self.assertEqual(cm.shape, (NUM_CLASSES, NUM_CLASSES))
        # y_true=0, y_pred=0 → cell [0,0] = 1
        self.assertEqual(cm[0, 0], 1)
        # y_true=0, y_pred=1 → cell [0,1] = 1
        self.assertEqual(cm[0, 1], 1)
        # y_true=1, y_pred=1 → cell [1,1] = 2
        self.assertEqual(cm[1, 1], 2)
        # Row sums must match actual class counts
        np.testing.assert_array_equal(cm.sum(axis=1), [2, 2, 2, 2])

    # ------------------------------------------------------------------
    # Test 4: Per-class metrics all four classes present
    # ------------------------------------------------------------------
    def test_04_per_class_metrics_all_classes_present(self):
        """4. All four class names must appear in per_class_metrics."""
        y_true = np.array([0, 1, 2, 3], dtype=np.int64)
        y_pred = np.array([0, 1, 2, 3], dtype=np.int64)
        result = evaluate_classifier(y_true, y_pred)

        for class_name in CLASS_NAMES:
            self.assertIn(class_name, result.per_class_metrics)
            metrics = result.per_class_metrics[class_name]
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1_score", metrics)
            self.assertIn("support", metrics)

    # ------------------------------------------------------------------
    # Test 5: Class imbalance — macro vs weighted diverge
    # ------------------------------------------------------------------
    def test_05_class_imbalance_macro_vs_weighted(self):
        """5. Imbalanced dataset: macro and weighted averages must differ."""
        # 100 Normal samples, 10 of each fault class
        y_true = np.array(
            [0] * 100 + [1] * 10 + [2] * 10 + [3] * 10, dtype=np.int64
        )
        # Predict everything as Normal → only class 0 recall is 1.0
        y_pred = np.zeros_like(y_true)

        result = evaluate_classifier(y_true, y_pred)

        # Weighted recall should be higher (dominated by class 0 which is perfect)
        # Macro recall should be lower (classes 1,2,3 recall = 0)
        self.assertGreater(result.weighted_recall, result.macro_recall)

    # ------------------------------------------------------------------
    # Test 6: Probability validation
    # ------------------------------------------------------------------
    def test_06_probability_validation(self):
        """6. Probability matrix validation catches all structural violations."""
        y_true = np.array([0, 1, 2, 3], dtype=np.int64)
        y_pred = np.array([0, 1, 2, 3], dtype=np.int64)
        valid_prob = np.array([
            [0.90, 0.05, 0.03, 0.02],
            [0.10, 0.70, 0.15, 0.05],
            [0.05, 0.10, 0.80, 0.05],
            [0.02, 0.03, 0.05, 0.90],
        ], dtype=np.float32)

        # Valid probabilities should pass
        result = evaluate_classifier(y_true, y_pred, y_prob=valid_prob)
        np.testing.assert_array_equal(result.y_prob, valid_prob)

        # Values outside [0, 1]
        bad_range = valid_prob.copy()
        bad_range[0, 0] = 1.5
        with self.assertRaises(ValueError):
            validate_probabilities(bad_range, n_samples=4)

        # Rows not summing to 1
        bad_sum = valid_prob.copy()
        bad_sum[0, 0] = 0.50  # row sum now ~0.60
        with self.assertRaises(ValueError):
            validate_probabilities(bad_sum, n_samples=4)

        # NaN values
        nan_prob = valid_prob.copy()
        nan_prob[1, 1] = np.nan
        with self.assertRaises(ValueError):
            validate_probabilities(nan_prob, n_samples=4)

        # Wrong number of samples
        with self.assertRaises(ValueError):
            validate_probabilities(valid_prob, n_samples=99)

        # Wrong number of classes (3 instead of 4)
        wrong_classes = valid_prob[:, :3]
        with self.assertRaises(ValueError):
            validate_probabilities(wrong_classes, n_samples=4)

    # ------------------------------------------------------------------
    # Test 7: Training history plot generation
    # ------------------------------------------------------------------
    def test_07_training_history_plot(self):
        """7. Training history dict with loss and accuracy produces a PNG figure."""
        history = {
            "train_loss": [0.9, 0.6, 0.4, 0.3, 0.25],
            "val_loss": [1.0, 0.7, 0.5, 0.45, 0.40],
            "train_accuracy": [0.5, 0.65, 0.75, 0.80, 0.84],
            "val_accuracy": [0.45, 0.60, 0.70, 0.73, 0.75],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = os.path.join(tmp_dir, "training_curves.png")
            plot_training_history(history, save_path=fig_path)
            self.assertTrue(os.path.exists(fig_path))

        # Empty history → ValueError
        with self.assertRaises(ValueError):
            plot_training_history({})

    # ------------------------------------------------------------------
    # Test 8: Empty / invalid input
    # ------------------------------------------------------------------
    def test_08_empty_and_invalid_inputs(self):
        """8. Empty or mismatched inputs raise meaningful errors."""
        # Empty arrays
        with self.assertRaises(ValueError):
            validate_predictions(np.array([], dtype=np.int64), np.array([], dtype=np.int64))

        # Mismatched shapes
        with self.assertRaises(ValueError):
            validate_predictions(
                np.array([0, 1, 2], dtype=np.int64),
                np.array([0, 1], dtype=np.int64),
            )

        # Non-array input
        with self.assertRaises(TypeError):
            validate_predictions([0, 1, 2], np.array([0, 1, 2]))  # type: ignore

        # 2D y_true
        with self.assertRaises(ValueError):
            validate_predictions(
                np.array([[0, 1], [2, 3]], dtype=np.int64),
                np.array([[0, 1], [2, 3]], dtype=np.int64),
            )

    # ------------------------------------------------------------------
    # Test 9: Confusion matrix plot and report export
    # ------------------------------------------------------------------
    def test_09_plot_and_export_outputs(self):
        """9. Confusion matrix plot, class performance plot, and CSV exports are created."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3], dtype=np.int64)
        result = evaluate_classifier(y_true, y_pred)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = os.path.join(tmp_dir, "figures", "confusion_matrix.png")
            plot_confusion_matrix(result.confusion_matrix, save_path=cm_path)
            self.assertTrue(os.path.exists(cm_path))

            perf_path = os.path.join(tmp_dir, "figures", "class_performance.png")
            plot_class_performance(result.per_class_metrics, save_path=perf_path)
            self.assertTrue(os.path.exists(perf_path))

            report_path = os.path.join(tmp_dir, "tables", "report.csv")
            save_classification_report(result, save_path=report_path)
            self.assertTrue(os.path.exists(report_path))
            df = pd.read_csv(report_path)
            self.assertIn("class", df.columns)

            summary_path = os.path.join(tmp_dir, "tables", "summary.csv")
            save_evaluation_summary(result, save_path=summary_path)
            self.assertTrue(os.path.exists(summary_path))
            df2 = pd.read_csv(summary_path)
            self.assertIn("accuracy", df2.columns)


if __name__ == "__main__":
    unittest.main()
