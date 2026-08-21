"""Baseline CNN training, validation, and evaluation pipeline for CWRU dataset.

Ensures strict test-set discipline:
    - Train set is used for backpropagation.
    - Validation set is used for epoch-level monitoring and generalization check.
    - Test set is strictly isolated and evaluated only once after training is complete.
"""

import os
import random
from typing import Dict, List, Tuple
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from src.cnn.config import (
    BATCH_SIZE,
    DATA_PATH,
    EPOCHS,
    FIGURE_DIR,
    INPUT_SHAPE,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_PATH,
    NUM_CLASSES,
    RANDOM_SEED,
    TABLE_DIR,
    TRAINING_HISTORY_PATH,
)
from src.cnn.model import (
    build_baseline_cnn,
    predict_classes,
    predict_probabilities,
)
from src.evaluation.cnn_evaluation import (
    CNNEvaluationResult,
    evaluate_classifier,
    plot_class_performance,
    plot_confusion_matrix,
    plot_training_history,
    save_classification_report,
    save_evaluation_summary,
)


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set global seeds across Python, NumPy, TensorFlow, and Keras for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def load_cwru_dataset(
    data_path: str = DATA_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed CWRU dataset splits from .npz file.

    Args:
        data_path: Path to cwru_dataset_v1.npz.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed CWRU dataset not found at '{data_path}'. "
            "Run 'python -m src.cwru_pipeline.preprocessing' first."
        )

    with np.load(data_path) as data:
        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        X_val = data["X_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.int64)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
    model_path: str = MODEL_PATH,
    history_csv_path: str = TRAINING_HISTORY_PATH,
    fig_dir: str = FIGURE_DIR,
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """Train the baseline 1D CNN model and save checkpoints and history curves.

    Strict Isolation:
        The test set (X_test, y_test) is NEVER passed to this training function.

    Args:
        X_train: Training input tensor of shape (N_train, 2048, 1).
        y_train: Training label array of shape (N_train,).
        X_val: Validation input tensor of shape (N_val, 2048, 1).
        y_val: Validation label array of shape (N_val,).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Optimizer learning rate.
        seed: Random seed.
        model_path: Destination path for saving trained model.
        history_csv_path: Destination path for saving training history CSV.
        fig_dir: Directory for training curve plot.

    Returns:
        Tuple[keras.Model, Dict[str, List[float]]]: Trained Keras model and history dictionary.
    """
    set_random_seed(seed)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Building baseline 1D CNN model (input_shape={INPUT_SHAPE}, classes={NUM_CLASSES})...")
    model = build_baseline_cnn(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=learning_rate,
    )

    print(
        f"Training CNN on {len(X_train)} samples, validating on {len(X_val)} samples "
        f"({epochs} epochs, batch_size={batch_size}, lr={learning_rate})..."
    )

    fit_history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    history_dict = {
        "train_loss": [float(v) for v in fit_history.history["loss"]],
        "val_loss": [float(v) for v in fit_history.history["val_loss"]],
        "train_accuracy": [float(v) for v in fit_history.history["accuracy"]],
        "val_accuracy": [float(v) for v in fit_history.history["val_accuracy"]],
    }

    # Save trained model
    model.save(model_path)
    print(f"Saved trained CNN model to: {model_path}")

    # Save training history to CSV
    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, epochs + 1)),
            "train_loss": history_dict["train_loss"],
            "val_loss": history_dict["val_loss"],
            "train_accuracy": history_dict["train_accuracy"],
            "val_accuracy": history_dict["val_accuracy"],
        }
    )
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training history to: {history_csv_path}")

    # Plot and save training history curves using existing evaluation framework
    training_curves_path = os.path.join(fig_dir, "cnn_training_curves.png")
    plot_training_history(history_dict, save_path=training_curves_path)
    print(f"Saved training curves to: {training_curves_path}")

    return model, history_dict


def evaluate_cnn_on_test_set(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fig_dir: str = FIGURE_DIR,
    table_dir: str = TABLE_DIR,
) -> CNNEvaluationResult:
    """Evaluate trained model on the isolated test set using the evaluation framework.

    Args:
        model: Trained CNN model.
        X_test: Test input tensor of shape (N_test, 2048, 1).
        y_test: Test label array of shape (N_test,).
        fig_dir: Directory for evaluation plots.
        table_dir: Directory for evaluation tables.

    Returns:
        CNNEvaluationResult: Structured evaluation metrics result.
    """
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    print(f"Evaluating model on {len(X_test)} test observations...")
    y_prob = predict_probabilities(model, X_test)
    y_pred = predict_classes(model, X_test)

    # Call canonical evaluation framework
    result = evaluate_classifier(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    # Generate figures
    cm_path = os.path.join(fig_dir, "cnn_confusion_matrix.png")
    plot_confusion_matrix(result.confusion_matrix, save_path=cm_path)
    print(f"Saved confusion matrix plot to: {cm_path}")

    perf_path = os.path.join(fig_dir, "cnn_class_performance.png")
    plot_class_performance(result.per_class_metrics, save_path=perf_path)
    print(f"Saved class performance plot to: {perf_path}")

    # Save tables
    report_csv = os.path.join(table_dir, "cnn_classification_report.csv")
    save_classification_report(result, save_path=report_csv)

    summary_csv = os.path.join(table_dir, "cnn_evaluation_summary.csv")
    save_evaluation_summary(result, save_path=summary_csv)

    return result


def run_training_pipeline(
    data_path: str = DATA_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
) -> Tuple[keras.Model, CNNEvaluationResult]:
    """Execute complete end-to-end baseline CNN training and test evaluation pipeline.

    Args:
        data_path: Path to processed CWRU dataset.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Optimizer learning rate.
        seed: Random seed.

    Returns:
        Tuple[keras.Model, CNNEvaluationResult]: Trained model and test evaluation result.
    """
    print("=== Executing CWRU Baseline CNN Training Pipeline ===")

    # 1. Load preprocessed dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_cwru_dataset(data_path)
    print(
        f"Dataset loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )

    # 2. Train model (using train and val sets only)
    model, history = train_cnn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    # 3. Final evaluation (using isolated test set only)
    test_result = evaluate_cnn_on_test_set(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n--- Final Test Set Evaluation Results ---")
    print(f"Accuracy:         {test_result.accuracy * 100:.2f}%")
    print(f"Macro F1-Score:   {test_result.macro_f1:.4f}")
    print(f"Weighted F1-Score:{test_result.weighted_f1:.4f}")
    print("\nPer-Class Performance:")
    for cname, m in test_result.per_class_metrics.items():
        print(
            f"  {cname:18s} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | "
            f"F1: {m['f1_score']:.4f} | Support: {m['support']}"
        )

    print("=== CWRU Baseline CNN Pipeline Complete ===")
    return model, test_result


if __name__ == "__main__":
    run_training_pipeline()
