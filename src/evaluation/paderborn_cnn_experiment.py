"""Paderborn CNN training and evaluation, reusing the canonical CWRU CNN architecture
and evaluation framework unmodified, parameterized for 3 Paderborn fault-location classes.
"""

import os
from typing import Dict, List, Tuple

import keras
import numpy as np

from src.cnn.config import BATCH_SIZE, EPOCHS, INPUT_SHAPE, LEARNING_RATE, RANDOM_SEED
from src.cnn.model import build_baseline_cnn, extract_embeddings, predict_classes, predict_probabilities
from src.cnn.train import set_random_seed
from src.evaluation.cnn_evaluation import (
    CNNEvaluationResult,
    evaluate_classifier,
    plot_class_performance,
    plot_confusion_matrix,
    plot_training_history,
    save_classification_report,
    save_evaluation_summary,
)
from src.paderborn_pipeline.classification_task import CLASS_NAMES, DATASET_V1_PATH

NUM_CLASSES = len(CLASS_NAMES)
ORDERED_CLASS_NAMES: List[str] = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

MODEL_PATH = "models/paderborn_cnn_baseline.keras"
TABLE_DIR = "results/tables"
FIGURE_DIR = "results/figures"
TRAINING_HISTORY_PATH = os.path.join(TABLE_DIR, "paderborn_cnn_training_history.csv")


def load_paderborn_dataset(
    data_path: str = DATASET_V1_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed Paderborn dataset not found at '{data_path}'. "
            "Run 'python -m src.paderborn_pipeline.classification_task' first."
        )
    with np.load(data_path) as data:
        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        X_val = data["X_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.int64)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_paderborn_cnn(
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
    set_random_seed(seed)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    model = build_baseline_cnn(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=learning_rate,
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

    model.save(model_path)

    import pandas as pd

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

    plot_training_history(history_dict, save_path=os.path.join(fig_dir, "paderborn_cnn_training_curves.png"))

    return model, history_dict


def evaluate_paderborn_cnn(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fig_dir: str = FIGURE_DIR,
    table_dir: str = TABLE_DIR,
) -> CNNEvaluationResult:
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    y_prob = predict_probabilities(model, X_test)
    y_pred = predict_classes(model, X_test)

    result = evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        class_names=ORDERED_CLASS_NAMES,
        y_prob=y_prob,
        num_classes=NUM_CLASSES,
    )

    plot_confusion_matrix(
        result.confusion_matrix,
        class_names=ORDERED_CLASS_NAMES,
        save_path=os.path.join(fig_dir, "paderborn_cnn_confusion_matrix.png"),
    )
    plot_class_performance(
        result.per_class_metrics,
        save_path=os.path.join(fig_dir, "paderborn_cnn_class_performance.png"),
    )
    save_classification_report(result, save_path=os.path.join(table_dir, "paderborn_cnn_classification_report.csv"))
    save_evaluation_summary(result, save_path=os.path.join(table_dir, "paderborn_cnn_evaluation_summary.csv"))

    return result


def run_paderborn_cnn_pipeline(
    data_path: str = DATASET_V1_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
) -> Tuple[keras.Model, CNNEvaluationResult]:
    print("=== Executing Paderborn CNN Training Pipeline (canonical architecture, 3 classes) ===")

    X_train, y_train, X_val, y_val, X_test, y_test = load_paderborn_dataset(data_path)
    print(f"Dataset loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    model, history = train_paderborn_cnn(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed,
    )

    test_result = evaluate_paderborn_cnn(model=model, X_test=X_test, y_test=y_test)

    print(f"Test accuracy: {test_result.accuracy * 100:.2f}%  Macro F1: {test_result.macro_f1:.4f}")
    for cname, m in test_result.per_class_metrics.items():
        print(f"  {cname:20s} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | Support: {m['support']}")

    print("=== Paderborn CNN Pipeline Complete ===")
    return model, test_result


if __name__ == "__main__":
    run_paderborn_cnn_pipeline()
