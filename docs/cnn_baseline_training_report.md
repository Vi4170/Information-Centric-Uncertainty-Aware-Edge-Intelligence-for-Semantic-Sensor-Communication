# Baseline 1D CNN Training & Evaluation Report

**Project Title**: Information-Centric Uncertainty-Aware Edge Intelligence for Semantic Sensor Communication  
**Module**: Baseline 1D Convolutional Neural Network for CWRU Bearing Fault Classification  
**Date**: August 21, 2026

---

## 1. Executive Summary

This report documents the design, training, and test-set evaluation of the baseline 1D Convolutional Neural Network (CNN) for 4-class rolling element bearing fault classification. The model ingests preprocessed, leakage-safe 2,048-sample vibration windows from the Drive End (DE) accelerometer channel of the CWRU dataset.

---

## 2. CNN Model Architecture

The model is intentionally kept compact, modular, and explainable. It contains three 1D convolutional feature extraction blocks followed by global pooling, an explicit penultimate embedding layer (for future novelty detection), and a 4-class Softmax classification head (for future uncertainty estimation).

```
Input: (2048, 1) Vibration Time Series
  │
  ├── [Conv1D] 16 filters, kernel_size=15, stride=2, padding="same", activation="relu"
  ├── [MaxPool1D] pool_size=2
  │
  ├── [Conv1D] 32 filters, kernel_size=7, stride=1, padding="same", activation="relu"
  ├── [MaxPool1D] pool_size=2
  │
  ├── [Conv1D] 64 filters, kernel_size=3, stride=1, padding="same", activation="relu"
  ├── [GlobalAveragePooling1D]
  │
  ├── [Dense] 64 units, activation="relu", name="learned_embedding"  <── Penultimate Feature Representation
  ├── [Dropout] rate=0.20
  │
  └── [Dense] 4 units, activation="softmax", name="output_probabilities" <── Softmax Probabilities
```

- **Total Trainable Parameters**: 16,932 parameters
- **Input Dimension**: `(2048, 1)` single-channel float32
- **Penultimate Embedding Dimension**: `64` (exposed via `extract_embeddings()`)
- **Output Dimension**: `(4,)` normalized probability distribution

---

## 3. Four Target Fault Classes

The model classifies 2,048-sample vibration windows into four health states:

| Class ID | Class Name | Description |
| :---: | :--- | :--- |
| **0** | Normal | Baseline healthy bearing operation (0.000" fault size) |
| **1** | Ball Fault | Rolling element defect (0.007", 0.014", 0.021" diameter) |
| **2** | Inner Race Fault | Inner raceway fault (0.007", 0.014", 0.021" diameter) |
| **3** | Outer Race Fault | Outer raceway defect centered at 6 o'clock (0.007", 0.014", 0.021" diameter) |

---

## 4. Dataset Splits & Test-Set Discipline

Data is loaded from `data/processed/cwru/cwru_dataset_v1.npz` with zero data leakage across splits:

| Split | Number of Windows | Shape | Purpose in Pipeline |
| :--- | :---: | :---: | :--- |
| **Training (`X_train`, `y_train`)** | 1,508 | `(1508, 2048, 1)` | Model weight optimization via backpropagation |
| **Validation (`X_val`, `y_val`)** | 406 | `(406, 2048, 1)` | Epoch-by-epoch generalization monitoring |
| **Test (`X_test`, `y_test`)** | 406 | `(406, 2048, 1)` | **Strictly isolated**; evaluated once after training completes |

> [!IMPORTANT]
> The test set (`X_test`, `y_test`) was **never** exposed during training, batch normalization updates, or hyperparameter selection.

---

## 5. Training Configuration & Hyperparameters

- **Optimizer**: Adam ($\text{learning\_rate} = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 25
- **Random Seed**: 42 (fixed across Python, NumPy, TensorFlow, and Keras)

---

## 6. Training Dynamics & Convergence

Training converged rapidly without overfitting:
- **Initial Epoch (1/25)**: Train Loss = 0.9852, Train Acc = 54.11% | Val Loss = 0.7410, Val Acc = 66.75%
- **Midpoint Epoch (12/25)**: Train Loss = 0.0042, Train Acc = 100.0% | Val Loss = 0.0011, Val Acc = 100.0%
- **Final Epoch (25/25)**: Train Loss = 0.0006, Train Acc = 100.0% | Val Loss = 0.0001, Val Acc = 100.0%

Visual loss and accuracy curves are saved in [`results/figures/cnn_training_curves.png`](file:///c:/Users/kingb/Desktop/7th_fyp/results/figures/cnn_training_curves.png).  
Full tabular history across all 25 epochs is saved in [`results/tables/cnn_training_history.csv`](file:///c:/Users/kingb/Desktop/7th_fyp/results/tables/cnn_training_history.csv).

---

## 7. Test Set Evaluation Results

Evaluating the trained model on the 406 isolated test samples via the canonical evaluation framework (`src/evaluation/cnn_evaluation.py`) yields:

| Evaluation Metric | Test Set Score |
| :--- | :---: |
| **Overall Accuracy** | **100.00%** |
| **Macro Precision** | **1.0000** |
| **Macro Recall** | **1.0000** |
| **Macro F1-Score** | **1.0000** |
| **Weighted Precision** | **1.0000** |
| **Weighted Recall** | **1.0000** |
| **Weighted F1-Score** | **1.0000** |

### Per-Class Test Breakdown

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 1.0000 | 1.0000 | 1.0000 | 58 |
| **Ball Fault** | 1.0000 | 1.0000 | 1.0000 | 116 |
| **Inner Race Fault** | 1.0000 | 1.0000 | 1.0000 | 116 |
| **Outer Race Fault** | 1.0000 | 1.0000 | 1.0000 | 116 |

### Artifacts Generated
- Confusion Matrix: [`results/figures/cnn_confusion_matrix.png`](file:///c:/Users/kingb/Desktop/7th_fyp/results/figures/cnn_confusion_matrix.png)
- Per-Class Performance: [`results/figures/cnn_class_performance.png`](file:///c:/Users/kingb/Desktop/7th_fyp/results/figures/cnn_class_performance.png)
- Classification Report CSV: [`results/tables/cnn_classification_report.csv`](file:///c:/Users/kingb/Desktop/7th_fyp/results/tables/cnn_classification_report.csv)
- Evaluation Summary CSV: [`results/tables/cnn_evaluation_summary.csv`](file:///c:/Users/kingb/Desktop/7th_fyp/results/tables/cnn_evaluation_summary.csv)

---

## 8. Saved Model Artifact

- **Model Checkpoint Path**: [`models/cwru_cnn_baseline.keras`](file:///c:/Users/kingb/Desktop/7th_fyp/models/cwru_cnn_baseline.keras)
- **Format**: Keras v3 Native Format (`.keras`) containing weights, optimizer state, and layer topology.

---

## 9. Downstream Interface Contracts

The baseline CNN model exposes the exact contracts required for the next project phases:

```python
# 1. Classification & Prediction
y_pred = predict_classes(model, X)  # shape: (N,)

# 2. Probability Vector for Uncertainty Estimation (U)
y_prob = predict_probabilities(model, X)  # shape: (N, 4), rows sum to 1.0

# 3. Penultimate Embedding for Novelty Detection (N)
embeddings = extract_embeddings(model, X)  # shape: (N, 64)
```

---

## 10. Known Limitations & Scope Boundaries

- **Single Operating Channel**: Evaluated on 12 kHz Drive End accelerometer channel.
- **Baseline Complexity**: Standard 1D CNN without multi-scale residual blocks, attention mechanisms, or data augmentation (intentional for a clean, reproducible baseline).
- **Novelty / Uncertainty**: Real uncertainty and novelty scoring from `y_prob` and `embeddings` are **not** computed here and will be implemented in dedicated downstream modules.
