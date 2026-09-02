# Novelty Detection Baseline Report

**Project Title**: Information-Centric Uncertainty-Aware Edge Intelligence for Semantic Sensor Communication  
**Module**: Baseline Distance-Based Novelty Detection Module (`src/novelty/`)  
**Date**: August 26, 2026

---

## 1. Purpose of Novelty Detection
In edge intelligence and semantic sensor communication, transmitting continuous raw sensor observations incurs heavy energy and bandwidth costs. Novelty detection quantifies how significantly a new sensor observation deviates from known normal baseline operating behavior. 

The resulting Novelty Score ($N \in [0, 1]$) acts as a primary input parameter to the Value of Information (VoI) Engine, enabling the edge device to prioritize novel or anomalous events for transmission while suppressing redundant normal data.

---

## 2. Why CNN Embeddings Are Used
Raw 2,048-sample vibration signals are high-dimensional, noisy, and challenging to compare directly using raw time-domain metrics. 

By passing vibration windows through the trained 1D Convolutional Neural Network (CNN), the penultimate dense layer extracts a compact **64-dimensional learned feature representation** (`learned_embedding`). In this 64-D embedding space, high-level spectral and temporal health signatures are structured, making distance-based novelty scoring highly effective and explainable.

---

## 3. Novelty Detection Method (V0.1 Baseline)
The baseline module (`DistanceNoveltyDetector`) implements a deterministic distance-to-centroid approach:
1. **Reference Centroid Computation**: Learns the mean 64-D feature vector $\boldsymbol{\mu}_{\text{ref}} \in \mathbb{R}^{64}$ from the normal baseline training embeddings:
   $$\boldsymbol{\mu}_{\text{ref}} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{e}_i^{\text{train, Normal}}$$
2. **Euclidean Distance Calculation**: For any observation embedding $\mathbf{e} \in \mathbb{R}^{64}$, computes unscaled Euclidean distance:
   $$d(\mathbf{e}) = \|\mathbf{e} - \boldsymbol{\mu}_{\text{ref}}\|_2$$
3. **Unit-Interval Normalization**: Scales raw distances using training set min-max bounds ($d_{\min}, d_{\max}$):
   $$N = \text{clip}\left( \frac{d(\mathbf{e}) - d_{\min}}{d_{\max} - d_{\min}}, 0.0, 1.0 \right)$$

---

## 4. Leakage Prevention Discipline
To ensure strict scientific validity and prevent data leakage:
- The reference centroid $\boldsymbol{\mu}_{\text{ref}}$ and normalization bounds $(d_{\min}, d_{\max})$ are learned **strictly from training set embeddings** (`X_train` / `y_train`).
- Validation (`X_val`) and test (`X_test`) set embeddings are **never** used during detector fitting.
- Fits and scores are executed independently per split.

---

## 5. Score Interpretation

| Novelty Score Range | Interpretation | Target System Action |
| :---: | :--- | :--- |
| $N \in [0.0, 0.15)$ | **Normal Baseline** (Highly similar to known normal state) | Low VoI score $\rightarrow$ DISCARD / BUFFER |
| $N \in [0.15, 0.60)$ | **Mild / Moderate Deviation** (Inner race fault signatures) | Medium VoI score $\rightarrow$ SUMMARY / BUFFER |
| $N \in [0.60, 1.00]$ | **High Novelty / Anomaly** (Outer race & ball fault signatures) | High VoI score $\rightarrow$ TRANSMIT |

---

## 6. Experimental Results & Verification

Evaluated on 64-D CNN embeddings extracted from the CWRU processed dataset (`data/processed/cwru/cwru_dataset_v1.npz`):

### Split Summary Table ([`results/tables/novelty_scores_summary.csv`](file:///c:/Users/kingb/Desktop/7th_fyp/results/tables/novelty_scores_summary.csv))

| Data Split / Condition | Sample Count | Mean Score | Min Score | Max Score | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Set Overall** | 1,508 | 0.6875 | 0.0000 | 1.0000 | 0.2401 |
| **Validation Set Overall** | 406 | 0.6393 | 0.0004 | 0.9780 | 0.2921 |
| **Test Set Overall** | 406 | 0.6331 | 0.0000 | 0.9224 | 0.2835 |
| **Test: Class 0 (Normal)** | 58 | **0.0044** | 0.0000 | 0.0140 | 0.0034 |
| **Test: Class 1 (Inner Race Fault)** | 116 | **0.8827** | 0.8205 | 0.9224 | 0.0190 |
| **Test: Class 2 (Ball Fault)** | 116 | **0.5699** | 0.5141 | 0.6153 | 0.0230 |
| **Test: Class 3 (Outer Race Fault)** | 116 | **0.7611** | 0.7271 | 0.7882 | 0.0118 |

### Key Observations
1. **Clear Separation**: Test set Normal samples achieve a mean novelty score of **0.0044** ($\approx 0.0$), whereas fault conditions achieve scores between **0.5699** and **0.8827**.
2. **Deterministic Scaling**: Min-max normalization maps the normal training distribution to 0.0 and caps extreme fault distances at 1.0.

### Generated Figures
- Diagnostic Distribution Plot: [`results/figures/novelty_score_distribution.png`](file:///c:/Users/kingb/Desktop/7th_fyp/results/figures/novelty_score_distribution.png)
- Per-Class Boxplot: [`results/figures/novelty_by_class.png`](file:///c:/Users/kingb/Desktop/7th_fyp/results/figures/novelty_by_class.png)

---

## 7. Limitations & Future Extensions
- **V0.1 Distance Metric**: Uses isotropic Euclidean distance to a single normal centroid.
- **Future Improvements**:
  - Mahalanobis distance to model feature covariances.
  - Multi-class / multi-modal centroids for varying speed and load regimes.
  - One-Class SVM or Autoencoder reconstruction error methods for V0.2.

---

## 8. Connection to Canonical VoI Engine
The calculated Novelty Score $N \in [0, 1]$ directly satisfies the input requirement for the canonical VoI Engine (`src/voi/ scoring.py`):

$$\text{VoI}_{\text{raw}} = w_N N + w_U U + w_R R + w_T T - w_C C$$

In upcoming integration phases, this real sensor novelty score $N$ will combine with real model uncertainty $U$ to drive semantic transmission decisions.
