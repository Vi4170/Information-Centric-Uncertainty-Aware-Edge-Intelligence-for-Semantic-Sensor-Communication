# Uncertainty Estimation Baseline Report

**Project Title**: Information-Centric Uncertainty-Aware Edge Intelligence for Semantic Sensor Communication  
**Module**: Baseline Predictive Entropy Uncertainty Module (`src/uncertainty/`)  
**Date**: August 26, 2026

---

## 1. Purpose of Uncertainty Estimation
In edge-based semantic sensor communication, determining whether sensor data is valuable enough to transmit depends not only on novelty but also on model confidence. 

Uncertainty Estimation quantifies the predictive ambiguity of the machine learning model. If an edge model is highly uncertain about its classification of a sensor observation, that observation carries higher information value ($U \in [0, 1]$), justifying transmission to a high-capacity central node for expert analysis.

---

## 2. Why CNN Probabilities Are Used
The baseline 1D CNN outputs a 4-class Softmax probability vector $\mathbf{p} = [p_0, p_1, p_2, p_3]^T$ where $\sum_{i=0}^3 p_i = 1.0$. 

These probabilities reflect how the model distributes its confidence across the four bearing health states (Normal, Ball Fault, Inner Race Fault, Outer Race Fault). When the model is confident, one probability dominates ($p_k \approx 1.0$). When the model is ambiguous, probabilities are dispersed across multiple classes.

---

## 3. Entropy-Based Method & Normalization
The baseline module (`src/uncertainty/uncertainty.py`) uses normalized Shannon Predictive Entropy:

1. **Shannon Entropy Calculation**:
   $$H(\mathbf{p}) = -\sum_{i=0}^{3} p_i \log_2(p_i)$$
   *(with safe $\varepsilon = 10^{-12}$ clipping near zero).*

2. **Unit-Interval Normalization**:
   Divided by the maximum theoretical entropy for a 4-class uniform distribution ($\log_2(4) = 2.0$):
   $$U = \frac{-\sum_{i=0}^{3} p_i \log_2(p_i)}{\log_2(4)} = \frac{-\sum_{i=0}^{3} p_i \log_2(p_i)}{2.0}$$

3. **Range Constraint**:
   Strictly clipped to $U \in [0.0, 1.0]$.

---

## 4. Score Interpretation

| Uncertainty Score ($U$) | Predictive State | System Interpretation |
| :---: | :--- | :--- |
| $U \approx 0.0$ | **Completely Confident** (e.g. $[1, 0, 0, 0]$) | Model prediction is clear and unambiguous. |
| $U \in (0.0, 0.50)$ | **Low to Moderate Ambiguity** | Slight probability spread across classes. |
| $U \to 1.0$ | **Maximum Uncertainty** (e.g. $[0.25, 0.25, 0.25, 0.25]$) | Model cannot confidently distinguish class. |

---

## 5. Experimental Results

Evaluated on 4-class Softmax probabilities extracted from `models/cwru_cnn_baseline.keras` evaluated on `data/processed/cwru/cwru_dataset_v1.npz`:

### Summary Table ([`results/tables/uncertainty_scores_summary.csv`](file:///c:/Users/kingb/Desktop/7th_fyp/results/tables/uncertainty_scores_summary.csv))

| Data Split / Condition | Sample Count | Mean Score | Median Score | Min Score | Max Score | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set Overall** | 1,508 | 0.000544 | 0.000243 | 0.000000 | 0.052570 | 0.001843 |
| **Validation Set Overall** | 406 | 0.000604 | 0.000230 | 0.000000 | 0.015722 | 0.001371 |
| **Test Set Overall** | 406 | 0.000871 | 0.000263 | 0.000001 | 0.028488 | 0.002695 |
| **Test: Class 0 (Normal)** | 58 | 0.000049 | 0.000048 | 0.000039 | 0.000058 | 0.000004 |
| **Test: Class 1 (Ball Fault)** | 116 | 0.000012 | 0.000008 | 0.000001 | 0.000134 | 0.000015 |
| **Test: Class 2 (Inner Race)** | 116 | **0.002443** | 0.000658 | 0.000159 | 0.028488 | 0.004647 |
| **Test: Class 3 (Outer Race)** | 116 | 0.000568 | 0.000462 | 0.000177 | 0.003945 | 0.000428 |

### Key Observations
1. **High Model Confidence**: Across all dataset splits, the mean predictive entropy is $< 0.001$, reflecting the baseline CNN's strong classification performance ($100\%$ test accuracy).
2. **Relative Class Variation**: Inner Race Fault (Class 2) exhibits the highest relative uncertainty among test samples (mean $U = 0.002443$), whereas Ball Fault (Class 1) exhibits the lowest uncertainty (mean $U = 0.000012$).

### Generated Visualizations
- Distribution Plot: [`results/figures/uncertainty_score_distribution.png`](file:///c:/Users/kingb/Desktop/7th_fyp/results/figures/uncertainty_score_distribution.png)
- Per-Class Boxplot: [`results/figures/uncertainty_by_class.png`](file:///c:/Users/kingb/Desktop/7th_fyp/results/figures/uncertainty_by_class.png)

---

## 6. Limitations & Future Extensions
- **V0.1 Method**: Predictive entropy measures softmax output dispersion (epistemic/aleatoric proxy).
- **Future Improvements**:
  - Monte Carlo (MC) Dropout entropy for model variance estimation.
  - Deep Ensembles variance.
  - Conformal Prediction sets.

---

## 7. Connection to Canonical VoI Engine
The Uncertainty Score $U \in [0, 1]$ directly satisfies the input requirement for the canonical VoI Engine (`src/voi/scoring.py`):

$$\text{VoI}_{\text{raw}} = w_N N + w_U U + w_R R + w_T T - w_C C$$

Together with the Novelty Score $N$, this real sensor uncertainty $U$ will contribute to dynamic Value of Information scoring in future integration phases.
