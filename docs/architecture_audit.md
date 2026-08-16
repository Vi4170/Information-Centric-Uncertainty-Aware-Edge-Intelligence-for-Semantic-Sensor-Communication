# Comprehensive Architecture Audit & Repository Status Report

**Project Title**: Information-Centric Uncertainty-Aware Edge Intelligence for Semantic Sensor Communication  
**Module**: System Architecture Audit, Interface Contracts, & Roadmap Document  
**Date**: August 16, 2026

---

## 1. Executive Summary

This document presents the complete architectural audit of the repository. A repository-wide content search and dependency trace were performed to map implemented components, identify missing modules, establish formal input/output contracts, and protect the canonical Value of Information (VoI) Engine.

### Key Audit Findings
1. **Canonical VoI Engine (`src/voi/`)**: Fully implemented, modularized, unit-tested (15/15 tests passing in `tests/test_voi.py`), and validated across 15 diagnostic stress tests.
2. **Synthetic Data Generator & Diagnostic Suite (`src/data_generation/`, `src/evaluation/`)**: Fully implemented and executing successfully, producing 1,000 synthetic observations across 6 behavioral scenarios and generating 11 PNG research figures and 8 CSV summary tables.
3. **Repository Content Search Result**: A deep keyword search for `CWRU`, `cwru`, `CNN`, `Conv1D`, `IsolationForest`, `OneClassSVM`, `Autoencoder`, and `anomaly` yielded zero occurrences in the source codebase. CWRU dataset handling, 2048-sample windowing, CNN feature extraction, and learning-based novelty/uncertainty estimation are currently **PLANNED / NOT YET IMPLEMENTED**.
4. **Zero Duplicate VoI Logic**: No duplicate, bypassing, or conflicting VoI scoring or decision logic exists in the workspace. The canonical implementation in `src/voi/` is confirmed as the single source of truth.
5. **No Premature Integration**: Missing components (CWRU preprocessing, CNN embeddings, real uncertainty, task relevance, temporal importance, communication cost) are cleanly documented with formal interface specifications rather than inserting placeholder or fake code.

---

## 2. Current Repository Structure

```
.
├── README.md                           # Project documentation & overview
├── requirements.txt                    # Project dependencies (numpy, pandas, matplotlib)
├── .gitignore                          # Standard git ignore file
│
├── src/
│   ├── voi/                            # CANONICAL VOI ENGINE (Protected)
│   │   ├── __init__.py
│   │   ├── normalization.py            # Numeric validation, bounds checking, unit-interval clipping
│   │   ├── scoring.py                  # Linear VoI scoring equation & ScoringResult container
│   │   ├── decision_policy.py          # DecisionAction enum & configurable PolicyThresholds
│   │   └── voi_engine.py               # High-level VoIEngine orchestrator (single & batch)
│   │
│   ├── data_generation/                # Synthetic data generation module
│   │   ├── __init__.py
│   │   └── synthetic_generator.py      # Reproducible 1,000-observation generator (seed 42, 6 scenarios)
│   │
│   ├── evaluation/                     # Diagnostic evaluation & plotting suite
│   │   ├── __init__.py
│   │   ├── diagnostics.py              # 15 diagnostic stress-testing functions
│   │   └── run_experiment.py           # Master experiment execution script
│   │
│   └── utils/                          # Common utility module placeholder
│       └── __init__.py
│
├── data/
│   └── synthetic/
│       └── synthetic_voi_dataset.csv   # Generated 1,000-row synthetic dataset
│
├── experiments/
│   └── voi/                            # Experiment script output directory
│
├── notebooks/
│   └── 01_synthetic_voi.ipynb          # Interactive research notebook with full diagnostic pipeline
│
├── results/
│   ├── figures/                        # 11 generated research plot figures (PNG)
│   ├── tables/                         # 8 generated evaluation summary tables (CSV)
│   └── logs/                           # System execution log directory
│
├── docs/
│   ├── architecture_audit.md           # Master architectural audit report (this file)
│   └── project_status.md               # Machine-readable status checklist
│
└── tests/
    └── test_voi.py                     # 15 automated unit tests for VoI Engine
```

---

## 3. Current vs Target Architecture

### Diagram A: Current Implemented Architecture
Shows ONLY components that currently exist, execute, and pass unit tests:

```
┌─────────────────────────────────────────────────────────────┐
│                    SYNTHETIC GENERATOR                      │
│        (src/data_generation/synthetic_generator.py)         │
│          Generates 1,000 Observations (Seed 42)             │
└──────────────────────────────┬──────────────────────────────┘
                               │ (N, U, R, T, C)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    CANONICAL VOI ENGINE                     │
│                        (src/voi/)                           │
│  ├── normalization.py  ──> Input Validation & Clipping     │
│  ├── scoring.py        ──> VoI_raw = 0.20(N+U+R+T-C)         │
│  ├── decision_policy.py──> DISCARD/BUFFER/SUMMARY/TRANSMIT  │
│  └── voi_engine.py     ──> High-Level Orchestrator API      │
└──────────────────────────────┬──────────────────────────────┘
                               │ VoIResult
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION & DIAGNOSTICS                    │
│                    (src/evaluation/)                        │
│  ├── diagnostics.py    ──> 15 Diagnostic Stress Tests       │
│  ├── run_experiment.py ──> Master Pipeline Execution        │
│  └── 01_synthetic_voi.ipynb ──> Interactive Notebook        │
└─────────────────────────────────────────────────────────────┘
```

### Diagram B: Target End-to-End Architecture
Shows the intended future system pipeline connecting CWRU learning to the VoI Engine and FSO channel:

```
                     CWRU Bearing Dataset
                              ↓
                   Dataset Preprocessing
                              ↓
                    2048-sample Windows
                              ↓
                      CNN Learning Model
                              ↓
             ┌────────────────┴────────────────┐
             ↓                                 ↓
    Learned Embedding                 Prediction Probabilities
             ↓                                 ↓
     Novelty Estimation              Uncertainty Estimation
      (Distance / OCSVM)             (1 - max_prob / MC Dropout)
             ↓                                 ↓
      Novelty Score (N)               Uncertainty Score (U)
             │                                 │
             └────────────────┬────────────────┘
                              ↓
                    Context / Task Relevance (R)
                              ↓
                    Temporal Importance (T)
                              ↓
                    Communication Cost Model (C)
                              ↓
            ┌──────────────────────────────────┐
            │       CANONICAL VOI ENGINE       │
            │            (src/voi/)            │
            └─────────────────┬────────────────┘
                              ↓
                          VoI Score
                              ↓
                       Decision Policy
                              ↓
            DISCARD / BUFFER / SUMMARY / TRANSMIT
                              ↓
                     FSO Channel Module
```

---

## 4. CWRU Pipeline Audit

- **Current Implementation Status**: `PLANNED / NOT YET IMPLEMENTED`
- **Repository Search Findings**: 0 references to CWRU, raw bearing files, or vibration preprocessing.
- **Required Functionality for Future Implementation**:
  - Raw `.mat` file loader for CWRU drive end / fan end vibration signals.
  - Non-overlapping or overlapping sliding window segmenter producing 2,048-sample vectors.
  - Normalization (standardization or min-max per channel).
  - Health state label mapping (Normal, Inner Race Fault, Ball Fault, Outer Race Fault).

---

## 5. Novelty Pipeline Audit

- **Current Implementation Status**: `PLANNED / NOT YET IMPLEMENTED`
- **Repository Search Findings**: 0 occurrences of `IsolationForest`, `OneClassSVM`, `Autoencoder`, or distance-based novelty modules in source code.
- **Required Functionality for Future Implementation**:
  - Reference distribution fitting on normal baseline operating conditions.
  - Embedding-based distance metric (e.g. Mahalanobis or Euclidean distance in CNN feature space).
  - Normalization of raw distance metrics into unit-interval Novelty scores $N \in [0, 1]$.

---

## 6. Uncertainty Pipeline Audit

- **Current Implementation Status**: `PLANNED / NOT YET IMPLEMENTED`
- **Repository Search Findings**: 0 occurrences of model confidence proxies, MC Dropout, Ensembles, Temperature Scaling, or Conformal Prediction.
- **Required Functionality for Future Implementation**:
  - Baseline Uncertainty Proxy: $U_{\text{baseline}} = 1 - \max_k P(y=k \mid \mathbf{x})$.
  - Advanced Uncertainty: Monte Carlo Dropout entropy or ensemble variance.
  - Normalization into unit-interval Uncertainty scores $U \in [0, 1]$.

---

## 7. Canonical VoI Engine Audit & Exact Input Contract

The implementation in `src/voi/` is the single canonical source of truth for VoI scoring and decision policies.

### Canonical Input Contract
Every observation passed to `VoIEngine.compute()` or `VoIEngine.compute_batch()` must adhere to the following normalized input vector:

$$\mathbf{x}_{\text{VoI}} = [N, U, R, T, C]^T$$

| Input Variable | Meaning | Scale | Current Source | Future Source |
| :--- | :--- | :---: | :--- | :--- |
| **Novelty ($N$)** | Difference from normal learned behavior | $[0, 1]$ | `synthetic_generator.py` | CWRU Embedding Novelty Module |
| **Uncertainty ($U$)** | Model prediction uncertainty | $[0, 1]$ | `synthetic_generator.py` | CNN Softmax / MC Dropout |
| **Task Relevance ($R$)** | Importance to application task | $[0, 1]$ | `synthetic_generator.py` | Context / Application Layer |
| **Temporal Importance ($T$)** | Significance of trend/change over time | $[0, 1]$ | `synthetic_generator.py` | Temporal Analysis Module |
| **Resource Cost ($C$)** | Communication channel expense | $[0, 1]$ | `synthetic_generator.py` | FSO / Communication Layer |

### Canonical Scoring Formulation (Baseline V0.1)
$$\text{VoI}_{\text{raw}} = w_N N + w_U U + w_R R + w_T T - w_C C$$
Default weights: $w_N = w_U = w_R = w_T = w_C = 0.20$.

---

## 8. Duplicate / Conflicting Implementations Audit

- **Audit Query**: Searched all python modules, scripts, notebooks, and tests for duplicate VoI mathematical scoring, alternative decision thresholds, or bypassing transmission logic.
- **Audit Result**: **Zero duplicate or conflicting VoI implementations exist**. `src/voi/` is confirmed to be the sole canonical VoI implementation in the project.

---

## 9. Future CNN Output Contract

To ensure seamless downstream integration without missing key outputs, the future CNN inference module must expose the following contract for each 2,048-sample window observation:

```python
@dataclass
class CNNInferenceOutput:
    observation_id: str             # Unique window identifier
    predicted_class: int            # Argmax predicted fault class (0=Normal, 1=IR, 2=B, 3=OR)
    class_probabilities: np.ndarray # Softmax probability vector (shape: [num_classes])
    learned_embedding: np.ndarray   # Penultimate layer feature vector (shape: [embedding_dim])
    ground_truth_label: int         # True fault label (for evaluation only)
    operating_condition: str        # Metadata (e.g. "0HP", "1HP", "2HP", "3HP")
```

### Downstream Usage Mapping
- `class_probabilities` $\longrightarrow$ Feeds Uncertainty Module ($U = 1 - \max P$).
- `learned_embedding` $\longrightarrow$ Feeds Novelty Module (Distance to Normal centroid).
- `operating_condition` $\longrightarrow$ Feeds Context / Task Relevance Layer ($R$).

---

## 10. System Output Contract Table

| Output | Producer | Current Implementation Status | Required by Component |
| :--- | :--- | :---: | :--- |
| **Prediction** | CNN Model | ❌ Not implemented (Planned) | Evaluation / Context Layer |
| **Class probabilities** | CNN Model | ❌ Not implemented (Planned) | Uncertainty Module |
| **Learned embedding** | CNN Model | ❌ Not implemented (Planned) | Novelty Module |
| **Novelty score ($N$)** | Novelty Module | ❌ Not implemented (Planned) | Canonical VoI Engine |
| **Uncertainty score ($U$)** | Uncertainty Module | ❌ Not implemented (Planned) | Canonical VoI Engine |
| **Task relevance ($R$)** | Context Layer | ❌ Not implemented (Planned) | Canonical VoI Engine |
| **Temporal importance ($T$)** | Temporal Module | ❌ Not implemented (Planned) | Canonical VoI Engine |
| **Communication cost ($C$)** | Communication Layer | ❌ Not implemented (Planned) | Canonical VoI Engine |
| **VoI score** | `src/voi/` | ✅ **Implemented & Validated** | Decision Policy |
| **Communication decision** | `src/voi/` | ✅ **Implemented & Validated** | FSO Channel Layer |

---

## 11. Completed Work Summary

1. **Modular VoI Engine (`src/voi/`)**: Complete implementation of normalization, linear scoring, configurable weights, decision thresholds, and orchestrator API.
2. **Synthetic Data Generator (`src/data_generation/`)**: Deterministic generator creating 1,000 observations across 6 behavioral scenarios with seed `42`.
3. **Diagnostic Evaluation Suite (`src/evaluation/`)**: 15 diagnostic stress tests analyzing reachability, scenario distributions, monotonicity, weight/threshold sensitivity, and clipping.
4. **Automated Unit Testing (`tests/test_voi.py`)**: 15 / 15 unit tests passing cleanly in $< 0.6$s.
5. **Research Visualizations & Tables (`results/`)**: 11 PNG plot figures and 8 CSV summary tables.
6. **Interactive Notebook (`notebooks/01_synthetic_voi.ipynb`)**: 10-section research notebook demonstrating the complete V0.1 pipeline.

---

## 12. Missing Work Summary

1. CWRU dataset raw file downloading, parsing, and 2,048-sample window preprocessing pipeline.
2. 1D-CNN architecture definition, training loop, validation, and model checkpointing.
3. CNN embedding extraction and probability vector exposure.
4. Learning-based Novelty Estimation module (Isolation Forest / One-Class SVM / Distance-based).
5. Learning-based Uncertainty Estimation module ($1 - \text{confidence}$ proxy & MC Dropout).
6. Application Context / Task Relevance mapping module.
7. Temporal Analysis module for window-to-window trend detection.
8. FSO Communication Channel model and packet-level transmission simulation.

---

## 13. Prioritized Development Roadmap

### P0 — Required Before VoI Integration (Learning Pipeline)
1. CWRU dataset downloading, parsing, and 2,048-sample window dataset pipeline.
2. 1D-CNN baseline model architecture, training, and evaluation scripts.
3. CNN feature extractor exposing `learned_embedding` and `class_probabilities`.
4. Novelty estimation module generating normalized $N \in [0, 1]$.
5. Uncertainty estimation module generating normalized $U \in [0, 1]$.

### P1 — Required for Meaningful End-to-End VoI Experiments
1. Task relevance context layer mapping operating conditions to $R \in [0, 1]$.
2. Temporal importance trend analysis module generating $T \in [0, 1]$.
3. Communication cost model generating $C \in [0, 1]$.
4. Integration script connecting real CWRU $(N, U)$ outputs to canonical `VoIEngine`.
5. End-to-end evaluation comparing communication reduction vs fault classification accuracy.

### P2 — Future Extensions & Refinements
1. Advanced uncertainty estimation (MC Dropout entropy, Deep Ensembles, Conformal Prediction).
2. Advanced novelty estimation (Autoencoders, Mahalanobis distance).
3. FSO channel simulation (attenuation, fog, alignment noise).
4. Weight optimization and threshold calibration experiments.
5. Hardware deployment profiling (Raspberry Pi / Edge device execution).

---

## 14. Single Recommended Next Task

**Recommended Next Task**:  
**Implement CWRU dataset preprocessing and establish the 2,048-sample windowing dataset pipeline.**

*Rationale*: The canonical VoI Engine (`src/voi/`) is fully implemented and validated. To transition from synthetic validation to real sensor data, the immediate prerequisite is establishing the CWRU dataset loading and window segmenter module (`src/cwru_pipeline/dataset.py`) so that the CNN learning model can be trained.

---

## 15. Risks, Concerns, & Research Warnings

> [!WARNING]
> 1. **Data Leakage in Novelty Reference Distributions**: When building the baseline reference distribution for novelty detection (e.g., fitting distance metrics or One-Class SVM), the reference model MUST be fit **strictly on normal baseline training data**. Evaluating novelty on test data using a reference distribution fit on the entire dataset constitutes data leakage.
> 2. **Baseline Threshold Calibration Gap**: Diagnostic tests revealed that under baseline equal weights ($0.20$), peak synthetic VoI score reaches $0.6576$. Under the baseline $0.70$ `TRANSMIT` threshold, $0\%$ of observations trigger immediate transmission. Threshold re-calibration or positive-weight normalization will be required during P1 integration.
