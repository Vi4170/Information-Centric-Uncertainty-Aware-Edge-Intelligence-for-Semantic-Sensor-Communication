# CWRU Bearing Dataset Preprocessing & Windowing Report

**Project Title**: Information-Centric Uncertainty-Aware Edge Intelligence for Semantic Sensor Communication  
**Module**: CWRU Vibration Dataset Preprocessing & 2,048-Sample Windowing Pipeline  
**Date**: August 16, 2026

---

## 1. Dataset Used
The Case Western Reserve University (CWRU) Bearing Data Center benchmark dataset was selected for real vibration signal preprocessing. 
- **Files Selected**: 40 baseline `.mat` recordings covering 4 health conditions (Normal, Inner Race Fault, Ball Fault, Outer Race Fault) across 4 motor load conditions (0 HP, 1 HP, 2 HP, 3 HP).
- **Fault Sizes**: 0.000" (Normal), 0.007", 0.014", and 0.021" (Faults).

---

## 2. Signal Selected
- **Vibration Channel**: Drive End (DE) time-series accelerometer signal (`X..._DE_time`).
- **Sampling Rate**: 12,000 Hz ($12\text{k}$).
- **Selection Rationale**: The Drive End channel represents the primary research benchmark standard for CWRU fault classification and anomaly detection. Sampling rate is explicitly verified upon loading.

---

## 3. Preprocessing Performed
1. **Signal Validation**: Signals are validated for 1D structure, finite values (rejecting `NaN` and `Inf`), float32 representation, and minimum sample length ($\ge 2048$).
2. **Leakage-Safe Z-Score Normalization**:
   - Training mean ($\mu_{\text{train}} = 0.000015$) and standard deviation ($\sigma_{\text{train}} = 0.279156$) are computed **strictly from the concatenated raw training set recordings**.
   - Validation and test sets are normalized using $(\mathbf{x} - \mu_{\text{train}}) / \sigma_{\text{train}}$ so that test data never influences training statistics.

---

## 4. Windowing Configuration
- **Window Size**: `WINDOW_SIZE = 2048` samples (~170.7 ms at 12 kHz).
- **Step Size**: `STEP_SIZE = 2048` samples (non-overlapping windows).
- **Incomplete Final Window Handling**: Incomplete tail windows (< 2048 samples) are explicitly discarded. Across all 40 recordings, a total of 48,640 tail samples were discarded (~1,216 samples per file), preserving exact 2,048-sample shapes.

---

## 5. Group-Level Dataset Split (Data Leakage Prevention)
- **Split Strategy**: Group-aware split performed at the **recording / source `.mat` file level** within each fault class using random seed `42`.
- **Target Ratios**: ~70% Train, ~15% Validation, ~15% Test.
- **Recording Allocation**:
  - **Train**: 26 `.mat` files (65.0% of windows)
  - **Validation**: 7 `.mat` files (17.5% of windows)
  - **Test**: 7 `.mat` files (17.5% of windows)
- **Leakage Verification**: `train_files ∩ val_files = ∅`, `train_files ∩ test_files = ∅`, and `val_files ∩ test_files = ∅`.

---

## 6. Final Dataset Summary & Array Shapes

Saved to: [`data/processed/cwru/cwru_dataset_v1.npz`](file:///c:/Users/kingb/Desktop/7th_fyp/data/processed/cwru/cwru_dataset_v1.npz)

| Split | Array Name | Window Count | Array Shape | Data Type |
| :--- | :--- | :---: | :---: | :---: |
| **Training** | `X_train` | 1,508 | `(1508, 2048, 1)` | `float32` |
| **Training Labels** | `y_train` | 1,508 | `(1508,)` | `int64` |
| **Validation** | `X_val` | 406 | `(406, 2048, 1)` | `float32` |
| **Validation Labels** | `y_val` | 406 | `(406,)` | `int64` |
| **Testing** | `X_test` | 406 | `(406, 2048, 1)` | `float32` |
| **Testing Labels** | `y_test` | 406 | `(406,)` | `int64` |
| **Total** | — | **2,320** | — | — |

---

## 7. Class Distribution Table

| Class Label | Health Condition | Total Windows | Train Windows | Val Windows | Test Windows |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Normal Baseline | 232 | 116 | 58 | 58 |
| **1** | Inner Race Fault | 696 | 464 | 116 | 116 |
| **2** | Ball Fault | 696 | 464 | 116 | 116 |
| **3** | Outer Race Fault | 696 | 464 | 116 | 116 |

---

## 8. Operating Conditions Breakdown

| Load (HP) | Motor Speed (RPM) | Source File Count | Windows Generated |
| :---: | :---: | :---: | :---: |
| **0 HP** | ~1797 RPM | 10 | 580 |
| **1 HP** | ~1772 RPM | 10 | 580 |
| **2 HP** | ~1750 RPM | 10 | 580 |
| **3 HP** | ~1730 RPM | 10 | 580 |

---

## 9. Data Quality Audit
- **Missing / Invalid Values**: 0 `NaN`, 0 `Inf`, 0 missing channel signals.
- **Traceability**: Every 2,048-sample window is stored with a unique `observation_id` (e.g. `cwru_105_w0000_train`) in [`data/processed/cwru/cwru_metadata.csv`](file:///c:/Users/kingb/Desktop/7th_fyp/data/processed/cwru/cwru_metadata.csv).
- **Diagnostic Visualization**: Saved to [`results/figures/cwru_sample_signals.png`](file:///c:/Users/kingb/Desktop/7th_fyp/results/figures/cwru_sample_signals.png).

---

## 10. Limitations & Recommended Next Task

### Limitations
- Baseline dataset focused on 12 kHz Drive End vibration signals.
- Fixed 2,048-sample non-overlapping windows.

### Recommended Next Task
**Implement and validate the baseline 1D CNN model using the processed 2,048-sample CWRU windows (`cwru_dataset_v1.npz`).**
