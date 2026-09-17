# Full Results Section — Slide Outline

Covers the entire project's results, across every dataset, through **Task 42**.
The Phase 2 communication-channel (FSO) experiment is explicitly out of scope —
not started, not included here.

Organized as ~6 sections / slides (adjust freely — each has 1-2 supporting charts).

---

## 1. Single-Modality Baseline — CWRU

**Message:** The canonical 5-factor VoI engine, its weights, and its decision
thresholds all originate here and are never modified for any later dataset.

- CNN: 100% test accuracy, 1.0 macro F1 (4-class bearing fault classification)
- VoI calibration (Task 14): before calibration TRANSMIT was structurally
  unreachable (max score 0.4835); after calibration (train/val-only, test
  held out) TRANSMIT reaches ~14% overall, concentrated correctly in Inner
  Race Fault — the class the relevance map itself flags as most critical
- Task Relevance is the dominant contributor to VoI (~48% share on test),
  Novelty ~32%, Temporal Importance ~20%, Uncertainty ~0% (CNN too
  confident to carry signal)

**Images:** `01_cwru_cnn_confusion_matrix.png`, `02_cwru_voi_calibration_decisions.png`, `03_cwru_voi_factor_dominance.png`

---

## 2. Cross-Dataset Validation — Paderborn & IMS

**Message:** The same unmodified VoI engine transfers to a second labeled
dataset with the same qualitative behavior, and a genuinely different
(unlabeled, run-to-failure) dataset stress-tests the novelty/temporal
factors alone.

- **Paderborn** (3-class, 64kHz): CNN 99.1% accuracy; full 5-factor VoI
  integration; TRANSMIT appears only for Outer Race Fault (3.51% of that
  class's test observations) — a real class-conditioned distinction (Inner
  Race Fault tops out at BUFFER/SUMMARY, TRANSMIT never reached), not
  fabricated
- **IMS** (run-to-failure bearings, no fault-type labels): no CNN/relevance/
  full VoI decision is possible without labels — instead, Novelty and
  Temporal Importance were tracked across each bearing's full run.
  Documented failures (bearings explicitly noted "documented_failure") show
  both scores trending upward toward the end of the run in most cases; one
  channel (3rd_test bearing 3) did not show a clear trend — reported as a
  genuine, unresolved limitation, not smoothed over

**Images:** `04_paderborn_cnn_confusion_matrix.png`, `05_paderborn_voi_decision_distribution.png`, `06_ims_novelty_temporal_trend.png` (8 stacked bearing-run panels — consider cropping to 2-3 representative panels, e.g. "1st_test bearing 4" and "2nd_test bearing 1", if slide space is tight)

---

## 3. Dataset Integration Status (corrected)

**Message:** Full transparency on what's actually been experimentally
validated versus what's only been preprocessed so far.

- CWRU, Paderborn, IMS: experimentally complete (to the extent each
  dataset's own label structure allows)
- XJTU-SY, MIMII-DG: preprocessing pipelines integrated only — no
  CNN/novelty/uncertainty/VoI experiments performed yet
- Multimodal KAIST: full pipeline complete through Task 42

**Note:** an earlier dashboard image making this same comparison was found
to be **stale** — it claimed Paderborn and IMS experiments were "not yet
performed," which was true when it was generated but is no longer accurate.
It was excluded rather than corrected-in-place; this new table replaces it.

**Image:** `07_dataset_integration_status.png`

---

## 4. Cross-Dataset VoI Consistency Check

**Message:** Task Relevance dominates the VoI score everywhere it's been
measured — the same qualitative behavior across three structurally
different datasets, using the exact same unmodified formula and weights.

- CWRU: 48% / Paderborn: 62% / Multimodal (avg of 4 fusion configs): ~51%
- Novelty and Temporal Importance fill most of the remainder; Uncertainty
  stays negligible everywhere the underlying CNN is confident
- This is reported as an **observed consistency**, not proof the weights are
  "correct" — no recalibration was performed to produce this consistency

**Image:** `08_cross_dataset_voi_factor_dominance.png`

---

## 5. Multimodal Fusion & Intelligence Factors (Tasks 31–39)

**Message:** The same five-factor pipeline extends to a real 3-sensor
fusion problem (vibration + motor current + temperature), with every
limitation disclosed rather than hidden.

- 4 fusion configurations, 5-class fault classification; fusion-head
  validation accuracy 84.9–90.5%
- **Disclosed limitation carried into every downstream chart**: relevance
  is only defined for 3 of 5 classes (Normal/BPFI/BPFO) — no defensible
  evidence exists for Misalignment/Unbalance's relative criticality, so
  those predictions get `UNDEFINED_RELEVANCE`, never a fabricated score.
  This is why the decision chart shows a large undefined share, especially
  for the two current-requiring configurations (the motor-current channel
  is genuinely corrupt for all 9 BPFO conditions in the raw data)

**Images:** `09_multimodal_fusion_accuracy.png`, `10_multimodal_decision_distribution.png`

---

## 6. Multimodal VoI Behaviour & End-to-End Validation (Tasks 40–42)

**Message:** The multimodal system's VoI behaves consistently with the
canonical formulation, and the full edge-intelligence pipeline is certified
ready for the Phase 2 communication-channel experiment.

- Investigated an unexpected finding rather than ignoring it: Uncertainty
  (positive weight) correlated *negatively* with VoI in this data — traced
  to Uncertainty co-occurring with lower-relevance Normal predictions, not
  a formula defect
- **Task 42 — 7/7 end-to-end checks pass**: deterministic observation IDs,
  train-only fitting, split integrity/no leakage, model reload consistency,
  artifact consistency, component/decision availability
- `pipeline_ready_for_communication_channel_experiment: true`

**Images:** `11_multimodal_voi_factor_dominance.png`, `12_multimodal_end_to_end_validation.png`

---

## Corrections made while assembling this deck (for transparency)

1. **Excluded a stale figure** (old "Dataset Integration Comparison" table)
   that claimed Paderborn/IMS experiments were not yet performed — they had
   been completed later in the project. Replaced with `07_dataset_integration_status.png`.
2. **Found and corrected a mislabeled figure**: `results/figures/paderborn_cnn_confusion_matrix.png`
   is generated by a shared plotting function (`src/evaluation/cnn_evaluation.py::plot_confusion_matrix`)
   that hardcodes the title "...CWRU 4-Class..." regardless of caller. The
   underlying numbers are genuinely Paderborn's (verified: row sums match
   Paderborn's exact test-class counts, 2251/3374/4133), so the chart used
   here (`04_paderborn_cnn_confusion_matrix.png`) was regenerated with the
   correct title from the same confusion-matrix values. A background task
   has been flagged to fix the underlying function so this doesn't recur
   for future datasets.

## Source data (for speaker reference, not for the slides)

- `results/tables/{cnn_evaluation_summary,voi_factor_dominance,voi_decision_distribution}.csv` (CWRU)
- `results/tables/paderborn_{cnn_evaluation_summary,voi_factor_dominance,voi_decision_distribution}.csv`
- `results/tables/{ims_temporal_progression_summary,ims_failure_proximity_correlation}.csv`
- `data/processed/multimodal/multimodal_fusion_training_history.json`
- `data/processed/multimodal/multimodal_communication_decision_summary.csv`
- `data/processed/multimodal/multimodal_voi_behaviour_dominance.csv`
- `data/processed/multimodal/multimodal_end_to_end_validation_config.json`
- `docs/voi_integration_analysis.md`, `docs/voi_calibration_report.md` (Task 13/14 narrative)
