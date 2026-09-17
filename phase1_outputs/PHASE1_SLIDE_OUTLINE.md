# Phase 1 Presentation — Slide Outline

Covers all work completed through **Task 42** (edge intelligence + VoI +
communication-decision pipeline, fully validated). The communication-channel
(FSO) experiment is explicitly **Phase 2** — not started, not included here.

---

## Slide 1 — System Overview & Architecture

**Message:** One canonical five-factor VoI engine, reused unmodified across
every dataset and modality.

- Pipeline: Sensors → Preprocessing → CNN/Modality representations → Fusion
  → Novelty + Uncertainty + Relevance + Temporal Importance + Communication
  Cost → **Canonical VoI Engine** → DISCARD / BUFFER / SUMMARY / TRANSMIT
- Two tracks validated: single-modality (CWRU baseline, cross-checked on
  Paderborn/IMS) and the new **multimodal KAIST rig** track (vibration +
  motor current + temperature, Tasks 31–42)
- The VoI formula, weights, and decision thresholds are never touched by
  new datasets or modalities — only the five input factors are computed
  per-domain and fed into the same unmodified engine

**Images:**
- `07_cwru_cnn_confusion_matrix.png` — baseline CNN performance (single-modality)

---

## Slide 2 — Single-Modality VoI Validation (CWRU baseline)

**Message:** Calibrating the five-factor weights turned an unreachable
TRANSMIT tier into a working, class-sensitive decision policy.

- Before calibration: TRANSMIT was structurally unreachable (max VoI score
  0.4835, below even the SUMMARY threshold)
- After calibration (train/val-only, test held out): TRANSMIT reaches
  ~14% overall, concentrated correctly in Inner Race Fault — the class the
  relevance map itself flags as most critical
- Task Relevance is the dominant contributor (~48% of positive VoI
  contribution on test), Novelty ~32%, Temporal Importance ~20%,
  Uncertainty ~0% (CNN too confident to carry signal)

**Images:**
- `05_single_modality_voi_calibration_decisions.png` — before/after decision distribution
- `06_single_modality_voi_factor_dominance.png` — factor contribution breakdown

---

## Slide 3 — Multimodal Fusion & Intelligence Factors (Tasks 31–39)

**Message:** The same five-factor pipeline extends cleanly to a real
3-sensor fusion problem, with every limitation disclosed rather than hidden.

- KAIST rig: vibration + motor current + temperature, 4 fusion
  configurations, 5-class fault classification (Normal/BPFI/BPFO/
  Misalignment/Unbalance)
- Fusion-head validation accuracy: 84.9–90.5% across configurations
- Full novelty/uncertainty/relevance/temporal/cost integration reused the
  existing canonical modules unmodified — no second VoI formula
- **Disclosed limitation carried into every downstream chart**: relevance
  is only defined for 3 of 5 classes (Normal/BPFI/BPFO); Misalignment/
  Unbalance predictions get `UNDEFINED_RELEVANCE`, never a fabricated
  score — this is why the decision chart shows a large undefined share,
  especially for the two current-requiring configurations (current channel
  genuinely corrupt for all 9 BPFO conditions)

**Images:**
- `01_multimodal_fusion_accuracy.png` — fusion-head train/val accuracy per configuration
- `02_multimodal_decision_distribution_test.png` — decision outcome per configuration (test split)

---

## Slide 4 — VoI Behaviour Analysis & End-to-End Validation (Tasks 40–42)

**Message:** The multimodal system's VoI behaves consistently with the
canonical formulation, and the full edge-intelligence pipeline is
certified ready for the Phase 2 communication-channel experiment.

- Task Relevance dominates the multimodal VoI score too (~56–62% share for
  vibration-anchored configs) — consistent with the single-modality result
  on Slide 2, same weight (0.35), same qualitative behavior
- Investigated an unexpected finding rather than ignoring it: Uncertainty
  (positive weight) correlated *negatively* with VoI in this data — traced
  to Uncertainty co-occurring with lower-relevance Normal predictions, not
  a formula defect
- **Task 42 — 7/7 end-to-end checks pass**: deterministic observation IDs,
  train-only fitting, split integrity/no leakage, model reload consistency,
  artifact consistency, component/decision availability
- `pipeline_ready_for_communication_channel_experiment: true`

**Images:**
- `03_multimodal_voi_factor_dominance.png` — factor dominance per configuration (test split)
- `04_end_to_end_validation_checklist.png` — Task 42 certification result

---

## Notes on source data (for speaker reference, not for the slides)

All numbers above are read directly from committed artifacts, never
recalculated for the presentation:
- `data/processed/multimodal/multimodal_fusion_training_history.json`
- `data/processed/multimodal/multimodal_communication_decision_summary.csv`
- `data/processed/multimodal/multimodal_voi_behaviour_dominance.csv`
- `data/processed/multimodal/multimodal_end_to_end_validation_config.json`
- `results/tables/voi_factor_dominance.csv` (single-modality CWRU)
- `docs/voi_integration_analysis.md`, `docs/voi_calibration_report.md` (Task 13/14 narrative)
