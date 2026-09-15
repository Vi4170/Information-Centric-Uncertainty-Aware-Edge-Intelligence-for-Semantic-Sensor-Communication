# Information-Centric Uncertainty-Aware Edge Intelligence for Semantic Sensor Communication

An edge-intelligence system for machine condition monitoring that does not blindly transmit every
sensor reading. It learns normal machine behaviour from vibration/audio data, scores each observation's
**Novelty**, **Uncertainty**, **Task Relevance**, **Temporal Importance**, and **Communication Cost**,
combines them into a single **Value-of-Information (VoI)** score, and a decision policy converts that
score into a communication action (`DISCARD` / `BUFFER` / `SUMMARY` / `TRANSMIT`). The intended final
communication medium is Free Space Optical (FSO) — not yet implemented; see [Status](#status--limitations).

Bearing/machine condition monitoring is the demonstration domain, not the research question itself. The
question is: can a learning-based Information Value system reduce unnecessary sensor communication while
still preserving task-relevant, novel, and uncertain events?

## Pipeline

```
Sensor data (vibration / audio)
        |
        v
Preprocessing & Windowing
        |
        v
      CNN Encoder ---------------------------+
        |                                    |
   class probabilities                 64-D learned embedding
        |                                    |
        v                                    v
  Uncertainty Estimator                Novelty Detector
        |                                    |
        +--------------+  +------------------+
                       |  |
        Task Relevance --+-- Temporal Importance --+-- Communication Cost
                       |                            |
                       v                            v
                        \--------- VoI Engine ------/
                                     |
                                     v
                              VoI score (0-1)
                                     |
                                     v
                             Decision Policy
                   DISCARD | BUFFER | SUMMARY | TRANSMIT
                                     |
                                     v
                                 Receiver
```

`VoI = 0.30*Novelty + 0.05*Uncertainty + 0.35*TaskRelevance + 0.20*TemporalImportance - 0.10*CommCost`
(clipped to `[0, 1]`), thresholded at `discard<0.25 / buffer<0.50 / summary<0.70 / transmit>=0.70`.
These weights/thresholds are the single canonical configuration (`src/voi/`), calibrated once on
train/validation data only, and are held fixed across every dataset experiment below.

## Datasets

| Dataset | Role | Status |
| :--- | :--- | :--- |
| CWRU | 4-class bearing fault classification; baseline VoI validation | Full pipeline (CNN -> VoI -> decision) integrated and validated |
| Paderborn | 3-class bearing fault classification; transfer/generalization check | Full pipeline integrated and validated, unmodified CNN/VoI config |
| IMS | Genuine run-to-failure vibration data; temporal degradation analysis | Novelty/Temporal Importance measured; no per-window labels exist, so classification/uncertainty/relevance/full VoI are honestly marked not scientifically valid rather than approximated |
| XJTU-SY | Run-to-failure vibration data | Code and manifest integrated; blocked on missing raw data in this environment, evaluated only structurally |
| MIMII-DG | Audio machine-sound anomaly data | Integrated and explicitly labeled MIMII-DG — a different dataset from the original MIMII (0 dB, 4 machine types), which was never integrated |

## Repository structure

```
src/
  cnn/            shared 1D CNN backbone (Conv1D x3 -> GAP -> 64-D embedding -> Dropout -> Softmax)
  novelty/        distance-to-reference-centroid novelty scoring
  uncertainty/    predictive entropy over CNN class probabilities
  relevance/      task-relevance scoring from class/probabilities
  temporal/       temporal importance (window-to-window signal change)
  communication/  communication cost estimation
  voi/            canonical VoI engine, scoring, normalization, decision policy (protected)
  integration/    wires the five factors into the VoI engine for CWRU
  continual/      gated continual-learning: adaptation buffer, condition monitor, versioned
                  multi-prototype novelty reference, safety/regression gate, admission
                  controller, head-only CNN adaptation, model registry
  cwru_pipeline/  paderborn_pipeline/  ims_pipeline/  xjtu_pipeline/  mimii_dg_pipeline/
                  per-dataset preprocessing (leakage-safe splitting/normalization)
  evaluation/     CNN evaluation + per-dataset VoI behaviour analyses + cross-dataset analysis
  data_generation/  utils/

dashboard/        Streamlit results dashboard (streamlit run dashboard/app.py)
docs/             one report per major module/task (architecture, calibration, per-dataset findings)
results/          tables/ figures/ logs/ continual/ cross_dataset/ per pipeline run
tests/            one test module per src/ package
run_project.py    orchestrates the pipelines end-to-end in dependency order
```

## Running

```bash
pip install -r requirements.txt
python -m pytest -q                 # full test suite
python run_project.py --all         # run every available pipeline stage
streamlit run dashboard/app.py      # visual results dashboard
```

`run_project.py` calls existing modules directly (no reimplementation) and skips any stage whose raw
data isn't present in the environment.

## Status & limitations

- 414 automated tests passing, 0 failing, 42 skipped (skipped tests are gated on raw XJTU-SY/MIMII-DG
  data not present in this environment, not deleted).
- A six-stage gated continual-learning architecture has been implemented and demonstrated end-to-end on
  CWRU (buffer -> condition monitor -> versioned novelty reference -> safety/regression gate -> admission
  controller -> head-only CNN adaptation + versioned model registry). The demonstrated "new condition" was
  already a class in the CNN's trained label space, so this validates the gating mechanics, not genuine
  unseen-class learning.
- The full cross-dataset evidence matrix, honest defensible-contribution statement, and explicit list of
  claims that must not be made are maintained as a machine-readable artifact at
  [`results/cross_dataset/task30_cross_dataset_analysis.json`](results/cross_dataset/task30_cross_dataset_analysis.json).
- Communication Cost is currently a constant placeholder; no physical or simulated FSO channel experiment
  has been run yet. This is the critical remaining gap before any communication-savings claim can be made.
