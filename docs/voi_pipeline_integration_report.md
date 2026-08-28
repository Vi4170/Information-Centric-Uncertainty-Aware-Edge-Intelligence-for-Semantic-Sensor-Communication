# VoI Pipeline Integration — Report (Task 10)

## 1. Purpose of the Integration Layer

The integration layer (`src/integration/`) provides a single entry point
that connects the five independently implemented VoI factor modules to the
existing canonical VoI Engine.

It is **orchestration only** — it does not implement, duplicate, or modify
any VoI mathematics, decision thresholds, or factor-scoring logic.

---

## 2. The Five Independent Inputs

| Factor                | Symbol | Range   | Source Module          |
|-----------------------|--------|---------|------------------------|
| Novelty               | N      | [0, 1]  | `src/novelty/`         |
| Prediction Uncertainty| U      | [0, 1]  | `src/uncertainty/`     |
| Task Relevance        | R      | [0, 1]  | `src/relevance/`       |
| Temporal Importance   | T      | [0, 1]  | `src/temporal/`        |
| Communication Cost    | C      | [0, 1]  | `src/communication/`   |

Each factor is computed independently by its own module and passed to the
integration layer as a pre-computed, normalised scalar.

---

## 3. Where Each Input Comes From (Conceptually)

- **Novelty (N):** Measures how unusual or out-of-distribution a sensor
  observation is relative to training data.  Produced by the novelty
  detection baseline using reconstruction error or distance metrics.

- **Uncertainty (U):** Quantifies the prediction confidence of the CNN
  classifier.  Computed via entropy or MC-Dropout–based methods.

- **Task Relevance (R):** Captures how relevant the observation is to the
  current application task (bearing fault monitoring).  Derived from the
  predicted class or class-probability distribution.

- **Temporal Importance (T):** Measures how much the current observation
  has changed relative to the immediately preceding observation.  Uses
  mean absolute difference normalised by a reference scale.

- **Communication Cost (C):** Estimates the resource burden of
  transmitting the observation.  Combines payload size, transmission time,
  and available bandwidth via a weighted sum.

---

## 4. All Five Factors Are Normalised to [0, 1]

The integration layer validates that every input is a finite float in
[0, 1] before forwarding to the engine.  This ensures compatibility with
the canonical `VoIInputs` validation in `src/voi/normalization.py`.

---

## 5. How the Integration Layer Passes Values to the Canonical VoI Engine

```
run_voi_pipeline(N, U, R, T, C)
        │
        ├─ validate each factor ∈ [0, 1]
        │
        ├─ instantiate VoIEngine(weights, thresholds)
        │
        └─ engine.compute(
               novelty=N,
               uncertainty=U,
               task_relevance=R,
               temporal_importance=T,
               resource_cost=C,
               timestamp=...,
           )
               │
               └─ returns canonical VoIResult
```

The five values are passed **unchanged** — no rescaling, transformation,
or recomputation occurs in the integration layer.

---

## 6. Why VoI Mathematics Are Not Duplicated

The canonical formula lives exclusively in `src/voi/scoring.py`:

$$\text{VoI}_\text{raw} = w_N \cdot N + w_U \cdot U + w_R \cdot R + w_T \cdot T - w_C \cdot C$$

The integration layer calls `VoIEngine.compute()`, which internally calls
`calculate_voi_score()`.  There is exactly one implementation of the VoI
formula in the entire codebase.

---

## 7. Canonical VoI Engine Output

`VoIEngine.compute()` returns a `VoIResult` dataclass containing:

| Field                | Type            | Description                          |
|----------------------|-----------------|--------------------------------------|
| `novelty`            | `float`         | Validated N                          |
| `uncertainty`        | `float`         | Validated U                          |
| `task_relevance`     | `float`         | Validated R                          |
| `temporal_importance`| `float`         | Validated T                          |
| `resource_cost`      | `float`         | Validated C                          |
| `raw_voi_score`      | `float`         | Unclipped VoI score                  |
| `voi_score`          | `float`         | Clipped to [0, 1]                    |
| `decision`           | `DecisionAction`| Communication action                 |
| `timestamp`          | `Any \| None`   | Observation timestamp                |
| `metadata`           | `dict`          | Weights and thresholds used          |

---

## 8. Canonical Decision Policy Output

The decision is produced by `evaluate_decision()` in
`src/voi/decision_policy.py` using configurable thresholds:

| VoI Score Range          | Decision    |
|--------------------------|-------------|
| 0.00 ≤ VoI < 0.25       | DISCARD     |
| 0.25 ≤ VoI < 0.50       | BUFFER      |
| 0.50 ≤ VoI < 0.70       | SUMMARY     |
| 0.70 ≤ VoI ≤ 1.00       | TRANSMIT    |

---

## 9. Architecture Diagram

```
CNN / Sensor Pipeline
        │
        ▼
Novelty ─────────── N ∈ [0,1]
Uncertainty ─────── U ∈ [0,1]
Task Relevance ──── R ∈ [0,1]
Temporal Importance  T ∈ [0,1]
Communication Cost ─ C ∈ [0,1]
        │
        ▼
┌──────────────────────────────┐
│   Integration Layer          │
│   src/integration/           │
│   (validation + delegation)  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Canonical VoI Engine       │
│   src/voi/voi_engine.py      │
│   (scoring + decision)       │
└──────────────┬───────────────┘
               │
               ▼
           VoIResult
               │
               ▼
  DISCARD / BUFFER / SUMMARY / TRANSMIT
```

---

## 10. Validation

The integration layer validates all five inputs before forwarding:

- Must be numeric (`int`, `float`, or numpy numeric types)
- Must be finite (rejects `NaN`, `Inf`, `-Inf`)
- Must be in [0, 1] (rejects negative or > 1 values)

Clear `TypeError` and `ValueError` messages identify which factor failed.

The canonical `VoIInputs` in `src/voi/normalization.py` performs its own
validation as a second safety net.

---

## 11. Reproducibility

The entire pipeline is deterministic:

- No random operations in any factor module or in the engine
- Same five inputs + same weights + same thresholds = same result, always

---

## 12. Separation of Responsibilities

| Responsibility              | Owner                         |
|-----------------------------|-------------------------------|
| Novelty scoring             | `src/novelty/`                |
| Uncertainty estimation      | `src/uncertainty/`            |
| Task relevance mapping      | `src/relevance/`              |
| Temporal importance calc    | `src/temporal/`               |
| Communication cost calc     | `src/communication/`          |
| Input validation            | `src/voi/normalization.py`    |
| VoI formula                 | `src/voi/scoring.py`          |
| Decision policy             | `src/voi/decision_policy.py`  |
| Engine orchestration        | `src/voi/voi_engine.py`       |
| **Factor → Engine bridge**  | **`src/integration/`**        |

The integration layer owns **only** the bridge.

---

## 13. Limitations

1. **Scalar-only (v0.1):** Processes one observation at a time.  Batch
   support can be added via `VoIEngine.compute_batch()` in a future
   iteration.
2. **No automatic factor computation:** The caller must compute all five
   factors before calling `run_voi_pipeline`.  There is no automatic
   pipeline that ingests raw sensor data end-to-end.
3. **No learned weighting:** Weights are static defaults (0.20 each)
   unless overridden per call.

---

## 14. Future: Integration with Real CNN / Pipeline Outputs

In a production deployment, the flow would be:

1. Raw sensor window arrives from the CWRU pipeline.
2. CNN classifies the window → class probabilities.
3. Novelty module scores the window.
4. Uncertainty module scores the prediction.
5. Relevance module maps the prediction to a relevance score.
6. Temporal module compares the window to the previous window.
7. Communication module estimates transmission cost.
8. **Integration layer** passes all five scores to the VoI Engine.
9. VoI Engine returns a decision.

This end-to-end pipeline is a future task.

---

## 15. Future: FSO / Channel Integration

When the FSO channel model is implemented, a sixth factor (channel
quality / link reliability) may be introduced.  The integration layer
and VoI Engine would be extended to accept this additional input.

This is out of scope for Task 10.
