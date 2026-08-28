# Temporal Importance — Baseline Report (v0.1)

## 1. Why Temporal Importance Is Needed

In a semantic sensor communication system, not all observations carry equal
informational value.  An observation's *timing* — specifically, how much
change it represents relative to recent history — can be a strong signal
of whether transmitting or processing it is worthwhile.

Consider a bearing vibration monitor:

- **Stable operation:** consecutive windows look nearly identical.
  Transmitting every window wastes bandwidth and energy on the edge device.
- **Gradual degradation:** successive windows slowly diverge, indicating
  a developing fault trend that merits closer attention.
- **Sudden fault onset:** a sharp, abrupt change between two consecutive
  windows signals a potentially critical event demanding immediate action.

Temporal Importance quantifies this "change significance" as a single
normalised scalar, enabling downstream decision-making (e.g. the Value-of-
Information engine) to weight observations appropriately.

---

## 2. Why an Observation's Timing / Change Can Matter

An observation is not just a snapshot — it exists in a temporal context.
Two windows with identical spectral content carry very different
informational value if one follows an hour of identical readings and the
other follows a sudden shift.

Temporal change detection captures:

| Scenario                      | Implication                              |
|-------------------------------|------------------------------------------|
| No change between windows     | Redundant information — low value         |
| Gradual drift                 | Possible trend — moderate value           |
| Abrupt discontinuity          | Possible event / fault onset — high value |

---

## 3. Relationship to Sensor Degradation and Trends

Temporal Importance is particularly relevant to condition monitoring:

- **Early degradation** often manifests as slowly increasing vibration
  amplitude across consecutive windows, producing a gradually rising
  Temporal Importance score.
- **Bearing spalling or seizure** can appear as a sudden spectral shift —
  a large spike in Temporal Importance.
- **Sensor drift or noise floor changes** are also captured, providing an
  implicit health proxy for the sensing hardware itself.

---

## 4. Current v0.1 Method

The baseline uses a **mean-absolute-difference** approach between
consecutive observations, normalised by a configurable reference scale.

This method was chosen because it is:

1. **Interpretable** — directly measures magnitude of change.
2. **Computationally inexpensive** — a single vectorised subtraction and
   mean per observation pair.
3. **Deployment-friendly** — no model training, no learned parameters, no
   GPU required.

> **Important:** This is a v0.1 baseline, not a final temporal model.
> The normalisation scale is a design parameter, not an optimised value.

---

## 5. Mathematical Formulation

Given a sequence of observations:

$$x_1, x_2, \ldots, x_T \quad \text{where } x_t \in \mathbb{R}^d$$

**Step 1 — Mean Absolute Difference:**

$$D_t = \frac{1}{d} \sum_{i=1}^{d} |x_t^{(i)} - x_{t-1}^{(i)}| \qquad t = 2, \ldots, T$$

**Step 2 — Normalised Temporal Importance:**

$$T_t = \text{clip}\!\left(\frac{D_t}{\text{TEMPORAL\_CHANGE\_SCALE}},\; 0,\; 1\right)$$

**First observation convention:**

$$T_1 = 0$$

---

## 6. Expected Input Representation

| Property            | Specification                                    |
|---------------------|--------------------------------------------------|
| Type                | `numpy.ndarray`                                  |
| Shape               | `(num_observations, observation_size)`            |
| Semantics           | Each row is one sequential observation / window   |
| Scalar observations | Use shape `(N, 1)`                               |
| CWRU windows        | Shape `(N, 2048)` — 2048-sample non-overlapping windows |
| Data type           | Any numeric dtype (cast to float64 internally)   |
| Constraints         | All values must be finite (no NaN / Inf)          |

---

## 7. Output Definition

| Property        | Specification                                            |
|-----------------|----------------------------------------------------------|
| Type            | `numpy.ndarray`, dtype `float64`                         |
| Shape           | `(num_observations,)` — one score per input observation  |
| Range           | `[0, 1]`                                                 |
| Interpretation  | 0 = no temporal change; 1 = change ≥ reference scale     |

---

## 8. First-Observation Behaviour

The first observation in a sequence has no predecessor.  Its Temporal
Importance is **deterministically set to 0.0**.  This is a design choice,
not a limitation — it reflects that without prior context, temporal change
is undefined.

---

## 9. Stable vs. Changing vs. Sudden-Change Examples

### 9.1 Constant (Stable) Signal

```
Observations:  [0.5, 0.5, 0.5, 0.5, 0.5]   (scalar, 5 steps)
D values:      [  —, 0.0, 0.0, 0.0, 0.0 ]
T scores:      [0.0, 0.0, 0.0, 0.0, 0.0 ]
```

All scores are zero — no temporal change detected.

### 9.2 Gradual Change

```
Observations:  [0.0, 0.1, 0.2, 0.3, 0.4]   (scale = 0.5)
D values:      [  —, 0.1, 0.1, 0.1, 0.1 ]
T scores:      [0.0, 0.2, 0.2, 0.2, 0.2 ]
```

Moderate, uniform temporal importance throughout.

### 9.3 Sudden Change

```
Observations:  [0.0, 0.0, 0.0, 5.0, 5.0]   (scale = 0.5)
D values:      [  —, 0.0, 0.0, 5.0, 0.0 ]
T scores:      [0.0, 0.0, 0.0, 1.0, 0.0 ]
```

A single spike at the transition — exactly the behaviour desired for
fault-onset detection.

---

## 10. Normalisation Strategy

The raw mean-absolute-difference *D* has unbounded range.  We normalise by
dividing by `TEMPORAL_CHANGE_SCALE` and clipping to [0, 1]:

$$T = \text{clip}(D / S, 0, 1)$$

- **`TEMPORAL_CHANGE_SCALE` (S):** a reference magnitude representing a
  "fully significant" change.  Default: **0.5** (calibrated for CWRU-scale
  accelerometer readings ≈ 0.1–1.0 g).
- Changes smaller than *S* produce proportionally smaller scores.
- Changes ≥ *S* saturate at 1.0.

> The default scale is a reasonable starting point but is **not optimised**.
> Practitioners should tune it to their signal domain.

---

## 11. Score Interpretation

| Score Range   | Interpretation                                         |
|---------------|--------------------------------------------------------|
| T = 0.0       | No change from previous observation (or first obs)     |
| 0 < T ≤ 0.2   | Minor fluctuation — likely noise or normal variation   |
| 0.2 < T ≤ 0.5 | Moderate change — possible trend or drift              |
| 0.5 < T ≤ 0.8 | Significant change — warrants attention                |
| 0.8 < T ≤ 1.0 | Large / sudden change — potential event or fault onset  |

These thresholds are illustrative.  The continuous score is intended to be
consumed by downstream components (e.g. VoI engine), not hard-thresholded.

---

## 12. Computational Simplicity

The entire computation for *N* observations of dimension *d* is:

1. One element-wise subtraction: O(N × d)
2. One absolute value: O(N × d)
3. One row-wise mean: O(N × d)
4. One element-wise division + clip: O(N)

**Total: O(N × d)** — linear in input size, no iteration, fully
vectorised with NumPy.

Memory footprint: one temporary `(N−1, d)` difference array.

This is well within the computational budget of resource-constrained
edge devices (e.g. Raspberry Pi, microcontrollers with NumPy-compatible
runtimes).

---

## 13. Limitations

1. **No trend awareness:** The method only considers *adjacent* pairs.
   A slow, monotonic drift across many windows may produce individually
   small scores even though the cumulative change is large.
2. **Scale sensitivity:** The normalisation scale must be chosen per
   signal domain.  A poorly chosen scale compresses or saturates scores.
3. **No frequency / spectral sensitivity:** Equal weight is given to all
   elements of the observation vector.  A change concentrated in a few
   high-frequency bins is treated the same as a broadband shift.
4. **Symmetric treatment of increase / decrease:** The absolute difference
   does not distinguish rising from falling signals.
5. **No memory beyond one step:** Only the immediately preceding
   observation is used — there is no notion of "recent history" or
   baseline.

---

## 14. Why This Is Only a Baseline

The v0.1 module intentionally prioritises simplicity and interpretability
over sophistication.  It serves as:

- A **functional placeholder** so that the VoI engine can incorporate
  temporal information from the start.
- A **performance lower bound** against which more advanced methods can
  be evaluated.
- A **correctness reference** — if a complex model cannot outperform
  this simple baseline, its added complexity is unjustified.

---

## 15. Future Improvements

| Improvement                          | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| **Trend-aware methods**              | Compute slope over a sliding window to detect monotonic trends              |
| **Sliding-window statistics**        | Use rolling mean / variance to establish a dynamic baseline                 |
| **Change-point detection**           | Apply CUSUM, PELT, or Bayesian online change-point methods                  |
| **Exponentially weighted importance**| EWMA-based scoring to give more weight to recent history                    |
| **Temporal models (RNN / Transformer)**| Learn temporal representations from labelled event sequences             |
| **Frequency-aware differencing**     | Weight spectral bands by fault-discriminative importance                    |
| **Application-specific context**     | Incorporate domain knowledge (e.g. operating-regime transitions)            |
| **Multi-scale temporal analysis**    | Combine short-horizon (1-step) and long-horizon (N-step) change measures    |
| **Adaptive normalisation**           | Learn or calibrate the change scale from data statistics                    |

---

## 16. Important Distinction

Temporal Importance measures the significance associated with **temporal
change** — how different the current observation is from its immediate
predecessor.

It is **not**:

| Concept                 | Module          | What It Measures                            |
|-------------------------|-----------------|---------------------------------------------|
| Novelty                 | `src/novelty/`  | How unusual the observation is overall       |
| Prediction Uncertainty  | `src/uncertainty/` | Confidence of the classification model    |
| Task Relevance          | `src/relevance/` | How relevant the observation is to the task |
| Resource Cost           | *(future)*      | Communication / computation cost             |

These are **separate factors** in the broader Value-of-Information
formulation and are combined at a higher level — not within this module.
