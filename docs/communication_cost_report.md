# Communication Cost — Baseline Report (v0.1)

## 1. Why Communication Cost Is Part of the Broader VoI Concept

In an Information-Centric Uncertainty-Aware Edge Intelligence system, the
decision to transmit a sensor observation is not purely about the
*information content* of that observation — it must also consider the
*resource cost* of the transmission.

A highly informative observation that is prohibitively expensive to
transmit may not justify its cost, while a moderately informative
observation that is cheap to communicate may be well worth sending.

The Value-of-Information (VoI) framework therefore needs both:

- **Information value factors** (novelty, uncertainty, task relevance,
  temporal importance)
- **Resource cost factors** (communication cost, and potentially
  computation cost)

Communication Cost captures the latter.

---

## 2. What "Communication Cost" Means in This Project

Communication Cost C ∈ [0, 1] quantifies the **relative resource burden**
of transmitting one observation from an edge sensor device to a remote
receiver.

It aggregates three intuitive cost drivers:

| Factor               | What It Captures                                        |
|----------------------|---------------------------------------------------------|
| Payload size         | How large the observation data is (bytes)               |
| Transmission time    | How long the transmission takes (seconds)               |
| Available bandwidth  | How congested/scarce the communication channel is       |

A higher score means a more expensive transmission.

---

## 3. Why Cost Should Remain Separate from Information Value

Mixing cost into the information-value score would conflate two
fundamentally different concepts:

- **Information value** answers: *"How much would we learn from this
  observation?"*
- **Communication cost** answers: *"How much would it cost to obtain
  that learning?"*

Keeping them separate allows the VoI engine to make principled trade-offs
(e.g. "send only if information value exceeds cost by a threshold") and
to re-weight them independently as deployment conditions change.

---

## 4. Parameters Considered

### 4.1 Payload Size

The number of bytes in the observation to be transmitted.  Larger payloads
consume more channel capacity and energy.

- Normalised as: `S = clip(payload_size / MAX_PAYLOAD_SIZE, 0, 1)`
- Default reference: **16 384 bytes** (one CWRU 2048-sample float64
  window)

### 4.2 Transmission Time

The estimated time (seconds) required to complete the transmission.
Longer transmissions occupy the channel and increase latency.

- Normalised as: `T = clip(transmission_time / MAX_TRANSMISSION_TIME, 0, 1)`
- Default reference: **1.0 second**

### 4.3 Available Bandwidth

The currently available channel capacity (bytes/second).  When bandwidth
is scarce, the cost of any transmission rises.

- Normalised as: `B = clip(1 − available_bandwidth / REFERENCE_BANDWIDTH, 0, 1)`
- Default reference: **1 000 000 B/s** (1 MB/s)

---

## 5. Mathematical Formulation

**Step 1 — Normalise each component to [0, 1]:**

$$S = \text{clip}\!\left(\frac{\text{payload\_size}}{\text{MAX\_PAYLOAD\_SIZE}},\; 0,\; 1\right)$$

$$T = \text{clip}\!\left(\frac{\text{transmission\_time}}{\text{MAX\_TRANSMISSION\_TIME}},\; 0,\; 1\right)$$

$$B = \text{clip}\!\left(1 - \frac{\text{available\_bandwidth}}{\text{REFERENCE\_BANDWIDTH}},\; 0,\; 1\right)$$

**Step 2 — Weighted combination:**

$$C = \text{clip}\!\left(w_S \cdot S + w_T \cdot T + w_B \cdot B,\; 0,\; 1\right)$$

where the default weights are:

| Weight       | Default | Symbol  |
|--------------|---------|---------|
| Payload size | 0.5     | $w_S$   |
| Time         | 0.3     | $w_T$   |
| Bandwidth    | 0.2     | $w_B$   |

The weights form a convex combination: $w_S + w_T + w_B = 1.0$.

---

## 6. Normalisation Process

Each raw input is divided by its reference limit and clipped to [0, 1]:

- Values **below** the reference limit produce a proportional score in
  (0, 1).
- Values **at or above** the reference limit saturate at 1.0.
- All three components are combined via the weighted sum.
- The final cost is clipped to [0, 1] for numerical safety.

---

## 7. Configurable Weights

The weights are stored in `src/communication/config.py` and can be
overridden per call via keyword arguments.

**Constraints:**
- Each weight must be in [0, 1].
- The three weights must sum to 1.0 (within a small tolerance).

> **Important:** The default weights (0.5 / 0.3 / 0.2) are initial
> design parameters, NOT optimised values.  They reflect a baseline
> assumption that payload size is the dominant cost driver, followed by
> time and then bandwidth pressure.

---

## 8. Score Interpretation

| Score   | Meaning                                                    |
|---------|------------------------------------------------------------|
| C = 0.0 | Negligible cost — zero payload, zero time, full bandwidth  |
| C ≈ 0.2 | Low cost — small observation, fast link, ample bandwidth   |
| C ≈ 0.5 | Moderate cost — mid-range on one or more dimensions        |
| C ≈ 0.8 | High cost — large payload and/or constrained channel       |
| C = 1.0 | Maximum cost — all components at or beyond reference limits|

---

## 9. Examples

### 9.1 Minimal Cost

```python
compute_communication_cost(
    payload_size=0,
    transmission_time=0,
    available_bandwidth=1_000_000,  # full reference bandwidth
)
# → C = 0.0
```

### 9.2 Maximum Cost

```python
compute_communication_cost(
    payload_size=16_384,   # at MAX_PAYLOAD_SIZE
    transmission_time=1.0, # at MAX_TRANSMISSION_TIME
    available_bandwidth=0, # no bandwidth
)
# → C = 0.5×1.0 + 0.3×1.0 + 0.2×1.0 = 1.0
```

### 9.3 Mid-Range

```python
compute_communication_cost(
    payload_size=8_192,     # 50% of max
    transmission_time=0.3,  # 30% of max
    available_bandwidth=500_000,  # 50% of reference
)
# S = 0.5,  T = 0.3,  B = 0.5
# C = 0.5×0.5 + 0.3×0.3 + 0.2×0.5 = 0.25 + 0.09 + 0.10 = 0.44
```

---

## 10. Deterministic Behaviour

The computation is entirely deterministic:

- No random operations.
- No learned parameters.
- Same inputs + same configuration = same output, always.

`RANDOM_SEED = 42` is included in the configuration for project-wide
consistency but is not used by the module itself.

---

## 11. Limitations

1. **Static weights:** The relative importance of payload size, time, and
   bandwidth is fixed (unless overridden per call).  In practice, the
   optimal weighting may depend on the current operating regime.
2. **No queuing / contention model:** The module does not consider
   multiple competing transmissions or queue depth.
3. **No energy model:** Power consumption is not included as a cost
   component.
4. **No compression awareness:** If the observation is compressed before
   transmission, the raw payload size may overestimate cost.
5. **Reference limits are not adaptive:** The normalisation references are
   static configuration values, not learned from traffic statistics.
6. **Scalar inputs only (v0.1):** The module processes one observation at
   a time.

---

## 12. Why the Baseline Is Not an FSO Model

**Communication Cost ≠ FSO Channel Quality.**

This module estimates the **resource burden** of transmitting data,
regardless of the physical channel technology.  It answers:

> *"Given the size of this observation and current channel conditions,
> how costly is it to send?"*

An FSO (Free-Space Optical) channel model would address a fundamentally
different question:

> *"Given atmospheric conditions (turbulence, fog, alignment), how
> reliably can data actually be delivered?"*

The FSO model would incorporate:

- Atmospheric attenuation coefficients
- Scintillation / turbulence (Rytov variance, Cn²)
- Geometric / pointing loss
- Bit-error rate (BER) estimation
- Fade statistics

These are **not** communication costs — they are channel-quality
indicators.  Merging them into a single score would obscure the
distinction between "expensive to send" and "unlikely to arrive."

---

## 13. Future FSO Integration

When an FSO channel model is implemented (future task), it will produce a
separate **channel quality** or **link reliability** score.  The VoI
engine can then combine:

| Factor                | Module               | Score |
|-----------------------|----------------------|-------|
| Novelty               | `src/novelty/`       | N     |
| Prediction Uncertainty| `src/uncertainty/`   | U     |
| Task Relevance        | `src/relevance/`     | R     |
| Temporal Importance   | `src/temporal/`      | T     |
| Communication Cost    | `src/communication/` | C     |
| Channel Quality (FSO) | *(future)*           | Q     |

into a unified VoI decision.

---

## 14. Important Conceptual Distinctions

Communication Cost is **not**:

| Concept                 | Module             | What It Measures                             |
|-------------------------|--------------------|----------------------------------------------|
| Novelty                 | `src/novelty/`     | How unusual the observation is overall        |
| Prediction Uncertainty  | `src/uncertainty/`  | Confidence of the classification model       |
| Task Relevance          | `src/relevance/`   | How relevant the observation is to the task   |
| Temporal Importance     | `src/temporal/`    | Change significance relative to prior obs     |
| Channel Quality (FSO)   | *(future)*         | Link reliability under atmospheric conditions |

These factors are **separate** components of the VoI formulation and are
combined at a higher level — not within this module.
