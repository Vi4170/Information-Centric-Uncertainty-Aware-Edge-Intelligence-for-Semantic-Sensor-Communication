# Task 7 — Task Relevance Module

## 1. Purpose and Motivation

The **Task Relevance** module addresses a fundamental question in semantic sensor communication:

> **How relevant is a given sensor observation to the current application task?**

In an Information-Centric, Uncertainty-Aware Edge Intelligence architecture, not all sensor observations carry equal value for decision-making. A "Normal" bearing health reading during routine monitoring conveys far less actionable information than a detected "Inner Race Fault." Task Relevance quantifies this distinction with a single score.

**Task Relevance Score:** R ∈ [0, 1]
- **R = 0** → Little or no relevance to the current task
- **R = 1** → Highly relevant to the current task

### Status

> [!IMPORTANT]
> **Task 7 is standalone.** The Task Relevance module is NOT yet integrated with the VoI (Value of Information) Engine. Integration is planned for a future task.

---

## 2. Distinction from Novelty and Uncertainty

Task Relevance is conceptually distinct from the other semantic metrics computed in this project:

| Metric | Question Answered | Source |
|--------|-------------------|--------|
| **Uncertainty (Task 6)** | How confident is the model in its prediction? | Predictive entropy of softmax probabilities |
| **Novelty (Task 5)** | How different is this observation from training data? | Distance from learned embedding centroid |
| **Task Relevance (Task 7)** | How important is this observation for the application task? | Predicted class identity and/or probability distribution |

- **Uncertainty** is model-intrinsic — it reflects the classifier's own confidence.
- **Novelty** is data-intrinsic — it reflects how far an observation is from known patterns.
- **Task Relevance** is application-extrinsic — it reflects the importance assigned by the application context to different types of observations.

A model can be highly confident (low uncertainty) about a Normal reading, which is also non-novel (low novelty), yet the observation itself is not task-relevant (low relevance) for a fault-detection application.

---

## 3. Current CWRU Application Context

The module operates within the CWRU (Case Western Reserve University) bearing fault diagnosis pipeline using the standard 4-class classification:

| Class ID | Condition | Description |
|----------|-----------|-------------|
| 0 | Normal | Healthy baseline bearing operation |
| 1 | Inner Race Fault | Fault on the inner race of the bearing |
| 2 | Ball Fault | Fault on a rolling element (ball) |
| 3 | Outer Race Fault | Fault on the outer race of the bearing |

---

## 4. Input / Output Definitions

### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `predicted_class` | `int` | Yes (class_mapping) | Predicted class ID in [0, 3] |
| `probabilities` | `np.ndarray` (1D, length 4) | Yes (probability_weighted) | Softmax probability vector |
| `relevance_map` | `Dict[int, float]` | No | Override class → relevance mapping |
| `strategy` | `str` | No | `"class_mapping"` or `"probability_weighted"` |

### Output

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| Relevance Score | `float` | [0, 1] | Task relevance of the observation |

---

## 5. Strategies

### 5.1 Class-Mapping Strategy (`"class_mapping"`)

Direct deterministic lookup from the predicted class ID to a configured relevance value:

```
R = relevance_map[predicted_class]
```

**API:** `relevance_from_class(predicted_class, relevance_map=None, strategy="class_mapping")`

This is the default strategy. It is simple, interpretable, and requires only the argmax prediction.

### 5.2 Probability-Weighted Strategy (`"probability_weighted"`)

Uses the full softmax probability distribution to compute a weighted relevance score:

```
R = Σ P(class_i) × relevance(class_i)   for i ∈ {0, 1, 2, 3}
```

**API:** `relevance_from_probabilities(probabilities, relevance_map=None, strategy="probability_weighted")`

This strategy is more nuanced — it accounts for prediction uncertainty by blending relevance across classes proportionally to their predicted likelihoods.

**Example:** If the classifier outputs P = [0.1, 0.6, 0.2, 0.1]:
```
R = 0.1 × 0.10 + 0.6 × 1.00 + 0.2 × 0.90 + 0.1 × 0.90
  = 0.01 + 0.60 + 0.18 + 0.09
  = 0.88
```

---

## 6. Initial Relevance Mapping

The baseline class-to-relevance mapping used in this implementation:

| Class ID | Condition | Relevance Value | Rationale |
|----------|-----------|-----------------|-----------|
| 0 | Normal | 0.10 | Routine observation, low actionable information |
| 1 | Inner Race Fault | 1.00 | Critical fault requiring immediate attention |
| 2 | Ball Fault | 0.90 | Significant fault condition |
| 3 | Outer Race Fault | 0.90 | Significant fault condition |

> [!NOTE]
> These values are **initial design parameters**, chosen based on domain reasoning about the relative importance of different bearing health conditions for a fault-monitoring task. They are **NOT** optimized, learned, or calibrated against downstream decision utility. They serve as a reasonable starting point for the baseline module.

The mapping is fully configurable via `src/relevance/config.py` and can be overridden at runtime through the `relevance_map` parameter.

---

## 7. Score Interpretation

| Score Range | Interpretation | Example |
|-------------|---------------|---------|
| 0.00 – 0.20 | Very low relevance | Confident Normal prediction |
| 0.20 – 0.50 | Low-to-moderate relevance | Ambiguous prediction leaning Normal |
| 0.50 – 0.80 | Moderate-to-high relevance | Mixed fault probability |
| 0.80 – 1.00 | High relevance | Confident fault prediction |

---

## 8. Validation Rules

The module enforces the following validation constraints:

### Class ID Validation
- Must be an integer type (`int` or `numpy.integer`)
- Must be in range [0, NUM_CLASSES - 1] (currently [0, 3])

### Probability Vector Validation
- Must be a `numpy.ndarray`
- Must be 1-dimensional
- Must contain exactly `NUM_CLASSES` (4) elements
- All values must be finite (no NaN or Inf)
- All values must be non-negative
- Sum must be approximately 1.0 (within configurable tolerance, default 1e-2)

### Relevance Map Validation
- Must be a `dict`
- Must contain all class IDs from 0 to NUM_CLASSES - 1
- All relevance values must be numeric, finite, and in [0, 1]

### Output Validation
- Resulting relevance score is verified to be finite and within [0, 1]

All validation failures produce descriptive `ValueError` or `TypeError` messages.

---

## 9. Limitations

1. **Static Mapping:** The current class-mapping strategy uses fixed, manually-assigned relevance values that do not adapt to changing operational conditions.

2. **No Temporal Context:** Relevance is computed independently for each observation without considering temporal patterns (e.g., a fault reading after many Normal readings may be more relevant).

3. **No Application-Specific Optimization:** The relevance values are design parameters, not optimized against any downstream decision utility or cost function.

4. **Single-Task Assumption:** The module assumes a single fixed application task (bearing fault monitoring). Different application contexts would require different relevance mappings.

5. **No Severity Weighting:** Within a fault class, all observations receive the same relevance regardless of fault severity indicators.

6. **No Cross-Module Integration:** Task Relevance does not currently incorporate uncertainty or novelty signals, though these could provide complementary information.

---

## 10. Future Improvements

1. **Learned Task Relevance:** Train a model to predict task relevance from features, potentially using reinforcement learning with downstream decision utility as the reward signal.

2. **Application-Specific Task Definitions:** Support multiple simultaneous application tasks (e.g., predictive maintenance vs. quality control) with different relevance mappings.

3. **Contextual Relevance:** Incorporate contextual factors such as current operating conditions, maintenance schedule, or system criticality level into the relevance computation.

4. **Temporal Relevance:** Model time-dependent relevance patterns — for example, increasing relevance for repeated fault observations or decreasing relevance for known, already-acknowledged faults.

5. **Validation Against Downstream Decision Utility:** Calibrate relevance values by measuring their impact on downstream VoI-based communication and decision policies.

6. **Severity-Aware Relevance:** Incorporate fault severity indicators (e.g., fault size, vibration amplitude) to differentiate relevance within the same fault class.

7. **Multi-Sensor Fusion:** Extend relevance estimation to consider information from multiple sensor channels or sensor nodes simultaneously.

8. **VoI Engine Integration:** Integrate the Task Relevance score as a component signal into the VoI Engine alongside Uncertainty and Novelty scores (planned for a future task).

---

## 11. Module Structure

```
src/relevance/
├── __init__.py           # Package init — exports public API
├── config.py             # Configuration constants and relevance mapping
└── relevance.py          # Core relevance estimation logic

tests/
└── test_relevance.py     # Comprehensive unit test suite (17 test cases)
```

---

## 12. Configuration Reference

All configurable parameters in `src/relevance/config.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUM_CLASSES` | `int` | 4 | Number of classification classes |
| `RANDOM_SEED` | `int` | 42 | Reproducibility seed |
| `PROB_TOLERANCE` | `float` | 1e-2 | Tolerance for probability sum validation |
| `DEFAULT_STRATEGY` | `str` | `"class_mapping"` | Default relevance estimation strategy |
| `CLASS_RELEVANCE_MAP` | `dict` | `{0: 0.10, 1: 1.00, 2: 0.90, 3: 0.90}` | Class ID → relevance value mapping |
