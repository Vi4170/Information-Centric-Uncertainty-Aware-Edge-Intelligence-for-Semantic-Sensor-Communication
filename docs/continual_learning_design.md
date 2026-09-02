# Continual Learning Architecture — Design (Task 16)

**Status**: design only. Nothing in this document has been implemented. No file under `src/` was modified for this task; `src/voi/` in particular is untouched and this design never proposes changing it.
**Date**: September 2, 2026
**Baseline this design builds on**: the validated CWRU → CNN → Novelty/Uncertainty/Relevance/Temporal/Cost → canonical VoI Engine → Decision pipeline, calibrated in Task 14 and validated in Task 15. CWRU is used here only as the concrete baseline to reason about — **CWRU alone does not demonstrate continual learning** (every recording is a single fixed condition throughout; there is no genuine condition drift to adapt to in this dataset). This design is written to be exercised by a future degradation dataset (IMS, XJTU-SY — see §5), not by CWRU.

---

## 1. Proposed Architecture

Continual learning is added as a **supervisory loop around the existing pipeline**, not a change to it. The existing pipeline (CNN → factors → VoI Engine → decision) keeps running unmodified on every incoming observation. Alongside it, a new **Condition Monitor** watches a rolling window of that pipeline's own outputs (novelty scores, predicted-class distribution, embeddings) for evidence of a *sustained* shift — as opposed to a single anomalous reading. When a sustained shift is detected and independently confirmed safe (§4), an **Adaptation Controller** updates only the components listed as adaptive in §2, through a versioned, gated procedure, and never touches `src/voi/`.

```
                    EXISTING PIPELINE (unchanged, Task 10/14/15 baseline)
   Sensor data --> CNN --> {probabilities, embedding}
                              |                |
                       Uncertainty          Novelty (reference: versioned, see §3)
                              |                |
                              +---> Relevance, Temporal, Cost --+
                                                                 v
                                                   canonical VoI Engine (src/voi/, FIXED)
                                                                 |
                                                            Decision (unchanged)

                    NEW SUPERVISORY LOOP (this design, not yet implemented)
   {novelty scores, predicted-class distribution, embeddings}  (read-only tap on the existing outputs above)
                              |
                              v
                    Condition Monitor (rolling-window drift detector, §3.1)
                              |
                    sustained shift detected?
                       |                  |
                      no                 yes
                       |                  v
                       |         Safety Gate (§4): is this data safe to learn from?
                       |                  |
                       |            yes   |   no -> log only, no adaptation
                       |                  v
                       |         Adaptation Controller (§2-3)
                       |          - extend novelty reference (append new prototype)
                       |          - (optional, gated) partial CNN fine-tune
                       |                  |
                       |                  v
                       |         Regression Gate (§4.3): re-evaluate on retained
                       |         validation set spanning ALL known conditions
                       |          pass -> commit new versioned artifact
                       |          fail -> reject, roll back, log
                       v                  v
                    (no change)   Pipeline now uses updated novelty reference
                                  and/or updated CNN on subsequent observations
```

`src/voi/` sits entirely inside the "existing pipeline, unchanged" box. It only ever consumes whatever the (possibly updated) Novelty/Uncertainty/Relevance/Temporal/Cost modules hand it — the same contract as today (`compute(novelty, uncertainty, task_relevance, temporal_importance, resource_cost)`, five floats in `[0, 1]`). Continual learning changes *what produces those five numbers over time*; it never changes how they're combined or decided on.

---

## 2. Adaptive vs. Fixed Components

| Component | Status | Notes |
| :--- | :--- | :--- |
| `src/voi/` (formula, weights, thresholds) | **Fixed** | Protected per project rules; never adapts automatically. Recalibration (Task 14-style) remains a separate, deliberate, human-reviewed task — not something continual learning triggers on its own. |
| Decision policy tiers (DISCARD/BUFFER/SUMMARY/TRANSMIT) | **Fixed** | Same reasoning as above. |
| Novelty reference (`DistanceNoveltyDetector`'s centroid) | **Adaptive** | Extended to a versioned, append-only *set* of prototypes (§3.2) — the lowest-risk, first-implemented adaptive component. |
| CNN convolutional backbone (`conv1d_1`, `conv1d_2`) | **Fixed** (by default) | Generic vibration feature extractors; kept frozen to minimize forgetting risk. Only revisited if Phase 2 (§6) proves Phase 1 insufficient. |
| CNN embedding layer + classifier head (`learned_embedding`, `output_probabilities`) | **Conditionally adaptive** | Only fine-tuned under the full gating procedure in §4, only with confirmed labels (§4.2), never automatically. |
| `src/relevance/` class-relevance mapping | **Fixed for CWRU's 4 classes** | Adaptive only in the sense that a *new* class (from a new dataset/condition) could be added with a human-assigned relevance weight — never inferred automatically, since relevance is an application-level judgment, not a learned statistic. |
| `src/temporal/`, `src/communication/` formulas | **Fixed** | Their *default parameters* (e.g. `DEFAULT_TEMPORAL_CHANGE_SCALE`) may need periodic manual recalibration per dataset (as Task 14 already established), but that is a scheduled task, not continual online adaptation. |
| `src/uncertainty/` estimator | **Fixed algorithm, monitored calibration** | The entropy formula itself doesn't adapt; but its *usefulness* must be re-checked after every CNN update (§4.4), since a newly fine-tuned model's confidence behaviour can change. |
| Condition Monitor's own drift thresholds | **Adaptive by design, but slow-moving** | Its baseline statistics are refreshed only when a new condition is accepted (never per-observation), to avoid the monitor drifting into blindness. |

---

## 3. Update Strategy

### 3.1 Detecting a new operating condition

A **single** high-novelty observation is not evidence of a new condition — it is indistinguishable from an ordinary fault (which is exactly what novelty is already supposed to flag) or sensor noise. The Condition Monitor instead looks for a **sustained** pattern over a rolling window of *N* consecutive observations (N configurable, e.g. one recording's worth):

1. **Novelty control-chart signal**: track the rolling mean of novelty scores restricted to observations the CNN currently predicts as Normal (or otherwise low-relevance). If this rolling mean exceeds the original reference distribution's `mean + k·std` (a standard statistical-process-control threshold) for the *entire* window (not a single spike), flag a candidate shift.
2. **Predicted-class-distribution shift**: in parallel, track a divergence measure (e.g. population stability index or KL divergence) between the *recent* window's predicted-class distribution and the *original training* class distribution. A genuine new operating condition typically shows both signals moving together (novelty rises **and** the classifier's output pattern becomes atypical); a transient fault burst typically shows high novelty briefly, classified confidently into an existing fault class, then reverts.
3. Only when **both** signals agree over the full window is a "candidate new condition" raised — this directly targets the project's stated concern (master doc §21): a new condition should not be permanently treated as anomalous, but a single anomalous reading also should not trigger relearning.

This is deliberately conservative: false negatives (missing a real new condition, catching it one window later) are preferred over false positives (relearning from what was actually just a fault burst or noise).

### 3.2 Updating the novelty/reference embedding model

The current `DistanceNoveltyDetector` fits **one** centroid from `Normal`-labelled training embeddings. Continually overwriting that single centroid with new data would silently redefine "normal" and could mask the very faults the system exists to catch. Instead:

- Maintain a **versioned, append-only set of reference prototypes**: `{prototype_v1 (original training fit), prototype_v2 (first accepted new condition), ...}`.
- Novelty score for an observation = distance to its **nearest** prototype among all accepted prototypes (not just the original one). An observation close to *any* previously-accepted normal mode is not novel; something far from *all* of them is.
- A new prototype is added only when the Condition Monitor's candidate shift (§3.1) also passes the Safety Gate (§4) and Regression Gate (§4.3) — i.e., a new prototype is a *committed, reviewed* addition, not a running average that drifts continuously.
- Old prototypes are **never deleted** by this process (only a separate, explicitly human-triggered consolidation/pruning step — out of scope for this design — could remove one), which is the primary mechanism preventing the reference model from forgetting what earlier conditions looked like.

### 3.3 Adapting CNN representations

Ordered from lowest to highest risk, and this design recommends implementing them **in this order**, validating each before moving to the next (see §6):

1. **No CNN change** (Phase 1, recommended first): only the novelty reference (§3.2) adapts. A new condition is recognized as "a new kind of normal-or-fault-adjacent embedding region" without asking the CNN to relabel anything. This alone lets the system stop treating a recognized-but-unlabelled new condition as perpetually maximally novel.
2. **Frozen-backbone partial fine-tuning** (Phase 2, gated, confirmed-labels only): unfreeze only the embedding dense layer (`learned_embedding`) and the classifier head (`output_probabilities`); keep `conv1d_1`/`conv1d_2`/`conv1d_3` frozen. Fine-tune on `(new confirmed-label samples) ∪ (rehearsal buffer, §4.5)`, low learning rate, few epochs, always followed by the Regression Gate (§4.3).
3. **Full-model fine-tuning or architecture growth** (explicitly not designed here): only worth considering if Phase 2 proves structurally insufficient (e.g., a genuinely new fault mechanism the frozen backbone can't represent at all). Flagged as future work, not part of this task's implementation plan.

---

## 4. Leakage / Forgetting Safeguards

### 4.1 When is an observation safe to use for learning?

- **Never** the CWRU (or any dataset's) designated test split — see §4.6.
- For extending the **novelty reference** (§3.2): safe once the observation belongs to a *cluster* that the Condition Monitor has flagged as sustained (§3.1) — i.e., safety here comes from **internal consistency/recurrence** (many similar observations agreeing with each other), not from any label, since by definition the CNN has no reliable label for a condition it wasn't trained on.
- For **CNN fine-tuning** (§3.3, Phase 2): safe only with a **confirmed label** (§4.2) — recurrence/consistency alone is not sufficient here, because a supervised loss trained on a wrong pseudo-label directly corrupts the classifier, unlike the geometric, error-tolerant nearest-prototype novelty update.

### 4.2 Pseudo-labels vs. confirmed labels

Two explicitly separate tiers, never mixed:

| Tier | Source | Allowed uses | Forbidden uses |
| :--- | :--- | :--- | :--- |
| **Pseudo-label** | The CNN's own predicted class; or a cluster/prototype ID assigned by the Condition Monitor to a new, still-unlabelled condition | Extending the novelty reference (§3.2); tagging observations for later human review; informing Temporal/Relevance analysis *diagnostically* | Any supervised loss update to the CNN's weights; assigning a new permanent class name or relevance weight |
| **Confirmed label** | A human (e.g. maintenance technician inspection), or an independently validated downstream process | CNN fine-tuning (§3.3 Phase 2); introducing a new permanent class into `src/relevance`'s `CLASS_RELEVANCE_MAP`-equivalent, with a human-assigned relevance weight | — |

Every pseudo-label-derived artifact (a new prototype, a provisional cluster tag) must be stored with an explicit `provisional: true` marker so it can be later pruned or promoted to confirmed without having contaminated anything supervised.

### 4.3 Preventing catastrophic forgetting

- **Regression gate (primary safeguard)**: before any adaptation (prototype addition or CNN fine-tune) is committed, re-evaluate the candidate updated system against a small, fixed, **retained validation set spanning every previously known condition** (not just the new one). If any previously-known condition's metrics (e.g. per-class F1, or per-class VoI-decision distribution) degrade beyond a defined tolerance, the update is **rejected and rolled back** — the running system keeps its previous, working state.
- **Rehearsal/replay buffer**: for any CNN fine-tuning step, mix a small, class-balanced sample of previously-seen data into the fine-tuning batch alongside the new confirmed-label data, so the gradient update doesn't overwrite old representations purely in service of the new condition.
- **Append-only reference versioning**: as in §3.2, old prototypes/model checkpoints are never overwritten in place — every accepted update produces a new version (`models/cwru_cnn_v{n}.keras`, `novelty_reference_v{n}.json`), so a bad update discovered later can always be rolled back to a known-good version.
- **(Candidate, not mandated)** Regularization-based forgetting mitigation (e.g. Elastic Weight Consolidation) is noted as a complementary technique worth evaluating in a future task if rehearsal alone proves insufficient — not designed in detail here to keep this design's first implementation phase small.

### 4.4 Uncertainty/calibration after adaptation

A CNN update (§3.3 Phase 2) can change how well-calibrated its confidence is — potentially in either direction. This design requires that **every** accepted CNN update be followed by re-running the equivalent of Task 13's uncertainty-distribution analysis on the retained validation set, and explicitly re-checking whether `src/uncertainty`'s entropy signal is still near-uninformative (as it is today) or has become more discriminating. If it changes materially, `VoIWeights.uncertainty` becomes a candidate for a fresh Task-14-style recalibration — a deliberate, reviewed, train/val-only step, not an automatic one. If a better uncertainty estimator (MC Dropout, ensembles) is introduced alongside continual learning, its own calibration must be refit using confirmed-label validation data post-adaptation, never test data.

### 4.5 Temporal Importance: volatility vs. genuine drift

Task 13/14/15 already established that the current `src/temporal/temporal.py` formula (mean absolute difference between *consecutive* windows) measures short-horizon **signal volatility** (how impulsive a single reading looks), not genuine **condition drift over time** — and that this is fine for CWRU's actual use of the score (impulsive fault signatures correlate with severity) but conceptually mismatched with what "temporal importance" should eventually mean for a degradation dataset.

This design proposes — **for a future task, not implemented here** — computing two distinct signals side by side rather than modifying the existing one:

- **Short-horizon volatility** (existing, unchanged): the current window-to-window score, kept exactly as-is since it already feeds the calibrated VoI formula correctly for CWRU-style data.
- **Long-horizon drift** (new, proposed): a much slower rolling statistic (e.g. an exponentially-weighted moving average of novelty or embedding-centroid distance) computed over many recordings/sessions rather than consecutive windows — this is what should genuinely answer "has the operating baseline shifted," and this signal, not the existing short-horizon one, should feed the Condition Monitor's §3.1 detection logic.

Keeping these separate avoids repeating Task 13's original problem (one formula asked to answer two different questions) and avoids modifying the existing, already-calibrated Temporal Importance factor as part of this design.

### 4.6 Preventing test-set leakage

- The designated test split (for CWRU, and for any future dataset) remains a **static, held-out benchmark forever** — structurally separate from any "deployment stream" or adaptation buffer the Condition Monitor and Adaptation Controller read from. No observation from a test split may ever enter a rehearsal buffer, a novelty-prototype fit, or a fine-tuning batch.
- Any new dataset (IMS, Paderborn, XJTU-SY) introduced for continual-learning experiments must define its own recording-level train/val/test split (matching the existing CWRU leakage-prevention methodology — Task 1) **before** any adaptation experiment touches it.
- Recalibration triggered by an adaptation cycle (§4.4) follows the same train/val-only discipline Task 14 already established — test data is touched only for periodic, final reporting, never for selecting a parameter.
- Concretely proposed for the next implementation task: a `deployment_log`/`adaptation_buffer` store, versioned and timestamped, kept structurally separate from `data/processed/<dataset>/`'s test arrays, with an explicit automated check (a new test) asserting no `observation_id` ever appears in both the test split and the adaptation buffer.

---

## 5. Dataset Requirements for Future Datasets

| Dataset | Why it fits this design | What it must provide |
| :--- | :--- | :--- |
| **IMS Bearing** | Run-to-failure lifecycle recordings — genuine condition drift over time, unlike CWRU's fixed-condition recordings. The natural first testbed for §3.1's drift detection and §4.5's long-horizon signal. | Timestamped, sequential recordings per bearing; recording-level metadata analogous to CWRU's `file_id`/`window_index`; its own recording-level train/val/test split before any adaptation experiment. |
| **XJTU-SY** | Similar accelerated-degradation lifecycle structure to IMS — a second, independent testbed for the same drift-detection and long-horizon-signal logic, useful for confirming the design generalizes rather than overfitting to IMS's specific degradation pattern. | Same structural requirements as IMS. |
| **Paderborn** | Multiple simultaneously-known operating conditions and fault types (not necessarily drift *over time*) — better suited to validating §3.1's "new operating condition" detection against conditions that are different-but-not-necessarily-sequential, and to testing whether the novelty-prototype approach (§3.2) generalizes across conditions rather than only within one recording's temporal order. | Condition/fault-type metadata per recording; its own recording-level split. |
| **MIMII (audio, later)** | A genuinely different modality — exercises whether this architecture's module boundaries (CNN/encoder → factors → VoI) hold up outside vibration, but is explicitly out of scope until the above are validated (per the project's own stated sequencing). | An audio-appropriate encoder replacing the CNN; everything downstream of the encoder (factors, VoI) should require no change if the module boundaries are respected. |

Each new dataset must independently re-derive any dataset-specific constant this design or Task 14 established empirically (e.g. a `DEFAULT_TEMPORAL_CHANGE_SCALE`-equivalent for its own signal statistics) — none of Task 14's CWRU-specific calibration values should be assumed to transfer.

---

## 6. Implementation Plan for the Next Task

Proposed as a sequence of small, independently-validated phases — each should be its own task, not one large implementation:

1. **Foundational, no model changes**: build the `deployment_log`/`adaptation_buffer` data structure and the leakage-prevention assertion test from §4.6. Nothing adaptive yet — this just proves the plumbing and the leakage guarantee exist before anything is allowed to learn from live data.
2. **Condition Monitor (read-only)**: implement §3.1's rolling novelty-control-chart and predicted-class-distribution-shift detector as a new, separate module (e.g. `src/continual/condition_detection.py`). Validate it only in *monitoring/logging* mode (raises flags, adapts nothing) against synthetic drift injected into held-out CWRU data or against a second dataset — never against the real CWRU test split.
3. **Novelty reference extension**: implement §3.2's versioned, append-only multi-prototype novelty reference, gated by the Safety Gate (§4.1) and Regression Gate (§4.3). This is the lowest-risk adaptive component and should be the first thing allowed to actually change system behaviour.
4. **Gated CNN fine-tuning**: only after (1)-(3) are demonstrated safe, implement §3.3 Phase 2 (frozen-backbone partial fine-tune, confirmed labels only, rehearsal buffer, full regression testing against every previously-known condition).
5. **Re-validate VoI behaviour after each phase**: re-run the Task 13/15 analysis methodology after every phase to check whether the calibrated weights/thresholds from Task 14 still produce sensible behaviour — treat this as a deliberate, reviewed checkpoint, not an automatic recalibration.
6. **Introduce a real degradation dataset (IMS first)** only once phases 1-4 are validated safely using CWRU-only or synthetic drift scenarios — this is when the design in this document gets its first genuine test, since CWRU itself cannot exercise real condition drift.

---

## 7. Test Result

`python -m unittest discover -s tests` — **139/139 passing**, unchanged from before this task. This was a design-only task: no file under `src/` or `tests/` was modified, so the repository's behaviour and test outcomes are identical to the Task 15 state. Confirms the design work did not disturb the validated, calibrated pipeline.
