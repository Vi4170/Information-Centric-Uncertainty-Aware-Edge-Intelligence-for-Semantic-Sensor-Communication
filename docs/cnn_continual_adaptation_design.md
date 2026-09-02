# Continual CNN Adaptation — Design (Task 22)

**Status**: design only. Nothing in this document has been implemented. No file under `src/` was modified for this task — `src/cnn/`, `src/voi/`, and every other protected module remain exactly as they were after Task 21 (commit `643f76f`).
**Baseline this design builds on**: the working, gated prototype-admission loop (Tasks 17–21: `AdaptationBuffer`, `ConditionMonitor`, `NoveltyReference`, `SafetyRegressionGate`, `GatedPrototypeAdmissionController`) and the current CNN training pipeline (`src/cnn/model.py`, `src/cnn/train.py`) — a single 1D CNN (`Conv1D×3 → GlobalAveragePooling1D → Dense(64, "learned_embedding") → Dropout → Dense(4, "output_probabilities")`), trained once via `train_cnn()`/`run_training_pipeline()`, saved to one fixed path (`models/cwru_cnn_baseline.keras`) with no versioning today.

---

## 1. Separation of Decisions

**Prototype admission ≠ CNN adaptation. These are two independent decisions with independent gates.**

Task 21 already proved this works one-directionally: `GatedPrototypeAdmissionController` can add a new prototype to `NoveltyReference` — recognizing a new operating condition exists — **without touching the CNN at all**. That capability is not incidental; it is the intended design. A new condition becoming "known" to the novelty reference and a new condition becoming something the *classifier* understands are different claims requiring different evidence:

- Admitting a prototype only claims: "this region of embedding space, produced by the *current, unchanged* CNN, recurs consistently and is worth recognizing as not-novel going forward."
- Adapting the CNN claims something much stronger: "the CNN's learned representation and/or classification boundary should change" — which risks the CNN's entire existing behavior, not just one factor's reference table.

Consequently, CNN adaptation requires its **own** explicit gated decision, evaluated separately from (and typically after) prototype admission, with its own regression evidence, its own ACCEPT/REJECT/REVIEW outcome, and its own rollback mechanism (§7). A prototype being accepted is never sufficient justification, by itself, to adapt the CNN — it only supplies part of the evidence (the confirmed-label data for the new condition) that a *future* CNN-adaptation proposal would need.

---

## 2. Adaptation Data

Reusing exactly the discipline already built and tested in Tasks 17/20/21, extended to the CNN-specific case:

- **Only `train`/`val`-split data already sitting in `AdaptationBuffer` may be used.** `AdaptationBuffer.add()` already refuses `split="test"` outright (Task 17); nothing about CNN adaptation changes or should be allowed to bypass that.
- **Permanent test data is never used for adaptation, at any stage** — not for training, not for the adaptation-internal validation split (§5), not for early stopping. It is touched only for the same kind of periodic, final, separate reporting Tasks 14/15 already established for VoI calibration.
- **Confirmed labels and pseudo-labels remain explicitly separated**, exactly as `LabelStatus.CONFIRMED` / `LabelStatus.PSEUDO` already distinguish in `AdaptationRecord` (Task 17) and as `SafetyRegressionGate`'s `confirmed_count`/`pseudo_count` reporting and `require_confirmed_labels` flag already enforce (Task 20).
- **Pseudo-labels must never automatically become supervised CNN training labels.** This is not a new rule to invent — it is Task 16's original design (§4.2) and is already mechanically enforceable today: a CNN-adaptation proposal must construct its `SafetyRegressionGate` with `require_confirmed_labels=True`, which already rejects any candidate evidence containing so much as one pseudo-labelled observation (Task 20, `test_06`). Prototype admission may keep `require_confirmed_labels=False` (pseudo-label recurrence is legitimate evidence there); CNN adaptation must not.
- **Every adaptation sample requires provenance** — reuse `AdaptationRecord` as-is (`observation_id`, `dataset`, `split`, `source_recording_id`, `label_status`, `label`). No new provenance type should be invented for CNN adaptation; the same record that would back a prototype's evidence is exactly what should back a CNN-adaptation candidate's training sample.

---

## 3. Rehearsal

**Purpose**: prevent catastrophic forgetting of previously known conditions when the CNN is nudged toward a new one — per `docs/continual_learning_design.md` §4.3's rehearsal/replay recommendation, not yet built.

| Property | Design |
| :--- | :--- |
| Per-condition representation | For each condition already represented by a `Prototype` in `NoveltyReference`, and for each original CWRU class, retain up to `K` rehearsal samples (an `AdaptationRecord` reference + its known label). |
| Deterministic sample selection | **No randomness.** For a given condition, rank its available confirmed-label samples by embedding distance to that condition's own prototype centroid (ascending — most representative first; ties broken by `observation_id` lexicographic order) and take the first `K`. This is fully reproducible from the same `NoveltyReference` + `AdaptationBuffer` state, with no seed to manage or record. |
| Memory limit | `K` per condition (engineering default `K=50`, **not experimentally calibrated** — chosen only to be small enough to bound memory and large enough to plausibly cover a condition's variability; revisit once real multi-condition adaptation data exists). |
| Balancing strategy | Equal `K` per condition, not per raw sample count — a condition with 10,000 buffered observations and one with 40 both contribute the same rehearsal weight, so a numerically dominant condition cannot crowd out a rare one during adaptation training. |
| Provenance | Every rehearsal sample is still a real `AdaptationRecord` with its original `source_recording_id`/`dataset`/`split`/`label_status` intact — rehearsal is a *selection* of existing evidence, never a synthesized or fabricated sample. |
| Versioning | A rehearsal set is a deterministic function of (a specific `NoveltyReference` version, a specific `AdaptationBuffer` state, `K`) — snapshot and record exactly those three alongside any candidate CNN's metadata (§8), so a rehearsal set used to train a given candidate is always reproducible and auditable, never an untracked ad hoc sample. |
| Leakage checks | Reuse `AdaptationBuffer.verify_no_test_leakage()` (Task 17) against the assembled rehearsal set before it is allowed anywhere near a training loop — the same function already used for buffer-level leakage checks, not a new one. |

---

## 4. CNN Adaptation Scope

Three options, evaluated against this project's specific architecture (`src/cnn/model.py`):

| Option | What's trainable | Forgetting risk | Embedding-space impact |
| :--- | :--- | :--- | :--- |
| **Classifier/head-only** | `output_probabilities` Dense layer only | Lowest (fewest parameters change) | **None — embedding space is mathematically identical before and after**, since every layer that produces the embedding (`conv1d_1/2/3`, `learned_embedding`) is frozen |
| Partial backbone | `learned_embedding` + `output_probabilities`, `conv1d_1/2/3` frozen | Moderate | Changes — every existing prototype's centroid becomes stale in the new space (§9) |
| Full fine-tuning | Everything | Highest | Changes substantially |

**Chosen first implementation: classifier/head-only adaptation.**

Rationale: this is the only option that makes §9's embedding-consistency problem disappear rather than requiring a solution for it. With the embedding backbone frozen, `extract_embeddings()` is provably a fixed function of its input regardless of any head-only adaptation — old prototypes, old novelty scores, and old distances all remain exactly as valid as they were before the adaptation, with zero re-encoding needed. Combined with having by far the fewest trainable parameters (a single 64→4 dense layer), this is the lowest-risk possible starting point for a first CNN-adaptation implementation, and keeps this task's scope aligned with "conservative first step" rather than committing to the harder embedding-consistency problem on day one.

**Known limitation, stated honestly rather than hidden**: head-only adaptation can only succeed if the new condition is already *linearly separable* within the existing, frozen 64-D embedding space — e.g., a new severity level or new operating parameter of an already-recognized fault mechanism. If a genuinely new fault mechanism produces vibration signatures the frozen backbone was never trained to extract features for, head-only adaptation may simply underperform, no matter how the classifier head is trained. **Partial-backbone adaptation is the designed escalation path** if and when real evidence (from the regression evaluation in §6) shows head-only adaptation is insufficient for a specific new condition — not a default assumed to always be reachable. Full fine-tuning is not part of this design's near-term plan at all; it is noted only as the theoretical ceiling, with the highest risk and least justification for a project at this stage.

---

## 5. Adaptation Dataset Split

| Split | Contents | Role |
| :--- | :--- | :--- |
| **Adaptation-training data** | Confirmed-label new-condition samples (from `AdaptationBuffer`, `split="train"`) + the rehearsal set (§3) | The only data any gradient update may use. |
| **Adaptation-validation data** | Confirmed-label new-condition samples with `split="val"` in `AdaptationBuffer`, held out *before* any training begins | Used only for early-stopping and for computing the candidate's regression-evaluation metrics (§6). **Never used to update model weights.** |
| **Permanent test data** | The CWRU (or future dataset's) fixed `X_test`/`y_test` | Untouched by the entire adaptation process. Touched only afterward, for the same kind of periodic, separate reporting already established for VoI calibration (Tasks 14/15) — never to select a threshold, a checkpoint, or an accept/reject decision. |

This mirrors the exact discipline `src/cwru_pipeline`'s recording-level train/val/test split already enforces for the *original* CNN training — adaptation just re-applies the same discipline one level up, using `AdaptationBuffer`'s own `split` field (which already refuses `"test"`) rather than inventing a new split concept.

---

## 6. Regression Evaluation

Before a candidate CNN may be activated, it is compared against the currently active CNN — never evaluated in isolation. Evidence gathered on the adaptation-validation data (§5) only, never on test data:

| Dimension | What is compared |
| :--- | :--- |
| Previously known conditions | Per-condition accuracy/F1 on rehearsal-representative validation samples for every condition the active CNN already handles |
| Newly learned condition | Accuracy/F1 on the new condition's own held-out adaptation-validation slice |
| Per-condition performance | Every condition individually — never collapsed into one number |
| Overall validation performance | A single aggregate figure, reported alongside, never in place of, the per-condition breakdown |
| Prediction behavior | Confusion-matrix shift: does the candidate confuse previously-well-separated conditions with each other now? |
| Embedding behavior | For head-only adaptation (§4): provably unchanged, so this check is a no-op confirmation, not a live measurement. For any future backbone-touching adaptation: measure drift as the change in distance from each previously-known condition's validation samples to that condition's *existing* prototype centroid — a large increase indicates the representation has moved even for data the CNN should still recognize well. |
| Uncertainty behavior | Re-run the Task 13/15-style predictive-entropy distribution analysis (`src/uncertainty`, unmodified) on the candidate's validation predictions and compare its shape/mean to the active CNN's — see §10. |

**Worst-condition regression governs the decision, never an average** — exactly the principle `SafetyRegressionGate`'s `RegressionCheckReport` (Task 20) already implements (`worst_condition_id`/`worst_regression`, taking the maximum per-condition drop, not a mean). **This is directly reusable, not something to reimplement**: a CNN-adaptation proposal should construct its own `SafetyRegressionGateConfig` (likely with stricter `review_regression_threshold`/`max_acceptable_regression` than the prototype-admission defaults, since a CNN mistake is more consequential than an unnecessary prototype) and call the *same* `SafetyRegressionGate.evaluate()`, passing `baseline_metrics`/`candidate_metrics` keyed by condition exactly as its existing interface already expects.

---

## 7. Rollback

```
current model (active)
      |
candidate model (trained separately, never overwrites the active model file)
      |
validation/regression evaluation (§6, via SafetyRegressionGate)
      |
   ACCEPT?
   /    \
 NO      YES
  |       |
retain   activate candidate
current  (atomic pointer swap, §8)
model
```

The mechanism that makes this atomic and safe is **never training in place**: a candidate is always written to its own versioned path (§8), and the currently-active model is identified only through a small, separate pointer record — never by a fixed, overwritten filename. "Activating" a candidate is rewriting that one pointer record, a single, small, atomic operation; it is not copying weights over the previous file. A rejected candidate's file may be retained (for audit/debugging) or discarded per a retention policy, but **it is never referenced by the active pointer**, so a rejected candidate can never end up serving traffic by any code path that simply "loads the model" — matching this task's explicit requirement that rejected candidates must never replace the active model.

---

## 8. Model Versioning

| Concept | Proposed scheme |
| :--- | :--- |
| Active CNN | `models/active_cnn_pointer.json` — a single small file: `{"active_version": n}`. Every runtime component that needs "the current CNN" reads this pointer, then loads `models/cnn/v{n}/model.keras`. Rewritten only on ACCEPT (§7). |
| Candidate CNN | `models/cnn/v{n+1}/model.keras` — written by the (future, not-yet-built) training step, evaluated, but never pointed to unless accepted. |
| Accepted CNN | Once accepted, the candidate's version number simply becomes the new value of `active_version` — there is no separate "accepted" artifact distinct from the candidate; acceptance is the pointer update itself. |
| NoveltyReference | Already versioned (Task 19, `NoveltyReference.version` + `.history`) and already persistable (`save_json`/`load_json`). Persist each accepted state as `data/novelty_reference/v{m}.json`. |

**Auditability across the two version lines**: every `models/cnn/v{n}/metadata.json` (proposed, alongside the model file) records which `NoveltyReference` version (`m`) it was validated against, and — critically, per §9 — whether that CNN version required the reference to be re-encoded, and if so, the new reference version (`m'`) produced as a result. This makes "which CNN was this reference built for, and which reference is this CNN compatible with" a directly answerable, recorded fact, never an implicit assumption.

---

## 9. CRITICAL: Embedding-Space Consistency

**The CNN produces the embeddings `NoveltyReference` stores centroids in. If the CNN's embedding-producing layers change, every existing prototype's centroid was computed in a coordinate system that no longer matches what the new CNN produces — and comparing a new observation's new-space embedding to an old-space centroid is meaningless, not just imprecise.** This must never be silently assumed away.

**This design's strategy, tied directly to §4's chosen scope:**

- **Classifier/head-only adaptation (the chosen first implementation) requires no action here at all** — the embedding backbone is frozen by construction, so `extract_embeddings()` is provably identical before and after. Every existing prototype remains exactly as valid as it was. This is stated in §4 as the primary reason head-only adaptation was chosen first: it is the only option that avoids this problem entirely rather than needing to solve it.
- **Any future backbone-touching adaptation (partial or full) MUST re-encode every existing prototype** using the newly accepted CNN before that CNN is trusted for novelty scoring. Concretely: for each existing `Prototype`, re-run the accepted CNN's `extract_embeddings()` over the *original observations that produced it* and recompute its centroid — producing a new `NoveltyReference` version, paired with the new CNN version (§8), while the old reference+old CNN pair remains available for rollback.

**A real gap this analysis surfaces, not papered over**: `NoveltyReference`'s current `Prototype` (Task 19) stores only the resulting **centroid** and `n_source_embeddings` — it does **not** retain which original observations produced it. Re-encoding as described above is therefore **not yet possible** with today's `Prototype` structure. This design does not implement a fix (that would be `src/continual/novelty_reference.py` changes outside this task's design-only scope), but explicitly flags the requirement: **before any backbone-touching CNN adaptation phase can be built, `Prototype` (or an adjacent provenance log) must be extended to retain enough of a pointer to its original source observations (e.g., their `observation_id`s) to allow re-encoding.** This is precisely why classifier/head-only adaptation is not merely "a reasonable first choice" but the only choice that can be implemented *without* first solving this open provenance gap.

---

## 10. Uncertainty

Even head-only adaptation changes the final Dense layer's weights, which changes the softmax output for every input — and therefore changes `src/uncertainty`'s predictive-entropy value for every input, even though the entropy **formula** itself (`compute_predictive_entropy`, protected, unmodified) does not change at all. This is expected and is not itself a problem; it must simply be **measured, not assumed**.

Validation procedure (part of §6's regression evaluation, not a separate mechanism): re-run the Task 13/15-style entropy distribution analysis on the candidate CNN's adaptation-validation predictions, and compare it to the active CNN's own distribution on the same data. Two outcomes are both acceptable results of this analysis, not failure conditions in themselves:

- Entropy stays similarly (un)informative — consistent with the CNN remaining about as confident as before.
- Entropy becomes more informative (e.g., the CNN is now less falsely overconfident on the previously-known conditions near the new one) — a genuinely interesting finding, but one that should trigger a **separate, deliberate, future Task-14-style recalibration** of `VoIWeights.uncertainty` if and when it's judged worth acting on. It must never trigger an automatic change to `src/voi/`.

---

## 11. VoI Compatibility

The canonical VoI formula, weights, and thresholds (`src/voi/`) are never touched by any part of this design, at any stage.

| May change after CNN adaptation | Stays fixed |
| :--- | :--- |
| Uncertainty (predictive entropy values — logits change, formula doesn't) | The VoI formula, weights, and decision thresholds (`src/voi/`) |
| Novelty (**only** if a future backbone-touching adaptation is accepted — see §9; unaffected by head-only adaptation) | `src/relevance/`'s class-relevance mapping (a fixed, human-assigned table, not learned) |
| Which observations receive which predicted class (and therefore which relevance value they're looked up under) — a *routing* change, not a change to the relevance table itself | `src/temporal/`'s formula (operates on raw signal, not CNN output) and `src/communication/`'s formula (independent of the CNN entirely) |

**Revalidation procedure after any accepted CNN update**: re-run the existing, unmodified `src/evaluation/voi_behaviour_analysis.py` (Task 13/14/15's methodology) against the *new* CNN's outputs on adaptation-validation data (never test data, until a final periodic check) to confirm VoI decisions still behave sensibly under the updated upstream signals. If this reveals the calibrated weights (Task 14) no longer produce sensible behavior, that is a trigger for a separate, deliberate, future recalibration task — never an automatic adjustment bundled into the CNN-acceptance decision itself.

---

## 12. Complete Continual-Learning Lifecycle

```
Known Condition A
      |
New Condition B detected                    <- ConditionMonitor (Task 18, existing)
      |
Gated prototype admission                    <- GatedPrototypeAdmissionController (Task 21, existing)
      |                                          -> NoveltyReference version increments
      |                                             CNN is NOT touched here (§1)
      v
CNN adaptation proposal                      <- NOT YET BUILT. Assembles confirmed-label
      |                                          evidence for condition B + rehearsal (§3)
New-condition data + rehearsal               <- NOT YET BUILT
      |
Candidate CNN                                <- NOT YET BUILT. Head-only training (§4),
      |                                          written to its own versioned path (§8),
      |                                          never overwriting the active model.
      v
Regression/safety evaluation                 <- Reuses SafetyRegressionGate (Task 20)
      |                                          with a stricter, CNN-specific config
   ACCEPT?
   /    \
 NO      YES
  |       |
rollback/retain   activate CNN               <- Atomic pointer swap (§7/§8)
previous CNN            |
                  revalidate VoI behaviour   <- Reuses voi_behaviour_analysis.py (§11)
                  on validation data (not automatic recalibration)
```

Everything left of "CNN adaptation proposal" already exists and is tested (Tasks 17–21, 228/228 passing). Everything from "CNN adaptation proposal" onward is this design's proposal for a **future** implementation task — none of it exists yet.

---

## 13. CWRU Limitation

Consistent with every prior task in this phase (16, 18, 19, 20, 21): **CWRU's condition structure is useful for demonstrating that this design's pipeline mechanics work — that data flows correctly, provenance is preserved, splits are respected, and decisions are auditable — but it does not, by itself, prove genuine temporal continual learning.** CWRU's conditions are recordings concatenated back-to-back, each internally homogeneous and fixed for its entire duration (Task 18's `docs/condition_monitor.md` already documented this exact limitation for the Condition Monitor: a `CANDIDATE_CONDITION_SHIFT` on CWRU's test stream reflects block/recording boundaries, not evolving operating conditions). Applying this CNN-adaptation design to CWRU would demonstrate "can the mechanism correctly adapt to a distinctly different, already-fully-formed block of data" — a real and useful mechanical test — but not "can the mechanism track a condition that is gradually degrading or drifting over time," which is the scenario continual learning is ultimately meant to handle.

**A genuinely degrading, time-evolving dataset — IMS or XJTU-SY, both already identified in `docs/continual_learning_design.md` §5 — will be required to validate this design's actual continual-learning value**, once implemented. Any performance numbers this design's future implementation produces on CWRU alone should be reported as pipeline-correctness evidence, not as continual-learning validation.

---

## Testing

No test file was added for this task. This is a documentation-only design task per its own scope ("do not implement CNN training or fine-tuning... do not modify the existing CNN"); no new code, configuration object, or data structure was introduced that would have anything meaningful to unit-test — creating a test file here would either test nothing real or would have to fabricate a CNN-training scenario, which this task explicitly prohibits ("do not create fake CNN-training tests"). The existing suite was re-run to confirm this task introduced no regressions: **228/228 passing**, unchanged from before this task (matching the same approach already used for Task 16, the prior documentation-only design task in this series).
