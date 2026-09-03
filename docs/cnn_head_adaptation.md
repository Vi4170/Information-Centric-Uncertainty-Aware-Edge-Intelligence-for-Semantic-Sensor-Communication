# Leakage-Safe CNN Head Adaptation — Phase 4C (Task 23)

**Module**: `src/continual/cnn_head_adaptation.py`
**Status**: implements the first, conservative tier of `docs/cnn_continual_adaptation_design.md` Section 4. Trains a candidate model only; never persists, versions, or activates one — that remains a separate future operation. Not connected to any automatic trigger.

## Architecture

```
active_model
     | clone_model_with_weights()        <- independent copy; active_model never touched
candidate_model
     | freeze_backbone()                  <- verified globally, not assumed
     |
gate.evaluate(observations=train_records, <- fail-closed pre-check, reuses
              condition_monitor_results)     SafetyRegressionGate (Task 20) unmodified
     |
candidate_model.fit(X_train, y_train,
                     validation_data=(X_val, y_val))
     |
CandidateAdaptationResult (candidate + freeze report + per-condition accuracy)
     |
(separate call) evaluate_candidate_regression()  <- second gate.evaluate() call,
                                                      full safety + regression re-check
```

Two public entry points: `train_candidate_head()` (produces the candidate) and `evaluate_candidate_regression()` (gates it against the active model) — deliberately separate, so a caller can inspect a candidate before deciding whether to gate it, and so producing a candidate is never conflated with deciding to use it.

## Trainable / Frozen Layers

| Layer | Status |
| :--- | :--- |
| `conv1d_1`, `maxpool_1`, `conv1d_2`, `maxpool_2`, `conv1d_3`, `global_pool` | Frozen |
| `learned_embedding` (`src.cnn.config.EMBEDDING_LAYER_NAME` — the layer `NoveltyReference` prototypes are built from) | Frozen |
| `dropout` | Frozen (no trainable weights regardless) |
| `output_probabilities` | **The only trainable layer** |

**Freeze verification is whole-model, not just the named layers.** `freeze_backbone()` doesn't just check that the layers it was told about ended up with zero trainable weights — it inspects `model.trainable_weights` (every trainable weight anywhere in the entire model) and confirms each one belongs to a head layer. This matters: a check that only re-examines the layers it was explicitly told to freeze would happily report success even if a caller's `backbone_layer_names` list forgot one — `tests/test_cnn_head_adaptation.py::test_02` demonstrates exactly this failure mode with a deliberately incomplete list, and confirms the whole-model check catches it. After `.compile()`, `model.trainable_variables` (what the optimizer will actually update) is separately confirmed to contain only the head layer's own weight/bias variables (`test_03`).

## Data Policy

- **Confirmed labels only, enforced twice.** `_validate_confirmed_only()` rejects any non-`CONFIRMED` record (or a `CONFIRMED` record with no label value) before anything else happens — a fast, direct, explicit check specific to this module. Independently, `gate.evaluate()`'s existing safety checks (Task 20, unmodified) are also run as a pre-training gate, catching the same class of problem again from a different angle plus everything else it already validates (split, provenance, count, sustained evidence).
- **Test-split data can never enter this component.** Not a new check invented for this task — `SafetyRegressionGate`'s existing split re-validation (which itself doesn't trust that any `AdaptationRecord` came from a properly-validated `AdaptationBuffer`) is reused unmodified via the pre-training gate call.
- **Sustained-shift evidence is required, not optional**, via `condition_monitor_results`. CNN adaptation is a higher-risk decision than prototype admission (Task 21), so it is held to at least the same evidence standard: an empty sequence fails closed at the gate exactly as it does for prototype admission — this is Task 20's existing fail-closed design, not a new rule.
- **Validation never participates in optimization.** `X_val`/`y_val` are passed only via Keras's `validation_data=` argument (the same mechanism `src/cnn/train.py`'s own `train_cnn()` already uses) and are used solely for the per-condition accuracy comparison this module returns. `tests/test_cnn_head_adaptation.py::test_12` proves this directly: training with the *same* training data but *different* validation data produces byte-identical candidate weights.
- **Permanent test data is never referenced anywhere in this module.**

## Rehearsal Policy

Implemented exactly per `docs/cnn_continual_adaptation_design.md` Section 3, and only because the prerequisites turned out to be available without needing anything from the still-open gap that document flagged (`NoveltyReference` doesn't retain a prototype's original source observations — see below): `select_rehearsal_samples()` needs only each prototype's **centroid** (already available via `NoveltyReference.get_prototype(id).centroid`) plus a caller-supplied confirmed-label candidate pool, not the prototype's original training data.

For each pool sample: assign it to whichever prototype `NoveltyReference.nearest_prototype()` (reused unmodified — no distance/centroid logic duplicated) reports as nearest, then within each resulting group keep the `k_per_condition` samples with the smallest such distance (most representative first; ties broken by index — no randomness anywhere). The pool itself is validated exactly like training data (confirmed-only, fails closed on an empty/absent reference).

## Candidate Model Isolation

`clone_model_with_weights()` calls `keras.models.clone_model()` (fresh, independent layer objects) then copies weight *values* via `set_weights()`/`get_weights()` — no weight array is ever shared between `active_model` and a candidate. `test_04` trains a full candidate end to end and then asserts every one of `active_model`'s weight arrays is still byte-identical to a snapshot taken before training began.

## Model Versioning

This module deliberately does **not** implement `docs/cnn_continual_adaptation_design.md` Section 8's file-versioning scheme (`models/cnn/v{n}/`, `active_cnn_pointer.json`) — that is orchestration/persistence logic, not adaptation logic, and remains a separate future task. `train_candidate_head()` returns an in-memory `keras.Model` and nothing else; no file is written, no pointer is read or updated, and nothing is "activated" merely because training succeeded (`test_05`).

## Regression Protection

`evaluate_candidate_regression()` builds `baseline_metrics`/`candidate_metrics` from the per-condition (per true class) validation accuracy `train_candidate_head()` already computed for both models, and passes them into the *same*, unmodified `SafetyRegressionGate.evaluate()` used by prototype admission (Task 20/21) — no new regression-comparison logic was written. This directly reuses the gate's existing worst-condition-regression rule (not an average), so a regression hidden in one previously-known class can't be masked by improvement in another. `test_16` demonstrates both a real (small, likely-neutral) run and a synthetic forced-excessive-regression scenario that correctly REJECTs.

## Embedding-Space Invariant

**This is the specific design property that makes head-only adaptation safe for `NoveltyReference` without any re-encoding.** Because the entire embedding-producing path is frozen, `extract_embeddings()` is a fixed function of its weights — and those weights never change during head-only adaptation. `test_13` confirms this directly: embeddings for the same probe inputs, extracted from the active model and from a candidate that has just been trained for 3 epochs, are asserted **exactly equal** (`np.testing.assert_array_equal`, not a tolerance-based comparison) — not merely close. `test_14` confirms the practical consequence through `NoveltyReference`'s own API: a prototype's distance to a fixed embedding is unaffected by adaptation, so **no existing prototype ever needs re-encoding after this specific kind of adaptation.**

This is also why this module doesn't need to solve `docs/cnn_continual_adaptation_design.md` Section 9's flagged open gap (`Prototype` doesn't retain which source observations produced it, so re-encoding isn't currently possible) — head-only adaptation is the one tier of the design that doesn't require re-encoding at all, which is precisely why it was chosen as the first implementation.

## Uncertainty

Not modified — `src/uncertainty/`'s entropy formula is untouched, and this module never imports it. What *does* change: the final Dense layer's weights, and therefore the softmax output (and therefore whatever `src/uncertainty` computes from it) for any input, even for a head-only adaptation. `test_15` confirms probabilities *can* change (training toward a class the active model isn't already confident about measurably shifts its output) — this is expected and is evidence the mechanism works, not a defect. Per the design doc, validating *how* uncertainty behaves after a real adaptation is part of the regression evaluation, using the existing Task 13-style methodology on validation data — not something this module computes itself.

## VoI

Not touched, anywhere. `src/voi/` is never imported by this module.

## Limitations

- **Only representable within the frozen embedding space.** If a new condition needs features the backbone was never trained to extract, head-only adaptation will not help, no matter how the head is trained — the honest limitation `docs/cnn_continual_adaptation_design.md` Section 4 already stated. Partial-backbone adaptation is the designed (not yet built) escalation path, and it *would* need the `Prototype` re-encoding fix this task deliberately avoided needing.
- **No persistence/versioning yet** — every candidate lives only in memory for the lifetime of the caller's process. Building the versioned file scheme is separate future work.
- **Per-condition accuracy is grouped by CWRU class label**, the most direct available notion of "condition" for a supervised classifier; it is not literally the same grouping as `NoveltyReference`'s prototype ids (which may represent a different partitioning of the embedding space) — a caller wiring this into a full controller should be deliberate about which grouping a given `condition_id`/metrics key actually means.
- **CWRU cannot demonstrate genuine continual learning** (see `test_20`'s docstring and every prior task in this phase) — the real-data smoke test proves this mechanism runs cleanly against real provenance, real weights, and real embeddings; it does not, and does not claim to, prove the system adapts to a genuinely evolving operating condition.

## Why Head-Only Adaptation Is the First Implementation

Restated from `docs/cnn_continual_adaptation_design.md` Section 4, now demonstrated rather than only argued: it is the only tier that makes the embedding-space-consistency problem disappear instead of requiring a solution for it (§ above), it has by far the fewest trainable parameters (260, versus the full model), and — as this task's implementation confirms — it is fully buildable today with the existing `AdaptationBuffer`, `ConditionMonitor`, `NoveltyReference`, and `SafetyRegressionGate`, reusing every one of them unmodified.
