# Gated Prototype Admission Controller — Phase 4B (Task 21)

**Module**: `src/continual/admission_controller.py`
**Status**: the first REAL, controlled continual-learning loop. Still not connected to the CNN, `src/voi/`, or automatic triggering — it must be driven explicitly by a caller.

## Controller Role

Everything built in Tasks 17–20 existed independently and unwired: a buffer that could hold eligible observations, a monitor that could flag a sustained shift, a reference that could hold multiple prototypes, and a gate that could decide whether evidence justified an update — but nothing connected them. This controller is that connection, and nothing else: it owns no novelty math, no distance computation, no split-validation algorithm of its own beyond a fast-fail guard clause, and no gate logic. It holds already-constructed instances of all four components (dependency injection) and coordinates the sequence in which they're called.

## Interaction Between Tasks 17–20

```
observations
     |
ConditionMonitor.observe()            <- Task 18, called unmodified
     |
candidate evidence (tracked per condition_id, inside this controller)
     |
AdaptationBuffer.add()                <- Task 17, called unmodified
     |
SafetyRegressionGate.evaluate()       <- Task 20, called unmodified
     |
   ACCEPT?
    /    \
  NO      YES
  |        |
STOP   NoveltyReference.add_prototype()   <- Task 19, called unmodified
            |
     new reference version
```

## Candidate Collection

`observe(observation_id, embedding, novelty, predicted_class, condition_id, dataset, split, source_recording_id, label_status, label=None, window_index=None)` is the single entry point per observation:

1. **Split is checked first, before anything else runs.** A forbidden/unrecognized split raises immediately — not even `ConditionMonitor`'s rolling window is touched. `AdaptationBuffer.add()` then re-validates independently as a second layer (defense in depth, not duplicated trust).
2. The embedding is validated (shape/finiteness) and the novelty/class pair is routed to `ConditionMonitor.observe()`.
3. Only if `AdaptationBuffer.add()` succeeds is the returned `AdaptationRecord`, the embedding, and the `ConditionMonitorResult` retained — grouped under the caller-supplied `condition_id`, entirely within this controller's own bookkeeping (`AdaptationBuffer` has no concept of "candidate condition"; it only permanently logs individual records).

Nothing here is "silently accumulated" — this controller does not decide on its own that an observation belongs to a coherent emerging condition; the caller assigns `condition_id` explicitly at `observe()` time, informed by whatever they already know (e.g. the `ConditionMonitorResult.status` just returned).

## Safety-Gate Dependency

`attempt_admission(condition_id, prototype_id, source_dataset, source_condition, baseline_metrics=None, candidate_metrics=None)` is a second, explicit, separate call — it never runs automatically inside `observe()`. It assembles the tracked evidence for `condition_id` (records, embeddings, monitor results) and calls `SafetyRegressionGate.evaluate()` exactly once. **There is exactly one call site for `NoveltyReference.add_prototype()` in this entire module, and it is reachable only when `gate_report.decision == GateDecision.ACCEPT`.** REJECT and REVIEW both take the same no-op path: the reference is untouched, and — importantly — the tracked evidence for `condition_id` is *also* left untouched, so the caller may keep observing and retry later rather than losing everything on one unlucky evaluation.

## Prototype Admission

On ACCEPT, the candidate embedding is the mean of every embedding tracked under `condition_id` (in observation order — deterministic), and this array is handed to `NoveltyReference.add_prototype()` unchanged — the exact same centroid computation Task 19 already performs on whatever it's given; this controller invents no new representation or aggregation rule. `source_split` for the new prototype is the *majority* split among the candidate's own records (ties broken deterministically by first-seen order) — candidate evidence may legitimately mix `train` and `val` (the gate does not require split homogeneity, only that every split is permitted), but `NoveltyReference` needs one split value for its own provenance field. On success, the controller clears its own bookkeeping for `condition_id` (the now-permanently-captured evidence shouldn't silently double-count into a future admission attempt) — `AdaptationBuffer`'s own permanent, append-only log is untouched, since it has no remove operation by design.

## Duplicate-Condition Protection

This controller adds no new "is this a duplicate?" logic of its own — it relies entirely on `SafetyRegressionGate`'s existing distinguishability check (Task 20), by making sure the gate call is always given a real `candidate_embedding` and the live `existing_reference`. If the candidate's mean embedding is too close to an already-known prototype, the gate's safety check fails regardless of how convincingly `ConditionMonitor` reported a sustained shift — a temporary variation within an already-known condition can never manufacture a duplicate prototype just because the monitor was noisy. `tests/test_admission_controller.py::test_06` demonstrates this directly: sustained-shift evidence with an embedding indistinguishable from existing prototype "A" is rejected.

## Leakage Safeguards

Layered, not single-point:

1. `observe()`'s own guard clause rejects a forbidden split before touching the monitor or buffer at all.
2. `AdaptationBuffer.add()` re-validates split and provenance independently.
3. `SafetyRegressionGate.evaluate()` re-validates split and provenance *again*, from scratch, specifically because `AdaptationRecord`'s own constructor performs no validation (a caller could bypass `AdaptationBuffer.add()` entirely by constructing one directly) — this gate does not trust that any record it receives necessarily came through a validated path.

A test-split observation therefore cannot enter a candidate at any of three independent points, and a batch mixing permitted and test-split calls can never let the test-split portion through (`test_09`, `test_10`).

## Rollback / No-Op Behaviour on REJECT/REVIEW

There is nothing to roll back: on REJECT or REVIEW, no mutation has occurred anywhere — not to `NoveltyReference`, not to the controller's own tracked evidence, not to `AdaptationBuffer` (which never has anything removed from it regardless of outcome). "Rollback" in the fuller sense `docs/continual_learning_design.md` describes (versioned model/reference artifacts a human can revert to) still belongs to a future phase; this controller's contribution is simply that REJECT/REVIEW are true no-ops, verified directly by `test_15`.

## Why CNN Adaptation Is Still Excluded

This task connects the four *reference-side* components into a working admission loop for the novelty reference only. It does not, and structurally cannot, touch the CNN: `admission_controller.py` never imports `src/cnn/`, never calls `model.fit`/`compile`, and the only representation it ever handles is a plain `np.ndarray` embedding the caller already computed elsewhere — exactly the same embedding contract every other module in this project already uses. Gated CNN fine-tuning (Task 16's Phase 4, "frozen-backbone partial fine-tune, confirmed labels only, rehearsal buffer, full regression testing") remains a separate, later, higher-risk task per `docs/continual_learning_design.md`'s own staged ordering — this is the first, lower-risk half of Phase 4, not the whole of it.

## What This Is (and Is Not)

**This is the first prototype-adaptation mechanism** — a novelty reference that can grow a new, explicitly validated prototype through a fully gated, auditable, append-only process. **It is not full continual learning**: the CNN's classifier, embeddings, and uncertainty estimator are all still fixed forever; only the novelty reference's *set of known conditions* can grow, and only through this controller's single, gated path. CWRU is used in one lightweight compatibility test (`test_16`) to confirm real provenance fields (dataset id, `file_id`, `split` values) plug into this controller correctly — it is not used to demonstrate, and does not demonstrate, genuine operating-condition drift or continual learning, consistent with every prior task in this phase (16, 18, 19, 20). No threshold anywhere in this controller or its tests was tuned against CWRU.
