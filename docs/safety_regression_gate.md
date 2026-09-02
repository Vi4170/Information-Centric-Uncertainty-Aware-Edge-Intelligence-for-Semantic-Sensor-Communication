# Safety + Regression Gate — Phase 4A (Task 20)

**Module**: `src/continual/safety_regression_gate.py`
**Status**: decision infrastructure only. Never commits anything. Not connected to the Condition Monitor, the Adaptation Buffer, `NoveltyReference.add_prototype()`, or any CNN training code.

## Purpose

Given evidence about a candidate condition, decide whether a **future** controller may commit an adaptation update. This module performs no adaptation itself — no prototype is created, no model is retrained, no reference is modified. It only evaluates evidence and returns an auditable decision.

```
candidate evidence
        |
  safety checks
        |
 regression checks
        |
   GateDecision
```

## Inputs

`SafetyRegressionGate.evaluate(condition_id, observations, condition_monitor_results, candidate_embedding=None, existing_reference=None, baseline_metrics=None, candidate_metrics=None)`:

- `observations`: a sequence of `AdaptationRecord` (Task 17's type — reused directly, not duplicated) backing the candidate condition.
- `condition_monitor_results`: a sequence of `ConditionMonitorResult` (Task 18's type, reused directly) gathered while observing the candidate.
- `candidate_embedding` / `existing_reference`: an optional representative embedding and the current `NoveltyReference` (Task 19's type), queried **read-only**.
- `baseline_metrics` / `candidate_metrics`: optional `{condition: performance}` dicts a future adaptation phase would supply — this gate never computes them and never retrains anything to get them.

## Safety Checks

| Check | What it verifies | Failure is fatal? |
| :--- | :--- | :---: |
| Sufficient observation count | `len(observations) >= min_observation_count` | Yes |
| Source split permitted / no test leakage | Every observation's `split` is re-checked against the allowed set — **not trusted from the caller** | Yes |
| Provenance validity | Every `AdaptationRecord`'s fields are re-validated from scratch (see "Why re-validate?" below) | Yes |
| Confirmed-vs-pseudo labels | Counted and reported separately, never conflated; optionally (`require_confirmed_labels`) required to be 100% confirmed | Yes, only if configured to require confirmed labels |
| Sustained evidence (not an isolated anomaly) | Fraction of supplied `ConditionMonitorResult`s reporting `CANDIDATE_CONDITION_SHIFT` must meet `min_sustained_fraction`. **No evidence supplied → fails closed**, it is never skipped. | Yes |
| Distinguishable from existing prototypes | Raw distance from `candidate_embedding` to the nearest existing prototype must meet `min_distinguishability_distance`. No prototypes yet → trivially passes. Prototypes exist but no embedding supplied → **fails closed**. | Yes |

### Why re-validate provenance instead of trusting the caller?

`AdaptationRecord` (Task 17) is a plain dataclass with **no validation in its own constructor** — the validation lives in `AdaptationBuffer.add()`/`add_from_dataframe()`. A caller could construct `AdaptationRecord(split="test", ...)` directly, bypassing the buffer entirely. Test-set protection must not depend on every caller routing through the buffer correctly, so this gate re-checks split and provenance validity itself, independent of how the records were produced. `tests/test_safety_regression_gate.py::test_05` constructs exactly such a bypassing record and confirms the gate still catches it.

## Regression Checks

Optional: if neither `baseline_metrics` nor `candidate_metrics` is supplied, this check is skipped and the decision rests on safety alone (e.g. for a prototype-only candidate with no performance claim attached). If **either** is supplied, **both** must be supplied, non-empty, share identical condition keys, and contain only finite numeric values — otherwise the candidate is rejected (never silently ignored).

Per-condition regression = `baseline[condition] - candidate[condition]` (positive = got worse). The **worst** (maximum) per-condition regression governs the decision — not an average — so a regression hidden in one previously-known condition cannot be masked by improvement elsewhere (`test_15`).

| Worst-case regression | Outcome |
| :---: | :--- |
| ≤ `review_regression_threshold` (incl. negative = improvement) | No regression concern |
| between `review_regression_threshold` and `max_acceptable_regression` | `REVIEW` |
| > `max_acceptable_regression` | `REJECT` |

## Decision States

```python
class GateDecision(str, Enum):
    REJECT = "reject"   # caller must not commit
    REVIEW = "review"   # no automatic update; needs human judgement
    ACCEPT = "accept"   # supplied evidence satisfies current criteria;
                        # a future controller MAY commit — nothing has
                        # already been committed by this decision.
```

Combination logic (`_combine`): any safety failure → `REJECT` (safety is never "borderline" — a leakage or provenance failure is not a judgement call). If safety passes: no regression evidence supplied → `ACCEPT`; invalid regression evidence → `REJECT`; excessive regression → `REJECT`; borderline regression → `REVIEW`; otherwise → `ACCEPT`.

## Rejection Behaviour

Every rejection reason is recorded in `SafetyCheckReport.failed_checks` / `RegressionCheckReport.failed_checks` (machine-readable strings) and summarized in `GateReport.reasons` (human-readable). Nothing is rejected silently.

## Rollback Semantics

There is nothing to roll back here — this gate never applies anything. "Rollback" belongs to whatever future controller actually commits an update (per `docs/continual_learning_design.md` §4.3's versioned artifacts / regression-gate framing); this module's entire contribution is making sure that controller has a clear, auditable ACCEPT/REJECT/REVIEW signal *before* it commits anything.

## Test-Set Protection

Enforced independently at two points: (1) every observation's `split` is re-validated against the allowed set regardless of what the caller claims, and (2) a batch containing even one test-split observation rejects the **entire** candidate, matching `AdaptationBuffer`'s own atomic-rejection philosophy (Task 17). Proven by `test_03` (pure test split), `test_04` (29 train + 1 test → still rejected), and `test_05` (a record that bypassed `AdaptationBuffer`'s own validation entirely).

## Relationship to AdaptationBuffer, ConditionMonitor, and NoveltyReference

```
Condition Monitor  --(ConditionMonitorResult stream)-->
Adaptation Buffer  --(AdaptationRecord evidence)-->        Safety + Regression Gate  --(GateDecision)-->  [future] Prototype Admission
NoveltyReference   --(read-only nearest_prototype query)-->
```

This gate consumes the *output types* of Tasks 17–19 directly (no duplicated parallel types) but never calls a mutating method on any of them: it never adds to an `AdaptationBuffer` (it doesn't even hold one — callers pass already-retrieved records), never calls `NoveltyReference.add_prototype()`, and the `ConditionMonitor` has no path back to this gate at all — nothing here is triggered automatically by a monitor alert.

## Engineering Defaults vs. Experimentally Validated Thresholds

**None of `SafetyRegressionGateConfig`'s defaults are experimentally calibrated.** They are conservative engineering defaults, consistent with how `docs/continual_learning_design.md` already frames `src/voi/`'s own provisional weights/thresholds:

- `min_observation_count=30`, `min_sustained_fraction=0.8` — round, conservative numbers, not fit to any dataset.
- `min_distinguishability_distance=0.0` — deliberately permissive (only rejects an exact duplicate), because Task 19 established that no calibrated distance scale exists yet for the multi-prototype reference (raw, unnormalized distances only). Do not raise this without first establishing what a meaningful separation actually looks like on real multi-condition data.
- `review_regression_threshold=0.01`, `max_acceptable_regression=0.02` — round percentage-point figures, not derived from any real adaptation experiment (none has been run).

These should be revisited once a real degradation dataset (IMS/XJTU-SY, per `docs/continual_learning_design.md` §5) provides genuine multi-condition adaptation evidence to calibrate against. CWRU is not used anywhere in this task and would not be an appropriate basis for calibrating these thresholds even if it were, since CWRU cannot exercise genuine condition drift (Tasks 16/18).

## What This Task Does Not Implement

Prototype admission (no `add_prototype()` call anywhere in this module), CNN fine-tuning or any CNN modification, automatic triggering from Condition Monitor alerts, an actual commit/rollback mechanism, and any experimentally-calibrated threshold. Those remain for later phases per `docs/continual_learning_design.md`.
