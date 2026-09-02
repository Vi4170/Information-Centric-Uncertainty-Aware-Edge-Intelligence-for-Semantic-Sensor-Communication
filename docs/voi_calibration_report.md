# VoI Calibration — Report (Task 14)

**Date**: September 2, 2026
**Scope**: calibration of default parameters only. The VoI formula (`raw = w_N·N + w_U·U + w_R·R + w_T·T − w_C·C`), the decision-policy tier structure, the five-factor architecture, and every module's own computation logic are unchanged. No new VoI implementation was created. No continual learning or FSO work was done.
**Method discipline**: every calibration decision below was derived from **training and validation data only**. The held-out test split was touched only once, at the very end, to produce the "after" numbers in §3 — using the exact same methodology (`src/evaluation/voi_behaviour_analysis.py`, unmodified since Task 13) as the "before" numbers in `docs/voi_integration_analysis.md`.

---

## 1. Changes Made

| File | Change | Protected? |
| :--- | :--- | :--- |
| `src/voi/scoring.py` | `VoIWeights` defaults: `novelty` 0.20→**0.30**, `uncertainty` 0.20→**0.05**, `task_relevance` 0.20→**0.35**, `temporal_importance` 0.20→**0.20** (unchanged), `resource_cost` 0.20→**0.10** | Yes — canonical, minimal, default-values-only |
| `src/temporal/config.py` | `DEFAULT_TEMPORAL_CHANGE_SCALE`: 0.5→**1.8** | No |
| `tests/test_voi.py` | Updated 3 assertions (`test_09`, `test_11`, `test_15`) that hardcoded the old default weight values | No |
| `src/evaluation/voi_behaviour_analysis.py` | Fixed a stale sensitivity-scenario label that hardcoded "w=0.20 each" (a documentation bug the calibration exposed, not a behavioural change) | No |
| `results/tables/*`, `results/figures/*` | Regenerated with calibrated defaults (same filenames as Task 13 — this task's "after" state) | No |
| `docs/voi_calibration_report.md` | This report | No |

**Explicitly not changed**: `src/voi/decision_policy.py` (thresholds stay 0.25 / 0.50 / 0.70 — see §2), `src/voi/normalization.py`, the VoI scoring formula itself, `src/communication/` (module and its config untouched), `src/uncertainty/` (estimator untouched), `src/novelty/`, `src/relevance/`, `src/cnn/`, `src/cwru_pipeline/`.

---

## 2. Rationale

Task 13 (`docs/voi_integration_analysis.md`) identified four structural issues. Each is addressed below strictly using train/val evidence.

### 2.1 Constant Communication Cost

Communication Cost is, and will remain, a constant (≈0.505) under the current nominal "one full raw CWRU window, full bandwidth" scenario — **every CWRU window is exactly the same size**, so there is nothing per-observation for the cost module to discriminate on without fabricating channel dynamics (explicitly out of scope: no FSO work in this task). Inventing per-observation cost variation without real channel telemetry would not be justified by genuine information-value behaviour — it would be exactly the kind of arbitrary tuning the task instructions prohibit.

The honest calibration response is therefore not to "fix" `src/communication/` (it already correctly computes cost from its inputs) but to stop letting a **non-discriminating constant** consume as much weight (0.20) as a genuinely discriminating factor. `resource_cost`'s weight was reduced to 0.10 — cost remains a real, included physical constraint (it still uniformly penalizes every transmission), it just no longer eats into the achievable ceiling as heavily as a factor that actually varies with the observation.

### 2.2 Ineffective Uncertainty Signal

Mean predictive entropy is ≤0.0009 on every split (train, val, and — confirmed only at the end — test), because this CNN is genuinely, correctly near-100%-confident (Task 6). This is not a bug to patch by inventing numbers; entropy-based uncertainty simply carries almost no information for *this* model. Replacing it with a better estimator (MC Dropout, deep ensembles, conformal prediction) would mean retraining/redesigning the uncertainty module, which is a separate, larger undertaking than a calibration task and was not attempted here.

The calibration response: reduce `uncertainty`'s weight from 0.20 to 0.05 — small enough that a near-zero signal no longer wastes a fifth of the formula's weight budget, but **not zero**, so a future improved uncertainty estimator can regain influence without requiring another weight change.

### 2.3 Temporal-Importance Behaviour

Task 13 found Temporal Importance saturating at ~1.0 for the large majority of fault windows, collapsing all fault severities into one indistinguishable bucket, because `DEFAULT_TEMPORAL_CHANGE_SCALE = 0.5` assumed a raw accelerometer "g" scale that doesn't match this pipeline's actual signal. Computing the raw (pre-normalization) mean-absolute-difference across **all training-split windows, label-free** (`data/processed/cwru`, `X_train`, per-recording sequences via `file_id`/`window_index`, same method `src/temporal/temporal.py` already uses) gives:

| Percentile | 50th | 75th | 90th | 95th | 99th |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Raw mean-abs-diff (train) | 1.214 | 1.595 | 1.775 | **1.806** | 1.829 |

The 95th percentile (≈1.8) was chosen as the new `DEFAULT_TEMPORAL_CHANGE_SCALE` — a scale derived purely from the observed training-signal statistics (no fault labels used in choosing it), replacing an assumed physical constant that never matched this dataset. This does not change what Temporal Importance conceptually measures (still window-to-window signal volatility, not condition drift over time — that limitation, noted in Task 13, remains and is listed in §5) — it only stops the *existing* formula from saturating almost everything to the same value.

### 2.4 Unreachable TRANSMIT Region

This was the compound effect of 2.1–2.3: a fixed cost penalty and a dead uncertainty term ate 0.40 of weight for near-zero return, while Temporal Importance's saturation meant even the most severe faults scored the same as moderate ones. Fixing the three root causes was checked, on train+val only, against the **existing, unmodified** decision thresholds (0.25 / 0.50 / 0.70) before touching anything in `src/voi/decision_policy.py`:

- Train+val raw-score ceiling rose from ≈0.60 (Task 13 analytical bound) to a train max of 0.7977 — genuinely above 0.70.
- Crucially, **the existing 0.70 threshold already produced a sensible three-way separation** once the inputs were fixed — Inner Race Fault (the class carrying the highest configured Task Relevance weight, 1.00, described as "critical" in `src/relevance/config.py`) separated cleanly above 0.70 on train+val, while Ball/Outer Race Fault (relevance 0.90, "significant but non-critical") stayed in the 0.50–0.70 SUMMARY band.

Because the existing thresholds already worked once the underlying factors were fixed, **`src/voi/decision_policy.py` was left untouched** — the smallest possible footprint for this fix. Thresholds were not lowered or otherwise tuned to hit any particular TRANSMIT percentage.

---

## 3. Before / After Metrics (test split, final evaluation)

### 3.1 Decision distribution

| Class | Before: DISCARD / BUFFER / SUMMARY / TRANSMIT | After: DISCARD / BUFFER / SUMMARY / TRANSMIT |
| :--- | :--- | :--- |
| Normal | 100% / 0% / 0% / 0% | 100% / 0% / 0% / 0% |
| Inner Race Fault | 0% / **100%** / 0% / 0% | 0% / 0% / 50.9% / **49.1%** |
| Ball Fault | 1.7% / 98.3% / 0% / 0% | 0% / 1.7% / **98.3%** / 0% |
| Outer Race Fault | 1.7% / 98.3% / 0% / 0% | 0% / 1.7% / **98.3%** / 0% |
| **Overall test** | 15.27% / 84.73% / **0%** / **0%** | 14.29% / 0.99% / **70.69%** / **14.04%** |

TRANSMIT went from 0 observations (0.0%) to 57 observations (14.04% of the test set) — concentrated entirely in Inner Race Fault, the class the (unmodified) relevance module already flags as most critical. SUMMARY, previously empty, now correctly captures the "significant but not critical" fault classes. Normal is unaffected (still 100% DISCARD) — the calibration did not change how the system treats routine data.

### 3.2 VoI score distribution

| Split | Before: mean / max | After: mean / max |
| :--- | :---: | :---: |
| Train | 0.398 / 0.4990 | 0.583 / 0.7977 |
| Validation | 0.370 / 0.4946 | 0.530 / 0.7929 |
| Test | 0.369 / 0.4835 | 0.543 / 0.7670 |

### 3.3 Normal vs. fault behaviour (test split)

| Class | Before: mean voi_score | After: mean voi_score |
| :--- | :---: | :---: |
| Normal | ≈0.000 | ≈0.0002 |
| Inner Race Fault | 0.472 | **0.704** |
| Ball Fault | 0.390 | 0.583 |
| Outer Race Fault | 0.428 | 0.612 |

Normal stayed essentially at zero in both configurations (correctly discarded); every fault class's score rose, with Inner Race Fault rising the most — the intended, class-aligned separation.

### 3.4 Factor contributions (test split, share of positive contribution to `raw_voi_score`)

| Factor | Before | After |
| :--- | :---: | :---: |
| Novelty | 27.2% | 32.0% |
| Uncertainty | 0.04% | 0.01% |
| Task Relevance | 35.0% | **48.1%** |
| Temporal Importance | 37.7% | 19.9% |
| Communication Cost (constant) | −0.101 | −0.0505 |

Task Relevance — the most directly task-aligned, intentionally-designed signal (r=0.984 with `voi_score`, unchanged from Task 13) — is now the clear leading factor, rather than Temporal Importance's volatility artifact dominating by accident. Uncertainty's contribution remains negligible in absolute terms (as expected — its *estimator* wasn't changed, only its weight), but it no longer wastes a fifth of the formula's budget.

### 3.5 Sensitivity check (read-only, test split, `results/tables/voi_sensitivity_analysis.csv`)

The same six-scenario exploration from Task 13 was re-run against the calibrated defaults: lowering the TRANSMIT threshold further (to 0.60 or 0.50) now increases TRANSMIT even more (42.6% / 84.7%) — confirming the ceiling problem is genuinely resolved, not just marginally patched. The alternative weight scenarios (further reducing uncertainty, cost, or boosting relevance) all *reduce* TRANSMIT relative to the new calibrated default, which is expected: those scenarios were Task 13's exploratory alternatives, not further refinements of this calibration.

---

## 4. Test Result

Full suite: **139/139 passing** after this task's changes (same count as before — no tests added, none removed).

Three pre-existing tests hardcoded the old default weight values and were updated to match the deliberately-changed defaults (not relaxed or weakened — their pass/fail logic and intent are unchanged, only the literal expected numbers that follow directly from the new weights):
- `tests/test_voi.py::test_09_boundary_values_0_and_1` — expected raw score at all-factors-saturated-except-cost updated from 0.8 (4×0.20) to 0.90 (0.30+0.05+0.35+0.20).
- `tests/test_voi.py::test_11_clipping_behavior` — expected raw score at cost=1.0 updated from −0.20 to −0.10 (new `resource_cost` weight).
- `tests/test_voi.py::test_15_decision_reachability_analysis` — this test's entire purpose was to document the V0.1 reachability problem (`assertLess(max, 0.70)`, `assertEqual(n_capable, 0)`); it now documents the V0.2 fix (`assertGreater(max, 0.70)`, `assertGreater(n_capable, 0)`) on the same seeded synthetic dataset used since Phase 1.

No genuine gap requiring a *new* test was found — the calibration changed default parameter values in already-tested code paths, not logic.

---

## 5. Remaining Limitations

1. **BUFFER is nearly empty for CWRU** (4/406 test observations, all borderline Ball/Outer cases). This is a consequence of the CNN's near-deterministic, near-100%-confident predictions, not something this calibration can fix — BUFFER will only be meaningfully populated by genuine prediction ambiguity, which this dataset/model combination rarely produces.
2. **Uncertainty's weight reduction is a mitigation, not a fix.** The entropy estimator itself is untouched and will stay near-zero for any similarly confident model. A genuinely more informative uncertainty signal (MC Dropout, ensembles, conformal prediction) remains future work, as already recommended in Task 13.
3. **Communication Cost is still literally constant.** Its weight was reduced, but it does not yet vary per observation. A real fix requires either actual channel-condition telemetry (FSO, explicitly out of scope here) or a transmission scheme where payload size genuinely depends on what's being sent.
4. **The temporal-importance recalibration (scale = 1.8) is CWRU-specific**, fit to this dataset's own signal-volatility statistics. It will need to be re-derived (via the same train-only, label-free percentile method) for any new dataset (IMS, Paderborn, XJTU-SY) with different signal characteristics — it should not be assumed to transfer.
5. **Temporal Importance's conceptual mismatch persists**: it still measures window-to-window signal volatility, not genuine operating-condition drift over time, since CWRU is a static-condition dataset. Its role should be revisited once a degradation dataset is introduced.
6. **The new weights (0.30 / 0.05 / 0.35 / 0.20 / 0.10) were chosen by inspection of train/val distributions and dominance shares, not by a formal optimization procedure** against a defined objective. They are a justified, evidence-based calibration, not a mathematically optimal one — consistent with this project's stated "provisional parameters, candidate for future calibration" framing for VoI weights and thresholds.
7. **This calibration is specific to the current CNN and CWRU.** Any future change to the CNN (retraining, a new architecture) or introduction of a new dataset should trigger a re-run of this same train/val-only calibration methodology, not an assumption that these exact weights remain correct.
