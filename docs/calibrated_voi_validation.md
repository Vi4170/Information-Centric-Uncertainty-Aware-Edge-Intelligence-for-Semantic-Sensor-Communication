# Calibrated VoI Validation — Report (Task 15)

**Analysis module**: `src/evaluation/calibration_validation.py` (new, read-only — modifies no weights, thresholds, scoring logic, or estimators)
**Date**: September 2, 2026
**Scope**: validation only. Confirms what Task 14 changed actually helped, and that it did so honestly (no test-data leakage, no formula changes, reproducible).

---

## Method

"Before" (Task 13) and "after" (Task 14) are compared on the exact same held-out test-split observations. Novelty, Uncertainty, Task Relevance, and Communication Cost are identical in both configurations — their own modules were never touched between Task 13 and Task 14 — so they are read directly from the already-committed `results/tables/voi_integration_per_observation.csv`. Only Temporal Importance differs (Task 14 changed its default scale), so it alone is recomputed here with the historical scale (0.5) via the unmodified `compute_temporal_importance()`. Both factor sets are then scored by two **explicit, local** `VoIEngine` instances — the historical Task 13 weights (0.20 each) and the current canonical defaults — never by mutating any module's defaults. This means the "before" numbers reported here are freshly recomputed from real data, not copied from `docs/voi_integration_analysis.md`; they match it exactly (cross-checked below), which is itself a consistency check.

---

## Verification Checklist

| Check | Result |
| :--- | :--- |
| Test data was not used to select Task 14 parameters | **Confirmed.** Task 14's weight choice and temporal-scale recalibration (`docs/voi_calibration_report.md`) were computed exclusively from `X_train`/`meta_train` and `X_val`/`meta_val` concatenations — verified by re-reading that task's own calibration computations, none of which referenced `X_test`. This task's "before" reconstruction only touches `X_test` for evaluation, never for parameter selection. |
| 0.25 / 0.50 / 0.70 decision thresholds unchanged | **Confirmed** by reading `src/voi/decision_policy.py` directly: `PolicyThresholds` defaults are still `discard_max=0.25, buffer_max=0.50, summary_max=0.70`. Not modified in Task 14 or this task. |
| Canonical VoI formula unchanged | **Confirmed** by reading `src/voi/scoring.py` directly: `calculate_voi_score` still computes `raw_score = w_N·N + w_U·U + w_R·R + w_T·T − w_C·C`, byte-identical to before Task 14 — only the `VoIWeights` dataclass's default field *values* changed. |
| Reproducibility of calibrated results | **Confirmed.** Re-running `src/evaluation/voi_behaviour_analysis.py` end-to-end and comparing every numeric factor/score column against the already-committed `voi_integration_per_observation.csv`: **max absolute difference = 0.00e+00**, and every decision label matched exactly. The pipeline is fully deterministic (fixed trained model, fixed novelty reference fit on train, no randomness in any factor computation). |

---

## 1. Whether Calibration Improved Behaviour

**Yes.** The calibrated system reaches TRANSMIT for genuinely high-value observations while preserving DISCARD/BUFFER-appropriate behaviour for low-value ones — the exact goal Task 14 was scoped to achieve — without any change to the formula, thresholds, or any factor estimator's own logic.

## 2. Evidence Supporting the Conclusion

### 2.1 Normal discard rate
**Unchanged: 100% → 100%.** All 58 test-set Normal observations are DISCARDed in both configurations — calibration did not disturb the one behaviour that was already correct.

### 2.2 Fault transmit/summary/discard rates (test split)

| Class | Before: DISCARD / BUFFER / SUMMARY / TRANSMIT | After: DISCARD / BUFFER / SUMMARY / TRANSMIT |
| :--- | :--- | :--- |
| Inner Race Fault | 0% / 100% / 0% / 0% | 0% / 0% / 50.9% / **49.1%** |
| Ball Fault | 1.7% / 98.3% / 0% / 0% | 0% / 1.7% / **98.3%** / 0% |
| Outer Race Fault | 1.7% / 98.3% / 0% / 0% | 0% / 1.7% / **98.3%** / 0% |

No fault observation is ever DISCARDed after calibration (0% for all three fault classes, down from 1.7% for Ball/Outer). Inner Race Fault — the class `src/relevance` configures as most critical (relevance weight 1.00) — is the only class reaching TRANSMIT, at essentially half its test observations.

### 2.3 VoI distributions by class (mean / min / max)

| Class | Before | After |
| :--- | :---: | :---: |
| Normal | 0.000 / 0.000 / 0.000 | 0.0002 / 0.000 / 0.0022 |
| Inner Race Fault | 0.472 / 0.270 / 0.483 | **0.704** / 0.557 / 0.767 |
| Ball Fault | 0.390 / 0.199 / 0.407 | 0.583 / 0.444 / 0.600 |
| Outer Race Fault | 0.428 / 0.227 / 0.437 | 0.612 / 0.487 / 0.656 |

Full table: `results/tables/calibration_validation_voi_score_comparison.csv`. Overall test mean rose from 0.369 to 0.543; overall max from 0.4835 to 0.7670. Every fault class's score range shifted upward; Normal stayed pinned near zero.

### 2.4 Factor contributions (test split, share of positive contribution)

| Factor | Before | After |
| :--- | :---: | :---: |
| Novelty | 27.2% | 32.0% |
| Uncertainty | 0.04% | 0.01% |
| Task Relevance | 35.0% | **48.1%** |
| Temporal Importance | 37.7% | 19.9% |
| Communication Cost (constant) | −0.101 | −0.0505 |

Task Relevance — the most directly task-aligned signal — is now the clear leading factor, rather than Temporal Importance's saturation-inflated share dominating by accident (Task 13's finding). Full table: `results/tables/calibration_validation_factor_dominance_comparison.csv`.

### 2.5 Maximum / minimum VoI
Before: min 0.000, max 0.4835 (below even the SUMMARY threshold). After: min 0.000, max 0.7670 — genuinely above the TRANSMIT threshold (0.70), with margin.

### 2.6 Transmission reduction
Against the naive "transmit everything" baseline (100% full transmission): after calibration, only **14.04%** of test observations are fully transmitted (TRANSMIT); the remaining **85.96%** are DISCARDed, BUFFERed, or sent as a compact SUMMARY rather than a full raw window. Before calibration this distinction was moot — nothing ever reached full TRANSMIT, so "transmission reduction" and "information loss" were the same thing (everything was either discarded or buffered, with no differentiated compact path). Full table: `results/tables/calibration_validation_transmission_reduction.csv`.

### 2.7 Are high-value observations preferentially transmitted?
**Yes, with a clean margin, not a borderline split.** Within Inner Race Fault alone: SUMMARY observations range `voi_score` ∈ [0.5567, 0.6615] (mean 0.650); TRANSMIT observations range ∈ [0.7457, 0.7670] (mean 0.760) — a real gap (0.66 to 0.75) with no observations in between, not a threshold cutting through a dense cluster. Across the full test set, the correlation between `voi_score` and decision-tier ordinal (DISCARD=0…TRANSMIT=3) is **0.971**, and between the raw importance proxy (N+R+T, unweighted) and decision tier is **0.979** — decisions track underlying importance closely and monotonically, as the deterministic threshold policy guarantees by construction, but the *data* additionally shows this isn't a knife-edge separation.

### 2.8 Is any class systematically discarded despite being important?
**No.** After calibration, `after_DISCARD_pct` is exactly 0.0% for all three fault classes (Inner Race, Ball, Outer Race) — every fault observation receives at least a SUMMARY-level response; only Normal is ever discarded.

---

## 3. Remaining Weaknesses

(Carried forward from `docs/voi_calibration_report.md` §5 — none were addressed in this validation-only task, as scoped.)

1. **BUFFER is nearly unused for CWRU** (4/406 test observations) — a consequence of this CNN's near-deterministic confidence, not something calibration can fix.
2. **Uncertainty's estimator is still unfixed** — its weight was reduced, but predictive entropy itself remains near-zero and uninformative for this model.
3. **Communication Cost is still a literal constant** (0.505 in both configurations) — its weight was reduced, but it does not vary per observation; a genuine fix needs real channel telemetry (FSO, out of scope).
4. **The temporal scale (1.8) is CWRU-specific** and will need re-deriving (same train-only, label-free percentile method) for any new dataset.
5. **Temporal Importance still measures signal volatility, not condition drift over time** — conceptually mismatched with its intended role until a degradation dataset (IMS/XJTU-SY) is introduced.
6. **The calibrated weights were chosen by evidence-based inspection, not formal optimization** against a defined objective function.
7. **All of the above findings are specific to the current CNN and CWRU** — they do not automatically transfer to a new model or dataset.

## 4. Readiness for Continual-Learning Experiments

**Conditionally ready, with caveats.** The core VoI decision pipeline is now validated to behave sensibly (transmits what's genuinely important, discards what genuinely isn't, is reproducible, and was calibrated without test-set leakage) — a reasonable, honest baseline to build continual learning on top of. However, continual learning will introduce operating conditions this validation never tested:

- Every number in this report is specific to the **current, static** CNN and the **current, static** novelty reference (fit once on `X_train`). Continual learning's entire premise is that the model/reference will change over time — this validation says nothing about whether the calibrated weights, or the temporal scale, remain appropriate after such an update.
- Temporal Importance's known conceptual mismatch (volatility vs. drift, §3.5) makes it a particularly important factor to re-examine once genuine condition drift (the object continual learning is meant to detect) becomes available — its current weight and formula were calibrated against a dataset that structurally cannot exercise that behaviour.
- The near-total absence of BUFFER usage and Uncertainty's continued weakness mean the system currently has little signal to fall back on for *novel-but-not-yet-understood* conditions — precisely the regime continual learning is supposed to handle gracefully (per the project's master plan, Section 20–21: "avoid permanently treating every new operating condition as an anomaly").

Recommendation: proceed to continual-learning architecture *design* (as the project roadmap's Task 15/16 originally envisioned before this validation was inserted), but treat the current calibration as a re-calibration candidate once real condition drift is introduced, not as a fixed, final configuration.
