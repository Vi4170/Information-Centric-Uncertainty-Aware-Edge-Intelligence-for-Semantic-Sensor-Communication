# VoI Behaviour Analysis — Report (Task 13)

**Analysis module**: `src/evaluation/voi_behaviour_analysis.py` (new, read-only — computes no VoI mathematics of its own)
**Date**: September 2, 2026
**Scope**: analysis only. `src/voi/` (weights, thresholds, scoring formula) was not modified; nothing about the live system's configuration changed as a result of this task.

---

## 1. Method

Every CWRU window in train/val/test is run through the existing, unmodified pipeline:

`CNN → {probabilities, 64-D embedding} → Novelty (src/novelty) + Uncertainty (src/uncertainty) + Task Relevance (src/relevance) + Temporal Importance (src/temporal) + Communication Cost (src/communication) → canonical VoIEngine (src/voi, default weights/thresholds) → decision`

Two assumptions were needed to compute factors the existing modules don't derive automatically from the CWRU dataset:

- **Communication Cost**: no channel/attenuation model exists yet (planned FSO future work), so every observation is scored under one fixed nominal scenario — a full raw window (`MAX_PAYLOAD_SIZE` = 16,384 bytes, i.e. `src/communication/config.py`'s own definition of "one CWRU window") transmitted at full `REFERENCE_BANDWIDTH`. This makes cost **constant by construction** (C ≈ 0.5049 for every observation), not by measurement. See §4 and §7.
- **Temporal Importance**: computed on the normalized raw window signal (matching the physical scale `DEFAULT_TEMPORAL_CHANGE_SCALE = 0.5` was calibrated against), sequenced per source recording (CWRU metadata's `file_id` + `window_index`) so a score is never computed across a recording boundary — the first window of every recording gets T=0 by the existing module's own convention.

The novelty reference centroid was fit strictly on training embeddings (as in Task 5); nothing here refits or recalibrates any protected module.

---

## 2. Key Findings

1. **Task Relevance and Novelty behave almost as intended, but as near-deterministic functions of predicted class**, not as continuous, nuanced signals — because the CNN is essentially always 100% confident (see finding 3), relevance collapses to its fixed per-class constant and novelty tracks the class-conditional embedding distance tightly (test-set std within each class ≤ 0.023, vs. ≥ 0.19 spread if a class is ambiguous). This is expected given the CNN's near-perfect accuracy, not a defect in either module.
2. **Temporal Importance is unexpectedly the single largest contributor to the VoI score** (see §3), but it is not measuring what the project's architecture intends it to measure for this dataset. `src/temporal`'s formula (mean absolute difference between consecutive windows) picks up the raw *signal volatility* of impulsive bearing-fault vibration — fault classes have mean T ≈ 0.98 (saturated) vs. Normal's T ≈ 0.24 — which happens to correlate with fault presence, but CWRU is a static-condition dataset (each recording is one fixed fault/load throughout), so there is no genuine "operating condition changed over time" signal to detect here. Temporal Importance is currently acting as a second, cruder novelty detector rather than a temporal-drift detector; that role only becomes meaningful once a degradation dataset (IMS/XJTU-SY, per the project roadmap) is introduced.
3. **Uncertainty is functionally dead in the current VoI formula.** Mean predictive entropy is ≤ 0.0009 across every split (consistent with the 100% test accuracy already reported in Task 6), so its weighted contribution to `raw_voi_score` is 0.0001–0.0002 — a 0.02–0.04% share of the total positive contribution (§3). This confirms, with real data, the concern the original VoI validation (V0.1, synthetic data) had already flagged.
4. **TRANSMIT is never reached — not once, on any split, for any class**, under the current default weights and thresholds (§3–4). The system currently only ever chooses between DISCARD and BUFFER.

---

## 3. Transmission / Decision Behaviour

| Split / Class | n | DISCARD | BUFFER | SUMMARY | TRANSMIT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Train | 1,508 | 8.75% | 91.25% | 0.0% | 0.0% |
| Validation | 406 | 15.27% | 84.73% | 0.0% | 0.0% |
| Test (overall) | 406 | 15.27% | 84.73% | 0.0% | 0.0% |
| Test: Normal | 58 | **100.0%** | 0.0% | 0.0% | 0.0% |
| Test: Inner Race Fault | 116 | 0.0% | **100.0%** | 0.0% | 0.0% |
| Test: Ball Fault | 116 | 1.72% | 98.28% | 0.0% | 0.0% |
| Test: Outer Race Fault | 116 | 1.72% | 98.28% | 0.0% | 0.0% |

The system does cleanly separate Normal (always DISCARD) from every fault class (almost always BUFFER) — a sensible qualitative split. But it never distinguishes *degree* of importance within or above that: no observation, including the most novel/relevant fault windows, ever reaches SUMMARY or TRANSMIT. Max observed `voi_score` across the entire dataset is **0.4995** (train) / **0.4835** (test) — below even the SUMMARY boundary (0.50) in the test set, let alone the TRANSMIT boundary (0.70).

Full per-split/per-class table: [`results/tables/voi_decision_distribution.csv`](../results/tables/voi_decision_distribution.csv). Visualization: [`results/figures/voi_decision_distribution.png`](../results/figures/voi_decision_distribution.png).

### Is the TRANSMIT threshold reachable?

Analytically, with equal weights (0.20 each) and `raw_voi_score = 0.2(N+U+R+T) − 0.2C`: the positive terms can sum to at most 0.8 (all four at 1.0), and C is fixed at ≈0.505 under the nominal cost scenario, giving a **hard ceiling of ≈0.60** — already below the 0.70 TRANSMIT threshold **regardless of how novel, relevant, or temporally significant an observation is**, as long as U stays near zero and C stays at its current fixed value. Empirically, the true ceiling is even lower (0.4995) because N, R, and T never simultaneously saturate at 1.0 for the same observation. **TRANSMIT is currently unreachable by construction, not because no CWRU observation is "important enough."**

---

## 4. Dominant Factors

Mean weighted contribution to `raw_voi_score` and correlation with the final `voi_score` (test set):

| Factor | Mean weighted contribution | Share of positive contribution | Correlation with `voi_score` |
| :--- | :---: | :---: | :---: |
| Temporal Importance | +0.1754 | **37.7%** | 0.941 |
| Task Relevance | +0.1629 | **35.0%** | 0.983 |
| Novelty | +0.1266 | 27.2% | 0.960 |
| Uncertainty | +0.0002 | **0.04%** | 0.034 (noisy, not meaningful) |
| Communication Cost | −0.1010 (constant) | n/a | 0.000 (constant, no variance to correlate) |

Temporal Importance and Task Relevance together account for ~73% of the positive signal driving the VoI score — meaning **two of the five intended independent factors currently do almost all the work**, and one (Uncertainty) contributes essentially nothing. This is not a balanced five-factor system in practice, even though all five are correctly wired in.

Full table: [`results/tables/voi_factor_dominance.csv`](../results/tables/voi_factor_dominance.csv). Visualization: [`results/figures/voi_factor_contribution.png`](../results/figures/voi_factor_contribution.png).

---

## 5. Threshold / Weight Sensitivity

A read-only exploration (`build_sensitivity_table`, `results/tables/voi_sensitivity_analysis.csv`) ran the test set through six alternative `VoIWeights`/`PolicyThresholds` configurations — **as separate local engine instances only; nothing in `src/voi/` or its defaults was changed**:

| Scenario | Mean VoI | DISCARD | BUFFER | SUMMARY | TRANSMIT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Default (w=0.20 each, thresholds 0.25/0.50/0.70) | 0.369 | 15.27% | 84.73% | 0.0% | 0.0% |
| Lower TRANSMIT threshold to 0.50 | 0.369 | 15.27% | 0.49% | 84.24% | 0.0% |
| Lower TRANSMIT threshold to 0.60 | 0.369 | 15.27% | 25.86% | 58.87% | 0.0% |
| Zero-out uncertainty weight, redistribute to novelty+relevance | 0.512 | 14.29% | 1.48% | 84.24% | 0.0% |
| Zero-out communication cost weight | 0.465 | 14.29% | 25.62% | 60.10% | 0.0% |
| Double task_relevance weight (renormalised) | 0.477 | 14.29% | 1.48% | 84.24% | 0.0% |

Observations:
- Even after **lowering the TRANSMIT threshold to 0.60**, or **removing the dead uncertainty weight entirely and reallocating it**, or **zeroing out the constant cost penalty**, TRANSMIT is still never reached (0.0% in every scenario tested). This confirms the ceiling problem in §3 is not fixable by threshold tuning alone — the positive factors themselves (or their weights) need headroom, or the fixed cost floor needs to shrink.
- The system is highly sensitive to whether uncertainty's weight is reallocated: mean VoI jumps from 0.369 → 0.512 (the single largest change of any scenario tested), because uncertainty was contributing almost nothing to begin with — redistributing its 0.20 weight to novelty/relevance directly raises scores that already had signal.
- Lowering the TRANSMIT threshold to 0.50 mostly converts BUFFER → SUMMARY, not BUFFER → TRANSMIT, because so few observations sit above 0.50 to begin with.

---

## 6. Generated Artifacts

**New tables** (`results/tables/`):
- `relevance_scores_summary.csv` — fills a gap: Task 7 (`src/relevance/`) had no existing distribution report, unlike novelty/uncertainty.
- `voi_integration_per_observation.csv` — full per-observation table (test split, 406 rows): `observation_id, class_id, class_name, novelty, uncertainty, task_relevance, temporal_importance, resource_cost, raw_voi_score, voi_score, decision`.
- `voi_integration_summary.csv` — per-split and per-test-class distribution statistics for all 5 factors + `raw_voi_score` + `voi_score`.
- `voi_decision_distribution.csv` — decision counts/percentages per split and per test class.
- `voi_factor_dominance.csv` — mean weighted contribution, share of positive contribution, and correlation with `voi_score`, per factor per split.
- `voi_sensitivity_analysis.csv` — the six-scenario exploration in §5.

**New figures** (`results/figures/`):
- `relevance_score_distribution.png`
- `voi_score_distribution.png` (train/val/test histograms with decision-threshold boundaries marked)
- `voi_score_by_class.png` (test-set boxplot by bearing health class)
- `voi_decision_distribution.png` (stacked bar of decision category by class)
- `voi_factor_contribution.png` (mean weighted contribution per factor)

**Reused, not regenerated** (already correct post class-label fix): `results/figures/novelty_score_distribution.png`, `results/figures/uncertainty_score_distribution.png`.

---

## 7. Test Result

Full suite: **139/139 passing**, unchanged from before this task. No test was added — this task's new code (`src/evaluation/voi_behaviour_analysis.py`) is a read-only orchestration/reporting script in the same category as the existing `run_novelty_pipeline` / `run_uncertainty_pipeline` functions, none of which are covered by the unit test suite either (verified during Task 11): their correctness is validated by inspecting the generated output, not by unit tests, consistent with the established convention in this repo. No genuine gap in the *unit-tested* modules (`src/voi/`, `src/novelty/`, `src/uncertainty/`, `src/relevance/`, `src/temporal/`, `src/communication/`) was discovered — every value this analysis consumed passed its own module's existing validation.

---

## 8. Recommended Changes for a Future Calibration Task (not implemented here)

1. **Communication Cost needs to stop being a constant.** Under the current nominal scenario, C is the same (≈0.505) for every observation, so it can never help distinguish important from unimportant observations — it only ever subtracts a fixed amount. Recommend either (a) making payload size reflect what's actually transmitted (e.g., a 64-D embedding = 256 bytes vs. a full 2048-sample raw window = 16,384 bytes, an ~64× difference already representable in the existing formula), or (b) deferring meaningful cost variation to the planned FSO channel model.
2. **Uncertainty needs either a better estimator or a smaller weight.** With this CNN, entropy-based uncertainty will stay near-zero regardless of application; per the original uncertainty report's own recommendation, MC Dropout / deep ensembles / conformal prediction may produce a more discriminating signal. Until then, its 0.20 weight is effectively wasted and could be reallocated.
3. **Temporal Importance's role should be reconsidered for this dataset.** It is currently the single largest contributor, but it's measuring signal volatility, not condition change over time — recommend explicitly documenting this distinction, and revisiting its formula/weight once a genuine degradation dataset (IMS/XJTU-SY) is introduced, per the project roadmap.
4. **The TRANSMIT threshold (or the positive-factor weights/ceiling) needs recalibration**, since it is currently structurally unreachable regardless of how important an observation is. Options include raising the ceiling (larger novelty/relevance weights, or allowing raw scores to exceed 1.0 before clipping in a redesigned formula) or lowering the threshold — but §5 shows threshold-only changes convert BUFFER→SUMMARY, not BUFFER→TRANSMIT, so weight/ceiling changes are likely necessary too.
5. **Re-run this analysis after any calibration change**, and additionally once continual learning or a second dataset is introduced, since all findings here are specific to CWRU's near-perfect CNN and static-condition recordings.

None of the above were implemented in this task, per scope.
