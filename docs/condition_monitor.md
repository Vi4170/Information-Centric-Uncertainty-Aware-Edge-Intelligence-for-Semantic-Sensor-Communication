# Condition Monitor — Phase 2 (Task 18)

**Module**: `src/continual/condition_monitor.py`
**Status**: read-only monitoring only. Not connected to `AdaptationBuffer`, `src/voi/`, `src/integration/`, or any adaptation logic. It computes nothing that feeds back into the CNN, novelty, uncertainty, relevance, temporal, or communication-cost modules — it only reads plain floats/ints it is handed.

## Purpose

Implements Phase 2 of `docs/continual_learning_design.md` §3.1: distinguish a **sustained** distribution/operating-condition shift from a single anomalous observation, using only outputs the existing pipeline already produces. It answers "has something changed for a while?", not "is this one reading a fault?" — fault detection is already novelty's/the CNN's job (Task 5/6) and is untouched here.

## Inputs

Per observation: a novelty score (`float`, `[0, 1]` — the same contract `src/novelty`'s `DistanceNoveltyDetector.score()` already returns) and a predicted class id (`int`, `[0, num_classes)` — the same contract `src/cnn`'s `predict_classes()` already returns). The monitor never calls those modules itself; a caller computes them and passes the results in via `observe(novelty, predicted_class)`.

At construction, it also needs **reference statistics** representing the known/in-control baseline: `reference_novelty_mean`/`reference_novelty_std` (a novelty distribution to compare against) and `reference_class_distribution` (a class-proportion baseline to compare against). This module does not compute these either — see "Choosing reference statistics" below for why that choice matters.

## Rolling-Window Behaviour

Observations are kept in a bounded, deterministic rolling window (`window_size`, default 30; oldest observation drops off as a new one arrives). Until the window has `window_size` observations, the monitor reports `INSUFFICIENT_HISTORY` and computes no signals — there is deliberately no way to get a shift verdict from too little evidence.

## Novelty Shift Detection

A per-observation control-chart limit is fixed at construction: `threshold = reference_novelty_mean + k · reference_novelty_std` (k defaults to 2, a standard "2-sigma" convention). A shift is flagged only if the **fraction of the current window** exceeding that limit is at least `novelty_fraction_threshold` (default **1.0 — the entire window**, per the design doc's explicit wording: "for the ENTIRE window (not a single spike)"). This is deliberately stricter than comparing the window's *mean* to the threshold: a single extreme value can pull a mean above the line while 4 out of 5 observations remain completely normal — the monitor's own tests (`test_03_isolated_novelty_spike_no_sustained_shift`) construct exactly this case and confirm it is correctly **not** flagged, even though the naive window mean would exceed the threshold.

## Predicted-Class Distribution Shift Detection

The current window's predicted-class histogram is compared to `reference_class_distribution` via the **Population Stability Index** (PSI) — a standard, widely used drift-monitoring statistic (`PSI > 0.2` conventionally means "significant shift"; this default was not tuned against CWRU). Both distributions are epsilon-smoothed so a class with zero occurrences in a small window doesn't produce `log(0)`.

## Sustained-Evidence Requirement

Both signals are combined into one `ConditionShiftStatus`:

| Novelty shift | Class shift | Status |
| :---: | :---: | :--- |
| — (< window_size observations) | — | `INSUFFICIENT_HISTORY` |
| No | No | `STABLE` |
| Yes | No | `NOVELTY_SHIFT_ONLY` |
| No | Yes | `CLASS_DISTRIBUTION_SHIFT_ONLY` |
| Yes | Yes | `CANDIDATE_CONDITION_SHIFT` |

Per `docs/continual_learning_design.md`, **only `CANDIDATE_CONDITION_SHIFT` represents both required signals agreeing** — high novelty alone, one fault, or one unusual window can never reach it. `NOVELTY_SHIFT_ONLY` and `CLASS_DISTRIBUTION_SHIFT_ONLY` are reported so a human/future controller can see a single signal building without treating it as confirmed evidence.

Every call to `observe()` returns a full `ConditionMonitorResult`: whether there's enough history, the window's novelty mean/threshold/fraction-above, the PSI value, both boolean shift flags, the combined status, and the raw window class counts — enough for a future regression/replay check to reason about without re-deriving anything.

## Choosing Reference Statistics (a real finding, not a design choice made in the abstract)

While validating against real CWRU pipeline outputs (below), computing `reference_novelty_mean`/`std` from the **full training set** (which is ~90% already-labelled fault windows, only ~7.7% Normal) produced a control-chart threshold of **1.167** — above the maximum possible novelty score of 1.0, making a novelty shift structurally unreachable. This makes sense in hindsight: the full training set already contains known, expected fault novelty by design, so it isn't a genuine "in-control baseline" to detect deviation from. Using the **Normal-only** subset of training novelty scores instead (mean ≈ 0.0053, std ≈ 0.0043) gives a small, meaningful threshold (≈0.0139) representing "how novel does genuinely normal data look" — the correct baseline for this kind of control-chart monitoring. This is documented here because it is an easy mistake to repeat when this monitor is wired up for a real dataset later.

## Limitations on CWRU (observed, not assumed)

**CWRU does not demonstrate genuine continual learning or operating-condition drift, and this validation does not claim it does.** Concretely, running the monitor (with the Normal-only reference above) over the real CWRU test split, in its natural per-recording order, showed:

- The first ~30 observations (the initial Normal-labelled recording file) report `INSUFFICIENT_HISTORY` then largely `CLASS_DISTRIBUTION_SHIFT_ONLY` — because CWRU's training composition is itself heavily fault-skewed (7.7% Normal / 30.8% each fault class), so **any homogeneous window, including an all-Normal one, diverges from that skewed baseline by PSI** — this is a property of the class-imbalance in the reference, not evidence of drift.
- Once the window fills with a single fault-type recording file, the monitor does report `CANDIDATE_CONDITION_SHIFT` for most of the remainder of the stream. This is **not** the interesting kind of sustained-shift detection the design doc is aimed at: CWRU's test stream is a concatenation of separately recorded, internally homogeneous, single-condition files placed back to back (58 Normal windows, then 58 Inner-Race-Fault windows, then 58 Ball-Fault windows, ...). Once a rolling window of 30 sits entirely inside one such block, of course it looks stably different from the training reference — that is just "this is fault data," which per-observation novelty already tells us (Task 5). There is no continuous deployment timeline in CWRU for a *genuine* new operating condition to emerge gradually within, so this result should be read as "the monitor is mechanically correct and runs cleanly on real pipeline outputs," not as "the monitor discovered new operating conditions in CWRU."
- The monitor was never observed to reach `CANDIDATE_CONDITION_SHIFT` and then fall back to `STABLE` mid-recording (the behaviour that would indicate a real, bounded "shift episode" rather than a static block-boundary artifact) — again consistent with CWRU having no real temporal drift to exercise that pattern.

A meaningful test of the *sustained vs. transient* distinction this monitor is built for requires a dataset with an actual mixed/evolving deployment timeline — e.g. IMS or XJTU-SY (both flagged in `docs/continual_learning_design.md` §5) — not CWRU.

## How Phase 2 Connects to the Future Adaptation Controller

This monitor produces a `ConditionShiftStatus` and does **nothing else** with it. The design doc's Adaptation Controller (Phase 3+, not implemented here) is what would eventually read a sustained `CANDIDATE_CONDITION_SHIFT` and decide whether to add observations to the `AdaptationBuffer` (Task 17) and whether it's safe to extend the novelty reference — gated by the Safety Gate and Regression Gate the design doc describes. None of that gating, and no automatic buffer insertion, exists yet; this task only produces the signal a future controller would consume.
