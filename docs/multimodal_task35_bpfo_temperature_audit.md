# Task 35 Corrective Audit — BPFO Temperature/Current Independent Validity

## Trigger

Task 35's original implementation excluded all 9 BPFO conditions (every
load x severity) from **every** fusion configuration that used
`motor_current` OR `temperature`, on the grounds that both modalities are
digitized into the same raw `.tdms` file, and that file's current channels
are known to be corrupt for BPFO. This audit was requested to verify,
independently and from the raw channels themselves, whether temperature was
genuinely also invalid for those files, or whether the exclusion of
temperature was an artifact of a single overly-broad validity check.

## Method

Every one of the 45 raw `temperature_current/*.tdms` files was read directly
with `nptdms`, bypassing `src/multimodal_pipeline` entirely. Each channel was
classified by its own `DAC~Channel~Type` property (`Temperature` or
`Current`) and inspected for: sample count, NaN count, min/max/mean/std.

## Findings

- **Exactly the 9 BPFO conditions** have any problem at all — confirmed by
  scanning all 45 files, not just the 9 already suspected.
- In every one of those 9 files: both **Temperature** channels have the
  full expected sample count (1,536,492, matching known-good files of the
  same 60s duration class), zero NaNs, and physically plausible, non-static
  values (e.g. 28-33 degC with real sample-to-sample variation, std
  0.08-0.33 degC).
- In every one of those same 9 files: only **1 of the 3 Current**
  phase channels has data; the other 2 (`Mod2/ai2`, `Mod2/ai3`) have exactly
  0 samples. Motor current requires all 3 U/V/W phases to be meaningful, so
  this modality is genuinely, unambiguously invalid for these 9 conditions.
- **Conclusion: temperature is valid for all 9 BPFO conditions; current is
  genuinely invalid for all 9 BPFO conditions.** These are independent
  facts about two different physical channel groups that happen to share
  one raw file — not a single fact about "the file."

## Root cause of the original (incorrect) exclusion

`src/multimodal_pipeline/preprocessing.py::_read_temperature_current_tdms`
validated all 5 channels (2 temperature + 3 current) for consistent,
non-zero length as a single blanket check, raising `CorruptRawFileError` for
the whole file if any one channel failed. Every caller that needed only one
modality's channels (`build_temperature_features_for_condition`,
`build_motor_current_windows_for_condition`, `fit_modality_normalization`,
`condition_window_availability`) inherited this coupling. Additionally,
`fusion.py::build_split_representation_table` explicitly gated the
temperature build behind `if current_valid`, on the documented (but
mistaken) assumption that a corrupt current channel implied an unusable
temperature reading in the same file.

## Fix

`_read_temperature_current_tdms` now accepts `require_temperature` /
`require_current` flags and validates each channel group independently
(`_validate_channel_group`). Each caller now requests validation only for
the modality it actually needs:

| Caller | require_temperature | require_current |
|---|---|---|
| `build_temperature_features_for_condition` | True | False |
| `build_motor_current_windows_for_condition` | False | True |
| `condition_window_availability` | True | False |
| `fit_modality_normalization("temperature")` | True | False |
| `fit_modality_normalization("motor_current")` | False | True |
| unflagged call (default) | True | True |

`fusion.py::build_split_representation_table` no longer gates the
temperature attempt behind current's outcome — it now tries and catches
`CorruptRawFileError` for each modality independently.

## Corrected behaviour (verified by test)

| Configuration | Requires current | Requires temperature | BPFO retained |
|---|---|---|---|
| `vibration_only` | no | no | yes (always was) |
| `vibration_current` | yes | no | **no** (current genuinely invalid) |
| `vibration_temperature` | no | yes | **yes** (corrected — was incorrectly excluded before) |
| `vibration_current_temperature` | yes | yes | **no** (current genuinely invalid) |

Only `vibration_temperature`'s trained fusion head and its
`class_coverage_by_configuration` / training-history entries needed
regeneration; `vibration_only`, `vibration_current`, and
`vibration_current_temperature` were unaffected by the fix (their required
modality set's actual validity did not change) and were left untouched.

## Side finding, disclosed but explicitly out of scope here

`fit_modality_normalization("motor_current")` previously would have crashed
with an uncaught `CorruptRawFileError` if re-run against today's raw data,
because it inherited the same blanket 5-channel check with no exception
handling around the per-condition loop. It now (a) validates only the
current-channel group for this modality, and (b) skips and records any
train condition whose current group is genuinely invalid, matching the
exclusion pattern already used elsewhere in this pipeline
(`representation.py::assemble_split_arrays`).

Recomputing normalization stats with the fix confirms **vibration and
temperature are numerically unchanged** (bit-identical mean/std to the
currently persisted `multimodal_normalization_params.json`) — those two
modalities' fitted statistics were never actually affected by the
current-channel corruption. **`motor_current`'s persisted statistic differs
slightly** (persisted: mean=0.001400, std=2.283664, `n_train_conditions=31`;
freshly computed with the fix: mean=-0.000038, std=2.275692,
`n_train_conditions=25`, 6 excluded) — the persisted value was computed by
code that included the lone valid U-phase channel's real samples from the 6
corrupt BPFO train conditions while silently contributing zero from the two
empty V/W-phase channels, rather than excluding those conditions outright.

This drift is small and does not change Task 34's reported near-chance
motor_current accuracy finding, but regenerating it would require
retraining the motor_current encoder (Task 34) to stay consistent with new
normalization statistics — outside this audit's scope, which is limited to
Task 35's fusion-level exclusion logic. Not fixed here; flagged for a
dedicated follow-up task.
