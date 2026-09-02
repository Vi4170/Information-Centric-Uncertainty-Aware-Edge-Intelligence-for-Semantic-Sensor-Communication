# Adaptation Buffer — Phase 1 (Task 17)

**Module**: `src/continual/adaptation_buffer.py`
**Status**: infrastructure only. Not connected to the production pipeline (`src/integration/voi_pipeline.py`), `src/voi/`, the CNN, or any factor module. No learning logic exists yet.

## Purpose

Implements Phase 1 of `docs/continual_learning_design.md`: a passive store that decides *which observations are even allowed to be considered* for future adaptation, before any condition detection, novelty-reference updating, or CNN fine-tuning exists. It records references and metadata only — never the raw sensor array.

## Accepted / Rejected

| Input | Result |
| :--- | :--- |
| `split="train"` or `split="val"` | Accepted |
| `split="test"` (single `add()` or any row in `add_from_dataframe()`) | **Rejected** — raises `ValueError`. A batch containing even one test row rejects the *entire* batch; nothing is partially ingested. |
| Any other split string (e.g. `"holdout"`) | Rejected — raises `ValueError` |
| Duplicate `observation_id` (already in the buffer, or duplicated within one batch) | Rejected — raises `ValueError`. Never silently overwritten. |
| Empty `observation_id` / `dataset` / `source_recording_id` | Rejected — raises `ValueError` |
| `label_status=CONFIRMED` with `label=None` | Rejected — raises `ValueError` (a confirmed label with no value is a contradiction) |
| `label_status=PSEUDO` with no label | Accepted — pseudo-labels are optional by nature |

Rejection is always explicit (an exception), never a silent filter.

## Metadata Retained (per `AdaptationRecord`)

`observation_id`, `dataset`, `split`, `source_recording_id`, `label_status` (`confirmed` / `pseudo`), `label` (optional), `window_index` (optional), `extra` (free-form dict for future regression/replay checks — populated by callers, never computed by this module).

## Leakage Safeguards

1. **Per-insert rejection**: every `add()` and `add_from_dataframe()` call validates `split` before anything is stored; `"test"` is hard-rejected.
2. **Independent second check**: `AdaptationBuffer.verify_no_test_leakage(test_observation_ids)` compares the buffer's contents against an *externally supplied, authoritative* list of test ids (e.g. loaded straight from a dataset's metadata) — a safety net that doesn't rely on trusting each record's own stored `split` field.
3. **Atomic batch ingestion**: a batch with any invalid row (test split, duplicate, already-present id) is rejected wholesale, so the buffer can never end up in a partially-contaminated state from one call.
4. **Structural separation**: the buffer persists to `data/adaptation_buffer/`, deliberately outside `data/processed/<dataset>/` where evaluation-only test arrays live.
5. Verified against real data: `tests/test_adaptation_buffer.py::test_21_real_cwru_metadata_integration` ingests the actual CWRU train+val metadata and asserts zero overlap with the real CWRU test observation ids.

## Determinism

Internally insertion-ordered (`dict`, Python 3.7+ preserves insertion order); no randomness anywhere. Two buffers built from identical input produce byte-identical `to_dataframe()` output (`tests/test_adaptation_buffer.py::test_15`). `save_json()` / `load_json()` round-trip without loss.

## How This Supports Later Phases

- **Phase 2 (Condition Monitor)** will read from this buffer (or an equivalent live stream) to evaluate rolling statistics — it never needs to re-derive leakage safety itself, since ineligible data can never have entered here.
- **Phase 3 (novelty reference extension)** will use `label_status=PSEUDO` records (cluster/recurrence-based) to propose new prototypes, and `get_by_recording()` to inspect a candidate condition's full recording.
- **Phase 4 (gated CNN fine-tuning)** will use `label_status=CONFIRMED` records only, per `docs/continual_learning_design.md` §4.2's strict pseudo-vs-confirmed separation — this buffer is what makes that separation enforceable rather than a documentation-only rule.
- **Every phase's regression gate** can reuse `extra` (once populated by those future phases) to compare pre/post-adaptation behaviour without re-plumbing metadata storage.

## Not Yet Implemented (explicitly out of scope for Phase 1)

Condition detection, novelty prototype updates, CNN fine-tuning, the Adaptation Controller itself, and any wiring into `src/voi/` or `src/integration/`.
