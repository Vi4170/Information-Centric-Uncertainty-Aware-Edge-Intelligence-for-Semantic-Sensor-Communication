# Versioned Multi-Prototype Novelty Reference — Phase 3 (Task 19)

**Module**: `src/continual/novelty_reference.py`
**Status**: infrastructure only. Not wired into `src/novelty/`, `src/voi/`, `src/integration/`, the Condition Monitor (Task 18), or the Adaptation Buffer (Task 17). No prototype is ever created automatically by anything in this repository.

## Why One Centroid Is Insufficient for Continual Learning

`src/novelty/novelty.py`'s `DistanceNoveltyDetector` fits **one** centroid from Normal-labelled training embeddings, fixed forever after `fit()`. That's correct for a static baseline, but continual learning's whole premise (per `docs/continual_learning_design.md`) is that new, legitimate operating conditions will appear over time. A single centroid can't represent "condition A is known AND condition B is now also known" — either you overwrite the centroid (silently redefining what "normal" means, which could mask real faults) or you never recognize condition B as anything but perpetually novel. Both are wrong. The fix the design doc proposes: keep a **growing set** of validated reference points, and score novelty as distance to the *nearest* one.

## Structure

```
Observation embedding
         |
    distance to Prototype A
    distance to Prototype B
    distance to Prototype C
         |
   nearest prototype
         |
   novelty distance (raw, unnormalized)
```

`NoveltyReference(embedding_dim)` holds an ordered list of `Prototype` objects. Each `Prototype` carries: `prototype_id`, its `centroid` (mean of the embeddings it was built from, read-only after creation), `embedding_dim`, `source_dataset`, `source_condition`, `source_split`, `n_source_embeddings`, `version_added`, and a free-form `extra` dict.

## Versioning

`version = number of prototypes successfully added so far` (0 = empty). Each `add_prototype()` call appends exactly one audit entry to `.history` and increments `.version` by 1:

```
version 1 -> {A}
version 2 -> {A, B}      (A byte-identical to version 1)
version 3 -> {A, B, C}   (A, B byte-identical to version 2)
```

There is deliberately **no** `remove_prototype` or `update_prototype` method — append-only is enforced structurally, not just by convention. `tests/test_novelty_reference.py::test_03` and `test_13` explicitly prove a prototype's centroid and lookup behaviour are unaffected by any number of later additions.

## Nearest-Prototype Behaviour

`nearest_prototype(embedding)` computes the raw Euclidean distance from the given embedding to every prototype's centroid (`distance_report()`) and returns whichever is smallest, deterministically (ties broken by insertion order — Python's `min()` guarantee). `distance_to_nearest()` is a convenience for just the number.

## Prototype Admission Boundary (what this task does NOT do)

Adding a prototype requires an explicit `add_prototype(...)` call from a caller. This module never creates one on its own:

- A Condition Monitor `CANDIDATE_CONDITION_SHIFT` (Task 18) does not, by itself, cause a prototype to be added — nothing connects the two.
- A pseudo-label does not cause a prototype to be added.
- A high novelty score does not cause a prototype to be added.
- A test-split observation **cannot** be used to create one — `source_split="test"` is rejected inside `add_prototype()` itself, reusing the exact same forbidden-split check `src/continual/adaptation_buffer.py` (Task 17) already established, rather than trusting caller discipline.

Deciding *when* it's safe to call `add_prototype()` is the future Safety Gate's job (`docs/continual_learning_design.md` §4.1) — a later phase, not this one.

## Normalization and Leakage

`distance_to_nearest()` / `distance_report()` return the **raw, unnormalized** Euclidean distance — not a `[0, 1]` novelty score. This is deliberate: `DistanceNoveltyDetector`'s existing `[0, 1]` scaling uses `d_min`/`d_max` bounds fit on the *single-centroid* distance distribution of the training set. Reusing those bounds here would be a category error (they describe a different quantity — distance to one fixed point, not distance to whichever of several growing prototypes is nearest), and computing *new* bounds would require deciding what a multi-prototype reference's "maximum expected distance" is before any second prototype has ever been validated — an assumption about future adaptation data this task has no principled basis for making yet. Rather than invent a calibration, this module exposes the raw distance and documents the gap; a normalized multi-prototype novelty score is left to whichever future task actually has real multi-condition adaptation data to calibrate against.

No prototype's centroid, bounds, or any other parameter is ever fit using test-split data — enforced by the same `source_split` check above, and demonstrated with real data conventions matching Tasks 17/18.

## Relationship to the Existing Novelty Module

`src/novelty/novelty.py` is unchanged and remains the CWRU baseline novelty implementation used everywhere it already was (`src/evaluation/voi_behaviour_analysis.py` and upstream). No adapter was needed: both modules consume the exact same input contract (2D `(N, embedding_dim)` float arrays from `src/cnn`'s `extract_embeddings()`), and the only logic in common — Euclidean distance to a centroid — is a one-line `np.linalg.norm` call, not substantial enough to warrant sharing an abstraction. The two modules can be used side by side for baseline comparison once a future task wires this one up.

## How This Connects to the Future Safety Gate

Per `docs/continual_learning_design.md` §4.1/§4.3, the future Adaptation Controller will call `add_prototype()` only after its Safety Gate (sustained, corroborated evidence — not one observation) and Regression Gate (re-evaluating retained validation data across every previously known condition) both pass. This module is what makes that gate's decision *enforceable and auditable* — every accepted prototype is permanently recorded with its provenance and version, and nothing bypasses `add_prototype()` to add one any other way.

## Naming Note

`src/novelty/distance_novelty.py` (part of an earlier, pre-existing two-tier implementation pattern in `src/novelty/` noted during Task 11's review) happens to define an unrelated internal dataclass also named `NoveltyReference` (holding `mean`/`precision`/`training_distances` for a Mahalanobis-distance detector). It is not imported anywhere by name outside that one file, so there is no practical namespace collision, but it is worth knowing the name is reused for a structurally different, unrelated purpose in that older, separate module.

## What This Task Does Not Implement

Condition detection (Task 18, already done, not connected here), the Safety Gate, the Regression Gate, the Adaptation Controller, any automatic prototype creation, and any normalized multi-prototype novelty score. **CWRU is not used to demonstrate continual learning in this task** — the synthetic-embedding tests in `tests/test_novelty_reference.py` are an infrastructure demonstration of the append-only mechanism, not an experimental result.
