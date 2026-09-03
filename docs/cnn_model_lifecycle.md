# CNN Model Lifecycle — Phase 4D (Task 24)

**Module**: `src/continual/model_registry.py`
**Status**: candidate activation, persistence, versioning, and rollback. Training (Task 23) stays completely separate — this module never trains anything.

## Candidate vs. Active Model

```
Active CNN (version n)
     |
train_candidate_head()          <- Task 23, unmodified, in-memory only
     |
candidate model (in memory)
     |
persist_candidate()             <- staged under candidates/{id}/, NEVER
     |                              touches versions/ or the active pointer
evaluate_candidate_regression() <- Task 23, unmodified -> GateReport
     |
   ACCEPT?
   /    \
 NO      YES
  |       |
STOP   activate_candidate()     <- the ONLY function that can advance
          |                        the active pointer
     version n+1 committed atomically
```

A **candidate** is a model that exists (in memory, then on disk in `candidates/`) but has no version number and cannot be loaded via `load_active_model()`. An **active model** is whatever `load_active_model()` returns — always exactly one, at a time, referenced by the pointer file. Nothing but `activate_candidate()` (and `rollback()`, targeting an *existing* version) can change which version is active.

## Model Versions

```
models/continual/
    versions/
        v1/
            model.keras
            metadata.json
        v2/
            model.keras
            metadata.json
    candidates/
        cand_xyz/
            model.keras
            metadata.json
    active_cnn_pointer.json
```

No existing versioning convention was found in the repository to reuse (`models/` held only the single, flat, unversioned `cwru_cnn_baseline.keras`), so this introduces one, following `docs/cnn_continual_adaptation_design.md` Section 8's proposal closely, placed under `models/continual/` — deliberately separate from the original baseline artifact, which this task never touches, and alongside `data/adaptation_buffer/`'s existing convention of giving continual-learning state its own clearly-labeled area.

Every version's `metadata.json` records: `version`, `parent_version`, `candidate_id`, `dataset`, `condition_id`, training/validation sample counts, per-condition accuracy (both the model that was active at training time and the candidate), `novelty_reference_version`, `embedding_backbone_hash`, a full `architecture_signature`, and — for activated versions — the exact `gate_decision` and `gate_reasons` that justified activation. This is what makes the adaptation history auditable: given any version, you can answer "what was it trained from, on what evidence, and why was it accepted" without external records.

## Activation Sequence and the ACCEPT-Only Rule

`activate_candidate(candidate_id, gate_report)` requires `gate_report.decision is GateDecision.ACCEPT` (the exact enum from `src/continual/safety_regression_gate.py`, not redefined). Everything else raises and leaves the registry untouched:

- `REJECT` or `REVIEW` → raises `ValueError`, no state change.
- `gate_report` missing (`None`) or not a `GateReport` → raises `TypeError`.
- `gate_report.decision` not a `GateDecision` (e.g. the plain string `"accept"`) → raises `TypeError`.

This module never decides acceptance itself — it only enforces that `SafetyRegressionGate` (Task 20/23, unmodified) already did. What it *does* decide, independently, is architecture/backbone **compatibility** (see below) — a check specific to model activation that the gate has no reason to know about.

## Atomicity and Failure Behaviour

**No partially-written state is ever observable.** Both the pointer file and every version/candidate directory are written to a `.tmp` path, verified, and only then atomically promoted:

- Pointer: write `active_cnn_pointer.json.tmp` (flush + `fsync`), then `os.replace()` onto the real path — atomic on both POSIX and Windows for a same-directory rename.
- A version/candidate directory: write to `v{n}.tmp/` (or `{candidate_id}.tmp/`), **reload the just-written model and confirm its weights exactly match the in-memory model being persisted**, and only then `os.replace()` the whole directory into its final name.

If any step fails — the model can't be saved, the reload doesn't match, metadata is invalid — the `.tmp` directory is removed and the exception propagates. Critically, this means **a failed or rejected activation never leaves a `v{n}/` directory behind at all**, which is what makes version numbers monotonic without gaps from failed attempts: `_next_version_number()` is computed by scanning existing `v{n}/` directories, so a version number is only ever "spent" by a `v{n}/` directory that actually, fully exists. `tests/test_model_registry.py::test_14` and `test_25` demonstrate this directly — a rejected candidate never occupies a slot, and no `.tmp` remnant survives a failed attempt.

## Rollback

`rollback(target_version)` only ever moves the pointer to a version that already exists (verified loadable before the pointer is touched) — it never deletes, modifies, or re-derives anything. `v1 → v2 → rollback(1) → v1` (`test_15`) leaves both `v1/` and `v2/` on disk; rolling forward again later is just another pointer write. Rollback to a version that was never registered raises immediately (`test_16`).

## Candidate Compatibility

Because Task 23 freezes the entire embedding backbone, an activated candidate's backbone is *supposed* to be byte-identical to its parent's — but this module doesn't merely trust that contract. `check_compatibility()` independently verifies, against the parent version's own recorded metadata:

| Check | How |
| :--- | :--- |
| Same architecture | Every layer's name and weight shapes match exactly |
| Same embedding dimensionality | The `learned_embedding` layer's output width matches |
| Same frozen-backbone weights | SHA-256 over every backbone layer's weight *values* matches (`compute_backbone_hash`) — catches the backbone being touched by anything, anywhere, not just a freeze-flag failing |
| Same output/class structure | Output shape matches |

Any mismatch fails closed (`ValueError`, nothing activated) — demonstrated for an embedding-dimension mismatch (`test_17`) and for a backbone-weight mismatch from a differently-initialized model (`test_18`), and a genuinely compatible frozen clone is confirmed to pass (`test_19`).

## Relationship to NoveltyReference

This module never imports a mutating `NoveltyReference` method and never calls one — `test_20` confirms a reference's version and prototypes are completely unaffected by a full persist-then-activate cycle. It only **records** whichever `novelty_reference_version` a caller supplies (the version that was valid for the model being activated) in that version's metadata — because Task 23's head-only adaptation never touches the embedding backbone, that recorded reference version remains exactly as valid after activation as before, with no re-encoding step required. This module's job is limited to writing that fact down for audit, not acting on it.

## Failure Behaviour Summary

In every tested failure case — malformed decision, REJECT/REVIEW, incompatible candidate, nonexistent version for rollback, corrupted metadata file, missing version — **the active model is unchanged afterward**, verified directly in each corresponding test by re-checking `get_active_version()` and/or comparing weight arrays.

## Limitations

- Only the compatibility properties listed above are checked; this is not a full behavioral equivalence proof (e.g. it does not re-run inference to compare outputs) — architecture and backbone-weight identity are the properties Task 23's design actually depends on, so those are what's verified.
- No automatic cleanup/retention policy for rejected candidates left in `candidates/` — they remain for audit indefinitely; pruning is left to a future task if it becomes necessary.
- No concurrent-writer protection beyond the atomic single-writer replace pattern (no file lock) — sufficient for this project's current single-process usage, not designed for multiple simultaneous adaptation processes.
- This module does not implement the trigger that decides *when* to attempt an adaptation, or wire itself into `GatedPrototypeAdmissionController` — it is deliberately limited to what happens once a candidate and a `GateReport` already exist, per this task's scope.
