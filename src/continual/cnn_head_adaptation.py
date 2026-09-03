"""Leakage-Safe CNN Head Adaptation (Task 23 -- CNN Continual Adaptation
Design, Tier 1: classifier/head-only).

Implements ONLY the first, conservative CNN-adaptation mechanism chosen in
docs/cnn_continual_adaptation_design.md Section 4: freeze the ENTIRE
embedding backbone (every Conv1D block plus the `learned_embedding` dense
layer) and train ONLY the final classifier head
(`output_probabilities`). This is the specific design choice that makes
Section 9's embedding-space-consistency problem disappear rather than
requiring a solution for it: with the backbone frozen, embeddings are
provably identical before and after adaptation, so every existing
NoveltyReference prototype remains valid with zero re-encoding.

This module does NOT:
    - touch src/voi/, src/cnn/model.py's architecture, or src/novelty/,
    - implement partial-backbone or full fine-tuning,
    - persist/version model files or activate a candidate automatically
      (per Task 22 Section 7/8, that is a separate, future, gated
      operation -- this module only ever returns an in-memory candidate),
    - modify ConditionMonitor, NoveltyReference, or
      SafetyRegressionGate -- it calls SafetyRegressionGate's existing,
      public evaluate() method, unmodified, both as a pre-training
      fail-closed check on the supplied evidence and as the post-training
      regression check.

Data flow:

    active_model
        | clone_model_with_weights()          <- independent copy,
        |                                         active_model never touched
    candidate_model
        | freeze_backbone()                    <- verified, not assumed
        |
    gate.evaluate(observations=train_records)  <- fail-closed pre-check,
        |                                          reuses Task 20 unmodified
    candidate_model.fit(X_train, y_train,
                         validation_data=(X_val, y_val))
        |
    candidate_model + LayerFreezeReport + per-condition accuracy
        |
    (optional, separate call) gate.evaluate(..., baseline_metrics=...,
                               candidate_metrics=...)  <- regression check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

from src.cnn.config import EMBEDDING_LAYER_NAME
from src.cnn.model import predict_classes
from src.continual.adaptation_buffer import AdaptationRecord, LabelStatus
from src.continual.novelty_reference import NoveltyReference
from src.continual.safety_regression_gate import SafetyRegressionGate

# The entire embedding-producing path -- everything up to and including
# the layer NoveltyReference's prototypes are built from
# (src.cnn.config.EMBEDDING_LAYER_NAME). "dropout" carries no trainable
# weights and sits after the embedding, but is listed here (not in the
# head) since it is conceptually part of the frozen representation path,
# not the classifier being adapted.
BACKBONE_LAYER_NAMES: Tuple[str, ...] = (
    "conv1d_1",
    "maxpool_1",
    "conv1d_2",
    "maxpool_2",
    "conv1d_3",
    "global_pool",
    EMBEDDING_LAYER_NAME,
    "dropout",
)

# The only layer whose weights this module ever trains.
HEAD_LAYER_NAMES: Tuple[str, ...] = ("output_probabilities",)


@dataclass(frozen=True)
class LayerFreezeReport:
    """Verified (not assumed) result of freezing the backbone."""

    backbone_layer_names: Tuple[str, ...]
    head_layer_names: Tuple[str, ...]
    backbone_trainable_param_count: int
    head_trainable_param_count: int
    all_backbone_frozen: bool


@dataclass(frozen=True)
class CandidateAdaptationResult:
    """Everything produced by one train_candidate_head() call.

    `candidate_model` is a fully independent keras.Model -- never the same
    object as, and never sharing weight arrays with, the active model this
    was cloned from. Nothing in this dataclass or this module writes to
    disk or mutates any "active model" pointer; activation remains a
    separate, future, gated operation (Task 22 Section 7/8).
    """

    candidate_model: keras.Model
    freeze_report: LayerFreezeReport
    history: Dict[str, List[float]]
    condition_id: str
    n_training_samples: int
    n_validation_samples: int
    training_record_ids: Tuple[str, ...]
    validation_record_ids: Tuple[str, ...]
    per_condition_accuracy_active: Dict[str, float]
    per_condition_accuracy_candidate: Dict[str, float]
    overall_accuracy_active: float
    overall_accuracy_candidate: float


@dataclass(frozen=True)
class RehearsalSelection:
    """Deterministic, bounded, per-condition rehearsal sample selection."""

    condition_id_to_indices: Dict[str, Tuple[int, ...]]
    condition_id_to_record_ids: Dict[str, Tuple[str, ...]]
    k_per_condition: int


def _validate_confirmed_only(records: Sequence[AdaptationRecord], context: str) -> None:
    """Fail closed unless every record is CONFIRMED with a non-None label.

    This is deliberately re-checked here (a simple, direct filter, not a
    reimplementation of AdaptationBuffer's/SafetyRegressionGate's fuller
    provenance/split validation, which are still delegated to
    SafetyRegressionGate.evaluate() below) so that this module never even
    attempts to build a training array from pseudo-labelled or unlabeled
    evidence, regardless of what a caller passes in.
    """
    if len(records) == 0:
        raise ValueError(f"{context}: no adaptation records supplied. Refusing to proceed.")

    non_confirmed = [r.observation_id for r in records if r.label_status is not LabelStatus.CONFIRMED]
    if non_confirmed:
        raise ValueError(
            f"{context}: CNN head adaptation requires CONFIRMED-label records only; found "
            f"{len(non_confirmed)} non-confirmed record(s) (e.g. {non_confirmed[:5]}). "
            "Pseudo-labelled records are never used for supervised CNN training."
        )

    missing_labels = [r.observation_id for r in records if r.label is None]
    if missing_labels:
        raise ValueError(
            f"{context}: {len(missing_labels)} CONFIRMED record(s) have no label value "
            f"(e.g. {missing_labels[:5]}). A confirmed label with no value is a contradiction."
        )


def clone_model_with_weights(model: keras.Model) -> keras.Model:
    """Return a fully independent copy of `model`: same architecture, same
    weight VALUES, but no shared weight arrays or layer objects. Mutating
    the returned model can never affect `model`.
    """
    candidate = keras.models.clone_model(model)
    candidate.set_weights(model.get_weights())
    return candidate


def freeze_backbone(
    model: keras.Model,
    backbone_layer_names: Sequence[str] = BACKBONE_LAYER_NAMES,
    head_layer_names: Sequence[str] = HEAD_LAYER_NAMES,
) -> LayerFreezeReport:
    """Set `.trainable` on the named layers and VERIFY the resulting
    trainable-parameter counts GLOBALLY -- never merely assume the flag
    took effect, and never trust only the layers we were told about.

    The verification deliberately does not just re-check the layers named
    in `backbone_layer_names` (a caller could pass an incomplete list and
    such a check would happily report success while a forgotten layer
    stayed trainable). Instead it inspects `model.trainable_weights` --
    every trainable weight anywhere in the whole model -- and confirms
    each one belongs to a head layer. This is what makes the freeze
    verification meaningful: it catches ANY unexpectedly-trainable layer,
    not only ones explicitly listed as "backbone".

    Args:
        model: The model to freeze in place (typically a freshly cloned candidate).
        backbone_layer_names: Layers to explicitly freeze (trainable = False).
        head_layer_names: Layers to leave trainable (trainable = True).

    Returns:
        LayerFreezeReport with the verified, whole-model parameter counts.

    Raises:
        ValueError: If any named layer does not exist in `model`.
    """
    for name in backbone_layer_names:
        model.get_layer(name).trainable = False
    for name in head_layer_names:
        model.get_layer(name).trainable = True

    head_weights = [w for name in head_layer_names for w in model.get_layer(name).trainable_weights]
    head_weight_ids = {id(w) for w in head_weights}

    # Whole-model check: every trainable weight in the ENTIRE model must
    # belong to a head layer. A layer left trainable by mistake -- even
    # one never named in backbone_layer_names at all -- shows up here.
    leaked_backbone_weights = [w for w in model.trainable_weights if id(w) not in head_weight_ids]

    backbone_trainable = sum(int(np.prod(w.shape)) for w in leaked_backbone_weights)
    head_trainable = sum(int(np.prod(w.shape)) for w in head_weights)

    return LayerFreezeReport(
        backbone_layer_names=tuple(backbone_layer_names),
        head_layer_names=tuple(head_layer_names),
        backbone_trainable_param_count=backbone_trainable,
        head_trainable_param_count=head_trainable,
        all_backbone_frozen=(backbone_trainable == 0),
    )


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[Dict[int, str]] = None) -> Dict[str, float]:
    """Accuracy for each class present in y_true, keyed by a string label
    (matching SafetyRegressionGate's Dict[str, float] metrics contract).
    """
    result: Dict[str, float] = {}
    for class_id in sorted(set(int(c) for c in y_true)):
        mask = y_true == class_id
        label = class_names.get(class_id, str(class_id)) if class_names else str(class_id)
        result[label] = float(np.mean(y_pred[mask] == class_id))
    return result


def train_candidate_head(
    active_model: keras.Model,
    gate: SafetyRegressionGate,
    condition_id: str,
    train_records: Sequence[AdaptationRecord],
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_records: Sequence[AdaptationRecord],
    X_val: np.ndarray,
    y_val: np.ndarray,
    condition_monitor_results: Sequence = (),
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    class_names: Optional[Dict[int, str]] = None,
    backbone_layer_names: Sequence[str] = BACKBONE_LAYER_NAMES,
    head_layer_names: Sequence[str] = HEAD_LAYER_NAMES,
) -> CandidateAdaptationResult:
    """Train a head-only-adapted CANDIDATE model. Never mutates `active_model`.

    Fails closed (raises) before any training occurs if: the training or
    validation record sets contain anything other than CONFIRMED,
    labelled evidence; the record/array lengths don't match; or
    `gate.evaluate()` -- reusing Task 20's existing, unmodified provenance/
    split/count/sustained-evidence validation -- rejects the training
    evidence on safety grounds (e.g. a test-split or unrecognized-split
    record present, insufficient count, invalid provenance, or no
    sustained-shift evidence supplied).

    Args:
        active_model: The currently active CNN. Never modified.
        gate: A SafetyRegressionGate instance used both as a pre-training
            fail-closed check on `train_records` (called with no
            regression metrics, so its decision reflects safety alone)
            and, separately, to hold whatever config a caller wants for a
            later regression re-check (see evaluate_candidate_regression()).
        condition_id: Identifier for this adaptation attempt (audit only,
            passed straight through to the gate).
        train_records / X_train / y_train: The confirmed-label adaptation
            training data. Every array's length must match len(train_records).
        val_records / X_val / y_val: The confirmed-label adaptation
            validation data -- used ONLY for Keras's `validation_data`
            monitoring and for the per-condition accuracy comparison
            returned here. Never included in any gradient update.
        condition_monitor_results: The same ConditionMonitorResult evidence
            that justified this condition's prototype admission (Task 21),
            passed straight through to the gate's sustained-evidence check.
            CNN adaptation is a higher-risk decision than prototype
            admission, so it is held to at least the same evidence
            standard -- this is required by the gate (Task 20), not
            optional, and an empty sequence will fail the pre-check
            (fail-closed), exactly as it does for prototype admission.
        epochs, batch_size, learning_rate, seed: Standard training
            hyperparameters for the head-only fit.
        class_names: Optional {class_id: name} mapping for readable
            per-condition accuracy keys; defaults to the string class id.

    Returns:
        CandidateAdaptationResult with the candidate model and full metadata.

    Raises:
        ValueError: On any data-validation failure (see above). No
            training is attempted and `active_model` is untouched.
        RuntimeError: If backbone-freeze verification fails (defensive;
            should be unreachable given freeze_backbone()'s own checks).
    """
    if len(X_train) != len(train_records) or len(y_train) != len(train_records):
        raise ValueError("X_train/y_train length must match len(train_records)")
    if len(X_val) != len(val_records) or len(y_val) != len(val_records):
        raise ValueError("X_val/y_val length must match len(val_records)")

    _validate_confirmed_only(train_records, context="train_records")
    _validate_confirmed_only(val_records, context="val_records")

    # Fail-closed pre-check: reuse SafetyRegressionGate's existing,
    # unmodified provenance/split/count/sustained-evidence validation
    # rather than re-implementing it a third time. No regression metrics
    # are supplied
    # here, so the resulting decision reflects safety alone.
    pre_check = gate.evaluate(
        condition_id=condition_id,
        observations=tuple(train_records),
        condition_monitor_results=tuple(condition_monitor_results),
    )
    if not pre_check.safety.passed:
        raise ValueError(
            f"Adaptation training data failed the safety gate pre-check: "
            f"{pre_check.safety.failed_checks}"
        )

    keras.utils.set_random_seed(seed)

    candidate_model = clone_model_with_weights(active_model)
    freeze_report = freeze_backbone(candidate_model, backbone_layer_names, head_layer_names)
    if not freeze_report.all_backbone_frozen:
        raise RuntimeError(
            f"Backbone freeze verification failed: {freeze_report.backbone_trainable_param_count} "
            "trainable parameters remain in backbone layers. Refusing to train."
        )

    candidate_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    fit_history = candidate_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    history = {k: [float(v) for v in vals] for k, vals in fit_history.history.items()}

    active_val_pred = predict_classes(active_model, X_val)
    candidate_val_pred = predict_classes(candidate_model, X_val)

    return CandidateAdaptationResult(
        candidate_model=candidate_model,
        freeze_report=freeze_report,
        history=history,
        condition_id=condition_id,
        n_training_samples=len(train_records),
        n_validation_samples=len(val_records),
        training_record_ids=tuple(r.observation_id for r in train_records),
        validation_record_ids=tuple(r.observation_id for r in val_records),
        per_condition_accuracy_active=_per_class_accuracy(y_val, active_val_pred, class_names),
        per_condition_accuracy_candidate=_per_class_accuracy(y_val, candidate_val_pred, class_names),
        overall_accuracy_active=float(np.mean(active_val_pred == y_val)),
        overall_accuracy_candidate=float(np.mean(candidate_val_pred == y_val)),
    )


def evaluate_candidate_regression(
    gate: SafetyRegressionGate,
    result: CandidateAdaptationResult,
    train_records: Sequence[AdaptationRecord],
    condition_monitor_results: Sequence = (),
):
    """Second, separate gate call: full safety + regression re-check using
    the per-condition accuracy already computed by train_candidate_head().

    This does NOT activate anything. It returns the same GateReport type
    every other continual-learning component already uses (Task 20/21) --
    ACCEPT means "a future controller may activate this candidate",
    REJECT/REVIEW mean it must not be activated. Reuses
    SafetyRegressionGate.evaluate() unmodified.
    """
    return gate.evaluate(
        condition_id=result.condition_id,
        observations=tuple(train_records),
        condition_monitor_results=tuple(condition_monitor_results),
        baseline_metrics=result.per_condition_accuracy_active,
        candidate_metrics=result.per_condition_accuracy_candidate,
    )


def select_rehearsal_samples(
    reference: NoveltyReference,
    model: keras.Model,
    pool_records: Sequence[AdaptationRecord],
    pool_X: np.ndarray,
    k_per_condition: int,
) -> RehearsalSelection:
    """Deterministic, bounded, per-condition rehearsal sample selection
    (docs/cnn_continual_adaptation_design.md Section 3).

    For each pool sample, assigns it to whichever existing NoveltyReference
    prototype is nearest (reusing NoveltyReference.nearest_prototype()
    unmodified -- no distance/centroid logic is duplicated here), then
    within each resulting group ranks samples by that same nearest-distance
    ascending (most representative first; ties broken by index) and keeps
    the first `k_per_condition`. No randomness anywhere.

    Args:
        reference: The current NoveltyReference (queried read-only; never mutated).
        model: The model whose extract_embeddings() produces the space
            `reference`'s prototypes live in (i.e. the ACTIVE model, not a
            candidate -- rehearsal selection must use the same embedding
            space the reference was built in).
        pool_records: Candidate rehearsal pool. Every record must be
            CONFIRMED with a non-test, permitted split (fails closed
            otherwise) -- reuses the same validation as train_candidate_head().
        pool_X: Raw input windows for `pool_records`, same order, same length.
        k_per_condition: Maximum samples retained per matched condition.

    Returns:
        RehearsalSelection mapping each matched prototype id to the
        (deterministically ordered) indices and record ids selected for it.

    Raises:
        ValueError: On invalid pool records, length mismatch, or an empty reference.
    """
    if len(pool_records) != len(pool_X):
        raise ValueError("pool_records and pool_X must have the same length")
    _validate_confirmed_only(pool_records, context="pool_records")
    if reference.version == 0:
        raise ValueError("Cannot select rehearsal samples: NoveltyReference has no prototypes yet")

    from src.cnn.model import extract_embeddings  # local import: keeps this module's top-level

    embeddings = extract_embeddings(model, pool_X)

    condition_id_to_indices: Dict[str, List[Tuple[int, float]]] = {}
    for i, embedding in enumerate(embeddings):
        prototype, distance = reference.nearest_prototype(embedding)
        condition_id_to_indices.setdefault(prototype.prototype_id, []).append((i, distance))

    final_indices: Dict[str, Tuple[int, ...]] = {}
    final_record_ids: Dict[str, Tuple[str, ...]] = {}
    for prototype_id, entries in condition_id_to_indices.items():
        entries.sort(key=lambda pair: (pair[1], pair[0]))  # distance asc, index asc for deterministic ties
        selected = [i for i, _ in entries[:k_per_condition]]
        final_indices[prototype_id] = tuple(selected)
        final_record_ids[prototype_id] = tuple(pool_records[i].observation_id for i in selected)

    return RehearsalSelection(
        condition_id_to_indices=final_indices,
        condition_id_to_record_ids=final_record_ids,
        k_per_condition=k_per_condition,
    )
