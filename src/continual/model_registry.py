"""Safe CNN Candidate Activation & Versioning (Task 24 -- CNN Continual
Adaptation Design, Phase 4D).

Implements the model lifecycle docs/cnn_continual_adaptation_design.md
Sections 7/8 describe: candidate training (Task 23, unmodified) stays
completely separate from candidate ACTIVATION, which this module owns.

    Active CNN (version n)
         |
    train_candidate_head()          <- Task 23, unmodified, in-memory only
         |
    candidate model (in memory)
         |
    persist_candidate()             <- staged under candidates/{id}/,
         |                              NEVER touches versions/ or the pointer
    (regression evaluation, Task 23's evaluate_candidate_regression() --
     produces a GateReport, unmodified)
         |
       ACCEPT?
       /    \
     NO      YES
      |       |
    STOP   activate_candidate()     <- the ONLY function that can advance
              |                        the active pointer; requires
              |                        gate_report.decision == ACCEPT
         version n+1 committed atomically, pointer flips only after
         the new version is fully written and verifiably reloadable

Rejected/REVIEW candidates remain in candidates/ (for audit) but never
consume a version number -- version numbers advance only inside
activate_candidate()'s successful path, so the sequence is always
v1 -> v2 -> v3 -> ..., never v1 -> candidate_v2 -> rejected -> v3.

This module does not retrain anything, does not change Task 23's
adaptation algorithm, does not touch NoveltyReference (it only RECORDS
which NoveltyReference version a model was paired with, in metadata), and
does not modify SafetyRegressionGate -- it only reads the GateDecision the
gate already produced.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np

from src.cnn.config import EMBEDDING_LAYER_NAME
from src.continual.cnn_head_adaptation import BACKBONE_LAYER_NAMES, CandidateAdaptationResult
from src.continual.config import (
    ACTIVE_POINTER_FILENAME,
    DEFAULT_MODEL_REGISTRY_DIR,
    MODEL_CANDIDATES_SUBDIR,
    MODEL_VERSIONS_SUBDIR,
)
from src.continual.safety_regression_gate import GateDecision, GateReport


@dataclass(frozen=True)
class ModelMetadata:
    """Auditable record for one model artifact -- either a staged candidate
    (`version` is None) or a committed, activatable version (`version` is set).
    """

    version: Optional[int]
    candidate_id: Optional[str]
    parent_version: Optional[int]
    dataset: str
    condition_id: str
    n_training_samples: int
    n_validation_samples: int
    per_condition_accuracy_active: Dict[str, float]
    per_condition_accuracy_candidate: Dict[str, float]
    novelty_reference_version: Optional[int]
    embedding_backbone_hash: str
    architecture_signature: Dict[str, Any]
    gate_decision: Optional[str] = None
    gate_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            version=data.get("version"),
            candidate_id=data.get("candidate_id"),
            parent_version=data.get("parent_version"),
            dataset=data["dataset"],
            condition_id=data["condition_id"],
            n_training_samples=data["n_training_samples"],
            n_validation_samples=data["n_validation_samples"],
            per_condition_accuracy_active=dict(data.get("per_condition_accuracy_active") or {}),
            per_condition_accuracy_candidate=dict(data.get("per_condition_accuracy_candidate") or {}),
            novelty_reference_version=data.get("novelty_reference_version"),
            embedding_backbone_hash=data["embedding_backbone_hash"],
            architecture_signature=dict(data["architecture_signature"]),
            gate_decision=data.get("gate_decision"),
            gate_reasons=tuple(data.get("gate_reasons") or ()),
        )


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of comparing a candidate's architecture/backbone against a parent version."""

    compatible: bool
    same_architecture: bool
    same_embedding_dim: bool
    same_backbone_weights: bool
    same_output_structure: bool
    reasons: Tuple[str, ...]


def compute_backbone_hash(model: keras.Model, backbone_layer_names: Tuple[str, ...] = BACKBONE_LAYER_NAMES) -> str:
    """Deterministic SHA-256 over every backbone layer's weight VALUES.

    Used to verify a candidate's frozen backbone is byte-identical to its
    parent's -- an independent integrity check, not merely trusting Task
    23's freeze contract.
    """
    hasher = hashlib.sha256()
    for name in backbone_layer_names:
        layer = model.get_layer(name)
        hasher.update(name.encode("utf-8"))
        for w in layer.weights:
            hasher.update(np.asarray(w).tobytes())
    return hasher.hexdigest()


def compute_architecture_signature(model: keras.Model) -> Dict[str, Any]:
    """A JSON-serializable structural fingerprint: layer names, every
    layer's weight shapes, input/output shape, and embedding dimensionality.
    """
    embedding_layer = model.get_layer(EMBEDDING_LAYER_NAME)
    return {
        "layer_names": [layer.name for layer in model.layers],
        "layer_weight_shapes": {layer.name: [list(w.shape) for w in layer.weights] for layer in model.layers},
        "input_shape": list(model.inputs[0].shape[1:]) if model.inputs else None,
        "output_shape": list(model.outputs[0].shape[1:]) if model.outputs else None,
        "embedding_dim": int(embedding_layer.output.shape[-1]),
    }


class ModelRegistry:
    """Owns the versioned CNN model registry: bootstrap, staging, atomic
    activation, and rollback. Never trains a model itself.
    """

    def __init__(
        self,
        registry_dir: str = DEFAULT_MODEL_REGISTRY_DIR,
        backbone_layer_names: Tuple[str, ...] = BACKBONE_LAYER_NAMES,
    ) -> None:
        self.registry_dir = registry_dir
        self.versions_dir = os.path.join(registry_dir, MODEL_VERSIONS_SUBDIR)
        self.candidates_dir = os.path.join(registry_dir, MODEL_CANDIDATES_SUBDIR)
        self.pointer_path = os.path.join(registry_dir, ACTIVE_POINTER_FILENAME)
        self.backbone_layer_names = backbone_layer_names

    # ------------------------------------------------------------------
    # Active pointer
    # ------------------------------------------------------------------

    def has_active_version(self) -> bool:
        return os.path.exists(self.pointer_path)

    def get_active_version(self) -> int:
        """Raises FileNotFoundError if no version has ever been registered."""
        if not self.has_active_version():
            raise FileNotFoundError(
                f"No active model version registered at '{self.pointer_path}'. "
                "Call register_initial_version() first."
            )
        with open(self.pointer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data["active_version"])

    def load_active_model(self) -> keras.Model:
        return self.load_model_version(self.get_active_version())

    def _write_pointer_atomic(self, version: int) -> None:
        """Write-temp-then-replace: readers can never observe a partially
        written pointer file. os.replace() is atomic on both POSIX and
        Windows for a same-directory rename.
        """
        os.makedirs(self.registry_dir, exist_ok=True)
        tmp_path = self.pointer_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"active_version": version}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.pointer_path)

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------

    def _version_dir(self, version: int) -> str:
        return os.path.join(self.versions_dir, f"v{version}")

    def list_versions(self) -> Tuple[int, ...]:
        if not os.path.isdir(self.versions_dir):
            return ()
        versions = [int(name[1:]) for name in os.listdir(self.versions_dir) if name.startswith("v") and name[1:].isdigit()]
        return tuple(sorted(versions))

    def _next_version_number(self) -> int:
        existing = self.list_versions()
        return (max(existing) + 1) if existing else 1

    def get_version_metadata(self, version: int) -> ModelMetadata:
        meta_path = os.path.join(self._version_dir(version), "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No metadata found for version {version} at '{meta_path}'")
        with open(meta_path, "r", encoding="utf-8") as f:
            return ModelMetadata.from_dict(json.load(f))

    def load_model_version(self, version: int) -> keras.Model:
        model_path = os.path.join(self._version_dir(version), "model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found for version {version} at '{model_path}'")
        return keras.models.load_model(model_path, compile=False)

    def register_initial_version(
        self,
        model: keras.Model,
        dataset: str,
        condition_id: str = "baseline",
        novelty_reference_version: Optional[int] = None,
    ) -> int:
        """Bootstrap version 1 from an existing (e.g. baseline) model.

        Raises:
            ValueError: If this registry already has any version registered
                (bootstrapping must only ever happen once, into an empty registry).
        """
        if self.list_versions():
            raise ValueError(
                f"Registry at '{self.registry_dir}' already has version(s) "
                f"{self.list_versions()}; register_initial_version() may only "
                "bootstrap an empty registry."
            )

        metadata = ModelMetadata(
            version=1,
            candidate_id=None,
            parent_version=None,
            dataset=dataset,
            condition_id=condition_id,
            n_training_samples=0,
            n_validation_samples=0,
            per_condition_accuracy_active={},
            per_condition_accuracy_candidate={},
            novelty_reference_version=novelty_reference_version,
            embedding_backbone_hash=compute_backbone_hash(model, self.backbone_layer_names),
            architecture_signature=compute_architecture_signature(model),
        )
        self._commit_version_dir(1, model, metadata)
        self._write_pointer_atomic(1)
        return 1

    # ------------------------------------------------------------------
    # Candidate staging (never touches versions/ or the pointer)
    # ------------------------------------------------------------------

    def persist_candidate(
        self,
        candidate_id: str,
        candidate_model: keras.Model,
        parent_version: int,
        dataset: str,
        condition_id: str,
        adaptation_result: CandidateAdaptationResult,
        novelty_reference_version: Optional[int] = None,
    ) -> str:
        """Stage a candidate independently of the active model / version sequence.

        Never touches versions/ or the active pointer. A rejected or
        under-review candidate simply remains here for audit -- it never
        consumes a version number.

        Raises:
            ValueError: If candidate_id already exists in staging, or
                parent_version does not exist in this registry.
        """
        staging_dir = os.path.join(self.candidates_dir, candidate_id)
        if os.path.isdir(staging_dir):
            raise ValueError(f"candidate_id '{candidate_id}' already exists in staging (never overwritten).")
        if parent_version not in self.list_versions():
            raise ValueError(f"parent_version {parent_version} does not exist in this registry {self.list_versions()}")

        metadata = ModelMetadata(
            version=None,
            candidate_id=candidate_id,
            parent_version=parent_version,
            dataset=dataset,
            condition_id=condition_id,
            n_training_samples=adaptation_result.n_training_samples,
            n_validation_samples=adaptation_result.n_validation_samples,
            per_condition_accuracy_active=dict(adaptation_result.per_condition_accuracy_active),
            per_condition_accuracy_candidate=dict(adaptation_result.per_condition_accuracy_candidate),
            novelty_reference_version=novelty_reference_version,
            embedding_backbone_hash=compute_backbone_hash(candidate_model, self.backbone_layer_names),
            architecture_signature=compute_architecture_signature(candidate_model),
        )

        tmp_dir = staging_dir + ".tmp"
        self._write_model_dir(tmp_dir, candidate_model, metadata)
        self._verify_reload_matches(tmp_dir, candidate_model)
        os.replace(tmp_dir, staging_dir)
        return candidate_id

    def load_candidate(self, candidate_id: str) -> Tuple[keras.Model, ModelMetadata]:
        staging_dir = os.path.join(self.candidates_dir, candidate_id)
        model_path = os.path.join(staging_dir, "model.keras")
        meta_path = os.path.join(staging_dir, "metadata.json")
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"No staged candidate found for candidate_id '{candidate_id}'")
        model = keras.models.load_model(model_path, compile=False)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = ModelMetadata.from_dict(json.load(f))
        return model, metadata

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def check_compatibility(self, candidate_model: keras.Model, parent_version: int) -> CompatibilityReport:
        """Verify a candidate's architecture/backbone against a parent version's
        recorded metadata -- independent of trusting that Task 23's freeze
        contract was honored.
        """
        parent_meta = self.get_version_metadata(parent_version)
        candidate_arch = compute_architecture_signature(candidate_model)
        candidate_hash = compute_backbone_hash(candidate_model, self.backbone_layer_names)
        parent_arch = parent_meta.architecture_signature

        same_architecture = (
            candidate_arch["layer_names"] == parent_arch["layer_names"]
            and candidate_arch["layer_weight_shapes"] == parent_arch["layer_weight_shapes"]
        )
        same_embedding_dim = candidate_arch["embedding_dim"] == parent_arch["embedding_dim"]
        same_backbone_weights = candidate_hash == parent_meta.embedding_backbone_hash
        same_output_structure = candidate_arch["output_shape"] == parent_arch["output_shape"]

        reasons = []
        if not same_architecture:
            reasons.append("architecture_mismatch")
        if not same_embedding_dim:
            reasons.append("embedding_dimension_mismatch")
        if not same_backbone_weights:
            reasons.append("backbone_weight_mismatch")
        if not same_output_structure:
            reasons.append("output_structure_mismatch")

        return CompatibilityReport(
            compatible=(len(reasons) == 0),
            same_architecture=same_architecture,
            same_embedding_dim=same_embedding_dim,
            same_backbone_weights=same_backbone_weights,
            same_output_structure=same_output_structure,
            reasons=tuple(reasons),
        )

    # ------------------------------------------------------------------
    # Activation -- the ONLY path that can advance the active pointer
    # ------------------------------------------------------------------

    def activate_candidate(self, candidate_id: str, gate_report: GateReport) -> int:
        """Promote a staged candidate to a new, active version.

        Requires `gate_report.decision == GateDecision.ACCEPT` -- REJECT,
        REVIEW, a missing report, or a malformed decision all raise and
        leave the registry completely unchanged. This function does not
        decide acceptance itself; it only enforces that SafetyRegressionGate
        already did (Task 20/23), plus an independent architecture/backbone
        compatibility check this module owns.

        The new version is fully written to a temporary directory, reloaded,
        and weight-verified BEFORE that directory is atomically renamed into
        its final `v{n}` name -- and the active pointer is updated only
        after that rename succeeds. If any step fails, no `v{n}` directory
        ever appears (so the version number is not consumed) and the
        pointer is untouched.

        Raises:
            TypeError: If gate_report is not a GateReport, or its decision
                is not a GateDecision.
            ValueError: If the decision is not ACCEPT, the candidate/parent
                is missing or invalid, or the candidate is architecturally
                incompatible with its parent version.
        """
        if not isinstance(gate_report, GateReport):
            raise TypeError(f"gate_report must be a GateReport, got {type(gate_report)}")
        if not isinstance(gate_report.decision, GateDecision):
            raise TypeError(f"gate_report.decision must be a GateDecision, got {type(gate_report.decision)}")
        if gate_report.decision is not GateDecision.ACCEPT:
            raise ValueError(
                f"Refusing to activate candidate '{candidate_id}': gate decision is "
                f"'{gate_report.decision.value}', not ACCEPT. No model may be activated "
                "on REJECT or REVIEW."
            )

        candidate_model, candidate_meta = self.load_candidate(candidate_id)

        parent_version = candidate_meta.parent_version
        if parent_version is None or parent_version not in self.list_versions():
            raise ValueError(
                f"Candidate '{candidate_id}' has an invalid or missing parent_version "
                f"({parent_version}); registered versions are {self.list_versions()}"
            )

        compatibility = self.check_compatibility(candidate_model, parent_version)
        if not compatibility.compatible:
            raise ValueError(
                f"Refusing to activate incompatible candidate '{candidate_id}': {compatibility.reasons}"
            )

        next_version = self._next_version_number()
        final_metadata = ModelMetadata(
            version=next_version,
            candidate_id=candidate_id,
            parent_version=parent_version,
            dataset=candidate_meta.dataset,
            condition_id=candidate_meta.condition_id,
            n_training_samples=candidate_meta.n_training_samples,
            n_validation_samples=candidate_meta.n_validation_samples,
            per_condition_accuracy_active=candidate_meta.per_condition_accuracy_active,
            per_condition_accuracy_candidate=candidate_meta.per_condition_accuracy_candidate,
            novelty_reference_version=candidate_meta.novelty_reference_version,
            embedding_backbone_hash=candidate_meta.embedding_backbone_hash,
            architecture_signature=candidate_meta.architecture_signature,
            gate_decision=gate_report.decision.value,
            gate_reasons=tuple(gate_report.reasons),
        )

        self._commit_version_dir(next_version, candidate_model, final_metadata)

        # Only after the new version is fully, verifiably committed does the
        # pointer move -- this is the single atomic "activation" instant.
        self._write_pointer_atomic(next_version)
        return next_version

    def rollback(self, target_version: int) -> int:
        """Point the active pointer back at an existing, already-accepted
        version. Never deletes or modifies any version's files.

        Raises:
            ValueError: If target_version does not exist in this registry.
        """
        if target_version not in self.list_versions():
            raise ValueError(f"Cannot roll back to version {target_version}: not in {self.list_versions()}")
        self.load_model_version(target_version)  # verify it is actually loadable before committing
        self._write_pointer_atomic(target_version)
        return target_version

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _commit_version_dir(self, version: int, model: keras.Model, metadata: ModelMetadata) -> None:
        """Write, verify, then atomically rename into place as v{version}/.
        If anything fails, the temp directory is cleaned up and no v{version}/
        directory is ever created -- so the version number is never consumed
        by a failed attempt.
        """
        final_dir = self._version_dir(version)
        tmp_dir = final_dir + ".tmp"
        try:
            self._write_model_dir(tmp_dir, model, metadata)
            self._verify_reload_matches(tmp_dir, model)
            os.replace(tmp_dir, final_dir)
        except Exception:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _write_model_dir(self, dir_path: str, model: keras.Model, metadata: ModelMetadata) -> None:
        os.makedirs(dir_path, exist_ok=True)
        model.save(os.path.join(dir_path, "model.keras"))
        meta_path = os.path.join(dir_path, "metadata.json")
        meta_tmp = meta_path + ".tmp"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(meta_tmp, meta_path)

    def _verify_reload_matches(self, dir_path: str, original_model: keras.Model) -> None:
        reloaded = keras.models.load_model(os.path.join(dir_path, "model.keras"), compile=False)
        for w_original, w_reloaded in zip(original_model.get_weights(), reloaded.get_weights()):
            if not np.array_equal(w_original, w_reloaded):
                raise RuntimeError(
                    f"Persisted model at '{dir_path}' failed integrity verification: "
                    "reloaded weights do not match the in-memory model."
                )
