"""Versioned Multi-Prototype Novelty Reference (Task 19 -- Continual Learning
Design, Phase 3).

Extends the *concept* of the existing single-centroid novelty baseline
(src/novelty/novelty.py's DistanceNoveltyDetector) to support MULTIPLE,
independently validated reference conditions, per
docs/continual_learning_design.md Section 3.2:

    "Maintain a versioned, append-only set of reference prototypes ...
    Novelty score for an observation = distance to its NEAREST prototype
    among all accepted prototypes ... A new prototype is added only when
    [an external] Safety Gate and Regression Gate [pass] ... Old prototypes
    are never deleted."

This module does NOT modify, wrap, or replace src/novelty/. The existing
single-centroid DistanceNoveltyDetector remains the baseline CWRU novelty
implementation, unchanged, and continues to be used exactly as before by
src/evaluation/voi_behaviour_analysis.py and everything upstream of it.
This module only computes Euclidean distance to a centroid -- a one-line
computation -- so there is no meaningful normalization/fitting logic to
share with src/novelty/'s min-max scaling (which is deliberately NOT
reproduced here; see "Normalization" below).

Prototype ADMISSION is entirely the caller's responsibility: this module
never creates a prototype on its own initiative. It exposes only an
explicit add_prototype() call. No Condition Monitor alert, pseudo-label,
or high-novelty observation triggers anything here -- that gating is the
future Safety Gate's job (docs/continual_learning_design.md Section 4),
not this task's.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.continual.config import ALLOWED_SPLITS, FORBIDDEN_SPLIT


@dataclass(eq=False)
class Prototype:
    """One validated reference condition's centroid, plus its provenance.

    Not a frozen dataclass (numpy arrays are not hashable, which would
    break dataclass-generated __eq__/__hash__); instead, `centroid` is
    marked read-only (`centroid.flags.writeable = False`) at construction
    so it cannot be mutated in place after the fact. Callers should treat
    every field as immutable by convention.
    """

    prototype_id: str
    centroid: np.ndarray  # shape (embedding_dim,), float64, read-only
    embedding_dim: int
    source_dataset: str
    source_condition: str
    source_split: str
    n_source_embeddings: int
    version_added: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation (centroid rendered as a plain list)."""
        return {
            "prototype_id": self.prototype_id,
            "centroid": self.centroid.tolist(),
            "embedding_dim": self.embedding_dim,
            "source_dataset": self.source_dataset,
            "source_condition": self.source_condition,
            "source_split": self.source_split,
            "n_source_embeddings": self.n_source_embeddings,
            "version_added": self.version_added,
            "extra": dict(self.extra),
        }


def _validate_embeddings(embeddings: np.ndarray, embedding_dim: int, name: str = "embeddings") -> np.ndarray:
    """Validate a 2D embedding batch: correct shape, dimensionality, and finiteness.

    Raises:
        TypeError: If embeddings is not a numpy array.
        ValueError: If embeddings is not 2D, empty, has the wrong last
            dimension, or contains non-finite values.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(embeddings)}")
    if embeddings.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (N, {embedding_dim}), got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"{name} column count ({embeddings.shape[1]}) does not match this reference's "
            f"embedding_dim ({embedding_dim})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    return embeddings


def _validate_single_embedding(embedding: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Validate a single 1D observation embedding.

    Raises:
        TypeError: If embedding is not a numpy array.
        ValueError: If embedding is not 1D, has the wrong length, or
            contains non-finite values.
    """
    if not isinstance(embedding, np.ndarray):
        raise TypeError(f"embedding must be a numpy array, got {type(embedding)}")
    if embedding.ndim != 1:
        raise ValueError(f"embedding must be a 1D array of length {embedding_dim}, got shape {embedding.shape}")
    if embedding.shape[0] != embedding_dim:
        raise ValueError(
            f"embedding length ({embedding.shape[0]}) does not match this reference's embedding_dim ({embedding_dim})"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains NaN or Inf values")
    return embedding


def _validate_split(split: str) -> str:
    """Reject the forbidden ("test") split; require one of the explicitly allowed splits.

    Reuses the exact same split convention as src/continual/adaptation_buffer.py
    (Task 17) rather than redefining "test" as a magic string a second time.
    """
    if not isinstance(split, str):
        raise TypeError(f"source_split must be a string, got {type(split)}")
    normalized = split.strip().lower()
    if normalized == FORBIDDEN_SPLIT:
        raise ValueError(
            f"Rejected: source_split='{split}' is the forbidden evaluation split. "
            "A prototype must never be created from test-split observations."
        )
    if normalized not in ALLOWED_SPLITS:
        raise ValueError(
            f"Unrecognized source_split '{split}'. Allowed splits are {ALLOWED_SPLITS}; "
            f"'{FORBIDDEN_SPLIT}' is explicitly forbidden."
        )
    return split


class NoveltyReference:
    """Versioned, append-only collection of validated novelty reference prototypes.

    version 0                  -> no prototypes yet
    version 1 (add "normal")   -> {normal}
    version 2 (add "cond_b")   -> {normal, cond_b}   (normal unchanged)
    version 3 (add "cond_c")   -> {normal, cond_b, cond_c}

    Nothing in this class ever removes or mutates a previously added
    Prototype. This is the sole mechanism enforcing the append-only
    guarantee -- there is deliberately no `remove_prototype` or
    `update_prototype` method.
    """

    def __init__(self, embedding_dim: int) -> None:
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self._prototypes: List[Prototype] = []
        self._history: List[Dict[str, Any]] = []

    @property
    def version(self) -> int:
        """Current version number: the number of prototypes successfully added so far."""
        return len(self._prototypes)

    @property
    def prototypes(self) -> Tuple[Prototype, ...]:
        """Read-only snapshot of every prototype added so far, in insertion order."""
        return tuple(self._prototypes)

    @property
    def history(self) -> Tuple[Dict[str, Any], ...]:
        """Read-only, append-only audit log: one entry per successful add_prototype() call."""
        return tuple(self._history)

    def prototype_ids(self) -> Tuple[str, ...]:
        """IDs of every prototype currently in the reference, in insertion order."""
        return tuple(p.prototype_id for p in self._prototypes)

    def get_prototype(self, prototype_id: str) -> Prototype:
        """Retrieve a single prototype by id.

        Raises:
            KeyError: If no prototype with that id exists.
        """
        for p in self._prototypes:
            if p.prototype_id == prototype_id:
                return p
        raise KeyError(f"No prototype with id '{prototype_id}' in this reference")

    def add_prototype(
        self,
        prototype_id: str,
        embeddings: np.ndarray,
        source_dataset: str,
        source_condition: str,
        source_split: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Prototype:
        """Append a new prototype, computed as the mean of the given embeddings.

        This is the ONLY way a prototype can be created -- there is no
        automatic path from a Condition Monitor alert, a pseudo-label, or
        any observation's novelty score. The caller must explicitly invoke
        this method with embeddings it has already decided are safe to use
        (that decision belongs to a future Safety Gate, not this module).

        Args:
            prototype_id: Unique identifier for the new prototype (e.g.
                "normal_v1", "condition_b"). Must not already exist in
                this reference.
            embeddings: 2D array of shape (N, embedding_dim), the
                validated embeddings this prototype's centroid is computed
                from. N >= 1.
            source_dataset: Source dataset identifier (e.g. "cwru").
            source_condition: Human-readable label for what this prototype
                represents (e.g. "normal_operation", "new_load_condition").
            source_split: Originating split. Must not be the forbidden
                ("test") split -- enforced here, not left to caller
                discipline.
            extra: Optional free-form metadata dict.

        Returns:
            The newly created Prototype.

        Raises:
            TypeError / ValueError: For invalid embeddings, an
                already-used prototype_id, or a forbidden/unrecognized
                source_split. No prototype is added if validation fails.
        """
        if not isinstance(prototype_id, str) or not prototype_id:
            raise ValueError("prototype_id must be a non-empty string")
        if prototype_id in self.prototype_ids():
            raise ValueError(
                f"prototype_id '{prototype_id}' already exists in this reference "
                "(append-only: duplicate ids are rejected, never overwritten)."
            )
        if not isinstance(source_dataset, str) or not source_dataset:
            raise ValueError("source_dataset must be a non-empty string")
        if not isinstance(source_condition, str) or not source_condition:
            raise ValueError("source_condition must be a non-empty string")

        validated_split = _validate_split(source_split)
        validated_embeddings = _validate_embeddings(embeddings, self.embedding_dim, name="embeddings")

        centroid = np.mean(validated_embeddings, axis=0).astype(np.float64)
        centroid.setflags(write=False)

        new_version = self.version + 1
        prototype = Prototype(
            prototype_id=prototype_id,
            centroid=centroid,
            embedding_dim=self.embedding_dim,
            source_dataset=source_dataset,
            source_condition=source_condition,
            source_split=validated_split,
            n_source_embeddings=int(validated_embeddings.shape[0]),
            version_added=new_version,
            extra=dict(extra) if extra else {},
        )

        # Append only -- no existing entry in self._prototypes is ever touched.
        self._prototypes.append(prototype)
        self._history.append(
            {
                "version": new_version,
                "action": "add_prototype",
                "prototype_id": prototype_id,
                "source_dataset": source_dataset,
                "source_condition": source_condition,
                "source_split": validated_split,
                "n_source_embeddings": prototype.n_source_embeddings,
            }
        )
        return prototype

    def distance_report(self, embedding: np.ndarray) -> Dict[str, float]:
        """Raw Euclidean distance from `embedding` to every known prototype's centroid.

        Deliberately UNNORMALIZED -- see module docstring / docs/multi_prototype_novelty.md
        for why a [0, 1] normalized score is not implemented here.

        Raises:
            ValueError: If this reference has no prototypes yet, or the
                embedding's shape/finiteness is invalid.
        """
        if not self._prototypes:
            raise ValueError("Cannot compute distances: this NoveltyReference has no prototypes yet")
        validated = _validate_single_embedding(embedding, self.embedding_dim)
        return {p.prototype_id: float(np.linalg.norm(validated - p.centroid)) for p in self._prototypes}

    def nearest_prototype(self, embedding: np.ndarray) -> Tuple[Prototype, float]:
        """Find the nearest prototype to `embedding` and its raw distance.

        Deterministic: ties are broken by insertion order (the first
        prototype added, among those tied for minimum distance, wins) --
        Python's min() over a list already guarantees this, and it is
        exercised explicitly by this module's tests.

        Returns:
            (nearest_prototype, raw_euclidean_distance)

        Raises:
            ValueError: If this reference has no prototypes yet, or the
                embedding's shape/finiteness is invalid.
        """
        distances = self.distance_report(embedding)
        nearest_id = min(self.prototype_ids(), key=lambda pid: distances[pid])
        return self.get_prototype(nearest_id), distances[nearest_id]

    def distance_to_nearest(self, embedding: np.ndarray) -> float:
        """Convenience: just the raw distance to the nearest prototype."""
        return self.nearest_prototype(embedding)[1]

    def to_dict(self) -> Dict[str, Any]:
        """Full serializable representation: every prototype plus the audit history."""
        return {
            "embedding_dim": self.embedding_dim,
            "version": self.version,
            "prototypes": [p.to_dict() for p in self._prototypes],
            "history": [dict(h) for h in self._history],
        }

    def save_json(self, path: str) -> None:
        """Persist this reference to a JSON file, losslessly."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "NoveltyReference":
        """Reconstruct a NoveltyReference previously written by save_json(), exactly.

        Reconstructs prototypes via the same append-only add_prototype()
        path is NOT used here (that recomputes a mean from raw embeddings,
        which are not what's persisted) -- instead each prototype's exact
        centroid and metadata are restored directly, and version/history
        are cross-checked for consistency.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ref = cls(embedding_dim=data["embedding_dim"])
        for proto_data in data["prototypes"]:
            centroid = np.array(proto_data["centroid"], dtype=np.float64)
            centroid.setflags(write=False)
            prototype = Prototype(
                prototype_id=proto_data["prototype_id"],
                centroid=centroid,
                embedding_dim=proto_data["embedding_dim"],
                source_dataset=proto_data["source_dataset"],
                source_condition=proto_data["source_condition"],
                source_split=proto_data["source_split"],
                n_source_embeddings=proto_data["n_source_embeddings"],
                version_added=proto_data["version_added"],
                extra=proto_data.get("extra") or {},
            )
            ref._prototypes.append(prototype)
        ref._history = [dict(h) for h in data["history"]]

        if ref.version != data["version"]:
            raise ValueError(
                f"Corrupt reference file: recorded version {data['version']} does not match "
                f"{ref.version} reconstructed prototypes"
            )
        return ref
