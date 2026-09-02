"""Adaptation Buffer (Task 17 -- Continual Learning Design, Phase 1).

Implements ONLY the infrastructure that controls which observations are
allowed to enter a *future* continual-learning adaptation process, per
Phase 1 of docs/continual_learning_design.md. This module:

    - accepts observations eligible for future adaptation,
    - retains an observation REFERENCE (id/recording/window index) plus
      metadata -- it never stores or requires the raw sensor array,
    - distinguishes confirmed-label samples from pseudo/unlabelled ones
      (docs/continual_learning_design.md Section 4.2),
    - preserves source recording and dataset identity,
    - preserves split information,
    - explicitly REJECTS (never silently filters) any observation from the
      "test" split,
    - is fully deterministic (insertion-ordered, no randomness),
    - exposes enough metadata (via `extra`) for a future regression/replay
      check to use.

It implements NO learning logic: no condition detection, no novelty
reference updates, no CNN fine-tuning, and it is not wired into the
production VoI pipeline (src/integration/voi_pipeline.py) or into
src/voi/. It is a passive data structure only.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from src.continual.config import ALLOWED_SPLITS, FORBIDDEN_SPLIT


class LabelStatus(str, Enum):
    """Whether an observation's label is human-confirmed or a provisional pseudo-label.

    Per docs/continual_learning_design.md Section 4.2: CONFIRMED labels may
    eventually be used to fine-tune supervised model weights; PSEUDO labels
    may only ever inform unsupervised/self-referential state. This module
    only stores the distinction -- it performs no learning of either kind.
    """

    CONFIRMED = "confirmed"
    PSEUDO = "pseudo"


@dataclass(frozen=True)
class AdaptationRecord:
    """One observation reference retained for possible future adaptation use.

    Attributes:
        observation_id: Unique identifier for the observation (matches the
            source dataset's own id convention, e.g. CWRU's
            'cwru_098_w0000_train').
        dataset: Source dataset identifier (e.g. "cwru").
        split: The originating dataset split ("train" or "val" only --
            "test" is rejected before a record can ever be constructed).
        source_recording_id: Identifier of the physical recording/file the
            observation was windowed from (e.g. CWRU's `file_id`). Required,
            never inferred, so recording-level identity is always traceable.
        label_status: LabelStatus.CONFIRMED or LabelStatus.PSEUDO.
        label: Class label, required when label_status is CONFIRMED (a
            confirmed label with no actual label value is a contradiction
            and is rejected). Optional when PSEUDO.
        window_index: Optional position of this window within its source
            recording (preserves within-recording order, when available).
        extra: Optional free-form metadata dict for future regression/replay
            checks (e.g. novelty score, VoI decision at ingestion time).
            This module never populates or reads `extra` itself.
    """

    observation_id: str
    dataset: str
    split: str
    source_recording_id: str
    label_status: LabelStatus
    label: Optional[Any] = None
    window_index: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict representation (LabelStatus rendered as its string value)."""
        data = asdict(self)
        data["label_status"] = self.label_status.value
        return data


def _validate_split(split: str) -> str:
    """Validate a split string, explicitly rejecting the forbidden ("test") split.

    Args:
        split: Proposed split value.

    Returns:
        The validated split string, unchanged.

    Raises:
        TypeError: If split is not a string.
        ValueError: If split is the forbidden ("test") split, or not one of
            the explicitly allowed splits.
    """
    if not isinstance(split, str):
        raise TypeError(f"split must be a string, got {type(split)}")

    normalized = split.strip().lower()

    if normalized == FORBIDDEN_SPLIT:
        raise ValueError(
            f"Rejected: split='{split}' is the forbidden evaluation split. "
            "Test-split observations must never enter the adaptation buffer."
        )

    if normalized not in ALLOWED_SPLITS:
        raise ValueError(
            f"Unrecognized split '{split}'. Allowed splits are {ALLOWED_SPLITS}; "
            f"'{FORBIDDEN_SPLIT}' is explicitly forbidden."
        )

    return split


def _validate_label_status(
    label_status: LabelStatus, label: Optional[Any]
) -> LabelStatus:
    """Validate the label_status/label combination.

    Raises:
        TypeError: If label_status is not a LabelStatus.
        ValueError: If label_status is CONFIRMED but no label is provided.
    """
    if not isinstance(label_status, LabelStatus):
        raise TypeError(f"label_status must be a LabelStatus, got {type(label_status)}")

    if label_status is LabelStatus.CONFIRMED and label is None:
        raise ValueError(
            "label_status=CONFIRMED requires a non-None label. "
            "A confirmed label with no label value is a contradiction."
        )

    return label_status


class AdaptationBuffer:
    """Deterministic, leakage-safe store of observations eligible for future adaptation.

    This buffer is intentionally minimal: it does not compute anything, fit
    anything, or connect to the CNN, novelty, uncertainty, relevance,
    temporal, communication-cost, or VoI modules. It only records WHICH
    observations are allowed to be considered later, and by what label tier.
    """

    def __init__(self) -> None:
        self._records: "dict[str, AdaptationRecord]" = {}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, observation_id: str) -> bool:
        return observation_id in self._records

    @property
    def confirmed_count(self) -> int:
        """Number of records with a human-confirmed label."""
        return sum(1 for r in self._records.values() if r.label_status is LabelStatus.CONFIRMED)

    @property
    def pseudo_count(self) -> int:
        """Number of records with only a provisional pseudo-label (or none)."""
        return sum(1 for r in self._records.values() if r.label_status is LabelStatus.PSEUDO)

    def add(
        self,
        observation_id: str,
        dataset: str,
        split: str,
        source_recording_id: str,
        label_status: LabelStatus,
        label: Optional[Any] = None,
        window_index: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> AdaptationRecord:
        """Add a single observation to the buffer.

        Args:
            observation_id: Unique identifier for the observation.
            dataset: Source dataset identifier (e.g. "cwru").
            split: Originating split. Must not be the forbidden ("test") split.
            source_recording_id: Identifier of the source recording/file.
            label_status: LabelStatus.CONFIRMED or LabelStatus.PSEUDO.
            label: Required if label_status is CONFIRMED.
            window_index: Optional within-recording window position.
            extra: Optional free-form metadata dict.

        Returns:
            The AdaptationRecord that was added.

        Raises:
            TypeError: For wrong argument types.
            ValueError: If split is the forbidden split, if
                observation_id/dataset/source_recording_id is empty, if
                label_status/label are contradictory, or if observation_id
                is already present in the buffer.
        """
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id must be a non-empty string")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("dataset must be a non-empty string")
        if not isinstance(source_recording_id, str) or not source_recording_id:
            raise ValueError("source_recording_id must be a non-empty string")

        if observation_id in self._records:
            raise ValueError(
                f"observation_id '{observation_id}' is already present in the buffer "
                "(duplicate insertion is rejected, not overwritten)."
            )

        validated_split = _validate_split(split)
        validated_status = _validate_label_status(label_status, label)

        record = AdaptationRecord(
            observation_id=observation_id,
            dataset=dataset,
            split=validated_split,
            source_recording_id=source_recording_id,
            label_status=validated_status,
            label=label,
            window_index=window_index,
            extra=dict(extra) if extra else {},
        )
        self._records[observation_id] = record
        return record

    def add_from_dataframe(
        self,
        df: pd.DataFrame,
        dataset: str,
        label_status: LabelStatus,
        id_column: str = "observation_id",
        split_column: str = "split",
        recording_column: str = "file_id",
        window_index_column: Optional[str] = "window_index",
        label_column: Optional[str] = None,
    ) -> List[AdaptationRecord]:
        """Add multiple observations from a metadata-style DataFrame.

        This call is ATOMIC: if any row's split is the forbidden ("test")
        split, or any row is otherwise invalid, NO rows from this call are
        added (explicit rejection of the whole batch, never a silent
        partial ingest) and a ValueError listing the offending
        observation_ids is raised.

        Args:
            df: DataFrame with at least `id_column`, `split_column`, and
                `recording_column`. Matches the schema of
                data/processed/cwru/cwru_metadata.csv.
            dataset: Source dataset identifier (e.g. "cwru").
            label_status: LabelStatus applied to every row in this batch.
            id_column: Column holding the observation id.
            split_column: Column holding the split value.
            recording_column: Column holding the source recording identifier.
            window_index_column: Optional column holding window index.
            label_column: Optional column holding the label. Required if
                label_status is CONFIRMED.

        Returns:
            The list of AdaptationRecords added.

        Raises:
            ValueError: If any row uses the forbidden split, if a required
                column is missing, or if any observation_id is already
                present in the buffer or duplicated within `df`.
        """
        required_columns = {id_column, split_column, recording_column}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

        if label_status is LabelStatus.CONFIRMED and label_column is None:
            raise ValueError("label_column is required when label_status is CONFIRMED")

        offending = df.loc[df[split_column].astype(str).str.strip().str.lower() == FORBIDDEN_SPLIT, id_column].tolist()
        if offending:
            raise ValueError(
                f"Rejected entire batch: {len(offending)} row(s) use the forbidden "
                f"'{FORBIDDEN_SPLIT}' split (e.g. {offending[:5]}). "
                "No rows from this call were added."
            )

        duplicate_ids = df[df[id_column].duplicated()][id_column].tolist()
        if duplicate_ids:
            raise ValueError(f"Rejected entire batch: duplicate observation_id(s) within the input: {duplicate_ids[:5]}")

        already_present = [oid for oid in df[id_column] if oid in self._records]
        if already_present:
            raise ValueError(
                f"Rejected entire batch: {len(already_present)} observation_id(s) already "
                f"present in the buffer (e.g. {already_present[:5]})."
            )

        # All rows validated -- safe to add.
        added: List[AdaptationRecord] = []
        for _, row in df.iterrows():
            label = row[label_column] if label_column is not None else None
            window_index = int(row[window_index_column]) if window_index_column and window_index_column in df.columns else None
            record = self.add(
                observation_id=str(row[id_column]),
                dataset=dataset,
                split=str(row[split_column]),
                source_recording_id=str(row[recording_column]),
                label_status=label_status,
                label=label,
                window_index=window_index,
            )
            added.append(record)
        return added

    def get_by_recording(self, source_recording_id: str) -> List[AdaptationRecord]:
        """Return all records originating from a given source recording."""
        return [r for r in self._records.values() if r.source_recording_id == source_recording_id]

    def observation_ids(self) -> Set[str]:
        """Return the set of all observation_ids currently in the buffer."""
        return set(self._records.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a DataFrame, in deterministic insertion order."""
        if not self._records:
            return pd.DataFrame(
                columns=["observation_id", "dataset", "split", "source_recording_id", "label_status", "label", "window_index", "extra"]
            )
        return pd.DataFrame([r.to_dict() for r in self._records.values()])

    def verify_no_test_leakage(self, test_observation_ids: Iterable[str]) -> None:
        """Assert that no id in `test_observation_ids` is present in this buffer.

        This is an independent, second safety check beyond the per-insert
        split rejection: it compares against an externally supplied,
        authoritative list of test observation ids (e.g. loaded directly
        from a dataset's metadata) rather than trusting each record's own
        stored `split` field.

        Args:
            test_observation_ids: Iterable of observation ids known to
                belong to a test split.

        Raises:
            AssertionError: If any overlap is found, naming the offending ids.
        """
        overlap = self.observation_ids() & set(test_observation_ids)
        assert not overlap, (
            f"Test-set leakage detected: {len(overlap)} observation(s) present in both "
            f"the adaptation buffer and the supplied test set: {sorted(overlap)[:5]}"
        )

    def save_json(self, path: str) -> None:
        """Persist the buffer to a JSON file, in deterministic insertion order."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._records.values()], f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "AdaptationBuffer":
        """Load a buffer previously written by save_json()."""
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        buffer = cls()
        for row in rows:
            buffer.add(
                observation_id=row["observation_id"],
                dataset=row["dataset"],
                split=row["split"],
                source_recording_id=row["source_recording_id"],
                label_status=LabelStatus(row["label_status"]),
                label=row.get("label"),
                window_index=row.get("window_index"),
                extra=row.get("extra") or {},
            )
        return buffer


def check_no_overlap(buffer_ids: Iterable[str], external_ids: Iterable[str]) -> Set[str]:
    """Pure helper: return the set intersection of two id collections.

    Used to independently verify that an adaptation buffer's observation
    ids share nothing with an external (e.g. test-split) id collection,
    without requiring an AdaptationBuffer instance.
    """
    return set(buffer_ids) & set(external_ids)
