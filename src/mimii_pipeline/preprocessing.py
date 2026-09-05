"""Leakage-safe, lazy MIMII-DG audio preprocessing.

The implementation is based on the actual downloaded MIMII-DG archive
structure inspected before implementation.

Design principles
-----------------
- Original MIMII-DG ZIP archives remain the raw source of truth.
- Original attribute CSV columns are preserved rather than discarded.
- WAV files are loaded lazily, one recording at a time.
- Audio remains a separate modality from vibration/current datasets.
- Section-level grouping is used as the conservative machine-identity /
  leakage boundary because the downloaded MIMII-DG release does not expose
  a separate physical-machine-ID field.
- Official train/test provenance is retained separately from project split.
- Normalization statistics are fitted from project-train normal recordings only.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import wave
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mimii_dg"

SUMMARY_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mimii_dg"
    / "mimii_dg_dataset_summary.json"
)

WINDOW_SIZE = 2048

EXPECTED_MACHINE_TYPES = (
    "bearing",
    "fan",
    "gearbox",
    "slider",
    "valve",
)

EXPECTED_SECTIONS = (
    "section_00",
    "section_01",
    "section_02",
)

EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_SAMPLE_WIDTH_BYTES = 2
EXPECTED_FRAMES_PER_RECORDING = 160000
EXPECTED_DURATION_SECONDS = 10.0

RANDOM_SEED = 42

# Conservative section-level grouping:
# section 00 -> project train
# section 01 -> project validation
# section 02 -> project test
#
# Official test recordings in sections 00 and 01 remain excluded from the
# model splits so that official test recordings cannot leak into train/val.
SECTION_TO_SPLIT = {
    "section_00": "train",
    "section_01": "val",
    "section_02": "test",
}


_FILENAME_RE = re.compile(
    r"^section_(?P<section>\d+)"
    r"_(?P<domain>source|target)"
    r"_(?P<official_split>train|test)"
    r"_(?P<label>normal|anomaly)"
    r"_(?P<recording_index>\d+)"
    r"(?P<suffix>.*)\.wav$",
    re.IGNORECASE,
)


def _normalise_machine_type(value: str) -> str:
    return value.strip().lower()


def _parse_filename(relative_path: str) -> Dict[str, object]:
    """Parse structured fields from an actual MIMII-DG filename."""
    name = Path(relative_path).name
    match = _FILENAME_RE.match(name)

    if match is None:
        raise ValueError(
            f"Unrecognized MIMII-DG filename format: {relative_path}"
        )

    section = f"section_{int(match.group('section')):02d}"

    return {
        "section": section,
        "domain": match.group("domain").lower(),
        "official_split": match.group("official_split").lower(),
        "label": match.group("label").lower(),
        "recording_index": int(match.group("recording_index")),
        "filename_suffix": match.group("suffix") or "",
    }


def _project_split(section: str, official_split: str) -> str:
    """Assign a leakage-safe project split."""
    if section == "section_02":
        return "test"

    if official_split == "train":
        return SECTION_TO_SPLIT[section]

    return "excluded_official_test"


def _zip_path_for_machine(
    machine_type: str,
    raw_dir: Path,
) -> Path:
    path = raw_dir / f"{machine_type}.zip"

    if not path.is_file():
        raise FileNotFoundError(
            f"MIMII-DG archive not found: {path}"
        )

    return path


def list_machine_types(
    raw_dir: str | Path = RAW_DATA_DIR,
) -> Tuple[str, ...]:
    """Return expected machine archives that are actually present."""
    raw_path = Path(raw_dir)

    found = []

    for machine_type in EXPECTED_MACHINE_TYPES:
        if (raw_path / f"{machine_type}.zip").is_file():
            found.append(machine_type)

    return tuple(found)


def _read_attribute_csv(
    archive: zipfile.ZipFile,
    csv_member: str,
) -> Dict[str, Dict[str, object]]:
    """Read and preserve every original column in an attribute CSV.

    MIMII-DG attribute files are not guaranteed to have identical schemas.
    Some contain:

        file_name,d1p,d1v

    while others can contain additional condition metadata such as:

        file_name,d1p,d1v,d2p,d2v

    Therefore all original columns are preserved.
    """
    raw = archive.read(csv_member).decode(
        "utf-8-sig",
        errors="replace",
    )

    reader = csv.DictReader(io.StringIO(raw))

    fieldnames = [
        str(name).strip()
        for name in (reader.fieldnames or [])
        if name is not None
    ]

    if "file_name" not in fieldnames:
        raise ValueError(
            f"{csv_member} has no file_name column. "
            f"Observed columns: {fieldnames}"
        )

    # Validate condition parameter/value pairing when present.
    lower_fields = {field.lower() for field in fieldnames}

    parameter_columns = sorted(
        name
        for name in fieldnames
        if re.fullmatch(r"d\d+p", name.lower())
    )

    for parameter_column in parameter_columns:
        number_match = re.search(
            r"d(\d+)p",
            parameter_column.lower(),
        )

        if number_match is None:
            continue

        value_column = f"d{number_match.group(1)}v"

        if value_column not in lower_fields:
            raise ValueError(
                f"{csv_member} contains {parameter_column} but no "
                f"matching {value_column}. Columns: {fieldnames}"
            )

    records: Dict[str, Dict[str, object]] = {}

    for row in reader:
        if row is None:
            continue

        filename = str(
            row.get("file_name", "")
        ).strip()

        if not filename:
            raise ValueError(
                f"{csv_member} contains an attribute row without file_name."
            )

        # Preserve every original field exactly as text.
        preserved = {
            field: (
                row.get(field, "").strip()
                if row.get(field) is not None
                else ""
            )
            for field in fieldnames
        }

        records[filename] = preserved

    return records


def _load_all_attributes(
    archive: zipfile.ZipFile,
    machine_type: str,
) -> Tuple[
    Dict[str, Dict[str, object]],
    Dict[str, str],
]:
    """Load all original attribute CSVs for a machine type."""
    attribute_map: Dict[str, Dict[str, object]] = {}
    attribute_source: Dict[str, str] = {}

    csv_members = sorted(
        name
        for name in archive.namelist()
        if name.startswith(f"{machine_type}/attributes_")
        and name.lower().endswith(".csv")
    )

    expected_members = [
        f"{machine_type}/attributes_00.csv",
        f"{machine_type}/attributes_01.csv",
        f"{machine_type}/attributes_02.csv",
    ]

    if csv_members != expected_members:
        raise ValueError(
            f"Unexpected attribute CSV structure for {machine_type}: "
            f"{csv_members}"
        )

    for csv_member in csv_members:
        rows = _read_attribute_csv(
            archive,
            csv_member,
        )

        for filename, metadata in rows.items():
            if filename in attribute_map:
                raise ValueError(
                    "Duplicate metadata entry across attribute files: "
                    f"{filename}"
                )

            attribute_map[filename] = metadata
            attribute_source[filename] = Path(csv_member).name

    return attribute_map, attribute_source


def _extract_condition_fields(
    attribute_row: Dict[str, object],
) -> Dict[str, Optional[str]]:
    """Extract every dNp/dNv pair into explicit normalized columns."""
    output: Dict[str, Optional[str]] = {}

    for key, value in attribute_row.items():
        match = re.fullmatch(
            r"d(\d+)(p|v)",
            key.lower(),
        )

        if match is None:
            continue

        number = match.group(1)
        suffix = match.group(2)

        output[f"condition_{suffix}_{number}"] = (
            str(value).strip()
            if value is not None
            and str(value).strip()
            else None
        )

    return output


def _primary_condition(
    attribute_row: Dict[str, object],
) -> Tuple[Optional[str], Optional[str]]:
    """Return d1p/d1v as the primary condition pair."""
    parameter = attribute_row.get("d1p")
    value = attribute_row.get("d1v")

    parameter_text = (
        str(parameter).strip()
        if parameter is not None
        and str(parameter).strip()
        else None
    )

    value_text = (
        str(value).strip()
        if value is not None
        and str(value).strip()
        else None
    )

    return parameter_text, value_text


def discover_recordings(
    machine_type: Optional[str] = None,
    raw_dir: str | Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Discover recordings without loading audio samples."""
    raw_path = Path(raw_dir)

    machines = (
        (_normalise_machine_type(machine_type),)
        if machine_type is not None
        else list_machine_types(raw_path)
    )

    if not machines:
        raise FileNotFoundError(
            f"No MIMII-DG archives found in {raw_path}"
        )

    rows: List[Dict[str, object]] = []

    for machine in machines:
        if machine not in EXPECTED_MACHINE_TYPES:
            raise ValueError(
                f"Unsupported MIMII-DG machine type: {machine}"
            )

        archive_path = _zip_path_for_machine(
            machine,
            raw_path,
        )

        with zipfile.ZipFile(archive_path) as archive:
            attribute_map, attribute_source = _load_all_attributes(
                archive,
                machine,
            )

            wav_members = sorted(
                name
                for name in archive.namelist()
                if name.startswith(f"{machine}/")
                and name.lower().endswith(".wav")
            )

            if not wav_members:
                raise ValueError(
                    f"No WAV recordings found in {archive_path}"
                )

            for relative_path in wav_members:
                parsed = _parse_filename(
                    relative_path
                )

                section = str(
                    parsed["section"]
                )

                official_split = str(
                    parsed["official_split"]
                )

                if section not in EXPECTED_SECTIONS:
                    raise ValueError(
                        f"Unexpected section {section} "
                        f"in {relative_path}"
                    )

                if relative_path not in attribute_map:
                    raise ValueError(
                        "Recording has no corresponding original "
                        f"attribute row: {relative_path}"
                    )

                raw_attribute = attribute_map[
                    relative_path
                ]

                primary_parameter, primary_value = (
                    _primary_condition(
                        raw_attribute
                    )
                )

                leakage_group_id = (
                    f"{machine}:{section}"
                )

                row: Dict[str, object] = {
                    "machine_type": machine,

                    # MIMII-DG does not expose a separate physical
                    # machine-ID field in the downloaded release.
                    # Do not fabricate one.
                    "machine_id": None,
                    "machine_id_source": (
                        "not_exposed_in_MIMII_DG_release"
                    ),

                    # Conservative leakage boundary supported by
                    # the dataset structure.
                    "leakage_group_id": leakage_group_id,

                    "section": section,
                    "domain": parsed["domain"],
                    "official_split": official_split,
                    "project_split": _project_split(
                        section,
                        official_split,
                    ),
                    "label": parsed["label"],
                    "recording_index": parsed[
                        "recording_index"
                    ],
                    "filename_suffix": parsed[
                        "filename_suffix"
                    ],
                    "condition_parameter": primary_parameter,
                    "condition_value": primary_value,
                    "original_filename": relative_path,
                    "attribute_csv": attribute_source[
                        relative_path
                    ],
                    "archive_path": str(
                        archive_path
                    ),
                    "modality": "audio",

                    # Complete original attribute row.
                    "attribute_metadata": dict(
                        raw_attribute
                    ),
                }

                # Explicitly expose all dNp/dNv fields while still retaining
                # the complete original metadata dictionary above.
                row.update(
                    _extract_condition_fields(
                        raw_attribute
                    )
                )

                rows.append(row)

    metadata = pd.DataFrame(rows)

    if metadata.empty:
        raise ValueError(
            "MIMII-DG discovery produced no recordings."
        )

    return metadata.sort_values(
        [
            "machine_type",
            "section",
            "project_split",
            "domain",
            "official_split",
            "label",
            "recording_index",
            "original_filename",
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_recording_metadata(
    machine_type: Optional[str] = None,
    raw_dir: str | Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Project-compatible metadata builder."""
    return discover_recordings(
        machine_type=machine_type,
        raw_dir=raw_dir,
    )


def read_mimii_wav(
    row: pd.Series | Dict[str, object],
    raw_dir: str | Path = RAW_DATA_DIR,
) -> np.ndarray:
    """Read exactly one recording lazily from its original ZIP archive."""
    machine_type = str(
        row["machine_type"]
    )

    relative_path = str(
        row["original_filename"]
    )

    archive_path = _zip_path_for_machine(
        machine_type,
        Path(raw_dir),
    )

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(
            relative_path,
            "r",
        ) as file_object:
            wav_bytes = file_object.read()

    with wave.open(
        io.BytesIO(wav_bytes),
        "rb",
    ) as wav_reader:
        channels = wav_reader.getnchannels()
        sample_rate = wav_reader.getframerate()
        sample_width = wav_reader.getsampwidth()
        frame_count = wav_reader.getnframes()
        compression = wav_reader.getcomptype()

        if channels != EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected mono audio, got {channels} channels for "
                f"{relative_path}"
            )

        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"Expected {EXPECTED_SAMPLE_RATE} Hz, got "
                f"{sample_rate} Hz for {relative_path}"
            )

        if sample_width != EXPECTED_SAMPLE_WIDTH_BYTES:
            raise ValueError(
                f"Expected 16-bit PCM, got {sample_width * 8}-bit "
                f"for {relative_path}"
            )

        if compression != "NONE":
            raise ValueError(
                f"Unsupported WAV compression {compression} "
                f"for {relative_path}"
            )

        raw = wav_reader.readframes(
            frame_count
        )

    audio = np.frombuffer(
        raw,
        dtype="<i2",
    ).astype(np.float32)

    if audio.size != frame_count:
        raise ValueError(
            f"Decoded sample count mismatch for {relative_path}: "
            f"{audio.size} != {frame_count}"
        )

    return audio


def window_audio(
    audio: np.ndarray,
    window_size: int = WINDOW_SIZE,
    drop_remainder: bool = True,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield fixed-size, non-overlapping mono audio windows."""
    array = np.asarray(
        audio,
        dtype=np.float32,
    )

    if array.ndim != 1:
        raise ValueError(
            f"Expected mono 1-D audio, got shape {array.shape}"
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive"
        )

    full_windows = (
        array.size // window_size
    )

    for index in range(full_windows):
        start = index * window_size
        stop = start + window_size

        yield index, array[start:stop]

    if (
        not drop_remainder
        and array.size % window_size
    ):
        start = (
            full_windows * window_size
        )

        yield full_windows, array[start:]


def _observation_id(
    row: pd.Series | Dict[str, object],
    window_index: int,
) -> str:
    """Create a globally unique, dataset-aware observation ID."""
    return "|".join(
        [
            str(row["machine_type"]),
            str(row["section"]),
            str(row["domain"]),
            str(row["official_split"]),
            str(row["label"]),
            str(row["recording_index"]),
            str(row["original_filename"]),
            str(window_index),
        ]
    )


def load_stream_windows(
    metadata_df: Optional[pd.DataFrame] = None,
    machine_type: Optional[str] = None,
    project_split: Optional[str] = None,
    label: Optional[str] = None,
    raw_dir: str | Path = RAW_DATA_DIR,
    max_recordings: Optional[int] = None,
    window_size: int = WINDOW_SIZE,
) -> Iterator[Tuple[np.ndarray, Dict[str, object]]]:
    """Lazily yield one audio window at a time."""
    metadata = (
        discover_recordings(
            machine_type=machine_type,
            raw_dir=raw_dir,
        )
        if metadata_df is None
        else metadata_df.copy()
    )

    if project_split is not None:
        metadata = metadata[
            metadata["project_split"].eq(
                project_split
            )
        ]

    if label is not None:
        metadata = metadata[
            metadata["label"].eq(label)
        ]

    if machine_type is not None:
        metadata = metadata[
            metadata["machine_type"].eq(
                _normalise_machine_type(
                    machine_type
                )
            )
        ]

    metadata = metadata.sort_values(
        [
            "machine_type",
            "section",
            "domain",
            "official_split",
            "recording_index",
        ],
        kind="stable",
    )

    if max_recordings is not None:
        metadata = metadata.head(
            max_recordings
        )

    for _, row in metadata.iterrows():
        audio = read_mimii_wav(
            row,
            raw_dir=raw_dir,
        )

        for window_index, window in window_audio(
            audio,
            window_size=window_size,
            drop_remainder=True,
        ):
            info = row.to_dict()

            info["window_index"] = (
                window_index
            )

            info["observation_id"] = (
                _observation_id(
                    row,
                    window_index,
                )
            )

            yield window, info


def fit_train_normalization(
    metadata_df: Optional[pd.DataFrame] = None,
    raw_dir: str | Path = RAW_DATA_DIR,
    max_recordings: Optional[int] = None,
) -> Tuple[float, float]:
    """Fit waveform mean/std using project-train normal recordings only."""
    metadata = (
        discover_recordings(
            raw_dir=raw_dir,
        )
        if metadata_df is None
        else metadata_df.copy()
    )

    train_metadata = metadata[
        metadata["project_split"].eq(
            "train"
        )
        & metadata["official_split"].eq(
            "train"
        )
        & metadata["label"].eq(
            "normal"
        )
    ].copy()

    if max_recordings is not None:
        train_metadata = train_metadata.head(
            max_recordings
        )

    if train_metadata.empty:
        raise ValueError(
            "No eligible training recordings for normalization."
        )

    count = 0
    mean = 0.0
    m2 = 0.0

    for _, row in train_metadata.iterrows():
        audio = read_mimii_wav(
            row,
            raw_dir=raw_dir,
        ).astype(np.float64)

        batch_count = int(
            audio.size
        )

        batch_mean = float(
            audio.mean()
        )

        batch_m2 = float(
            (
                (audio - batch_mean) ** 2
            ).sum()
        )

        if count == 0:
            count = batch_count
            mean = batch_mean
            m2 = batch_m2
            continue

        total = (
            count + batch_count
        )

        delta = (
            batch_mean - mean
        )

        mean += (
            delta
            * batch_count
            / total
        )

        m2 += (
            batch_m2
            + delta * delta
            * count
            * batch_count
            / total
        )

        count = total

    if count < 2:
        raise ValueError(
            "Insufficient training samples for normalization."
        )

    variance = (
        m2 / count
    )

    std = math.sqrt(
        max(variance, 0.0)
    )

    if (
        not np.isfinite(mean)
        or not np.isfinite(std)
    ):
        raise ValueError(
            "Non-finite normalization statistics."
        )

    if std <= 0:
        raise ValueError(
            "Training audio standard deviation must be positive."
        )

    return float(mean), float(std)


def apply_normalization(
    audio: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Apply already-fitted train-only normalization."""
    if std <= 0:
        raise ValueError(
            "std must be positive"
        )

    return (
        np.asarray(
            audio,
            dtype=np.float32,
        )
        - mean
    ) / std


def verify_split_disjoint(
    metadata_df: pd.DataFrame,
) -> Dict[str, int]:
    """Verify leakage-group disjointness across project splits."""
    usable = metadata_df[
        metadata_df["project_split"].isin(
            ["train", "val", "test"]
        )
    ].copy()

    groups = (
        usable.groupby(
            "leakage_group_id"
        )["project_split"]
        .nunique()
    )

    leaking_groups = groups[
        groups > 1
    ]

    if not leaking_groups.empty:
        raise AssertionError(
            "Leakage groups cross project splits: "
            f"{list(leaking_groups.index)}"
        )

    return {
        split: int(
            usable["project_split"].eq(
                split
            ).sum()
        )
        for split in [
            "train",
            "val",
            "test",
        ]
    }


def verify_no_recording_crosses_split(
    metadata_df: pd.DataFrame,
) -> bool:
    """Verify that no original recording crosses project splits."""
    usable = metadata_df[
        metadata_df["project_split"].isin(
            ["train", "val", "test"]
        )
    ]

    collisions = (
        usable.groupby(
            "original_filename"
        )["project_split"]
        .nunique()
    )

    leaking_recordings = collisions[
        collisions > 1
    ]

    if not leaking_recordings.empty:
        raise AssertionError(
            "Recording leakage detected: "
            f"{list(leaking_recordings.index)[:10]}"
        )

    return True


def verify_observation_id_uniqueness(
    metadata_df: pd.DataFrame,
    max_recordings: int = 10,
) -> bool:
    """Check observation IDs on a bounded metadata sample."""
    seen = set()

    sample = metadata_df.head(
        max_recordings
    )

    windows_per_recording = (
        EXPECTED_FRAMES_PER_RECORDING
        // WINDOW_SIZE
    )

    for _, row in sample.iterrows():
        for window_index in range(
            windows_per_recording
        ):
            observation_id = (
                _observation_id(
                    row,
                    window_index,
                )
            )

            if observation_id in seen:
                raise AssertionError(
                    f"Duplicate observation ID: "
                    f"{observation_id}"
                )

            seen.add(
                observation_id
            )

    return True


def _value_counts_dict(
    series: pd.Series,
) -> Dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(
            dropna=False
        ).to_dict().items()
    }


def _condition_schema(
    metadata: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Report original condition fields encountered."""
    fields = set()

    for row in metadata[
        "attribute_metadata"
    ]:
        if isinstance(row, dict):
            fields.update(
                row.keys()
            )

    return {
        "original_attribute_fields": sorted(
            fields
        ),
        "explicit_condition_fields": sorted(
            column
            for column in metadata.columns
            if column.startswith(
                "condition_"
            )
        ),
    }


def discover_all_machines_summary(
    raw_dir: str | Path = RAW_DATA_DIR,
) -> Dict[str, object]:
    """Build dataset statistics and provenance."""
    metadata = discover_recordings(
        raw_dir=raw_dir
    )

    split_counts = _value_counts_dict(
        metadata[
            "project_split"
        ]
    )

    label_counts = _value_counts_dict(
        metadata[
            "label"
        ]
    )

    machine_counts = _value_counts_dict(
        metadata[
            "machine_type"
        ]
    )

    section_counts = _value_counts_dict(
        metadata[
            "section"
        ]
    )

    leakage_group_counts = _value_counts_dict(
        metadata[
            "leakage_group_id"
        ]
    )

    machine_summary: Dict[str, object] = {}

    for machine_type in EXPECTED_MACHINE_TYPES:
        subset = metadata[
            metadata["machine_type"].eq(
                machine_type
            )
        ]

        section_summary: Dict[str, object] = {}

        for section in EXPECTED_SECTIONS:
            section_df = subset[
                subset["section"].eq(
                    section
                )
            ]

            condition_pairs = (
                section_df[
                    [
                        "condition_parameter",
                        "condition_value",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    [
                        "condition_parameter",
                        "condition_value",
                    ],
                    kind="stable",
                )
                .to_dict(
                    "records"
                )
            )

            section_summary[
                section
            ] = {
                "recordings_total": int(
                    len(section_df)
                ),
                "official_split_counts": (
                    _value_counts_dict(
                        section_df[
                            "official_split"
                        ]
                    )
                ),
                "project_split_counts": (
                    _value_counts_dict(
                        section_df[
                            "project_split"
                        ]
                    )
                ),
                "label_counts": (
                    _value_counts_dict(
                        section_df[
                            "label"
                        ]
                    )
                ),
                "domain_counts": (
                    _value_counts_dict(
                        section_df[
                            "domain"
                        ]
                    )
                ),
                "conditions_observed": (
                    condition_pairs
                ),
                "attribute_csvs": sorted(
                    section_df[
                        "attribute_csv"
                    ]
                    .unique()
                    .tolist()
                ),
                "leakage_group_ids": sorted(
                    section_df[
                        "leakage_group_id"
                    ]
                    .unique()
                    .tolist()
                ),
            }

        machine_summary[
            machine_type
        ] = {
            "recordings_total": int(
                len(subset)
            ),
            "sections": section_summary,
        }

    windows_per_recording = (
        EXPECTED_FRAMES_PER_RECORDING
        // WINDOW_SIZE
    )

    dropped_tail = (
        EXPECTED_FRAMES_PER_RECORDING
        % WINDOW_SIZE
    )

    # Run leakage checks while creating the summary.
    verify_split_disjoint(
        metadata
    )

    verify_no_recording_crosses_split(
        metadata
    )

    summary = {
        "dataset": "MIMII-DG",

        "modality": "audio",

        "provenance": {
            "source": (
                "Hitachi Ltd. official MIMII-DG release"
            ),
            "zenodo_record": (
                "https://zenodo.org/records/6529888"
            ),
            "doi": (
                "10.5281/zenodo.6529888"
            ),
            "archives": [
                "bearing.zip",
                "fan.zip",
                "gearbox.zip",
                "slider.zip",
                "valve.zip",
            ],
            "license": (
                "CC BY-NC-SA 4.0"
            ),
        },

        "raw_data": {
            "raw_directory": str(
                Path(
                    raw_dir
                ).resolve()
            ),
            "archives_are_preserved": True,
            "original_attribute_csvs_preserved_inside_archives": True,
            "extraction_required_for_pipeline": False,
            "processing_strategy": (
                "lazy per-recording access directly "
                "from original ZIP archives"
            ),
        },

        "dataset_structure": {
            "machine_types": list(
                EXPECTED_MACHINE_TYPES
            ),
            "machine_type_count": len(
                EXPECTED_MACHINE_TYPES
            ),
            "sections": list(
                EXPECTED_SECTIONS
            ),
            "sections_per_machine_type": len(
                EXPECTED_SECTIONS
            ),
            "official_splits_observed": sorted(
                metadata[
                    "official_split"
                ]
                .unique()
                .tolist()
            ),
            "domains_observed": sorted(
                metadata[
                    "domain"
                ]
                .unique()
                .tolist()
            ),
            "labels_observed": sorted(
                metadata[
                    "label"
                ]
                .unique()
                .tolist()
            ),
        },

        "recordings": {
            "total": int(
                len(metadata)
            ),
            "by_machine_type": (
                machine_counts
            ),
            "by_section": (
                section_counts
            ),
            "by_label": (
                label_counts
            ),
            "by_project_split": (
                split_counts
            ),
            "by_leakage_group": (
                leakage_group_counts
            ),
        },

        "audio": {
            "channels": (
                EXPECTED_CHANNELS
            ),
            "sample_rate_hz": (
                EXPECTED_SAMPLE_RATE
            ),
            "sample_width_bytes": (
                EXPECTED_SAMPLE_WIDTH_BYTES
            ),
            "bits_per_sample": (
                EXPECTED_SAMPLE_WIDTH_BYTES
                * 8
            ),
            "frames_per_recording": (
                EXPECTED_FRAMES_PER_RECORDING
            ),
            "duration_seconds": (
                EXPECTED_DURATION_SECONDS
            ),
            "window_size_samples": (
                WINDOW_SIZE
            ),
            "windows_per_recording": (
                windows_per_recording
            ),
            "dropped_tail_samples_per_recording": (
                dropped_tail
            ),
            "windowing": (
                "non-overlapping fixed windows; "
                "incomplete tail dropped"
            ),
        },

        "metadata": {
            "machine_type_preserved": True,
            "machine_id_present_in_release": False,
            "machine_id_preservation_status": (
                "not_exposed; not fabricated"
            ),
            "machine_id_source_field": (
                "machine_id_source"
            ),
            "leakage_group_identifier": (
                "machine_type + section"
            ),
            "section_preserved": True,
            "domain_preserved": True,
            "official_split_preserved": True,
            "recording_index_preserved": True,
            "original_filename_preserved": True,
            "attribute_csv_preserved": True,
            "all_original_attribute_columns_preserved": True,
            "condition_parameter_primary_field": (
                "d1p"
            ),
            "condition_value_primary_field": (
                "d1v"
            ),
            "additional_condition_fields_preserved": True,
            "normal_anomaly_label_preserved": True,
            "audio_modality_separate": True,
        },

        "condition_schema": _condition_schema(
            metadata
        ),

        "split_policy": {
            "strategy": (
                "section-level grouped split"
            ),
            "group_key": [
                "leakage_group_id"
            ],
            "group_definition": (
                "machine_type + section"
            ),
            "mapping": {
                "section_00": "train",
                "section_01": "val",
                "section_02": "test",
            },
            "train_source": (
                "section_00 official training "
                "recordings only"
            ),
            "validation_source": (
                "section_01 official training "
                "recordings only"
            ),
            "test_source": (
                "section_02 recordings, with "
                "official train/test provenance retained"
            ),
            "section_00_official_test": (
                "excluded_official_test"
            ),
            "section_01_official_test": (
                "excluded_official_test"
            ),
            "reason": (
                "prevents leakage between machine-type/"
                "section groups and preserves MIMII-DG "
                "domain structure"
            ),
        },

        "normalization": {
            "method": (
                "global waveform standardization"
            ),
            "fit_scope": (
                "project train split normal recordings only"
            ),
            "validation_used_for_fit": False,
            "test_used_for_fit": False,
            "excluded_official_test_used_for_fit": False,
            "per_recording_adaptation": False,
        },

        "leakage_checks": {
            "leakage_group_disjoint": True,
            "recording_disjoint": True,
            "official_test_not_used_for_train_validation": True,
            "raw_original_split_preserved": True,
        },

        "machines": machine_summary,
    }

    return summary


def save_mimii_dataset_summary(
    raw_dir: str | Path = RAW_DATA_DIR,
    path: str | Path = SUMMARY_JSON_PATH,
) -> Dict[str, object]:
    """Save the MIMII-DG dataset summary JSON."""
    summary = discover_all_machines_summary(
        raw_dir=raw_dir
    )

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    return summary