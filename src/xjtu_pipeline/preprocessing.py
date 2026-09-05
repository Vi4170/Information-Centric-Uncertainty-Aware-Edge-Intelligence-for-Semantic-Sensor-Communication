"""XJTU-SY bearing run-to-failure dataset preprocessing.

This module provides:
- dataset discovery and bearing metadata
- official provenance and failure metadata
- bearing-level leakage-safe train/validation/test splitting
- per-channel vibration windowing
- training-only normalization
- deterministic observation identifiers
- dataset statistics and summary generation

Dataset facts confirmed from the author-provided
``Introduction_to_XJTU-SY_Bearing_Dataset.pdf`` and the actual downloaded
XJTU-SY dataset package:

- 15 LDK UER204 rolling element bearings
- 3 operating conditions
- 5 bearings per operating condition
- 9,216 CSV recordings in total
- 2 vibration channels per recording
- Sampling frequency: 25.6 kHz
- 32,768 samples per recording
- Recording duration: 1.28 seconds
- Recordings approximately every 1 minute
- Failure modes documented at bearing/run level in the author PDF

Provenance:
Biao Wang, Yaguo Lei, Naipeng Li,
"A Hybrid Prognostics Approach for Estimating Remaining Useful Life of
Rolling Element Bearings", IEEE Transactions on Reliability, 2018.
DOI: 10.1109/TR.2018.2882682.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.cnn.config import INPUT_SHAPE


# ---------------------------------------------------------------------------
# Dataset identity and provenance
# ---------------------------------------------------------------------------

DATASET_NAME: str = "xjtu"

RAW_DATA_DIR: str = os.path.join("data", "raw", "xjtu")
PROCESSED_DATA_DIR: str = os.path.join("data", "processed", "xjtu")
SUMMARY_JSON_PATH: str = os.path.join(
    PROCESSED_DATA_DIR,
    "xjtu_dataset_summary.json",
)

OFFICIAL_DATA_SOURCE_NAME: str = (
    "Original XJTU-SY Bearing Dataset distribution by "
    "Xi'an Jiaotong University (XJTU) and Changxing Sumyoung Technology "
    "Co., Ltd. (SY)"
)

OFFICIAL_DATA_SOURCE_DESCRIPTION: str = (
    "Original author-distributed XJTU-SY Bearing Dataset package, "
    "with the included author documentation and multi-part RAR archive."
)

OFFICIAL_AUTHORS: Tuple[str, ...] = (
    "Biao Wang",
    "Yaguo Lei",
    "Naipeng Li",
)

OFFICIAL_CONTACT: str = (
    "Biao Wang (wangbiaoxjtu@outlook.com), "
    "Prof. Yaguo Lei (yaguolei@mail.xjtu.edu.cn)"
)

OFFICIAL_CITATION: str = (
    'Biao Wang, Yaguo Lei, Naipeng Li, Ningbo Li, '
    '"A Hybrid Prognostics Approach for Estimating Remaining Useful Life '
    'of Rolling Element Bearings", IEEE Transactions on Reliability, '
    "2018. DOI: 10.1109/TR.2018.2882682."
)

OFFICIAL_DOCUMENTATION: str = (
    "Introduction_to_XJTU-SY_Bearing_Dataset.pdf"
)

OFFICIAL_DOCUMENTATION_SECTION: str = "Section 2.4"


# ---------------------------------------------------------------------------
# Reproducibility and split configuration
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# Five bearings per condition:
# 3 train + 1 validation + 1 test.
SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")

SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.60,
    "val": 0.20,
    "test": 0.20,
}


# ---------------------------------------------------------------------------
# Sampling and windowing constants
# ---------------------------------------------------------------------------

SAMPLING_RATE_HZ: int = 25600
SAMPLES_PER_FILE: int = 32768

RECORDING_DURATION_SECONDS: float = (
    SAMPLES_PER_FILE / SAMPLING_RATE_HZ
)

SAMPLING_PERIOD_MINUTES: int = 1

WINDOW_SIZE: int = INPUT_SHAPE[0]
STEP_SIZE: int = WINDOW_SIZE

WINDOWS_PER_FILE: int = (
    SAMPLES_PER_FILE // WINDOW_SIZE
)

EXPECTED_TOTAL_FILES: int = 9216


# ---------------------------------------------------------------------------
# Bearing information
# ---------------------------------------------------------------------------

BEARING_TYPE: str = "LDK UER204"

BEARING_PARAMS: Dict[str, object] = {
    "outer_race_diameter_mm": 39.80,
    "inner_race_diameter_mm": 29.30,
    "bearing_mean_diameter_mm": 34.55,
    "ball_diameter_mm": 7.92,
    "number_of_balls": 8,
    "contact_angle_deg": 0,
    "load_rating_static_kN": 6.65,
    "load_rating_dynamic_kN": 12.82,
}

N_BEARINGS_PER_CONDITION: int = 5
EXPECTED_TOTAL_BEARINGS: int = 15


# ---------------------------------------------------------------------------
# Channel / modality information
# ---------------------------------------------------------------------------

CHANNEL_NAMES: Tuple[str, ...] = (
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
)

MODALITY_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "vibration": CHANNEL_NAMES,
}


# ---------------------------------------------------------------------------
# Operating conditions
# ---------------------------------------------------------------------------

OPERATING_CONDITIONS: Tuple[str, ...] = (
    "35Hz12kN",
    "37.5Hz11kN",
    "40Hz10kN",
)

OPERATING_CONDITION_DETAILS: Dict[str, Dict[str, object]] = {
    "35Hz12kN": {
        "rotation_speed_hz": 35.0,
        "rotation_speed_rpm": 2100,
        "radial_force_kN": 12.0,
    },
    "37.5Hz11kN": {
        "rotation_speed_hz": 37.5,
        "rotation_speed_rpm": 2250,
        "radial_force_kN": 11.0,
    },
    "40Hz10kN": {
        "rotation_speed_hz": 40.0,
        "rotation_speed_rpm": 2400,
        "radial_force_kN": 10.0,
    },
}


# ---------------------------------------------------------------------------
# Bearing registry
#
# Failure modes and lifetimes are preserved from the author-provided PDF.
# These are bearing/run-level labels, NOT per-file or per-window labels.
# ---------------------------------------------------------------------------

XJTU_BEARING_REGISTRY: Dict[str, Dict[str, object]] = {
    # 35 Hz / 12 kN
    "Bearing1_1": {
        "operating_condition": "35Hz12kN",
        "n_files": 123,
        "bearing_lifetime": "2 h 3 min",
        "fault_element": "Outer race",
    },
    "Bearing1_2": {
        "operating_condition": "35Hz12kN",
        "n_files": 161,
        "bearing_lifetime": "2 h 41 min",
        "fault_element": "Outer race",
    },
    "Bearing1_3": {
        "operating_condition": "35Hz12kN",
        "n_files": 158,
        "bearing_lifetime": "2 h 38 min",
        "fault_element": "Outer race",
    },
    "Bearing1_4": {
        "operating_condition": "35Hz12kN",
        "n_files": 122,
        "bearing_lifetime": "2 h 2 min",
        "fault_element": "Cage",
    },
    "Bearing1_5": {
        "operating_condition": "35Hz12kN",
        "n_files": 52,
        "bearing_lifetime": "52 min",
        "fault_element": "Inner race and outer race",
    },

    # 37.5 Hz / 11 kN
    "Bearing2_1": {
        "operating_condition": "37.5Hz11kN",
        "n_files": 491,
        "bearing_lifetime": "8 h 11 min",
        "fault_element": "Inner race",
    },
    "Bearing2_2": {
        "operating_condition": "37.5Hz11kN",
        "n_files": 161,
        "bearing_lifetime": "2 h 41 min",
        "fault_element": "Outer race",
    },
    "Bearing2_3": {
        "operating_condition": "37.5Hz11kN",
        "n_files": 533,
        "bearing_lifetime": "8 h 53 min",
        "fault_element": "Cage",
    },
    "Bearing2_4": {
        "operating_condition": "37.5Hz11kN",
        "n_files": 42,
        "bearing_lifetime": "42 min",
        "fault_element": "Outer race",
    },
    "Bearing2_5": {
        "operating_condition": "37.5Hz11kN",
        "n_files": 339,
        "bearing_lifetime": "5 h 39 min",
        "fault_element": "Outer race",
    },

    # 40 Hz / 10 kN
    "Bearing3_1": {
        "operating_condition": "40Hz10kN",
        "n_files": 2538,
        "bearing_lifetime": "42 h 18 min",
        "fault_element": "Outer race",
    },
    "Bearing3_2": {
        "operating_condition": "40Hz10kN",
        "n_files": 2496,
        "bearing_lifetime": "41 h 36 min",
        "fault_element": "Inner race, ball, cage and outer race",
    },
    "Bearing3_3": {
        "operating_condition": "40Hz10kN",
        "n_files": 371,
        "bearing_lifetime": "6 h 11 min",
        "fault_element": "Inner race",
    },
    "Bearing3_4": {
        "operating_condition": "40Hz10kN",
        "n_files": 1515,
        "bearing_lifetime": "25 h 15 min",
        "fault_element": "Inner race",
    },
    "Bearing3_5": {
        "operating_condition": "40Hz10kN",
        "n_files": 114,
        "bearing_lifetime": "1 h 54 min",
        "fault_element": "Outer race",
    },
}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _registry_total_files() -> int:
    """Return the total number of expected CSV files."""
    return int(
        sum(
            int(entry["n_files"])
            for entry in XJTU_BEARING_REGISTRY.values()
        )
    )


# ---------------------------------------------------------------------------
# Bearing-level deterministic split
# ---------------------------------------------------------------------------

def _compute_bearing_split_assignment(
    seed: int = RANDOM_SEED,
) -> Dict[str, str]:
    """Assign complete bearings to train/validation/test.

    Within each operating condition:
        3 bearings -> train
        1 bearing  -> val
        1 bearing  -> test

    The assignment is deterministic for a fixed seed.
    """
    rng = np.random.RandomState(seed)
    assignment: Dict[str, str] = {}

    for condition_index, _condition in enumerate(
        OPERATING_CONDITIONS,
        start=1,
    ):
        bearing_indices = list(
            range(1, N_BEARINGS_PER_CONDITION + 1)
        )
        rng.shuffle(bearing_indices)

        for position, bearing_index in enumerate(
            bearing_indices
        ):
            bearing_id = (
                f"Bearing{condition_index}_{bearing_index}"
            )

            if position < 3:
                split = "train"
            elif position == 3:
                split = "val"
            else:
                split = "test"

            assignment[bearing_id] = split

    return assignment


BEARING_SPLIT_ASSIGNMENT: Dict[str, str] = (
    _compute_bearing_split_assignment()
)


def split_for_bearing(bearing_id: str) -> str:
    """Return the deterministic split assigned to a bearing."""
    if bearing_id not in BEARING_SPLIT_ASSIGNMENT:
        raise ValueError(
            f"Unknown bearing_id '{bearing_id}'. "
            f"Expected one of "
            f"{sorted(BEARING_SPLIT_ASSIGNMENT)}"
        )

    return BEARING_SPLIT_ASSIGNMENT[bearing_id]


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def list_bearing_ids(
    operating_condition: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return sorted bearing IDs, optionally filtered by condition."""
    ids = sorted(XJTU_BEARING_REGISTRY.keys())

    if operating_condition is None:
        return tuple(ids)

    if operating_condition not in OPERATING_CONDITIONS:
        raise ValueError(
            f"Unknown operating_condition "
            f"'{operating_condition}'. "
            f"Expected one of {OPERATING_CONDITIONS}"
        )

    return tuple(
        bearing_id
        for bearing_id in ids
        if XJTU_BEARING_REGISTRY[bearing_id][
            "operating_condition"
        ] == operating_condition
    )


def discover_bearing_files(
    bearing_id: str,
    raw_dir: str = RAW_DATA_DIR,
) -> List[str]:
    """Discover chronologically ordered CSV filenames for one bearing.

    The actual dataset uses integer filenames:
        1.csv, 2.csv, ..., N.csv

    The integer file number is treated as the chronological acquisition
    order, consistent with the author documentation.
    """
    if bearing_id not in XJTU_BEARING_REGISTRY:
        raise ValueError(
            f"Unknown bearing_id '{bearing_id}'"
        )

    registry = XJTU_BEARING_REGISTRY[bearing_id]
    condition = str(
        registry["operating_condition"]
    )

    bearing_dir = os.path.join(
        raw_dir,
        condition,
        bearing_id,
    )

    if not os.path.isdir(bearing_dir):
        raise FileNotFoundError(
            f"XJTU-SY bearing directory not found: "
            f"'{bearing_dir}'."
        )

    csv_files: List[Tuple[int, str]] = []

    for filename in os.listdir(bearing_dir):
        if not filename.lower().endswith(".csv"):
            continue

        stem = filename[:-4]

        try:
            file_number = int(stem)
        except ValueError:
            continue

        csv_files.append(
            (file_number, filename)
        )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under '{bearing_dir}'."
        )

    csv_files.sort(
        key=lambda item: item[0]
    )

    numbers = [
        number
        for number, _filename in csv_files
    ]

    expected_numbers = list(
        range(1, len(numbers) + 1)
    )

    if numbers != expected_numbers:
        raise ValueError(
            f"Unexpected file numbering for "
            f"'{bearing_id}'. "
            f"Expected consecutive numbering "
            f"starting at 1, found first/last "
            f"{numbers[:3]} ... {numbers[-3:]}"
        )

    return [
        filename
        for _number, filename in csv_files
    ]


def _bearing_dir(
    bearing_id: str,
    raw_dir: str = RAW_DATA_DIR,
) -> str:
    """Return the full filesystem path for a bearing directory."""
    if bearing_id not in XJTU_BEARING_REGISTRY:
        raise ValueError(
            f"Unknown bearing_id '{bearing_id}'"
        )

    condition = str(
        XJTU_BEARING_REGISTRY[bearing_id][
            "operating_condition"
        ]
    )

    return os.path.join(
        raw_dir,
        condition,
        bearing_id,
    )


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_xjtu_csv(
    path: str,
) -> np.ndarray:
    """Read one raw XJTU-SY CSV file.

    The actual CSV files contain a header row:
        Horizontal_vibration_signals
        Vertical_vibration_signals

    Returns:
        np.ndarray of shape (32768, 2), dtype float32.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"XJTU-SY CSV file not found: '{path}'"
        )

    frame = pd.read_csv(
        path,
        header=0,
        dtype=np.float64,
    )

    actual_columns = tuple(
        str(column).strip()
        for column in frame.columns
    )

    if actual_columns != CHANNEL_NAMES:
        raise ValueError(
            f"Unexpected CSV columns in '{path}': "
            f"{actual_columns}. "
            f"Expected {CHANNEL_NAMES}."
        )

    data = frame.to_numpy(
        dtype=np.float32,
        copy=False,
    )

    if data.shape != (
        SAMPLES_PER_FILE,
        len(CHANNEL_NAMES),
    ):
        raise ValueError(
            f"Unexpected shape {data.shape} in "
            f"'{path}'. Expected "
            f"({SAMPLES_PER_FILE}, "
            f"{len(CHANNEL_NAMES)})."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"Non-finite values detected in '{path}'."
        )

    return data


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def window_channel_signal(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> np.ndarray:
    """Create fixed-size windows from a single 1D signal."""
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError(
            f"Expected a 1D signal, got shape "
            f"{signal.shape}."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive."
        )

    if step_size <= 0:
        raise ValueError(
            "step_size must be positive."
        )

    if signal.size < window_size:
        return np.empty(
            (0, window_size, 1),
            dtype=np.float32,
        )

    starts = range(
        0,
        signal.size - window_size + 1,
        step_size,
    )

    windows = np.stack(
        [
            signal[
                start:start + window_size
            ]
            for start in starts
        ],
        axis=0,
    )

    return windows.astype(
        np.float32,
        copy=False,
    ).reshape(
        -1,
        window_size,
        1,
    )


# ---------------------------------------------------------------------------
# Observation IDs
# ---------------------------------------------------------------------------

def _observation_id(
    bearing_id: str,
    channel_name: str,
    file_number: int,
    window_index: int,
) -> str:
    """Generate a deterministic observation identifier."""
    return (
        f"xjtu_{bearing_id}_"
        f"{channel_name}_"
        f"f{file_number:04d}_"
        f"w{window_index:02d}"
    )


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_bearing_metadata(
    bearing_id: str,
    channel_index: int,
    raw_dir: str = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Build window-level metadata without loading raw signal arrays."""
    if bearing_id not in XJTU_BEARING_REGISTRY:
        raise ValueError(
            f"Unknown bearing_id '{bearing_id}'"
        )

    if not (
        0 <= channel_index < len(CHANNEL_NAMES)
    ):
        raise ValueError(
            f"channel_index {channel_index} out of range "
            f"[0, {len(CHANNEL_NAMES)})"
        )

    channel_name = CHANNEL_NAMES[
        channel_index
    ]

    registry = XJTU_BEARING_REGISTRY[
        bearing_id
    ]

    split = split_for_bearing(
        bearing_id
    )

    filenames = discover_bearing_files(
        bearing_id,
        raw_dir=raw_dir,
    )

    rows: List[dict] = []

    for chronological_index, filename in enumerate(
        filenames
    ):
        file_number = int(
            os.path.splitext(filename)[0]
        )

        for window_index in range(
            WINDOWS_PER_FILE
        ):
            rows.append(
                {
                    "observation_id": _observation_id(
                        bearing_id,
                        channel_name,
                        file_number,
                        window_index,
                    ),
                    "dataset": DATASET_NAME,
                    "bearing_id": bearing_id,
                    "operating_condition": registry[
                        "operating_condition"
                    ],
                    "channel_name": channel_name,
                    "channel_index": channel_index,
                    "source_file": filename,
                    "file_number": file_number,
                    "chronological_order_index": (
                        chronological_index
                    ),
                    "window_index": window_index,
                    "split": split,
                    "sampling_rate_hz": (
                        SAMPLING_RATE_HZ
                    ),
                    "window_size": WINDOW_SIZE,
                    "bearing_type": BEARING_TYPE,
                    "fault_element": registry[
                        "fault_element"
                    ],
                    "bearing_lifetime": registry[
                        "bearing_lifetime"
                    ],
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# On-demand signal loading + windowing
# ---------------------------------------------------------------------------

def load_stream_windows(
    bearing_id: str,
    channel_index: int,
    raw_dir: str = RAW_DATA_DIR,
    file_indices: Optional[
        Sequence[int]
    ] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load selected CSV files and convert one channel into windows.

    CSV files are processed one at a time. ``file_indices`` allows callers
    to select only specific chronological files from a bearing, which keeps
    large-bearings processing bounded to the requested subset.

    Returns:
        X:
            Array of shape (N, WINDOW_SIZE, 1)
        metadata_df:
            One metadata row per returned window
    """
    if bearing_id not in XJTU_BEARING_REGISTRY:
        raise ValueError(
            f"Unknown bearing_id '{bearing_id}'"
        )

    if not (
        0 <= channel_index < len(CHANNEL_NAMES)
    ):
        raise ValueError(
            f"channel_index {channel_index} out of range "
            f"[0, {len(CHANNEL_NAMES)})"
        )

    channel_name = CHANNEL_NAMES[
        channel_index
    ]

    registry = XJTU_BEARING_REGISTRY[
        bearing_id
    ]

    split = split_for_bearing(
        bearing_id
    )

    filenames = discover_bearing_files(
        bearing_id,
        raw_dir=raw_dir,
    )

    n_files_total = len(filenames)

    if file_indices is None:
        selected_indices = list(
            range(n_files_total)
        )
    else:
        selected_indices = sorted(
            set(
                int(index)
                for index in file_indices
            )
        )

    for index in selected_indices:
        if not (
            0 <= index < n_files_total
        ):
            raise ValueError(
                f"file index {index} out of range "
                f"for bearing '{bearing_id}' with "
                f"{n_files_total} files"
            )

    bearing_directory = _bearing_dir(
        bearing_id,
        raw_dir=raw_dir,
    )

    window_arrays: List[np.ndarray] = []
    metadata_rows: List[dict] = []

    for file_index in selected_indices:
        filename = filenames[file_index]

        file_number = int(
            os.path.splitext(filename)[0]
        )

        path = os.path.join(
            bearing_directory,
            filename,
        )

        data = read_xjtu_csv(path)

        channel_signal = data[
            :, channel_index
        ]

        windows = window_channel_signal(
            channel_signal,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
        )

        for window_index in range(
            windows.shape[0]
        ):
            metadata_rows.append(
                {
                    "observation_id": _observation_id(
                        bearing_id,
                        channel_name,
                        file_number,
                        window_index,
                    ),
                    "dataset": DATASET_NAME,
                    "bearing_id": bearing_id,
                    "operating_condition": registry[
                        "operating_condition"
                    ],
                    "channel_name": channel_name,
                    "channel_index": channel_index,
                    "source_file": filename,
                    "file_number": file_number,
                    "chronological_order_index": (
                        file_index
                    ),
                    "window_index": window_index,
                    "split": split,
                    "sampling_rate_hz": (
                        SAMPLING_RATE_HZ
                    ),
                    "window_size": WINDOW_SIZE,
                    "bearing_type": BEARING_TYPE,
                    "fault_element": registry[
                        "fault_element"
                    ],
                    "bearing_lifetime": registry[
                        "bearing_lifetime"
                    ],
                }
            )

        window_arrays.append(windows)

    if window_arrays:
        X = np.concatenate(
            window_arrays,
            axis=0,
        )
    else:
        X = np.empty(
            (0, WINDOW_SIZE, 1),
            dtype=np.float32,
        )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    return X, metadata_df


# ---------------------------------------------------------------------------
# Training-only normalization
# ---------------------------------------------------------------------------

def fit_train_normalization(
    X: np.ndarray,
    metadata_df: pd.DataFrame,
) -> Tuple[float, float]:
    """Fit z-score statistics using training rows only."""
    if len(X) != len(metadata_df):
        raise ValueError(
            "X and metadata_df must contain the "
            "same number of rows."
        )

    if "split" not in metadata_df.columns:
        raise ValueError(
            "metadata_df must contain a 'split' column."
        )

    train_mask = (
        metadata_df["split"] == "train"
    ).to_numpy()

    if not train_mask.any():
        raise ValueError(
            "Cannot fit normalization: "
            "no train-split rows present."
        )

    train_values = X[train_mask]

    if train_values.size == 0:
        raise ValueError(
            "Cannot fit normalization: "
            "training data is empty."
        )

    mean = float(
        np.mean(train_values)
    )

    std = float(
        np.std(train_values)
    )

    if not np.isfinite(mean):
        raise ValueError(
            "Training mean is not finite."
        )

    if (
        not np.isfinite(std)
        or std == 0.0
    ):
        std = 1.0

    return mean, std


def apply_normalization(
    X: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Apply previously fitted training-only normalization."""
    if not np.isfinite(mean):
        raise ValueError(
            "Normalization mean must be finite."
        )

    if (
        not np.isfinite(std)
        or std == 0.0
    ):
        raise ValueError(
            "Normalization std must be finite and non-zero."
        )

    return (
        np.asarray(X, dtype=np.float32)
        - mean
    ) / std


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_split_disjoint(
    metadata_df: pd.DataFrame,
) -> Dict[str, int]:
    """Verify that no observation ID occurs in multiple splits."""
    ids_by_split = {
        split_name: set(
            metadata_df.loc[
                metadata_df["split"] == split_name,
                "observation_id",
            ]
        )
        for split_name in SPLIT_NAMES
    }

    overlaps = {
        "train_val_overlap": len(
            ids_by_split["train"]
            & ids_by_split["val"]
        ),
        "train_test_overlap": len(
            ids_by_split["train"]
            & ids_by_split["test"]
        ),
        "val_test_overlap": len(
            ids_by_split["val"]
            & ids_by_split["test"]
        ),
    }

    if any(overlaps.values()):
        raise AssertionError(
            f"XJTU-SY split overlap detected: "
            f"{overlaps}"
        )

    return overlaps


def verify_no_bearing_crosses_split(
    metadata_df: pd.DataFrame,
) -> bool:
    """Verify that every bearing appears in exactly one split."""
    required = {
        "bearing_id",
        "split",
    }

    missing = required.difference(
        metadata_df.columns
    )

    if missing:
        raise ValueError(
            f"metadata_df is missing required "
            f"columns: {sorted(missing)}"
        )

    grouped = (
        metadata_df
        .groupby("bearing_id")["split"]
        .nunique()
    )

    offenders = grouped[
        grouped > 1
    ]

    if not offenders.empty:
        raise AssertionError(
            "Bearing(s) cross split boundaries: "
            f"{offenders.index.tolist()}"
        )

    return True


def verify_observation_id_uniqueness(
    metadata_df: pd.DataFrame,
) -> bool:
    """Verify that all observation IDs are unique."""
    if (
        "observation_id"
        not in metadata_df.columns
    ):
        raise ValueError(
            "metadata_df must contain "
            "'observation_id'."
        )

    if not metadata_df[
        "observation_id"
    ].is_unique:
        duplicates = metadata_df.loc[
            metadata_df[
                "observation_id"
            ].duplicated(),
            "observation_id",
        ].tolist()

        raise AssertionError(
            "Duplicate observation_id(s) found: "
            f"{duplicates[:5]}"
        )

    return True


# ---------------------------------------------------------------------------
# Dataset statistics / summary
# ---------------------------------------------------------------------------

def discover_all_bearings_summary(
    raw_dir: str = RAW_DATA_DIR,
) -> Dict[str, object]:
    """Build a comprehensive dataset summary."""
    registry_total_files = (
        _registry_total_files()
    )

    bearings_summary: Dict[
        str, Dict[str, object]
    ] = {}

    total_found_files = 0

    for bearing_id in sorted(
        XJTU_BEARING_REGISTRY
    ):
        registry = XJTU_BEARING_REGISTRY[
            bearing_id
        ]

        bearing_directory = _bearing_dir(
            bearing_id,
            raw_dir=raw_dir,
        )

        available = os.path.isdir(
            bearing_directory
        )

        n_files_found: Optional[int] = None

        if available:
            filenames = discover_bearing_files(
                bearing_id,
                raw_dir=raw_dir,
            )
            n_files_found = len(filenames)
            total_found_files += n_files_found

        bearings_summary[bearing_id] = {
            "operating_condition": registry[
                "operating_condition"
            ],
            "n_files_expected": int(
                registry["n_files"]
            ),
            "n_files_found": n_files_found,
            "raw_data_available": available,
            "bearing_lifetime": registry[
                "bearing_lifetime"
            ],
            "fault_element": registry[
                "fault_element"
            ],
            "split": BEARING_SPLIT_ASSIGNMENT[
                bearing_id
            ],
        }

    return {
        "dataset": DATASET_NAME,
        "data_source_name": (
            OFFICIAL_DATA_SOURCE_NAME
        ),
        "data_source_description": (
            OFFICIAL_DATA_SOURCE_DESCRIPTION
        ),
        "authors": list(
            OFFICIAL_AUTHORS
        ),
        "contact": OFFICIAL_CONTACT,
        "citation": OFFICIAL_CITATION,
        "documentation_file": (
            OFFICIAL_DOCUMENTATION
        ),
        "documentation_section": (
            OFFICIAL_DOCUMENTATION_SECTION
        ),
        "bearing_type": BEARING_TYPE,
        "bearing_params": BEARING_PARAMS,
        "n_bearings_per_condition": (
            N_BEARINGS_PER_CONDITION
        ),
        "n_bearings_total": len(
            XJTU_BEARING_REGISTRY
        ),
        "n_files_total_expected": (
            registry_total_files
        ),
        "n_files_total_found": (
            total_found_files
        ),
        "operating_conditions": list(
            OPERATING_CONDITIONS
        ),
        "operating_condition_details": (
            OPERATING_CONDITION_DETAILS
        ),
        "sampling_rate_hz": (
            SAMPLING_RATE_HZ
        ),
        "sampling_rate_source": (
            "Author-provided "
            "Introduction_to_XJTU-SY_Bearing_Dataset.pdf, "
            "Section 2.4"
        ),
        "samples_per_file": (
            SAMPLES_PER_FILE
        ),
        "recording_duration_seconds": (
            RECORDING_DURATION_SECONDS
        ),
        "sampling_period_minutes": (
            SAMPLING_PERIOD_MINUTES
        ),
        "channel_names": list(
            CHANNEL_NAMES
        ),
        "modality_channels": {
            key: list(value)
            for key, value in MODALITY_CHANNELS.items()
        },
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "windows_per_file": (
            WINDOWS_PER_FILE
        ),
        "expected_windows_total_per_channel": (
            registry_total_files
            * WINDOWS_PER_FILE
        ),
        "expected_windows_total_all_channels": (
            registry_total_files
            * WINDOWS_PER_FILE
            * len(CHANNEL_NAMES)
        ),
        "random_seed": RANDOM_SEED,
        "split_ratios": dict(
            SPLIT_RATIOS
        ),
        "bearing_split_assignment": dict(
            BEARING_SPLIT_ASSIGNMENT
        ),
        "split_methodology": (
            "Bearing-level leakage-safe split. "
            "Within each operating condition, "
            "3 bearings are assigned to train, "
            "1 to validation, and 1 to test using "
            "a deterministic seeded shuffle. "
            "Every CSV recording and every derived "
            "window from a bearing remains in that "
            "bearing's split. No window-level or "
            "file-level random splitting is used."
        ),
        "normalization_methodology": (
            "Z-score normalization is fitted using "
            "training-split rows only. Validation "
            "and test data never contribute to the "
            "fitted mean or standard deviation. "
            "Statistics are channel-specific because "
            "channels are processed separately."
        ),
        "processing_methodology": (
            "Metadata is constructed without loading "
            "signal arrays. Signal loading is performed "
            "on demand, one CSV file at a time, with "
            "optional file_indices selection."
        ),
        "raw_data_available": (
            total_found_files == registry_total_files
        ),
        "bearings": bearings_summary,
    }


def save_xjtu_dataset_summary(
    raw_dir: str = RAW_DATA_DIR,
    path: str = SUMMARY_JSON_PATH,
) -> Dict[str, object]:
    """Generate and save the XJTU-SY dataset summary."""
    summary = discover_all_bearings_summary(
        raw_dir=raw_dir
    )

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
            default=(
                lambda value: (
                    value.item()
                    if isinstance(value, np.generic)
                    else str(value)
                )
            ),
        )

    return summary


if __name__ == "__main__":
    result = save_xjtu_dataset_summary()

    print(
        f"Saved XJTU-SY dataset summary to: "
        f"{SUMMARY_JSON_PATH}"
    )

    print(
        f"Bearings: "
        f"{result['n_bearings_total']}"
    )

    print(
        f"Expected CSV files: "
        f"{result['n_files_total_expected']}"
    )

    print(
        f"Found CSV files: "
        f"{result['n_files_total_found']}"
    )