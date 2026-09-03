from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.cnn.config import INPUT_SHAPE
from src.continual.adaptation_buffer import check_no_overlap

DATASET_NAME: str = "ims"
RAW_DATA_DIR: str = os.path.join("data", "raw", "ims")
PROCESSED_DATA_DIR: str = os.path.join("data", "processed", "ims")
SUMMARY_JSON_PATH: str = os.path.join(PROCESSED_DATA_DIR, "ims_dataset_summary.json")

WINDOW_SIZE: int = INPUT_SHAPE[0]
STEP_SIZE: int = WINDOW_SIZE
SAMPLING_RATE_HZ: int = 20000
IMS_RAW_SAMPLES_PER_FILE: int = 20480
WINDOWS_PER_FILE: int = IMS_RAW_SAMPLES_PER_FILE // WINDOW_SIZE
NUM_BEARINGS: int = 4
FILENAME_TIMESTAMP_FORMAT: str = "%Y.%m.%d.%H.%M.%S"

RUN_IDS: Tuple[str, ...] = ("1st_test", "2nd_test", "3rd_test")

RUN_SUBDIR: Dict[str, str] = {
    "1st_test": "1st_test",
    "2nd_test": "2nd_test",
    "3rd_test": os.path.join("4th_test", "txt"),
}

CHANNELS_PER_BEARING: Dict[str, int] = {
    "1st_test": 2,
    "2nd_test": 1,
    "3rd_test": 1,
}

RUN_FAILURE_DESCRIPTIONS: Dict[str, str] = {
    "1st_test": (
        "Vendor readme (IMS Bearing Data, NSF I/UCR Center for Intelligent Maintenance "
        "Systems): at the end of the test-to-failure run, inner race defect occurred in "
        "bearing 3 and roller element defect occurred in bearing 4. Run-level provenance "
        "only, not a per-window label: no timestamped onset is documented."
    ),
    "2nd_test": (
        "Vendor readme: at the end of the test-to-failure run, outer race failure occurred "
        "in bearing 1. Run-level provenance only, not a per-window label: no timestamped "
        "onset is documented."
    ),
    "3rd_test": (
        "Vendor readme describes a 'Set No. 3' of 4,448 files spanning 2004-03-04 to "
        "2004-04-04, ending in an outer race failure in bearing 3. The archive actually "
        "provided for this integration extracts (via its own internal folder name, "
        "'4th_test') to 6,324 files spanning 2004-03-04 to 2004-04-18 -- more files and a "
        "later end date than the readme's Set No. 3 description. This mismatch is preserved "
        "here explicitly rather than silently assumed away: the readme's failure "
        "description is carried as informational provenance only and its exact "
        "applicability to the full extracted run is unverified."
    ),
}

INITIAL_SPLIT_FRACTION: float = 0.2
TEST_SPLIT_FRACTION: float = 0.2

SPLIT_NAMES: Tuple[str, ...] = ("initial", "adaptation", "test")


def parse_ims_filename_timestamp(filename: str) -> datetime:
    return datetime.strptime(filename, FILENAME_TIMESTAMP_FORMAT)


def _run_dir(run_id: str, raw_dir: str) -> str:
    if run_id not in RUN_IDS:
        raise ValueError(f"Unknown run_id '{run_id}'. Expected one of {RUN_IDS}")
    return os.path.join(raw_dir, RUN_SUBDIR[run_id])


def discover_run_files(run_id: str, raw_dir: str = RAW_DATA_DIR) -> List[str]:
    run_dir = _run_dir(run_id, raw_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"IMS run directory not found: '{run_dir}'. Extract the '{run_id}' raw archive "
            f"before calling discover_run_files()."
        )
    filenames = [f for f in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, f))]
    parsed = []
    for filename in filenames:
        try:
            timestamp = parse_ims_filename_timestamp(filename)
        except ValueError:
            continue
        parsed.append((timestamp, filename))
    if not parsed:
        raise FileNotFoundError(f"No timestamp-named IMS snapshot files found under '{run_dir}'.")
    parsed.sort(key=lambda pair: pair[0])
    return [filename for _, filename in parsed]


def compute_chronological_split_boundaries(
    n_files: int,
    initial_fraction: float = INITIAL_SPLIT_FRACTION,
    test_fraction: float = TEST_SPLIT_FRACTION,
) -> Dict[str, Tuple[int, int]]:
    if n_files < 3:
        raise ValueError(
            f"Cannot construct an initial/adaptation/test chronological split from "
            f"{n_files} file(s); at least 3 are required."
        )
    n_initial = max(1, int(n_files * initial_fraction))
    n_test = max(1, int(n_files * test_fraction))
    if n_initial + n_test >= n_files:
        n_test = 1
        n_initial = max(1, n_files - n_test - 1)
    n_adaptation = n_files - n_initial - n_test
    return {
        "initial": (0, n_initial),
        "adaptation": (n_initial, n_initial + n_adaptation),
        "test": (n_initial + n_adaptation, n_files),
    }


def assign_split_for_index(file_index: int, boundaries: Dict[str, Tuple[int, int]]) -> str:
    for split_name, (lo, hi) in boundaries.items():
        if lo <= file_index < hi:
            return split_name
    raise ValueError(f"file_index {file_index} does not fall within any split boundary {boundaries}")


def _bearing_columns(run_id: str, bearing_id: int) -> List[int]:
    if not (1 <= bearing_id <= NUM_BEARINGS):
        raise ValueError(f"bearing_id must be in [1, {NUM_BEARINGS}], got {bearing_id}")
    per_bearing = CHANNELS_PER_BEARING[run_id]
    start = (bearing_id - 1) * per_bearing
    return list(range(start, start + per_bearing))


def channel_count_for_bearing(run_id: str, bearing_id: int) -> int:
    return len(_bearing_columns(run_id, bearing_id))


def _observation_id(run_id: str, bearing_id: int, channel_index: int, file_index: int, window_index: int) -> str:
    return f"ims_{run_id}_b{bearing_id}_c{channel_index}_f{file_index:05d}_w{window_index:02d}"


def build_stream_metadata(
    run_id: str,
    bearing_id: int,
    channel_index: int,
    raw_dir: str = RAW_DATA_DIR,
) -> pd.DataFrame:
    columns = _bearing_columns(run_id, bearing_id)
    if not (0 <= channel_index < len(columns)):
        raise ValueError(
            f"channel_index {channel_index} invalid for run '{run_id}' bearing {bearing_id} "
            f"(this bearing has {len(columns)} channel(s))"
        )
    filenames = discover_run_files(run_id, raw_dir)
    n_files = len(filenames)
    boundaries = compute_chronological_split_boundaries(n_files)

    rows = []
    for file_index, filename in enumerate(filenames):
        split = assign_split_for_index(file_index, boundaries)
        timestamp = parse_ims_filename_timestamp(filename)
        for window_index in range(WINDOWS_PER_FILE):
            rows.append(
                {
                    "observation_id": _observation_id(run_id, bearing_id, channel_index, file_index, window_index),
                    "dataset": DATASET_NAME,
                    "run_id": run_id,
                    "bearing_id": bearing_id,
                    "channel_index": channel_index,
                    "source_file": filename,
                    "file_timestamp": timestamp.isoformat(),
                    "chronological_order_index": file_index,
                    "window_index": window_index,
                    "split": split,
                    "sampling_rate_hz": SAMPLING_RATE_HZ,
                    "window_size": WINDOW_SIZE,
                    "label_available": False,
                    "condition_label": None,
                    "run_failure_description": RUN_FAILURE_DESCRIPTIONS[run_id],
                }
            )
    return pd.DataFrame(rows)


def read_ims_file(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMS raw snapshot file not found: '{path}'")
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"IMS raw snapshot file '{path}' did not parse to a 2D array, got shape {data.shape}")
    return data


def window_channel_signal(channel_signal: np.ndarray, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE) -> np.ndarray:
    n_samples = len(channel_signal)
    num_windows = (n_samples - window_size) // step_size + 1
    if num_windows <= 0:
        return np.zeros((0, window_size, 1), dtype=np.float32)
    windows = np.stack(
        [channel_signal[i * step_size : i * step_size + window_size] for i in range(num_windows)],
        axis=0,
    )
    return windows.reshape(num_windows, window_size, 1).astype(np.float32)


def load_stream_windows(
    run_id: str,
    bearing_id: int,
    channel_index: int,
    raw_dir: str = RAW_DATA_DIR,
    file_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    columns = _bearing_columns(run_id, bearing_id)
    raw_column = columns[channel_index] if 0 <= channel_index < len(columns) else None
    if raw_column is None:
        raise ValueError(
            f"channel_index {channel_index} invalid for run '{run_id}' bearing {bearing_id} "
            f"(this bearing has {len(columns)} channel(s))"
        )

    filenames = discover_run_files(run_id, raw_dir)
    n_files_total = len(filenames)
    boundaries = compute_chronological_split_boundaries(n_files_total)

    selected_indices = sorted(set(range(n_files_total) if file_indices is None else file_indices))
    for idx in selected_indices:
        if not (0 <= idx < n_files_total):
            raise ValueError(f"file index {idx} out of range for run '{run_id}' with {n_files_total} files")

    run_dir = _run_dir(run_id, raw_dir)
    window_arrays = []
    rows = []
    for file_index in selected_indices:
        filename = filenames[file_index]
        data = read_ims_file(os.path.join(run_dir, filename))
        channel_signal = data[:, raw_column]
        windows = window_channel_signal(channel_signal)
        split = assign_split_for_index(file_index, boundaries)
        timestamp = parse_ims_filename_timestamp(filename)
        for window_index in range(windows.shape[0]):
            rows.append(
                {
                    "observation_id": _observation_id(run_id, bearing_id, channel_index, file_index, window_index),
                    "dataset": DATASET_NAME,
                    "run_id": run_id,
                    "bearing_id": bearing_id,
                    "channel_index": channel_index,
                    "source_file": filename,
                    "file_timestamp": timestamp.isoformat(),
                    "chronological_order_index": file_index,
                    "window_index": window_index,
                    "split": split,
                    "sampling_rate_hz": SAMPLING_RATE_HZ,
                    "window_size": WINDOW_SIZE,
                    "label_available": False,
                    "condition_label": None,
                    "run_failure_description": RUN_FAILURE_DESCRIPTIONS[run_id],
                }
            )
        window_arrays.append(windows)

    X = np.concatenate(window_arrays, axis=0) if window_arrays else np.zeros((0, WINDOW_SIZE, 1), dtype=np.float32)
    metadata_df = pd.DataFrame(rows)
    return X, metadata_df


def fit_initial_normalization(X: np.ndarray, metadata_df: pd.DataFrame) -> Tuple[float, float]:
    initial_mask = (metadata_df["split"] == "initial").to_numpy()
    if not initial_mask.any():
        raise ValueError("Cannot fit normalization: no 'initial'-split rows present in metadata_df")
    initial_values = X[initial_mask]
    mean = float(np.mean(initial_values))
    std = float(np.std(initial_values))
    if std == 0.0 or np.isnan(std):
        std = 1.0
    return mean, std


def apply_normalization(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def verify_split_disjoint(metadata_df: pd.DataFrame) -> Dict[str, int]:
    ids_by_split = {
        split_name: set(metadata_df.loc[metadata_df["split"] == split_name, "observation_id"])
        for split_name in SPLIT_NAMES
    }
    overlaps = {
        "initial_adaptation_overlap": len(check_no_overlap(ids_by_split["initial"], ids_by_split["adaptation"])),
        "initial_test_overlap": len(check_no_overlap(ids_by_split["initial"], ids_by_split["test"])),
        "adaptation_test_overlap": len(check_no_overlap(ids_by_split["adaptation"], ids_by_split["test"])),
    }
    if any(overlaps.values()):
        raise AssertionError(f"IMS split overlap detected: {overlaps}")
    return overlaps


def verify_chronological_split_order(metadata_df: pd.DataFrame) -> bool:
    for run_id, group in metadata_df.groupby("run_id"):
        indices_by_split = {
            split_name: group.loc[group["split"] == split_name, "chronological_order_index"]
            for split_name in SPLIT_NAMES
        }
        if indices_by_split["initial"].empty or indices_by_split["test"].empty:
            raise AssertionError(f"Run '{run_id}' is missing an initial or test split allocation")
        if not indices_by_split["adaptation"].empty:
            if indices_by_split["initial"].max() >= indices_by_split["adaptation"].min():
                raise AssertionError(f"Run '{run_id}': initial split does not precede adaptation split")
            if indices_by_split["adaptation"].max() >= indices_by_split["test"].min():
                raise AssertionError(f"Run '{run_id}': adaptation split does not precede test split")
        elif indices_by_split["initial"].max() >= indices_by_split["test"].min():
            raise AssertionError(f"Run '{run_id}': initial split does not precede test split")
    return True


def verify_observation_id_uniqueness(metadata_df: pd.DataFrame) -> bool:
    if not metadata_df["observation_id"].is_unique:
        duplicates = metadata_df.loc[metadata_df["observation_id"].duplicated(), "observation_id"].tolist()
        raise AssertionError(f"Duplicate observation_id(s) found: {duplicates[:5]}")
    return True


def verify_no_test_leakage_into_normalization(metadata_df: pd.DataFrame) -> int:
    initial_ids = set(metadata_df.loc[metadata_df["split"] == "initial", "observation_id"])
    test_ids = set(metadata_df.loc[metadata_df["split"] == "test", "observation_id"])
    overlap = check_no_overlap(initial_ids, test_ids)
    if overlap:
        raise AssertionError(f"Normalization-fitting rows overlap with the permanent test split: {sorted(overlap)[:5]}")
    return len(overlap)


def discover_all_runs_summary(raw_dir: str = RAW_DATA_DIR) -> Dict[str, object]:
    runs_summary = {}
    for run_id in RUN_IDS:
        filenames = discover_run_files(run_id, raw_dir)
        n_files = len(filenames)
        boundaries = compute_chronological_split_boundaries(n_files)
        first_timestamp = parse_ims_filename_timestamp(filenames[0])
        last_timestamp = parse_ims_filename_timestamp(filenames[-1])
        runs_summary[run_id] = {
            "run_subdir": RUN_SUBDIR[run_id],
            "n_files": n_files,
            "first_file": filenames[0],
            "last_file": filenames[-1],
            "first_timestamp": first_timestamp.isoformat(),
            "last_timestamp": last_timestamp.isoformat(),
            "num_bearings": NUM_BEARINGS,
            "channels_per_bearing": CHANNELS_PER_BEARING[run_id],
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "raw_samples_per_file": IMS_RAW_SAMPLES_PER_FILE,
            "window_size": WINDOW_SIZE,
            "windows_per_file": WINDOWS_PER_FILE,
            "split_file_boundaries": {name: list(bounds) for name, bounds in boundaries.items()},
            "split_file_counts": {name: bounds[1] - bounds[0] for name, bounds in boundaries.items()},
            "split_window_counts": {
                name: (bounds[1] - bounds[0]) * WINDOWS_PER_FILE * NUM_BEARINGS * CHANNELS_PER_BEARING[run_id]
                for name, bounds in boundaries.items()
            },
            "failure_description": RUN_FAILURE_DESCRIPTIONS[run_id],
        }
    return {
        "dataset": DATASET_NAME,
        "initial_split_fraction": INITIAL_SPLIT_FRACTION,
        "test_split_fraction": TEST_SPLIT_FRACTION,
        "split_methodology": (
            "Chronological, per-run, file-level split. Each run's timestamp-sorted files are "
            "partitioned into a contiguous leading 'initial' block (baseline/reference), a "
            "contiguous middle 'adaptation' block (future-evidence stream), and a contiguous "
            "trailing 'test' block (permanent, post-hoc-only). No shuffling. No split ever "
            "reorders or mixes files across runs. All channels/bearings sampled from the same "
            "file inherit that file's split, since they share one timestamp."
        ),
        "runs": runs_summary,
    }


def save_ims_dataset_summary(raw_dir: str = RAW_DATA_DIR, path: str = SUMMARY_JSON_PATH) -> Dict[str, object]:
    summary = discover_all_runs_summary(raw_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


if __name__ == "__main__":
    result = save_ims_dataset_summary()
    print(f"Saved IMS dataset summary to: {SUMMARY_JSON_PATH}")
