"""CWRU vibration signal preprocessing, group-aware leakage-safe split, and windowing module.

Performs signal validation, leakage-safe recording-level splitting, train-only z-score
normalization, 2,048-sample non-overlapping windowing, and dataset serialization.
"""

import json
import math
import os
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cwru_pipeline.config import (
    BASELINE_FAULT_CLASSES,
    FIGURE_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    STEP_SIZE,
    TARGET_SPLIT_RATIOS,
    WINDOW_SIZE,
)
from src.cwru_pipeline.dataset import CWRUFileMetadata, discover_and_load_cwru_dataset


def validate_signal(signal: np.ndarray, file_identifier: str) -> np.ndarray:
    """Validate raw vibration time-series array for numerical integrity.

    Args:
        signal: Input 1D vibration array.
        file_identifier: Identifier name for error reporting.

    Returns:
        np.ndarray: Validated float32 array.

    Raises:
        TypeError: If signal is not a numpy array.
        ValueError: If signal is empty, not 1D, contains NaN/Inf, or length < WINDOW_SIZE.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"Signal for '{file_identifier}' must be a numpy array, got {type(signal)}")

    if signal.size == 0:
        raise ValueError(f"Signal for '{file_identifier}' is empty")

    if signal.ndim != 1:
        raise ValueError(
            f"Signal for '{file_identifier}' must be 1D, got shape {signal.shape}"
        )

    val_float = signal.astype(np.float32)

    if np.isnan(val_float).any():
        raise ValueError(f"Signal for '{file_identifier}' contains NaN values")

    if np.isinf(val_float).any():
        raise ValueError(f"Signal for '{file_identifier}' contains Infinite values")

    if len(val_float) < WINDOW_SIZE:
        raise ValueError(
            f"Signal for '{file_identifier}' length ({len(val_float)}) is less than "
            f"minimum required window size ({WINDOW_SIZE})"
        )

    return val_float


def create_leakage_safe_split(
    file_records: List[Tuple[np.ndarray, CWRUFileMetadata]],
    split_ratios: Dict[str, float] = TARGET_SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> Tuple[
    List[Tuple[np.ndarray, CWRUFileMetadata]],
    List[Tuple[np.ndarray, CWRUFileMetadata]],
    List[Tuple[np.ndarray, CWRUFileMetadata]],
]:
    """Perform deterministic recording-level (group-aware) split by source .mat file ID.

    Ensures that windows from the same .mat recording NEVER appear simultaneously across
    training, validation, or test sets.

    Args:
        file_records: List of (signal, CWRUFileMetadata) tuples.
        split_ratios: Target split ratios dictionary (default 70/15/15).
        seed: Random seed for reproducible group shuffling.

    Returns:
        Tuple of (train_records, val_records, test_records).
    """
    np.random.seed(seed)

    # Group file records by fault label to maintain class representation
    class_groups: Dict[int, List[Tuple[np.ndarray, CWRUFileMetadata]]] = {}
    for sig, meta in file_records:
        class_groups.setdefault(meta.fault_label, []).append((sig, meta))

    train_records, val_records, test_records = [], [], []

    for label, records in class_groups.items():
        # Shuffle recordings deterministically per class
        shuffled = records.copy()
        np.random.shuffle(shuffled)

        n_files = len(shuffled)
        if n_files == 1:
            # Single recording fallback to train
            train_records.extend(shuffled)
        elif n_files == 2:
            train_records.append(shuffled[0])
            val_records.append(shuffled[1])
        elif n_files == 3:
            train_records.append(shuffled[0])
            val_records.append(shuffled[1])
            test_records.append(shuffled[2])
        else:
            # Approximate 70 / 15 / 15 split on file count
            n_train = max(1, int(round(n_files * split_ratios["train"])))
            n_val = max(1, int(round(n_files * split_ratios["val"])))
            if n_train + n_val >= n_files:
                n_train = n_files - 2
                n_val = 1

            train_records.extend(shuffled[:n_train])
            val_records.extend(shuffled[n_train : n_train + n_val])
            test_records.extend(shuffled[n_train + n_val :])

    return train_records, val_records, test_records


def fit_train_normalization(
    train_records: List[Tuple[np.ndarray, CWRUFileMetadata]]
) -> Tuple[float, float]:
    """Compute mean and standard deviation strictly from raw training recordings.

    Validation and test recordings NEVER influence normalization statistics to prevent data leakage.

    Args:
        train_records: List of (signal, metadata) tuples in the training split.

    Returns:
        Tuple[float, float]: (train_mean, train_std).
    """
    if not train_records:
        raise ValueError("Cannot fit normalization parameters on empty train_records")

    concat_train = np.concatenate([sig for sig, _ in train_records])
    train_mean = float(np.mean(concat_train))
    train_std = float(np.std(concat_train))

    if train_std == 0.0 or np.isnan(train_std):
        train_std = 1.0

    return train_mean, train_std


def normalize_signal(signal: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize signal array using pre-computed train statistics (z-score normalization).

    Args:
        signal: Input 1D vibration array.
        mean: Training mean.
        std: Training standard deviation.

    Returns:
        np.ndarray: Normalized float32 signal array.
    """
    return ((signal - mean) / std).astype(np.float32)


def window_signal(
    signal: np.ndarray,
    metadata: CWRUFileMetadata,
    split_name: str,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> Tuple[np.ndarray, List[dict], int]:
    """Slice a 1D vibration array into fixed-length windows preserving per-window metadata.

    Incomplete final windows (< window_size samples) are explicitly discarded and logged.

    Args:
        signal: Preprocessed 1D vibration array.
        metadata: Source CWRUFileMetadata object.
        split_name: Split label ("train", "val", or "test").
        window_size: Length of each window (2048).
        step_size: Sliding step size (2048 for non-overlapping).

    Returns:
        Tuple containing:
        - np.ndarray: Windowed array of shape (num_windows, window_size, 1).
        - List[dict]: List of per-window metadata dictionaries.
        - int: Count of discarded tail samples.
    """
    n_samples = len(signal)
    num_windows = (n_samples - window_size) // step_size + 1

    if num_windows <= 0:
        return np.zeros((0, window_size, 1), dtype=np.float32), [], n_samples

    discarded_samples = n_samples - (num_windows * step_size + (window_size - step_size))

    windows_list = []
    window_meta_list = []

    for w_idx in range(num_windows):
        start = w_idx * step_size
        end = start + window_size
        w_data = signal[start:end].reshape(window_size, 1)
        windows_list.append(w_data)

        obs_id = f"cwru_{metadata.file_id:03d}_w{w_idx:04d}_{split_name}"
        w_meta = {
            "observation_id": obs_id,
            "split": split_name,
            "window_index": w_idx,
            "source_file": metadata.source_file,
            "file_id": metadata.file_id,
            "fault_label": metadata.fault_label,
            "fault_type": metadata.fault_type,
            "fault_size": metadata.fault_size,
            "load_hp": metadata.load_hp,
            "rpm": metadata.rpm,
            "sampling_rate": metadata.sampling_rate,
            "sensor_location": metadata.sensor_location,
        }
        window_meta_list.append(w_meta)

    windows_arr = np.array(windows_list, dtype=np.float32)
    return windows_arr, window_meta_list, discarded_samples


def generate_cwru_sample_signals_plot(
    X_train: np.ndarray,
    y_train: np.ndarray,
    figure_path: str = os.path.join(FIGURE_DIR, "cwru_sample_signals.png"),
) -> None:
    """Generate visual plot showing raw/preprocessed vibration windows across fault classes.

    Args:
        X_train: Processed training window array of shape (N, 2048, 1).
        y_train: Training fault label array.
        figure_path: Output PNG figure path.
    """
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    time_axis = np.arange(WINDOW_SIZE) / 12000.0 * 1000.0  # Time in ms

    for label, class_name in BASELINE_FAULT_CLASSES.items():
        indices = np.where(y_train == label)[0]
        if len(indices) > 0:
            sample_idx = indices[0]
            signal = X_train[sample_idx, :, 0]
            axes[label].plot(time_axis, signal, linewidth=1.0, color=f"C{label}")
            axes[label].set_title(
                f"Class {label}: {class_name} (Sample Window index {sample_idx})",
                fontsize=11,
                fontweight="bold",
            )
            axes[label].set_ylabel("Norm. Vib.", fontsize=9)
            axes[label].grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Time (ms) [2048 samples @ 12 kHz]", fontsize=11)
    fig.suptitle(
        "CWRU Preprocessed Vibration Window Baseline Signals Across 4 Classes",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Saved CWRU sample signals plot to: {figure_path}")


def run_cwru_preprocessing_pipeline(
    raw_dir: str = RAW_DATA_DIR,
    processed_dir: str = PROCESSED_DATA_DIR,
    fig_dir: str = FIGURE_DIR,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    seed: int = RANDOM_SEED,
) -> Dict[str, Union[int, float, str, dict]]:
    """Run end-to-end CWRU dataset preprocessing, windowing, and serialization pipeline.

    Args:
        raw_dir: Path to raw CWRU .mat file directory.
        processed_dir: Path to output directory for processed arrays and metadata.
        fig_dir: Directory for diagnostic figures.
        window_size: Window length (2048).
        step_size: Sliding step size (2048).
        seed: Random seed.

    Returns:
        Dict: Machine-readable pipeline summary metadata dictionary.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=== Executing CWRU Dataset Preprocessing Pipeline ===")

    # 1. Discover and load valid baseline recordings
    valid_records, excluded_records = discover_and_load_cwru_dataset(raw_dir=raw_dir)

    if not valid_records:
        raise RuntimeError(
            f"No valid baseline CWRU .mat files found in '{raw_dir}'. "
            f"Please place raw CWRU .mat files in '{raw_dir}'."
        )

    # 2. Validate all loaded signals
    validated_records = []
    for sig, meta in valid_records:
        val_sig = validate_signal(sig, meta.source_file)
        validated_records.append((val_sig, meta))

    # 3. Group-level leakage-safe dataset split (by source .mat recording)
    train_recs, val_recs, test_recs = create_leakage_safe_split(
        validated_records, seed=seed
    )

    # 4. Fit z-score normalization on TRAIN recordings ONLY
    train_mean, train_std = fit_train_normalization(train_recs)
    print(f"Learned Train Normalization Statistics: mean = {train_mean:.6f}, std = {train_std:.6f}")

    # 5. Normalize and window each split
    def process_split(
        records: List[Tuple[np.ndarray, CWRUFileMetadata]], split_name: str
    ):
        all_windows = []
        all_meta = []
        total_discarded = 0

        for raw_sig, meta in records:
            norm_sig = normalize_signal(raw_sig, train_mean, train_std)
            w_arr, w_meta, disc = window_signal(
                norm_sig,
                metadata=meta,
                split_name=split_name,
                window_size=window_size,
                step_size=step_size,
            )
            if len(w_arr) > 0:
                all_windows.append(w_arr)
                all_meta.extend(w_meta)
            total_discarded += disc

        if all_windows:
            X = np.concatenate(all_windows, axis=0)
            y = np.array([m["fault_label"] for m in all_meta], dtype=np.int64)
        else:
            X = np.zeros((0, window_size, 1), dtype=np.float32)
            y = np.zeros((0,), dtype=np.int64)

        return X, y, all_meta, total_discarded

    X_train, y_train, meta_train, disc_train = process_split(train_recs, "train")
    X_val, y_val, meta_val, disc_val = process_split(val_recs, "val")
    X_test, y_test, meta_test, disc_test = process_split(test_recs, "test")

    # 6. Save processed dataset arrays (.npz)
    npz_path = os.path.join(processed_dir, "cwru_dataset_v1.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    print(f"Saved processed dataset arrays to: {npz_path}")

    # 7. Save per-window metadata CSV
    all_window_metadata = meta_train + meta_val + meta_test
    meta_df = pd.DataFrame(all_window_metadata)
    meta_csv_path = os.path.join(processed_dir, "cwru_metadata.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"Saved per-window metadata table to: {meta_csv_path}")

    # 8. Generate sample signals visualization plot
    fig_path = os.path.join(fig_dir, "cwru_sample_signals.png")
    if len(X_train) > 0:
        generate_cwru_sample_signals_plot(X_train, y_train, figure_path=fig_path)

    # 9. Calculate distribution metrics for summary
    total_files = len(validated_records)
    train_files = list({m.source_file for _, m in train_recs})
    val_files = list({m.source_file for _, m in val_recs})
    test_files = list({m.source_file for _, m in test_recs})

    total_windows = len(X_train) + len(X_val) + len(X_test)
    total_discarded_samples = disc_train + disc_val + disc_test

    class_dist = {
        BASELINE_FAULT_CLASSES[c]: int((y_train == c).sum() + (y_val == c).sum() + (y_test == c).sum())
        for c in BASELINE_FAULT_CLASSES.keys()
    }

    summary_dict = {
        "pipeline_version": "v1.0",
        "dataset_name": "CWRU Bearing Vibration Dataset (12k DE Baseline)",
        "window_size": window_size,
        "step_size": step_size,
        "sampling_rate_hz": 12000,
        "preferred_channel": "DE",
        "random_seed": seed,
        "train_mean": train_mean,
        "train_std": train_std,
        "source_files_processed": total_files,
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "total_windows": total_windows,
        "shape_X_train": list(X_train.shape),
        "shape_X_val": list(X_val.shape),
        "shape_X_test": list(X_test.shape),
        "class_distribution_total": class_dist,
        "discarded_tail_samples": total_discarded_samples,
    }

    summary_json_path = os.path.join(processed_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved pipeline summary log to: {summary_json_path}")

    print("=== CWRU Preprocessing Pipeline Execution Complete ===")
    return summary_dict


if __name__ == "__main__":
    run_cwru_preprocessing_pipeline()
