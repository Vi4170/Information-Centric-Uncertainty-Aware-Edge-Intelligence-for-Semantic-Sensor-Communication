"""Task 33 -- leakage-safe, modality-specific preprocessing for the
canonical multisensor observation (vibration + temperature + motor_current)
defined in ``src/multimodal_pipeline/observation_schema.py``.

Reuses this repository's existing conventions rather than inventing new
ones: ``window_channel_signal`` and ``apply_normalization`` are imported
unchanged from ``src.ims_pipeline.preprocessing`` (the same functions
Paderborn's own preprocessing already reuses), and train-only global
mean/std normalization mirrors CWRU/Paderborn/XJTU-SY exactly -- one
(mean, std) pair per modality, fit only from train-split raw values.

Modality-specific treatment (Task 33 requirements 1-3):

- **Vibration**: windowed at its own measured, exact 25,600.0 Hz rate into
  (2048, 4) raw-amplitude windows, channel order preserved from the paper
  (x_A, y_A, x_B, y_B) -- unchanged from Task 32.
- **Motor current**: windowed independently at its OWN measured rate
  (~25,608 Hz, read from each file's own metadata, not assumed) into
  (window_size, 3) raw-amplitude windows, kept as a separate array from
  vibration at every step -- never concatenated into vibration's channel
  dimension. Its 3 channels are labeled U/V/W-phase per the published
  paper (not self-described in the raw file -- flagged, same as Task 32's
  vibration channel-order caveat).
- **Temperature**: NOT windowed at its native ADC rate. Task 32 confirmed
  temperature and motor_current are digitized by the same NI module at the
  same high rate (~25,608 Hz) purely because they share one multiplexed
  acquisition chassis -- that is an acquisition-hardware artifact, not
  evidence that temperature carries high-frequency *information*. A
  thermocouple's physical time constant is far slower than an 80ms window,
  so storing 2049 nearly-identical raw samples per window would be
  "artificially high-frequency" in exactly the sense Task 33 prohibits.
  Instead, each temperature window is reduced to its per-window MEAN per
  channel -- computed from real, measured samples over the window's actual
  elapsed-time interval (no interpolation, no fabricated values), which
  both respects the physical bandwidth of temperature and satisfies
  "preserve the actual physical time interval represented by each
  window" precisely. This interpretation is stated explicitly here rather
  than silently assumed; see the Task 33 final report for the same note.

Genuine Task 32 defect found and fixed while implementing this module:
``observation_schema._n_windows_for_duration`` used floor division on raw
floats (``duration_seconds // WINDOW_DURATION_SECONDS``), which
undercounts by exactly one window per condition due to binary
floating-point representation (``300.0 // 0.08 == 3749.0`` even though
``300.0 / 0.08 == 3750.0`` exactly) -- confirmed against real vibration
file lengths. Fixed with a round() there; observation_schema's own tests
were updated to match. Motor_current/temperature genuinely DO support one
fewer window than vibration for the same condition in some cases (their
real, measured sample rate is not identical to vibration's) -- this is a
real physical difference, not a bug, and is handled here as a legitimate
missing-modality case (see ``condition_window_availability``), never
padded or interpolated.

Efficiency (Task 33 requirement 11): no bulk processed-window array is
materialized or saved. Normalization is fit by streaming through each
train-split condition's raw file exactly once (bounded memory: one file's
arrays at a time, never all train data concatenated). Only the small
fitted (mean, std) triple per modality is persisted to disk. Per-condition
window arrays are built on demand by the ``build_*_windows_for_condition``
functions below, for a caller (Task 34) to consume incrementally --
mirroring the existing ``load_stream_windows`` pattern already used by
CWRU/Paderborn/XJTU-SY/IMS.

src/voi/ is not imported or modified anywhere in this module.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.io import loadmat

from src.ims_pipeline.preprocessing import apply_normalization, window_channel_signal
from src.multimodal_pipeline import observation_schema as schema

DATASET_NAME: str = "multimodal"

NORMALIZATION_PARAMS_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_normalization_params.json"
)

TEMPERATURE_CHANNEL_NAMES: Tuple[str, ...] = ("Temperature_housing_A", "Temperature_housing_B")
MOTOR_CURRENT_CHANNEL_NAMES: Tuple[str, ...] = ("U-phase", "V-phase", "W-phase")
TEMPERATURE_CURRENT_CHANNEL_NAME_SOURCE: str = (
    "Published paper (Jung et al., 2023) only. The raw .tdms channels are "
    "named generically by NI device/module/AI-index (e.g. "
    "'cDAQ9185-1F486B5Mod1/ai0'), not by housing or phase -- channel "
    "identity within each physical type (Temperature vs Current) is "
    "assigned here by acquisition order, matching the paper's documented "
    "column order. Not independently verifiable from file metadata alone, "
    "same caveat as Task 32's vibration channel order."
)

TEMPERATURE_SUMMARY_STAT: str = "mean"


# ---------------------------------------------------------------------------
# Raw file readers (read-only -- raw data is never modified)
# ---------------------------------------------------------------------------

def _read_vibration_mat(path: str) -> Dict[str, np.ndarray]:
    """Read one raw vibration .mat file into {channel_name: 1D float32 array}."""
    m = loadmat(path, simplify_cells=True)
    if "Signal" not in m:
        raise ValueError(f"'{path}' does not contain the expected top-level 'Signal' struct")
    values = np.asarray(m["Signal"]["y_values"]["values"])
    if values.ndim != 2 or values.shape[1] != len(schema.VIBRATION_CHANNEL_ORDER):
        raise ValueError(
            f"'{path}': expected y_values.values with {len(schema.VIBRATION_CHANNEL_ORDER)} "
            f"columns, found shape {values.shape}"
        )
    return {
        name: values[:, i].astype(np.float32)
        for i, name in enumerate(schema.VIBRATION_CHANNEL_ORDER)
    }


def _read_temperature_current_tdms(path: str) -> Dict[str, np.ndarray]:
    """Read one raw temperature+current .tdms file into
    {channel_name: 1D float32 array}, covering both TEMPERATURE_CHANNEL_NAMES
    and MOTOR_CURRENT_CHANNEL_NAMES. Channels are matched to a physical name
    by their own 'DAC~Channel~Type' property (Temperature/Current), then
    assigned in acquisition order -- never by trusting a hardcoded device
    path string, so this is robust to minor path/module numbering changes.
    """
    tdms = TdmsFile.read(path)
    temperature_arrays: List[np.ndarray] = []
    current_arrays: List[np.ndarray] = []
    for group in tdms.groups():
        for ch in group.channels():
            ch_type = ch.properties.get("DAC~Channel~Type")
            if ch_type == "Temperature":
                temperature_arrays.append(ch[:].astype(np.float32))
            elif ch_type == "Current":
                current_arrays.append(ch[:].astype(np.float32))

    if len(temperature_arrays) != len(TEMPERATURE_CHANNEL_NAMES):
        raise ValueError(
            f"'{path}': expected {len(TEMPERATURE_CHANNEL_NAMES)} Temperature channels, "
            f"found {len(temperature_arrays)}"
        )
    if len(current_arrays) != len(MOTOR_CURRENT_CHANNEL_NAMES):
        raise ValueError(
            f"'{path}': expected {len(MOTOR_CURRENT_CHANNEL_NAMES)} Current channels, "
            f"found {len(current_arrays)}"
        )

    result: Dict[str, np.ndarray] = {}
    for name, arr in zip(TEMPERATURE_CHANNEL_NAMES, temperature_arrays):
        result[name] = arr
    for name, arr in zip(MOTOR_CURRENT_CHANNEL_NAMES, current_arrays):
        result[name] = arr
    return result


def _measured_temperature_current_rate_hz(path: str) -> float:
    """The real, per-file measured sampling rate for temperature/current,
    read from that file's own metadata -- never assumed equal to vibration's."""
    tdms = TdmsFile.read(path)
    for group in tdms.groups():
        for ch in group.channels():
            increment = ch.properties.get("wf_increment")
            if increment:
                return 1.0 / float(increment)
    raise ValueError(f"'{path}': no channel with a 'wf_increment' property found")


# ---------------------------------------------------------------------------
# Condition / split lookup (reuses Task 32's discovery + split logic exactly)
# ---------------------------------------------------------------------------

def _n_real_windows(n_samples: int, window_size: int) -> int:
    """Real achievable non-overlapping window count for an array of this
    length -- the same formula window_channel_signal itself uses."""
    if n_samples < window_size:
        return 0
    return (n_samples - window_size) // window_size + 1


def get_condition_registry(raw_dir: str = schema.RAW_DATA_DIR) -> pd.DataFrame:
    """All conditions meeting the minimum-modality requirement, with their
    train/val/test split attached (Task 32's condition-level split,
    unmodified)."""
    conditions = schema.discover_conditions(raw_dir)
    valid = conditions[conditions["meets_minimum_modality_requirement"]].copy()
    split_assignment = schema._compute_condition_split_assignment(valid["condition_code"].tolist())
    valid["split"] = valid["condition_code"].map(split_assignment)
    return valid


def get_condition_row(condition_code: str, raw_dir: str = schema.RAW_DATA_DIR) -> pd.Series:
    registry = get_condition_registry(raw_dir)
    matches = registry[registry["condition_code"] == condition_code]
    if matches.empty:
        raise ValueError(f"Unknown or incomplete condition_code '{condition_code}'")
    return matches.iloc[0]


def condition_window_availability(condition_row: pd.Series) -> Dict[str, object]:
    """Real, measured window counts per modality for one condition -- the
    authoritative source of truth for 'how many windows exist', not Task
    32's precomputed (estimate-based) sample_range columns.

    vibration and motor_current/temperature can legitimately differ by a
    small amount (their real measured sample rates are not identical) --
    this function makes that explicit rather than hiding it.
    """
    vib_channels = _read_vibration_mat(condition_row["vibration_path"])
    vib_n_samples = len(next(iter(vib_channels.values())))
    n_vibration = _n_real_windows(vib_n_samples, schema.WINDOW_SIZE_SAMPLES)

    tc_rate = _measured_temperature_current_rate_hz(condition_row["temperature_current_path"])
    tc_window_size = int(round(schema.WINDOW_DURATION_SECONDS * tc_rate))
    tc_channels = _read_temperature_current_tdms(condition_row["temperature_current_path"])
    tc_n_samples = len(next(iter(tc_channels.values())))
    n_temperature_current = _n_real_windows(tc_n_samples, tc_window_size)

    n_common = min(n_vibration, n_temperature_current)
    return {
        "n_vibration": n_vibration,
        "n_temperature_current": n_temperature_current,
        "tc_window_size_samples": tc_window_size,
        "tc_measured_sampling_rate_hz": tc_rate,
        "n_common": n_common,
        "vibration_only_window_indices": list(range(n_common, n_vibration)),
        "temperature_current_only_window_indices": list(range(n_common, n_temperature_current)),
    }


# ---------------------------------------------------------------------------
# Train-only normalization (streamed, one file at a time)
# ---------------------------------------------------------------------------

def fit_modality_normalization(
    modality: str,
    raw_dir: str = schema.RAW_DATA_DIR,
) -> Dict[str, object]:
    """Fit ONE global (mean, std) for a modality from train-split raw
    values only, streaming through each train condition's raw file exactly
    once (never concatenating all train data into one array). Matches the
    global-mean/std convention already used by CWRU/Paderborn/XJTU-SY/IMS.
    """
    if modality not in schema.CANONICAL_MODALITIES:
        raise ValueError(f"Unknown modality '{modality}'. Expected one of {schema.CANONICAL_MODALITIES}")

    registry = get_condition_registry(raw_dir)
    train_rows = registry[registry["split"] == "train"]
    if train_rows.empty:
        raise ValueError("Cannot fit normalization: no train-split conditions found")

    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for _, row in train_rows.iterrows():
        if modality == "vibration":
            channels = _read_vibration_mat(row["vibration_path"])
        else:
            all_channels = _read_temperature_current_tdms(row["temperature_current_path"])
            names = (
                MOTOR_CURRENT_CHANNEL_NAMES if modality == "motor_current" else TEMPERATURE_CHANNEL_NAMES
            )
            channels = {name: all_channels[name] for name in names}

        for arr in channels.values():
            arr64 = arr.astype(np.float64)
            total_sum += float(arr64.sum())
            total_sumsq += float(np.square(arr64).sum())
            total_count += arr64.size

    if total_count == 0:
        raise ValueError(f"Cannot fit normalization for '{modality}': zero samples encountered")

    mean = total_sum / total_count
    variance = max(total_sumsq / total_count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if std == 0.0 or not np.isfinite(std):
        std = 1.0

    return {
        "modality": modality,
        "mean": float(mean),
        "std": float(std),
        "n_samples": int(total_count),
        "n_train_conditions": int(len(train_rows)),
        "fit_only_on_split": "train",
    }


def fit_all_modality_normalizations(raw_dir: str = schema.RAW_DATA_DIR) -> Dict[str, Dict[str, object]]:
    return {modality: fit_modality_normalization(modality, raw_dir) for modality in schema.CANONICAL_MODALITIES}


def save_normalization_params(
    raw_dir: str = schema.RAW_DATA_DIR,
    path: str = NORMALIZATION_PARAMS_JSON_PATH,
) -> Dict[str, Dict[str, object]]:
    params = fit_all_modality_normalizations(raw_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)
    return params


def load_normalization_params(path: str = NORMALIZATION_PARAMS_JSON_PATH) -> Dict[str, Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normalization params not found at '{path}'. Run save_normalization_params() first.")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Per-condition, per-modality window builders (on-demand, one file read each)
# ---------------------------------------------------------------------------

def _base_metadata_row(condition_row: pd.Series, modality: str, window_index: int) -> dict:
    return {
        "observation_id": schema._observation_id(condition_row["condition_code"], window_index),
        "dataset": DATASET_NAME,
        "source_session_id": condition_row["condition_code"],
        "condition_code": condition_row["condition_code"],
        "modality": modality,
        "load_nm": condition_row["load_nm"],
        "fault_type": condition_row["fault_type"],
        "severity_token": condition_row["severity_token"],
        "split": condition_row["split"],
        "window_index": window_index,
    }


def build_vibration_windows_for_condition(
    condition_row: pd.Series,
    normalization: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Returns (X, metadata_df): X has shape (n_windows, WINDOW_SIZE_SAMPLES,
    len(VIBRATION_CHANNEL_ORDER)). Channel order and metadata are preserved
    exactly as documented in Task 32; normalization (if given) is applied
    only after windowing, never altering the underlying raw file."""
    channels = _read_vibration_mat(condition_row["vibration_path"])
    per_channel_windows = [
        window_channel_signal(channels[name], window_size=schema.WINDOW_SIZE_SAMPLES, step_size=schema.WINDOW_SIZE_SAMPLES)
        for name in schema.VIBRATION_CHANNEL_ORDER
    ]
    n_windows = per_channel_windows[0].shape[0]
    X = np.concatenate(per_channel_windows, axis=-1)  # (n_windows, WINDOW_SIZE_SAMPLES, n_channels)

    if normalization is not None:
        X = apply_normalization(X, normalization["mean"], normalization["std"])

    rows = []
    for window_index in range(n_windows):
        row = _base_metadata_row(condition_row, "vibration", window_index)
        row.update(
            {
                "source_recording_id": condition_row["vibration_filename"],
                "channel_names": list(schema.VIBRATION_CHANNEL_ORDER),
                "channel_order_source": schema.VIBRATION_CHANNEL_ORDER_SOURCE,
                "sampling_rate_hz": schema.VIBRATION_MEASURED_SAMPLING_RATE_HZ,
                "window_size_samples": schema.WINDOW_SIZE_SAMPLES,
                "window_start_seconds": window_index * schema.WINDOW_DURATION_SECONDS,
                "window_end_seconds": (window_index + 1) * schema.WINDOW_DURATION_SECONDS,
                "sample_range": (window_index * schema.WINDOW_SIZE_SAMPLES, (window_index + 1) * schema.WINDOW_SIZE_SAMPLES),
                "normalized": normalization is not None,
            }
        )
        rows.append(row)
    return X, pd.DataFrame(rows)


def build_motor_current_windows_for_condition(
    condition_row: pd.Series,
    normalization: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Returns (X, metadata_df): X has shape (n_windows, tc_window_size,
    len(MOTOR_CURRENT_CHANNEL_NAMES)). Processed entirely independently
    from vibration -- its own file read, own measured sampling rate, own
    window size, never concatenated with vibration's channels. n_windows
    may be one less than vibration's for the same condition; see
    condition_window_availability()."""
    tc_rate = _measured_temperature_current_rate_hz(condition_row["temperature_current_path"])
    tc_window_size = int(round(schema.WINDOW_DURATION_SECONDS * tc_rate))
    channels = _read_temperature_current_tdms(condition_row["temperature_current_path"])

    per_channel_windows = [
        window_channel_signal(channels[name], window_size=tc_window_size, step_size=tc_window_size)
        for name in MOTOR_CURRENT_CHANNEL_NAMES
    ]
    n_windows = per_channel_windows[0].shape[0]
    X = np.concatenate(per_channel_windows, axis=-1)  # (n_windows, tc_window_size, 3)

    if normalization is not None:
        X = apply_normalization(X, normalization["mean"], normalization["std"])

    rows = []
    for window_index in range(n_windows):
        row = _base_metadata_row(condition_row, "motor_current", window_index)
        row.update(
            {
                "source_recording_id": condition_row["temperature_current_filename"],
                "channel_names": list(MOTOR_CURRENT_CHANNEL_NAMES),
                "channel_name_source": TEMPERATURE_CURRENT_CHANNEL_NAME_SOURCE,
                "sampling_rate_hz": tc_rate,
                "window_size_samples": tc_window_size,
                "window_start_seconds": window_index * schema.WINDOW_DURATION_SECONDS,
                "window_end_seconds": (window_index + 1) * schema.WINDOW_DURATION_SECONDS,
                "sample_range": (window_index * tc_window_size, (window_index + 1) * tc_window_size),
                "normalized": normalization is not None,
            }
        )
        rows.append(row)
    return X, pd.DataFrame(rows)


def build_temperature_features_for_condition(
    condition_row: pd.Series,
    normalization: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Returns (X, metadata_df): X has shape (n_windows,
    len(TEMPERATURE_CHANNEL_NAMES)) -- one summary statistic (mean) per
    channel per window, NOT a raw high-rate trace. See module docstring
    for why: temperature's real informative bandwidth is far below its
    ADC digitization rate, so a per-window mean is the physically
    appropriate representation, not an information-losing shortcut."""
    tc_rate = _measured_temperature_current_rate_hz(condition_row["temperature_current_path"])
    tc_window_size = int(round(schema.WINDOW_DURATION_SECONDS * tc_rate))
    channels = _read_temperature_current_tdms(condition_row["temperature_current_path"])

    per_channel_windows = [
        window_channel_signal(channels[name], window_size=tc_window_size, step_size=tc_window_size)
        for name in TEMPERATURE_CHANNEL_NAMES
    ]
    n_windows = per_channel_windows[0].shape[0]
    # (n_windows, tc_window_size, 1) per channel -> mean over the time axis -> (n_windows,)
    per_channel_means = [w[:, :, 0].mean(axis=1) for w in per_channel_windows]
    X = np.stack(per_channel_means, axis=-1).astype(np.float32)  # (n_windows, n_channels)

    if normalization is not None:
        X = apply_normalization(X, normalization["mean"], normalization["std"])

    rows = []
    for window_index in range(n_windows):
        row = _base_metadata_row(condition_row, "temperature", window_index)
        row.update(
            {
                "source_recording_id": condition_row["temperature_current_filename"],
                "channel_names": list(TEMPERATURE_CHANNEL_NAMES),
                "channel_name_source": TEMPERATURE_CURRENT_CHANNEL_NAME_SOURCE,
                "raw_sampling_rate_hz": tc_rate,
                "summary_statistic": TEMPERATURE_SUMMARY_STAT,
                "n_raw_samples_averaged": tc_window_size,
                "window_start_seconds": window_index * schema.WINDOW_DURATION_SECONDS,
                "window_end_seconds": (window_index + 1) * schema.WINDOW_DURATION_SECONDS,
                "physical_time_interval_seconds": schema.WINDOW_DURATION_SECONDS,
                "normalized": normalization is not None,
            }
        )
        rows.append(row)
    return X, pd.DataFrame(rows)


if __name__ == "__main__":
    params = save_normalization_params()
    for modality, p in params.items():
        print(f"{modality}: mean={p['mean']:.6f} std={p['std']:.6f} "
              f"(n_samples={p['n_samples']}, n_train_conditions={p['n_train_conditions']})")
