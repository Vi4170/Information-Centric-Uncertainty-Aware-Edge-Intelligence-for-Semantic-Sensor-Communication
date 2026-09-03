from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.cnn.config import INPUT_SHAPE
from src.continual.adaptation_buffer import check_no_overlap
from src.continual.config import RANDOM_SEED
from src.cwru_pipeline.config import TARGET_SPLIT_RATIOS
from src.ims_pipeline.preprocessing import apply_normalization, window_channel_signal

DATASET_NAME: str = "paderborn"
RAW_DATA_DIR: str = os.path.join("data", "raw", "paderborn")
PROCESSED_DATA_DIR: str = os.path.join("data", "processed", "paderborn")
SUMMARY_JSON_PATH: str = os.path.join(PROCESSED_DATA_DIR, "paderborn_dataset_summary.json")

WINDOW_SIZE: int = INPUT_SHAPE[0]
STEP_SIZE: int = WINDOW_SIZE
SAMPLING_RATE_HZ: int = 64000
SUPPORTING_SAMPLING_RATE_HZ: int = 4000
BEARING_TYPE: str = "6203"

OPERATING_CONDITIONS: Tuple[str, ...] = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
N_MEASUREMENTS_PER_CONDITION: int = 20

VIBRATION_CHANNEL_NAME: str = "vibration_1"
MOTOR_CURRENT_CHANNEL_NAMES: Tuple[str, ...] = ("phase_current_1", "phase_current_2")
SUPPORTING_CHANNEL_NAMES: Tuple[str, ...] = ("force", "speed", "torque", "temp_2_bearing_module")

MODALITY_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "vibration": (VIBRATION_CHANNEL_NAME,),
    "motor_current": MOTOR_CURRENT_CHANNEL_NAMES,
}

SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")

PADERBORN_BEARING_REGISTRY: Dict[str, Dict[str, object]] = {
    "K001": {"health_state": "healthy", "damage_modes": (), "damage_components": (), "damage_methods": ()},
    "K002": {"health_state": "healthy", "damage_modes": (), "damage_components": (), "damage_methods": ()},
    "K003": {"health_state": "healthy", "damage_modes": (), "damage_components": (), "damage_methods": ()},
    "K004": {"health_state": "healthy", "damage_modes": (), "damage_components": (), "damage_methods": ()},
    "K005": {"health_state": "healthy", "damage_modes": (), "damage_components": (), "damage_methods": ()},
    "K006": {"health_state": "healthy", "damage_modes": (), "damage_components": (), "damage_methods": ()},
    "KA01": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("OR",), "damage_methods": ("EDM machining",)},
    "KA03": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("OR",), "damage_methods": ("electric engraver",)},
    "KA04": {"health_state": "damaged", "damage_modes": ("fatigue",), "damage_components": ("OR",), "damage_methods": ("lifetime test",)},
    "KA05": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("OR",), "damage_methods": ("electric engraver",)},
    "KA06": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("OR",), "damage_methods": ("electric engraver",)},
    "KA07": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("OR",), "damage_methods": ("drilled",)},
    "KA08": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("AR",), "damage_methods": ("drilled",)},
    "KA09": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("OR",), "damage_methods": ("drilled",)},
    "KA15": {"health_state": "damaged", "damage_modes": ("plastic deformation",), "damage_components": ("OR",), "damage_methods": ("lifetime test",)},
    "KA16": {"health_state": "damaged", "damage_modes": ("fatigue", "fatigue"), "damage_components": ("OR", "OR"), "damage_methods": ("lifetime test", "lifetime test")},
    "KA22": {"health_state": "damaged", "damage_modes": ("fatigue",), "damage_components": ("OR",), "damage_methods": ("lifetime test",)},
    "KA30": {"health_state": "damaged", "damage_modes": ("plastic deformation",), "damage_components": ("OR",), "damage_methods": ("lifetime test",)},
    "KB23": {"health_state": "damaged", "damage_modes": ("fatigue", "fatigue", "fatigue"), "damage_components": ("IR", "IR", "OR"), "damage_methods": ("lifetime test", "lifetime test", "lifetime test")},
    "KB24": {"health_state": "damaged", "damage_modes": ("fatigue", "plastic deformation"), "damage_components": ("IR", "OR"), "damage_methods": ("lifetime test", "lifetime test")},
    "KB27": {"health_state": "damaged", "damage_modes": ("plastic deformation", "plastic deformation"), "damage_components": ("OR", "IR"), "damage_methods": ("lifetime test", "lifetime test")},
    "KI01": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("IR",), "damage_methods": ("n/a",)},
    "KI03": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("IR",), "damage_methods": ("electric engraver",)},
    "KI04": {"health_state": "damaged", "damage_modes": ("fatigue", "plastic deformation"), "damage_components": ("IR", "OR"), "damage_methods": ("lifetime test", "lifetime test")},
    "KI05": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("IR",), "damage_methods": ("electric engraver",)},
    "KI07": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("IR",), "damage_methods": ("electric engraver",)},
    "KI08": {"health_state": "damaged", "damage_modes": ("artificial",), "damage_components": ("IR",), "damage_methods": ("electric engraver",)},
    "KI14": {"health_state": "damaged", "damage_modes": ("fatigue", "plastic deformation"), "damage_components": ("IR", "OR"), "damage_methods": ("lifetime test", "lifetime test")},
    "KI16": {"health_state": "damaged", "damage_modes": ("fatigue",), "damage_components": ("IR",), "damage_methods": ("lifetime test",)},
    "KI17": {"health_state": "damaged", "damage_modes": ("fatigue", "fatigue"), "damage_components": ("IR", "IR"), "damage_methods": ("lifetime test", "lifetime test")},
    "KI18": {"health_state": "damaged", "damage_modes": ("fatigue",), "damage_components": ("IR",), "damage_methods": ("lifetime test",)},
    "KI21": {"health_state": "damaged", "damage_modes": ("fatigue",), "damage_components": ("IR",), "damage_methods": ("lifetime test",)},
}


def _derive_damage_category(damage_modes: Sequence[str]) -> str:
    if not damage_modes:
        return "n/a"
    mode_set = set(damage_modes)
    if mode_set <= {"artificial"}:
        return "artificial"
    if mode_set <= {"fatigue", "plastic deformation"}:
        return "real"
    raise ValueError(f"Unrecognized or mixed damage mode set {mode_set}; refusing to derive damage_category")


def list_bearing_codes(health_state: Optional[str] = None) -> Tuple[str, ...]:
    codes = sorted(PADERBORN_BEARING_REGISTRY.keys())
    if health_state is None:
        return tuple(codes)
    return tuple(c for c in codes if PADERBORN_BEARING_REGISTRY[c]["health_state"] == health_state)


def _compute_measurement_split_assignment(
    n: int = N_MEASUREMENTS_PER_CONDITION,
    split_ratios: Dict[str, float] = TARGET_SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> Dict[int, str]:
    numbers = list(range(1, n + 1))
    rng = np.random.RandomState(seed)
    shuffled = numbers.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(round(n * split_ratios["train"])))
    n_val = max(1, int(round(n * split_ratios["val"])))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError(f"Degenerate split for n={n}: train={n_train}, val={n_val}, test={n_test}")
    assignment: Dict[int, str] = {}
    for x in shuffled[:n_train]:
        assignment[x] = "train"
    for x in shuffled[n_train : n_train + n_val]:
        assignment[x] = "val"
    for x in shuffled[n_train + n_val :]:
        assignment[x] = "test"
    return assignment


MEASUREMENT_SPLIT_ASSIGNMENT: Dict[int, str] = _compute_measurement_split_assignment()


def split_for_measurement(measurement_number: int) -> str:
    if measurement_number not in MEASUREMENT_SPLIT_ASSIGNMENT:
        raise ValueError(f"measurement_number {measurement_number} has no split assignment")
    return MEASUREMENT_SPLIT_ASSIGNMENT[measurement_number]


def discover_measurement_files(bearing_code: str, operating_condition: str, raw_dir: str = RAW_DATA_DIR) -> List[Tuple[int, str]]:
    if bearing_code not in PADERBORN_BEARING_REGISTRY:
        raise ValueError(f"Unknown bearing_code '{bearing_code}'")
    if operating_condition not in OPERATING_CONDITIONS:
        raise ValueError(f"Unknown operating_condition '{operating_condition}'. Expected one of {OPERATING_CONDITIONS}")
    bearing_dir = os.path.join(raw_dir, bearing_code)
    if not os.path.isdir(bearing_dir):
        raise FileNotFoundError(
            f"Paderborn bearing directory not found: '{bearing_dir}'. Extract the '{bearing_code}' raw archive "
            f"before calling discover_measurement_files()."
        )
    pattern = re.compile(rf"^{re.escape(operating_condition)}_{re.escape(bearing_code)}_(\d+)\.mat$")
    matches = []
    for filename in os.listdir(bearing_dir):
        m = pattern.match(filename)
        if m:
            matches.append((int(m.group(1)), filename))
    if not matches:
        raise FileNotFoundError(
            f"No measurement files matching '{operating_condition}_{bearing_code}_<n>.mat' found under '{bearing_dir}'."
        )
    matches.sort(key=lambda pair: pair[0])
    return matches


def read_paderborn_mat_file(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Paderborn raw measurement file not found: '{path}'")
    mat = loadmat(path)
    data_keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(data_keys) != 1:
        raise ValueError(f"Expected exactly one top-level data variable in '{path}', found {data_keys}")
    top = mat[data_keys[0]][0, 0]
    if top.dtype.names is None or "Y" not in top.dtype.names:
        raise ValueError(f"'{path}' does not contain the expected 'Y' channel struct (fields found: {top.dtype.names})")
    channels = top["Y"][0]
    result: Dict[str, np.ndarray] = {}
    for i in range(channels.shape[0]):
        entry = channels[i]
        name_field = entry["Name"]
        if name_field.size == 0:
            raise ValueError(f"Unnamed channel at index {i} in '{path}'")
        name = str(name_field[0])
        result[name] = np.asarray(entry["Data"]).reshape(-1).astype(np.float64)
    return result


def _observation_id(bearing_code: str, operating_condition: str, measurement_number: int, channel_name: str, window_index: int) -> str:
    return f"paderborn_{bearing_code}_{operating_condition}_m{measurement_number:02d}_{channel_name}_w{window_index:03d}"


def build_measurement_metadata(
    bearing_code: str,
    operating_condition: str,
    modality: str,
    channel_index: int,
    raw_dir: str = RAW_DATA_DIR,
) -> pd.DataFrame:
    if modality not in MODALITY_CHANNELS:
        raise ValueError(f"Unknown modality '{modality}'. Expected one of {tuple(MODALITY_CHANNELS)}")
    channel_names = MODALITY_CHANNELS[modality]
    if not (0 <= channel_index < len(channel_names)):
        raise ValueError(f"channel_index {channel_index} invalid for modality '{modality}' ({len(channel_names)} channel(s))")
    channel_name = channel_names[channel_index]

    measurements = discover_measurement_files(bearing_code, operating_condition, raw_dir)
    reg = PADERBORN_BEARING_REGISTRY[bearing_code]
    damage_category = _derive_damage_category(reg["damage_modes"])

    rows = []
    for measurement_number, filename in measurements:
        rows.append(
            {
                "dataset": DATASET_NAME,
                "bearing_code": bearing_code,
                "bearing_type": BEARING_TYPE,
                "health_state": reg["health_state"],
                "damage_category": damage_category,
                "damage_modes": reg["damage_modes"],
                "damage_components": reg["damage_components"],
                "damage_methods": reg["damage_methods"],
                "operating_condition": operating_condition,
                "measurement_number": measurement_number,
                "modality": modality,
                "channel_name": channel_name,
                "channel_index": channel_index,
                "source_file": filename,
                "split": split_for_measurement(measurement_number),
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "window_size": WINDOW_SIZE,
            }
        )
    return pd.DataFrame(rows)


def load_stream_windows(
    bearing_code: str,
    operating_condition: str,
    modality: str,
    channel_index: int,
    raw_dir: str = RAW_DATA_DIR,
    measurement_numbers: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    measurement_metadata = build_measurement_metadata(bearing_code, operating_condition, modality, channel_index, raw_dir)
    if measurement_numbers is not None:
        selected = sorted(set(measurement_numbers))
        measurement_metadata = measurement_metadata[measurement_metadata["measurement_number"].isin(selected)]
        if measurement_metadata.empty:
            raise ValueError(f"No measurements matched measurement_numbers={measurement_numbers}")

    channel_name = MODALITY_CHANNELS[modality][channel_index]
    bearing_dir = os.path.join(raw_dir, bearing_code)

    window_arrays = []
    rows = []
    for _, meta_row in measurement_metadata.iterrows():
        path = os.path.join(bearing_dir, meta_row["source_file"])
        channels = read_paderborn_mat_file(path)
        if channel_name not in channels:
            raise ValueError(f"Channel '{channel_name}' not found in '{path}' (available: {sorted(channels)})")
        windows = window_channel_signal(channels[channel_name], window_size=WINDOW_SIZE, step_size=STEP_SIZE)
        for window_index in range(windows.shape[0]):
            row = meta_row.to_dict()
            row["window_index"] = window_index
            row["observation_id"] = _observation_id(
                bearing_code, operating_condition, int(meta_row["measurement_number"]), channel_name, window_index
            )
            rows.append(row)
        window_arrays.append(windows)

    X = np.concatenate(window_arrays, axis=0) if window_arrays else np.zeros((0, WINDOW_SIZE, 1), dtype=np.float32)
    metadata_df = pd.DataFrame(rows)
    return X, metadata_df


def fit_train_normalization(X: np.ndarray, metadata_df: pd.DataFrame) -> Tuple[float, float]:
    train_mask = (metadata_df["split"] == "train").to_numpy()
    if not train_mask.any():
        raise ValueError("Cannot fit normalization: no 'train'-split rows present in metadata_df")
    train_values = X[train_mask]
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if std == 0.0 or np.isnan(std):
        std = 1.0
    return mean, std


def verify_split_disjoint(metadata_df: pd.DataFrame) -> Dict[str, int]:
    ids_by_split = {
        split_name: set(metadata_df.loc[metadata_df["split"] == split_name, "observation_id"])
        for split_name in SPLIT_NAMES
    }
    overlaps = {
        "train_val_overlap": len(check_no_overlap(ids_by_split["train"], ids_by_split["val"])),
        "train_test_overlap": len(check_no_overlap(ids_by_split["train"], ids_by_split["test"])),
        "val_test_overlap": len(check_no_overlap(ids_by_split["val"], ids_by_split["test"])),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Paderborn split overlap detected: {overlaps}")
    return overlaps


def verify_no_measurement_crosses_split(metadata_df: pd.DataFrame) -> bool:
    grouped = metadata_df.groupby(["bearing_code", "operating_condition", "measurement_number"])["split"].nunique()
    offenders = grouped[grouped > 1]
    if not offenders.empty:
        raise AssertionError(f"Measurement(s) crossing split boundaries: {offenders.index.tolist()[:5]}")
    return True


def estimate_full_representation_size(
    mean_windows_per_measurement: float = 125.27,
    bytes_per_sample: int = 4,
) -> Dict[str, float]:
    n_bearings = len(PADERBORN_BEARING_REGISTRY)
    n_measurements = n_bearings * len(OPERATING_CONDITIONS) * N_MEASUREMENTS_PER_CONDITION
    n_channels = sum(len(v) for v in MODALITY_CHANNELS.values())
    total_windows = n_measurements * n_channels * mean_windows_per_measurement
    total_bytes = total_windows * WINDOW_SIZE * bytes_per_sample
    return {
        "n_bearings": n_bearings,
        "n_measurements": n_measurements,
        "n_channels_all_modalities": n_channels,
        "estimated_mean_windows_per_measurement": mean_windows_per_measurement,
        "estimated_total_windows_all_modalities": total_windows,
        "estimated_total_bytes_all_modalities": total_bytes,
    }


def discover_all_bearings_summary(raw_dir: str = RAW_DATA_DIR) -> Dict[str, object]:
    bearings_summary = {}
    for bearing_code in list_bearing_codes():
        reg = PADERBORN_BEARING_REGISTRY[bearing_code]
        bearing_dir = os.path.join(raw_dir, bearing_code)
        available = os.path.isdir(bearing_dir)
        n_files = len([f for f in os.listdir(bearing_dir) if f.endswith(".mat")]) if available else None
        bearings_summary[bearing_code] = {
            "health_state": reg["health_state"],
            "damage_category": _derive_damage_category(reg["damage_modes"]),
            "damage_modes": list(reg["damage_modes"]),
            "damage_components": list(reg["damage_components"]),
            "damage_methods": list(reg["damage_methods"]),
            "raw_data_available": available,
            "n_mat_files_found": n_files,
        }
    return {
        "dataset": DATASET_NAME,
        "bearing_type": BEARING_TYPE,
        "n_healthy_states": len(list_bearing_codes("healthy")),
        "n_damaged_states": len(list_bearing_codes("damaged")),
        "operating_conditions": list(OPERATING_CONDITIONS),
        "n_measurements_per_condition": N_MEASUREMENTS_PER_CONDITION,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "supporting_sampling_rate_hz": SUPPORTING_SAMPLING_RATE_HZ,
        "window_size": WINDOW_SIZE,
        "modality_channels": {k: list(v) for k, v in MODALITY_CHANNELS.items()},
        "supporting_channel_names": list(SUPPORTING_CHANNEL_NAMES),
        "measurement_split_assignment": {str(k): v for k, v in MEASUREMENT_SPLIT_ASSIGNMENT.items()},
        "split_ratios": dict(TARGET_SPLIT_RATIOS),
        "split_methodology": (
            "Measurement-level, leakage-safe split. Every (bearing_code, operating_condition) stratum shares "
            "the same 20 measurement numbers, so a single deterministic seeded assignment of measurement_number "
            "to train/val/test (computed once, reused everywhere) guarantees every stratum is represented in "
            "every split while a single source measurement (.mat file) -- and every window generated from it -- "
            "belongs to exactly one split. No random shuffling of individual windows; splitting happens strictly "
            "at the measurement level."
        ),
        "expected_full_representation_size": estimate_full_representation_size(),
        "random_seed": RANDOM_SEED,
        "bearings": bearings_summary,
    }


def save_paderborn_dataset_summary(raw_dir: str = RAW_DATA_DIR, path: str = SUMMARY_JSON_PATH) -> Dict[str, object]:
    summary = discover_all_bearings_summary(raw_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    return summary


if __name__ == "__main__":
    save_paderborn_dataset_summary()
    print(f"Saved Paderborn dataset summary to: {SUMMARY_JSON_PATH}")
