"""Paderborn 3-class fault-location classification task definition and dataset assembly.

Reuses src.paderborn_pipeline.preprocessing unmodified for discovery, windowing,
measurement-level leakage-safe splitting, and train-only normalization. This module
only adds the classification-task-specific decisions: which operating condition and
modality to use, which bearing codes map to which class, and which bearing codes are
excluded because no single unambiguous fault-location label applies to them.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.paderborn_pipeline.preprocessing import (
    PADERBORN_BEARING_REGISTRY,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    apply_normalization,
    fit_train_normalization,
    load_stream_windows,
    verify_no_measurement_crosses_split,
    verify_split_disjoint,
)
from src.ims_pipeline.preprocessing import verify_observation_id_uniqueness

EXPERIMENT_OPERATING_CONDITION: str = "N15_M07_F10"
EXPERIMENT_MODALITY: str = "vibration"
EXPERIMENT_CHANNEL_INDEX: int = 0

CLASS_NAMES: Dict[int, str] = {0: "Healthy", 1: "Inner Race Fault", 2: "Outer Race Fault"}

BEARING_CLASS_LABELS: Dict[str, int] = {
    "K001": 0, "K002": 0, "K003": 0, "K004": 0, "K005": 0, "K006": 0,
    "KI01": 1, "KI03": 1, "KI05": 1, "KI07": 1, "KI08": 1,
    "KI16": 1, "KI17": 1, "KI18": 1, "KI21": 1,
    "KA01": 2, "KA03": 2, "KA04": 2, "KA05": 2, "KA06": 2, "KA07": 2,
    "KA09": 2, "KA15": 2, "KA16": 2, "KA22": 2, "KA30": 2,
}

EXCLUDED_BEARING_CODES: Dict[str, str] = {
    "KA08": (
        "damage_components=('AR',) is not a recognized OR/IR fault-location code "
        "in the Paderborn bearing registry; ambiguous, excluded rather than assumed."
    ),
    "KB23": "damage_components=('IR','IR','OR') mixes inner- and outer-race damage; no single unambiguous fault-location label applies.",
    "KB24": "damage_components=('IR','OR') mixes inner- and outer-race damage; no single unambiguous fault-location label applies.",
    "KB27": "damage_components=('OR','IR') mixes inner- and outer-race damage; no single unambiguous fault-location label applies.",
    "KI04": "damage_components=('IR','OR') mixes inner- and outer-race damage; no single unambiguous fault-location label applies.",
    "KI14": "damage_components=('IR','OR') mixes inner- and outer-race damage; no single unambiguous fault-location label applies.",
}

DATASET_V1_PATH: str = os.path.join(PROCESSED_DATA_DIR, "paderborn_dataset_v1.npz")
METADATA_PATH: str = os.path.join(PROCESSED_DATA_DIR, "paderborn_metadata.csv")
MANIFEST_PATH: str = os.path.join(PROCESSED_DATA_DIR, "paderborn_experiment_manifest.json")


def _validate_bearing_taxonomy() -> None:
    all_codes = set(PADERBORN_BEARING_REGISTRY.keys())
    included = set(BEARING_CLASS_LABELS.keys())
    excluded = set(EXCLUDED_BEARING_CODES.keys())
    if included & excluded:
        raise ValueError(f"Bearing code(s) both included and excluded: {sorted(included & excluded)}")
    if included | excluded != all_codes:
        raise ValueError(
            f"Bearing taxonomy does not cover the full registry. "
            f"Missing: {sorted(all_codes - included - excluded)}, "
            f"Unknown: {sorted((included | excluded) - all_codes)}"
        )


_validate_bearing_taxonomy()


def build_classification_dataset(
    raw_dir: str = RAW_DATA_DIR,
    bearing_class_labels: Dict[str, int] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if bearing_class_labels is None:
        bearing_class_labels = BEARING_CLASS_LABELS
    window_arrays = []
    metadata_frames = []
    for bearing_code, class_id in sorted(bearing_class_labels.items()):
        X_bearing, meta_bearing = load_stream_windows(
            bearing_code=bearing_code,
            operating_condition=EXPERIMENT_OPERATING_CONDITION,
            modality=EXPERIMENT_MODALITY,
            channel_index=EXPERIMENT_CHANNEL_INDEX,
            raw_dir=raw_dir,
        )
        meta_bearing = meta_bearing.copy()
        meta_bearing["fault_label"] = class_id
        meta_bearing["recording_id"] = (
            meta_bearing["bearing_code"] + "_m" + meta_bearing["measurement_number"].astype(str).str.zfill(2)
        )
        window_arrays.append(X_bearing)
        metadata_frames.append(meta_bearing)

    X = np.concatenate(window_arrays, axis=0)
    metadata_df = pd.concat(metadata_frames, axis=0, ignore_index=True)
    return X, metadata_df


def save_paderborn_classification_dataset(
    raw_dir: str = RAW_DATA_DIR,
    dataset_path: str = DATASET_V1_PATH,
    metadata_path: str = METADATA_PATH,
    manifest_path: str = MANIFEST_PATH,
) -> Dict[str, object]:
    X, metadata_df = build_classification_dataset(raw_dir=raw_dir)

    verify_observation_id_uniqueness(metadata_df)
    split_counts = verify_split_disjoint(metadata_df)
    verify_no_measurement_crosses_split(metadata_df)

    mean, std = fit_train_normalization(X, metadata_df)
    X_normalized = apply_normalization(X, mean, std)

    split_masks = {name: (metadata_df["split"] == name).to_numpy() for name in ("train", "val", "test")}
    y = metadata_df["fault_label"].to_numpy(dtype=np.int64)

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    np.savez(
        dataset_path,
        X_train=X_normalized[split_masks["train"]],
        y_train=y[split_masks["train"]],
        X_val=X_normalized[split_masks["val"]],
        y_val=y[split_masks["val"]],
        X_test=X_normalized[split_masks["test"]],
        y_test=y[split_masks["test"]],
    )

    metadata_df.to_csv(metadata_path, index=False)

    class_distribution_total = {
        CLASS_NAMES[class_id]: int((y == class_id).sum()) for class_id in CLASS_NAMES
    }
    class_distribution_by_split = {
        split_name: {
            CLASS_NAMES[class_id]: int(((y == class_id) & split_masks[split_name]).sum())
            for class_id in CLASS_NAMES
        }
        for split_name in ("train", "val", "test")
    }

    manifest = {
        "dataset": "paderborn",
        "experiment": "3-class fault-location classification (Healthy / Inner Race / Outer Race)",
        "operating_condition": EXPERIMENT_OPERATING_CONDITION,
        "modality": EXPERIMENT_MODALITY,
        "channel_index": EXPERIMENT_CHANNEL_INDEX,
        "window_size": int(X.shape[1]),
        "class_names": CLASS_NAMES,
        "included_bearing_codes": sorted(BEARING_CLASS_LABELS.keys()),
        "excluded_bearing_codes": EXCLUDED_BEARING_CODES,
        "n_bearing_codes_included": len(BEARING_CLASS_LABELS),
        "n_bearing_codes_excluded": len(EXCLUDED_BEARING_CODES),
        "total_windows": int(len(metadata_df)),
        "split_sizes": {name: int(mask.sum()) for name, mask in split_masks.items()},
        "class_distribution_total": class_distribution_total,
        "class_distribution_by_split": class_distribution_by_split,
        "normalization": {
            "method": "global scalar z-score, fit on train split only",
            "train_mean": mean,
            "train_std": std,
        },
        "leakage_checks": {
            "split_disjoint_overlaps": split_counts,
            "no_measurement_crosses_split": True,
            "observation_ids_unique": True,
        },
        "random_seed": 42,
        "provenance": {
            "data_source_name": "University of Paderborn Bearing DataCenter",
            "data_source_url": "https://groups.uni-paderborn.de/kat/BearingDataCenter/",
        },
    }

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


if __name__ == "__main__":
    result = save_paderborn_classification_dataset()
    print(json.dumps(result, indent=2, sort_keys=True))
