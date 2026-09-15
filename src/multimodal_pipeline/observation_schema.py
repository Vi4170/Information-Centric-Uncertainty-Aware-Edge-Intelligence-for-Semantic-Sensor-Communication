"""Task 32 -- the canonical multisensor observation for the primary
multimodal dataset (Mendeley DOI 10.17632/ztmf3m7h5x.6).

Prerequisite: Task 31 (``src/multimodal_pipeline/dataset_audit.py``). This
module uses ONLY the modalities and correspondence Task 31 established as
scientifically valid, and goes one step further: raw data has since been
placed under ``data/raw/multimodal/`` (it was not available during Task 31),
so several Task 31 conclusions that were "not_documented_by_source" have now
been checked directly against the real files. Findings from that direct
inspection, used below:

1. Vibration (``.mat``) and temperature+current (``.tdms``) files, joined by
   condition code, share the same calendar recording date for all 45
   conditions (verified exhaustively, not sampled) -- session-level
   correspondence for this triple is CONFIRMED, not just plausible.
2. Their absolute recording-start times are close but NOT fixed: on
   2021-06-25 the vibration system started ~3m23s-3m25s after the
   temperature/current system across four different files that day, but on
   2021-06-28 the gap was ~30 minutes. No single constant offset can be
   assumed, so absolute-clock alignment across the two acquisition systems
   is NOT used here.
3. The two systems' sample clocks also differ slightly: vibration is
   exactly 25,600.0 Hz every file (``x_values.increment`` = 3.90625e-05s
   exactly); temperature/current is consistently ~25,608.2 Hz
   (``wf_increment`` = 3.905e-05s, giving a few thousand extra samples over
   a 300s file). This is a small, real, non-random rate mismatch between two
   independently clocked DAQ systems, not measurement noise.
4. Acoustic recordings, despite sharing condition-code filenames with
   vibration, were made on entirely different calendar dates for all 5
   acoustic files (2021-07-05 / 2021-07-06, versus the original vibration
   sessions in late June 2021) -- CONFIRMED to be a separate, later
   measurement campaign, not the same physical run. Acoustic is therefore
   excluded from the canonical multisensor observation entirely (Task 31
   had already flagged it as unverified/likely-excluded; this is now a
   confirmed exclusion, not a cautious one).
5. Recording duration is NOT uniform: BPFI/BPFO = 60s, Misalignment /
   Unbalance = 120s at every load, but Normal = 300s at 0 Nm and 120s at
   2/4 Nm (verified against every one of the 45 vibration files).
6. The raw vibration archive contains a real filename typo for every one of
   the 5 "Unbalance" conditions at 2 Nm ONLY: ``2Nm_Unbalalnce_*.mat``
   (extra "al"). The temperature/current archive spells it correctly at
   every load. This module normalizes the condition code but preserves the
   literal on-disk filename for traceability -- it never silently assumes a
   clean join.
7. Vibration's 4 channels are labeled generically in the raw file
   ("Point1".."Point4"), not by housing/axis name. The
   x_A/y_A/x_B/y_B channel order used below comes only from the published
   paper (Jung et al., 2023), not from self-describing metadata in the raw
   file itself -- flagged as such, not silently trusted as verified.

Given (2) and (3), NO sample-index or absolute-timestamp alignment between
vibration/temperature/current is treated as established. The only
legitimate temporal reference is elapsed time since each recording's own
start (t=0 at that recording's first sample), used identically across the
two DAQ systems for a given window index. This is a deliberate, documented
choice, not a claim of true physical synchronization -- see
``ALIGNMENT_STRATEGY`` below.

This module builds a metadata-only observation index (no raw signal arrays
are loaded or windowed here). It does not implement modality encoders,
fusion, or VoI scoring -- those are explicitly out of scope for Task 32.
``src/voi/`` is not imported or modified anywhere in this module.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.cnn.config import INPUT_SHAPE
from src.multimodal_pipeline.dataset_audit import (
    FAULT_TYPES,
    LOAD_LEVELS_NM,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Canonical modality set (audit questions 8-11 resolved for Task 32)
# ---------------------------------------------------------------------------

CANONICAL_MODALITIES: Tuple[str, ...] = ("vibration", "temperature", "motor_current")

ACOUSTIC_EXCLUSION_REASON: str = (
    "Confirmed by direct inspection of raw file metadata: every one of the "
    "5 acoustic .mat files (function_record.creation_time) was recorded on "
    "2021-07-05 or 2021-07-06, while the corresponding vibration files for "
    "the same nominal condition code were recorded between 2021-06-25 and "
    "2021-06-30 -- a separate, later measurement campaign, not the same "
    "physical experimental run. Acoustic is excluded from the canonical "
    "multisensor observation entirely."
)

VIBRATION_SUBDIR: str = "vibration"
TEMPERATURE_CURRENT_SUBDIR: str = "temperature_current"
ACOUSTIC_SUBDIR: str = "acoustic"

OBSERVATION_INDEX_CSV_PATH: str = os.path.join(PROCESSED_DATA_DIR, "multimodal_observation_index.csv")
OBSERVATION_SCHEMA_JSON_PATH: str = os.path.join(PROCESSED_DATA_DIR, "multimodal_observation_schema.json")


# ---------------------------------------------------------------------------
# Filename tokens as they actually appear on disk (verified against real
# files -- NOT the human-readable engineering-unit strings used in Task 31's
# SEVERITY_LEVELS, which describe the physical severities, not the filename
# tokens that encode them).
# ---------------------------------------------------------------------------

FAULT_TYPE_FILENAME_TOKEN: Dict[str, str] = {
    "Normal": "Normal",
    "BPFI": "BPFI",
    "BPFO": "BPFO",
    "Misalignment": "Misalign",
    "Unbalance": "Unbalance",
}

SEVERITY_FILENAME_TOKENS: Dict[str, Tuple[str, ...]] = {
    "Normal": (),
    "BPFI": ("03", "10", "30"),
    "BPFO": ("03", "10", "30"),
    "Misalignment": ("01", "03", "05"),
    "Unbalance": ("0583mg", "1169mg", "1751mg", "2239mg", "3318mg"),
}

# The only known on-disk filename quirk: the vibration archive misspells
# "Unbalance" as "Unbalalnce" for every 2 Nm condition (and only at 2 Nm).
# Both spellings are tried when locating a vibration file; whichever one
# actually exists is recorded verbatim in the observation index.
VIBRATION_FAULT_TOKEN_ALTERNATES: Dict[str, Tuple[str, ...]] = {
    "Unbalance": ("Unbalance", "Unbalalnce"),
}

# Measured directly from every one of the 45 vibration files (see module
# docstring point 5). Keyed by (fault_type, load_nm); Normal is the only
# fault type whose duration depends on load.
CONDITION_DURATION_SECONDS: Dict[Tuple[str, int], float] = {
    **{("BPFI", load): 60.0 for load in LOAD_LEVELS_NM},
    **{("BPFO", load): 60.0 for load in LOAD_LEVELS_NM},
    **{("Misalignment", load): 120.0 for load in LOAD_LEVELS_NM},
    **{("Unbalance", load): 120.0 for load in LOAD_LEVELS_NM},
    ("Normal", 0): 300.0,
    ("Normal", 2): 120.0,
    ("Normal", 4): 120.0,
}


# ---------------------------------------------------------------------------
# Sampling / windowing constants (audit questions 5, 8; "Important" clause
# on differing window sizes per modality)
# ---------------------------------------------------------------------------

# Both vibration and temperature/current are documented (Task 31) and
# measured (this task) at ~25.6 kHz. Vibration's measured rate is exactly
# 25600.0 Hz in every file; temperature/current's measured rate is
# consistently ~25608.2 Hz (see module docstring point 3) -- a small, real
# inter-system clock-rate mismatch, not interpolated or corrected away here.
NOMINAL_SAMPLING_RATE_HZ: int = 25600
VIBRATION_MEASURED_SAMPLING_RATE_HZ: float = 25600.0
TEMPERATURE_CURRENT_MEASURED_INCREMENT_SECONDS: float = 3.905e-05  # ~25608.2 Hz

# Reuses the existing project-wide CNN window convention (src/cnn/config.py)
# rather than inventing a new window size -- the same 2048-sample window is
# already used, unmodified, for CWRU, Paderborn, and XJTU-SY.
WINDOW_SIZE_SAMPLES: int = INPUT_SHAPE[0]
WINDOW_DURATION_SECONDS: float = WINDOW_SIZE_SAMPLES / NOMINAL_SAMPLING_RATE_HZ

# If a modality with a different sampling rate is ever reintroduced (e.g.
# acoustic at 51,200 Hz), its window sample count must be scaled so the
# PHYSICAL time interval covered stays identical to WINDOW_DURATION_SECONDS,
# not the sample count. Documented here per the Task 32 "Important" clause,
# even though acoustic is not used by CANONICAL_MODALITIES today.
ACOUSTIC_SAMPLING_RATE_HZ: int = 51200
ACOUSTIC_WINDOW_SIZE_SAMPLES: int = round(WINDOW_DURATION_SECONDS * ACOUSTIC_SAMPLING_RATE_HZ)

VIBRATION_CHANNEL_ORDER: Tuple[str, ...] = (
    "x_direction_housing_A",
    "y_direction_housing_A",
    "x_direction_housing_B",
    "y_direction_housing_B",
)
VIBRATION_CHANNEL_ORDER_SOURCE: str = (
    "Published paper (Jung et al., 2023) only. The raw .mat file's own "
    "channel labels are generic ('Point1'..'Point4'), not self-describing "
    "-- this ordering is NOT independently verifiable from file metadata "
    "alone."
)

# ---------------------------------------------------------------------------
# Alignment strategy (audit questions 6, 7, 9)
# ---------------------------------------------------------------------------

ALIGNMENT_STRATEGY: Dict[str, object] = {
    "common_temporal_reference": (
        "Elapsed time since each recording's OWN first sample (t=0 at that "
        "file's start), NOT absolute wall-clock time and NOT a shared "
        "sample index."
    ),
    "why_not_absolute_clock": (
        "Measured absolute start-time offset between the vibration and "
        "temperature/current systems varies across sessions (~3.4 minutes "
        "on one day, ~30 minutes on another) -- no fixed correction exists."
    ),
    "why_not_shared_sample_index": (
        "The two systems' measured sample rates differ slightly (25600.0 Hz "
        "vs ~25608.2 Hz); over a multi-minute recording this accumulates a "
        "real sample-index drift of thousands of samples, so 'sample N of "
        "vibration' is not 'sample N of current' even within one file."
    ),
    "per_modality_window_extraction": (
        "For window index i, each modality independently selects the "
        "sample range covering elapsed time "
        "[i * WINDOW_DURATION_SECONDS, (i+1) * WINDOW_DURATION_SECONDS) "
        "using that modality's OWN measured sampling rate. No "
        "interpolation or resampling of raw values is performed -- only "
        "the sample INDEX range selected differs per modality to cover "
        "the same physical time interval."
    ),
    "residual_limitation": (
        "This assumes each recording's own t=0 corresponds to the same "
        "physical moment (e.g. reaching steady-state operation) across "
        "modalities, which is plausible for a single continuous "
        "steady-condition recording but is not independently verified. "
        "This is a documented, defensible default, not a proven exact "
        "synchronization -- large-gap interpolation across sessions is "
        "never performed."
    ),
    "acoustic": "Not applicable -- acoustic is excluded from the canonical observation entirely.",
}


# ---------------------------------------------------------------------------
# Split configuration (audit questions 13, 14)
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")
SPLIT_RATIOS: Dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

MIN_REQUIRED_MODALITIES: Tuple[str, ...] = ("vibration", "temperature_current")


# ---------------------------------------------------------------------------
# Raw-data discovery (metadata only -- no signal arrays are loaded)
# ---------------------------------------------------------------------------

def raw_data_available(raw_dir: str = RAW_DATA_DIR) -> bool:
    """Whether the vibration and temperature_current raw subdirectories exist and are non-empty."""
    vib_dir = os.path.join(raw_dir, VIBRATION_SUBDIR)
    tc_dir = os.path.join(raw_dir, TEMPERATURE_CURRENT_SUBDIR)
    return (
        os.path.isdir(vib_dir)
        and any(f.endswith(".mat") for f in os.listdir(vib_dir))
        and os.path.isdir(tc_dir)
        and any(f.endswith(".tdms") for f in os.listdir(tc_dir))
    )


_CONDITION_FILENAME_RE = re.compile(r"^(?P<load>\d+)Nm_(?P<rest>.+)$")


def _parse_condition_filename_stem(stem: str) -> Optional[Tuple[int, str, str]]:
    """Parse '{load}Nm_{fault_token}[_{severity_token}]' into (load, fault_token, severity_token).

    Returns None if the stem does not match the expected pattern at all
    (never raises -- callers decide how to handle unparsed files).
    """
    match = _CONDITION_FILENAME_RE.match(stem)
    if match is None:
        return None
    load = int(match.group("load"))
    rest = match.group("rest")
    if rest == "Normal":
        return load, "Normal", ""
    parts = rest.split("_", 1)
    if len(parts) != 2:
        return None
    fault_token, severity_token = parts
    return load, fault_token, severity_token


def _fault_type_for_token(fault_token: str) -> Optional[str]:
    """Reverse-lookup a canonical fault_type from an on-disk filename token,
    accounting for known alternate spellings (e.g. the 'Unbalalnce' typo)."""
    for fault_type, canonical_token in FAULT_TYPE_FILENAME_TOKEN.items():
        if fault_token == canonical_token:
            return fault_type
    for fault_type, alternates in VIBRATION_FAULT_TOKEN_ALTERNATES.items():
        if fault_token in alternates:
            return fault_type
    return None


def condition_code(load_nm: int, fault_type: str, severity_token: str) -> str:
    """The canonical, typo-normalized condition code used as an observation key."""
    fault_file_token = FAULT_TYPE_FILENAME_TOKEN[fault_type]
    if fault_type == "Normal":
        return f"{load_nm}Nm_{fault_file_token}"
    return f"{load_nm}Nm_{fault_file_token}_{severity_token}"


def discover_conditions(raw_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """Scan the actual raw vibration/ and temperature_current/ directories and
    join them into one condition-level table by (load, fault_type, severity).

    Every row is built from files that genuinely exist on disk -- nothing is
    assumed present. A condition missing either required modality is still
    returned as a row (so callers can see and count exclusions), with the
    missing path recorded as None.
    """
    vib_dir = os.path.join(raw_dir, VIBRATION_SUBDIR)
    tc_dir = os.path.join(raw_dir, TEMPERATURE_CURRENT_SUBDIR)
    acoustic_dir = os.path.join(raw_dir, ACOUSTIC_SUBDIR)

    if not raw_data_available(raw_dir):
        raise FileNotFoundError(
            f"Required raw multimodal directories not found under '{raw_dir}'. "
            f"Expected '{VIBRATION_SUBDIR}/' and '{TEMPERATURE_CURRENT_SUBDIR}/' "
            "each containing their respective raw files."
        )

    vib_files = {f[:-4]: f for f in os.listdir(vib_dir) if f.endswith(".mat")}
    tc_files = {f[:-5]: f for f in os.listdir(tc_dir) if f.endswith(".tdms")}
    acoustic_files: Dict[str, str] = {}
    if os.path.isdir(acoustic_dir):
        acoustic_files = {f[:-4]: f for f in os.listdir(acoustic_dir) if f.endswith(".mat")}

    rows: List[dict] = []
    seen_conditions: set = set()

    for load in LOAD_LEVELS_NM:
        for fault_type in FAULT_TYPES:
            severities = SEVERITY_FILENAME_TOKENS[fault_type] or ("",)
            for severity_token in severities:
                cond = condition_code(load, fault_type, severity_token)
                if cond in seen_conditions:
                    continue
                seen_conditions.add(cond)

                # Temperature/current: always spelled correctly.
                tc_filename = f"{cond}.tdms"
                tc_path = os.path.join(tc_dir, tc_filename) if tc_filename[:-5] in tc_files else None

                # Vibration: try the canonical spelling, then any known
                # alternate (e.g. the 2 Nm 'Unbalalnce' typo).
                vib_path = None
                vib_filename_used = None
                candidate_stems = [cond]
                if fault_type in VIBRATION_FAULT_TOKEN_ALTERNATES:
                    for alt_token in VIBRATION_FAULT_TOKEN_ALTERNATES[fault_type]:
                        alt_cond = (
                            f"{load}Nm_{alt_token}"
                            if fault_type == "Normal"
                            else f"{load}Nm_{alt_token}_{severity_token}"
                        )
                        if alt_cond not in candidate_stems:
                            candidate_stems.append(alt_cond)
                for stem in candidate_stems:
                    if stem in vib_files:
                        vib_filename_used = vib_files[stem]
                        vib_path = os.path.join(vib_dir, vib_filename_used)
                        break

                acoustic_stem = cond
                acoustic_filename_used = acoustic_files.get(acoustic_stem)
                acoustic_path = (
                    os.path.join(acoustic_dir, acoustic_filename_used)
                    if acoustic_filename_used
                    else None
                )

                duration = CONDITION_DURATION_SECONDS.get((fault_type, load))

                rows.append(
                    {
                        "condition_code": cond,
                        "load_nm": load,
                        "fault_type": fault_type,
                        "severity_token": severity_token,
                        "vibration_path": vib_path,
                        "vibration_filename": vib_filename_used,
                        "temperature_current_path": tc_path,
                        "temperature_current_filename": tc_filename if tc_path else None,
                        "acoustic_path": acoustic_path,
                        "acoustic_filename": acoustic_filename_used,
                        "acoustic_excluded_reason": (
                            ACOUSTIC_EXCLUSION_REASON if acoustic_path else None
                        ),
                        "duration_seconds": duration,
                        "modalities_available": [
                            m
                            for m, p in (("vibration", vib_path), ("temperature_current", tc_path))
                            if p is not None
                        ],
                        "meets_minimum_modality_requirement": (
                            vib_path is not None and tc_path is not None
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split assignment -- condition-level, deterministic (audit questions 13, 14)
# ---------------------------------------------------------------------------

def _compute_condition_split_assignment(
    condition_codes: List[str],
    seed: int = RANDOM_SEED,
) -> Dict[str, str]:
    """Deterministically assign whole conditions (never windows within a
    condition) to train/val/test, in proportion to SPLIT_RATIOS."""
    codes = sorted(condition_codes)
    rng = np.random.RandomState(seed)
    shuffled = list(codes)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * SPLIT_RATIOS["train"])
    n_val = round(n * SPLIT_RATIOS["val"])
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    assignment: Dict[str, str] = {}
    for i, code in enumerate(shuffled):
        if i < n_train:
            assignment[code] = "train"
        elif i < n_train + n_val:
            assignment[code] = "val"
        else:
            assignment[code] = "test"
    return assignment


# ---------------------------------------------------------------------------
# Observation index construction (audit questions 1-5, 10, 12, 15)
# ---------------------------------------------------------------------------

def _observation_id(condition: str, window_index: int) -> str:
    return f"multimodal_{condition}_w{window_index:04d}"


def _n_windows_for_duration(duration_seconds: Optional[float]) -> int:
    if duration_seconds is None:
        return 0
    return int(duration_seconds // WINDOW_DURATION_SECONDS)


def build_observation_index(raw_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """Build the full per-observation (per-window) metadata table.

    Only conditions meeting MIN_REQUIRED_MODALITIES (vibration AND
    temperature_current both present) produce observations. No raw signal
    array is loaded or read here -- this is metadata construction only.
    """
    conditions = discover_conditions(raw_dir)
    split_assignment = _compute_condition_split_assignment(
        conditions.loc[conditions["meets_minimum_modality_requirement"], "condition_code"].tolist()
    )

    rows: List[dict] = []
    for _, cond_row in conditions.iterrows():
        if not cond_row["meets_minimum_modality_requirement"]:
            continue

        condition = cond_row["condition_code"]
        split = split_assignment[condition]
        n_windows = _n_windows_for_duration(cond_row["duration_seconds"])

        for window_index in range(n_windows):
            window_start_s = window_index * WINDOW_DURATION_SECONDS
            window_end_s = window_start_s + WINDOW_DURATION_SECONDS

            vib_start_sample = int(round(window_start_s * VIBRATION_MEASURED_SAMPLING_RATE_HZ))
            vib_end_sample = vib_start_sample + WINDOW_SIZE_SAMPLES

            tc_rate = 1.0 / TEMPERATURE_CURRENT_MEASURED_INCREMENT_SECONDS
            tc_start_sample = int(round(window_start_s * tc_rate))
            tc_window_size = int(round(WINDOW_DURATION_SECONDS * tc_rate))
            tc_end_sample = tc_start_sample + tc_window_size

            rows.append(
                {
                    "observation_id": _observation_id(condition, window_index),
                    "dataset": "multimodal",
                    "source_session_id": condition,
                    "condition_code": condition,
                    "load_nm": cond_row["load_nm"],
                    "fault_type": cond_row["fault_type"],
                    "severity_token": cond_row["severity_token"],
                    "modalities_included": list(CANONICAL_MODALITIES),
                    "vibration_source_file": cond_row["vibration_filename"],
                    "temperature_current_source_file": cond_row["temperature_current_filename"],
                    "acoustic_source_file": cond_row["acoustic_filename"],
                    "acoustic_included": False,
                    "acoustic_excluded_reason": cond_row["acoustic_excluded_reason"],
                    "window_index": window_index,
                    "window_start_seconds": window_start_s,
                    "window_end_seconds": window_end_s,
                    "window_duration_seconds": WINDOW_DURATION_SECONDS,
                    "vibration_sample_range": (vib_start_sample, vib_end_sample),
                    "vibration_window_size_samples": WINDOW_SIZE_SAMPLES,
                    "vibration_sampling_rate_hz": VIBRATION_MEASURED_SAMPLING_RATE_HZ,
                    "temperature_current_sample_range": (tc_start_sample, tc_end_sample),
                    "temperature_current_window_size_samples": tc_window_size,
                    "temperature_current_sampling_rate_hz": round(tc_rate, 2),
                    "common_temporal_reference": ALIGNMENT_STRATEGY["common_temporal_reference"],
                    "split": split,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leakage-prevention verification (audit question 14)
# ---------------------------------------------------------------------------

def verify_no_condition_crosses_split(observation_df: pd.DataFrame) -> bool:
    grouped = observation_df.groupby("condition_code")["split"].nunique()
    offenders = grouped[grouped > 1]
    if not offenders.empty:
        raise AssertionError(f"Condition(s) cross split boundaries: {offenders.index.tolist()}")
    return True


def verify_split_disjoint(observation_df: pd.DataFrame) -> Dict[str, int]:
    ids_by_split = {
        split_name: set(observation_df.loc[observation_df["split"] == split_name, "observation_id"])
        for split_name in SPLIT_NAMES
    }
    overlaps = {
        "train_val_overlap": len(ids_by_split["train"] & ids_by_split["val"]),
        "train_test_overlap": len(ids_by_split["train"] & ids_by_split["test"]),
        "val_test_overlap": len(ids_by_split["val"] & ids_by_split["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Split overlap detected: {overlaps}")
    return overlaps


def verify_observation_id_uniqueness(observation_df: pd.DataFrame) -> bool:
    if not observation_df["observation_id"].is_unique:
        duplicates = observation_df.loc[
            observation_df["observation_id"].duplicated(), "observation_id"
        ].tolist()
        raise AssertionError(f"Duplicate observation_id(s) found: {duplicates[:5]}")
    return True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_observation_index(
    raw_dir: str = RAW_DATA_DIR,
    csv_path: str = OBSERVATION_INDEX_CSV_PATH,
    schema_path: str = OBSERVATION_SCHEMA_JSON_PATH,
) -> pd.DataFrame:
    """Build, verify, and persist the observation index plus a human/machine
    -readable schema-definition document alongside it."""
    observation_df = build_observation_index(raw_dir)

    verify_observation_id_uniqueness(observation_df)
    verify_split_disjoint(observation_df)
    verify_no_condition_crosses_split(observation_df)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    observation_df.to_csv(csv_path, index=False)

    schema_doc = {
        "task": 32,
        "title": "Canonical multisensor observation definition",
        "prerequisite": "Task 31 (src/multimodal_pipeline/dataset_audit.py)",
        "canonical_modalities": list(CANONICAL_MODALITIES),
        "acoustic_exclusion_reason": ACOUSTIC_EXCLUSION_REASON,
        "window_size_samples": WINDOW_SIZE_SAMPLES,
        "window_duration_seconds": WINDOW_DURATION_SECONDS,
        "nominal_sampling_rate_hz": NOMINAL_SAMPLING_RATE_HZ,
        "vibration_channel_order": list(VIBRATION_CHANNEL_ORDER),
        "vibration_channel_order_source": VIBRATION_CHANNEL_ORDER_SOURCE,
        "alignment_strategy": ALIGNMENT_STRATEGY,
        "split_ratios": SPLIT_RATIOS,
        "random_seed": RANDOM_SEED,
        "min_required_modalities": list(MIN_REQUIRED_MODALITIES),
        "n_conditions_discovered": int(
            discover_conditions(raw_dir)["meets_minimum_modality_requirement"].sum()
        ),
        "n_observations": int(len(observation_df)),
        "n_observations_by_split": observation_df["split"].value_counts().to_dict(),
    }
    os.makedirs(os.path.dirname(schema_path) or ".", exist_ok=True)
    with open(schema_path, "w", encoding="utf-8") as handle:
        json.dump(schema_doc, handle, indent=2, sort_keys=True, default=str)

    return observation_df


if __name__ == "__main__":
    df = save_observation_index()
    print(f"Saved observation index ({len(df)} observations) to: {OBSERVATION_INDEX_CSV_PATH}")
    print(f"Saved observation schema to: {OBSERVATION_SCHEMA_JSON_PATH}")
    print(df["split"].value_counts())
