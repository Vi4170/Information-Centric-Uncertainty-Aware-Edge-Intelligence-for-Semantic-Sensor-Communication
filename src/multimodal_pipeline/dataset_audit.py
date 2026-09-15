"""Task 31 -- structural/provenance audit of the primary multimodal dataset.

Dataset audited:
    "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating
    Machine Under Varying Load Conditions for Fault Diagnosis"
    Mendeley Data, DOI 10.17632/ztmf3m7h5x.6 (Version 6, KAIST).

This module does NOT implement preprocessing, fusion, windowing, CNN input
construction, VoI scoring, or communication logic -- that is explicitly out
of scope for Task 31. It only records, in a machine-readable form, what is
and is not legitimately known about this dataset's structure and modality
correspondence, so a later task can decide how to build a genuine multisensor
observation without inventing anything.

No raw data for this dataset exists in this environment (verified: no
``data/raw/multimodal`` directory). Every fact below therefore comes from
the dataset's own published documentation, not from inspecting real files:

    - Mendeley Data landing page: https://data.mendeley.com/datasets/ztmf3m7h5x/6
    - Companion Data in Brief paper: Jung, W., Kim, S.-H., Yun, S., Bae, J.,
      Park, Y.-H. (2023). "Vibration, acoustic, temperature, and motor
      current dataset of rotating machine under varying operating
      conditions for fault diagnosis." Data in Brief, 48, 109049.
      https://doi.org/10.1016/j.dib.2023.109049
      (open-access mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC10036499/)

Every field below is tagged with one of the ``AUDIT_STATUS_*`` values so a
reader (human or test) can tell a directly-published fact apart from a
derived count, an undocumented gap, or something that requires raw data to
verify. Nothing here is invented: where the source is silent, the field says
so explicitly rather than filling in a plausible-looking value.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Audit status vocabulary
# ---------------------------------------------------------------------------

AUDIT_STATUS_CONFIRMED = "confirmed_from_published_source"
AUDIT_STATUS_DERIVED = "derived_from_published_source"
AUDIT_STATUS_NOT_DOCUMENTED = "not_documented_by_source"
AUDIT_STATUS_UNVERIFIED_NO_RAW_DATA = "unverified_no_raw_data_in_environment"
AUDIT_STATUS_EXCLUDED = "excluded_from_fusion_scope"

VALID_AUDIT_STATUSES = frozenset(
    {
        AUDIT_STATUS_CONFIRMED,
        AUDIT_STATUS_DERIVED,
        AUDIT_STATUS_NOT_DOCUMENTED,
        AUDIT_STATUS_UNVERIFIED_NO_RAW_DATA,
        AUDIT_STATUS_EXCLUDED,
    }
)


# ---------------------------------------------------------------------------
# Dataset identity and provenance (audit question 14)
# ---------------------------------------------------------------------------

DATASET_NAME: str = "multimodal"

RAW_DATA_DIR: str = os.path.join("data", "raw", "multimodal")
PROCESSED_DATA_DIR: str = os.path.join("data", "processed", "multimodal")
AUDIT_JSON_PATH: str = os.path.join(PROCESSED_DATA_DIR, "multimodal_dataset_audit.json")

OFFICIAL_DATASET_TITLE: str = (
    "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating "
    "Machine Under Varying Load Conditions for Fault Diagnosis"
)
OFFICIAL_DOI: str = "10.17632/ztmf3m7h5x.6"
OFFICIAL_DATASET_URL: str = "https://data.mendeley.com/datasets/ztmf3m7h5x/6"
OFFICIAL_DATASET_VERSION: int = 6
OFFICIAL_DATASET_PUBLICATION_DATE: str = "2023-02-08"
OFFICIAL_LICENSE: str = "CC BY 4.0"
OFFICIAL_INSTITUTION: str = "Korea Advanced Institute of Science and Technology (KAIST)"
OFFICIAL_AUTHORS: Tuple[str, ...] = (
    "Wonho Jung",
    "Seong-Hu Kim",
    "SungHyun Yun",
    "Jaewoong Bae",
    "Yong-Hwa Park",
)
OFFICIAL_COMPANION_PAPER_CITATION: str = (
    "Jung, W., Kim, S.-H., Yun, S., Bae, J., Park, Y.-H. (2023). "
    "\"Vibration, acoustic, temperature, and motor current dataset of "
    "rotating machine under varying operating conditions for fault "
    "diagnosis.\" Data in Brief, 48, 109049. "
    "https://doi.org/10.1016/j.dib.2023.109049"
)
OFFICIAL_COMPANION_PAPER_OPEN_ACCESS_URL: str = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10036499/"
)


# ---------------------------------------------------------------------------
# Test rig (audit question 6, context for question 7)
# ---------------------------------------------------------------------------

TEST_RIG_DESCRIPTION: str = (
    "Siemens 3-HP three-phase AC motor (rated 1770 rpm) -> 2.07x gearbox -> "
    "shaft carrying four rotor disks -> two bearing housings (A, B) -> "
    "air-cooled hysteresis brake (AHB-3A) with a torque meter (Datum M425) "
    "for load application."
)

LOAD_LEVELS_NM: Tuple[int, ...] = (0, 2, 4)
CONSTANT_SPEED_RPM: int = 3010
VARYING_SPEED_RANGE_RPM: Tuple[int, int] = (680, 2460)


# ---------------------------------------------------------------------------
# Modality registry (audit questions 1-3, 7)
#
# "part1_load" = the varying-load dataset (normal/fault x severity x load).
# "part2_speed" = the separate varying-speed dataset. The two parts use
# different file formats, different naming conventions, and (for motor
# current) a different sampling rate -- they are documented here as
# structurally distinct sub-datasets, not merged.
# ---------------------------------------------------------------------------

MODALITY_REGISTRY: Dict[str, Dict[str, object]] = {
    "vibration": {
        "file_format": "mat",
        "columns": (
            "Time Stamp",
            "x_direction_housing_A",
            "y_direction_housing_A",
            "x_direction_housing_B",
            "y_direction_housing_B",
        ),
        "units": "g (gravitational acceleration)",
        "acquisition_device": (
            "4x PCB352C34 ICP accelerometers (2 axes x 2 housings) via a "
            "Siemens SCADAS Mobile 5PM50 data acquisition system"
        ),
        "sampling_rate_hz": {"part1_load": 25600, "part2_speed": None},
        "sampling_rate_status": {
            "part1_load": AUDIT_STATUS_CONFIRMED,
            "part2_speed": AUDIT_STATUS_NOT_DOCUMENTED,
        },
        "sampling_rate_note": (
            "part2_speed vibration sampling rate is not explicitly restated "
            "for the varying-speed sub-dataset in the fetched source text; "
            "it is NOT assumed to equal the part1_load rate."
        ),
        "coverage": "part1_load (all conditions) and part2_speed (constant + varying speed)",
        "present_in_part1_load": True,
        "present_in_part2_speed": True,
    },
    "acoustic": {
        "file_format": "mat",
        "columns": ("Time Stamp", "values"),
        "units": "Pa",
        "acquisition_device": (
            "1x PCB378B02 microphone, positioned near bearing housing A, via "
            "the same Siemens SCADAS Mobile 5PM50 system as vibration"
        ),
        "sampling_rate_hz": {"part1_load": 51200, "part2_speed": None},
        "sampling_rate_status": {
            "part1_load": AUDIT_STATUS_CONFIRMED,
            "part2_speed": AUDIT_STATUS_NOT_DOCUMENTED,
        },
        "sampling_rate_note": "Acoustic data does not exist in part2_speed at all.",
        "coverage": (
            "part1_load ONLY, and only at the 0 Nm load level (the authors "
            "state acoustic data was restricted to no-load conditions "
            "specifically to avoid noise from the air-cooled hysteresis "
            "brake). 5 files total: 1 Normal + a documented subset of 2 "
            "fault types at 2 severities each, all at 0 Nm."
        ),
        "present_in_part1_load": True,
        "present_in_part2_speed": False,
    },
    "temperature": {
        "file_format": "tdms",
        "columns": (
            "Time Stamp",
            "Temperature_housing_A",
            "Temperature_housing_B",
            "U-phase",
            "V-phase",
            "W-phase",
        ),
        "units": "degrees Celsius (temperature columns); ampere (current columns)",
        "acquisition_device": "2x K-type thermocouples via an NI 9211 module",
        "sampling_rate_hz": {"part1_load": 25600, "part2_speed": None},
        "sampling_rate_status": {
            "part1_load": AUDIT_STATUS_CONFIRMED,
            "part2_speed": AUDIT_STATUS_NOT_DOCUMENTED,
        },
        "sampling_rate_note": "Temperature data does not exist in part2_speed at all.",
        "coverage": (
            "part1_load ONLY. Shares one .tdms file with motor_current -- "
            "temperature and motor current are the same physical recording, "
            "not two files that must be separately aligned."
        ),
        "present_in_part1_load": True,
        "present_in_part2_speed": False,
        "shares_raw_file_with": "motor_current",
    },
    "motor_current": {
        "file_format": {"part1_load": "tdms", "part2_speed": "csv"},
        "columns": (
            "Time Stamp",
            "Temperature_housing_A",
            "Temperature_housing_B",
            "U-phase",
            "V-phase",
            "W-phase",
        ),
        "units": "ampere (A)",
        "acquisition_device": "3x Hioki CT6700 current-transformer sensors via an NI9775 module",
        "sampling_rate_hz": {"part1_load": 25600, "part2_speed": 100000},
        "sampling_rate_status": {
            "part1_load": AUDIT_STATUS_CONFIRMED,
            "part2_speed": AUDIT_STATUS_CONFIRMED,
        },
        "coverage": "part1_load (shares file with temperature) and part2_speed (standalone CSV, paired with an RPM reference file)",
        "present_in_part1_load": True,
        "present_in_part2_speed": True,
        "shares_raw_file_with": "temperature (part1_load only)",
    },
}

ALL_MODALITIES: Tuple[str, ...] = tuple(sorted(MODALITY_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# Health/fault labels and severities -- part1_load (audit questions 5-7)
#
# File naming convention (author-documented): "{load}Nm_{fault_type}_{severity}"
# e.g. "0Nm_BPFI_03", "2Nm_Normal". This naming triple is treated as the
# recording/session identifier: one triple = one physical experimental run.
#
# The per-fault-type severity counts below are directly stated by the
# source. The resulting file-count totals are DERIVED (multiplied out) and
# cross-checked against the source's own reported totals (45 vibration
# files, 45 temperature/current files) as an internal consistency check.
# ---------------------------------------------------------------------------

FAULT_TYPES: Tuple[str, ...] = ("Normal", "BPFI", "BPFO", "Misalignment", "Unbalance")

SEVERITY_LEVELS: Dict[str, Tuple[str, ...]] = {
    "Normal": (),
    "BPFI": ("0.3mm", "1.0mm", "3.0mm"),
    "BPFO": ("0.3mm", "1.0mm", "3.0mm"),
    "Misalignment": ("0.1mm", "0.3mm", "0.5mm"),
    "Unbalance": ("583mg", "1169mg", "1751mg", "2239mg", "3318mg"),
}

FAULT_TYPE_DESCRIPTION: Dict[str, str] = {
    "Normal": "Healthy baseline condition, no induced fault.",
    "BPFI": "Bearing inner race fault, induced via a machined crack of the given depth.",
    "BPFO": "Bearing outer race fault, induced via a machined crack of the given depth.",
    "Misalignment": "Shaft misalignment, induced via the given lateral offset.",
    "Unbalance": "Rotor unbalance, induced via an added mass of the given weight.",
}

LABEL_GRANULARITY_NOTE: str = (
    "Labels are file-level (one condition code per recording), not "
    "per-sample or per-window. Fault severities are artificially seeded "
    "(machined cracks / fixed offsets / added masses) and static for the "
    "duration of a recording -- this is a multi-class static-severity "
    "design, not a run-to-failure degradation trajectory like IMS/XJTU-SY."
)


def _n_conditions_for_fault(fault_type: str) -> int:
    """Number of distinct (fault_type, severity) condition codes."""
    severities = SEVERITY_LEVELS[fault_type]
    return 1 if not severities else len(severities)


PART1_CONDITION_CODES: Tuple[str, ...] = tuple(
    sorted(
        f"{fault}_{severity}" if severity else fault
        for fault in FAULT_TYPES
        for severity in (SEVERITY_LEVELS[fault] or (None,))
    )
)

N_CONDITIONS_PART1: int = sum(_n_conditions_for_fault(f) for f in FAULT_TYPES)
N_VIBRATION_FILES_PART1_EXPECTED: int = N_CONDITIONS_PART1 * len(LOAD_LEVELS_NM)
N_TEMPERATURE_CURRENT_FILES_PART1_EXPECTED: int = N_VIBRATION_FILES_PART1_EXPECTED

# Directly stated by the source (Data in Brief paper); kept separate from
# the derived count above so the two can be cross-checked against each
# other rather than one silently overwriting the other.
N_VIBRATION_FILES_PART1_SOURCE_STATED: int = 45
N_TEMPERATURE_CURRENT_FILES_PART1_SOURCE_STATED: int = 45

N_ACOUSTIC_FILES_PART1_SOURCE_STATED: int = 5
ACOUSTIC_COVERAGE_NOTE: str = (
    "Source states 5 acoustic files: 1 Normal (0 Nm) plus 2 fault types at "
    "2 severities each, all restricted to 0 Nm. The specific fault-type and "
    "severity identities of that subset are NOT stated in the fetched "
    "source text and are deliberately not enumerated here -- listing "
    "specific values would be inventing a correspondence the source does "
    "not confirm."
)

# ---------------------------------------------------------------------------
# part2_speed -- structurally separate sub-dataset (audit questions 6, 11)
# ---------------------------------------------------------------------------

PART2_SPEED_FILE_COUNTS: Dict[str, int] = {
    "vibration_constant_speed": 4,
    "vibration_varying_speed": 28,
    "motor_current_varying_speed": 28,
    "rpm_reference": 28,
}
PART2_SPEED_FAULT_TYPES: Tuple[str, ...] = ("normal", "inner", "outer", "ball")
PART2_SPEED_FORMAT: str = "csv"
PART2_SPEED_NOTE: str = (
    "A separate sub-dataset covering constant (3010 rpm) and varying "
    "(680-2460 rpm) speed conditions. Contains only vibration + motor "
    "current + an RPM reference signal (CSV format) -- no acoustic or "
    "temperature channel exists for these recordings at all. Motor current "
    "is sampled at 100 kHz here versus 25.6 kHz in part1_load. Treated as "
    "structurally distinct from part1_load: different file format, "
    "different naming convention, different current sampling rate, and a "
    "reduced modality set."
)


# ---------------------------------------------------------------------------
# Fusion-legitimacy conclusions (audit questions 8-11)
# ---------------------------------------------------------------------------

MODALITIES_CONFIRMED_SAME_PHYSICAL_EXPERIMENT: Tuple[str, ...] = (
    "vibration",
    "temperature",
    "motor_current",
)
SAME_EXPERIMENT_BASIS: str = (
    "The source states these three modalities were acquired simultaneously "
    "on the same testbed for each condition; independently, their file "
    "counts (45 = 45) and shared {load}Nm_{fault}_{severity} naming "
    "convention are mutually consistent with this claim for part1_load."
)

CROSS_MODAL_SYNCHRONIZATION_CONCLUSION: Dict[str, object] = {
    "session_level_correspondence": {
        "modalities": ["vibration", "temperature", "motor_current"],
        "status": AUDIT_STATUS_CONFIRMED,
        "basis": SAME_EXPERIMENT_BASIS,
        "meaning": (
            "A given {load}Nm_{fault}_{severity} condition code legitimately "
            "identifies one shared physical experimental run across these "
            "three modalities. It is legitimate to treat one condition code "
            "as one multimodal session."
        ),
    },
    "sample_level_synchronization": {
        "modalities": ["vibration", "temperature", "motor_current"],
        "status": AUDIT_STATUS_NOT_DOCUMENTED,
        "basis": (
            "Vibration/acoustic are acquired by the Siemens SCADAS system; "
            "temperature/motor_current are acquired by separate NI modules "
            "(NI 9211, NI9775). Each produces its own 'Time Stamp' column. "
            "The source does not document a shared trigger, shared clock, "
            "or any explicit synchronization procedure between the SCADAS "
            "and NI acquisition systems."
        ),
        "meaning": (
            "Per-sample or per-timestamp fusion (e.g. aligning vibration "
            "sample N with a specific current sample by matching Time Stamp "
            "values across the two DAQ systems) is NOT established by the "
            "source and MUST NOT be assumed. Any future implementation must "
            "either verify this directly against raw file timestamps or "
            "fuse at a coarser (e.g. windowed/resampled) granularity that "
            "does not depend on cross-DAQ sample alignment."
        ),
    },
    "acoustic_synchronization": {
        "modalities": ["acoustic"],
        "status": AUDIT_STATUS_EXCLUDED,
        "basis": ACOUSTIC_COVERAGE_NOTE,
        "meaning": (
            "Acoustic can only be session-matched to vibration/temperature/"
            "motor_current for the narrow 0 Nm subset it exists in, and "
            "even there the exact matching condition codes are not "
            "confirmed by the fetched source text."
        ),
    },
}

MODALITIES_SAFELY_COMBINABLE: Dict[str, object] = {
    "part1_load_primary_triple": {
        "modalities": ["vibration", "temperature", "motor_current"],
        "granularity": "session-level (one condition code = one observation's source run)",
        "status": AUDIT_STATUS_DERIVED,
        "n_sessions": N_VIBRATION_FILES_PART1_SOURCE_STATED,
    },
    "part1_load_with_acoustic": {
        "modalities": ["vibration", "temperature", "motor_current", "acoustic"],
        "granularity": "session-level, 0 Nm subset only",
        "status": AUDIT_STATUS_UNVERIFIED_NO_RAW_DATA,
        "n_sessions": N_ACOUSTIC_FILES_PART1_SOURCE_STATED,
        "note": (
            "Combinable in principle, but which specific condition codes "
            "overlap cannot be confirmed without either the raw acoustic "
            "filenames or a file manifest -- neither is present in this "
            "environment."
        ),
    },
    "part2_speed": {
        "modalities": ["vibration", "motor_current"],
        "granularity": "session-level",
        "status": AUDIT_STATUS_EXCLUDED,
        "note": "Structurally separate from part1_load; see PART2_SPEED_NOTE.",
    },
}

EXCLUSIONS: Tuple[str, ...] = (
    "acoustic is excluded from the primary fusion scope for every load != 0 Nm, "
    "and for every fault/severity combination not among its documented "
    "5-file subset (exact identities undocumented).",
    "part2_speed is excluded from the primary (4-modality) fusion scope: it has "
    "no acoustic or temperature channel, uses a different motor_current "
    "sampling rate (100 kHz vs 25.6 kHz), a different file format (CSV), and "
    "a different naming/session convention than part1_load.",
    "sample-level (tick-by-tick) cross-DAQ timestamp synchronization between "
    "{vibration, acoustic} (SCADAS) and {temperature, motor_current} (NI) is "
    "excluded from any legitimate-fusion claim until directly verified "
    "against raw file timestamps -- it is not documented by the source.",
)

MISSING_DATA_AND_ALIGNMENT_ISSUES: Dict[str, object] = {
    "status": AUDIT_STATUS_NOT_DOCUMENTED,
    "note": (
        "The fetched source text does not discuss missing data, dropped "
        "samples, or sensor drift during acquisition. This audit cannot "
        "confirm the absence of such issues either -- no raw files or file "
        "manifest are present in this environment to check directly."
    ),
}

LEAKAGE_RISK_NOTES: Tuple[str, ...] = (
    "If this dataset is used for train/val/test splitting, splits must be "
    "assigned at the condition-code (session) level, never at the "
    "window/sample level -- consistent with every other dataset already "
    "integrated in this repository (CWRU, Paderborn, IMS, XJTU-SY).",
    "Fault severities are artificially seeded and static per recording "
    "(not a naturally progressive degradation), so a model evaluated only "
    "within the same severity levels it was trained on risks learning a "
    "severity-code shortcut rather than a genuine fault signature; this is "
    "a modeling-evaluation risk for a future task, not a data-leakage risk "
    "in the train/val/test sense.",
    "part1_load and part2_speed must not be silently pooled: they represent "
    "different experimental designs (induced static severity vs. varying "
    "operating speed) and mixing them without accounting for that would "
    "conflate two different sources of variation.",
)


# ---------------------------------------------------------------------------
# Raw-data presence check (no raw data expected in this environment)
# ---------------------------------------------------------------------------

def raw_data_root_available(raw_dir: str = RAW_DATA_DIR) -> bool:
    """Whether ANY raw data for this dataset is present locally."""
    return os.path.isdir(raw_dir) and any(os.scandir(raw_dir))


# ---------------------------------------------------------------------------
# Audit assembly
# ---------------------------------------------------------------------------

def build_dataset_audit() -> Dict[str, object]:
    """Assemble the full Task 31 audit as a single machine-readable dict.

    Every top-level numbered key corresponds directly to one of the 15 audit
    questions in the Task 31 specification.
    """
    raw_available = raw_data_root_available()

    return {
        "task": 31,
        "title": "Audit and lock the primary multimodal dataset",
        "dataset": DATASET_NAME,
        "status_vocabulary": {
            AUDIT_STATUS_CONFIRMED: "Directly stated by the dataset's published documentation.",
            AUDIT_STATUS_DERIVED: "Computed from confirmed facts via a documented, non-arbitrary transformation.",
            AUDIT_STATUS_NOT_DOCUMENTED: "The published source does not state this; not assumed or invented.",
            AUDIT_STATUS_UNVERIFIED_NO_RAW_DATA: "Would require inspecting real files; no raw data exists in this environment.",
            AUDIT_STATUS_EXCLUDED: "Deliberately excluded from the primary multisensor-fusion scope, with a stated reason.",
        },
        "raw_data_available_in_environment": raw_available,
        "1_available_modalities": {
            "status": AUDIT_STATUS_CONFIRMED,
            "modalities": ALL_MODALITIES,
            "detail": MODALITY_REGISTRY,
        },
        "2_raw_file_formats_and_directory_structure": {
            "status": AUDIT_STATUS_CONFIRMED,
            "part1_load_naming_convention": "{load}Nm_{fault_type}_{severity}.{ext} (severity omitted for Normal)",
            "part2_speed_naming_convention": "{signal}_{condition}_{subset}.csv",
            "formats_by_modality": {
                "vibration": "mat",
                "acoustic": "mat",
                "temperature": "tdms (shared file with motor_current)",
                "motor_current": {"part1_load": "tdms (shared file with temperature)", "part2_speed": "csv"},
            },
            "explicit_folder_hierarchy_documented": {
                "status": AUDIT_STATUS_NOT_DOCUMENTED,
                "note": "No explicit folder/directory hierarchy is documented by the source; only file-naming conventions are.",
            },
        },
        "3_sampling_rates": {
            modality: {
                "sampling_rate_hz": info["sampling_rate_hz"],
                "status": info["sampling_rate_status"],
            }
            for modality, info in MODALITY_REGISTRY.items()
        },
        "4_timestamp_availability_and_resolution": {
            "column_present_in_every_modality": {
                "status": AUDIT_STATUS_CONFIRMED,
                "note": "Every raw file format (vibration/acoustic .mat, temperature+current .tdms) documents a 'Time Stamp' column.",
            },
            "nominal_resolution_derived_from_sampling_rate": {
                "status": AUDIT_STATUS_DERIVED,
                "vibration_seconds": 1.0 / 25600,
                "acoustic_seconds": 1.0 / 51200,
                "temperature_current_part1_seconds": 1.0 / 25600,
                "motor_current_part2_seconds": 1.0 / 100000,
            },
            "absolute_epoch_or_shared_clock_format": {
                "status": AUDIT_STATUS_NOT_DOCUMENTED,
                "note": "Whether 'Time Stamp' is absolute/wall-clock, or relative-to-recording-start, and whether it is comparable across the two separate DAQ systems, is not documented by the source.",
            },
        },
        "5_recording_session_experiment_identifiers": {
            "status": AUDIT_STATUS_CONFIRMED,
            "part1_load": "the {load}Nm_{fault_type}_{severity} filename triple",
            "part2_speed": "the {condition}_{subset} filename pair",
        },
        "6_machine_operating_load_conditions": {
            "status": AUDIT_STATUS_CONFIRMED,
            "part1_load_levels_nm": LOAD_LEVELS_NM,
            "part2_constant_speed_rpm": CONSTANT_SPEED_RPM,
            "part2_varying_speed_range_rpm": VARYING_SPEED_RANGE_RPM,
            "test_rig": TEST_RIG_DESCRIPTION,
        },
        "7_health_fault_labels_and_granularity": {
            "status": AUDIT_STATUS_CONFIRMED,
            "fault_types": FAULT_TYPES,
            "fault_type_description": FAULT_TYPE_DESCRIPTION,
            "severity_levels": SEVERITY_LEVELS,
            "label_granularity": LABEL_GRANULARITY_NOTE,
        },
        "8_modalities_from_same_physical_experiment": {
            "status": AUDIT_STATUS_CONFIRMED,
            "confirmed_same_experiment": MODALITIES_CONFIRMED_SAME_PHYSICAL_EXPERIMENT,
            "basis": SAME_EXPERIMENT_BASIS,
            "acoustic_caveat": ACOUSTIC_COVERAGE_NOTE,
        },
        "9_cross_modal_synchronization_legitimacy": CROSS_MODAL_SYNCHRONIZATION_CONCLUSION,
        "10_modalities_safely_combinable": MODALITIES_SAFELY_COMBINABLE,
        "11_exclusions": EXCLUSIONS,
        "12_missing_data_and_alignment_issues": MISSING_DATA_AND_ALIGNMENT_ISSUES,
        "13_dataset_level_leakage_risks": LEAKAGE_RISK_NOTES,
        "14_source_provenance_metadata": {
            "status": AUDIT_STATUS_CONFIRMED,
            "title": OFFICIAL_DATASET_TITLE,
            "doi": OFFICIAL_DOI,
            "url": OFFICIAL_DATASET_URL,
            "version": OFFICIAL_DATASET_VERSION,
            "publication_date": OFFICIAL_DATASET_PUBLICATION_DATE,
            "license": OFFICIAL_LICENSE,
            "institution": OFFICIAL_INSTITUTION,
            "authors": OFFICIAL_AUTHORS,
            "companion_paper_citation": OFFICIAL_COMPANION_PAPER_CITATION,
            "companion_paper_open_access_url": OFFICIAL_COMPANION_PAPER_OPEN_ACCESS_URL,
        },
        "15_sufficiency_for_task_32": {
            "status": AUDIT_STATUS_DERIVED,
            "conclusion": "conditionally_sufficient",
            "reasoning": (
                "The vibration + temperature + motor_current triple is "
                "legitimately session-matched (same experimental run, "
                "consistent file counts, shared naming) for all 45 "
                "part1_load conditions, and is sufficient to design a "
                "genuine 3-modality multisensor observation once raw data "
                "is obtained. A full 4-modality observation including "
                "acoustic is NOT supported across the general condition "
                "space -- only for a narrow, currently-unconfirmed 0 Nm "
                "subset -- and must not be forced. Sample-level "
                "cross-DAQ timestamp synchronization is unverified and "
                "must be checked directly against raw files before any "
                "fine-grained (sub-session) fusion is implemented. Raw "
                "data acquisition into this environment is a prerequisite "
                "for any part of Task 32 beyond structural design."
            ),
            "blocking_for_task_32": not raw_available,
        },
        "part2_speed_sub_dataset": {
            "status": AUDIT_STATUS_EXCLUDED,
            "file_counts": PART2_SPEED_FILE_COUNTS,
            "fault_types": PART2_SPEED_FAULT_TYPES,
            "format": PART2_SPEED_FORMAT,
            "note": PART2_SPEED_NOTE,
        },
        "internal_consistency_checks": {
            "n_conditions_part1_derived": N_CONDITIONS_PART1,
            "n_vibration_files_part1_derived": N_VIBRATION_FILES_PART1_EXPECTED,
            "n_vibration_files_part1_source_stated": N_VIBRATION_FILES_PART1_SOURCE_STATED,
            "n_vibration_files_match": (
                N_VIBRATION_FILES_PART1_EXPECTED == N_VIBRATION_FILES_PART1_SOURCE_STATED
            ),
            "n_temperature_current_files_part1_derived": N_TEMPERATURE_CURRENT_FILES_PART1_EXPECTED,
            "n_temperature_current_files_part1_source_stated": N_TEMPERATURE_CURRENT_FILES_PART1_SOURCE_STATED,
            "n_temperature_current_files_match": (
                N_TEMPERATURE_CURRENT_FILES_PART1_EXPECTED
                == N_TEMPERATURE_CURRENT_FILES_PART1_SOURCE_STATED
            ),
        },
    }


def save_dataset_audit(path: str = AUDIT_JSON_PATH) -> Dict[str, object]:
    """Build and persist the Task 31 audit artifact."""
    audit = build_dataset_audit()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, default=str)

    return audit


if __name__ == "__main__":
    result = save_dataset_audit()
    print(f"Saved multimodal dataset audit to: {AUDIT_JSON_PATH}")
    print(f"Raw data available in this environment: {result['raw_data_available_in_environment']}")
    print(f"Task 32 readiness: {result['15_sufficiency_for_task_32']['conclusion']}")
