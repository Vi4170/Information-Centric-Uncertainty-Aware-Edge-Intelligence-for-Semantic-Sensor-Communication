"""MIMII-DG dataset integration pipeline."""

from .preprocessing import (
    RAW_DATA_DIR,
    SUMMARY_JSON_PATH,
    WINDOW_SIZE,
    list_machine_types,
    discover_recordings,
    read_mimii_wav,
    window_audio,
    build_recording_metadata,
    load_stream_windows,
    fit_train_normalization,
    apply_normalization,
    verify_split_disjoint,
    verify_no_recording_crosses_split,
    discover_all_machines_summary,
    save_mimii_dataset_summary,
)

__all__ = [
    "RAW_DATA_DIR",
    "SUMMARY_JSON_PATH",
    "WINDOW_SIZE",
    "list_machine_types",
    "discover_recordings",
    "read_mimii_wav",
    "window_audio",
    "build_recording_metadata",
    "load_stream_windows",
    "fit_train_normalization",
    "apply_normalization",
    "verify_split_disjoint",
    "verify_no_recording_crosses_split",
    "discover_all_machines_summary",
    "save_mimii_dataset_summary",
]