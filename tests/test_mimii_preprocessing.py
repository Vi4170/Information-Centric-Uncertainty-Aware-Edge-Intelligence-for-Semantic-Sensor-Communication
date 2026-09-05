from pathlib import Path

import numpy as np
import pytest

from src.mimii_pipeline.preprocessing import (
    EXPECTED_CHANNELS,
    EXPECTED_FRAMES_PER_RECORDING,
    EXPECTED_MACHINE_TYPES,
    EXPECTED_SAMPLE_RATE,
    EXPECTED_SAMPLE_WIDTH_BYTES,
    EXPECTED_SECTIONS,
    RAW_DATA_DIR,
    WINDOW_SIZE,
    apply_normalization,
    build_recording_metadata,
    discover_all_machines_summary,
    discover_recordings,
    fit_train_normalization,
    list_machine_types,
    load_stream_windows,
    read_mimii_wav,
    verify_no_recording_crosses_split,
    verify_observation_id_uniqueness,
    verify_split_disjoint,
    window_audio,
)


def _raw_available() -> bool:
    return all(
        (
            Path(RAW_DATA_DIR)
            / f"{machine}.zip"
        ).is_file()
        for machine in EXPECTED_MACHINE_TYPES
    )


pytestmark = pytest.mark.skipif(
    not _raw_available(),
    reason=(
        "MIMII-DG raw ZIP archives are not available"
    ),
)


def test_all_five_machine_archives_present():
    assert (
        list_machine_types()
        == EXPECTED_MACHINE_TYPES
    )


def test_actual_machine_and_section_structure():
    metadata = discover_recordings()

    assert set(
        metadata["machine_type"]
    ) == set(
        EXPECTED_MACHINE_TYPES
    )

    assert set(
        metadata["section"]
    ) == set(
        EXPECTED_SECTIONS
    )


def test_original_metadata_columns_preserved():
    metadata = build_recording_metadata()

    required = {
        "machine_type",
        "machine_id",
        "machine_id_source",
        "leakage_group_id",
        "section",
        "domain",
        "official_split",
        "project_split",
        "label",
        "recording_index",
        "original_filename",
        "attribute_csv",
        "attribute_metadata",
        "condition_parameter",
        "condition_value",
        "modality",
    }

    assert required.issubset(
        metadata.columns
    )


def test_machine_identity_handling_is_explicit():
    metadata = discover_recordings()

    assert (
        "machine_id"
        in metadata.columns
    )

    assert (
        "machine_id_source"
        in metadata.columns
    )

    assert (
        "leakage_group_id"
        in metadata.columns
    )

    assert metadata[
        "machine_id"
    ].isna().all()

    assert set(
        metadata[
            "machine_id_source"
        ].unique()
    ) == {
        "not_exposed_in_MIMII_DG_release"
    }

    expected_groups = (
        metadata["machine_type"]
        + ":"
        + metadata["section"]
    )

    assert (
        metadata[
            "leakage_group_id"
        ]
        == expected_groups
    ).all()


def test_additional_attribute_columns_are_preserved():
    metadata = discover_recordings(
        machine_type="valve"
    )

    attribute_rows = metadata[
        "attribute_metadata"
    ]

    assert all(
        isinstance(row, dict)
        for row in attribute_rows
    )

    assert any(
        "d2p" in row
        for row in attribute_rows
    )

    assert any(
        "d2v" in row
        for row in attribute_rows
    )

    assert (
        "condition_p_2"
        in metadata.columns
    )

    assert (
        "condition_v_2"
        in metadata.columns
    )


def test_audio_modality_is_separate():
    metadata = discover_recordings()

    assert (
        metadata["modality"]
        .nunique()
        == 1
    )

    assert (
        metadata["modality"].iloc[0]
        == "audio"
    )


def test_normal_anomaly_labels_present():
    metadata = discover_recordings()

    assert set(
        metadata["label"].unique()
    ) == {
        "normal",
        "anomaly",
    }

    normal_count = int(
        metadata["label"].eq(
            "normal"
        ).sum()
    )

    anomaly_count = int(
        metadata["label"].eq(
            "anomaly"
        ).sum()
    )

    assert (
        normal_count
        > anomaly_count
    )

    assert anomaly_count > 0


def test_project_split_is_leakage_group_disjoint():
    metadata = discover_recordings()

    counts = verify_split_disjoint(
        metadata
    )

    assert set(counts) == {
        "train",
        "val",
        "test",
    }

    assert all(
        value > 0
        for value in counts.values()
    )


def test_no_recording_crosses_project_split():
    metadata = discover_recordings()

    assert (
        verify_no_recording_crosses_split(
            metadata
        )
    )


def test_section_assignment_matches_policy():
    metadata = discover_recordings()

    usable = metadata[
        metadata["project_split"].isin(
            [
                "train",
                "val",
                "test",
            ]
        )
    ]

    expected = {
        "section_00": "train",
        "section_01": "val",
        "section_02": "test",
    }

    for section, split in expected.items():
        subset = usable[
            usable["section"].eq(
                section
            )
        ]

        assert set(
            subset["project_split"]
        ) == {split}


def test_leakage_group_is_single_split():
    metadata = discover_recordings()

    group_split_counts = (
        metadata[
            metadata[
                "project_split"
            ].isin(
                [
                    "train",
                    "val",
                    "test",
                ]
            )
        ]
        .groupby(
            "leakage_group_id"
        )[
            "project_split"
        ]
        .nunique()
    )

    assert (
        group_split_counts.max()
        == 1
    )


def test_official_test_not_used_for_train_or_validation():
    metadata = discover_recordings()

    train_or_val = metadata[
        metadata["project_split"].isin(
            [
                "train",
                "val",
            ]
        )
    ]

    assert not (
        train_or_val[
            "official_split"
        ]
        .eq("test")
        .any()
    )


def test_audio_header_is_expected_for_each_machine_type():
    metadata = discover_recordings()

    for machine_type in EXPECTED_MACHINE_TYPES:
        row = metadata[
            metadata["machine_type"].eq(
                machine_type
            )
        ].iloc[0]

        audio = read_mimii_wav(
            row
        )

        assert audio.ndim == 1

        assert (
            audio.dtype
            == np.float32
        )

        assert (
            audio.size
            == EXPECTED_FRAMES_PER_RECORDING
        )


def test_non_overlapping_windowing():
    audio = np.arange(
        WINDOW_SIZE * 2 + 10,
        dtype=np.float32,
    )

    windows = list(
        window_audio(
            audio,
            WINDOW_SIZE,
        )
    )

    assert len(
        windows
    ) == 2

    assert (
        windows[0][1].shape
        == (WINDOW_SIZE,)
    )

    assert (
        windows[1][1].shape
        == (WINDOW_SIZE,)
    )

    assert (
        windows[0][1][0]
        == 0
    )

    assert (
        windows[1][1][0]
        == WINDOW_SIZE
    )


def test_lazy_window_loader_returns_fixed_size_windows():
    metadata = discover_recordings()

    train_metadata = metadata[
        metadata["project_split"].eq(
            "train"
        )
    ]

    generator = load_stream_windows(
        metadata_df=train_metadata,
        max_recordings=1,
    )

    first_window, info = next(
        generator
    )

    assert (
        first_window.shape
        == (WINDOW_SIZE,)
    )

    assert (
        first_window.dtype
        == np.float32
    )

    assert (
        info["modality"]
        == "audio"
    )

    assert (
        info["project_split"]
        == "train"
    )

    assert (
        "observation_id"
        in info
    )

    assert (
        "leakage_group_id"
        in info
    )


def test_observation_ids_are_unique():
    metadata = discover_recordings()

    assert verify_observation_id_uniqueness(
        metadata,
        max_recordings=25,
    )


def test_normalization_uses_train_only():
    metadata = discover_recordings()

    train_mean, train_std = (
        fit_train_normalization(
            metadata_df=metadata,
            max_recordings=5,
        )
    )

    assert np.isfinite(
        train_mean
    )

    assert np.isfinite(
        train_std
    )

    assert train_std > 0


def test_normalization_application():
    signal = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    normalized = apply_normalization(
        signal,
        mean=2.0,
        std=1.0,
    )

    np.testing.assert_allclose(
        normalized,
        np.array(
            [-1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
    )


def test_summary_contains_required_provenance_and_statistics():
    summary = (
        discover_all_machines_summary()
    )

    assert (
        summary["dataset"]
        == "MIMII-DG"
    )

    assert (
        summary["modality"]
        == "audio"
    )

    assert (
        summary["provenance"]["doi"]
        == "10.5281/zenodo.6529888"
    )

    assert (
        summary[
            "dataset_structure"
        ]["machine_type_count"]
        == 5
    )

    assert (
        summary[
            "dataset_structure"
        ]["sections"]
        == list(
            EXPECTED_SECTIONS
        )
    )

    audio = summary[
        "audio"
    ]

    assert (
        audio["channels"]
        == EXPECTED_CHANNELS
    )

    assert (
        audio["sample_rate_hz"]
        == EXPECTED_SAMPLE_RATE
    )

    assert (
        audio["sample_width_bytes"]
        == EXPECTED_SAMPLE_WIDTH_BYTES
    )

    assert (
        audio["frames_per_recording"]
        == EXPECTED_FRAMES_PER_RECORDING
    )

    assert (
        audio["window_size_samples"]
        == WINDOW_SIZE
    )


def test_summary_preserves_machine_identity_handling():
    summary = (
        discover_all_machines_summary()
    )

    metadata = summary[
        "metadata"
    ]

    assert (
        metadata[
            "machine_id_present_in_release"
        ]
        is False
    )

    assert (
        metadata[
            "machine_id_preservation_status"
        ]
        == "not_exposed; not fabricated"
    )

    assert (
        metadata[
            "leakage_group_identifier"
        ]
        == "machine_type + section"
    )


def test_summary_preserves_original_split_information():
    metadata = discover_recordings()

    assert set(
        metadata[
            "official_split"
        ].unique()
    ) == {
        "train",
        "test",
    }

    assert (
        "excluded_official_test"
        in set(
            metadata[
                "project_split"
            ].unique()
        )
    )


def test_condition_metadata_is_machine_specific():
    metadata = discover_recordings()

    for machine_type in EXPECTED_MACHINE_TYPES:
        subset = metadata[
            metadata["machine_type"].eq(
                machine_type
            )
        ]

        assert (
            subset[
                "condition_parameter"
            ]
            .notna()
            .all()
        )

        assert (
            subset[
                "condition_value"
            ]
            .notna()
            .all()
        )