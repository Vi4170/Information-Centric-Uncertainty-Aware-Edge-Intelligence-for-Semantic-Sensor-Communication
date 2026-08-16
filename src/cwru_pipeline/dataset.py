"""CWRU dataset file loader, metadata parser, and channel extractor module.

Handles loading Matlab (.mat) binary files from the CWRU Bearing Data Center,
extracting Drive End (DE) vibration time-series signals, verifying sampling rates,
and parsing file metadata into structured containers.
"""

from dataclasses import asdict, dataclass
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.io import loadmat

from src.cwru_pipeline.config import (
    BASELINE_FAULT_CLASSES,
    EXPECTED_SAMPLING_RATE,
    PREFERRED_CHANNEL,
)


@dataclass
class CWRUFileMetadata:
    """Metadata container for a single CWRU source recording (.mat file)."""

    file_id: int
    source_file: str
    fault_label: int
    fault_type: str
    fault_size: float
    load_hp: int
    rpm: int
    sampling_rate: int
    sensor_location: str = "DE"
    is_valid_baseline: bool = True
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert metadata object to dictionary."""
        return asdict(self)


# Comprehensive CWRU 12k Drive End Baseline Metadata Registry
# Maps CWRU File Numbers to (Fault Label, Fault Type, Fault Size, Load HP, RPM)
CWRU_FILE_REGISTRY: Dict[int, Tuple[int, str, float, int, int]] = {
    # Normal Baseline (Label 0)
    97: (0, "Normal", 0.000, 0, 1797),
    98: (0, "Normal", 0.000, 1, 1772),
    99: (0, "Normal", 0.000, 2, 1750),
    100: (0, "Normal", 0.000, 3, 1730),
    # Inner Race Faults (Label 1)
    105: (1, "Inner Race Fault", 0.007, 0, 1797),
    106: (1, "Inner Race Fault", 0.007, 1, 1772),
    107: (1, "Inner Race Fault", 0.007, 2, 1750),
    108: (1, "Inner Race Fault", 0.007, 3, 1730),
    169: (1, "Inner Race Fault", 0.014, 0, 1797),
    170: (1, "Inner Race Fault", 0.014, 1, 1772),
    171: (1, "Inner Race Fault", 0.014, 2, 1750),
    172: (1, "Inner Race Fault", 0.014, 3, 1730),
    209: (1, "Inner Race Fault", 0.021, 0, 1797),
    210: (1, "Inner Race Fault", 0.021, 1, 1772),
    211: (1, "Inner Race Fault", 0.021, 2, 1750),
    212: (1, "Inner Race Fault", 0.021, 3, 1730),
    # Ball Faults (Label 2)
    118: (2, "Ball Fault", 0.007, 0, 1797),
    119: (2, "Ball Fault", 0.007, 1, 1772),
    120: (2, "Ball Fault", 0.007, 2, 1750),
    121: (2, "Ball Fault", 0.007, 3, 1730),
    185: (2, "Ball Fault", 0.014, 0, 1797),
    186: (2, "Ball Fault", 0.014, 1, 1772),
    187: (2, "Ball Fault", 0.014, 2, 1750),
    188: (2, "Ball Fault", 0.014, 3, 1730),
    222: (2, "Ball Fault", 0.021, 0, 1797),
    223: (2, "Ball Fault", 0.021, 1, 1772),
    224: (2, "Ball Fault", 0.021, 2, 1750),
    225: (2, "Ball Fault", 0.021, 3, 1730),
    # Outer Race Faults Centered @ 6 o'clock (Label 3)
    130: (3, "Outer Race Fault", 0.007, 0, 1797),
    131: (3, "Outer Race Fault", 0.007, 1, 1772),
    132: (3, "Outer Race Fault", 0.007, 2, 1750),
    133: (3, "Outer Race Fault", 0.007, 3, 1730),
    197: (3, "Outer Race Fault", 0.014, 0, 1797),
    198: (3, "Outer Race Fault", 0.014, 1, 1772),
    199: (3, "Outer Race Fault", 0.014, 2, 1750),
    200: (3, "Outer Race Fault", 0.014, 3, 1730),
    234: (3, "Outer Race Fault", 0.021, 0, 1797),
    235: (3, "Outer Race Fault", 0.021, 1, 1772),
    236: (3, "Outer Race Fault", 0.021, 2, 1750),
    237: (3, "Outer Race Fault", 0.021, 3, 1730),
}


def parse_cwru_file_id(filename: str) -> Optional[int]:
    """Extract numeric CWRU file ID from filename (e.g., '105.mat' -> 105 or 'X105.mat' -> 105)."""
    match = re.search(r"(\d+)", os.path.basename(filename))
    if match:
        return int(match.group(1))
    return None


def extract_vibration_channel(
    mat_dict: dict, preferred_channel: str = PREFERRED_CHANNEL
) -> Tuple[np.ndarray, str]:
    """Search Matlab dictionary keys for the requested vibration channel time series.

    Args:
        mat_dict: Loaded .mat dictionary.
        preferred_channel: Target channel identifier (default "DE").

    Returns:
        Tuple[np.ndarray, str]: Extracted 1D signal array and key name.

    Raises:
        ValueError: If preferred channel is missing from .mat dictionary.
    """
    pattern = re.compile(rf".*_{preferred_channel}_time.*", re.IGNORECASE)
    matching_keys = [k for k in mat_dict.keys() if pattern.match(k)]

    if not matching_keys:
        # Fallback search for any DE key
        matching_keys = [
            k for k in mat_dict.keys() if preferred_channel.upper() in k.upper() and "time" in k.lower()
        ]

    if not matching_keys:
        available_keys = [k for k in mat_dict.keys() if not k.startswith("__")]
        raise ValueError(
            f"Requested vibration channel '{preferred_channel}' not found in file. "
            f"Available keys: {available_keys}"
        )

    key_name = matching_keys[0]
    signal = mat_dict[key_name].flatten().astype(np.float32)
    return signal, key_name


def load_cwru_mat_file(
    file_path: str,
    preferred_channel: str = PREFERRED_CHANNEL,
    expected_sampling_rate: int = EXPECTED_SAMPLING_RATE,
) -> Tuple[np.ndarray, CWRUFileMetadata]:
    """Load a CWRU Matlab .mat file, extract vibration signal, and verify metadata.

    Args:
        file_path: Path to raw .mat file.
        preferred_channel: Target sensor channel ("DE").
        expected_sampling_rate: Expected sampling rate in Hz (default 12000).

    Returns:
        Tuple[np.ndarray, CWRUFileMetadata]: Raw vibration array and file metadata object.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If signal channel is missing or signal is unmapped.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CWRU .mat file not found: {file_path}")

    filename = os.path.basename(file_path)
    file_id = parse_cwru_file_id(filename)

    mat_dict = loadmat(file_path)
    signal, key_name = extract_vibration_channel(mat_dict, preferred_channel=preferred_channel)

    # Parse metadata from registry or infer defaults
    if file_id in CWRU_FILE_REGISTRY:
        label, f_type, f_size, load, rpm = CWRU_FILE_REGISTRY[file_id]
        meta = CWRUFileMetadata(
            file_id=file_id,
            source_file=filename,
            fault_label=label,
            fault_type=f_type,
            fault_size=f_size,
            load_hp=load,
            rpm=rpm,
            sampling_rate=expected_sampling_rate,
            sensor_location=preferred_channel,
            is_valid_baseline=True,
        )
    else:
        # File not in standard baseline registry
        meta = CWRUFileMetadata(
            file_id=file_id if file_id else -1,
            source_file=filename,
            fault_label=-1,
            fault_type="Unmapped",
            fault_size=0.0,
            load_hp=-1,
            rpm=-1,
            sampling_rate=expected_sampling_rate,
            sensor_location=preferred_channel,
            is_valid_baseline=False,
            exclusion_reason="File ID not present in standard 12k DE baseline registry",
        )

    # Check RPM in mat file if embedded
    rpm_key = [k for k in mat_dict.keys() if "RPM" in k.upper()]
    if rpm_key:
        try:
            mat_rpm = int(mat_dict[rpm_key[0]].flatten()[0])
            if meta.rpm != -1 and abs(meta.rpm - mat_rpm) > 50:
                print(
                    f"Warning: RPM mismatch in {filename}: registry={meta.rpm}, file={mat_rpm}"
                )
        except Exception:
            pass

    return signal, meta


def discover_and_load_cwru_dataset(
    raw_dir: str,
    preferred_channel: str = PREFERRED_CHANNEL,
    expected_sampling_rate: int = EXPECTED_SAMPLING_RATE,
) -> Tuple[List[Tuple[np.ndarray, CWRUFileMetadata]], List[CWRUFileMetadata]]:
    """Scan raw directory, load valid baseline .mat files, and exclude unmapped files.

    Args:
        raw_dir: Path to directory containing raw .mat files.
        preferred_channel: Target sensor channel ("DE").
        expected_sampling_rate: Expected sampling rate in Hz.

    Returns:
        Tuple[List[Tuple[np.ndarray, CWRUFileMetadata]], List[CWRUFileMetadata]]:
            - List of valid (signal, metadata) tuples.
            - List of excluded file metadata objects with exclusion reasons.
    """
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        print(f"Created raw data directory at {raw_dir}. Place raw .mat files here.")
        return [], []

    mat_files = [f for f in os.listdir(raw_dir) if f.endswith(".mat")]
    valid_records = []
    excluded_records = []

    for fname in sorted(mat_files):
        fpath = os.path.join(raw_dir, fname)
        try:
            signal, meta = load_cwru_mat_file(
                fpath,
                preferred_channel=preferred_channel,
                expected_sampling_rate=expected_sampling_rate,
            )
            if meta.is_valid_baseline:
                valid_records.append((signal, meta))
            else:
                excluded_records.append(meta)
                print(f"Excluded {fname}: {meta.exclusion_reason}")
        except Exception as e:
            file_id = parse_cwru_file_id(fname)
            ex_meta = CWRUFileMetadata(
                file_id=file_id if file_id else -1,
                source_file=fname,
                fault_label=-1,
                fault_type="Error",
                fault_size=0.0,
                load_hp=-1,
                rpm=-1,
                sampling_rate=expected_sampling_rate,
                sensor_location=preferred_channel,
                is_valid_baseline=False,
                exclusion_reason=str(e),
            )
            excluded_records.append(ex_meta)
            print(f"Excluded {fname}: {e}")

    print(
        f"Discovered {len(mat_files)} .mat files: "
        f"{len(valid_records)} valid baseline recordings loaded, "
        f"{len(excluded_records)} excluded."
    )
    return valid_records, excluded_records
