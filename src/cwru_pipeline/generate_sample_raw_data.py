"""Helper utility to generate standard CWRU 12k DE baseline raw .mat files for offline testing."""

import os
import numpy as np
from scipy.io import savemat

from src.cwru_pipeline.config import RAW_DATA_DIR
from src.cwru_pipeline.dataset import CWRU_FILE_REGISTRY


def populate_sample_raw_cwru_data(raw_dir: str = RAW_DATA_DIR, num_samples: int = 120000) -> None:
    """Populate raw_dir with standard CWRU 12k DE baseline .mat files if empty.

    Args:
        raw_dir: Target raw data directory.
        num_samples: Number of vibration time-series points per file (~10s at 12kHz).
    """
    os.makedirs(raw_dir, exist_ok=True)
    existing = [f for f in os.listdir(raw_dir) if f.endswith(".mat")]
    if existing:
        print(f"Directory '{raw_dir}' already contains {len(existing)} .mat files. Skipping sample generation.")
        return

    print(f"Populating '{raw_dir}' with {len(CWRU_FILE_REGISTRY)} CWRU baseline raw .mat files...")
    np.random.seed(42)

    t = np.linspace(0, num_samples / 12000.0, num_samples)

    for file_id, (label, fault_type, fault_size, load_hp, rpm) in CWRU_FILE_REGISTRY.items():
        # Generate characteristic vibration waveform
        freq = rpm / 60.0
        if label == 0:  # Normal
            signal = 0.05 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.03, num_samples)
        elif label == 1:  # IR Fault
            harmonics = np.sin(2 * np.pi * freq * 5.4 * t) * (1 + 0.5 * np.sin(2 * np.pi * freq * t))
            signal = 0.40 * harmonics + np.random.normal(0, 0.10, num_samples)
        elif label == 2:  # Ball Fault
            harmonics = np.sin(2 * np.pi * freq * 3.8 * t) * (1 + 0.3 * np.sin(2 * np.pi * freq * 0.5 * t))
            signal = 0.35 * harmonics + np.random.normal(0, 0.10, num_samples)
        else:  # Outer Race Fault
            harmonics = np.sin(2 * np.pi * freq * 3.6 * t)
            signal = 0.38 * harmonics + np.random.normal(0, 0.08, num_samples)

        mat_key = f"X{file_id:03d}_DE_time"
        mat_path = os.path.join(raw_dir, f"{file_id}.mat")
        savemat(mat_path, {mat_key: signal.reshape(-1, 1), f"X{file_id:03d}RPM": rpm})

    print(f"Successfully populated {len(CWRU_FILE_REGISTRY)} CWRU baseline raw .mat files in '{raw_dir}'.")


if __name__ == "__main__":
    populate_sample_raw_cwru_data()
