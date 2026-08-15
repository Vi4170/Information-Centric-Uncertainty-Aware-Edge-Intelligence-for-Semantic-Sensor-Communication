from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "CWRU"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "CWRU"

METADATA_FILE = PROCESSED_DIR / "cwru_metadata.csv"

OUTPUT_FILE = PROCESSED_DIR / "cwru_windows.npz"


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 2048
STEP_SIZE = 2048

MAX_WINDOWS_PER_FILE = 100


# ============================================================
# SIGNAL EXTRACTION
# ============================================================

def find_drive_end_signal(mat_data):
    """
    Find the Drive-End vibration signal in a CWRU .mat file.
    """

    candidates = []

    for key, value in mat_data.items():

        if key.startswith("__"):
            continue

        if not isinstance(value, np.ndarray):
            continue

        if value.size < 1000:
            continue

        key_lower = key.lower()

        if "_de_time" in key_lower:
            candidates.append(value)

    if not candidates:
        raise ValueError("No Drive-End vibration signal found.")

    signal = candidates[0].squeeze()

    return signal.astype(np.float64)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_signal(signal):
    """
    Standardize a vibration signal.
    """

    scaler = StandardScaler()

    signal = signal.reshape(-1, 1)

    normalized = scaler.fit_transform(signal)

    return normalized.squeeze()


# ============================================================
# WINDOWING
# ============================================================

def create_windows(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Split a 1-D signal into fixed-size windows.
    """

    windows = []

    for start in range(
        0,
        len(signal) - window_size + 1,
        step_size
    ):

        window = signal[start:start + window_size]

        windows.append(window)

        if len(windows) >= MAX_WINDOWS_PER_FILE:
            break

    return np.asarray(windows, dtype=np.float32)


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def main():

    print("=" * 60)
    print("CWRU SIGNAL PREPROCESSING")
    print("=" * 60)

    print(f"\nRaw dataset:")
    print(RAW_DIR)

    print(f"\nMetadata:")
    print(METADATA_FILE)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    metadata = pd.read_csv(METADATA_FILE)

    print(f"\nMetadata rows: {len(metadata)}")

    all_windows = []
    all_labels = []
    all_files = []

    processed_files = 0
    failed_files = 0

    for index, row in metadata.iterrows():

        filename = row["filename"]
        fault_type = row["fault_type"]

        file_path = RAW_DIR / filename

        print(
            f"\n[{index + 1}/{len(metadata)}] "
            f"{filename}"
        )

        if not file_path.exists():

            print("  ERROR: File not found.")

            failed_files += 1

            continue

        try:

            mat_data = loadmat(file_path)

            signal = find_drive_end_signal(mat_data)

            print(
                f"  Signal samples: {len(signal):,}"
            )

            signal = normalize_signal(signal)

            windows = create_windows(signal)

            print(
                f"  Windows created: {len(windows)}"
            )

            for window in windows:

                all_windows.append(window)
                all_labels.append(fault_type)
                all_files.append(filename)

            processed_files += 1

        except Exception as error:

            print(f"  ERROR: {error}")

            failed_files += 1

    # --------------------------------------------------------
    # CONVERT TO ARRAYS
    # --------------------------------------------------------

    X = np.asarray(all_windows, dtype=np.float32)

    y = np.asarray(all_labels)

    files = np.asarray(all_files)

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"\nFiles processed: {processed_files}")
    print(f"Files failed:    {failed_files}")

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nClass distribution:")

    print(
        pd.Series(y).value_counts()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        files=files
    )

    print("\nProcessed dataset saved to:")

    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()