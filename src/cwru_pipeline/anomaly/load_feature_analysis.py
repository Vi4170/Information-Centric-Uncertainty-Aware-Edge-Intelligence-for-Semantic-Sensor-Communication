import numpy as np
import re


DATA_PATH = "data/processed/CWRU/cwru_splits.npz"
FEATURE_PATH = "results/anomaly/extracted_features.npz"
OUTPUT_PATH = "results/anomaly/load_feature_analysis.npz"


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(DATA_PATH, allow_pickle=True)
features = np.load(FEATURE_PATH, allow_pickle=True)


# ============================================================
# COMBINE ALL SPLITS
# ============================================================

F_train = features["F_train"]
F_val = features["F_val"]
F_test = features["F_test"]

y_train = features["y_train"]
y_val = features["y_val"]
y_test = features["y_test"]

files_train = data["files_train"]
files_val = data["files_val"]
files_test = data["files_test"]


F = np.concatenate(
    [F_train, F_val, F_test],
    axis=0
)

y = np.concatenate(
    [y_train, y_val, y_test],
    axis=0
)

files = np.concatenate(
    [files_train, files_val, files_test],
    axis=0
)


# ============================================================
# FEATURE NAMES
# ============================================================

names = [
    "mean",
    "std",
    "variance",
    "rms",
    "peak",
    "peak_to_peak",
    "abs_mean",
    "skewness",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "band_1",
    "band_2",
    "band_3",
    "band_4",
    "band_5",
    "band_6",
    "band_7",
]


# ============================================================
# EXTRACT LOAD
# ============================================================

def get_load(filename):

    match = re.search(
        r"_(0HP|1HP|2HP|3HP)",
        str(filename)
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


loads = np.array([
    get_load(f)
    for f in files
])


# ============================================================
# BASIC DATA CHECK
# ============================================================

print("=" * 70)
print("LOAD-WISE NORMAL FEATURE ANALYSIS")
print("=" * 70)

print("\nCombined feature matrix:", F.shape)
print("Combined labels:", y.shape)
print("Combined filenames:", files.shape)

print("\nDataset composition:")

for load in ["0HP", "1HP", "2HP", "3HP"]:

    load_mask = loads == load

    normal_count = np.sum(
        load_mask & (y == "Normal")
    )

    anomaly_count = np.sum(
        load_mask & (y != "Normal")
    )

    print(
        f"{load}: "
        f"total={np.sum(load_mask)}, "
        f"normal={normal_count}, "
        f"anomaly={anomaly_count}"
    )


# ============================================================
# NORMAL FEATURE MEANS BY LOAD
# ============================================================

print("\n" + "=" * 70)
print("NORMAL FEATURE MEANS BY LOAD")
print("=" * 70)

normal_by_load = {}


for load in ["0HP", "1HP", "2HP", "3HP"]:

    mask = (
        (loads == load)
        & (y == "Normal")
    )

    F_load = F[mask]

    normal_by_load[load] = F_load

    print("\n" + "-" * 70)
    print(
        f"{load} | Normal samples: {len(F_load)}"
    )
    print("-" * 70)

    for i, name in enumerate(names):

        mean_value = np.mean(F_load[:, i])
        std_value = np.std(F_load[:, i])

        print(
            f"{name:22s}"
            f" mean={mean_value:12.6f}"
            f" std={std_value:12.6f}"
        )


# ============================================================
# LOAD VARIATION RATIO
# ============================================================

print("\n")
print("=" * 70)
print("LOAD VARIATION RATIO")
print("=" * 70)

print(
    f"{'FEATURE':22s}"
    f"{'MIN MEAN':>14s}"
    f"{'MAX MEAN':>14s}"
    f"{'RATIO':>14s}"
)

print("-" * 70)


load_names = ["0HP", "1HP", "2HP", "3HP"]

variation_ratios = []


for i, name in enumerate(names):

    means = np.array([
        np.mean(normal_by_load[load][:, i])
        for load in load_names
    ])

    min_mean = np.min(np.abs(means))
    max_mean = np.max(np.abs(means))

    ratio = max_mean / (
        min_mean + 1e-8
    )

    variation_ratios.append(ratio)

    print(
        f"{name:22s}"
        f"{min_mean:14.6f}"
        f"{max_mean:14.6f}"
        f"{ratio:14.3f}"
    )


# ============================================================
# LOAD-INVARIANT FEATURE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("MOST LOAD-STABLE FEATURES")
print("=" * 70)

ranked_indices = np.argsort(
    variation_ratios
)

for rank, i in enumerate(
    ranked_indices[:10],
    start=1
):

    print(
        f"{rank:2d}. "
        f"{names[i]:22s} "
        f"ratio={variation_ratios[i]:.3f}"
    )


# ============================================================
# SAVE
# ============================================================

np.savez(
    OUTPUT_PATH,

    features=F,
    labels=y,
    files=files,
    loads=loads,

    load_0hp_normal=normal_by_load["0HP"],
    load_1hp_normal=normal_by_load["1HP"],
    load_2hp_normal=normal_by_load["2HP"],
    load_3hp_normal=normal_by_load["3HP"],

    feature_names=np.array(names),

    variation_ratios=np.array(
        variation_ratios
    ),
)


print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("LOAD FEATURE ANALYSIS COMPLETE")
print("=" * 70)