"""IMS run-to-failure temporal / novelty progression analysis.

IMS provides only run-level failure descriptions (which bearing failed, by
which mechanism, at the end of a run) with no timestamped onset and no
per-window ground truth (src/ims_pipeline/preprocessing.py's
RUN_FAILURE_DESCRIPTIONS is explicit about this). Treating that run-level
information as a per-window label, or inventing a time-threshold label, would
misrepresent the data. No such label is constructed anywhere in this module.

Consequently, the canonical supervised CNN classifier is not trained on IMS
here: there is no legitimate per-window target to train it against. Task
Relevance and Uncertainty both require a trained class-probability
distribution, which therefore also does not exist for IMS; they are reported
as not computed, not approximated. Communication Cost is a constant,
config-driven placeholder unrelated to labels, so it transfers unchanged. The
full 5-term canonical VoI score is consequently not computed for IMS -- see
the module-level ANALYSIS_TARGETS and the manifest written by
run_ims_temporal_analysis() for the explicit accounting.

Novelty and Temporal Importance require no labels and are computed directly
on raw (leakage-safe, train-only-normalized) window signal -- not a CNN
embedding, since no legitimately trained embedding exists for IMS. The
Novelty reference centroid is fit strictly on each target's own "initial"
split only, per src/ims_pipeline/preprocessing.py's existing chronological,
leakage-safe split (unmodified). Temporal Importance uses the existing
DEFAULT_TEMPORAL_CHANGE_SCALE unmodified (not re-derived for IMS).
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ims_pipeline.preprocessing import (
    RUN_FAILURE_DESCRIPTIONS,
    SPLIT_NAMES,
    WINDOW_SIZE,
    apply_normalization,
    fit_initial_normalization,
    load_stream_windows,
    verify_chronological_split_order,
    verify_no_test_leakage_into_normalization,
    verify_observation_id_uniqueness,
    verify_split_disjoint,
)
from src.novelty.novelty import DistanceNoveltyDetector
from src.temporal.temporal import compute_temporal_importance
from src.communication.cost import compute_communication_cost
from src.communication.config import MAX_PAYLOAD_SIZE, REFERENCE_BANDWIDTH

TABLE_DIR = "results/tables"
FIGURE_DIR = "results/figures"
MANIFEST_PATH = os.path.join("data", "processed", "ims", "ims_temporal_experiment_manifest.json")

NOMINAL_PAYLOAD_BYTES = MAX_PAYLOAD_SIZE
NOMINAL_BANDWIDTH = REFERENCE_BANDWIDTH

ANALYSIS_TARGETS: List[Dict[str, object]] = [
    {"run_id": "1st_test", "bearing_id": 3, "channel_index": 0, "role": "documented_failure",
     "failure_note": "Vendor readme: inner race defect in bearing 3 by end of run (run-level only)."},
    {"run_id": "1st_test", "bearing_id": 4, "channel_index": 0, "role": "documented_failure",
     "failure_note": "Vendor readme: roller element defect in bearing 4 by end of run (run-level only)."},
    {"run_id": "1st_test", "bearing_id": 1, "channel_index": 0, "role": "control",
     "failure_note": "No documented failure for this bearing in this run."},
    {"run_id": "2nd_test", "bearing_id": 1, "channel_index": 0, "role": "documented_failure",
     "failure_note": "Vendor readme: outer race failure in bearing 1 by end of run (run-level only)."},
    {"run_id": "2nd_test", "bearing_id": 2, "channel_index": 0, "role": "control",
     "failure_note": "No documented failure for this bearing in this run."},
    {"run_id": "3rd_test", "bearing_id": 3, "channel_index": 0, "role": "documented_failure",
     "failure_note": "Vendor readme: outer race failure in bearing 3 by end of run (run-level only)."},
    {"run_id": "3rd_test", "bearing_id": 1, "channel_index": 0, "role": "control",
     "failure_note": "No documented failure for this bearing in this run."},
    {"run_id": "4th_test", "bearing_id": 3, "channel_index": 0, "role": "undocumented_continuation",
     "failure_note": (
         "Not a documented failure event: this run_id is the undocumented continuation of the "
         "same physical rig/bearing as 3rd_test's bearing 3 (see RUN_FAILURE_DESCRIPTIONS['4th_test']); "
         "vendor readme asserts no label for this region."
     )},
]

NOT_COMPUTED_COMPONENTS = {
    "uncertainty": (
        "Requires a trained class-probability distribution. IMS provides no legitimate "
        "per-window label to train a classifier against; not approximated."
    ),
    "task_relevance": (
        "Requires a trained class-probability distribution (relevance_from_probabilities) or a "
        "predicted class id (relevance_from_class). Same reason as uncertainty: no legitimate "
        "per-window classifier exists for IMS."
    ),
    "voi_score_and_decision": (
        "The canonical VoI formula requires all five components. With task_relevance and "
        "uncertainty unavailable, the composite VoI score and DISCARD/BUFFER/SUMMARY/TRANSMIT "
        "decision policy are not computed for IMS rather than substituted with placeholder values."
    ),
}


def _compute_cost_batch(n: int) -> np.ndarray:
    cost = compute_communication_cost(
        payload_size=NOMINAL_PAYLOAD_BYTES,
        transmission_time=NOMINAL_PAYLOAD_BYTES / NOMINAL_BANDWIDTH,
        available_bandwidth=NOMINAL_BANDWIDTH,
    )
    return np.full(n, cost, dtype=np.float32)


def analyze_target(target: Dict[str, object]) -> pd.DataFrame:
    run_id = target["run_id"]
    bearing_id = target["bearing_id"]
    channel_index = target["channel_index"]

    X, meta = load_stream_windows(run_id, bearing_id, channel_index)
    order = meta.sort_values(["chronological_order_index", "window_index"]).index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    X = X[order]

    verify_observation_id_uniqueness(meta)
    verify_split_disjoint(meta)
    verify_chronological_split_order(meta)
    verify_no_test_leakage_into_normalization(meta)

    mean, std = fit_initial_normalization(X, meta)
    X_norm = apply_normalization(X, mean, std)
    flat = X_norm.reshape(X_norm.shape[0], -1)

    initial_mask = (meta["split"] == "initial").to_numpy()
    detector = DistanceNoveltyDetector(embedding_dim=flat.shape[1], reference_class=None)
    detector.fit(flat[initial_mask])
    novelty = detector.score(flat)

    temporal_importance = compute_temporal_importance(flat)
    resource_cost = _compute_cost_batch(len(X))

    n_files = int(meta["chronological_order_index"].max()) + 1
    trajectory_position = meta["chronological_order_index"].to_numpy(dtype=np.float64) / max(n_files - 1, 1)

    result = meta.copy()
    result["run_id"] = run_id
    result["bearing_id"] = bearing_id
    result["channel_index"] = channel_index
    result["role"] = target["role"]
    result["failure_note"] = target["failure_note"]
    result["trajectory_position"] = trajectory_position
    result["novelty"] = novelty
    result["temporal_importance"] = temporal_importance
    result["resource_cost"] = resource_cost
    result["normalization_mean"] = mean
    result["normalization_std"] = std
    return result


def build_progression_summary(all_results: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in all_results:
        run_id = df["run_id"].iloc[0]
        bearing_id = df["bearing_id"].iloc[0]
        role = df["role"].iloc[0]
        for split_name in SPLIT_NAMES:
            split_df = df[df["split"] == split_name]
            if split_df.empty:
                continue
            row = {
                "run_id": run_id, "bearing_id": bearing_id, "role": role, "split": split_name,
                "n": len(split_df),
            }
            for col in ("novelty", "temporal_importance", "resource_cost"):
                row[f"{col}_mean"] = float(split_df[col].mean())
                row[f"{col}_median"] = float(split_df[col].median())
                row[f"{col}_std"] = float(split_df[col].std()) if len(split_df) > 1 else 0.0
                row[f"{col}_min"] = float(split_df[col].min())
                row[f"{col}_max"] = float(split_df[col].max())
            rows.append(row)
    return pd.DataFrame(rows)


def build_failure_proximity_correlation(all_results: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in all_results:
        row = {
            "run_id": df["run_id"].iloc[0],
            "bearing_id": df["bearing_id"].iloc[0],
            "role": df["role"].iloc[0],
            "failure_note": df["failure_note"].iloc[0],
            "n_windows": len(df),
            "corr_trajectory_position_novelty": float(df["trajectory_position"].corr(df["novelty"])),
            "corr_trajectory_position_temporal_importance": float(
                df["trajectory_position"].corr(df["temporal_importance"])
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_progression(all_results: List[pd.DataFrame], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(len(all_results), 1, figsize=(11, 3.2 * len(all_results)), sharex=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, df in zip(axes, all_results):
        run_id = df["run_id"].iloc[0]
        bearing_id = df["bearing_id"].iloc[0]
        role = df["role"].iloc[0]
        x = df["chronological_order_index"].to_numpy()

        ax.plot(x, df["novelty"].to_numpy(), color="#c00000", alpha=0.6, linewidth=0.8, label="Novelty")
        ax.plot(x, df["temporal_importance"].to_numpy(), color="#4472c4", alpha=0.6, linewidth=0.8, label="Temporal Importance")

        for split_name in ("initial", "adaptation", "test"):
            split_positions = np.flatnonzero((df["split"] == split_name).to_numpy())
            if split_positions.size == 0:
                continue
            ax.axvline(x[split_positions[0]], color="gray", linestyle="--", alpha=0.5)

        ax.set_title(f"{run_id} bearing {bearing_id} ({role})", fontsize=10)
        ax.set_ylabel("Score [0,1]", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Chronological file index", fontsize=10)
    fig.suptitle("IMS Novelty and Temporal Importance vs. Run Progression", fontsize=13, y=1.0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_ims_temporal_analysis() -> Dict[str, object]:
    print("=== Executing IMS Run-to-Failure Temporal/Novelty Analysis ===")

    all_results = [analyze_target(target) for target in ANALYSIS_TARGETS]

    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    summary_df = build_progression_summary(all_results)
    summary_path = os.path.join(TABLE_DIR, "ims_temporal_progression_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved progression summary table to: {summary_path}")

    correlation_df = build_failure_proximity_correlation(all_results)
    correlation_path = os.path.join(TABLE_DIR, "ims_failure_proximity_correlation.csv")
    correlation_df.to_csv(correlation_path, index=False)
    print(f"Saved failure proximity correlation table to: {correlation_path}")

    per_obs_cols = [
        "observation_id", "run_id", "bearing_id", "channel_index", "role", "split",
        "chronological_order_index", "window_index", "trajectory_position",
        "novelty", "temporal_importance", "resource_cost",
    ]
    test_rows = pd.concat([df[df["split"] == "test"][per_obs_cols] for df in all_results], ignore_index=True)
    per_obs_path = os.path.join(TABLE_DIR, "ims_per_observation_test_split.csv")
    test_rows.to_csv(per_obs_path, index=False)
    print(f"Saved per-observation (test split) table to: {per_obs_path}")

    plot_path = os.path.join(FIGURE_DIR, "ims_novelty_temporal_over_time.png")
    plot_progression(all_results, plot_path)
    print(f"Saved progression plot to: {plot_path}")

    manifest = {
        "dataset": "ims",
        "experiment": "run-to-failure temporal and novelty progression analysis (no supervised classification)",
        "analysis_targets": ANALYSIS_TARGETS,
        "reference_population": (
            "Per-(run_id, bearing_id, channel_index) target: mean of that target's own 'initial' "
            "split raw-window signal (flattened, train-only normalized). No cross-run or "
            "cross-bearing pooling of reference populations."
        ),
        "normalization": {
            "method": "global scalar z-score per target, fit on 'initial' split only",
            "per_target_train_mean_std": {
                f"{t['run_id']}_b{t['bearing_id']}_c{t['channel_index']}": {
                    "mean": float(df["normalization_mean"].iloc[0]),
                    "std": float(df["normalization_std"].iloc[0]),
                }
                for t, df in zip(ANALYSIS_TARGETS, all_results)
            },
        },
        "novelty_methodology": (
            "DistanceNoveltyDetector operating on raw flattened window signal "
            f"(embedding_dim={WINDOW_SIZE}), not a CNN embedding, since no legitimately trained "
            "embedding exists for IMS (see module docstring). Formula, fitting, and scoring code "
            "unmodified from src/novelty/novelty.py."
        ),
        "temporal_methodology": (
            "compute_temporal_importance applied unmodified with the existing default "
            "DEFAULT_TEMPORAL_CHANGE_SCALE (not re-derived for IMS), on raw flattened window "
            "signal in verified chronological order within each target."
        ),
        "not_computed": NOT_COMPUTED_COMPONENTS,
        "leakage_checks": (
            "verify_observation_id_uniqueness, verify_split_disjoint, verify_chronological_split_order, "
            "and verify_no_test_leakage_into_normalization (all unmodified, from src/ims_pipeline/preprocessing.py) "
            "were run and passed for every analysis target; see the full test suite for enforcement."
        ),
        "random_seed": 42,
    }

    import json

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    print(f"Saved experiment manifest to: {MANIFEST_PATH}")

    print("=== IMS Temporal/Novelty Analysis Complete ===")
    return manifest


if __name__ == "__main__":
    run_ims_temporal_analysis()
