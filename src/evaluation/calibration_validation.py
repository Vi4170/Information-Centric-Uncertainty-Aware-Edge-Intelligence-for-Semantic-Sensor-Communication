"""Calibrated VoI Validation (Task 15).

Read-only comparison of the pre-calibration (Task 13) vs calibrated
(Task 14) VoI system on the held-out CWRU test split.

This module implements NO new mathematics and modifies NO defaults. It
constructs two EXPLICIT, local configurations -- the historical Task 13
parameters and the current (Task 14) canonical defaults -- and evaluates
both, using the unmodified canonical src/voi/voi_engine.VoIEngine and the
unmodified src/temporal/temporal.compute_temporal_importance, on the exact
same test-split observations already produced by
src/evaluation/voi_behaviour_analysis.py (Task 13/14).

"Before" reconstruction: Novelty, Uncertainty, Task Relevance, and
Communication Cost are identical in both configurations (their own modules
were never changed between Task 13 and Task 14) and are loaded directly
from the already-generated, already-committed
results/tables/voi_integration_per_observation.csv (the current, Task 14,
test-split factor table). Only Temporal Importance differs (it depends on
DEFAULT_TEMPORAL_CHANGE_SCALE, which Task 14 changed), so it alone is
recomputed here with the historical scale (0.5) via the unmodified
compute_temporal_importance() function, using the same per-recording
windowing already used in Task 13/14. Both factor sets are then scored by
two explicit VoIEngine instances (historical Task 13 weights, and current
canonical defaults) -- never by mutating any module's defaults.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.temporal.temporal import compute_temporal_importance
from src.voi.voi_engine import VoIEngine
from src.voi.scoring import VoIWeights
from src.voi.decision_policy import PolicyThresholds

METADATA_PATH = "data/processed/cwru/cwru_metadata.csv"
DATA_PATH = "data/processed/cwru/cwru_dataset_v1.npz"
PER_OBS_PATH = "results/tables/voi_integration_per_observation.csv"
TABLE_DIR = "results/tables"
FIGURE_DIR = "results/figures"

CLASS_NAMES = {0: "Normal", 1: "Inner Race Fault", 2: "Ball Fault", 3: "Outer Race Fault"}

# Historical Task 13 configuration -- an EXPLICIT local instance for
# comparison only. Does not touch src/voi/scoring.py's current defaults.
PRE_CALIBRATION_WEIGHTS = VoIWeights(
    novelty=0.20, uncertainty=0.20, task_relevance=0.20, temporal_importance=0.20, resource_cost=0.20
)
PRE_CALIBRATION_TEMPORAL_SCALE = 0.5


def _temporal_batch(X: np.ndarray, meta: pd.DataFrame, scale: float) -> np.ndarray:
    """Recompute Temporal Importance with an explicit scale, per recording sequence."""
    flat = X.reshape(X.shape[0], -1)
    scores = np.zeros(X.shape[0], dtype=np.float64)
    for _, group in meta.groupby("file_id"):
        idx = group.index.to_numpy()
        order = np.argsort(group["window_index"].to_numpy())
        ordered_idx = idx[order]
        scores[ordered_idx] = compute_temporal_importance(flat[ordered_idx], temporal_change_scale=scale)
    return scores.astype(np.float32)


def build_before_after_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (before_df, after_df): paired per-observation VoI results for the test split."""
    after_df = pd.read_csv(PER_OBS_PATH)

    data = np.load(DATA_PATH)
    meta_all = pd.read_csv(METADATA_PATH)
    meta_test = meta_all[meta_all["split"] == "test"].reset_index(drop=True)
    X_test = data["X_test"]

    assert (meta_test["observation_id"].to_numpy() == after_df["observation_id"].to_numpy()).all(), (
        "Row-order mismatch between metadata and the existing per-observation table"
    )

    temporal_before = _temporal_batch(X_test, meta_test, PRE_CALIBRATION_TEMPORAL_SCALE)

    before_factors = after_df[
        ["observation_id", "class_id", "class_name", "novelty", "uncertainty", "task_relevance", "resource_cost"]
    ].copy()
    before_factors["temporal_importance"] = temporal_before

    pre_engine = VoIEngine(weights=PRE_CALIBRATION_WEIGHTS, thresholds=PolicyThresholds())
    before_result = pre_engine.compute_batch(
        before_factors[["novelty", "uncertainty", "task_relevance", "temporal_importance", "resource_cost"]]
    )
    before_df = before_factors.copy()
    before_df["raw_voi_score"] = before_result["raw_voi_score"]
    before_df["voi_score"] = before_result["voi_score"]
    before_df["decision"] = before_result["decision"]

    return before_df, after_df


def decision_distribution_table(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("test (overall)", before_df, after_df)]
    for class_id, name in CLASS_NAMES.items():
        groups.append(
            (
                f"test_class_{class_id}_{name.replace(' ', '_')}",
                before_df[before_df["class_id"] == class_id],
                after_df[after_df["class_id"] == class_id],
            )
        )
    for group_name, b_df, a_df in groups:
        row = {"group": group_name, "n": len(a_df)}
        for action in ("DISCARD", "BUFFER", "SUMMARY", "TRANSMIT"):
            b_pct = round(100.0 * (b_df["decision"] == action).sum() / len(b_df), 2) if len(b_df) else 0.0
            a_pct = round(100.0 * (a_df["decision"] == action).sum() / len(a_df), 2) if len(a_df) else 0.0
            row[f"before_{action}_pct"] = b_pct
            row[f"after_{action}_pct"] = a_pct
        rows.append(row)
    return pd.DataFrame(rows)


def voi_score_stats_table(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("test (overall)", before_df, after_df)]
    for class_id, name in CLASS_NAMES.items():
        groups.append(
            (
                f"test_class_{class_id}_{name.replace(' ', '_')}",
                before_df[before_df["class_id"] == class_id],
                after_df[after_df["class_id"] == class_id],
            )
        )
    for group_name, b_df, a_df in groups:
        rows.append(
            {
                "group": group_name,
                "before_mean": float(b_df["voi_score"].mean()),
                "before_min": float(b_df["voi_score"].min()),
                "before_max": float(b_df["voi_score"].max()),
                "after_mean": float(a_df["voi_score"].mean()),
                "after_min": float(a_df["voi_score"].min()),
                "after_max": float(a_df["voi_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def factor_dominance_table(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df, weights in (
        ("before", before_df, PRE_CALIBRATION_WEIGHTS),
        ("after", after_df, VoIWeights()),
    ):
        terms = {
            "novelty": weights.novelty,
            "uncertainty": weights.uncertainty,
            "task_relevance": weights.task_relevance,
            "temporal_importance": weights.temporal_importance,
        }
        contributions = {name: (w * df[name]) for name, w in terms.items()}
        contributions["resource_cost"] = -weights.resource_cost * df["resource_cost"]
        positive_total = sum(c.mean() for name, c in contributions.items() if name != "resource_cost")
        row = {"config": label}
        for name, contribution in contributions.items():
            row[f"{name}_mean_contribution"] = float(contribution.mean())
            if name != "resource_cost" and positive_total > 0:
                row[f"{name}_share_pct"] = round(100.0 * contribution.mean() / positive_total, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def transmission_reduction_table(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
    """Compare against the naive "transmit everything" baseline (100% TRANSMIT)."""
    rows = []
    for label, df in (("before", before_df), ("after", after_df)):
        n = len(df)
        full_transmit_equivalent = (df["decision"] == "TRANSMIT").sum()
        not_fully_transmitted = n - full_transmit_equivalent
        rows.append(
            {
                "config": label,
                "n": n,
                "naive_baseline_pct_transmitted": 100.0,
                "actual_pct_full_transmit": round(100.0 * full_transmit_equivalent / n, 2),
                "pct_not_fully_transmitted (DISCARD+BUFFER+SUMMARY)": round(100.0 * not_fully_transmitted / n, 2),
            }
        )
    return pd.DataFrame(rows)


def plot_before_after_decision(before_df: pd.DataFrame, after_df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    actions = ["DISCARD", "BUFFER", "SUMMARY", "TRANSMIT"]
    colors = {"DISCARD": "#888888", "BUFFER": "#5b9bd5", "SUMMARY": "#ffc000", "TRANSMIT": "#c00000"}
    class_ids = list(CLASS_NAMES.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, df, title in ((axes[0], before_df, "Before Calibration (Task 13)"), (axes[1], after_df, "After Calibration (Task 14)")):
        bottom = np.zeros(len(class_ids))
        x = np.arange(len(class_ids))
        for action in actions:
            counts = [(df[df["class_id"] == c]["decision"] == action).sum() for c in class_ids]
            ax.bar(x, counts, bottom=bottom, label=action, color=colors[action])
            bottom += np.array(counts)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Class {i}:\n{CLASS_NAMES[i]}" for i in class_ids], fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    axes[0].set_ylabel("Number of Test Observations", fontsize=11)
    axes[1].legend(fontsize=9, loc="upper right")
    fig.suptitle("Test Set VoI Decision Distribution: Before vs After Calibration", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved before/after decision comparison plot to: {save_path}")


def plot_before_after_voi_distribution(before_df: pd.DataFrame, after_df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    thresholds = PolicyThresholds()
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 25)
    ax.hist(before_df["voi_score"], bins=bins, alpha=0.5, label="Before (Task 13)", color="#7f7f7f")
    ax.hist(after_df["voi_score"], bins=bins, alpha=0.5, label="After (Task 14)", color="#2ca02c")
    for boundary in (thresholds.discard_max, thresholds.buffer_max, thresholds.summary_max):
        ax.axvline(boundary, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("VoI Score [0, 1]", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Test Set VoI Score Distribution: Before vs After Calibration", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved before/after VoI distribution plot to: {save_path}")


def check_reproducibility() -> bool:
    """Re-run the calibrated (after) pipeline end-to-end and confirm identical output
    to the already-committed results/tables/voi_integration_per_observation.csv.
    """
    from src.evaluation.voi_behaviour_analysis import run_voi_behaviour_analysis

    existing_df = pd.read_csv(PER_OBS_PATH).sort_values("observation_id").reset_index(drop=True)
    run_voi_behaviour_analysis()
    rerun_df = pd.read_csv(PER_OBS_PATH).sort_values("observation_id").reset_index(drop=True)

    numeric_cols = ["novelty", "uncertainty", "task_relevance", "temporal_importance", "resource_cost", "raw_voi_score", "voi_score"]
    max_abs_diff = float((existing_df[numeric_cols] - rerun_df[numeric_cols]).abs().max().max())
    decisions_match = (existing_df["decision"] == rerun_df["decision"]).all()

    print(f"Reproducibility check: max abs diff across all numeric factor/score columns = {max_abs_diff:.2e}")
    print(f"Reproducibility check: all decisions identical across reruns = {decisions_match}")
    return max_abs_diff < 1e-9 and bool(decisions_match)


def run_calibration_validation() -> None:
    print("=== Executing Calibrated VoI Validation (Task 15) ===")
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    before_df, after_df = build_before_after_tables()

    decision_df = decision_distribution_table(before_df, after_df)
    decision_path = os.path.join(TABLE_DIR, "calibration_validation_decision_comparison.csv")
    decision_df.to_csv(decision_path, index=False)
    print(f"Saved decision comparison table to: {decision_path}")

    stats_df = voi_score_stats_table(before_df, after_df)
    stats_path = os.path.join(TABLE_DIR, "calibration_validation_voi_score_comparison.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved VoI score comparison table to: {stats_path}")

    dominance_df = factor_dominance_table(before_df, after_df)
    dominance_path = os.path.join(TABLE_DIR, "calibration_validation_factor_dominance_comparison.csv")
    dominance_df.to_csv(dominance_path, index=False)
    print(f"Saved factor dominance comparison table to: {dominance_path}")

    transmission_df = transmission_reduction_table(before_df, after_df)
    transmission_path = os.path.join(TABLE_DIR, "calibration_validation_transmission_reduction.csv")
    transmission_df.to_csv(transmission_path, index=False)
    print(f"Saved transmission reduction table to: {transmission_path}")

    plot_before_after_decision(before_df, after_df, os.path.join(FIGURE_DIR, "voi_calibration_before_after_decision.png"))
    plot_before_after_voi_distribution(before_df, after_df, os.path.join(FIGURE_DIR, "voi_calibration_before_after_distribution.png"))

    is_reproducible = check_reproducibility()
    repro_df = pd.DataFrame(
        [{"check": "calibrated pipeline rerun matches committed results", "reproducible": is_reproducible}]
    )
    repro_path = os.path.join(TABLE_DIR, "calibration_validation_reproducibility.csv")
    repro_df.to_csv(repro_path, index=False)
    print(f"Saved reproducibility check to: {repro_path}")

    print("=== Calibrated VoI Validation Complete ===")


if __name__ == "__main__":
    run_calibration_validation()
