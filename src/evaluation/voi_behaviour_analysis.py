"""VoI Behaviour Analysis (Task 13).

Read-only diagnostic analysis of the already-integrated
CWRU -> CNN -> Novelty + Uncertainty + Relevance + Temporal + Communication
Cost -> canonical VoI Engine -> Decision pipeline.

This module computes NO new mathematics of its own: it calls the existing
src/novelty, src/uncertainty, src/relevance, src/temporal, src/communication
modules to obtain the five normalised VoI factors for every CWRU window, then
feeds them through the canonical src/voi/voi_engine.VoIEngine using its
DEFAULT weights and thresholds (never modified here), and reports on the
resulting behaviour: score distributions, decision rates, and which factors
drive (or fail to reach) the decision.

Communication Cost assumption: no FSO/channel-variability model exists yet
(planned future work), so every observation is scored under one nominal
"clear channel, single full raw window" scenario -- a fixed payload of
MAX_PAYLOAD_SIZE bytes (src/communication/config.py's own definition of a
single CWRU window) transmitted at the full REFERENCE_BANDWIDTH. This makes
Communication Cost effectively constant across observations by construction,
not by measurement -- see the analysis report for why that matters.

Temporal Importance assumption: computed on the normalised raw window signal
(matching the physical scale src/temporal/config.py's DEFAULT_TEMPORAL_CHANGE_SCALE
was calibrated against), sequenced per source recording (CWRU metadata's
`file_id` + `window_index`) so scores are never computed across a recording
boundary.
"""

import os
from typing import Tuple

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.cnn.model import extract_embeddings, predict_probabilities
from src.novelty.novelty import DistanceNoveltyDetector
from src.uncertainty.uncertainty import compute_predictive_entropy
from src.relevance.relevance import relevance_from_probabilities
from src.temporal.temporal import compute_temporal_importance
from src.communication.cost import compute_communication_cost
from src.communication.config import MAX_PAYLOAD_SIZE, REFERENCE_BANDWIDTH
from src.voi.voi_engine import VoIEngine
from src.voi.scoring import VoIWeights
from src.voi.decision_policy import PolicyThresholds

MODEL_PATH = "models/cwru_cnn_baseline.keras"
DATA_PATH = "data/processed/cwru/cwru_dataset_v1.npz"
METADATA_PATH = "data/processed/cwru/cwru_metadata.csv"
TABLE_DIR = "results/tables"
FIGURE_DIR = "results/figures"

# Ground-truth class ordering, matching src/cwru_pipeline/config.py and the
# corrected src/novelty, src/uncertainty, src/evaluation labels.
CLASS_NAMES = {0: "Normal", 1: "Inner Race Fault", 2: "Ball Fault", 3: "Outer Race Fault"}

# Nominal "clear channel, one full raw window" communication scenario used
# for every observation (see module docstring).
NOMINAL_PAYLOAD_BYTES = MAX_PAYLOAD_SIZE
NOMINAL_BANDWIDTH = REFERENCE_BANDWIDTH


def _compute_relevance_batch(probabilities: np.ndarray) -> np.ndarray:
    """Score task relevance per observation via the unmodified src/relevance API."""
    return np.array(
        [relevance_from_probabilities(row) for row in probabilities], dtype=np.float32
    )


def _compute_temporal_batch(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Score temporal importance per recording sequence, never across recording boundaries."""
    flat = X.reshape(X.shape[0], -1)
    scores = np.zeros(X.shape[0], dtype=np.float64)
    for _, group in meta.groupby("file_id"):
        idx = group.index.to_numpy()
        order = np.argsort(group["window_index"].to_numpy())
        ordered_idx = idx[order]
        scores[ordered_idx] = compute_temporal_importance(flat[ordered_idx])
    return scores.astype(np.float32)


def _compute_cost_batch(n: int) -> np.ndarray:
    """Constant nominal communication cost for n observations (see module docstring)."""
    cost = compute_communication_cost(
        payload_size=NOMINAL_PAYLOAD_BYTES,
        transmission_time=NOMINAL_PAYLOAD_BYTES / NOMINAL_BANDWIDTH,
        available_bandwidth=NOMINAL_BANDWIDTH,
    )
    return np.full(n, cost, dtype=np.float32)


def compute_all_factors(
    model: keras.Model,
    X: np.ndarray,
    meta: pd.DataFrame,
    novelty_detector: DistanceNoveltyDetector,
) -> pd.DataFrame:
    """Compute all five VoI factors for one dataset split.

    Args:
        model: Trained CNN model.
        X: Input window tensor, shape (N, 2048, 1).
        meta: Per-observation metadata rows for this split (same row order as X).
        novelty_detector: A DistanceNoveltyDetector already fitted on TRAIN embeddings only.

    Returns:
        DataFrame with one row per observation and columns for every VoI factor.
    """
    probabilities = predict_probabilities(model, X)
    embeddings = extract_embeddings(model, X)

    novelty = novelty_detector.score(embeddings)
    uncertainty = compute_predictive_entropy(probabilities)
    task_relevance = _compute_relevance_batch(probabilities)
    temporal_importance = _compute_temporal_batch(X, meta)
    resource_cost = _compute_cost_batch(len(X))

    return pd.DataFrame(
        {
            "observation_id": meta["observation_id"].to_numpy(),
            "class_id": meta["fault_label"].to_numpy(),
            "class_name": meta["fault_label"].map(CLASS_NAMES).to_numpy(),
            "novelty": novelty,
            "uncertainty": uncertainty,
            "task_relevance": task_relevance,
            "temporal_importance": temporal_importance,
            "resource_cost": resource_cost,
        }
    )


def run_voi_engine_batch(factors_df: pd.DataFrame, engine: VoIEngine) -> pd.DataFrame:
    """Run every observation's factors through the canonical VoI Engine (unmodified)."""
    result_df = engine.compute_batch(
        factors_df[
            ["novelty", "uncertainty", "task_relevance", "temporal_importance", "resource_cost"]
        ]
    )
    result_df["observation_id"] = factors_df["observation_id"].to_numpy()
    result_df["class_id"] = factors_df["class_id"].to_numpy()
    result_df["class_name"] = factors_df["class_name"].to_numpy()
    return result_df


def _split_stats_row(split_name: str, df: pd.DataFrame, cols) -> dict:
    row = {"group": split_name}
    for col in cols:
        row[f"{col}_mean"] = float(df[col].mean())
        row[f"{col}_median"] = float(df[col].median())
        row[f"{col}_std"] = float(df[col].std()) if len(df) > 1 else 0.0
        row[f"{col}_min"] = float(df[col].min())
        row[f"{col}_max"] = float(df[col].max())
    return row


def build_summary_table(all_results: dict) -> pd.DataFrame:
    """Per-split and per-test-class summary statistics for all factors and VoI score."""
    cols = [
        "novelty",
        "uncertainty",
        "task_relevance",
        "temporal_importance",
        "resource_cost",
        "raw_voi_score",
        "voi_score",
    ]
    rows = []
    for split_name in ("train", "val", "test"):
        rows.append(_split_stats_row(split_name, all_results[split_name], cols))

    test_df = all_results["test"]
    for class_id, class_name in CLASS_NAMES.items():
        class_df = test_df[test_df["class_id"] == class_id]
        if len(class_df) > 0:
            rows.append(
                _split_stats_row(f"test_class_{class_id}_{class_name.replace(' ', '_')}", class_df, cols)
            )

    return pd.DataFrame(rows)


def build_decision_distribution_table(all_results: dict) -> pd.DataFrame:
    """Decision-category counts and percentages per split and per test class."""
    rows = []
    for split_name in ("train", "val", "test"):
        df = all_results[split_name]
        counts = df["decision"].value_counts()
        total = len(df)
        row = {"group": split_name, "n": total}
        for action in ("DISCARD", "BUFFER", "SUMMARY", "TRANSMIT"):
            c = int(counts.get(action, 0))
            row[f"{action}_count"] = c
            row[f"{action}_pct"] = round(100.0 * c / total, 2) if total else 0.0
        rows.append(row)

    test_df = all_results["test"]
    for class_id, class_name in CLASS_NAMES.items():
        class_df = test_df[test_df["class_id"] == class_id]
        if len(class_df) == 0:
            continue
        counts = class_df["decision"].value_counts()
        total = len(class_df)
        row = {"group": f"test_class_{class_id}_{class_name.replace(' ', '_')}", "n": total}
        for action in ("DISCARD", "BUFFER", "SUMMARY", "TRANSMIT"):
            c = int(counts.get(action, 0))
            row[f"{action}_count"] = c
            row[f"{action}_pct"] = round(100.0 * c / total, 2) if total else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def build_dominance_table(all_results: dict, weights: VoIWeights) -> pd.DataFrame:
    """Mean weighted contribution of each VoI term, its share of total positive
    contribution, and its Pearson correlation with the final VoI score -- per split.
    """
    terms = {
        "novelty": weights.novelty,
        "uncertainty": weights.uncertainty,
        "task_relevance": weights.task_relevance,
        "temporal_importance": weights.temporal_importance,
    }
    rows = []
    for split_name in ("train", "val", "test"):
        df = all_results[split_name]
        contributions = {name: (w * df[name]) for name, w in terms.items()}
        contributions["resource_cost"] = -weights.resource_cost * df["resource_cost"]

        positive_total = sum(
            c.mean() for name, c in contributions.items() if name != "resource_cost"
        )
        row = {"group": split_name}
        for name, contribution in contributions.items():
            row[f"{name}_mean_contribution"] = float(contribution.mean())
            if name != "resource_cost" and positive_total > 0:
                row[f"{name}_share_of_positive_contribution_pct"] = round(
                    100.0 * contribution.mean() / positive_total, 2
                )
            row[f"{name}_corr_with_voi_score"] = float(
                df[name].corr(df["voi_score"]) if df[name].std() > 0 else 0.0
            )
        rows.append(row)

    return pd.DataFrame(rows)


def build_sensitivity_table(all_results: dict) -> pd.DataFrame:
    """Read-only exploration of how decision rates would change under alternative
    weight / threshold configurations. Does NOT modify src/voi/ defaults -- these
    are separate local VoIEngine instances used only for this diagnostic table.
    """
    test_factors = all_results["test"][
        ["novelty", "uncertainty", "task_relevance", "temporal_importance", "resource_cost"]
    ]

    scenarios = {
        "default (w=0.20 each, thresholds 0.25/0.50/0.70)": (VoIWeights(), PolicyThresholds()),
        "lower TRANSMIT threshold to 0.50": (
            VoIWeights(),
            PolicyThresholds(discard_max=0.25, buffer_max=0.35, summary_max=0.50),
        ),
        "lower TRANSMIT threshold to 0.60": (
            VoIWeights(),
            PolicyThresholds(discard_max=0.25, buffer_max=0.40, summary_max=0.60),
        ),
        "zero-out uncertainty weight, redistribute to novelty+relevance": (
            VoIWeights(novelty=0.30, uncertainty=0.0, task_relevance=0.30, temporal_importance=0.20, resource_cost=0.20),
            PolicyThresholds(),
        ),
        "zero-out communication cost weight": (
            VoIWeights(novelty=0.20, uncertainty=0.20, task_relevance=0.20, temporal_importance=0.20, resource_cost=0.0),
            PolicyThresholds(),
        ),
        "double task_relevance weight (renormalised)": (
            VoIWeights(novelty=0.15, uncertainty=0.15, task_relevance=0.40, temporal_importance=0.15, resource_cost=0.15),
            PolicyThresholds(),
        ),
    }

    rows = []
    for scenario_name, (weights, thresholds) in scenarios.items():
        engine = VoIEngine(weights=weights, thresholds=thresholds)
        result_df = engine.compute_batch(test_factors)
        counts = result_df["decision"].value_counts()
        total = len(result_df)
        row = {"scenario": scenario_name, "n": total, "mean_voi_score": float(result_df["voi_score"].mean())}
        for action in ("DISCARD", "BUFFER", "SUMMARY", "TRANSMIT"):
            c = int(counts.get(action, 0))
            row[f"{action}_pct"] = round(100.0 * c / total, 2) if total else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_relevance_distribution(all_results: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 25)
    colors = {"train": "navy", "val": "orange", "test": "green"}
    for split_name, color in colors.items():
        ax.hist(
            all_results[split_name]["task_relevance"],
            bins=bins,
            alpha=0.5,
            label=f"{split_name.capitalize()} Scores",
            color=color,
        )
    ax.set_xlabel("Task Relevance Score [0, 1]", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Task Relevance Score Distribution Across Dataset Splits", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved relevance distribution plot to: {save_path}")


def plot_voi_score_distribution(all_results: dict, thresholds: PolicyThresholds, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 25)
    colors = {"train": "navy", "val": "orange", "test": "green"}
    for split_name, color in colors.items():
        ax.hist(
            all_results[split_name]["voi_score"],
            bins=bins,
            alpha=0.5,
            label=f"{split_name.capitalize()} Scores",
            color=color,
        )
    for boundary, label in (
        (thresholds.discard_max, "DISCARD|BUFFER"),
        (thresholds.buffer_max, "BUFFER|SUMMARY"),
        (thresholds.summary_max, "SUMMARY|TRANSMIT"),
    ):
        ax.axvline(boundary, color="red", linestyle="--", alpha=0.7)
        ax.text(boundary, ax.get_ylim()[1] * 0.95, label, rotation=90, va="top", ha="right", fontsize=8, color="red")
    ax.set_xlabel("VoI Score [0, 1]", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("VoI Score Distribution Across Dataset Splits (with decision thresholds)", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved VoI score distribution plot to: {save_path}")


def plot_voi_score_by_class(test_df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    data_by_class, labels_list = [], []
    for class_id, name in CLASS_NAMES.items():
        mask = test_df["class_id"] == class_id
        if mask.sum() > 0:
            data_by_class.append(test_df.loc[mask, "voi_score"])
            labels_list.append(f"Class {class_id}: {name}")
    ax.boxplot(data_by_class, tick_labels=labels_list, patch_artist=True)
    ax.set_ylabel("VoI Score [0, 1]", fontsize=11)
    ax.set_title("Test Set VoI Scores by Bearing Health Condition", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved VoI score by class plot to: {save_path}")


def plot_decision_distribution(test_df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    actions = ["DISCARD", "BUFFER", "SUMMARY", "TRANSMIT"]
    class_ids = list(CLASS_NAMES.keys())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(class_ids))
    colors = {"DISCARD": "#888888", "BUFFER": "#5b9bd5", "SUMMARY": "#ffc000", "TRANSMIT": "#c00000"}
    x = np.arange(len(class_ids))
    for action in actions:
        counts = []
        for class_id in class_ids:
            class_df = test_df[test_df["class_id"] == class_id]
            counts.append((class_df["decision"] == action).sum())
        ax.bar(x, counts, bottom=bottom, label=action, color=colors[action])
        bottom += np.array(counts)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {i}:\n{CLASS_NAMES[i]}" for i in class_ids], fontsize=9)
    ax.set_ylabel("Number of Test Observations", fontsize=11)
    ax.set_title("Test Set VoI Decision Distribution by Bearing Health Condition", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved decision distribution plot to: {save_path}")


def plot_factor_contribution(dominance_df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    row = dominance_df[dominance_df["group"] == "test"].iloc[0]
    factors = ["novelty", "uncertainty", "task_relevance", "temporal_importance", "resource_cost"]
    contributions = [row[f"{f}_mean_contribution"] for f in factors]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4472c4", "#ed7d31", "#70ad47", "#ffc000", "#c00000"]
    ax.bar(factors, contributions, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Weighted Contribution to raw_voi_score", fontsize=11)
    ax.set_title("Mean Weighted Factor Contribution to VoI Score (Test Set)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    for i, v in enumerate(contributions):
        ax.text(i, v + (0.002 if v >= 0 else -0.008), f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved factor contribution plot to: {save_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_voi_behaviour_analysis(
    model_path: str = MODEL_PATH,
    data_path: str = DATA_PATH,
    metadata_path: str = METADATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full read-only VoI behaviour analysis and write all artifacts.

    Returns:
        Tuple of (summary_df, decision_df, dominance_df, sensitivity_df).
    """
    print("=== Executing VoI Behaviour Analysis (Task 13) ===")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained CNN model not found at '{model_path}'.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed CWRU dataset not found at '{data_path}'.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"CWRU metadata not found at '{metadata_path}'.")

    model = keras.models.load_model(model_path, compile=False)
    data = np.load(data_path)
    meta_all = pd.read_csv(metadata_path)

    X_splits = {"train": data["X_train"], "val": data["X_val"], "test": data["X_test"]}
    y_splits = {"train": data["y_train"], "val": data["y_val"], "test": data["y_test"]}
    meta_splits = {
        name: meta_all[meta_all["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")
    }

    for name in ("train", "val", "test"):
        assert (meta_splits[name]["fault_label"].to_numpy() == y_splits[name]).all(), (
            f"Metadata/label misalignment detected in split '{name}'"
        )

    # Novelty reference (class 0 = Normal) fitted STRICTLY on training embeddings.
    print("Fitting novelty reference on TRAINING embeddings only...")
    train_embeddings = extract_embeddings(model, X_splits["train"])
    novelty_detector = DistanceNoveltyDetector(reference_class=0)
    novelty_detector.fit(train_embeddings, y_splits["train"])

    print("Computing all five VoI factors per split...")
    factor_frames = {
        name: compute_all_factors(model, X_splits[name], meta_splits[name], novelty_detector)
        for name in ("train", "val", "test")
    }

    engine = VoIEngine()  # canonical default weights (0.20 each) and thresholds -- unmodified
    print("Running canonical VoI Engine (default weights/thresholds) over all splits...")
    all_results = {name: run_voi_engine_batch(factor_frames[name], engine) for name in ("train", "val", "test")}

    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # --- Relevance distribution summary (fills a gap: src/relevance has no
    # existing distribution report, unlike novelty/uncertainty). ---
    relevance_rows = []
    for name in ("train", "val", "test"):
        relevance_rows.append(
            {
                "split": name,
                "mean": float(factor_frames[name]["task_relevance"].mean()),
                "median": float(factor_frames[name]["task_relevance"].median()),
                "min": float(factor_frames[name]["task_relevance"].min()),
                "max": float(factor_frames[name]["task_relevance"].max()),
                "std": float(factor_frames[name]["task_relevance"].std()),
            }
        )
    test_factors = factor_frames["test"]
    for class_id, name in CLASS_NAMES.items():
        mask = test_factors["class_id"] == class_id
        if mask.sum() > 0:
            c = test_factors.loc[mask, "task_relevance"]
            relevance_rows.append(
                {
                    "split": f"test_class_{class_id}_{name.replace(' ', '_')}",
                    "mean": float(c.mean()),
                    "median": float(c.median()),
                    "min": float(c.min()),
                    "max": float(c.max()),
                    "std": float(c.std()) if len(c) > 1 else 0.0,
                }
            )
    relevance_summary_df = pd.DataFrame(relevance_rows)
    relevance_summary_path = os.path.join(TABLE_DIR, "relevance_scores_summary.csv")
    relevance_summary_df.to_csv(relevance_summary_path, index=False)
    print(f"Saved relevance summary table to: {relevance_summary_path}")
    plot_relevance_distribution(factor_frames, os.path.join(FIGURE_DIR, "relevance_score_distribution.png"))

    # --- Per-observation table (test split) ---
    per_obs_path = os.path.join(TABLE_DIR, "voi_integration_per_observation.csv")
    all_results["test"].to_csv(per_obs_path, index=False)
    print(f"Saved per-observation VoI table to: {per_obs_path}")

    # --- Summary / decision / dominance / sensitivity tables ---
    summary_df = build_summary_table(all_results)
    summary_path = os.path.join(TABLE_DIR, "voi_integration_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved VoI integration summary table to: {summary_path}")

    decision_df = build_decision_distribution_table(all_results)
    decision_path = os.path.join(TABLE_DIR, "voi_decision_distribution.csv")
    decision_df.to_csv(decision_path, index=False)
    print(f"Saved VoI decision distribution table to: {decision_path}")

    dominance_df = build_dominance_table(all_results, engine.weights)
    dominance_path = os.path.join(TABLE_DIR, "voi_factor_dominance.csv")
    dominance_df.to_csv(dominance_path, index=False)
    print(f"Saved VoI factor dominance table to: {dominance_path}")

    sensitivity_df = build_sensitivity_table(all_results)
    sensitivity_path = os.path.join(TABLE_DIR, "voi_sensitivity_analysis.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"Saved VoI sensitivity analysis table to: {sensitivity_path}")

    # --- Plots ---
    plot_voi_score_distribution(all_results, engine.thresholds, os.path.join(FIGURE_DIR, "voi_score_distribution.png"))
    plot_voi_score_by_class(all_results["test"], os.path.join(FIGURE_DIR, "voi_score_by_class.png"))
    plot_decision_distribution(all_results["test"], os.path.join(FIGURE_DIR, "voi_decision_distribution.png"))
    plot_factor_contribution(dominance_df, os.path.join(FIGURE_DIR, "voi_factor_contribution.png"))

    print("=== VoI Behaviour Analysis Complete ===")
    return summary_df, decision_df, dominance_df, sensitivity_df


if __name__ == "__main__":
    run_voi_behaviour_analysis()
