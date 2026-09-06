"""Paderborn VoI Behaviour Analysis.

Read-only diagnostic analysis of the canonical CNN -> Novelty + Uncertainty +
Relevance + Temporal + Communication Cost -> canonical VoI Engine -> Decision
pipeline applied to Paderborn, mirroring src/evaluation/voi_behaviour_analysis.py
(the CWRU analysis) structurally. It calls the existing src/novelty, src/uncertainty,
src/temporal, src/communication, src/voi modules unmodified, with the canonical
VoIEngine's default (unmodified) weights and thresholds.

Task Relevance deviation: src/relevance/relevance.py's public functions validate
their relevance_map against the module-level NUM_CLASSES=4 default regardless of
the map actually passed in (tests/test_relevance.py::test_16 asserts this 4-class-only
validation is the intended, tested contract), so they cannot accept a 3-class
Paderborn relevance map. This module instead applies the identical documented
formula (R = sum(P(class_i) * relevance(class_i))) inline against a Paderborn-specific
relevance map inherited unchanged, per shared fault-type name, from CWRU's
CLASS_RELEVANCE_MAP (Healthy<-Normal=0.10, Inner Race=1.00, Outer Race=0.90;
Ball Fault's value is inapplicable since Paderborn has no ball-fault class), without
touching src/relevance/ itself.

Temporal Importance deviation: DEFAULT_TEMPORAL_CHANGE_SCALE (1.8) was calibrated
against CWRU's own training-split signal-change distribution. It is used here
UNCHANGED (not re-derived for Paderborn) to test transfer of the existing
implementation as-is; if it saturates or collapses on Paderborn data, that is
reported as a legitimate finding, not corrected.
"""

import os
from typing import Dict

import keras
import numpy as np
import pandas as pd

from src.cnn.model import extract_embeddings, predict_probabilities
from src.novelty.novelty import DistanceNoveltyDetector
from src.uncertainty.uncertainty import compute_predictive_entropy
from src.temporal.temporal import compute_temporal_importance
from src.communication.cost import compute_communication_cost
from src.communication.config import MAX_PAYLOAD_SIZE, REFERENCE_BANDWIDTH
from src.voi.voi_engine import VoIEngine
from src.paderborn_pipeline.classification_task import CLASS_NAMES, DATASET_V1_PATH, METADATA_PATH

MODEL_PATH = "models/paderborn_cnn_baseline.keras"
TABLE_DIR = "results/tables"
FIGURE_DIR = "results/figures"

NUM_CLASSES = len(CLASS_NAMES)

PADERBORN_RELEVANCE_MAP: Dict[int, float] = {0: 0.10, 1: 1.00, 2: 0.90}

NOMINAL_PAYLOAD_BYTES = MAX_PAYLOAD_SIZE
NOMINAL_BANDWIDTH = REFERENCE_BANDWIDTH


def _relevance_from_probabilities(probabilities: np.ndarray, relevance_map: Dict[int, float]) -> float:
    weights = np.array([relevance_map[i] for i in range(len(relevance_map))], dtype=np.float32)
    score = float(np.dot(probabilities, weights))
    return float(np.clip(score, 0.0, 1.0))


def _compute_relevance_batch(probabilities: np.ndarray) -> np.ndarray:
    return np.array(
        [_relevance_from_probabilities(row, PADERBORN_RELEVANCE_MAP) for row in probabilities],
        dtype=np.float32,
    )


def _compute_temporal_batch(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    flat = X.reshape(X.shape[0], -1)
    scores = np.zeros(X.shape[0], dtype=np.float64)
    for _, group in meta.groupby("recording_id"):
        idx = group.index.to_numpy()
        order = np.argsort(group["window_index"].to_numpy())
        ordered_idx = idx[order]
        scores[ordered_idx] = compute_temporal_importance(flat[ordered_idx])
    return scores.astype(np.float32)


def _compute_cost_batch(n: int) -> np.ndarray:
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
    probabilities = predict_probabilities(model, X)
    embeddings = extract_embeddings(model, X)

    novelty = novelty_detector.score(embeddings)
    uncertainty = compute_predictive_entropy(probabilities, num_classes=NUM_CLASSES)
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
    result_df = engine.compute_batch(
        factors_df[["novelty", "uncertainty", "task_relevance", "temporal_importance", "resource_cost"]]
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
    cols = ["novelty", "uncertainty", "task_relevance", "temporal_importance", "resource_cost", "raw_voi_score", "voi_score"]
    rows = []
    for split_name in ("train", "val", "test"):
        rows.append(_split_stats_row(split_name, all_results[split_name], cols))

    test_df = all_results["test"]
    for class_id, class_name in CLASS_NAMES.items():
        class_df = test_df[test_df["class_id"] == class_id]
        if len(class_df) > 0:
            rows.append(_split_stats_row(f"test_class_{class_id}_{class_name.replace(' ', '_')}", class_df, cols))

    return pd.DataFrame(rows)


def build_decision_distribution_table(all_results: dict) -> pd.DataFrame:
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


def build_dominance_table(all_results: dict, weights) -> pd.DataFrame:
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

        positive_total = sum(c.mean() for name, c in contributions.items() if name != "resource_cost")
        row = {"group": split_name}
        for name, contribution in contributions.items():
            row[f"{name}_mean_contribution"] = float(contribution.mean())
            if name != "resource_cost" and positive_total > 0:
                row[f"{name}_share_of_positive_contribution_pct"] = round(100.0 * contribution.mean() / positive_total, 2)
            row[f"{name}_corr_with_voi_score"] = float(df[name].corr(df["voi_score"]) if df[name].std() > 0 else 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def run_paderborn_voi_behaviour_analysis(
    model_path: str = MODEL_PATH,
    data_path: str = DATASET_V1_PATH,
    metadata_path: str = METADATA_PATH,
):
    print("=== Executing Paderborn VoI Behaviour Analysis ===")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained Paderborn CNN model not found at '{model_path}'.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed Paderborn dataset not found at '{data_path}'.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Paderborn metadata not found at '{metadata_path}'.")

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

    print("Fitting novelty reference on TRAINING embeddings only (reference_class=0=Healthy)...")
    train_embeddings = extract_embeddings(model, X_splits["train"])
    novelty_detector = DistanceNoveltyDetector(reference_class=0)
    novelty_detector.fit(train_embeddings, y_splits["train"])

    print("Computing all five VoI factors per split...")
    factor_frames = {
        name: compute_all_factors(model, X_splits[name], meta_splits[name], novelty_detector)
        for name in ("train", "val", "test")
    }

    engine = VoIEngine()
    print("Running canonical VoI Engine (default weights/thresholds, unmodified) over all splits...")
    all_results = {name: run_voi_engine_batch(factor_frames[name], engine) for name in ("train", "val", "test")}

    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

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
    relevance_summary_path = os.path.join(TABLE_DIR, "paderborn_relevance_scores_summary.csv")
    relevance_summary_df.to_csv(relevance_summary_path, index=False)
    print(f"Saved relevance summary table to: {relevance_summary_path}")

    per_obs_path = os.path.join(TABLE_DIR, "paderborn_voi_integration_per_observation.csv")
    all_results["test"].to_csv(per_obs_path, index=False)
    print(f"Saved per-observation VoI table to: {per_obs_path}")

    summary_df = build_summary_table(all_results)
    summary_path = os.path.join(TABLE_DIR, "paderborn_voi_integration_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved VoI integration summary table to: {summary_path}")

    decision_df = build_decision_distribution_table(all_results)
    decision_path = os.path.join(TABLE_DIR, "paderborn_voi_decision_distribution.csv")
    decision_df.to_csv(decision_path, index=False)
    print(f"Saved VoI decision distribution table to: {decision_path}")

    dominance_df = build_dominance_table(all_results, engine.weights)
    dominance_path = os.path.join(TABLE_DIR, "paderborn_voi_factor_dominance.csv")
    dominance_df.to_csv(dominance_path, index=False)
    print(f"Saved VoI factor dominance table to: {dominance_path}")

    print("=== Paderborn VoI Behaviour Analysis Complete ===")
    return summary_df, decision_df, dominance_df


if __name__ == "__main__":
    run_paderborn_voi_behaviour_analysis()
