"""Evaluation and summary metrics module for VoI Engine results."""

from typing import Dict, Any
import pandas as pd
from src.voi.decision_policy import DecisionAction


def compute_summary_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary evaluation metrics from a VoI results DataFrame.

    Args:
        results_df: DataFrame output from VoIEngine.compute_batch().

    Returns:
        Dict[str, Any]: Metrics dictionary containing total count, action counts,
                        action percentages, mean/min/max VoI scores.
    """
    total_obs = len(results_df)
    decision_counts = results_df["decision"].value_counts().to_dict()

    n_discard = decision_counts.get(DecisionAction.DISCARD.value, 0)
    n_buffer = decision_counts.get(DecisionAction.BUFFER.value, 0)
    n_summary = decision_counts.get(DecisionAction.SUMMARY.value, 0)
    n_transmit = decision_counts.get(DecisionAction.TRANSMIT.value, 0)

    pct_transmitted = (n_transmit / total_obs) * 100.0 if total_obs > 0 else 0.0
    pct_suppressed = ((n_discard + n_buffer + n_summary) / total_obs) * 100.0 if total_obs > 0 else 0.0

    stats = {
        "total_observations": total_obs,
        "n_discard": n_discard,
        "n_buffer": n_buffer,
        "n_summary": n_summary,
        "n_transmit": n_transmit,
        "pct_transmitted": round(pct_transmitted, 2),
        "pct_suppressed": round(pct_suppressed, 2),
        "mean_voi_score": round(float(results_df["voi_score"].mean()), 4),
        "mean_raw_voi_score": round(float(results_df["raw_voi_score"].mean()), 4),
        "min_voi_score": round(float(results_df["voi_score"].min()), 4),
        "max_voi_score": round(float(results_df["voi_score"].max()), 4),
        "min_raw_voi_score": round(float(results_df["raw_voi_score"].min()), 4),
        "max_raw_voi_score": round(float(results_df["raw_voi_score"].max()), 4),
    }
    return stats
