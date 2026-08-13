"""Synthetic dataset generator for the VoI Engine research prototype.

Generates synthetic time-series observations across six distinct behavioral scenarios
to test and validate the mathematical scoring and decision policy behavior of the VoI Engine.
Uses a deterministic random seed (42) for reproducible experimental evaluation.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd


def generate_synthetic_dataset(
    output_path: Optional[str] = "data/synthetic/synthetic_voi_dataset.csv",
    num_observations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset of VoI input observations.

    Generates six sequential behavioral scenarios:
    - Scenario A: Normal / Low-information period
    - Scenario B: Sudden important event
    - Scenario C: Gradual degradation / trend
    - Scenario D: High uncertainty event
    - Scenario E: High communication-cost event
    - Scenario F: High novelty but low task relevance

    Args:
        output_path: Target CSV file path to save generated dataset.
        num_observations: Total number of timesteps/observations to generate (default 1000).
        seed: Random seed for reproducibility (default 42).

    Returns:
        pd.DataFrame: Generated synthetic dataset DataFrame.
    """
    np.random.seed(seed)

    timestamps = [f"2026-08-14T{i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}Z" for i in range(num_observations)]

    novelty = np.zeros(num_observations)
    uncertainty = np.zeros(num_observations)
    task_relevance = np.zeros(num_observations)
    temporal_importance = np.zeros(num_observations)
    resource_cost = np.zeros(num_observations)
    scenario_labels = [""] * num_observations

    # Compute dynamic lengths for each scenario
    n_a = int(num_observations * 0.20)
    n_b = int(num_observations * 0.15)
    n_c = int(num_observations * 0.20)
    n_d = int(num_observations * 0.15)
    n_e = int(num_observations * 0.15)
    n_f = num_observations - (n_a + n_b + n_c + n_d + n_e)

    # Scenario A: Normal / Low Information
    # Low N, U, R, T, low/moderate C
    idx_a = slice(0, n_a)
    novelty[idx_a] = np.random.uniform(0.05, 0.20, n_a)
    uncertainty[idx_a] = np.random.uniform(0.05, 0.20, n_a)
    task_relevance[idx_a] = np.random.uniform(0.05, 0.20, n_a)
    temporal_importance[idx_a] = np.random.uniform(0.05, 0.20, n_a)
    resource_cost[idx_a] = np.random.uniform(0.10, 0.30, n_a)
    for i in range(0, n_a):
        scenario_labels[i] = "Scenario A: Normal / Low Info"

    # Scenario B: Sudden Important Event
    # High N, R, T, elevated U, moderate C
    s_b = n_a
    e_b = s_b + n_b
    idx_b = slice(s_b, e_b)
    novelty[idx_b] = np.random.uniform(0.75, 0.95, n_b)
    uncertainty[idx_b] = np.random.uniform(0.60, 0.85, n_b)
    task_relevance[idx_b] = np.random.uniform(0.75, 0.95, n_b)
    temporal_importance[idx_b] = np.random.uniform(0.75, 0.95, n_b)
    resource_cost[idx_b] = np.random.uniform(0.20, 0.40, n_b)
    for i in range(s_b, e_b):
        scenario_labels[i] = "Scenario B: Sudden Important Event"

    # Scenario C: Gradual Degradation / Trend
    # Novelty, Relevance, Temporal Importance ramp up from 0.15 to 0.85
    s_c = e_b
    e_c = s_c + n_c
    idx_c = slice(s_c, e_c)
    ramp = np.linspace(0.15, 0.85, n_c)
    noise_c = np.random.normal(0, 0.03, n_c)
    novelty[idx_c] = ramp + noise_c
    task_relevance[idx_c] = ramp + noise_c
    temporal_importance[idx_c] = ramp + noise_c
    uncertainty[idx_c] = np.random.uniform(0.20, 0.40, n_c)
    resource_cost[idx_c] = np.random.uniform(0.20, 0.40, n_c)
    for i in range(s_c, e_c):
        scenario_labels[i] = "Scenario C: Gradual Trend"

    # Scenario D: High Uncertainty Event
    # Moderate N, High U, High R, Low/mod T, C
    s_d = e_c
    e_d = s_d + n_d
    idx_d = slice(s_d, e_d)
    novelty[idx_d] = np.random.uniform(0.50, 0.70, n_d)
    uncertainty[idx_d] = np.random.uniform(0.85, 0.98, n_d)
    task_relevance[idx_d] = np.random.uniform(0.70, 0.90, n_d)
    temporal_importance[idx_d] = np.random.uniform(0.20, 0.40, n_d)
    resource_cost[idx_d] = np.random.uniform(0.15, 0.35, n_d)
    for i in range(s_d, e_d):
        scenario_labels[i] = "Scenario D: High Uncertainty"

    # Scenario E: High Communication-Cost Event
    # High N, R, T, but High Cost C
    s_e = e_d
    e_e = s_e + n_e
    idx_e = slice(s_e, e_e)
    novelty[idx_e] = np.random.uniform(0.80, 0.95, n_e)
    uncertainty[idx_e] = np.random.uniform(0.30, 0.50, n_e)
    task_relevance[idx_e] = np.random.uniform(0.80, 0.95, n_e)
    temporal_importance[idx_e] = np.random.uniform(0.80, 0.95, n_e)
    resource_cost[idx_e] = np.random.uniform(0.85, 0.98, n_e)
    for i in range(s_e, e_e):
        scenario_labels[i] = "Scenario E: High Cost Event"

    # Scenario F: High Novelty but Low Task Relevance
    # Very High N, Very Low R
    s_f = e_e
    e_f = num_observations
    idx_f = slice(s_f, e_f)
    novelty[idx_f] = np.random.uniform(0.85, 0.98, n_f)
    uncertainty[idx_f] = np.random.uniform(0.20, 0.40, n_f)
    task_relevance[idx_f] = np.random.uniform(0.02, 0.15, n_f)
    temporal_importance[idx_f] = np.random.uniform(0.20, 0.40, n_f)
    resource_cost[idx_f] = np.random.uniform(0.20, 0.40, n_f)
    for i in range(s_f, e_f):
        scenario_labels[i] = "Scenario F: High Novelty / Low Relevance"

    # Ensure all values are strictly clipped to [0, 1]
    novelty = np.clip(novelty, 0.0, 1.0)
    uncertainty = np.clip(uncertainty, 0.0, 1.0)
    task_relevance = np.clip(task_relevance, 0.0, 1.0)
    temporal_importance = np.clip(temporal_importance, 0.0, 1.0)
    resource_cost = np.clip(resource_cost, 0.0, 1.0)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "novelty": np.round(novelty, 4),
            "uncertainty": np.round(uncertainty, 4),
            "task_relevance": np.round(task_relevance, 4),
            "temporal_importance": np.round(temporal_importance, 4),
            "resource_cost": np.round(resource_cost, 4),
            "scenario": scenario_labels,
        }
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Synthetic dataset with {len(df)} observations saved to: {output_path}")

    return df


if __name__ == "__main__":
    generate_synthetic_dataset()
