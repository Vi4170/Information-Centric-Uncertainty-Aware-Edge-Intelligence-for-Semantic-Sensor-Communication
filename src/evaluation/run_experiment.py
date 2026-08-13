"""Experiment runner for VoI Engine Version 0.1 prototype.

Processes synthetic dataset, runs full diagnostics pass, saves summary tables and figures under results/.
"""

import os
import pandas as pd

from src.evaluation import compute_summary_statistics
from src.evaluation.diagnostics import (
    generate_v01_validation_summary,
    plot_scenario_decisions,
    run_clipping_analysis,
    run_correlation_analysis,
    run_cost_sensitivity_experiment,
    run_decision_reachability_analysis,
    run_high_novelty_low_relevance_test,
    run_monotonicity_experiments,
    run_scenario_analysis,
    run_threshold_sensitivity_experiment,
    run_weight_sensitivity_experiment,
)
from src.voi.voi_engine import VoIEngine


def run_synthetic_experiment(
    data_path: str = "data/synthetic/synthetic_voi_dataset.csv",
    output_fig_dir: str = "results/figures",
    output_table_dir: str = "results/tables",
) -> pd.DataFrame:
    """Run full experimental pipeline and diagnostics suite over synthetic dataset.

    Args:
        data_path: Path to synthetic dataset CSV.
        output_fig_dir: Directory to save generated plot figures.
        output_table_dir: Directory to save summary tables.

    Returns:
        pd.DataFrame: Enriched results DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic dataset not found at {data_path}")

    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_table_dir, exist_ok=True)

    df_raw = pd.read_csv(data_path)
    engine = VoIEngine()

    results_df = engine.compute_batch(df_raw)

    print("=== Executing VoI Engine V0.1 Diagnostic Pass ===")

    # 1. Reachability Analysis
    run_decision_reachability_analysis(results_df, fig_dir=output_fig_dir, table_dir=output_table_dir)

    # 2. Scenario Analysis Table
    run_scenario_analysis(results_df, table_dir=output_table_dir)

    # 3. Scenario Decision Plot
    plot_scenario_decisions(results_df, fig_dir=output_fig_dir)

    # 4. Monotonicity Experiments
    run_monotonicity_experiments(engine, fig_dir=output_fig_dir)

    # 5. High Novelty vs Low Relevance Test
    run_high_novelty_low_relevance_test(engine, table_dir=output_table_dir)

    # 6. Resource Cost Experiment
    run_cost_sensitivity_experiment(engine, table_dir=output_table_dir)

    # 7. Weight Sensitivity Experiment
    run_weight_sensitivity_experiment(df_raw, fig_dir=output_fig_dir, table_dir=output_table_dir)

    # 8. Clipping Analysis
    run_clipping_analysis(results_df, table_dir=output_table_dir)

    # 9. Threshold Sensitivity Experiment
    run_threshold_sensitivity_experiment(results_df, table_dir=output_table_dir)

    # 10. Correlations
    corrs = run_correlation_analysis(results_df)
    print("Variable Correlations with VoI Score:")
    for k, v in corrs.items():
        print(f"  {k}: {v}")

    # 11. Summary Statistics Table
    stats = compute_summary_statistics(results_df)
    stats_df = pd.DataFrame([stats])
    stats_csv_path = os.path.join(output_table_dir, "summary_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved summary statistics to: {stats_csv_path}")

    # 12. Sample Results Table
    sample_df = results_df[
        [
            "timestamp",
            "novelty",
            "uncertainty",
            "task_relevance",
            "temporal_importance",
            "resource_cost",
            "raw_voi_score",
            "voi_score",
            "decision",
        ]
    ].head(20)
    sample_csv_path = os.path.join(output_table_dir, "sample_results.csv")
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"Saved sample results table to: {sample_csv_path}")

    # 13. V0.1 Validation Summary Table
    generate_v01_validation_summary(table_dir=output_table_dir)

    print("=== Diagnostics Suite Execution Complete ===")

    return results_df


if __name__ == "__main__":
    run_synthetic_experiment()
