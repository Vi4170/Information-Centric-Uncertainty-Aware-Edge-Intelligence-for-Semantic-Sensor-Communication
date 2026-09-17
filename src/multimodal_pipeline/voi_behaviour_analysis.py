"""Task 41 -- multimodal VoI behaviour analysis.

Read-only diagnostic analysis of Task 39's already-computed VoI results
(``multimodal_voi_integration_results.csv``), structurally mirroring
``src/evaluation/voi_behaviour_analysis.py`` (CWRU) and
``src/evaluation/paderborn_voi_behaviour_analysis.py`` (Paderborn): factor
summary statistics, correlations with the VoI score, decision-distribution
by class, and per-factor "dominance" (mean weighted contribution share of
the canonical, unmodified VoI formula).

This module computes NOTHING new about novelty, uncertainty, relevance,
temporal importance, communication cost, or the VoI score/decision itself
-- it only reads Task 39's per-observation table (via Task 40's loader,
reused rather than reimplemented) and aggregates/correlates the values
already there. ``src/voi/`` is imported only to read the canonical,
unmodified default weights (for the dominance breakdown) -- never to
recompute a score or to change a weight/threshold.

Per the Task 41 mandate, this module explicitly does NOT:
  - recalibrate any weight or threshold (no write path into src/voi/ exists
    here at all);
  - optimize anything against these results;
  - convert an observed correlation into a causal claim (report text is
    phrased as "observed association", never "causes" or "explains" beyond
    what the linear formula's own algebra guarantees by construction);
  - hide or silently interpret unexpected findings -- any surprising
    finding is investigated in-line and reported under "investigation",
    separate from "observed_behavior" and "limitation".

Only relevance-DEFINED rows are included in every VoI-score-dependent
statistic (correlations, dominance, voi_by_decision) -- undefined-relevance
rows have no VoI score by construction (Task 38/39's disclosed gap) and are
reported separately as their own coverage figure, never imputed or dropped
silently from the denominator of the factor-availability figures.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.multimodal_pipeline import communication_decision as cd
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline.uncertainty_relevance import EXPECTED_FAULT_TYPE_LABELS
from src.multimodal_pipeline.voi_integration import UNDEFINED_RELEVANCE_DECISION
from src.voi.decision_policy import PolicyThresholds
from src.voi.scoring import VoIWeights

FACTOR_COLUMNS: Tuple[str, ...] = (
    "novelty",
    "uncertainty",
    "task_relevance",
    "temporal_importance",
    "resource_cost",
)
VOI_COLUMNS: Tuple[str, ...] = ("raw_voi_score", "voi_score")
DECISION_ORDER: Tuple[str, ...] = ("DISCARD", "BUFFER", "SUMMARY", "TRANSMIT")

BEHAVIOUR_ANALYSIS_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_voi_behaviour_analysis.json"
)
FACTOR_SUMMARY_CSV_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_voi_behaviour_factor_summary.csv"
)
DOMINANCE_CSV_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_voi_behaviour_dominance.csv"
)


# ---------------------------------------------------------------------------
# Basic statistics helpers
# ---------------------------------------------------------------------------


def _stats(series: pd.Series) -> Dict[str, float]:
    if len(series) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def build_factor_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/std/min/max for every factor + VoI score, per
    (fusion_config, split). Factor columns always available (n = split size);
    VoI columns restricted to the relevance-defined subset (n = n_defined)."""
    rows: List[dict] = []
    for config_name, config_group in results_df.groupby("fusion_config"):
        for split_name in ("train", "val", "test"):
            split_group = config_group[config_group["split"] == split_name]
            defined = split_group[split_group["relevance_defined"]]
            row: Dict[str, object] = {"fusion_config": config_name, "split": split_name, "n": len(split_group)}
            for col in ("novelty", "uncertainty", "temporal_importance", "resource_cost"):
                s = _stats(split_group[col])
                for k, v in s.items():
                    row[f"{col}_{k}"] = v
            # task_relevance and both VoI columns are only meaningful on the
            # relevance-defined subset (undefined rows are NaN by design).
            for col in ("task_relevance", "raw_voi_score", "voi_score"):
                s = _stats(defined[col])
                for k, v in s.items():
                    row[f"{col}_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def build_correlation_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of each factor with voi_score, on the relevance-
    defined subset only (undefined rows have no voi_score). A constant
    column (std == 0) has an undefined correlation and is reported as None,
    never silently coerced to 0."""
    rows: List[dict] = []
    for config_name, config_group in results_df.groupby("fusion_config"):
        for split_name in ("train", "val", "test"):
            defined = config_group[(config_group["split"] == split_name) & (config_group["relevance_defined"])]
            row: Dict[str, object] = {"fusion_config": config_name, "split": split_name, "n_valid": len(defined)}
            for col in FACTOR_COLUMNS:
                if len(defined) < 2 or defined[col].std() == 0 or defined["voi_score"].std() == 0:
                    row[f"corr_{col}_vs_voi_score"] = None
                else:
                    row[f"corr_{col}_vs_voi_score"] = float(defined[col].corr(defined["voi_score"]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_voi_by_decision_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """VoI score statistics grouped by the decision it produced -- a sanity
    view of "VoI vs final decision", not a new relationship (the decision is
    a deterministic function of voi_score via canonical thresholds, so this
    is expected to reproduce PolicyThresholds' boundaries, not an
    independent finding)."""
    rows: List[dict] = []
    for config_name, config_group in results_df.groupby("fusion_config"):
        for split_name in ("train", "val", "test"):
            split_group = config_group[config_group["split"] == split_name]
            for decision in DECISION_ORDER:
                subset = split_group[split_group["decision"] == decision]
                s = _stats(subset["voi_score"])
                rows.append({"fusion_config": config_name, "split": split_name, "decision": decision, **s})
    return pd.DataFrame(rows)


def build_class_conditioned_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """VoI score and decision distribution grouped by the observation's TRUE
    fault_type (not predicted class) -- same true-label convention as Task
    40's class_distribution_by_decision, for consistency across the two
    reports."""
    rows: List[dict] = []
    for config_name, config_group in results_df.groupby("fusion_config"):
        for split_name in ("train", "val", "test"):
            split_group = config_group[config_group["split"] == split_name]
            for class_name in EXPECTED_FAULT_TYPE_LABELS:
                class_group = split_group[split_group["fault_type"] == class_name]
                if len(class_group) == 0:
                    continue
                defined = class_group[class_group["relevance_defined"]]
                decision_counts = class_group["decision"].value_counts().to_dict()
                row = {
                    "fusion_config": config_name,
                    "split": split_name,
                    "fault_type": class_name,
                    "n": len(class_group),
                    "n_relevance_defined": len(defined),
                    "voi_score_mean": float(defined["voi_score"].mean()) if len(defined) else None,
                    "voi_score_median": float(defined["voi_score"].median()) if len(defined) else None,
                }
                for decision in DECISION_ORDER + (UNDEFINED_RELEVANCE_DECISION,):
                    row[f"decision_{decision}_count"] = int(decision_counts.get(decision, 0))
                rows.append(row)
    return pd.DataFrame(rows)


def build_dominance_table(results_df: pd.DataFrame, weights: VoIWeights) -> pd.DataFrame:
    """Mean weighted contribution of each factor to the VoI score, and each
    factor's share of the total positive contribution -- reused verbatim
    from Paderborn's own dominance-check pattern
    (``src/evaluation/paderborn_voi_behaviour_analysis.py::build_dominance_table``),
    computed on the relevance-defined subset (the only rows that actually
    received a VoI score)."""
    terms = {
        "novelty": weights.novelty,
        "uncertainty": weights.uncertainty,
        "task_relevance": weights.task_relevance,
        "temporal_importance": weights.temporal_importance,
    }
    rows: List[dict] = []
    for config_name, config_group in results_df.groupby("fusion_config"):
        for split_name in ("train", "val", "test"):
            defined = config_group[(config_group["split"] == split_name) & (config_group["relevance_defined"])]
            row: Dict[str, object] = {"fusion_config": config_name, "split": split_name, "n_valid": len(defined)}
            if len(defined) == 0:
                rows.append(row)
                continue

            contributions = {name: (w * defined[name]) for name, w in terms.items()}
            contributions["resource_cost"] = -weights.resource_cost * defined["resource_cost"]

            positive_total = sum(c.mean() for name, c in contributions.items() if name != "resource_cost")
            for name, contribution in contributions.items():
                row[f"{name}_mean_contribution"] = float(contribution.mean())
                if name != "resource_cost" and positive_total > 0:
                    row[f"{name}_share_of_positive_contribution_pct"] = round(
                        100.0 * contribution.mean() / positive_total, 4
                    )
                if defined[name].std() > 0 and defined["voi_score"].std() > 0:
                    row[f"{name}_corr_with_voi_score"] = float(defined[name].corr(defined["voi_score"]))
                else:
                    row[f"{name}_corr_with_voi_score"] = None
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-configuration comparison (test split only -- held-out, most relevant
# for "does this generalize" style comparisons)
# ---------------------------------------------------------------------------


def build_cross_config_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for config_name, config_group in results_df.groupby("fusion_config"):
        test_group = config_group[config_group["split"] == "test"]
        defined = test_group[test_group["relevance_defined"]]
        decision_counts = test_group["decision"].value_counts()
        n = len(test_group)
        rows.append(
            {
                "fusion_config": config_name,
                "n_test": n,
                "n_relevance_defined_test": len(defined),
                "pct_relevance_defined_test": round(100.0 * len(defined) / n, 4) if n else 0.0,
                "voi_score_mean_test": float(defined["voi_score"].mean()) if len(defined) else None,
                "voi_score_max_test": float(defined["voi_score"].max()) if len(defined) else None,
                "transmit_count_test": int(decision_counts.get("TRANSMIT", 0)),
                "transmit_pct_test": round(100.0 * decision_counts.get("TRANSMIT", 0) / n, 4) if n else 0.0,
                "discard_pct_test": round(100.0 * decision_counts.get("DISCARD", 0) / n, 4) if n else 0.0,
                "undefined_relevance_pct_test": round(
                    100.0 * decision_counts.get(UNDEFINED_RELEVANCE_DECISION, 0) / n, 4
                )
                if n
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dominant-factor and TRANSMIT-rarity investigation (descriptive only)
# ---------------------------------------------------------------------------


def _dominant_factor_findings(dominance_df: pd.DataFrame) -> List[dict]:
    """For each (config, split) with valid data, identifies which positive
    factor has the largest mean-contribution share -- a descriptive
    observation, not a recalibration decision."""
    findings: List[dict] = []
    share_cols = [c for c in dominance_df.columns if c.endswith("_share_of_positive_contribution_pct")]
    for _, row in dominance_df.iterrows():
        if row.get("n_valid", 0) == 0:
            continue
        shares = {c.replace("_share_of_positive_contribution_pct", ""): row[c] for c in share_cols if pd.notna(row.get(c))}
        if not shares:
            continue
        dominant = max(shares, key=shares.get)
        findings.append(
            {
                "fusion_config": row["fusion_config"],
                "split": row["split"],
                "dominant_factor": dominant,
                "dominant_factor_share_pct": shares[dominant],
                "all_shares_pct": shares,
            }
        )
    return findings


_EXPECTED_CORRELATION_SIGN: Dict[str, int] = {
    "novelty": 1,
    "uncertainty": 1,
    "task_relevance": 1,
    "temporal_importance": 1,
    "resource_cost": -1,
}


def _counterintuitive_correlation_investigation(
    results_df: pd.DataFrame, correlation_df: pd.DataFrame
) -> List[dict]:
    """Every canonical weight is non-negative except resource_cost (which is
    subtracted), so each factor's own weighted TERM can never itself lower
    the score. An observed NEGATIVE correlation between a positively-
    weighted factor and the overall voi_score is therefore not something
    that formula term can cause by itself -- it can only arise from how
    that factor happens to co-vary with the other factors across real
    observations (confounding). Rather than asserting a cause, this checks
    the most likely confounder in this formula -- task_relevance, since it
    carries the largest canonical weight (0.35) and dominates the score per
    the dominance table -- and reports the correlation, explicitly labeled
    as a candidate explanation, not a proven cause."""
    findings: List[dict] = []
    for _, row in correlation_df.iterrows():
        if row["n_valid"] < 2:
            continue
        config_name, split_name = row["fusion_config"], row["split"]
        for factor, expected_sign in _EXPECTED_CORRELATION_SIGN.items():
            if factor == "task_relevance":
                continue  # can't be its own confounder
            corr_col = f"corr_{factor}_vs_voi_score"
            corr = row.get(corr_col)
            if corr is None or pd.isna(corr):
                continue
            observed_sign = 1 if corr > 0 else (-1 if corr < 0 else 0)
            if observed_sign != 0 and observed_sign != expected_sign:
                defined = results_df[
                    (results_df["fusion_config"] == config_name)
                    & (results_df["split"] == split_name)
                    & (results_df["relevance_defined"])
                ]
                confound_corr = (
                    float(defined[factor].corr(defined["task_relevance"]))
                    if defined[factor].std() > 0 and defined["task_relevance"].std() > 0
                    else None
                )
                findings.append(
                    {
                        "fusion_config": config_name,
                        "split": split_name,
                        "factor": factor,
                        "expected_correlation_sign": "positive" if expected_sign > 0 else "negative",
                        "observed_corr_with_voi_score": float(corr),
                        "candidate_confounder": "task_relevance",
                        "factor_corr_with_task_relevance": confound_corr,
                        "note": (
                            f"{factor}'s own canonical weight is non-negative, so its "
                            "weighted term cannot itself lower voi_score. The observed "
                            f"negative association with voi_score is consistent with "
                            f"{factor} co-occurring with lower task_relevance predictions "
                            "in this data (task_relevance dominates the score) -- reported "
                            "as an observed association with a plausible confounder, not "
                            "a causal claim."
                        ),
                    }
                )
    return findings


def _transmit_rarity_investigation(cross_config_df: pd.DataFrame, thresholds: PolicyThresholds) -> Dict[str, object]:
    """Investigates the low TRANSMIT rate observed on real data (rather than
    silently reporting it without explanation) by checking how close the
    observed maximum VoI score gets to the TRANSMIT threshold."""
    per_config = []
    for _, row in cross_config_df.iterrows():
        max_score = row["voi_score_max_test"]
        per_config.append(
            {
                "fusion_config": row["fusion_config"],
                "transmit_pct_test": row["transmit_pct_test"],
                "voi_score_max_test": max_score,
                "gap_below_transmit_threshold": (
                    round(thresholds.summary_max - max_score, 4) if max_score is not None else None
                ),
            }
        )
    return {
        "transmit_threshold": thresholds.summary_max,
        "per_configuration": per_config,
        "note": (
            "TRANSMIT is rare but not structurally unreachable on this "
            "dataset (unlike CWRU's pre-calibration finding in "
            "docs/voi_integration_analysis.md) -- every configuration has at "
            "least some TRANSMIT observations in at least one split. Low "
            "TRANSMIT counts on val/test are consistent with the disclosed "
            "relevance-coverage and class-support gaps (zero Normal val/test "
            "support, zero BPFI test support, near-total relevance-undefined "
            "test coverage for the two current-requiring configurations) "
            "already documented in Tasks 34-38, not a new defect introduced "
            "here. This is an observation, not a recalibration -- the "
            "existing weights/thresholds are not changed by this task."
        ),
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def build_behaviour_analysis_report(
    results_df: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    weights = VoIWeights()
    thresholds = PolicyThresholds()

    factor_summary_df = build_factor_summary_table(results_df)
    correlation_df = build_correlation_table(results_df)
    voi_by_decision_df = build_voi_by_decision_table(results_df)
    class_conditioned_df = build_class_conditioned_table(results_df)
    dominance_df = build_dominance_table(results_df, weights)
    cross_config_df = build_cross_config_comparison(results_df)

    dominant_factor_findings = _dominant_factor_findings(dominance_df)
    transmit_investigation = _transmit_rarity_investigation(cross_config_df, thresholds)
    counterintuitive_correlation_findings = _counterintuitive_correlation_investigation(results_df, correlation_df)

    report: Dict[str, object] = {
        "task": 41,
        "title": "Multimodal VoI behaviour analysis",
        "scope": (
            "Read-only analysis of Task 39's already-computed VoI results. "
            "No weight, threshold, or formula in src/voi/ is changed or "
            "recalibrated by this task."
        ),
        "weights_used": asdict(weights),
        "thresholds_used": asdict(thresholds),
        "correlations": correlation_df.to_dict(orient="records"),
        "voi_by_decision": voi_by_decision_df.to_dict(orient="records"),
        "class_conditioned": class_conditioned_df.to_dict(orient="records"),
        "cross_configuration_comparison_test_split": cross_config_df.to_dict(orient="records"),
        "dominant_factor_findings": dominant_factor_findings,
        "transmit_rarity_investigation": transmit_investigation,
        "counterintuitive_correlation_investigation": counterintuitive_correlation_findings,
        "interpretation_notes": {
            "observed_behavior": (
                "Task Relevance and Novelty are the largest positive "
                "contributors to the VoI score across configurations in the "
                "dominant_factor_findings below (consistent with their "
                "larger canonical weights, 0.35 and 0.30 respectively, "
                "versus Uncertainty's 0.05) -- see per-config/split shares "
                "for exact figures, which are NOT uniform across "
                "configurations or splits."
            ),
            "interpretation": (
                "A larger weighted contribution share reflects the "
                "canonical formula's fixed weights combined with each "
                "factor's own observed value distribution on this dataset; "
                "it does not imply that factor is more 'correct' or more "
                "causally responsible for a given decision -- the VoI score "
                "is a fixed linear combination by construction, so high "
                "correlation between an individual factor and voi_score is "
                "partly guaranteed by that formula, not independent "
                "evidence of a causal or predictive relationship."
            ),
            "limitation": (
                "These figures are computed only on relevance-defined "
                "observations. The undefined-relevance subset (Misalignment/"
                "Unbalance predictions) never receives a VoI score and is "
                "excluded from every correlation/dominance/voi_by_decision "
                "figure -- reported separately as its own coverage "
                "percentage (see Task 40's report), never imputed into "
                "these statistics."
            ),
        },
        "not_performed_in_this_task": ["weight_recalibration", "threshold_recalibration", "voi_formula_modification"],
    }
    return report, factor_summary_df, dominance_df


def run_task41(results_path: str = None) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    results_path = results_path or cd.VOI_INTEGRATION_RESULTS_CSV_PATH
    results_df = cd.load_voi_integration_results(results_path)
    return build_behaviour_analysis_report(results_df)


def save_behaviour_analysis_report(report: Dict[str, object], path: str = BEHAVIOUR_ANALYSIS_JSON_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)


def save_factor_summary(factor_summary_df: pd.DataFrame, path: str = FACTOR_SUMMARY_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    factor_summary_df.to_csv(path, index=False)


def save_dominance_table(dominance_df: pd.DataFrame, path: str = DOMINANCE_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dominance_df.to_csv(path, index=False)


if __name__ == "__main__":
    print("=== Executing Task 41: Multimodal VoI Behaviour Analysis ===")
    report, factor_summary_df, dominance_df = run_task41()
    save_behaviour_analysis_report(report)
    save_factor_summary(factor_summary_df)
    save_dominance_table(dominance_df)

    print("\n=== Dominant factor per (config, split) ===")
    for finding in report["dominant_factor_findings"]:
        print(
            f"  {finding['fusion_config']}/{finding['split']}: "
            f"{finding['dominant_factor']} ({finding['dominant_factor_share_pct']:.2f}% of positive contribution)"
        )

    print("\n=== Cross-configuration comparison (test split) ===")
    for row in report["cross_configuration_comparison_test_split"]:
        print(
            f"  {row['fusion_config']}: n={row['n_test']} "
            f"relevance_defined={row['pct_relevance_defined_test']:.2f}% "
            f"transmit={row['transmit_pct_test']:.2f}% "
            f"discard={row['discard_pct_test']:.2f}%"
        )

    print(f"\nSaved report to: {BEHAVIOUR_ANALYSIS_JSON_PATH}")
    print(f"Saved factor summary to: {FACTOR_SUMMARY_CSV_PATH}")
    print(f"Saved dominance table to: {DOMINANCE_CSV_PATH}")
