"""Task 40 -- multimodal communication decision reporting.

Task 39 already applied the EXISTING canonical decision policy
(``src.voi.decision_policy.evaluate_decision``, called internally by the
unmodified ``VoIEngine.compute_batch``) to every relevance-defined
observation, and recorded ``voi_score``/``decision`` per row in
``multimodal_voi_integration_results.csv``. This module does not
recompute, re-derive, or re-apply any decision -- it only reports on the
decisions Task 39 already made: counts, percentages, VoI-score
distributions, class composition per decision, relevance coverage, and
which observations never entered the decision pipeline at all (modality-
invalid for a given fusion configuration).

No new decision logic, threshold, or VoI formula is introduced anywhere in
this module. ``src/voi/`` is imported only to VERIFY (never to recompute
for the report) that every recorded decision is still consistent with the
canonical, unmodified ``evaluate_decision`` policy -- a regression guard,
not a second implementation.

Undefined relevance (``decision == "UNDEFINED_RELEVANCE"``, Task 38/39's
disclosed gap for Misalignment/Unbalance) is always reported as its own
explicit category alongside DISCARD/BUFFER/SUMMARY/TRANSMIT -- never folded
into DISCARD, never excluded from the percentage denominator (which would
silently inflate the other categories' shares), and never treated as 0.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.multimodal_pipeline import fusion as fu
from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline.novelty import (
    verify_no_condition_split_leakage,
    verify_no_observation_id_leakage,
)
from src.multimodal_pipeline.uncertainty_relevance import EXPECTED_FAULT_TYPE_LABELS
from src.multimodal_pipeline.voi_integration import (
    UNDEFINED_RELEVANCE_DECISION,
    VOI_INTEGRATION_RESULTS_CSV_PATH,
)
from src.voi.decision_policy import PolicyThresholds, evaluate_decision

DECISION_CATEGORIES: Tuple[str, ...] = ("DISCARD", "BUFFER", "SUMMARY", "TRANSMIT", UNDEFINED_RELEVANCE_DECISION)
_VALID_ENGINE_DECISIONS: Tuple[str, ...] = ("DISCARD", "BUFFER", "SUMMARY", "TRANSMIT")

DECISION_CONFIG_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_communication_decision_config.json"
)
DECISION_SUMMARY_CSV_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_communication_decision_summary.csv"
)

_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "observation_id",
    "condition_code",
    "fault_type",
    "split",
    "fusion_config",
    "predicted_fault_type",
    "novelty",
    "uncertainty",
    "task_relevance",
    "temporal_importance",
    "resource_cost",
    "relevance_defined",
    "raw_voi_score",
    "voi_score",
    "decision",
)


# ---------------------------------------------------------------------------
# Loading and structural validation (this module never regenerates Task 39's
# own per-observation table -- it only reads and reports on it)
# ---------------------------------------------------------------------------


def load_voi_integration_results(path: str = VOI_INTEGRATION_RESULTS_CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Task 39 results not found at '{path}'. Task 40 requires Task 39's "
            "output to exist first -- it does not fabricate or recompute it."
        )
    df = pd.read_csv(path)
    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise AssertionError(f"Task 39 results table missing required column(s): {missing}")
    return df


# ---------------------------------------------------------------------------
# Invariant verification (regression guard against src/voi/, never a second
# implementation of the decision policy)
# ---------------------------------------------------------------------------


def verify_decisions_match_canonical_policy(
    results_df: pd.DataFrame,
    thresholds: PolicyThresholds = None,
) -> None:
    """Raise if any relevance-defined row's stored decision disagrees with
    what the EXISTING, unmodified evaluate_decision() would produce for its
    own stored voi_score. Confirms Task 39 did not drift from the canonical
    policy -- never recomputes VoI itself."""
    thresholds = thresholds or PolicyThresholds()
    defined = results_df[results_df["relevance_defined"]]
    for voi_score, stored_decision, observation_id in zip(
        defined["voi_score"], defined["decision"], defined["observation_id"]
    ):
        expected = evaluate_decision(float(voi_score), thresholds=thresholds).value
        if expected != stored_decision:
            raise AssertionError(
                f"Observation '{observation_id}': stored decision '{stored_decision}' "
                f"disagrees with canonical evaluate_decision()='{expected}' for "
                f"voi_score={voi_score}"
            )


def verify_undefined_relevance_never_fabricated(results_df: pd.DataFrame) -> None:
    """Raise if any row's relevance_defined flag disagrees with the finiteness
    of its own voi_score/decision -- i.e. undefined relevance must ALWAYS mean
    NaN voi_score and decision == UNDEFINED_RELEVANCE_DECISION, and defined
    relevance must ALWAYS mean a finite voi_score and a real decision."""
    undefined = results_df[~results_df["relevance_defined"]]
    if len(undefined):
        if not undefined["voi_score"].isna().all():
            raise AssertionError("Undefined-relevance row(s) have a non-NaN voi_score")
        if not (undefined["decision"] == UNDEFINED_RELEVANCE_DECISION).all():
            raise AssertionError("Undefined-relevance row(s) have a decision other than UNDEFINED_RELEVANCE")
        if not undefined["predicted_fault_type"].isin(["Misalignment", "Unbalance"]).all():
            raise AssertionError("Undefined-relevance row(s) found for a class with a defined relevance value")

    defined = results_df[results_df["relevance_defined"]]
    if len(defined):
        if defined["voi_score"].isna().any():
            raise AssertionError("Relevance-defined row(s) have a NaN voi_score")
        if not defined["decision"].isin(_VALID_ENGINE_DECISIONS).all():
            raise AssertionError("Relevance-defined row(s) have a decision outside the four canonical actions")


def verify_no_leakage(results_df: pd.DataFrame) -> None:
    """Reuses Task 36's own leakage checks (not reimplemented) per fusion
    configuration."""
    for config_name, group in results_df.groupby("fusion_config"):
        meta_by_split = {
            split_name: group.loc[group["split"] == split_name, ["observation_id"]]
            for split_name in ("train", "val", "test")
        }
        verify_no_observation_id_leakage(meta_by_split["train"], meta_by_split["val"], meta_by_split["test"])
        combined = group[["condition_code", "split"]]
        verify_no_condition_split_leakage(combined)


# ---------------------------------------------------------------------------
# Report tables
# ---------------------------------------------------------------------------


def _decision_counts_and_pcts(group: pd.DataFrame) -> Dict[str, object]:
    n = len(group)
    counts = {cat: int((group["decision"] == cat).sum()) for cat in DECISION_CATEGORIES}
    pcts = {cat: (round(100.0 * c / n, 4) if n else 0.0) for cat, c in counts.items()}
    return {"n": n, "counts": counts, "percentages": pcts}


def _voi_distribution(group: pd.DataFrame) -> Dict[str, object]:
    defined = group[group["relevance_defined"]]
    if len(defined) == 0:
        return {"n_valid": 0, "voi_score": None, "raw_voi_score": None}

    def _stats(series: pd.Series) -> Dict[str, float]:
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
        }

    return {
        "n_valid": int(len(defined)),
        "voi_score": _stats(defined["voi_score"]),
        "raw_voi_score": _stats(defined["raw_voi_score"]),
    }


def _relevance_coverage(group: pd.DataFrame) -> Dict[str, object]:
    n = len(group)
    n_defined = int(group["relevance_defined"].sum())
    n_undefined = n - n_defined
    return {
        "n": n,
        "n_defined": n_defined,
        "n_undefined": n_undefined,
        "pct_defined": round(100.0 * n_defined / n, 4) if n else 0.0,
        "pct_undefined": round(100.0 * n_undefined / n, 4) if n else 0.0,
    }


def _class_distribution_by_decision(group: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Grouped by the observation's TRUE fault_type (matching the CWRU/
    Paderborn per-class reporting precedent), not the model's predicted
    class. A true BPFO/BPFI/Normal observation can therefore legitimately
    appear under UNDEFINED_RELEVANCE if the model misclassified it as
    Misalignment/Unbalance -- that is a real, reportable misclassification
    outcome, not a bug in this table."""
    result: Dict[str, Dict[str, int]] = {}
    for decision in DECISION_CATEGORIES:
        subset = group[group["decision"] == decision]
        counts = {name: 0 for name in EXPECTED_FAULT_TYPE_LABELS}
        counts.update(subset["fault_type"].value_counts().to_dict())
        result[decision] = {k: int(v) for k, v in counts.items()}
    return result


def _unavailable_observations(
    config_name: str,
    split_name: str,
    n_in_results: int,
    observation_index: pd.DataFrame,
) -> Dict[str, object]:
    """Observations that meet Task 32's minimum modality requirement
    (vibration + temperature_current) for this split but never entered
    this fusion configuration's results table -- excluded upstream because
    this configuration's own required modality (motor_current and/or
    temperature) was invalid for that observation's condition. Never
    silently dropped from the report; always counted here."""
    n_canonical = int((observation_index["split"] == split_name).sum())
    n_unavailable = n_canonical - n_in_results
    return {
        "n_canonical_observations": n_canonical,
        "n_in_decision_pipeline": n_in_results,
        "n_unavailable_for_this_configuration": n_unavailable,
        "pct_unavailable": round(100.0 * n_unavailable / n_canonical, 4) if n_canonical else 0.0,
    }


def build_communication_decision_report(
    results_df: pd.DataFrame,
    observation_index: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Builds the full nested Task 40 report plus a flat summary table.

    Runs the invariant verifications first -- refuses to report on a
    results table that has drifted from the canonical decision policy or
    silently mishandled undefined relevance."""
    verify_decisions_match_canonical_policy(results_df)
    verify_undefined_relevance_never_fabricated(results_df)
    verify_no_leakage(results_df)

    per_config: Dict[str, object] = {}
    summary_rows: List[dict] = []

    for config_name, config_group in results_df.groupby("fusion_config"):
        per_split: Dict[str, object] = {}
        for split_name in ("train", "val", "test"):
            split_group = config_group[config_group["split"] == split_name]
            decision_info = _decision_counts_and_pcts(split_group)
            voi_info = _voi_distribution(split_group)
            relevance_info = _relevance_coverage(split_group)
            class_by_decision = _class_distribution_by_decision(split_group)
            unavailable_info = _unavailable_observations(
                config_name, split_name, decision_info["n"], observation_index
            )

            per_split[split_name] = {
                "n_observations": decision_info["n"],
                "decision_counts": decision_info["counts"],
                "decision_percentages": decision_info["percentages"],
                "voi_distribution": voi_info,
                "relevance_coverage": relevance_info,
                "class_distribution_by_decision": class_by_decision,
                "unavailable_observations": unavailable_info,
            }

            row = {
                "fusion_config": config_name,
                "split": split_name,
                "n": decision_info["n"],
                "n_relevance_defined": relevance_info["n_defined"],
                "n_relevance_undefined": relevance_info["n_undefined"],
                "n_unavailable_for_this_configuration": unavailable_info["n_unavailable_for_this_configuration"],
            }
            for cat in DECISION_CATEGORIES:
                row[f"{cat}_count"] = decision_info["counts"][cat]
                row[f"{cat}_pct"] = decision_info["percentages"][cat]
            summary_rows.append(row)

        per_config[config_name] = {"splits": per_split}

    report: Dict[str, object] = {
        "task": 40,
        "title": "Multimodal communication decision reporting",
        "decision_policy_source": "src.voi.decision_policy.evaluate_decision via VoIEngine.compute_batch (unmodified, canonical) -- verified, not recomputed",
        "decision_categories": list(DECISION_CATEGORIES),
        "undefined_relevance_handling": (
            "UNDEFINED_RELEVANCE is always reported as its own category, "
            "included in the percentage denominator (never excluded, never "
            "folded into DISCARD, never treated as 0)."
        ),
        "configurations": per_config,
    }
    summary_df = pd.DataFrame(summary_rows)
    return report, summary_df


def save_decision_config(report: Dict[str, object], path: str = DECISION_CONFIG_JSON_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)


def save_decision_summary(summary_df: pd.DataFrame, path: str = DECISION_SUMMARY_CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    summary_df.to_csv(path, index=False)


def run_task40(
    results_path: str = VOI_INTEGRATION_RESULTS_CSV_PATH,
    observation_index_path: str = schema.OBSERVATION_INDEX_CSV_PATH,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    results_df = load_voi_integration_results(results_path)
    observation_index = pd.read_csv(observation_index_path, usecols=["observation_id", "split"])
    return build_communication_decision_report(results_df, observation_index)


if __name__ == "__main__":
    print("=== Executing Task 40: Multimodal Communication Decision Report ===")
    report, summary_df = run_task40()
    save_decision_config(report)
    save_decision_summary(summary_df)

    for config_name, info in report["configurations"].items():
        print(f"=== {config_name} ===")
        for split_name, split_info in info["splits"].items():
            print(
                f"  {split_name}: n={split_info['n_observations']} "
                f"decisions={split_info['decision_counts']} "
                f"unavailable={split_info['unavailable_observations']['n_unavailable_for_this_configuration']}"
            )
    print(f"Saved config to: {DECISION_CONFIG_JSON_PATH}")
    print(f"Saved summary to: {DECISION_SUMMARY_CSV_PATH}")
