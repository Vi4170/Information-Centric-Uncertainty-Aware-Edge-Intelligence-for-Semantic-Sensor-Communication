"""Task 38 -- multimodal relevance: evidence audit and canonical 5-class map.

======================================================================
Task 38 mandate
======================================================================
Determine whether defensible evidence exists for task-relevance values
for the two currently undefined multimodal fault classes:

    3  Misalignment  (rotor/shaft misalignment)
    4  Unbalance     (rotor mass imbalance)

and, if it does, finalize the complete five-class map; if it does not,
formally keep the mapping PARTIAL and preserve the undefined values.

======================================================================
Evidence audit (sources searched -- Task 38 scope)
======================================================================

Source 1 -- repository canonical map (src/relevance/config.py):
    CLASS_RELEVANCE_MAP covers exactly 4 CWRU bearing-race fault classes
    (Normal, Inner Race Fault, Ball Fault, Outer Race Fault). The module's
    own docstring says these are "initial design parameters, NOT optimized,
    learned, or calibrated." No Misalignment or Unbalance entry exists.
    Verdict: no evidence for Misalignment/Unbalance here.

Source 2 -- repository documentation (docs/task_relevance_report.md):
    Searched for "Misalignment", "Unbalance", "rotor", "shaft".
    The report addresses bearing-race-fault severity ordering only; it
    makes no claim about shaft/rotor failure modes.
    Verdict: no evidence for Misalignment/Unbalance here.

Source 3 -- dataset companion paper (Jung et al. 2023, Data in Brief
    48:109049, DOI 10.1016/j.dib.2023.109049):
    This is the only project-referenced external source for this dataset
    (cited in src/multimodal_pipeline/dataset_audit.py and
    relevance_prediction_audit.py). It is a data-acquisition methodology
    descriptor: it documents the experimental rig, sensor placement, and
    recording protocols. It assigns NO fault-priority ranking, severity
    weighting, or criticality ordering to any condition. Specifically, it
    does NOT compare the operational risk or detection urgency of
    Misalignment vs. Unbalance vs. bearing-race faults.
    Verdict: no evidence for Misalignment/Unbalance here.

Source 4 -- engineering literature (general domain knowledge):
    Misalignment and Unbalance are genuinely common and operationally
    important rotating-machine fault modes. However, their relative
    criticality compared to bearing defects is:
      * context-dependent (machine type, operating speed, load profile,
        maintenance policy, consequence of failure);
      * not established by any source cited in or tied to this
        repository's experimental design;
      * not calibrated against this specific KAIST dataset's operating
        conditions (3-speed, 0/2/4 Nm load, 5-fault-type scheme).
    Importing domain knowledge from the general literature WITHOUT a
    specific authoritative grounding for THIS dataset's conditions would
    be exactly the "arbitrary values / guessed values" the task prohibits.
    Verdict: no defensible evidence that can be legitimately applied here.

======================================================================
Conclusion (Task 38)
======================================================================
Grade B -- PARTIAL MAPPING ONLY.

The three defined values (Normal, BPFI, BPFO) are kept unchanged from
Task 37's already-established map, reused by genuine physical fault-
category correspondence with CWRU's own CLASS_RELEVANCE_MAP:

    0  Normal        -> 0.10  (same as CWRU Normal: low relevance, routine)
    1  BPFI          -> 1.00  (same as CWRU Inner Race Fault: critical defect)
    2  BPFO          -> 0.90  (same as CWRU Outer Race Fault; BPFO = outer-
                                race defect, not the ball-fault value despite
                                both being 0.90 in CWRU's map)
    3  Misalignment  -> None  (undefined -- no defensible source found)
    4  Unbalance     -> None  (undefined -- no defensible source found)

The two undefined values are NOT fabricated, NOT set to zero, NOT
imputed as the mean or median of the defined values, NOT copied from
another class. They remain None and are propagated as NaN in any
downstream numeric pipeline, with explicit downstream handling at every
consumer (Task 39 VoI integration excludes undefined-relevance
observations from the VoI score, does not replace NaN with a number).

This module is the single canonical source of the 5-class map for all
of Tasks 38-42. src/multimodal_pipeline/uncertainty_relevance.py's own
MULTIMODAL_RELEVANCE_MAP is imported here rather than duplicated,
keeping the map defined in exactly one place (uncertainty_relevance.py,
which was the first to establish it), and re-exported with the Task 38
evidence metadata attached.

src/voi/, src/relevance/, src/cnn/, src/novelty/, src/uncertainty/ are
not imported for modification and are not changed by this module.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.multimodal_pipeline import observation_schema as schema
from src.multimodal_pipeline.uncertainty_relevance import (
    EXPECTED_FAULT_TYPE_LABELS,
    MULTIMODAL_RELEVANCE_MAP,
    UNDEFINED_RELEVANCE_CLASSES,
)

# ---------------------------------------------------------------------------
# Re-export the canonical map (single source of truth: uncertainty_relevance)
# ---------------------------------------------------------------------------

FIVE_CLASS_RELEVANCE_MAP: Dict[int, Optional[float]] = MULTIMODAL_RELEVANCE_MAP
"""The 5-class relevance map for Tasks 38-42.  None = undefined (never zero).

Provenance: established in Task 37 (src/multimodal_pipeline/uncertainty_relevance.py),
re-exported here with full evidence documentation for Task 38.
"""

DEFINED_CLASSES: Tuple[str, ...] = tuple(
    EXPECTED_FAULT_TYPE_LABELS[i]
    for i, v in FIVE_CLASS_RELEVANCE_MAP.items()
    if v is not None
)
"""Fault types that have a defined (non-None, non-NaN) relevance value."""

UNDEFINED_CLASSES: Tuple[str, ...] = UNDEFINED_RELEVANCE_CLASSES
"""Fault types whose relevance value is explicitly undefined (None / NaN)."""

# ---------------------------------------------------------------------------
# Evidence record (one entry per fault class)
# ---------------------------------------------------------------------------

#: Per-class evidence and rationale, keyed by fault_type string name.
CLASS_EVIDENCE: Dict[str, Dict[str, object]] = {
    "Normal": {
        "relevance_value": FIVE_CLASS_RELEVANCE_MAP[0],
        "status": "defined",
        "source": "CWRU CLASS_RELEVANCE_MAP[0] (src/relevance/config.py)",
        "reuse_basis": (
            "Genuine physical equivalence: 'Normal' (no fault) in the KAIST dataset "
            "is the same operational condition as 'Normal' in CWRU's bearing-fault "
            "scheme -- neither has an active fault, so low relevance (0.10) applies "
            "equally. Reused verbatim, not imputed."
        ),
        "why_not_fabricated": "N/A -- value exists in authoritative project source.",
    },
    "BPFI": {
        "relevance_value": FIVE_CLASS_RELEVANCE_MAP[1],
        "status": "defined",
        "source": "CWRU CLASS_RELEVANCE_MAP[1] 'Inner Race Fault' (src/relevance/config.py)",
        "reuse_basis": (
            "Genuine physical equivalence: BPFI = ball-pass-frequency inner-race defect, "
            "exactly the same bearing defect location as CWRU's 'Inner Race Fault'. "
            "Maximum relevance (1.00) applies for the same fault-criticality reason. "
            "Reused verbatim."
        ),
        "why_not_fabricated": "N/A -- value exists in authoritative project source.",
    },
    "BPFO": {
        "relevance_value": FIVE_CLASS_RELEVANCE_MAP[2],
        "status": "defined",
        "source": "CWRU CLASS_RELEVANCE_MAP[3] 'Outer Race Fault' (src/relevance/config.py)",
        "reuse_basis": (
            "Genuine physical equivalence: BPFO = ball-pass-frequency outer-race defect, "
            "exactly the same bearing defect location as CWRU's 'Outer Race Fault' (index 3). "
            "High relevance (0.90) applies. NOTE: CWRU index 2 (Ball Fault) coincidentally "
            "also has value 0.90 in CWRU's map, but BPFO is NOT a ball defect -- the source "
            "is CLASS_RELEVANCE_MAP[3] (Outer Race Fault), not [2] (Ball Fault)."
        ),
        "why_not_fabricated": "N/A -- value exists in authoritative project source.",
    },
    "Misalignment": {
        "relevance_value": None,
        "status": "undefined",
        "source": "None found",
        "sources_searched": [
            "src/relevance/config.py (CLASS_RELEVANCE_MAP -- 4-class, bearing-race only)",
            "docs/task_relevance_report.md (bearing-race severity only, no rotor/shaft coverage)",
            "Jung et al. 2023, Data in Brief 48:109049 (data-acquisition descriptor; no fault priority)",
            "General rotating-machine fault literature (context-dependent; not tied to this dataset)",
        ],
        "why_not_fabricated": (
            "Misalignment (rotor/shaft misalignment) is a genuinely different failure-mode "
            "category from bearing-race defects. Its operational criticality relative to "
            "BPFI/BPFO is context-dependent (machine type, speed, load, maintenance policy) "
            "and is not established by any source cited in or tied to this repository's "
            "experimental design. No authoritative ranking exists in: this repo's own "
            "CLASS_RELEVANCE_MAP, its documentation, or the only project-cited external "
            "paper for this dataset (Jung et al. 2023). Using an arbitrary value, a mean "
            "of defined values, or copying BPFI/BPFO's value without basis would violate "
            "the task requirement to not fabricate. Kept as None."
        ),
    },
    "Unbalance": {
        "relevance_value": None,
        "status": "undefined",
        "source": "None found",
        "sources_searched": [
            "src/relevance/config.py (CLASS_RELEVANCE_MAP -- 4-class, bearing-race only)",
            "docs/task_relevance_report.md (bearing-race severity only, no rotor/shaft coverage)",
            "Jung et al. 2023, Data in Brief 48:109049 (data-acquisition descriptor; no fault priority)",
            "General rotating-machine fault literature (context-dependent; not tied to this dataset)",
        ],
        "why_not_fabricated": (
            "Unbalance (rotor mass imbalance) is a genuinely different failure-mode "
            "category from bearing-race defects. Same reasoning as Misalignment: no "
            "authoritative source in this repository or its cited external references "
            "establishes a relevance value for this class in this specific experimental "
            "context. Kept as None."
        ),
    },
}

# ---------------------------------------------------------------------------
# Task 38 mapping status
# ---------------------------------------------------------------------------

MAPPING_STATUS: str = "B_PARTIAL_MAPPING_ONLY"
MAPPING_CONCLUSION: str = (
    "Task 38 evidence audit found no defensible basis for defining relevance "
    "values for Misalignment or Unbalance. The five-class map remains PARTIAL: "
    "Normal=0.10, BPFI=1.00, BPFO=0.90 are defined by genuine physical fault-"
    "category correspondence with CWRU's CLASS_RELEVANCE_MAP; Misalignment and "
    "Unbalance remain explicitly undefined (None). No fabrication."
)

RELEVANCE_AUDIT_JSON_PATH: str = os.path.join(
    schema.PROCESSED_DATA_DIR, "multimodal_relevance_audit.json"
)


# ---------------------------------------------------------------------------
# Coverage utilities
# ---------------------------------------------------------------------------


def relevance_is_defined(fault_type: str) -> bool:
    """Return True iff the given fault_type has a defined (non-None) relevance value."""
    idx = {name: i for i, name in enumerate(EXPECTED_FAULT_TYPE_LABELS)}
    if fault_type not in idx:
        raise ValueError(
            f"Unknown fault_type '{fault_type}'. Expected one of {EXPECTED_FAULT_TYPE_LABELS}"
        )
    return FIVE_CLASS_RELEVANCE_MAP[idx[fault_type]] is not None


def relevance_value_for_class(predicted_class: int) -> Optional[float]:
    """Return the relevance value (float) or None for the given integer class index."""
    if predicted_class not in FIVE_CLASS_RELEVANCE_MAP:
        raise ValueError(
            f"predicted_class {predicted_class} not in map. Valid: {list(FIVE_CLASS_RELEVANCE_MAP)}"
        )
    return FIVE_CLASS_RELEVANCE_MAP[predicted_class]


def compute_relevance_coverage_summary(results_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Compute per-config, per-split relevance coverage from the Task 37 results DataFrame.

    'Defined' = observation whose relevance_score is finite (not NaN).
    'Undefined' = NaN relevance_score (predicted Misalignment or Unbalance).
    Never treats undefined as zero. Returns counts and percentages only.
    """
    # Import here to avoid pulling in nptdms/heavy deps at module load time
    from src.multimodal_pipeline import fusion as fu  # noqa: PLC0415

    coverage: Dict[str, Dict[str, object]] = {}
    for config_name in fu.FUSION_CONFIGS:
        coverage[config_name] = {}
        for split in ("train", "val", "test"):
            sub = results_df[
                (results_df["fusion_config"] == config_name) & (results_df["split"] == split)
            ]
            n = int(len(sub))
            if n == 0:
                coverage[config_name][split] = {
                    "n": 0,
                    "n_defined": 0,
                    "n_undefined": 0,
                    "pct_defined": 0.0,
                    "pct_undefined": 0.0,
                }
                continue
            defined_mask = sub["relevance_score"].notna()
            n_defined = int(defined_mask.sum())
            n_undefined = n - n_defined
            coverage[config_name][split] = {
                "n": n,
                "n_defined": n_defined,
                "n_undefined": n_undefined,
                "pct_defined": round(100.0 * n_defined / n, 4),
                "pct_undefined": round(100.0 * n_undefined / n, 4),
            }
    return coverage


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_relevance_audit_record(
    uncertainty_relevance_csv_path: Optional[str] = None,
) -> Dict[str, object]:
    """Build the Task 38 relevance audit record.

    If the Task 37 results CSV is available, also computes coverage statistics.
    Coverage is informational only -- it does not change the mapping conclusion.
    """
    from src.multimodal_pipeline.uncertainty_relevance import (
        UNCERTAINTY_RELEVANCE_RESULTS_CSV_PATH,
    )

    csv_path = uncertainty_relevance_csv_path or UNCERTAINTY_RELEVANCE_RESULTS_CSV_PATH

    coverage: Optional[Dict] = None
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        coverage = compute_relevance_coverage_summary(results_df)

    record: Dict[str, object] = {
        "task": "38",
        "title": "Multimodal relevance: evidence audit and 5-class map",
        "mapping_status": MAPPING_STATUS,
        "mapping_conclusion": MAPPING_CONCLUSION,
        "five_class_map": {
            EXPECTED_FAULT_TYPE_LABELS[k]: v for k, v in FIVE_CLASS_RELEVANCE_MAP.items()
        },
        "defined_classes": list(DEFINED_CLASSES),
        "undefined_classes": list(UNDEFINED_CLASSES),
        "per_class_evidence": CLASS_EVIDENCE,
        "relevance_coverage_by_config_and_split": coverage,
        "downstream_handling_of_undefined": (
            "Task 39 VoI integration: observations whose predicted class is "
            "Misalignment or Unbalance receive relevance_score=NaN and are NOT "
            "passed to the VoI engine (which rejects NaN inputs). These observations "
            "are given voi_score=NaN and decision='UNDEFINED_RELEVANCE' in the output. "
            "They are counted and reported separately in Task 40's decision summary. "
            "NaN is never replaced with 0 or any other number."
        ),
        "protected_modules_unchanged": [
            "src/voi/",
            "src/relevance/",
            "src/cnn/",
            "src/novelty/",
            "src/uncertainty/",
        ],
    }
    return record


def save_relevance_audit(
    record: Dict[str, object],
    path: str = RELEVANCE_AUDIT_JSON_PATH,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    rec = build_relevance_audit_record()
    save_relevance_audit(rec)
    print("Task 38 relevance audit complete.")
    print("Mapping status:", rec["mapping_status"])
    print("Defined classes:", rec["defined_classes"])
    print("Undefined classes:", rec["undefined_classes"])
    if rec["relevance_coverage_by_config_and_split"]:
        print("\nRelevance coverage (pct_defined):")
        for cfg, splits in rec["relevance_coverage_by_config_and_split"].items():
            for sp, info in splits.items():
                print(f"  {cfg}/{sp}: {info['pct_defined']:.2f}% defined ({info['n_defined']}/{info['n']})")
