"""Streamlit results dashboard for the Information-Centric Uncertainty-Aware Edge Intelligence project.

Run with: streamlit run dashboard/app.py
Reads only from artifacts already produced by the project's pipelines (see dashboard/data_loader.py).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import data_loader as dl

st.set_page_config(page_title="Edge Intelligence VoI Dashboard", layout="wide")

STAGES = [
    "Dataset",
    "Preprocessing",
    "CNN",
    "Novelty",
    "Uncertainty",
    "Relevance",
    "Temporal Importance",
    "Communication Cost",
    "VoI",
    "Decision",
    "Communication",
    "Continual Learning",
]

STAGE_PURPOSE = {
    "Dataset": "Raw bearing-vibration recordings (CWRU / IMS / Paderborn), each with its own sampling rate, channel layout, and documented provenance.",
    "Preprocessing": "Leakage-safe windowing, splitting (file-, chronological-, or measurement-level depending on the dataset), and normalization fitted only on the training portion.",
    "CNN": "1D convolutional classifier trained on CWRU windows; also produces a 64-D learned embedding used by downstream novelty scoring.",
    "Novelty": "How different an observation's CNN embedding is from the reference ('Normal') condition -- a proxy for 'have we seen something like this before?'.",
    "Uncertainty": "Normalized Shannon entropy of the CNN's predicted class probabilities -- a proxy for 'how confident is the model?'.",
    "Relevance": "How actionable the predicted condition is (e.g. a fault class matters more than 'Normal').",
    "Temporal Importance": "How much the signal changed since the previous observation -- a proxy for 'is something changing right now?'.",
    "Communication Cost": "Normalized cost of transmitting this observation (payload size, transmission time, bandwidth pressure).",
    "VoI": "Combines Novelty, Uncertainty, Relevance, and Temporal Importance, minus Communication Cost, into one weighted score.",
    "Decision": "Maps the VoI score to a discrete action: DISCARD, BUFFER, SUMMARY, or TRANSMIT.",
    "Communication": "The actual edge-to-cloud transmission implied by TRANSMIT decisions (no physical/FSO channel simulation exists yet).",
    "Continual Learning": "A supervisory loop (condition monitoring, gated prototype admission, safety/regression gate, CNN head adaptation, versioned activation) layered on top, independent of the VoI math.",
}


def render_pipeline_diagram():
    cols = st.columns(len(STAGES))
    for col, stage in zip(cols, STAGES):
        with col:
            st.markdown(
                f"<div style='text-align:center;font-size:0.75rem;border:1px solid #8884;"
                f"border-radius:6px;padding:6px'><b>{stage}</b></div>",
                unsafe_allow_html=True,
            )
    st.caption(" -> ".join(STAGES))


def render_overview():
    st.title("Project Overview")
    st.write(
        "Information-Centric Uncertainty-Aware Edge Intelligence for Semantic Sensor Communication -- "
        "a bearing-condition-monitoring pipeline that decides, per observation, whether an edge device "
        "should discard, buffer, summarize, or transmit data to the cloud, based on an estimated Value "
        "of Information (VoI)."
    )
    render_pipeline_diagram()
    st.subheader("What each stage does")
    for stage in STAGES:
        st.markdown(f"**{stage}** -- {STAGE_PURPOSE[stage]}")

    st.subheader("What has actually been run")
    status = dl.get_pipeline_stage_status()
    status_df = pd.DataFrame(
        [{"Stage": k, "Status": "done" if v else "not run / not available"} for k, v in status.items()]
    )
    st.table(status_df)
    st.info(
        "Only CWRU has been carried through CNN -> Novelty -> Uncertainty -> VoI -> Continual-Learning "
        "experiments so far. IMS and Paderborn are integrated datasets with no experiments performed on "
        "them yet -- see the Dataset Results section."
    )


def _dataset_section(title, info):
    st.subheader(title)
    if info["status"] == dl.STATUS_MISSING:
        st.warning(f"{title}: no processed dataset artifact found.")
        return
    st.markdown(f"**Experiment status:** {info['experiment_status']}")
    summary = info["summary"]
    with st.expander("Dataset statistics", expanded=True):
        st.json(summary, expanded=False)
    if info.get("split_description"):
        st.markdown(f"**Split / leakage-control strategy:** {info['split_description']}")


def render_dataset_results():
    st.title("Dataset Results")
    tabs = st.tabs(["CWRU", "IMS", "Paderborn"])
    with tabs[0]:
        _dataset_section("CWRU Bearing Vibration Dataset", dl.get_cwru_dataset_info())
    with tabs[1]:
        _dataset_section("IMS Bearing Run-to-Failure Dataset", dl.get_ims_dataset_info())
    with tabs[2]:
        _dataset_section("Paderborn University Bearing DataCenter", dl.get_paderborn_dataset_info())


def render_model_results():
    st.title("Model Results")
    st.markdown("### Canonical CNN (`src/cnn`) -- used by Novelty, Uncertainty, and the VoI pipeline")
    canonical = dl.get_canonical_cnn_results()
    if canonical["status"] == dl.STATUS_MISSING:
        st.warning("No canonical CNN results found.")
    else:
        info = canonical["model_info"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total parameters", info["total_params"] if info["total_params"] is not None else "unavailable")
        acc_row = canonical["evaluation_summary"].iloc[0]
        c2.metric("Test accuracy", f"{acc_row['accuracy']*100:.2f}%")
        c3.metric("Test weighted F1", f"{acc_row['weighted_f1']:.4f}")
        st.caption(canonical["note"])
        st.dataframe(canonical["classification_report"], use_container_width=True)
        fc1, fc2, fc3 = st.columns(3)
        if canonical["confusion_matrix_figure"]:
            fc1.image(canonical["confusion_matrix_figure"], caption="Confusion matrix")
        if canonical["training_curves_figure"]:
            fc2.image(canonical["training_curves_figure"], caption="Training curves")
        if canonical["class_performance_figure"]:
            fc3.image(canonical["class_performance_figure"], caption="Per-class performance")
        with st.expander("Training history (per epoch)"):
            st.dataframe(canonical["training_history"], use_container_width=True)

    st.markdown("---")
    st.markdown(
        "### Legacy edge/cloud CNN track (`src/cwru_pipeline/models`) -- a separate experiment, "
        "different train/val/test split (1043 test samples vs. the canonical 406)"
    )
    legacy = dl.get_legacy_edge_cloud_results()
    if legacy["status"] == dl.STATUS_MISSING:
        st.info("Not available.")
    else:
        st.warning(
            "This is a distinct pipeline from the canonical CNN above -- do not compare its accuracy "
            "directly against the 100% canonical test accuracy; they use different data splits."
        )
        b, c = legacy["baseline_cnn"], legacy["cloud_cnn"]
        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown("**Edge model (baseline CNN)**")
            st.metric("Test accuracy", f"{b['accuracy']*100:.2f}%")
            st.metric("Weighted F1", f"{b['f1_weighted']:.4f}")
            if legacy["figures"]["baseline_confusion_matrix"]:
                st.image(legacy["figures"]["baseline_confusion_matrix"])
        with lc2:
            st.markdown("**Cloud model**")
            st.metric("Test accuracy", f"{c['test_accuracy']*100:.2f}%")
            st.metric("Parameters", f"{c['num_parameters']:,}")
        with st.expander("Formal Value-of-Information (edge/cloud) results"):
            if legacy["formal_voi_results"]:
                fr = legacy["formal_voi_results"]["formal_voi_primary"]
                st.write(
                    f"Transmission rate {fr['transmission_rate']*100:.1f}%, "
                    f"resulting accuracy {fr['accuracy']*100:.2f}% "
                    f"(edge-only {legacy['formal_voi_results']['edge_only']['accuracy']*100:.2f}%, "
                    f"cloud-only {legacy['formal_voi_results']['cloud_only']['accuracy']*100:.2f}%)."
                )
                if legacy["figures"]["formal_voi_pareto"]:
                    st.image(legacy["figures"]["formal_voi_pareto"])


def _info_component_block(title, result):
    st.subheader(title)
    if result["status"] == dl.STATUS_NOT_PERFORMED:
        st.info(f"{title}: configuration exists but no experiment has been run yet.")
    st.markdown(f"**How it is calculated:** {result.get('method', '')}")
    if "config" in result:
        st.json(result["config"], expanded=False)
    if result.get("summary") is not None:
        st.markdown("**Current result (CWRU test split unless labeled otherwise):**")
        st.dataframe(result["summary"], use_container_width=True)
    for fig_key in ("distribution_figure", "by_class_figure"):
        if result.get(fig_key):
            st.image(result[fig_key])


def render_information_value_components():
    st.title("Information Value Components")
    tabs = st.tabs(["Novelty", "Uncertainty", "Task Relevance", "Temporal Importance", "Communication Cost"])
    with tabs[0]:
        st.write("**What it is:** a measure of how unfamiliar an observation is relative to a known reference condition.")
        st.write("**Why it exists:** high-novelty observations are more likely to represent an emerging fault and are worth spending communication budget on.")
        _info_component_block("Novelty", dl.get_novelty_results())
    with tabs[1]:
        st.write("**What it is:** the model's own uncertainty about its prediction.")
        st.write("**Why it exists:** low-confidence predictions may warrant a second opinion from a more capable cloud model.")
        _info_component_block("Uncertainty", dl.get_uncertainty_results())
    with tabs[2]:
        st.write("**What it is:** how actionable the predicted condition is for the monitoring task.")
        st.write("**Why it exists:** not all information is equally worth transmitting -- a confirmed fault matters more than routine 'Normal' readings.")
        _info_component_block("Task Relevance", dl.get_relevance_results())
    with tabs[3]:
        st.write("**What it is:** how much the signal changed since the previous observation.")
        st.write("**Why it exists:** a sudden change can indicate an event worth reporting even before a fault is confirmed.")
        _info_component_block("Temporal Importance", dl.get_temporal_results())
    with tabs[4]:
        st.write("**What it is:** the normalized cost of transmitting an observation.")
        st.write("**Why it exists:** VoI must trade off the value of information against the cost of sending it.")
        _info_component_block("Communication Cost", dl.get_communication_cost_results())


def render_voi_results():
    st.title("VoI Results")
    cfg = dl.get_voi_engine_config()
    st.subheader("Current VoI weights and decision thresholds (src/voi)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weights**")
        st.table(pd.DataFrame([cfg["weights"]]).T.rename(columns={0: "weight"}))
    with c2:
        st.markdown("**Decision thresholds**")
        st.table(pd.DataFrame([cfg["thresholds"]]).T.rename(columns={0: "VoI score upper bound"}))
    st.code(cfg["formula"], language="text")

    st.markdown("---")
    st.subheader("Synthetic VoI diagnostics (illustrative, not real sensor data)")
    synth = dl.get_voi_synthetic_results()
    if synth["status"] == dl.STATUS_AVAILABLE:
        st.caption(synth["note"])
        st.dataframe(synth["scenario_analysis"], use_container_width=True)
        with st.expander("Weight / threshold sensitivity"):
            st.dataframe(synth["weight_sensitivity"], use_container_width=True)
            st.dataframe(synth["threshold_sensitivity"], use_container_width=True)
        with st.expander("V0.1 validation summary (12 checks)"):
            st.dataframe(synth["v01_validation_summary"], use_container_width=True)

    st.markdown("---")
    st.subheader("CWRU VoI integration (real data, current calibrated weights)")
    integ = dl.get_voi_cwru_integration_results()
    if integ["status"] == dl.STATUS_AVAILABLE:
        st.dataframe(integ["decision_distribution"], use_container_width=True)
        st.dataframe(integ["factor_dominance"], use_container_width=True)
        fc1, fc2 = st.columns(2)
        if integ["figures"]["voi_decision_distribution"]:
            fc1.image(integ["figures"]["voi_decision_distribution"])
        if integ["figures"]["voi_factor_contribution"]:
            fc2.image(integ["figures"]["voi_factor_contribution"])
        st.caption(
            f"Full per-observation table has {integ['n_per_observation_rows']} rows "
            f"(results/tables/voi_integration_per_observation.csv); showing first 20."
        )
        st.dataframe(integ["per_observation_sample"], use_container_width=True)

    st.markdown("---")
    st.subheader("Before / after calibration (Task 13 vs Task 14, independently re-validated in Task 15)")
    calib = dl.get_voi_calibration_results()
    if calib["status"] == dl.STATUS_AVAILABLE:
        st.caption(calib["note"])
        st.dataframe(calib["decision_comparison"], use_container_width=True)
        st.dataframe(calib["transmission_reduction"], use_container_width=True)
        if calib["figures"]["before_after_decision"]:
            st.image(calib["figures"]["before_after_decision"])
        st.dataframe(calib["reproducibility"], use_container_width=True)


def render_continual_learning():
    st.title("Continual Learning")
    result = dl.get_continual_learning_results()
    if result["status"] == dl.STATUS_MISSING:
        st.warning("No continual-learning experiment results found.")
        return
    st.warning(result["note"])
    r = result["raw"]

    st.subheader("Condition monitoring")
    det = r["detection_result"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Novelty threshold", f"{det['novelty_threshold']:.5f}")
    c2.metric("Observations presented", det["n_observations_presented"])
    c3.metric("Sustained-shift fraction", f"{det['sustained_fraction_observed']*100:.1f}%")
    st.write("Status counts:", det["status_counts"])

    st.subheader("Gated prototype admission")
    adm = r["admission_result"]
    st.write(f"Decision: **{adm['decision'].upper()}** -- prototype `{adm['prototype_id']}` added "
             f"(reference version {adm['reference_version_before']} -> {adm['reference_version_after']}).")
    st.json(adm["gate_report"]["safety"], expanded=False)

    st.subheader("Safety / regression gate")
    reg = r["regression_result"]
    st.write(f"Decision: **{reg['decision'].upper()}** -- worst per-condition regression: "
             f"{reg['regression']['worst_regression']} ({reg['regression']['worst_condition_id']}).")

    st.subheader("CNN head adaptation")
    cand = r["candidate_result"]
    c1, c2 = st.columns(2)
    c1.metric("Backbone trainable params", cand["backbone_trainable_param_count"])
    c2.metric("Head trainable params", cand["head_trainable_param_count"])
    st.write("Per-condition accuracy (candidate):", cand["per_condition_accuracy_candidate"])

    st.subheader("Model versioning / activation")
    act = r["activation_result"]
    st.write(f"Activated: **{act['activated']}** -- reason: {act['reason']} -- "
             f"model version {r['model_version_before']} -> {r['model_version_after']}.")

    st.subheader("Post-hoc test-set performance (before vs after adaptation)")
    post = r["post_hoc_test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Baseline model accuracy", f"{post['baseline_model_accuracy']*100:.2f}%")
    c2.metric("Final active model accuracy", f"{post['final_active_model_accuracy']*100:.2f}%")

    st.subheader("Leakage verification")
    st.json(r["leakage_verification"], expanded=False)

    with st.expander("Full raw experiment result (JSON)"):
        st.json(r, expanded=False)


def render_comparison():
    st.title("Dataset / Experiment Comparison")
    st.subheader("Dataset integration comparison")
    cwru = dl.get_cwru_dataset_info()
    ims = dl.get_ims_dataset_info()
    pb = dl.get_paderborn_dataset_info()
    rows = []
    for name, info in (("CWRU", cwru), ("IMS", ims), ("Paderborn", pb)):
        if info["status"] != dl.STATUS_AVAILABLE:
            continue
        rows.append(
            {
                "Dataset": name,
                "Sampling rate (Hz)": info.get("sampling_rate_hz"),
                "Window size": info.get("window_size"),
                "Experiment status": info["experiment_status"],
            }
        )
    st.table(pd.DataFrame(rows))
    st.caption(
        "Only dataset-integration facts are compared here. CNN/novelty/uncertainty/VoI metrics exist "
        "for CWRU only, so no cross-dataset model-performance comparison is possible yet. "
        "XJTU-SY and MIMII are not integrated into this repository -- no code, raw data, or "
        "processed artifacts for either exist yet, so they are omitted rather than shown with "
        "placeholder rows."
    )

    st.subheader("Edge vs. cloud vs. VoI-mediated accuracy")
    st.warning(
        "The canonical pipeline (src/cnn + src/voi) and the legacy edge/cloud pipeline "
        "(src/cwru_pipeline) use different train/val/test splits (406 vs. 1043 test windows) and "
        "different VoI formulations. Only compare rows within the same track below."
    )
    legacy = dl.get_legacy_edge_cloud_results()
    if legacy["status"] == dl.STATUS_AVAILABLE and legacy["formal_voi_results"]:
        fr = legacy["formal_voi_results"]
        st.markdown("**Legacy edge/cloud track (test split, n=1043)**")
        st.table(
            pd.DataFrame(
                [
                    {"Configuration": "Edge only", "Accuracy": fr["edge_only"]["accuracy"], "Transmission rate": 0.0},
                    {"Configuration": "Cloud only", "Accuracy": fr["cloud_only"]["accuracy"], "Transmission rate": 1.0},
                    {
                        "Configuration": "Formal VoI (cost=0.05)",
                        "Accuracy": fr["formal_voi_primary"]["accuracy"],
                        "Transmission rate": fr["formal_voi_primary"]["transmission_rate"],
                    },
                    {
                        "Configuration": "Rate-matched uncertainty baseline",
                        "Accuracy": fr["rate_matched_uncertainty"]["accuracy"],
                        "Transmission rate": fr["rate_matched_uncertainty"]["transmission_rate"],
                    },
                ]
            )
        )


def render_methodology():
    st.title("Methodology -- How We Got Each Result")
    st.write(
        "Every number in this dashboard traces back through the same chain. Below is that chain for "
        "the three result families this project currently has real evidence for."
    )
    chains = {
        "CNN test accuracy (canonical)": [
            "Raw CWRU .mat files (data/raw/cwru/, obtained from the CWRU Bearing Data Center)",
            "src/cwru_pipeline/preprocessing.py: file-level train/val/test split, 2048-sample non-overlapping windows, train-only normalization -> data/processed/cwru/cwru_dataset_v1.npz",
            "src/cnn/model.py + src/cnn/train.py: 1D CNN trained on X_train/y_train, evaluated once on the held-out X_test/y_test",
            "src/evaluation/cnn_evaluation.py: classification report + confusion matrix computed from true vs. predicted test labels",
            "results/tables/cnn_classification_report.csv, results/figures/cnn_confusion_matrix.png",
            "Displayed as-is in the Model Results section above -- no recalculation in the dashboard.",
        ],
        "VoI decision distribution (CWRU, calibrated)": [
            "CNN embeddings + predicted probabilities for the CWRU test split",
            "src/novelty/novelty.py, src/uncertainty/uncertainty.py, src/relevance/relevance.py, src/temporal/temporal.py, src/communication/cost.py: the five VoI input factors, each computed independently",
            "src/integration/voi_pipeline.py -> src/voi/voi_engine.py: VoIEngine.compute_batch() combines the five factors via VoIWeights and maps the result to a DecisionAction via PolicyThresholds",
            "results/tables/voi_integration_summary.csv, voi_decision_distribution.csv, voi_factor_dominance.csv",
            "Displayed as-is in the VoI Results section above.",
        ],
        "Continual-learning experiment (Task 25)": [
            "CWRU CNN embeddings + predictions, split by recording into a 'known' (Normal) and 'new' (Inner Race Fault) condition stream",
            "src/continual/condition_monitor.py: rolling novelty + PSI check flags a candidate condition shift",
            "src/continual/admission_controller.py -> src/continual/safety_regression_gate.py: gates whether a new prototype is safe to admit",
            "src/continual/cnn_head_adaptation.py: trains a frozen-backbone candidate head on confirmed labels + rehearsal samples only",
            "src/continual/model_registry.py: versions and (if accepted) activates the candidate",
            "src/continual/cwru_continual_experiment.py orchestrates all of the above -> results/continual/task25_cwru_continual_experiment.json",
            "Displayed as-is in the Continual Learning section above.",
        ],
    }
    for title, steps in chains.items():
        st.subheader(title)
        for i, step in enumerate(steps, start=1):
            st.markdown(f"{i}. {step}")

    st.subheader("Test coverage for this project")
    tests = dl.get_test_suite_summary()
    if tests["status"] == dl.STATUS_AVAILABLE:
        st.write(f"{tests['n_test_files']} test files under tests/, covering every module referenced above.")
        with st.expander("Test files"):
            st.write(tests["test_files"])


PAGES = {
    "1. Project Overview": render_overview,
    "2. Dataset Results": render_dataset_results,
    "3. Model Results": render_model_results,
    "4. Information Value Components": render_information_value_components,
    "5. VoI Results": render_voi_results,
    "6. Continual Learning": render_continual_learning,
    "7. Dataset / Experiment Comparison": render_comparison,
    "8. Methodology": render_methodology,
}


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Section", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()
