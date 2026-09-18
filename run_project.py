"""Master orchestration entry point for the Information-Centric Uncertainty-Aware Edge Intelligence project.

Runs the project's existing pipelines in dependency order:
Dataset -> Preprocessing -> CNN -> Novelty -> Uncertainty -> VoI -> Continual Learning
-> Paderborn/IMS experiments -> Multimodal fusion + VoI track (Tasks 31-42) -> Dashboard.

Calls the existing project modules directly; it does not reimplement any research logic.
Each stage is skipped if its output artifact already exists, unless --force is given.
XJTU-SY and MIMII-DG have preprocessing pipelines only (src/xjtu_pipeline/,
src/mimii_dg_pipeline/) -- no CNN/VoI experiment exists for either yet, so there is
no stage for them here.

Usage:
    python run_project.py --all                       # everything below, in order
    python run_project.py --datasets
    python run_project.py --cwru --ims --paderborn
    python run_project.py --cnn --novelty --uncertainty --voi --continual
    python run_project.py --paderborn-experiments --ims-experiments
    python run_project.py --multimodal                # Tasks 31-42, needs data/raw/multimodal/
    python run_project.py --all --force
    python run_project.py --dashboard
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE_OK = "done"
STAGE_SKIPPED = "skipped (already available)"
STAGE_MISSING_INPUT = "skipped (required input not found)"
STAGE_FAILED = "failed"


def _p(*parts):
    return PROJECT_ROOT.joinpath(*parts)


def _exists(*parts):
    return _p(*parts).exists()


def _raw_cwru_available():
    d = _p("data", "raw", "cwru")
    return d.is_dir() and any(d.glob("*.mat"))


def _raw_ims_available():
    d = _p("data", "raw", "ims")
    return all((d / sub).is_dir() for sub in ("1st_test", "2nd_test")) and (
        (d / "3rd_test").is_dir() or (d / "4th_test").is_dir()
    )


def _raw_paderborn_available():
    d = _p("data", "raw", "paderborn")
    return d.is_dir() and all((d / code).is_dir() for code in ("K001", "KA01", "KB23", "KI04"))


def _raw_multimodal_available():
    d = _p("data", "raw", "multimodal")
    return (
        (d / "vibration").is_dir()
        and any((d / "vibration").glob("*.mat"))
        and (d / "temperature_current").is_dir()
        and any((d / "temperature_current").glob("*.tdms"))
    )


def run_stage(name, output_check, run_fn, force, input_check=None, input_missing_message=""):
    print(f"\n--- {name} ---")
    if input_check is not None and not input_check():
        print(f"SKIP: {input_missing_message}")
        return STAGE_MISSING_INPUT
    if not force and output_check():
        print("SKIP: output artifact already exists (use --force to recompute).")
        return STAGE_SKIPPED
    try:
        run_fn()
    except Exception as exc:
        print(f"FAILED: {exc!r}")
        return STAGE_FAILED
    if output_check():
        print("OK.")
        return STAGE_OK
    print("FAILED: stage completed without raising, but expected output artifact is still missing.")
    return STAGE_FAILED


def stage_cwru_preprocessing(force):
    from src.cwru_pipeline.preprocessing import run_cwru_preprocessing_pipeline

    return run_stage(
        "CWRU preprocessing",
        lambda: _exists("data", "processed", "cwru", "summary.json"),
        run_cwru_preprocessing_pipeline,
        force,
        input_check=_raw_cwru_available,
        input_missing_message="data/raw/cwru not found (raw CWRU .mat files must be added manually).",
    )


def stage_ims_preprocessing(force):
    from src.ims_pipeline.preprocessing import save_ims_dataset_summary

    return run_stage(
        "IMS preprocessing",
        lambda: _exists("data", "processed", "ims", "ims_dataset_summary.json"),
        save_ims_dataset_summary,
        force,
        input_check=_raw_ims_available,
        input_missing_message="data/raw/ims not found or incomplete (extracted IMS archives must be added manually).",
    )


def stage_paderborn_preprocessing(force):
    from src.paderborn_pipeline.preprocessing import save_paderborn_dataset_summary

    return run_stage(
        "Paderborn preprocessing",
        lambda: _exists("data", "processed", "paderborn", "paderborn_dataset_summary.json"),
        save_paderborn_dataset_summary,
        force,
        input_check=_raw_paderborn_available,
        input_missing_message="data/raw/paderborn not found or incomplete (extracted Paderborn archives must be added manually).",
    )


def stage_cnn(force):
    from src.cnn.train import run_training_pipeline

    return run_stage(
        "CNN training/evaluation",
        lambda: _exists("models", "cwru_cnn_baseline.keras")
        and _exists("results", "tables", "cnn_evaluation_summary.csv"),
        run_training_pipeline,
        force,
        input_check=lambda: _exists("data", "processed", "cwru", "cwru_dataset_v1.npz"),
        input_missing_message="data/processed/cwru/cwru_dataset_v1.npz not found; run CWRU preprocessing first.",
    )


def stage_novelty(force):
    from src.novelty.novelty import run_novelty_pipeline

    return run_stage(
        "Novelty estimation",
        lambda: _exists("results", "tables", "novelty_scores_summary.csv"),
        run_novelty_pipeline,
        force,
        input_check=lambda: _exists("models", "cwru_cnn_baseline.keras"),
        input_missing_message="CNN model not found; run the CNN stage first.",
    )


def stage_uncertainty(force):
    from src.uncertainty.uncertainty import run_uncertainty_pipeline

    return run_stage(
        "Uncertainty estimation",
        lambda: _exists("results", "tables", "uncertainty_scores_summary.csv"),
        run_uncertainty_pipeline,
        force,
        input_check=lambda: _exists("models", "cwru_cnn_baseline.keras"),
        input_missing_message="CNN model not found; run the CNN stage first.",
    )


def _voi_outputs_exist():
    return (
        _exists("results", "tables", "voi_decision_distribution.csv")
        and _exists("results", "tables", "voi_integration_summary.csv")
        and _exists("results", "tables", "calibration_validation_decision_comparison.csv")
    )


def _voi_inputs_available():
    return _exists("models", "cwru_cnn_baseline.keras") and _exists(
        "data", "processed", "cwru", "cwru_dataset_v1.npz"
    )


def stage_voi(force):
    from src.data_generation.synthetic_generator import generate_synthetic_dataset
    from src.evaluation.run_experiment import run_synthetic_experiment
    from src.evaluation.voi_behaviour_analysis import run_voi_behaviour_analysis
    from src.evaluation.calibration_validation import run_calibration_validation

    def _run():
        if force or not _exists("data", "synthetic", "synthetic_voi_dataset.csv"):
            generate_synthetic_dataset()
        run_synthetic_experiment()
        run_voi_behaviour_analysis()
        run_calibration_validation()

    return run_stage(
        "VoI diagnostics + CWRU integration + calibration validation",
        _voi_outputs_exist,
        _run,
        force,
        input_check=_voi_inputs_available,
        input_missing_message="CNN model or CWRU processed dataset not found; run the CNN stage first.",
    )


def stage_continual(force):
    from src.continual.cwru_continual_experiment import run_cwru_experiment, save_result_json

    def _run():
        with tempfile.TemporaryDirectory(prefix="cwru_continual_experiment_registry_") as tmp_registry_dir:
            experiment_result = run_cwru_experiment(registry_dir=tmp_registry_dir)
        save_result_json(experiment_result)

    return run_stage(
        "Continual-learning experiment (Task 25)",
        lambda: _exists("results", "continual", "task25_cwru_continual_experiment.json"),
        _run,
        force,
        input_check=lambda: _exists("models", "cwru_cnn_baseline.keras")
        and _exists("data", "processed", "cwru", "cwru_dataset_v1.npz"),
        input_missing_message="CNN model or CWRU processed dataset not found; run the CNN stage first.",
    )


def stage_paderborn_experiments(force):
    from src.paderborn_pipeline.classification_task import DATASET_V1_PATH, save_paderborn_classification_dataset
    from src.evaluation.paderborn_cnn_experiment import run_paderborn_cnn_pipeline
    from src.evaluation.paderborn_voi_behaviour_analysis import run_paderborn_voi_behaviour_analysis

    def _run():
        if force or not _exists("data", "processed", "paderborn", "paderborn_dataset_v1.npz"):
            save_paderborn_classification_dataset()
        run_paderborn_cnn_pipeline(data_path=str(DATASET_V1_PATH))
        run_paderborn_voi_behaviour_analysis()

    return run_stage(
        "Paderborn CNN + VoI experiments",
        lambda: _exists("models", "paderborn_cnn_baseline.keras")
        and _exists("results", "tables", "paderborn_voi_decision_distribution.csv"),
        _run,
        force,
        input_check=_raw_paderborn_available,
        input_missing_message="data/raw/paderborn not found or incomplete (extracted Paderborn archives must be added manually).",
    )


def stage_ims_experiments(force):
    from src.evaluation.ims_temporal_analysis import run_ims_temporal_analysis

    return run_stage(
        "IMS novelty/temporal-importance analysis",
        lambda: _exists("results", "tables", "ims_temporal_progression_summary.csv"),
        run_ims_temporal_analysis,
        force,
        input_check=lambda: _exists("data", "processed", "ims", "ims_dataset_summary.json"),
        input_missing_message="IMS not preprocessed; run --ims first.",
    )


_MULTIMODAL_MODULES = (
    "src.multimodal_pipeline.dataset_audit",
    "src.multimodal_pipeline.observation_schema",
    "src.multimodal_pipeline.preprocessing",
    "src.multimodal_pipeline.representation",
    "src.multimodal_pipeline.fusion",
    "src.multimodal_pipeline.novelty",
    "src.multimodal_pipeline.uncertainty_relevance",
    "src.multimodal_pipeline.relevance",
    "src.multimodal_pipeline.voi_integration",
    "src.multimodal_pipeline.communication_decision",
    "src.multimodal_pipeline.voi_behaviour_analysis",
    "src.multimodal_pipeline.end_to_end_validation",
)


def stage_multimodal(force):
    def _run():
        for module in _MULTIMODAL_MODULES:
            subprocess.run([sys.executable, "-m", module], cwd=str(PROJECT_ROOT), check=True)

    return run_stage(
        "Multimodal fusion + VoI track (Tasks 31-42)",
        lambda: _exists("data", "processed", "multimodal", "multimodal_end_to_end_validation_config.json"),
        _run,
        force,
        input_check=_raw_multimodal_available,
        input_missing_message="data/raw/multimodal not found or incomplete (extracted KAIST vibration/temperature_current archives must be added manually).",
    )


def stage_dashboard():
    print("\n--- Launching dashboard ---")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(_p("dashboard", "app.py"))], cwd=str(PROJECT_ROOT))
    return STAGE_OK


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Run every applicable stage (datasets, CNN, novelty, uncertainty, VoI, continual learning).")
    parser.add_argument("--datasets", action="store_true", help="Run CWRU + IMS + Paderborn preprocessing.")
    parser.add_argument("--cwru", action="store_true", help="Run CWRU preprocessing.")
    parser.add_argument("--ims", action="store_true", help="Run IMS preprocessing.")
    parser.add_argument("--paderborn", action="store_true", help="Run Paderborn preprocessing.")
    parser.add_argument("--cnn", action="store_true", help="Train/evaluate the canonical CWRU CNN.")
    parser.add_argument("--novelty", action="store_true", help="Run novelty estimation.")
    parser.add_argument("--uncertainty", action="store_true", help="Run uncertainty estimation.")
    parser.add_argument("--voi", action="store_true", help="Run VoI synthetic diagnostics + CWRU integration.")
    parser.add_argument("--continual", action="store_true", help="Run the CWRU continual-learning experiment (Task 25).")
    parser.add_argument("--paderborn-experiments", action="store_true", help="Train Paderborn CNN + run its VoI behaviour analysis.")
    parser.add_argument("--ims-experiments", action="store_true", help="Run IMS novelty/temporal-importance run-to-failure analysis.")
    parser.add_argument("--multimodal", action="store_true", help="Run the full multimodal fusion + VoI track (Tasks 31-42).")
    parser.add_argument("--dashboard", action="store_true", help="Launch the Streamlit results dashboard after any requested stages.")
    parser.add_argument("--force", action="store_true", help="Recompute stages even if their output artifacts already exist.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_datasets = args.all or args.datasets or args.cwru or args.ims or args.paderborn
    run_cwru = args.all or args.datasets or args.cwru
    run_ims = args.all or args.datasets or args.ims
    run_paderborn = args.all or args.datasets or args.paderborn
    run_cnn = args.all or args.cnn
    run_novelty = args.all or args.novelty
    run_uncertainty = args.all or args.uncertainty
    run_voi = args.all or args.voi
    run_continual = args.all or args.continual
    run_paderborn_experiments = args.all or args.paderborn_experiments
    run_ims_experiments = args.all or args.ims_experiments
    run_multimodal = args.all or args.multimodal

    if not any(
        [
            run_datasets,
            run_cnn,
            run_novelty,
            run_uncertainty,
            run_voi,
            run_continual,
            run_paderborn_experiments,
            run_ims_experiments,
            run_multimodal,
            args.dashboard,
        ]
    ):
        parser.print_help()
        return 0

    results = {}
    if run_cwru:
        results["CWRU preprocessing"] = stage_cwru_preprocessing(args.force)
    if run_ims:
        results["IMS preprocessing"] = stage_ims_preprocessing(args.force)
    if run_paderborn:
        results["Paderborn preprocessing"] = stage_paderborn_preprocessing(args.force)
    if run_cnn:
        results["CNN training/evaluation"] = stage_cnn(args.force)
    if run_novelty:
        results["Novelty estimation"] = stage_novelty(args.force)
    if run_uncertainty:
        results["Uncertainty estimation"] = stage_uncertainty(args.force)
    if run_voi:
        results["VoI diagnostics + CWRU integration + calibration validation"] = stage_voi(args.force)
    if run_continual:
        results["Continual-learning experiment"] = stage_continual(args.force)
    if run_paderborn_experiments:
        results["Paderborn CNN + VoI experiments"] = stage_paderborn_experiments(args.force)
    if run_ims_experiments:
        results["IMS novelty/temporal-importance analysis"] = stage_ims_experiments(args.force)
    if run_multimodal:
        results["Multimodal fusion + VoI track (Tasks 31-42)"] = stage_multimodal(args.force)

    print("\n=== Execution Summary ===")
    for stage, status in results.items():
        print(f"{stage:45s} {status}")
    if not results:
        print("(no pipeline stages requested)")

    failed = [s for s, status in results.items() if status == STAGE_FAILED]

    if args.dashboard:
        stage_dashboard()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
