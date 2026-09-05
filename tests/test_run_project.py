import unittest
from unittest import mock

import run_project


class TestRunStage(unittest.TestCase):
    def test_skips_when_output_already_exists_and_not_forced(self):
        run_fn = mock.Mock()
        status = run_project.run_stage("demo", lambda: True, run_fn, force=False)
        self.assertEqual(status, run_project.STAGE_SKIPPED)
        run_fn.assert_not_called()

    def test_reruns_when_output_exists_but_forced(self):
        run_fn = mock.Mock()
        status = run_project.run_stage("demo", lambda: True, run_fn, force=True)
        self.assertEqual(status, run_project.STAGE_OK)
        run_fn.assert_called_once()

    def test_skips_when_required_input_missing(self):
        run_fn = mock.Mock()
        status = run_project.run_stage(
            "demo", lambda: False, run_fn, force=False, input_check=lambda: False, input_missing_message="no input"
        )
        self.assertEqual(status, run_project.STAGE_MISSING_INPUT)
        run_fn.assert_not_called()

    def test_runs_when_output_missing_and_input_available(self):
        produced = {"done": False}

        def run_fn():
            produced["done"] = True

        status = run_project.run_stage("demo", lambda: produced["done"], run_fn, force=False)
        self.assertEqual(status, run_project.STAGE_OK)

    def test_reports_failed_on_exception(self):
        def run_fn():
            raise RuntimeError("boom")

        status = run_project.run_stage("demo", lambda: False, run_fn, force=False)
        self.assertEqual(status, run_project.STAGE_FAILED)

    def test_reports_failed_when_output_still_missing_after_run(self):
        status = run_project.run_stage("demo", lambda: False, lambda: None, force=False)
        self.assertEqual(status, run_project.STAGE_FAILED)


class TestRawDataAvailabilityChecks(unittest.TestCase):
    def _with_tmp_root(self, tmp_path):
        return mock.patch.object(run_project, "PROJECT_ROOT", tmp_path)

    def test_raw_cwru_available_requires_mat_files(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self._with_tmp_root(tmp_path):
                self.assertFalse(run_project._raw_cwru_available())
                cwru_dir = tmp_path / "data" / "raw" / "cwru"
                cwru_dir.mkdir(parents=True)
                self.assertFalse(run_project._raw_cwru_available())
                (cwru_dir / "100.mat").write_bytes(b"")
                self.assertTrue(run_project._raw_cwru_available())

    def test_raw_paderborn_available_requires_all_probe_codes(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self._with_tmp_root(tmp_path):
                base = tmp_path / "data" / "raw" / "paderborn"
                for code in ("K001", "KA01", "KB23"):
                    (base / code).mkdir(parents=True)
                self.assertFalse(run_project._raw_paderborn_available())
                (base / "KI04").mkdir(parents=True)
                self.assertTrue(run_project._raw_paderborn_available())


class TestArgParsing(unittest.TestCase):
    def test_all_flag_parses(self):
        parser = run_project.build_arg_parser()
        args = parser.parse_args(["--all", "--force"])
        self.assertTrue(args.all)
        self.assertTrue(args.force)
        self.assertFalse(args.dashboard)

    def test_individual_stage_flags_parse(self):
        parser = run_project.build_arg_parser()
        args = parser.parse_args(["--cwru", "--ims", "--continual"])
        self.assertTrue(args.cwru)
        self.assertTrue(args.ims)
        self.assertTrue(args.continual)
        self.assertFalse(args.paderborn)

    def test_no_args_prints_help_and_returns_zero(self):
        self.assertEqual(run_project.main([]), 0)


class TestMainSkipsWhenArtifactsAlreadyExist(unittest.TestCase):
    def test_all_pipeline_stages_skip_cleanly_on_a_fully_computed_repo(self):
        exit_code = run_project.main(["--all"])
        self.assertEqual(exit_code, 0)


class TestVoiStageCoversFullDependencyChain(unittest.TestCase):
    def test_voi_outputs_exist_requires_integration_and_calibration_artifacts(self):
        self.assertTrue(run_project._voi_outputs_exist())

    def test_voi_outputs_exist_is_false_if_calibration_artifact_missing(self):
        real_exists = run_project._exists

        def fake_exists(*parts):
            if parts == ("results", "tables", "calibration_validation_decision_comparison.csv"):
                return False
            return real_exists(*parts)

        with mock.patch.object(run_project, "_exists", side_effect=fake_exists):
            self.assertFalse(run_project._voi_outputs_exist())

    def test_voi_inputs_available_on_this_repo(self):
        self.assertTrue(run_project._voi_inputs_available())

    def test_stage_voi_skips_without_calling_any_pipeline_stage_when_all_outputs_present(self):
        with mock.patch("src.data_generation.synthetic_generator.generate_synthetic_dataset") as gen, mock.patch(
            "src.evaluation.run_experiment.run_synthetic_experiment"
        ) as synth, mock.patch(
            "src.evaluation.voi_behaviour_analysis.run_voi_behaviour_analysis"
        ) as integ, mock.patch(
            "src.evaluation.calibration_validation.run_calibration_validation"
        ) as calib:
            status = run_project.stage_voi(force=False)
        self.assertEqual(status, run_project.STAGE_SKIPPED)
        gen.assert_not_called()
        synth.assert_not_called()
        integ.assert_not_called()
        calib.assert_not_called()


if __name__ == "__main__":
    unittest.main()
