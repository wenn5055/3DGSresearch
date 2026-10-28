"""Verify sweep matching, validation-only selection, reuse and resource waiting without a GPU."""

import contextlib
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import yaml

import tune_sh_learning_rate as tune


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.root = tune.runner.ROOT / "outputs" / f"tuning-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.cleanup)
        (self.root / "config").mkdir()
        (self.root / "data").mkdir()
        (self.root / "3dgs_trainer.py").write_text("# fixture trainer\n")
        frames = [{"file_path": "./val/r_0"}, {"file_path": "./val/r_1"}]
        (self.root / "data/transforms_val.json").write_text(json.dumps({"frames": frames}))
        self.path = self.root / "config/base.yaml"
        self.path.write_text(yaml.safe_dump({
            "paths": {"data": "../data", "output": "../unused.png", "init_ply": None},
            "runtime": {"arch": "cpu", "seed": 7},
            "model": {"width": 16, "height": 16, "capacity": 4, "initial_splats": 1, "sh_degree": 3},
            "training": {"iterations": 3},
        }))
        self.config = tune.runner.TrainingConfig.load(self.path).printable()
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        baseline_config = deepcopy(self.config)
        baseline_config["paths"]["output"] = str(self.baseline / "model.png")
        (self.baseline / "config.yaml").write_text(yaml.safe_dump(baseline_config))
        (self.baseline / "source-files.json").write_text(json.dumps([
            {"path": path, "sha256": digest} for path, digest in tune.training_sources(self.root).items()]))
        self.record(self.baseline, baseline_config)
        self.args = tune.build_parser().parse_args([str(self.path), "--baseline-run", str(self.baseline)])

    def cleanup(self):
        if not self.root.resolve().is_relative_to((tune.runner.ROOT / "outputs").resolve()):
            raise ValueError("Refusing to clean a fixture outside outputs/")
        shutil.rmtree(self.root)

    def record(self, path, config, degree=3, steps=3):
        record = {"status": "completed", "config": config,
                  "training_result": {"completed_iterations": steps, "sh_degree": degree},
                  "evaluation": {"manifest": "transforms_test.json"},
                  "mean_metrics": {"psnr_db": 999.0, "ssim": 1.0}}
        (path / "run.json").write_text(json.dumps(record))

    def metrics(self, path, psnr):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "psnr_db", "ssim"])
            writer.writeheader()
            writer.writerows([{"file": f"./val/r_{i}", "psnr_db": psnr + i, "ssim": 0.8} for i in range(2)])

    def test_preparation_freezes_only_learning_rate_changes_and_reuses_baseline(self):
        directory = tune.prepare(self.args, self.root)
        plan = json.loads((directory / "sweep.json").read_text())
        self.assertEqual(plan["status"], "prepared")
        self.assertEqual(plan["candidates"][0]["baseline_run"], str(self.baseline))
        self.assertEqual(len(plan["candidates"]), 3)
        for candidate in plan["candidates"]:
            config = tune.runner.TrainingConfig.load(Path(candidate["config"])).printable()
            self.assertEqual(tune.comparable(config), tune.comparable(self.config))
            self.assertEqual(config["learning_rates"]["feature_rest"], candidate["rate"])
        self.assertFalse((directory / "best-config.yaml").exists())

    def test_validation_ranking_ignores_test_scores_and_incomplete_degree(self):
        directory = tune.prepare(self.args, self.root)
        calls = []

        def baseline_validation(directory, candidate, config):
            path = directory / "baseline-validation.csv"
            self.metrics(path, 20.0)
            return self.baseline, path

        def fake_run(args, root):
            self.assertEqual(args.eval_manifest, "transforms_val.json")
            config = tune.runner.TrainingConfig.load(args.config).printable()
            rate = config["learning_rates"]["feature_rest"]
            calls.append(rate)
            folder = self.root / "outputs" / f"candidate-{len(calls)}"
            folder.mkdir()
            self.record(folder, config, degree=2 if rate > 0.000125 else 3)
            self.metrics(folder / "test-metrics.csv", 30.0 if rate > 0.000125 else 21.0)
            return folder

        with contextlib.redirect_stdout(io.StringIO()), patch.object(tune, "validate_baseline", side_effect=baseline_validation), \
                patch.object(tune.runner, "run_experiment", side_effect=fake_run):
            tune.execute(directory, self.root)
            tune.execute(directory, self.root)
        plan = json.loads((directory / "sweep.json").read_text())
        self.assertEqual(calls, [0.0000625, 0.00025])
        self.assertEqual(plan["best_rate"], 0.0000625)
        self.assertEqual(plan["status"], "completed")
        self.assertFalse(plan["candidates"][2]["eligible"])
        best = yaml.safe_load((directory / "best-config.yaml").read_text())
        self.assertEqual(best["learning_rates"]["feature_rest"], 0.0000625)
        self.assertEqual(plan["candidates"][0]["psnr_db"], 20.5)

    def test_rejects_mismatched_baseline_or_changed_sources(self):
        config = yaml.safe_load((self.baseline / "config.yaml").read_text())
        config["runtime"]["seed"] = 9
        (self.baseline / "config.yaml").write_text(yaml.safe_dump(config))
        with self.assertRaises(ValueError):
            tune.prepare(self.args, self.root)
        config["runtime"]["seed"] = 7
        (self.baseline / "config.yaml").write_text(yaml.safe_dump(config))
        directory = tune.prepare(self.args, self.root)
        (self.root / "3dgs_trainer.py").write_text("# changed method\n")
        with patch.object(tune.runner, "run_experiment") as launch, self.assertRaises(ValueError):
            tune.execute(directory, self.root)
        launch.assert_not_called()
        self.assertEqual(json.loads((directory / "sweep.json").read_text())["status"], "failed")

    def test_reused_baseline_evaluation_explicitly_uses_validation_and_saved_sh_degree(self):
        directory = tune.prepare(self.args, self.root)
        candidate = json.loads((directory / "sweep.json").read_text())["candidates"][0]
        with patch.object(tune.subprocess, "run") as launch:
            tune.validate_baseline(directory, candidate, self.config)
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--manifest") + 1], "transforms_val.json")
        self.assertEqual(command[command.index("--sh-degree") + 1], "3")

    def test_waits_for_active_runs_before_proceeding(self):
        directory = tune.prepare(self.args, self.root)
        plan = json.loads((directory / "sweep.json").read_text())
        record = tune.read_record(self.baseline)
        record["status"] = "running"
        (self.baseline / "run.json").write_text(json.dumps(record))

        def complete(_):
            record["status"] = "completed"
            (self.baseline / "run.json").write_text(json.dumps(record))

        with contextlib.redirect_stdout(io.StringIO()), patch.object(tune.time, "sleep", side_effect=complete) as sleep:
            tune.wait_for_runs(directory, plan)
        sleep.assert_called_once_with(10)

    def test_invalid_rates_and_wrong_split_are_rejected(self):
        for rates in ([0.0], [float("nan")], [0.1, 0.1]):
            self.args.rates = rates
            with self.assertRaises(ValueError):
                tune.prepare(self.args, self.root)
        path = self.root / "wrong.csv"
        path.write_text("file,psnr_db,ssim\n./test/r_0,20,0.8\n./test/r_1,21,0.9\n")
        with self.assertRaises(ValueError):
            tune.read_metrics(path, self.config)


if __name__ == "__main__":
    unittest.main()
