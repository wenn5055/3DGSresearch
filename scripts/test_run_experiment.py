"""Exercise experiment orchestration with tiny subprocess fixtures, without a GPU."""

import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid
import zipfile

import yaml

import run_experiment as runner


TRAINER = '''
from pathlib import Path
import sys
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
assert Path(config['paths']['data']).is_absolute()
assert Path(config['paths']['init_ply']).is_file()
Path(config['paths']['output']).with_suffix('.ply').write_text('ply\\n')
degree = config['model'].get('sh_degree', 0)
if degree:
    Path(config['paths']['output']).with_suffix('.sh.npz').write_text('fixture SH sidecar')
    print(f'Final SH degree: {degree} (configured maximum: {degree})')
print('trainer stderr retained', file=sys.stderr)
print('Training finished after 2 iterations in 0.12s (0.060s/iter).')
'''

EVALUATOR = '''
import argparse
import csv
import json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('ply')
p.add_argument('data')
for flag in ('manifest', 'arch', 'width', 'height', 'background', 'sh-degree',
             'lpips-net', 'output-dir', 'csv', 'tile-pair-capacity'):
    p.add_argument('--' + flag)
p.add_argument('--metrics', nargs='+')
a = p.parse_args()
assert a.arch == 'cpu' and a.width == '32' and a.height == '16'
assert a.background == 'black'
if a.sh_degree is None:
    assert Path(a.ply).with_suffix('.sh.npz').is_file()
else:
    assert a.sh_degree == '0'
Path('evaluation.called').write_text('called')
frames = json.loads((Path(a.data) / a.manifest).read_text())['frames']
keys = ['view', 'file', 'mse'] + [('psnr_db' if m == 'psnr' else m) for m in a.metrics]
with Path(a.csv).open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for i, frame in enumerate(frames):
        all_metrics = {'psnr_db': 30 + i * 2, 'ssim': 0.8 + i * 0.1, 'lpips': 0.2 - i * 0.1}
        row = {'view': i, 'file': frame['file_path'], 'mse': 0.001}
        row.update({key: all_metrics[key] for key in keys[3:]})
        w.writerow(row)
print('evaluation complete')
'''


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        scratch = runner.ROOT / "outputs"
        scratch.mkdir(exist_ok=True)
        # Ordinary mkdir inherits workspace permissions on restricted Windows hosts.
        self.root = scratch / f"runner-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.run_name = "scene-Oct28-1305-recorded-run"
        self.addCleanup(self.cleanup_fixture)
        (self.root / "config").mkdir()
        data = self.root / "data" / "scene"
        data.mkdir(parents=True)
        pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]]
        for split in ("train", "val", "test"):
            (data / split).mkdir()
            frames = []
            for index in range(2 if split == "test" else 1):
                (data / split / f"r_{index}.png").write_bytes(b"fixture image")
                frames.append({"file_path": f"./{split}/r_{index}", "transform_matrix": pose})
            (data / f"transforms_{split}.json").write_text(
                json.dumps({"camera_angle_x": 0.7, "frames": frames}))
        (data / "init.ply").write_text("fixture initialization")
        self.config = self.root / "config" / "baseline.yaml"
        self.config.write_text(yaml.safe_dump({
            "paths": {"data": "../data/scene", "init_ply": "../data/scene/init.ply",
                      "output": "../existing.png"},
            "runtime": {"arch": "cpu", "seed": 4},
            "model": {"width": 32, "height": 16, "background": "black",
                      "capacity": 4, "initial_splats": 1, "tile_pair_capacity": 64},
            "training": {"iterations": 2}, "reporting": {"eval_views": 1}}))
        (self.root / "existing.png").write_bytes(b"preserve original output")
        (self.root / "3dgs_trainer.py").write_text(TRAINER)
        (self.root / "3dgs_evaluator.py").write_text(EVALUATOR)
        (self.root / "3dgs_gradient_check_gpu.py").write_text("print('gradients passed')")

    def cleanup_fixture(self):
        if not self.root.resolve().is_relative_to((runner.ROOT / "outputs").resolve()):
            raise ValueError("Refusing to clean a fixture outside outputs/")
        shutil.rmtree(self.root)

    def run_fixture(self, *flags):
        args = runner.build_parser().parse_args([
            str(self.config), "--name", "recorded-run", "--metrics", "psnr", "ssim", *flags])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                patch.object(runner, "capture", return_value={"exit_code": None, "error": "fixture"}), \
                patch.object(runner, "datetime", wraps=datetime) as clock:
            clock.now.side_effect = lambda tz: datetime(2026, 10, 28, 5, 5, tzinfo=timezone.utc).astimezone(tz)
            return runner.run_experiment(args, self.root)

    def record(self):
        return json.loads((self.root / "outputs" / self.run_name / "run.json").read_text())

    def test_complete_run_matches_evaluation_and_preserves_source_config(self):
        original = self.config.read_bytes()
        directory = self.run_fixture("--check-gradients", "--note", "test baseline")
        record = self.record()
        self.assertEqual(record["status"], "completed")
        self.assertEqual([stage["name"] for stage in record["stages"]], ["gradients", "train", "eval"])
        self.assertEqual(record["mean_metrics"]["psnr_db"], 31)
        self.assertAlmostEqual(record["mean_metrics"]["ssim"], 0.85)
        self.assertNotIn("lpips", record["mean_metrics"])
        self.assertEqual(record["training_result"]["completed_iterations"], 2)
        self.assertEqual(record["evaluation"]["views"], 2)
        self.assertIn("trainer stderr retained", (directory / "train.log").read_text())
        self.assertEqual(self.config.read_bytes(), original)
        self.assertEqual((self.root / "existing.png").read_bytes(), b"preserve original output")
        resolved = yaml.safe_load((directory / "config.yaml").read_text())
        self.assertEqual(resolved["paths"]["output"], str(directory / "model.png"))
        self.assertTrue(Path(resolved["paths"]["data"]).is_absolute())
        self.assertIn("position_initial", resolved["learning_rates"])
        self.assertTrue((directory / "input-files.json").is_file())
        with zipfile.ZipFile(directory / "source.zip") as archive:
            self.assertIn("3dgs_trainer.py", archive.namelist())
            self.assertEqual(archive.read("config/baseline.yaml"), original)

    def test_prepare_only_does_not_launch_commands(self):
        directory = self.run_fixture("--prepare-only")
        self.assertEqual(self.record()["status"], "prepared")
        self.assertEqual(self.record()["stages"], [])
        self.assertFalse((self.root / "evaluation.called").exists())
        self.assertFalse((directory / "model.ply").exists())
        self.assertIn("--width", self.record()["planned_commands"]["eval"])

    def test_sh_run_uses_sidecar_autodetection_and_records_the_saved_degree(self):
        config = yaml.safe_load(self.config.read_text())
        config["model"]["sh_degree"] = 3
        self.config.write_text(yaml.safe_dump(config))
        directory = self.run_fixture()
        record = self.record()
        self.assertNotIn("--sh-degree", record["planned_commands"]["eval"])
        self.assertEqual(record["training_result"]["sh_degree"], 3)
        self.assertTrue((directory / "model.sh.npz").is_file())

    def test_missing_sh_sidecar_prevents_rgb_fallback(self):
        config = yaml.safe_load(self.config.read_text())
        config["model"]["sh_degree"] = 3
        self.config.write_text(yaml.safe_dump(config))
        (self.root / "3dgs_trainer.py").write_text(
            "from pathlib import Path\nimport sys, yaml\n"
            "config=yaml.safe_load(Path(sys.argv[1]).read_text())\n"
            "Path(config['paths']['output']).with_suffix('.ply').write_text('ply')\n")
        with self.assertRaises(FileNotFoundError):
            self.run_fixture()
        self.assertEqual(self.record()["status"], "failed")
        self.assertFalse((self.root / "evaluation.called").exists())

    def test_gradient_failure_prevents_training_and_evaluation(self):
        (self.root / "3dgs_gradient_check_gpu.py").write_text("import sys; print('bad gradient'); sys.exit(7)")
        with self.assertRaises(runner.StageFailed) as caught:
            self.run_fixture("--check-gradients")
        self.assertEqual(caught.exception.returncode, 7)
        self.assertEqual(self.record()["status"], "failed")
        self.assertEqual([stage["name"] for stage in self.record()["stages"]], ["gradients"])
        self.assertFalse((self.root / "evaluation.called").exists())

    def test_training_failure_retains_log_and_prevents_evaluation(self):
        (self.root / "3dgs_trainer.py").write_text("import sys; print('failure detail'); sys.exit(5)")
        with self.assertRaises(runner.StageFailed):
            self.run_fixture()
        record = self.record()
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stages"][0]["exit_code"], 5)
        self.assertIn("failure detail", (self.root / "outputs" / self.run_name / "train.log").read_text())
        self.assertFalse((self.root / "evaluation.called").exists())

    def test_missing_model_prevents_evaluation_even_with_zero_exit(self):
        (self.root / "3dgs_trainer.py").write_text("print('exited without a model')")
        with self.assertRaises(FileNotFoundError):
            self.run_fixture()
        self.assertEqual(self.record()["status"], "failed")
        self.assertFalse((self.root / "evaluation.called").exists())

    def test_evaluation_failure_preserves_training_artifacts(self):
        (self.root / "3dgs_evaluator.py").write_text("import sys; print('metric failure'); sys.exit(6)")
        with self.assertRaises(runner.StageFailed):
            self.run_fixture()
        self.assertEqual(self.record()["status"], "failed")
        self.assertEqual(self.record()["training_result"]["completed_iterations"], 2)
        self.assertTrue((self.root / "outputs" / self.run_name / "model.ply").is_file())

    def test_existing_run_is_never_overwritten(self):
        directory = self.root / "outputs" / self.run_name
        directory.mkdir(parents=True)
        (directory / "sentinel").write_text("keep me")
        new_directory = self.run_fixture("--prepare-only")
        self.assertEqual(new_directory.name, f"{self.run_name}-2")
        self.assertEqual(json.loads((new_directory / "run.json").read_text())["run_id"], new_directory.name)
        self.assertEqual((directory / "sentinel").read_text(), "keep me")
        self.assertFalse((directory / "run.json").exists())

    def test_missing_image_fails_before_training_with_a_record(self):
        (self.root / "data/scene/test/r_1.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_fixture()
        self.assertEqual(self.record()["status"], "failed")
        self.assertEqual(self.record()["stages"], [])

    def test_invalid_run_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_fixture("--name", "../outside")
        self.assertFalse((self.root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
