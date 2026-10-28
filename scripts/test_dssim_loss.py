"""Check reflected D-SSIM adjoints, renderer integration, and the loss curriculum."""

import contextlib
from copy import deepcopy
import importlib
import io
import os
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "shared")]
from dssim_loss import DSSIMComputer
from training_config import TrainingConfig

trainer_module = importlib.import_module("3dgs_trainer")
checks = importlib.import_module("3dgs_gradient_check_gpu")
DEVICE = os.environ.get("DSSIM_TEST_DEVICE", "cpu")


def reference_dssim(image, target, window):
    """Independent float64 SSIM reference, including repeated reflected border samples."""
    def box(values):
        radius = window // 2
        padded = np.pad(values, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
        patches = np.lib.stride_tricks.sliding_window_view(padded, (window, window), axis=(0, 1))
        return patches.mean(axis=(-2, -1))

    x, y = np.asarray(image, np.float64), np.asarray(target, np.float64)
    mx, my = box(x), box(y)
    vx = np.maximum(box(x * x) - mx * mx, 0.0)
    vy = np.maximum(box(y * y) - my * my, 0.0)
    cov = box(x * y) - mx * my
    ssim = ((2 * mx * my + 0.0001) * (2 * cov + 0.0009)
            / ((mx * mx + my * my + 0.0001) * (vx + vy + 0.0009)))
    return float(np.mean((1.0 - ssim) / 2.0))


class DSSIMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / "outputs" / "dssim-test-warp-cache")
        wp.config.use_precompiled_headers = False
        wp.init()

    def compute(self, images, targets, view_ids, window, weight=0.2, **kwargs):
        batches, height, width, _ = images.shape
        computer = DSSIMComputer(width, height, batches, DEVICE)
        image = wp.array(images.reshape(-1, 3), dtype=wp.vec3, device=DEVICE)
        target = wp.array(targets.reshape(-1, 3), dtype=wp.vec3, device=DEVICE)
        ids = wp.array(view_ids, dtype=wp.int32, device=DEVICE)
        mse_gradient = 2.0 * (images - targets[view_ids]) / images.size
        gradient = wp.array(mse_gradient.reshape(-1, 3), dtype=wp.vec3, device=DEVICE)
        value = computer.compute(image, target, ids, gradient, window, batches, weight, **kwargs)
        return value, gradient.numpy().reshape(images.shape)

    def test_loss_and_blended_gradient_match_reference_at_borders_and_across_views(self):
        rng = np.random.default_rng(11)
        for height, width, window in ((5, 6, 3), (3, 4, 11), (1, 4, 11), (1, 1, 1)):
            with self.subTest(height=height, width=width, window=window):
                images = rng.uniform(0.1, 0.9, (2, height, width, 3)).astype(np.float32)
                targets = rng.uniform(0.1, 0.9, (3, height, width, 3)).astype(np.float32)
                ids = np.array([2, 0], np.int32)
                value, gradient = self.compute(images, targets, ids, window)
                expected = np.mean([reference_dssim(x, y, window) for x, y in zip(images, targets[ids])])
                self.assertAlmostEqual(value, expected, delta=3e-5)

                def objective(values):
                    mse = np.mean((values - targets[ids]) ** 2)
                    dssim = np.mean([reference_dssim(x, y, window) for x, y in zip(values, targets[ids])])
                    return 0.8 * mse + 0.2 * dssim

                locations = ((0, 0, 0, 0), (0, height - 1, width - 1, 1),
                             (1, 0, width - 1, 2), (1, height // 2, width // 2, 0))
                for location in locations:
                    plus, minus = images.astype(np.float64), images.astype(np.float64)
                    plus[location] += 1e-5
                    minus[location] -= 1e-5
                    numeric = (objective(plus) - objective(minus)) / 2e-5
                    self.assertAlmostEqual(float(gradient[location]), numeric, delta=3e-5)

    def test_flat_identical_and_anticorrelated_images_stay_finite(self):
        rng = np.random.default_rng(12)
        for image, target in ((np.full((1, 5, 6, 3), 0.4, np.float32),) * 2,
                              (rng.random((1, 5, 6, 3), dtype=np.float32),) * 2):
            value, gradient = self.compute(image, target, np.array([0], np.int32), 11)
            self.assertAlmostEqual(value, 0.0, delta=1e-4)
            np.testing.assert_allclose(gradient, 0.0, atol=3e-5)
        image = rng.random((1, 5, 6, 3), dtype=np.float32)
        value, gradient = self.compute(image, 1 - image, np.array([0], np.int32), 3, weight=1.0)
        self.assertTrue(np.isfinite(value) and np.all(np.isfinite(gradient)))

    def test_no_download_still_computes_gradients_and_evaluation_preserves_them(self):
        rng = np.random.default_rng(13)
        image = rng.random((1, 4, 5, 3), dtype=np.float32)
        target = rng.random((1, 4, 5, 3), dtype=np.float32)
        ids = np.array([0], np.int32)
        _, expected = self.compute(image, target, ids, 3)
        value, actual = self.compute(image, target, ids, 3, download_loss=False)
        self.assertIsNone(value)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        _, untouched = self.compute(image, target, ids, 3, compute_gradient=False)
        np.testing.assert_allclose(untouched, 2 * (image - target) / image.size, atol=1e-7)
        _, weight_zero = self.compute(image, target, ids, 3, weight=0.0)
        np.testing.assert_allclose(weight_zero, untouched, atol=1e-7)

    def test_blended_renderer_parameter_gradients_and_dense_stage(self):
        trainer = checks.make_trainer(8, np.array([0.2, 0.3, 0.4]), (1.0, 1.0, 1.0), DEVICE)
        trainer.loss_config.update(mode="mse_dssim", dssim_window=3)
        trainer.sparse_enabled = True  # Stage 1 must still force the dense renderer.
        trainer.advance_dssim_stage()
        trainer.loss_and_gradients(np.array([0], np.int32))
        self.assertEqual(trainer.last_loss_phase, "dssim")
        analytic = (float(trainer.optimizers.mean_grad.numpy()[0, 0]),
                    float(trainer.optimizers.scale_grad.numpy()[0, 1]),
                    float(trainer.optimizers.opacity_grad.numpy()[0]),
                    float(trainer.optimizers.color_grad.numpy()[0, 1]))
        for expected, field, dtype, index in zip(analytic,
                (trainer.means, trainer.log_scales, trainer.opacity, trainer.color),
                (wp.vec3, wp.vec3, wp.float32, wp.vec3),
                ((0, 0), (0, 1), 0, (0, 1))):
            numeric = checks.finite_difference(trainer, field, dtype, index)
            self.assertAlmostEqual(expected, numeric, delta=max(2e-4, abs(numeric) * 0.015))
        objective, mse, dssim = trainer.evaluate_components([0])
        self.assertAlmostEqual(objective, 0.8 * mse + 0.2 * dssim, places=6)
        self.assertAlmostEqual(trainer.render_single_view_image(0)[1], mse, places=6)

    def test_curriculum_gate_and_unreachable_gate_do_not_stop_warmup(self):
        for threshold, iterations, eval_every, expected_stage in ((0.0, 4, 1, "dssim"),
                                                                  (100.0, 4, 1, "mse"),
                                                                  (0.0, 2, 3, "mse")):
            with self.subTest(threshold=threshold, iterations=iterations, eval_every=eval_every):
                directory = ROOT / "outputs" / f"dssim-test-{uuid.uuid4().hex}"
                directory.mkdir()
                try:
                    trainer = checks.make_trainer(8, np.array([0.2, 0.3, 0.4]), (1.0, 1.0, 1.0), DEVICE)
                    trainer.loss_config.update(mode="mse_dssim", dssim_window=3)
                    values = TrainingConfig._merge(TrainingConfig.DEFAULTS, {
                        "model": {"width": 8, "height": 8, "capacity": 1, "initial_splats": 1},
                        "training": {"iterations": iterations},
                        "loss": {"mode": "mse_dssim", "dssim_window": 3},
                        "densification": {"interval": 0}, "multiview_adc": {"enabled": False},
                        "adaptive": {"dssim_psnr_threshold": threshold},
                        "reporting": {"eval_every": eval_every, "snapshot_every": 1,
                                      "log_every": 1, "save_ply": False, "loss_ema_decay": 0.0},
                        "convergence": {"min_iterations": 1, "patience": 1, "min_delta": 1.0},
                    })
                    config = TrainingConfig(values)
                    config.validate()
                    log = io.StringIO()
                    with contextlib.redirect_stdout(log):
                        trainer_module.run_training(trainer, config, trainer.targets_host, [0],
                                                    directory, directory / "model.png")
                    self.assertEqual(trainer.loss_stage, expected_stage)
                    if expected_stage == "mse":
                        self.assertIn(f"Training finished after {iterations:,} iterations", log.getvalue())
                        self.assertIn("D-SSIM was not activated", log.getvalue())
                    else:
                        self.assertIn("advancing loss stage 'mse' -> 'dssim'", log.getvalue())
                        self.assertIn("loss_phase=dssim", log.getvalue())
                    self.assertTrue((directory / "model.png").is_file())
                finally:
                    if not directory.resolve().is_relative_to((ROOT / "outputs").resolve()):
                        raise ValueError("Refusing to clean a fixture outside outputs/")
                    shutil.rmtree(directory)

    def test_loss_configuration_validation_and_default_mse(self):
        self.assertEqual(TrainingConfig.DEFAULTS["loss"]["mode"], "mse")
        for section, key, value in (("loss", "mode", "unknown"), ("loss", "dssim_weight", float("nan")),
                                   ("loss", "dssim_weight", 1.1), ("loss", "dssim_window", 4),
                                   ("loss", "dssim_window", 33), ("loss", "dssim_window", True),
                                   ("adaptive", "dssim_psnr_threshold", float("inf"))):
            values = deepcopy(TrainingConfig.DEFAULTS)
            values[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                TrainingConfig(values).validate()


if __name__ == "__main__":
    unittest.main()
