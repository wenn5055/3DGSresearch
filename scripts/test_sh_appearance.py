"""SH basis, full renderer adjoints, density control, staged training, and checkpoint tests."""

import argparse
import contextlib
from copy import deepcopy
import importlib
import io
import math
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
from gaussian_set import GaussianSet
from sh_appearance import SHAppearance, SH_BASIS, load_sh_sidecar
from training_config import TrainingConfig

trainer_module = importlib.import_module("3dgs_trainer")
evaluator = importlib.import_module("3dgs_evaluator")
checks = importlib.import_module("3dgs_gradient_check_gpu")
DEVICE = os.environ.get("SH_TEST_DEVICE", "cpu")


def reference_basis(direction, degree):
    """Independent real-SH reference from associated Legendre recurrence, without phase signs."""
    x, y, z = np.asarray(direction, np.float64)
    z = float(np.clip(z, -1, 1))
    phi = math.atan2(y, x)
    values = []
    for band in range(1, degree + 1):
        for order in range(-band, band + 1):
            m = abs(order)
            diagonal = math.prod(range(1, 2 * m, 2)) * max(0.0, 1 - z * z) ** (m / 2)
            if band == m:
                polynomial = diagonal
            else:
                previous, polynomial = diagonal, (2 * m + 1) * z * diagonal
                for l in range(m + 2, band + 1):
                    previous, polynomial = polynomial, ((2 * l - 1) * z * polynomial - (l + m - 1) * previous) / (l - m)
            scale = math.sqrt((2 * band + 1) / (4 * math.pi) * math.factorial(band - m) / math.factorial(band + m))
            angular = 1.0
            if order:
                angular = math.sqrt(2) * (math.sin(m * phi) if order < 0 else math.cos(m * phi))
            values.append(scale * polynomial * angular)
    return np.asarray(values)


class SHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / "outputs" / "sh-test-warp-cache")
        wp.config.use_precompiled_headers = False
        wp.init()

    def setUp(self):
        self.directory = ROOT / "outputs" / f"sh-test-{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup_fixture)

    def cleanup_fixture(self):
        if not self.directory.resolve().is_relative_to((ROOT / "outputs").resolve()):
            raise ValueError("Refusing to clean a fixture outside outputs/")
        shutil.rmtree(self.directory)

    def trainer(self, degree=3, sparse=False, capacity=1, blended=False):
        targets = np.broadcast_to(np.array([0.2, 0.3, 0.4], np.float32), (1, 16, 16, 3)).copy()
        cameras = np.eye(4, dtype=np.float32)[None]
        trainer = trainer_module.WarpImageTrainer(
            targets, cameras, 16.0, 1.0, capacity, 1, DEVICE, 7,
            init_scene=checks.one_splat_scene(), sh_degree=degree,
            training_options={"sparse": {"enabled": sparse, "samples_per_tile": 256},
                              "loss": {"mode": "mse_dssim" if blended else "mse", "dssim_window": 3}},
        )
        if degree:
            trainer.sh.degree = degree
            values = np.random.default_rng(41).normal(0, 0.03, (capacity * trainer.sh.rest_count, 3)).astype(np.float32)
            trainer.sh.coefficients.assign(values)
        if blended:
            trainer.advance_dssim_stage()
        return trainer

    def test_basis_matches_independent_legendre_reference_and_camera_convention(self):
        appearance = SHAppearance(1, 3, 0.000125, DEVICE)
        active = wp.array([1], dtype=wp.int32, device=DEVICE)
        rgb = wp.array([[0.4, 0.5, 0.6]], dtype=wp.vec3, device=DEVICE)
        angle = 0.6
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                             [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], np.float32)
        centre = np.array([2.0, -1.0, 3.0], np.float32)
        camera = np.eye(4, dtype=np.float32)
        camera[:3, :3] = rotation
        camera[:3, 3] = -rotation @ centre
        cameras = wp.array(camera[None], dtype=wp.mat44, device=DEVICE)
        for direction in ((0.2, -0.4, 0.8), (-0.3, 0.8, -0.4), (0.8, 0.1, 0.3)):
            d = np.asarray(direction, np.float64)
            d /= np.linalg.norm(d)
            means = wp.array([centre - d], dtype=wp.vec3, device=DEVICE)
            reference = reference_basis(d, 3)
            for degree in (1, 2, 3):
                appearance.degree = degree
                for index in range(15):
                    values = np.zeros((15, 3), np.float32)
                    values[index] = 0.1
                    appearance.coefficients.assign(values)
                    colour = appearance.forward(means, rgb, active, cameras, 0).numpy()[0]
                    expected = np.array([0.4, 0.5, 0.6]) + (0.1 * reference[index] if index < appearance.live_count else 0)
                    np.testing.assert_allclose(colour, expected, atol=2e-6)

    def test_coefficients_and_geometry_gradients_for_all_degrees_and_losses(self):
        for degree, blended in ((1, False), (2, False), (3, False), (3, True)):
            with self.subTest(degree=degree, blended=blended):
                trainer = self.trainer(degree, blended=blended)
                trainer.loss_and_gradients(np.array([0], np.int32))
                cases = [(trainer.means, wp.vec3, (0, axis), float(trainer.optimizers.mean_grad.numpy()[0, axis]))
                         for axis in range(3)]
                cases += [(trainer.color, wp.vec3, (0, 1), float(trainer.optimizers.color_grad.numpy()[0, 1]))]
                for index in (0, trainer.sh.rest_count - 1):
                    cases.append((trainer.sh.coefficients, wp.vec3, (index, 1), float(trainer.sh.gradient.numpy()[index, 1])))
                for field, dtype, index, analytic in cases:
                    numeric = checks.finite_difference(trainer, field, dtype, index)
                    self.assertAlmostEqual(analytic, numeric, delta=max(3e-4, abs(numeric) * 0.02))

    def test_appearance_only_position_adjoint_is_tangential(self):
        trainer = self.trainer(3)
        coefficients = trainer.sh.coefficients.numpy()
        mean = trainer.means.numpy()[0].astype(np.float64)
        g = np.array([0.2, -0.4, 0.7], np.float32)
        colour_gradient = wp.array(g, dtype=wp.float32, device=DEVICE)
        mean_gradient = wp.zeros(3, dtype=wp.float32, device=DEVICE)
        trainer.sh.backward(trainer.means, trainer.active_device, trainer.cameras, 0,
                            colour_gradient, mean_gradient)
        actual = mean_gradient.numpy()

        def objective(position):
            direction = -position / np.linalg.norm(position)
            return float(np.dot(reference_basis(direction, 3) @ coefficients, g))

        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = 1e-5
            numeric = (objective(mean + delta) - objective(mean - delta)) / 2e-5
            self.assertAlmostEqual(float(actual[axis]), numeric, delta=2e-5)
        self.assertAlmostEqual(float(np.dot(actual, mean)), 0.0, delta=1e-6)
        expected = reference_basis(-mean / np.linalg.norm(mean), 3)[:, None] * g[None]
        np.testing.assert_allclose(trainer.sh.gradient.numpy(), expected, atol=2e-6)

    def test_sparse_dense_full_coverage_agree_with_sh3(self):
        trainer = self.trainer(3)
        dense = trainer.dense_loss_and_gradients([0])
        expected = [trainer.optimizers.mean_grad.numpy(), trainer.optimizers.scale_grad.numpy(),
                    trainer.optimizers.color_grad.numpy(), trainer.sh.gradient.numpy()]
        # Allocate sparse buffers at construction, preserving the exact seed and coefficients.
        sparse_trainer = self.trainer(3, sparse=True)
        sparse = sparse_trainer.sparse_loss_and_gradients([0])
        actual = [sparse_trainer.optimizers.mean_grad.numpy(), sparse_trainer.optimizers.scale_grad.numpy(),
                  sparse_trainer.optimizers.color_grad.numpy(), sparse_trainer.sh.gradient.numpy()]
        self.assertAlmostEqual(dense, sparse, delta=2e-6)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=3e-4, atol=3e-6)

    def test_locked_bands_and_clamped_colour_have_zero_gradients(self):
        trainer = self.trainer(3)
        trainer.sh.degree = 1
        trainer.loss_and_gradients([0])
        np.testing.assert_array_equal(trainer.sh.gradient.numpy()[3:], 0)
        before = trainer.sh.coefficients.numpy().copy()
        trainer.sh.step(trainer.active_device)
        np.testing.assert_array_equal(trainer.sh.coefficients.numpy()[3:], before[3:])
        np.testing.assert_array_equal(trainer.sh.moment.numpy()[3:], 0)
        self.assertEqual(trainer.sh.steps, [1, 0, 0])
        trainer.advance_sh_stage()
        trainer.loss_and_gradients([0])
        trainer.sh.step(trainer.active_device)
        self.assertEqual(trainer.sh.steps, [2, 1, 0])
        self.assertTrue(np.any(trainer.sh.moment.numpy()[3:8]))
        np.testing.assert_array_equal(trainer.sh.moment.numpy()[8:], 0)
        trainer.sh.degree = 3
        values = np.zeros((15, 3), np.float32)
        values[1, 1] = -10.0  # z-direction is negative, so green saturates above 1.
        trainer.sh.coefficients.assign(values)
        trainer.loss_and_gradients([0])
        np.testing.assert_array_equal(trainer.sh.gradient.numpy()[:, 1], 0)
        self.assertEqual(float(trainer.optimizers.color_grad.numpy()[0, 1]), 0.0)

    def test_degree_zero_matches_rgb_even_with_unused_coefficients(self):
        rgb = self.trainer(0)
        sh = self.trainer(3)
        sh.sh.degree = 0
        self.assertAlmostEqual(rgb.loss_and_gradients([0]), sh.loss_and_gradients([0]), delta=2e-6)
        np.testing.assert_array_equal(rgb.image.numpy(), sh.image.numpy())
        # CUDA atomic accumulation can differ by a few float32 ulps across identical launches.
        np.testing.assert_allclose(rgb.optimizers.mean_grad.numpy(), sh.optimizers.mean_grad.numpy(), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(rgb.optimizers.color_grad.numpy(), sh.optimizers.color_grad.numpy(), rtol=2e-6, atol=1e-7)

    def test_density_control_copies_coefficients_and_saves_active_rows(self):
        trainer = self.trainer(3, capacity=4)
        original = trainer.sh.coefficients.numpy().reshape(4, 15, 3).copy()
        trainer.sh.steps = [3, 2, 1]
        trainer.sh.moment.assign(np.ones((60, 3), np.float32))
        children = trainer.densify_and_prune(np.ones((4, 3)), 1.0)
        np.testing.assert_array_equal(children, [1])
        np.testing.assert_array_equal(trainer.sh.coefficients.numpy().reshape(4, 15, 3)[1], original[0])
        np.testing.assert_array_equal(trainer.sh.moment.numpy(), 1)
        self.assertEqual(trainer.sh.steps, [3, 2, 1])
        logits = trainer.opacity.numpy()
        logits[0] = -12
        trainer.opacity.assign(logits)
        trainer.densify_and_prune(np.zeros((4, 3)), 0.0)
        path = self.directory / "model.ply"
        trainer.gaussian_set().to_ply(path)
        sidecar = trainer.save_sidecars(path)[0]
        values, maximum, degree = load_sh_sidecar(sidecar, 1)
        self.assertEqual((maximum, degree), (3, 3))
        np.testing.assert_array_equal(values[0], original[0])

    def test_checkpoint_roundtrip_and_evaluator_degree_selection(self):
        trainer = self.trainer(3)
        path = self.directory / "model.ply"
        trainer.gaussian_set().to_ply(path)
        scene = GaussianSet.from_ply(path)
        args = argparse.Namespace(ply=path, background="white", sh_npz=None,
                                  sh_degree=None, tile_pair_capacity=None)
        for degree in (0, 1, 2, 3):
            trainer.sh.degree = degree
            trainer.save_sidecars(path)
            loaded, actual_degree = evaluator.build_trainer(args, scene, trainer.targets_host,
                trainer.cameras.numpy(), trainer.focal, 1 / 0.35, DEVICE)
            self.assertEqual(actual_degree, degree)
            np.testing.assert_allclose(loaded.render_single_view_image(0)[0],
                                       trainer.render_single_view_image(0)[0], atol=2e-6)
        args.sh_degree = 2
        loaded, degree = evaluator.build_trainer(args, scene, trainer.targets_host,
            trainer.cameras.numpy(), trainer.focal, 1 / 0.35, DEVICE)
        self.assertEqual(degree, 2)
        trainer.sh.degree = 2
        np.testing.assert_allclose(loaded.render_single_view_image(0)[0], trainer.render_single_view_image(0)[0], atol=2e-6)
        trainer.save_sidecars(path)
        args.sh_degree = 3
        with self.assertRaises(ValueError):
            evaluator.build_trainer(args, scene, trainer.targets_host, trainer.cameras.numpy(), trainer.focal, 1, DEVICE)
        args.sh_degree = 0
        path.with_suffix('.sh.npz').write_bytes(b'invalid ignored sidecar')
        _, degree = evaluator.build_trainer(args, scene, trainer.targets_host, trainer.cameras.numpy(), trainer.focal, 1, DEVICE)
        self.assertEqual(degree, 0)
        path.with_suffix('.sh.npz').unlink()
        args.sh_degree = 1
        with self.assertRaises(FileNotFoundError):
            evaluator.build_trainer(args, scene, trainer.targets_host, trainer.cameras.numpy(), trainer.focal, 1, DEVICE)

    def test_sidecar_validation_and_legacy_degree2(self):
        path = self.directory / 'model.sh.npz'
        np.savez(path, sh_rest=np.zeros((1, 8, 3), np.float32))
        _, maximum, degree = load_sh_sidecar(path, 1)
        self.assertEqual((maximum, degree), (2, 2))
        for values, extra in ((np.zeros((2, 8, 3)), {}), (np.zeros((1, 7, 3)), {}),
                              (np.full((1, 15, 3), np.nan), {}),
                              (np.zeros((1, 15, 3)), {'sh_degree': 4}),
                              (np.zeros((1, 15, 3)), {'basis': 'unknown'}),
                              (np.zeros((1, 15, 3)), {'direction': 'camera_to_splat'})):
            np.savez(path, sh_rest=values, **extra)
            with self.assertRaises(ValueError):
                load_sh_sidecar(path, 1)

    def test_loss_then_sh_curriculum_and_unreached_degree(self):
        for reachable in (True, False):
            trainer = self.trainer(3)
            trainer.sh.coefficients.zero_()
            trainer.sh.degree = 0
            trainer.loss_config.update(mode='mse_dssim', dssim_window=3)
            values = TrainingConfig._merge(TrainingConfig.DEFAULTS, {
                'model': {'width': 16, 'height': 16, 'capacity': 1, 'initial_splats': 1, 'sh_degree': 3},
                'loss': {'mode': 'mse_dssim', 'dssim_window': 3},
                'training': {'iterations': 6}, 'densification': {'interval': 0},
                'multiview_adc': {'enabled': False},
                'adaptive': {'dssim_psnr_threshold': 0.0, 'sh1_psnr_threshold': 0.1 if reachable else 90.,
                             'sh2_psnr_threshold': 0.2 if reachable else 91., 'sh3_psnr_threshold': 0.3 if reachable else 92.},
                'convergence': {'min_iterations': 1, 'patience': 1, 'min_delta': 1.0},
                'reporting': {'eval_every': 1, 'log_every': 1, 'snapshot_every': 100,
                              'save_ply': False, 'loss_ema_decay': 0.0},
            })
            config = TrainingConfig(values)
            config.validate()
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                trainer_module.run_training(trainer, config, trainer.targets_host, [0],
                                            self.directory, self.directory / 'model.png')
            if reachable:
                self.assertEqual(trainer.active_sh_degree, 3)
                for degree in (1, 2, 3):
                    self.assertIn(f'{degree + 1:6d}: held-out MSE EMA', log.getvalue())
                self.assertTrue(np.any(trainer.sh.coefficients.numpy()[8:]))
            else:
                self.assertEqual(trainer.active_sh_degree, 0)
                self.assertIn('Training finished after 6 iterations', log.getvalue())
                self.assertIn('SH gates were not all reached', log.getvalue())

    def test_configuration_degrees_and_gate_ordering(self):
        self.assertEqual(TrainingConfig.DEFAULTS['model']['sh_degree'], 0)
        for degree in (0, 1, 2, 3):
            values = deepcopy(TrainingConfig.DEFAULTS)
            values['model']['sh_degree'] = degree
            TrainingConfig(values).validate()
        for section, key, value in (('model', 'sh_degree', 4), ('model', 'sh_degree', True),
                                   ('learning_rates', 'feature_rest', -0.1),
                                   ('adaptive', 'sh1_psnr_threshold', 24.),
                                   ('adaptive', 'sh2_psnr_threshold', 26.)):
            values = deepcopy(TrainingConfig.DEFAULTS)
            values['model']['sh_degree'] = 3
            values['loss']['mode'] = 'mse_dssim'
            values[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                TrainingConfig(values).validate()

    def test_density_inherits_sh_moments_without_resetting_band_clocks(self):
        trainer = self.trainer(2, capacity=4)
        trainer.training_options['multiview_adc']['enabled'] = False
        trainer.step([0])
        appearance = trainer.sh
        steps = appearance.steps.copy()
        moment = appearance.moment.numpy().reshape(4, appearance.rest_count, 3).copy()
        variance = appearance.variance.numpy().reshape(4, appearance.rest_count, 3).copy()
        children = trainer.densify_and_prune(np.ones((4, 3)), 1.)
        self.assertEqual(children.tolist(), [1])
        self.assertEqual(appearance.steps, steps)
        for current, old in ((appearance.moment, moment), (appearance.variance, variance)):
            rows = current.numpy().reshape(4, appearance.rest_count, 3)
            np.testing.assert_array_equal(rows[0], old[0])
            np.testing.assert_array_equal(rows[1], old[0])


if __name__ == '__main__':
    unittest.main()
