"""Geometry-loss gradient, masking, scheduling, data isolation and integration checks."""

from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import numpy as np
from PIL import Image
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from cp4282_gs.surfels.config import SurfelTrainingConfig, GEOMETRY_DEFAULTS
from cp4282_gs.surfels.geometry_losses import GeometryLoss, load_training_alpha
from cp4282_gs.surfels.checks import check_gradients, make_trainer
from cp4282_gs.surfels.kernels import GBuffer
from cp4282_gs.surfels.workflow import run_training
from cp4282_gs.surfels.trainer import SurfelTrainer
from cp4282_gs.surfels.checks import reference

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


class GeometryLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def setUp(self):
        self.directory = ROOT / 'outputs' / f'geometry-test-{uuid.uuid4().hex}'
        self.directory.mkdir()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        assert self.directory.resolve().is_relative_to((ROOT / 'outputs').resolve())
        shutil.rmtree(self.directory)

    def plane(self, rotation=None, **options):
        width = 6
        x, y = np.meshgrid((np.arange(width) + 0.5 - width / 2) / width,
                           (np.arange(width) + 0.5 - width / 2) / width)
        depth = 1.2 / (1.0 - 0.15 * x + 0.1 * y)
        normal = np.array([0.15, -0.1, -1.0]); normal /= np.linalg.norm(normal)
        matrix = np.eye(4, dtype=np.float32)
        if rotation is not None:
            matrix[:3, :3] = rotation
            normal = rotation.T @ normal
        values = np.zeros((width * width, 9), np.float32)
        values[:, 3:6] = 0.8 * normal
        values[:, 6] = 0.8 * depth.reshape(-1)
        values[:, 7] = 0.8
        buffers = wp.array(values, dtype=GBuffer, device=DEVICE, requires_grad=True)
        cameras = wp.array(matrix[None], dtype=wp.mat44, device=DEVICE)
        settings = dict(GEOMETRY_DEFAULTS, normal_start_iteration=0, normal_ramp_iterations=0,
                        mask_weight=0.1, normal_depth_weight=0.03)
        settings.update(options)
        loss = GeometryLoss(width, width, DEVICE, settings, np.ones((1, width, width), np.float32))
        return loss, buffers, cameras

    def test_consistent_perspective_plane_and_world_camera_rotation(self):
        for rotation in (None, np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])):
            loss, buffers, cameras = self.plane(rotation=rotation)
            result = loss.compute(buffers, cameras, 0, 6., 0)
            self.assertEqual(result['normal_pixels'], 16)
            self.assertLess(abs(result['normal_depth']), 1e-6)
            self.assertAlmostEqual(result['mask'], -np.log(0.8), places=5)

    def test_gbuffer_normal_depth_and_coverage_gradients(self):
        loss, buffers, cameras = self.plane(normal_depth_weight=0.07)
        original = buffers.numpy()
        original[14, 3] += 0.2
        original[15, 6] += 0.005
        buffers.assign(original)
        result = loss.compute(buffers, cameras, 0, 6., 0)
        self.assertGreater(result['normal_depth'], 0)
        analytic = buffers.grad.numpy().copy()
        for pixel, channel in [(14, 3), (15, 6), (15, 7)]:
            epsilon = 0.0002
            plus, minus = original.copy(), original.copy()
            plus[pixel, channel] += epsilon; minus[pixel, channel] -= epsilon
            buffers.assign(plus)
            high = loss.compute(buffers, cameras, 0, 6., 0, backward=False)['weighted']
            buffers.assign(minus)
            low = loss.compute(buffers, cameras, 0, 6., 0, backward=False)['weighted']
            numeric = (high - low) / (2 * epsilon)
            self.assertAlmostEqual(float(analytic[pixel, channel]), numeric, delta=max(0.0002, abs(numeric) * 0.02))

    def test_zero_coverage_and_empty_foreground_are_finite(self):
        loss, buffers, cameras = self.plane()
        buffers.zero_()
        result = loss.compute(buffers, cameras, 0, 6., 0)
        self.assertEqual(result['normal_pixels'], 0)
        self.assertEqual(result['normal_depth'], 0)
        self.assertTrue(np.isfinite(result['weighted']))
        self.assertTrue(np.isfinite(buffers.grad.numpy()).all())
        # A known background pixel penalizes increasing coverage, even if RGB is white.
        loss.alpha.zero_()
        buffers.grad.zero_()
        result = loss.compute(buffers, cameras, 0, 6., 0)
        self.assertTrue(np.all(buffers.grad.numpy()[:, 7] > 0))

    def test_foreground_holes_depth_discontinuities_and_low_coverage_are_excluded(self):
        for channel, value in ((7, 0.01), (6, 5.0)):
            loss, buffers, cameras = self.plane()
            values = buffers.numpy(); values[14, channel] = value; buffers.assign(values)
            result = loss.compute(buffers, cameras, 0, 6., 0)
            self.assertLess(result['normal_pixels'], 16)
            self.assertEqual(loss.valid.numpy()[14], 0)
        loss, buffers, cameras = self.plane()
        alpha = loss.alpha.numpy(); alpha[14] = 0; loss.alpha.assign(alpha)
        loss.compute(buffers, cameras, 0, 6., 0)
        self.assertEqual(loss.valid.numpy()[14], 0)

    def test_normal_schedule_and_held_out_gradient_guard(self):
        loss, buffers, cameras = self.plane(normal_start_iteration=2000, normal_ramp_iterations=2000)
        self.assertEqual(loss.normal_weight(1999), 0)
        self.assertEqual(loss.normal_weight(2000), 0)
        self.assertAlmostEqual(loss.normal_weight(3000), 0.015)
        self.assertAlmostEqual(loss.normal_weight(4000), 0.03)
        with self.assertRaises(ValueError):
            loss.compute(buffers, cameras, 1, 6., 4000)

    def test_geometry_and_photometric_gradients_combine_through_splats(self):
        check_gradients(DEVICE, 'rgb', overlapping=True, geometry=True, verbose=False)
        check_gradients(DEVICE, 'deferred', blended=True, overlapping=True, geometry=True, verbose=False)

    def test_mask_loss_removes_white_floater_that_rgb_cannot_penalize(self):
        scene = reference.one_splat_scene()
        trainer = SurfelTrainer(np.ones((1, 4, 4, 3), np.float32), np.eye(4, dtype=np.float32)[None],
            4., 1., 1, 1, DEVICE, 7, init_scene=scene, training_view_count=1,
            training_alpha=np.zeros((1, 4, 4), np.float32), geometry_options={'mask_weight': 0.1},
            training_options={'compact_box': {'enabled': False}}, surfel_options={'save_debug': False})
        trainer.color.assign(np.ones((1, 3), np.float32))
        image, _ = trainer.render_single_view_image(0)
        np.testing.assert_allclose(image, 1., atol=1e-6)
        before = float(trainer.buffers.numpy()[:, 7].mean())
        opacity = float(trainer.opacity.numpy()[0])
        for iteration in range(1, 6):
            trainer.step([0], iteration=iteration)
        trainer.render_single_view_image(0)
        self.assertLess(float(trainer.opacity.numpy()[0]), opacity)
        self.assertLess(float(trainer.buffers.numpy()[:, 7].mean()), before)

    def test_normal_loss_supplies_orientation_gradient_without_colour_error(self):
        trainer = make_trainer(DEVICE, geometry=True)
        trainer.geometry_options['mask_weight'] = 0.0
        trainer.color.assign(np.ones((1, 3), np.float32))
        trainer.targets.assign(np.ones((16, 3), np.float32))
        trainer.loss_and_gradients([0])
        self.assertLess(trainer.last_photometric_loss, 1e-12)
        self.assertGreater(trainer.last_geometry_terms['normal_depth'], 0)
        self.assertGreater(np.linalg.norm(trainer.optimizers.quaternion_grad_flat.numpy()), 1e-5)

    def test_training_masks_are_read_from_alpha_and_missing_alpha_is_rejected(self):
        (self.directory / 'train').mkdir()
        manifest = {'frames': [{'file_path': './train/r_0'}]}
        (self.directory / 'transforms_train.json').write_text(json.dumps(manifest))
        rgba = np.full((6, 6, 4), 255, np.uint8); rgba[:, :3, 3] = 0
        Image.fromarray(rgba).save(self.directory / 'train/r_0.png')
        mask = load_training_alpha(self.directory, 6, 6)
        np.testing.assert_array_equal(mask[0], rgba[..., 3] / 255)
        # No validation/test manifests exist: mask loading is training-only.
        Image.fromarray(rgba[..., :3]).save(self.directory / 'train/r_0.png')
        with self.assertRaisesRegex(ValueError, 'source alpha'):
            load_training_alpha(self.directory, 6, 6)

    def test_workflow_logs_losses_and_keeps_validation_objective_photometric(self):
        trainer = make_trainer(DEVICE, geometry=True, overlapping=True)
        config = SurfelTrainingConfig(SurfelTrainingConfig._merge(SurfelTrainingConfig.DEFAULTS, {
            'paths': {'output': str(self.directory / 'model.png')},
            'model': {'width': 4, 'height': 4, 'capacity': 2, 'initial_splats': 2},
            'training': {'iterations': 3}, 'densification': {'interval': 0},
            'multiview_adc': {'enabled': False}, 'convergence': {'enabled': False},
            'geometry_loss': trainer.geometry_options,
            'reporting': {'eval_every': 1, 'snapshot_every': 3, 'log_every': 1},
        }))
        config.validate()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            run_training(trainer, config, 1, [0])
        self.assertIn('normal_pixels=', log.getvalue())
        photo, _, _ = trainer.evaluate_components([0])
        training = trainer.loss_and_gradients([0])
        self.assertGreater(training, photo)

    def test_invalid_loss_configuration(self):
        for key, value in [('mask_weight', -1), ('normal_depth_weight', float('nan')),
                           ('normal_start_iteration', -1), ('normal_ramp_iterations', True),
                           ('min_coverage', 0), ('foreground_threshold', 2)]:
            config = deepcopy(SurfelTrainingConfig.DEFAULTS)
            config['geometry_loss'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                SurfelTrainingConfig(config).validate()


if __name__ == '__main__':
    unittest.main()
