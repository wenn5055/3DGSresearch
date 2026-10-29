"""Geometry, deferred-shading adjoints, initialization isolation, and checkpoint regressions."""

import argparse
import contextlib
from copy import deepcopy
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
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from gaussian_set import GaussianSet
from splat_math import quaternion_to_matrix
from cp4282_gs.surfels.checks import make_trainer, check_gradients
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels.trainer import GaussianRGBControl, SurfelTrainer, read_surfel_sidecar
from cp4282_gs.surfels.workflow import run_training
from cp4282_gs.surfels import kernels as k
import evaluate_surfels

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


class SurfelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def setUp(self):
        self.directory = ROOT / 'outputs' / f'surfel-test-{uuid.uuid4().hex}'
        self.directory.mkdir()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if not self.directory.resolve().is_relative_to((ROOT / 'outputs').resolve()):
            raise ValueError('Refusing to clean a fixture outside outputs/')
        shutil.rmtree(self.directory)

    def test_rgb_geometry_gradients(self):
        check_gradients(DEVICE, 'rgb', verbose=False)

    def test_deferred_material_environment_and_overlapping_surface_gradients(self):
        check_gradients(DEVICE, 'deferred', overlapping=True, verbose=False)
        check_gradients(DEVICE, 'deferred', blended=True, verbose=False)

    def test_exact_removed_axis_and_normal_sign_invariance(self):
        trainer = make_trainer(DEVICE, 'deferred')
        trainer.loss_and_gradients([0])
        original = trainer.image.numpy().copy()
        self.assertEqual(float(trainer.optimizers.scale_grad.numpy()[0, 2]), 0.0)
        self.assertEqual(float(trainer.gaussian_set().scales[0, 2]), 0.0)
        logs = trainer.log_scales.numpy()
        for placeholder in (-40.0, 10.0):
            logs[0, 2] = placeholder
            trainer.log_scales.assign(logs)
            trainer.render_single_view_image(0)
            np.testing.assert_allclose(trainer.image.numpy(), original, atol=2e-6)
        q = trainer.quaternions.numpy()[0]
        # Right-multiply by a 180-degree local-x rotation: same plane, opposite local normal.
        flipped = np.array([[-q[1], q[0], q[3], -q[2]]], np.float32)
        trainer.quaternions.assign(flipped)
        trainer.render_single_view_image(0)
        np.testing.assert_allclose(trainer.image.numpy(), original, atol=2e-6)

    def test_gbuffer_depth_normal_coverage_and_uncovered_background(self):
        trainer = make_trainer(DEVICE)
        trainer.render_single_view_image(0)
        values = trainer.buffers.numpy()
        coverage = values[:, 7]
        self.assertTrue(np.all((coverage >= 0) & (coverage <= 1)))
        visible = coverage > 1e-6
        np.testing.assert_allclose(values[visible, 6] / coverage[visible], 1.2, atol=1e-6)
        normal = quaternion_to_matrix(trainer.quaternions.numpy())[0, :, 2]
        if np.dot(normal, -trainer.means.numpy()[0]) < 0:
            normal = -normal
        np.testing.assert_allclose(values[visible, 3:6] / coverage[visible, None],
                                   np.broadcast_to(normal, (int(visible.sum()), 3)), atol=2e-6)
        logits = trainer.opacity.numpy()
        logits[:] = -12
        trainer.opacity.assign(logits)
        image, _ = trainer.render_single_view_image(0)
        np.testing.assert_array_equal(image, np.ones((4, 4, 3), np.float32))
        np.testing.assert_array_equal(trainer.buffers.numpy(), 0)

    def test_training_only_initialization_ignores_held_out_colours(self):
        targets = np.concatenate((np.full((1, 4, 4, 3), 0.2, np.float32),
                                  np.full((1, 4, 4, 3), 0.9, np.float32)))
        cameras = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
        trainer = SurfelTrainer(targets, cameras, 4., 1., 2, 1, DEVICE, 7,
            training_view_count=1, surfel_options={'mode': 'deferred', 'environment_width': 8,
                                                  'environment_height': 4, 'save_debug': False})
        np.testing.assert_allclose(trainer.color.numpy()[0], 0.2, atol=1e-6)
        environment = trainer.environment.numpy()
        np.testing.assert_allclose(1 / (1 + np.exp(-environment)), 0.2, atol=1e-6)

    def test_gaussian_control_uses_training_statistics_and_preserves_volume(self):
        targets = np.concatenate((np.full((1, 4, 4, 3), 0.2, np.float32), np.full((1, 4, 4, 3), 0.9, np.float32)))
        control = GaussianRGBControl(targets, np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy(),
                                      4., 1., 1, 1, DEVICE, 7, training_view_count=1,
                                      training_options={'sparse': {'enabled': False}})
        np.testing.assert_allclose(control.color.numpy()[0], 0.2, atol=1e-6)
        self.assertGreater(float(control.gaussian_set().scales[0, 2]), 0)
        self.assertEqual(control.surfel_stage, 'gaussian_rgb')

    def test_splitting_stays_in_tangent_plane_and_copies_materials(self):
        trainer = make_trainer(DEVICE, 'deferred', capacity=4)
        origin = trainer.means.numpy()[0].copy()
        normal = quaternion_to_matrix(trainer.quaternions.numpy())[0, :, 2]
        reflection = trainer.reflection_logits.numpy()[0]
        trainer.densify_and_prune(np.ones((4, 3)), 1.0)
        self.assertEqual(int(trainer.active.sum()), 2)
        points = trainer.means.numpy()[:2]
        np.testing.assert_allclose((points - origin) @ normal, 0, atol=3e-7)
        np.testing.assert_allclose(trainer.reflection_logits.numpy()[1], reflection, atol=1e-7)
        np.testing.assert_array_equal(trainer.gaussian_set().scales[:, 2], 0)

    def test_materials_only_update_when_deferred_is_active(self):
        trainer = make_trainer(DEVICE, 'deferred')
        trainer.specular_active = False
        before = trainer.environment.numpy().copy()
        strength = trainer.reflection_logits.numpy().copy()
        trainer.step([0])
        np.testing.assert_array_equal(trainer.environment.numpy(), before)
        np.testing.assert_array_equal(trainer.reflection_logits.numpy(), strength)
        trainer.activate_specular()
        trainer.step([0], iteration=2)
        self.assertTrue(np.any(trainer.environment.numpy() != before))
        self.assertTrue(np.any(trainer.reflection_logits.numpy() != strength))

    def test_checkpoint_evaluation_and_rgb_to_deferred_transfer(self):
        for mode in ('rgb', 'deferred'):
            trainer = make_trainer(DEVICE, mode)
            trainer.render_single_view_image(0)
            expected = trainer.image.numpy().copy()
            path = self.directory / f'{mode}.ply'
            trainer.gaussian_set().to_ply(path)
            trainer.save_sidecars(path)
            args = argparse.Namespace(ply=path, background='white', tile_pair_capacity=None)
            scene = GaussianSet.from_ply(path)
            loaded = evaluate_surfels.build_trainer(args, scene, trainer.targets_host,
                                                    trainer.cameras.numpy(), trainer.focal, DEVICE)
            actual, _ = loaded.render_single_view_image(0)
            np.testing.assert_allclose(actual.reshape(-1, 3), expected, atol=2e-6)
            self.assertEqual(loaded.surfel_stage, trainer.surfel_stage)
        target = make_trainer(DEVICE, 'deferred')
        target.load_materials(self.directory / 'rgb.surfel.npz', 1)
        self.assertTrue(np.all(np.isfinite(target.environment.numpy())))
        with self.assertRaises(ValueError):
            read_surfel_sidecar(self.directory / 'rgb.surfel.npz', 2)

    def test_shader_has_coverage_aware_blend_and_no_background_reflection(self):
        values = np.zeros((2, 9), np.float32)
        values[:, :3] = [0.1, 0.2, 0.3]
        values[:, 5] = -0.5
        values[:, 7] = 0.5
        values[:, 8] = 0.25  # normalized reflection strength 0.5.
        values[1] = 0
        buffers = wp.array(values, dtype=k.GBuffer, device=DEVICE)
        env = wp.zeros(32, dtype=wp.vec3, device=DEVICE)  # sigmoid(0) = 0.5.
        cameras = wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=DEVICE)
        image = wp.empty(2, dtype=wp.vec3, device=DEVICE)
        wp.launch(k.deferred_shader, dim=2, inputs=[buffers, env, 8, 4, cameras, 0, 2, 1, 2., 1, wp.vec3(1.)], outputs=[image], device=DEVICE)
        np.testing.assert_allclose(image.numpy()[0], np.array([0.1, 0.2, 0.3]) * 0.5 + 0.125 + 0.5, atol=1e-6)
        np.testing.assert_array_equal(image.numpy()[1], [1., 1., 1.])

    def test_workflow_runs_rgb_and_deferred_as_separate_passes(self):
        for mode in ('rgb', 'deferred'):
            trainer = make_trainer(DEVICE, mode)
            trainer.surfel_options['save_debug'] = True
            trainer.specular_active = False
            trainer.loss_config.update(mode='mse_dssim', dssim_window=3)
            config = SurfelTrainingConfig(SurfelTrainingConfig._merge(SurfelTrainingConfig.DEFAULTS, {
                'paths': {'output': str(self.directory / f'{mode}.png')},
                'model': {'width': 4, 'height': 4, 'capacity': 1, 'initial_splats': 1},
                'training': {'iterations': 4}, 'loss': {'mode': 'mse_dssim', 'dssim_window': 3},
                'surfel': {'mode': mode, 'specular_start_iteration': 2},
                'densification': {'interval': 0}, 'multiview_adc': {'enabled': False},
                'adaptive': {'dssim_psnr_threshold': 0.0}, 'convergence': {'enabled': False},
                'reporting': {'eval_every': 1, 'snapshot_every': 2, 'log_every': 1, 'loss_ema_decay': 0.0},
            }))
            config.validate()
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                run_training(trainer, config, 1, [0])
            self.assertEqual(trainer.surfel_stage, mode)
            self.assertEqual(trainer.loss_stage, 'dssim')
            self.assertIn(f'Final surfel stage: {mode}', log.getvalue())
            self.assertTrue((self.directory / f'{mode}.surfel.npz').is_file())
            self.assertTrue((self.directory / f'{mode}.normals.png').is_file())
            self.assertTrue((self.directory / f'{mode}.buffers.npz').is_file())

    def test_invalid_project_configuration_is_rejected(self):
        for section, key, value in (('model', 'sh_degree', 2), ('sparse', 'enabled', True),
                                   ('surfel', 'mode', 'unknown'), ('surfel', 'initial_reflection', 1),
                                   ('surfel', 'environment_height', 0), ('surfel', 'reflection_lr', float('nan'))):
            values = deepcopy(SurfelTrainingConfig.DEFAULTS)
            values[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                SurfelTrainingConfig(values).validate()


if __name__ == '__main__':
    unittest.main()
