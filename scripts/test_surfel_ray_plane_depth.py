"""Rendered plane depths, full depth adjoints, and checkpoint compatibility."""

from copy import deepcopy
import os
from pathlib import Path
import sys
import shutil
import unittest
import uuid

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src')]
from cp4282_gs.surfels.checks import make_trainer, reference
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels import kernels as k
from cp4282_gs.surfels.trainer import read_surfel_sidecar

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


@wp.kernel
def sample_depths(means: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                  cameras: wp.array(dtype=wp.mat44), rays: wp.array(dtype=wp.vec3),
                  depths: wp.array(dtype=wp.float32)):
    i = wp.tid()
    depths[i] = k.ray_plane_depth(means[i], normals[i], cameras[0], rays[i])


class RayPlaneDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def test_rendered_tilted_plane_depth_and_normals_in_two_camera_frames(self):
        for camera_angle in (0.0, 0.2):
            with self.subTest(camera_angle=camera_angle):
                trainer = make_trainer(DEVICE, geometry=True)
                theta = 0.35
                trainer.quaternions.assign(np.array([[np.cos(theta / 2), 0, np.sin(theta / 2), 0]], np.float32))
                trainer.log_scales.assign(np.log(np.array([[1., 1., 0.1]], np.float32)))
                c, s = np.cos(camera_angle), np.sin(camera_angle)
                camera = np.array([[c, 0, s, 0.1], [0, 1, 0, -0.05], [-s, 0, c, 0.3], [0, 0, 0, 1]], np.float32)
                point = np.array([0.02, 0.03, 1.2], np.float32)
                mean = camera[:3, :3].T @ (point - camera[:3, 3])
                trainer.cameras.assign(camera[None])
                trainer.means.assign(mean[None])
                trainer._forward(0)
                centre = trainer.buffers.numpy().copy()
                centre_image = trainer.image.numpy().copy()
                centre_loss = trainer.geometry_loss.compute(trainer.buffers, trainer.cameras, 0, 4., 0, backward=False)
                trainer.surfel_options['depth_mode'] = 'ray_plane'
                trainer._forward(0)
                plane = trainer.buffers.numpy()
                normal = camera[:3, :3] @ np.array([np.sin(theta), 0, np.cos(theta)])
                x, y = np.meshgrid((np.arange(4) + 0.5 - 2) / 4, (np.arange(4) + 0.5 - 2) / 4)
                rays = np.column_stack((x.ravel(), y.ravel(), np.ones(16)))
                expected = (normal @ point) / (rays @ normal)
                np.testing.assert_allclose(plane[:, 6] / plane[:, 7], expected, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(centre[:, 6] / centre[:, 7], 1.2, atol=2e-6)
                np.testing.assert_array_equal(plane[:, [0, 1, 2, 3, 4, 5, 7, 8]], centre[:, [0, 1, 2, 3, 4, 5, 7, 8]])
                np.testing.assert_array_equal(trainer.image.numpy(), centre_image)
                plane_loss = trainer.geometry_loss.compute(trainer.buffers, trainer.cameras, 0, 4., 0, backward=False)
                self.assertEqual(plane_loss['normal_pixels'], 4)
                self.assertLess(abs(plane_loss['normal_depth']), 2e-6)
                self.assertGreater(centre_loss['normal_depth'], 0.05)

    def test_overlapping_depth_adjoint_matches_finite_differences(self):
        trainer = make_trainer(DEVICE, overlapping=True)
        trainer.surfel_options['depth_mode'] = 'ray_plane'
        offsets, pairs = trainer._forward(0)
        weights = np.linspace(0.03, 0.11, 16, dtype=np.float32)
        gradients = np.zeros((16, 9), np.float32)
        gradients[:, 6] = weights
        trainer.buffers.grad.assign(gradients)
        trainer.optimizers.zero_gradients()
        trainer.normal_gradient.zero_()
        trainer.reflection_gradient.zero_()
        wp.launch(k.raster_backward, 16,
                  inputs=[trainer.means, trainer.log_scales, trainer.quaternions, trainer.opacity, trainer.color,
                          trainer.attributes, trainer.cameras, 0, offsets, pairs, 4, 4, trainer.tiles_x, 4., 1]
                          + trainer.compact_box_args + [trainer.buffers, trainer.buffers.grad],
                  outputs=[trainer.optimizers.mean_grad_flat, trainer.optimizers.scale_grad_flat,
                           trainer.optimizers.quaternion_grad_flat, trainer.optimizers.opacity_grad,
                           trainer.optimizers.color_grad_flat, trainer.normal_gradient, trainer.reflection_gradient],
                  device=DEVICE)
        wp.launch(k.normal_backward, trainer.capacity,
                  inputs=[trainer.means, trainer.quaternions, trainer.cameras, 0, trainer.normal_gradient],
                  outputs=[trainer.optimizers.mean_grad_flat, trainer.optimizers.quaternion_grad_flat], device=DEVICE)
        cases = [(trainer.means, trainer.optimizers.mean_grad_flat.numpy().reshape(-1, 3)),
                 (trainer.quaternions, trainer.optimizers.quaternion_grad_flat.numpy().reshape(-1, 4)),
                 (trainer.log_scales, trainer.optimizers.scale_grad_flat.numpy().reshape(-1, 3)),
                 (trainer.opacity, trainer.optimizers.opacity_grad.numpy())]
        self.assertGreater(np.linalg.norm(trainer.normal_gradient.numpy()), 0.01)
        for field, analytic in cases:
            original = field.numpy().copy()
            for index in np.ndindex(original.shape):
                epsilon = 0.0005
                high_low = []
                for sign in (1, -1):
                    values = original.copy()
                    values[index] += sign * epsilon
                    field.assign(values)
                    trainer._forward(0)
                    high_low.append(float(weights @ trainer.buffers.numpy()[:, 6]))
                field.assign(original)
                numeric = (high_low[0] - high_low[1]) / (2 * epsilon)
                with self.subTest(shape=original.shape, index=index):
                    self.assertAlmostEqual(float(analytic[index]), numeric, delta=max(0.0002, abs(numeric) * 0.006))

    def test_combined_normal_and_image_loss_gradients(self):
        trainer = make_trainer(DEVICE, 'deferred', blended=True, overlapping=True, geometry=True)
        trainer.surfel_options['depth_mode'] = 'ray_plane'
        trainer.geometry_options['normal_depth_weight'] = 1.0
        trainer.loss_and_gradients([0])
        self.assertGreater(trainer.last_geometry_terms['normal_pixels'], 0)
        cases = [(trainer.means, wp.vec3, trainer.optimizers.mean_grad_flat.numpy().reshape(-1, 3)),
                 (trainer.quaternions, wp.vec4, trainer.optimizers.quaternion_grad_flat.numpy().reshape(-1, 4)),
                 (trainer.opacity, wp.float32, trainer.optimizers.opacity_grad.numpy())]
        for field, dtype, analytic in cases:
            for index in np.ndindex(analytic.shape):
                numeric = reference.finite_difference(trainer, field, dtype, index, 0.0005)
                with self.subTest(shape=analytic.shape, index=index):
                    self.assertAlmostEqual(float(analytic[index]), numeric, delta=max(0.0004, abs(numeric) * 0.02))

    def test_grazing_parallel_and_behind_camera_fall_back_to_centre(self):
        means = np.array([[0, 0, 1.2]] * 4, np.float32)
        normals = np.array([[1, 0, 0], [1, 0, 1e-6], [1, 0, 1], [0, 0, 1]], np.float32)
        rays = np.array([[0, 0, 1], [0, 0, 1], [-2, 0, 1], [0.2, 0.1, 1]], np.float32)
        depths = wp.zeros(4, dtype=wp.float32, device=DEVICE)
        wp.launch(sample_depths, 4,
                  inputs=[wp.array(means, dtype=wp.vec3, device=DEVICE),
                          wp.array(normals, dtype=wp.vec3, device=DEVICE),
                          wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=DEVICE),
                          wp.array(rays, dtype=wp.vec3, device=DEVICE)], outputs=[depths], device=DEVICE)
        np.testing.assert_allclose(depths.numpy(), 1.2, atol=1e-7)

    def test_checkpoint_preserves_depth_mode_and_legacy_defaults(self):
        trainer = make_trainer(DEVICE)
        trainer.surfel_options['depth_mode'] = 'ray_plane'
        trainer._forward(0)
        expected = trainer.buffers.numpy().copy()
        directory = ROOT / 'outputs' / f'ray-plane-test-{uuid.uuid4().hex}'
        directory.mkdir()
        try:
            path = directory / 'model.png'
            sidecar = trainer.save_sidecars(path)[0]
            self.assertEqual(read_surfel_sidecar(sidecar, trainer.capacity)['depth_mode'], 'ray_plane')
            trainer.surfel_options['depth_mode'] = 'centre'
            trainer.load_materials(sidecar, trainer.capacity, for_evaluation=True)
            trainer._forward(0)
            np.testing.assert_array_equal(trainer.buffers.numpy(), expected)
            with np.load(sidecar, allow_pickle=False) as data:
                legacy = {name: data[name] for name in data.files if name != 'depth_mode'}
            np.savez_compressed(sidecar, **legacy)
            trainer.load_materials(sidecar, trainer.capacity, for_evaluation=True)
            self.assertEqual(trainer.surfel_options['depth_mode'], 'centre')
        finally:
            if not directory.resolve().is_relative_to((ROOT / 'outputs').resolve()):
                raise ValueError('Refusing to clean outside outputs/')
            shutil.rmtree(directory)

    def test_preset_changes_only_depth_mode_and_output(self):
        original = SurfelTrainingConfig.load(ROOT / 'config/materials_red-rose-silver_deferred_consistency.yaml').printable()
        updated = SurfelTrainingConfig.load(ROOT / 'config/materials_red-rose-silver_deferred_ray_plane.yaml').printable()
        self.assertEqual(updated['surfel']['depth_mode'], 'ray_plane')
        for config in (original, updated):
            config['paths'].pop('output')
            config['surfel'].pop('depth_mode')
        self.assertEqual(original, updated)
        config = deepcopy(SurfelTrainingConfig.DEFAULTS)
        config['surfel']['depth_mode'] = 'invalid'
        with self.assertRaises(ValueError):
            SurfelTrainingConfig(config).validate()


if __name__ == '__main__':
    unittest.main()
