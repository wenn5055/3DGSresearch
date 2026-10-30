"""HDR shader adjoints, display/alpha semantics, optimizer and checkpoint checks."""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from cp4282_gs.surfels.checks import make_trainer, check_gradients
from cp4282_gs.surfels.hdr_shader import deferred_hdr_shader, effective_blend
from cp4282_gs.surfels.kernels import GBuffer
from cp4282_gs.surfels.trainer import read_surfel_sidecar
from cp4282_gs.surfels.config import SurfelTrainingConfig
from gaussian_set import GaussianSet
from evaluate_surfels import build_trainer

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


class HDRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def test_material_geometry_and_angular_adjoints(self):
        check_gradients(DEVICE, 'deferred', overlapping=True, shading_model='hdr_angular')
        check_gradients(DEVICE, 'deferred', blended=True, shading_model='hdr_angular')

    def test_hdr_with_mask_and_normal_regularization_adjoints(self):
        check_gradients(DEVICE, 'deferred', blended=True, overlapping=True,
                        geometry=True, shading_model='hdr_angular')

    def test_display_conversion_precedes_alpha_composition(self):
        # Pure reflection of 0.25 linear radiance with 40% foreground coverage.
        values = np.zeros((1, 9), np.float32)
        values[0, 5] = -.4
        values[0, 7] = .4
        values[0, 8] = .4
        buffers = wp.array(values, dtype=GBuffer, device=DEVICE)
        environment = wp.array(np.full((4, 3), np.log(np.expm1(.25)), np.float32), dtype=wp.vec3, device=DEVICE)
        gain = wp.zeros(16, dtype=wp.float32, device=DEVICE)
        cameras = wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=DEVICE)
        image = wp.empty(1, dtype=wp.vec3, device=DEVICE)
        wp.launch(deferred_hdr_shader, 1, inputs=[buffers, environment, 2, 2, cameras, 0,
            1, 1, 1., 1, wp.vec3(1.), gain], outputs=[image], device=DEVICE)
        expected = .4 * (1.055 * .25 ** (1 / 2.4) - .055) + .6
        np.testing.assert_allclose(image.numpy(), expected, atol=2e-6)
        blend = effective_blend(values.reshape(1, 1, 9), np.eye(4), 1., np.zeros(16))
        np.testing.assert_allclose(blend, 1 - 1e-6, atol=1e-7)

    def test_gain_updates_and_checkpoint_roundtrip(self):
        trainer = make_trainer(DEVICE, 'deferred', shading_model='hdr_angular')
        before = trainer.angular_gain.numpy().copy()
        trainer.step([0])
        self.assertTrue(np.any(trainer.angular_gain.numpy() != before))
        trainer.render_single_view_image(0)
        expected = trainer.image.numpy().copy()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.ply'
            trainer.gaussian_set().to_ply(path)
            trainer.save_sidecars(path)
            metadata = read_surfel_sidecar(path.with_suffix('.surfel.npz'), 1)
            self.assertEqual(metadata['shading_model'], 'hdr_angular')
            args = argparse.Namespace(ply=path, background='white', tile_pair_capacity=None)
            loaded = build_trainer(args, GaussianSet.from_ply(path), trainer.targets_host,
                                   trainer.cameras.numpy(), trainer.focal, DEVICE)
            loaded.training_options['compact_box'] = dict(trainer.training_options['compact_box'])
            actual, _ = loaded.render_single_view_image(0)
            np.testing.assert_allclose(actual.reshape(-1, 3), expected, atol=2e-6)
            np.testing.assert_array_equal(loaded.angular_gain.numpy(), trainer.angular_gain.numpy())

    def test_gain_frozen_when_specular_inactive(self):
        trainer = make_trainer(DEVICE, 'deferred', shading_model='hdr_angular')
        trainer.specular_active = False
        before = trainer.angular_gain.numpy().copy()
        trainer.step([0])
        np.testing.assert_array_equal(trainer.angular_gain.numpy(), before)

    def test_hdr_capacity_and_uncovered_background(self):
        values = np.zeros((2, 9), np.float32)
        values[0, 5] = -1
        values[0, 7:9] = 1
        buffers = wp.array(values, dtype=GBuffer, device=DEVICE)
        environment = wp.array(np.full((4, 3), 4., np.float32), dtype=wp.vec3, device=DEVICE)
        gain = wp.zeros(16, dtype=wp.float32, device=DEVICE)
        cameras = wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=DEVICE)
        image = wp.empty(2, dtype=wp.vec3, device=DEVICE)
        wp.launch(deferred_hdr_shader, 2, inputs=[buffers, environment, 2, 2, cameras, 0,
            2, 1, 2., 1, wp.vec3(.2, .3, .4), gain], outputs=[image], device=DEVICE)
        np.testing.assert_allclose(image.numpy()[0], 1., atol=1e-6)
        np.testing.assert_allclose(image.numpy()[1], [.2, .3, .4], atol=1e-7)

    def test_densification_preserves_global_angular_optimizer(self):
        trainer = make_trainer(DEVICE, 'deferred', capacity=4, shading_model='hdr_angular')
        trainer.step([0])
        before = [x.numpy().copy() for x in (*trainer.angular_optimizer.m, *trainer.angular_optimizer.v)]
        clock = trainer.angular_optimizer.t
        gain = trainer.angular_gain.numpy().copy()
        trainer.densify_and_prune(np.ones((4, 3)), 1.)
        self.assertEqual(trainer.angular_optimizer.t, clock)
        np.testing.assert_array_equal(trainer.angular_gain.numpy(), gain)
        for actual, expected in zip((*trainer.angular_optimizer.m, *trainer.angular_optimizer.v), before):
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_frozen_geometry_and_visibility_are_exact_while_appearance_learns(self):
        trainer = make_trainer(DEVICE, 'deferred', shading_model='hdr_angular', freeze_geometry=True)
        fields = (trainer.means, trainer.log_scales, trainer.quaternions, trainer.opacity)
        initial = [field.numpy().copy() for field in fields]
        trainer.render_single_view_image(0)
        buffers = trainer.buffers.numpy().copy()
        environment = trainer.environment.numpy().copy()
        for iteration in range(1, 5):
            trainer.step([0], iteration=iteration)
        for field, value in zip(fields, initial):
            np.testing.assert_array_equal(field.numpy(), value)
        trainer.render_single_view_image(0)
        np.testing.assert_array_equal(trainer.buffers.numpy()[:, 3:8], buffers[:, 3:8])
        self.assertTrue(np.any(trainer.environment.numpy() != environment))

    def test_frozen_geometry_rejects_structural_schedules(self):
        values = SurfelTrainingConfig._merge(SurfelTrainingConfig.DEFAULTS, {'surfel': {'freeze_geometry': True}})
        with self.assertRaisesRegex(ValueError, 'Frozen-geometry'):
            SurfelTrainingConfig(values).validate()

    def test_frozen_geometry_still_trains_sh_coefficients(self):
        trainer = make_trainer(DEVICE, 'rgb', freeze_geometry=True, sh_degree=3)
        trainer.sh.degree = 3
        before = trainer.sh.coefficients.numpy().copy()
        geometry = trainer.quaternions.numpy().copy()
        trainer.step([0])
        np.testing.assert_array_equal(trainer.quaternions.numpy(), geometry)
        self.assertTrue(np.any(trainer.sh.coefficients.numpy() != before))


if __name__ == '__main__':
    unittest.main()
