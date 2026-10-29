"""Paper image-loss derivatives, reflection normal gradients and propagation rules."""

from copy import deepcopy
import contextlib
import io
import os
from pathlib import Path
import sys
import shutil
import uuid
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from cp4282_gs.surfels.checks import make_trainer, reference
from cp4282_gs.surfels.geometry_losses import l1_image_loss
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels.workflow import run_training

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


class PaperNormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def trainer(self, propagation=False):
        trainer = make_trainer(DEVICE, 'deferred', overlapping=True, geometry=True, capacity=3)
        trainer.geometry_options['normal_depth_weight'] = 0
        trainer.surfel_options['photometric_loss'] = 'l1_dssim'
        trainer.surfel_options['normal_propagation'].update(
            enabled=propagation, interval=1, until_iteration=5, opacity_freeze_iterations=2)
        return trainer

    def test_l1_value_sign_and_zero_subgradient(self):
        values = np.array([[0.2, 0.5, 0.9], [0.8, 0.5, 0.1]], np.float32)
        image = wp.array(values, dtype=wp.vec3, device=DEVICE)
        target = wp.array(np.full((2, 3), 0.5, np.float32), dtype=wp.vec3, device=DEVICE)
        loss = wp.zeros(1, dtype=wp.float32, device=DEVICE)
        gradient = wp.zeros(2, dtype=wp.vec3, device=DEVICE)
        wp.launch(l1_image_loss, 2, inputs=[image, target, 0, 2, 1], outputs=[loss, gradient], device=DEVICE)
        self.assertAlmostEqual(float(loss.numpy()[0]), float(np.abs(values - 0.5).mean()), places=6)
        np.testing.assert_allclose(gradient.numpy(), np.sign(values - 0.5) / 6, atol=1e-7)

    def test_paper_objective_matches_l1_dssim_and_retains_mask(self):
        trainer = self.trainer()
        photo, mse, structural = trainer.evaluate_components([0])
        image = trainer.image.numpy().reshape(4, 4, 3)
        target = trainer.targets_host[0]
        self.assertAlmostEqual(photo, 0.8 * float(np.abs(image - target).mean()) + 0.2 * structural, places=6)
        self.assertAlmostEqual(mse, float(np.mean((image - target) ** 2)), places=6)
        objective = trainer.loss_and_gradients([0])
        self.assertEqual(trainer.loss_stage, 'l1_dssim')
        self.assertAlmostEqual(objective, photo + 0.1 * trainer.last_geometry_terms['mask'], places=6)
        self.assertEqual(trainer.last_geometry_terms['normal_weight'], 0)
        self.assertEqual(trainer.last_geometry_terms['normal_depth'], 0)

    def test_normals_receive_gradients_through_reflections(self):
        trainer = self.trainer()
        trainer.geometry_options['mask_weight'] = 0
        trainer.specular_active = False
        trainer.loss_and_gradients([0])
        np.testing.assert_array_equal(trainer.normal_gradient.numpy(), 0)
        trainer.specular_active = True
        trainer.loss_and_gradients([0])
        self.assertGreater(float(np.linalg.norm(trainer.normal_gradient.numpy())), 1e-5)

    def test_combined_paper_and_mask_gradients_match_finite_differences(self):
        trainer = self.trainer()
        trainer.loss_and_gradients([0])
        qindex = int(np.argmax(np.abs(trainer.optimizers.quaternion_grad_flat.numpy()[:4])))
        env = trainer.environment.grad.numpy()
        eindex = np.unravel_index(np.abs(env).argmax(), env.shape)
        cases = [(trainer.means, wp.vec3, (0, 0), float(trainer.optimizers.mean_grad.numpy()[0, 0]), 0.001),
                 (trainer.quaternions, wp.vec4, (0, qindex), float(trainer.optimizers.quaternion_grad_flat.numpy()[qindex]), 0.001),
                 (trainer.opacity, wp.float32, 0, float(trainer.optimizers.opacity_grad.numpy()[0]), 0.002),
                 (trainer.color, wp.vec3, (0, 1), float(trainer.optimizers.color_grad.numpy()[0, 1]), 0.001),
                 (trainer.reflection_logits, wp.float32, 0, float(trainer.reflection_gradient.numpy()[0]), 0.005),
                 (trainer.environment, wp.vec3, eindex, float(env[eindex]), 0.005)]
        for field, dtype, index, analytic, epsilon in cases:
            numeric = reference.finite_difference(trainer, field, dtype, index, epsilon)
            self.assertAlmostEqual(analytic, numeric, delta=max(5e-4, abs(numeric) * 0.04))

    def test_propagation_preserves_geometry_axes_and_inactive_slots(self):
        trainer = self.trainer(propagation=True)
        trainer.opacity.assign(np.log(np.array([0.2, 0.95, 0.3]) / (1 - np.array([0.2, 0.95, 0.3]))).astype(np.float32))
        strengths = np.array([0.25, 0.0001, 0.4], np.float32)
        trainer.reflection_logits.assign(np.log(strengths / (1 - strengths)))
        trainer.log_scales.assign(np.log(np.full((3, 3), 0.1, np.float32)))
        trainer.color.assign(np.full((3, 3), 0.5, np.float32))
        means, rotations = trainer.means.numpy().copy(), trainer.quaternions.numpy().copy()
        event = trainer.propagate_normals(1)
        self.assertEqual(event, {'expanded': 1, 'colours_reset': 1})
        np.testing.assert_allclose(np.exp(trainer.log_scales.numpy()[0]), [0.15, 0.15, 0.1], atol=1e-6)
        np.testing.assert_allclose(np.exp(trainer.log_scales.numpy()[1:]), 0.1, atol=1e-6)
        np.testing.assert_allclose(trainer.color.numpy()[[0, 2]], 0.5)
        self.assertGreater(np.linalg.norm(trainer.color.numpy()[1] - 0.5), 0)
        np.testing.assert_array_equal(trainer.means.numpy(), means)
        np.testing.assert_array_equal(trainer.quaternions.numpy(), rotations)
        opacity = 1 / (1 + np.exp(-trainer.opacity.numpy()))
        np.testing.assert_allclose(opacity, [0.9, 0.95, 0.3], atol=1e-6)
        strength = 1 / (1 + np.exp(-trainer.reflection_logits.numpy()))
        np.testing.assert_allclose(strength, [0.25, 0.001, 0.4], atol=1e-6)

    def test_opacity_resets_interleave_and_freeze_expires(self):
        trainer = self.trainer(propagation=True)
        trainer.propagate_normals(1)
        before = trainer.opacity.numpy().copy()
        trainer.step([0], iteration=2)
        np.testing.assert_array_equal(trainer.opacity.numpy(), before)
        self.assertEqual(trainer.optimizers.opacity_optimizer.lr, 0)
        trainer.step([0], iteration=4)
        self.assertEqual(trainer.optimizers.opacity_optimizer.lr, trainer.learning_rates['opacity'])
        scales, colour = trainer.log_scales.numpy().copy(), trainer.color.numpy().copy()
        trainer.reset_opacity()
        self.assertIsNone(trainer.propagate_normals(3, opacity_reset=True))
        np.testing.assert_array_equal(trainer.log_scales.numpy(), scales)
        np.testing.assert_array_equal(trainer.color.numpy(), colour)
        self.assertEqual(trainer._opacity_frozen_until, 0)
        self.assertIsNone(trainer.propagate_normals(6))

    def test_configuration_rejects_mixing_old_penalty_with_propagation(self):
        config = deepcopy(SurfelTrainingConfig.DEFAULTS)
        config['surfel'].update(mode='deferred', photometric_loss='l1_dssim')
        config['surfel']['normal_propagation']['enabled'] = True
        config['geometry_loss']['normal_depth_weight'] = 0.01
        with self.assertRaisesRegex(ValueError, 'replaces'):
            SurfelTrainingConfig(config).validate()

    def test_workflow_preserves_final_opacity_instead_of_clamping_last_step(self):
        trainer = self.trainer(propagation=True)
        directory = ROOT / 'outputs' / f'paper-workflow-{uuid.uuid4().hex}'
        directory.mkdir()
        def cleanup():
            assert directory.resolve().is_relative_to((ROOT / 'outputs').resolve())
            shutil.rmtree(directory)
        self.addCleanup(cleanup)
        config = SurfelTrainingConfig(SurfelTrainingConfig._merge(SurfelTrainingConfig.DEFAULTS, {
            'paths': {'output': str(directory / 'model.png')},
            'training': {'iterations': 6}, 'model': {'capacity': 3, 'initial_splats': 2, 'width': 4, 'height': 4},
            'surfel': trainer.surfel_options, 'geometry_loss': trainer.geometry_options,
            'densification': {'interval': 0, 'opacity_reset_interval': 3},
            'multiview_adc': {'enabled': False}, 'convergence': {'enabled': False},
            'reporting': {'eval_every': 1, 'snapshot_every': 6, 'log_every': 1},
        }))
        config.validate()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(trainer, 'reset_opacity', wraps=trainer.reset_opacity) as reset:
            run_training(trainer, config, 1, [0])
        self.assertEqual(reset.call_count, 1)  # Step 3, never the final step 6.


if __name__ == '__main__':
    unittest.main()
