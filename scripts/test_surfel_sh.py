"""End-to-end gradients, staging, inheritance and serialization of late base SH."""

import argparse
import contextlib
from copy import deepcopy
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from cp4282_gs.surfels.checks import make_trainer, reference
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels.trainer import SurfelTrainer
from cp4282_gs.surfels.workflow import run_training
from gaussian_set import GaussianSet
from evaluate_surfels import build_trainer

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


def sh_trainer(mode='deferred', capacity=2):
    source = make_trainer(DEVICE, mode, overlapping=True)
    trainer = SurfelTrainer(source.targets_host, source.cameras.numpy(), source.focal,
        source.scene_radius, capacity, 2, DEVICE, 7, init_scene=source.gaussian_set(),
        sh_degree=3, training_view_count=1,
        training_options={'compact_box': {'enabled': False}},
        surfel_options=source.surfel_options)
    trainer.environment.assign(source.environment.numpy())
    coefficients = np.zeros((capacity, 15, 3), np.float32)
    coefficients[:2] = np.random.default_rng(42).normal(0, 0.08, (2, 15, 3))
    trainer.sh.coefficients.assign(coefficients.reshape(-1, 3))
    return trainer


class SurfelSHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def test_full_image_gradients_include_sh_position_opacity_and_material_paths(self):
        for mode in ('rgb', 'deferred'):
            trainer = sh_trainer(mode)
            trainer.sh.degree = 3
            trainer.loss_and_gradients([0])
            coefficient_gradient = trainer.sh.gradient.numpy()
            index = np.unravel_index(np.argmax(np.abs(coefficient_gradient)), coefficient_gradient.shape)
            cases = [(trainer.sh.coefficients, wp.vec3, index, float(coefficient_gradient[index])),
                     (trainer.means, wp.vec3, (0, 0), float(trainer.optimizers.mean_grad.numpy()[0, 0])),
                     (trainer.means, wp.vec3, (0, 2), float(trainer.optimizers.mean_grad.numpy()[0, 2])),
                     (trainer.opacity, wp.float32, 0, float(trainer.optimizers.opacity_grad.numpy()[0]))]
            if mode == 'deferred':
                cases.append((trainer.reflection_logits, wp.float32, 0, float(trainer.reflection_gradient.numpy()[0])))
            for field, dtype, location, analytic in cases:
                with self.subTest(mode=mode, location=location, dtype=str(dtype)):
                    numeric = reference.finite_difference(trainer, field, dtype, location, 0.002)
                    self.assertAlmostEqual(analytic, numeric, delta=max(3e-4, abs(numeric) * 0.03))

    def test_zero_degree_is_identical_and_locked_bands_do_not_update(self):
        trainer = sh_trainer()
        trainer.render_single_view_image(0)
        expected = trainer.image.numpy().copy()
        trainer.sh.coefficients.zero_()
        trainer.render_single_view_image(0)
        np.testing.assert_array_equal(trainer.image.numpy(), expected)
        trainer.sh.degree = 1
        trainer.step([0])
        values = trainer.sh.coefficients.numpy().reshape(2, 15, 3)
        self.assertTrue(np.any(values[:, :3] != 0))
        np.testing.assert_array_equal(values[:, 3:], 0)
        self.assertEqual(trainer.sh.steps, [1, 0, 0])

    def test_clone_and_reused_slot_inherit_coefficients_and_adam_trajectory(self):
        trainer = sh_trainer(capacity=4)
        trainer.sh.degree = 3
        trainer.step([0])
        expected = [x.numpy().reshape(4, 15, 3).copy() for x in
                    (trainer.sh.coefficients, trainer.sh.moment, trainer.sh.variance)]
        children = trainer.densify_and_prune(np.array([[2, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]]), 0.5)
        self.assertEqual(list(children), [2])
        for field, values in zip((trainer.sh.coefficients, trainer.sh.moment, trainer.sh.variance), expected):
            np.testing.assert_array_equal(field.numpy().reshape(4, 15, 3)[2], values[0])

    def test_evaluation_restores_degree_and_rejects_missing_or_untrained_appearance(self):
        trainer = sh_trainer()
        trainer.sh.degree = 2
        expected, _ = trainer.render_single_view_image(0)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            path = Path(directory) / 'model.ply'
            trainer.gaussian_set().to_ply(path)
            trainer.save_sidecars(path)
            args = argparse.Namespace(ply=path, background='white', tile_pair_capacity=None, sh_degree=None)
            scene = GaussianSet.from_ply(path)
            loaded = build_trainer(args, scene, trainer.targets_host, trainer.cameras.numpy(), trainer.focal, DEVICE)
            self.assertEqual(loaded.active_sh_degree, 2)
            image, _ = loaded.render_single_view_image(0)
            np.testing.assert_allclose(image, expected, atol=2e-6)
            args.sh_degree = 3
            with self.assertRaises(ValueError):
                build_trainer(args, scene, trainer.targets_host, trainer.cameras.numpy(), trainer.focal, DEVICE)
            args.sh_degree = None
            path.with_suffix('.sh.npz').unlink()
            with self.assertRaises(FileNotFoundError):
                build_trainer(args, scene, trainer.targets_host, trainer.cameras.numpy(), trainer.focal, DEVICE)

    def test_scheduled_bands_unlock_in_workflow_and_checkpoint(self):
        trainer = sh_trainer()
        trainer.sh.coefficients.zero_()
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            values = SurfelTrainingConfig._merge(SurfelTrainingConfig.DEFAULTS, {
                'paths': {'output': str(Path(directory) / 'model.png')},
                'model': {'width': 4, 'height': 4, 'capacity': 2, 'initial_splats': 2, 'sh_degree': 3},
                'training': {'iterations': 8}, 'convergence': {'enabled': False},
                'densification': {'interval': 0, 'opacity_reset_interval': 0},
                'multiview_adc': {'enabled': False},
                'surfel': {'mode': 'deferred', 'specular_start_iteration': 1,
                           'sh_start_iteration': 3, 'sh_band_interval': 2},
                'reporting': {'log_every': 1, 'eval_every': 1, 'snapshot_every': 8},
            })
            config = SurfelTrainingConfig(values)
            config.validate()
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                run_training(trainer, config, 1, [0])
            self.assertEqual(trainer.active_sh_degree, 3)
            for iteration, degree in ((3, 1), (5, 2), (7, 3)):
                self.assertIn(f'{iteration:6d}: unlocking late base SH degree {degree}.', log.getvalue())
            self.assertEqual(trainer.sh.steps, [6, 4, 2])
            self.assertTrue((Path(directory) / 'model.sh.npz').exists())

    def test_config_requires_late_complete_schedule_and_no_overlap_with_propagation(self):
        values = deepcopy(SurfelTrainingConfig.DEFAULTS)
        values['training']['iterations'] = 30000
        values['model']['sh_degree'] = 3
        values['surfel'].update(mode='deferred', sh_start_iteration=20001)
        SurfelTrainingConfig(values).validate()
        for change in ({'sh_start_iteration': None}, {'sh_start_iteration': 1000},
                       {'sh_start_iteration': 29000}, {'sh_band_interval': 0},
                       {'normal_propagation': dict(values['surfel']['normal_propagation'], enabled=True)}):
            invalid = deepcopy(values)
            invalid['surfel'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SurfelTrainingConfig(invalid).validate()


if __name__ == '__main__':
    unittest.main()
