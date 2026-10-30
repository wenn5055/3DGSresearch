"""Child-state policy must cover geometry, SH and reflection without changing inference."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src')]
from cp4282_gs.surfels.checks import make_trainer
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels.trainer import GaussianRGBControl, SurfelTrainer
from training_config import TrainingConfig

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')
POLICIES = ('inherit', 'fresh_children', 'legacy_reset')


def trainer_for(kind, policy):
    source = make_trainer(DEVICE, 'deferred', overlapping=True)
    options = deepcopy(source.surfel_options)
    if kind == 'gaussian':
        options['mode'] = 'rgb'
    initial = source.gaussian_set()
    if kind == 'gaussian':
        initial.scales[:, 2] = .03
    cls = GaussianRGBControl if kind == 'gaussian' else SurfelTrainer
    trainer = cls(source.targets_host, source.cameras.numpy(), source.focal,
        source.scene_radius, 4, 2, DEVICE, 7, init_scene=initial,
        training_view_count=1, sh_degree=3, child_optimizer_state=policy,
        surfel_options=options, training_options={'compact_box': {'enabled': False}, 'sparse': {'enabled': False},
                                                  'multiview_adc': {'enabled': False}})
    if kind == 'surfel':
        trainer.environment.assign(source.environment.numpy())
    trainer.sh.degree = 3
    coefficients = np.random.default_rng(42).normal(0, .08, (4 * 15, 3)).astype(np.float32)
    trainer.sh.coefficients.assign(coefficients)
    return trainer


def seed_history(trainer):
    optimizers = list(trainer.optimizers.optimizers)
    if isinstance(trainer, SurfelTrainer):
        optimizers.append(trainer.reflection_optimizer)
    arrays = [array for optimizer in optimizers for array in (*optimizer.m, *optimizer.v)]
    arrays += [trainer.optimizers.quaternion_m, trainer.optimizers.quaternion_v]
    for array in arrays:
        values = array.numpy()
        values[:] = np.arange(1, 5, dtype=np.float32).reshape((4,) + (1,) * (values.ndim - 1))
        array.assign(values)
    for array in (trainer.sh.moment, trainer.sh.variance):
        values = array.numpy().reshape(4, 15, 3)
        values[:] = np.arange(1, 5, dtype=np.float32)[:, None, None]
        array.assign(values.reshape(-1, 3))
    for optimizer in optimizers:
        optimizer.t = 37
    trainer.optimizers.quaternion_adam_t = 37
    trainer.sh.steps = [11, 9, 7]
    return optimizers, arrays


class ChildOptimizerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def test_clone_and_split_state_for_gaussian_and_deferred_sh(self):
        for kind in ('gaussian', 'surfel'):
            for policy in POLICIES:
                for split in (False, True):
                    with self.subTest(kind=kind, policy=policy, split=split):
                        trainer = trainer_for(kind, policy)
                        trainer.step([0])
                        optimizers, arrays = seed_history(trainer)
                        before = [array.numpy().copy() for array in arrays]
                        sh_before = [array.numpy().reshape(4, 15, 3).copy()
                                     for array in (trainer.sh.moment, trainer.sh.variance)]
                        coefficients = trainer.sh.coefficients.numpy().reshape(4, 15, 3).copy()
                        environment = None
                        if kind == 'surfel':
                            environment = [x.numpy().copy() for x in (*trainer.environment_optimizer.m, *trainer.environment_optimizer.v)]
                            environment_clock = trainer.environment_optimizer.t
                        gradients = np.zeros((4, 3), np.float32)
                        gradients[0] = 1
                        children = trainer.densify_and_prune(gradients, .5, percent_dense=.001 if split else 1.)
                        self.assertEqual(children.tolist(), [2])
                        for array, old in zip(arrays, before):
                            expected = old.copy()
                            if policy == 'legacy_reset':
                                expected[:] = 0
                            else:
                                expected[2] = old[0] if policy == 'inherit' else 0
                            np.testing.assert_array_equal(array.numpy(), expected)
                        for array, old in zip((trainer.sh.moment, trainer.sh.variance), sh_before):
                            expected = old.copy()
                            if policy == 'legacy_reset':
                                expected[:] = 0
                            else:
                                expected[2] = old[0] if policy == 'inherit' else 0
                            np.testing.assert_array_equal(array.numpy().reshape(4, 15, 3), expected)
                        np.testing.assert_array_equal(trainer.sh.coefficients.numpy().reshape(4, 15, 3)[2], coefficients[0])
                        clock = 0 if policy == 'legacy_reset' else 37
                        self.assertTrue(all(optimizer.t == clock for optimizer in optimizers))
                        self.assertEqual(trainer.optimizers.quaternion_adam_t, clock)
                        self.assertEqual(trainer.sh.steps, [0, 0, 0] if policy == 'legacy_reset' else [11, 9, 7])
                        if environment is not None:
                            self.assertEqual(trainer.environment_optimizer.t, environment_clock)
                            for current, old in zip((*trainer.environment_optimizer.m, *trainer.environment_optimizer.v), environment):
                                np.testing.assert_array_equal(current.numpy(), old)

    def test_zero_child_events_preserve_all_histories(self):
        for policy in POLICIES:
            trainer = trainer_for('surfel', policy)
            trainer.step([0])
            optimizers, arrays = seed_history(trainer)
            before = [array.numpy().copy() for array in arrays]
            sh_before = trainer.sh.moment.numpy().copy()
            children = trainer.densify_and_prune(np.zeros((4, 3), np.float32), 0.)
            self.assertEqual(len(children), 0)
            for array, old in zip(arrays, before):
                np.testing.assert_array_equal(array.numpy(), old)
            np.testing.assert_array_equal(trainer.sh.moment.numpy(), sh_before)
            self.assertTrue(all(optimizer.t == 37 for optimizer in optimizers))
            self.assertEqual(trainer.sh.steps, [11, 9, 7])

    def test_initial_render_is_identical_across_policies(self):
        for kind in ('gaussian', 'surfel'):
            images = [trainer_for(kind, policy).render_single_view_image(0)[0] for policy in POLICIES]
            for image in images[1:]:
                np.testing.assert_array_equal(image, images[0])

    def test_config_policy_validation_and_default(self):
        self.assertEqual(TrainingConfig.DEFAULTS['densification']['child_optimizer_state'], 'fresh_children')
        for policy in POLICIES:
            values = deepcopy(SurfelTrainingConfig.DEFAULTS)
            values['densification']['child_optimizer_state'] = policy
            SurfelTrainingConfig(values).validate()
        for policy in ('unknown', None, True):
            values = deepcopy(SurfelTrainingConfig.DEFAULTS)
            values['densification']['child_optimizer_state'] = policy
            with self.assertRaises(ValueError):
                SurfelTrainingConfig(values).validate()


if __name__ == '__main__':
    unittest.main()
