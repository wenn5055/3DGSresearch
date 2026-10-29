"""Regression checks for boundary recovery, state inheritance and quiet refinement."""

import contextlib
from copy import deepcopy
import io
import os
from pathlib import Path
import sys
import shutil
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src')]
from cp4282_gs.surfels.checks import make_trainer, reference
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels.geometry import base
from cp4282_gs.surfels.workflow import run_training

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


class OptimizerRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.use_precompiled_headers = False
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.init()

    def setUp(self):
        self.directory = ROOT / 'outputs' / f'optimizer-refinement-test-{uuid.uuid4().hex}'
        self.directory.mkdir()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if not self.directory.resolve().is_relative_to((ROOT / 'outputs').resolve()):
            raise ValueError('Refusing to clean outside outputs/')
        shutil.rmtree(self.directory)

    def test_clamped_rgb_recovers_inward_in_surfel_dense_and_sparse_paths(self):
        for mode in ('surfel', 'gaussian_dense', 'gaussian_sparse'):
            for endpoint in (0., 1.):
                with self.subTest(mode=mode, endpoint=endpoint):
                    target = 1. - endpoint
                    if mode == 'surfel':
                        trainer = make_trainer(DEVICE, 'rgb')
                        trainer.targets_host[:] = target
                        trainer.targets.assign(trainer.targets_host.reshape(-1, 3))
                    else:
                        trainer = base.WarpImageTrainer(
                            np.full((1, 16, 16, 3), target, np.float32),
                            np.eye(4, dtype=np.float32)[None], 16., 1., 1, 1, DEVICE, 7,
                            init_scene=reference.one_splat_scene(),
                            training_options={'sparse': {'enabled': mode == 'gaussian_sparse',
                                                         'samples_per_tile': 256},
                                              'compact_box': {'enabled': False}})
                    colours = trainer.color.numpy()
                    colours[:] = endpoint
                    trainer.color.assign(colours)
                    before = trainer.evaluate([0])
                    trainer.step([0])
                    after = trainer.evaluate([0])
                    channel = float(trainer.color.numpy()[0, 1])
                    self.assertTrue(0 < channel < 1)
                    self.assertLess(after, before)

    def test_density_preserves_live_history_and_inherits_reused_slot_history(self):
        for split in (False, True):
            with self.subTest(split=split):
                trainer = make_trainer(DEVICE, 'deferred', overlapping=True, capacity=4)
                trainer.training_options['multiview_adc']['enabled'] = False
                trainer.step([0])
                optimizers = (*trainer.optimizers.optimizers, trainer.reflection_optimizer)
                state = [(o.t, [x.numpy().copy() for x in (*o.m, *o.v)]) for o in optimizers]
                qm, qv = trainer.optimizers.quaternion_m.numpy(), trainer.optimizers.quaternion_v.numpy()
                clock = trainer.optimizers.quaternion_adam_t
                gradients = np.zeros((4, 3), np.float32)
                gradients[0] = 1
                children = trainer.densify_and_prune(gradients, .5, percent_dense=.001 if split else 1.)
                self.assertEqual(children.tolist(), [2])
                for optimizer, (old_clock, arrays) in zip(optimizers, state):
                    self.assertEqual(optimizer.t, old_clock)
                    for current, old in zip((*optimizer.m, *optimizer.v), arrays):
                        values = current.numpy()
                        np.testing.assert_array_equal(values[:2], old[:2])
                        np.testing.assert_array_equal(values[2], old[0])
                self.assertEqual(trainer.optimizers.quaternion_adam_t, clock)
                for current, old in ((trainer.optimizers.quaternion_m, qm), (trainer.optimizers.quaternion_v, qv)):
                    np.testing.assert_array_equal(current.numpy()[:2], old[:2])
                    np.testing.assert_array_equal(current.numpy()[2], old[0])

    def test_rejected_density_candidates_do_not_reset_adam(self):
        trainer = make_trainer(DEVICE, 'deferred', capacity=3)
        trainer.step([0])
        trainer.training_options['multiview_adc']['enabled'] = True
        before = trainer.optimizers.quaternion_m.numpy().copy()
        with patch.object(trainer, 'multiview_adc_scores', return_value=(np.zeros(1), np.zeros(1))):
            children = trainer.densify_and_prune(np.ones((3, 3)), .2, view_ids=[0])
        self.assertEqual(len(children), 0)
        self.assertEqual(trainer.optimizers.quaternion_adam_t, 1)
        self.assertEqual(trainer.reflection_optimizer.t, 1)
        np.testing.assert_array_equal(trainer.optimizers.quaternion_m.numpy(), before)

    def test_quiet_refinement_stops_interventions_and_optionally_prunes(self):
        for prune_interval in (0, 1):
            with self.subTest(prune_interval=prune_interval):
                directory = self.directory / str(prune_interval)
                directory.mkdir()
                trainer = make_trainer(DEVICE, 'deferred', capacity=8)
                trainer.training_options['multiview_adc']['enabled'] = False
                config = SurfelTrainingConfig(SurfelTrainingConfig._merge(SurfelTrainingConfig.DEFAULTS, {
                    'paths': {'output': str(Path(directory) / 'model.png')},
                    'model': {'width': 4, 'height': 4, 'capacity': 8, 'initial_splats': 1},
                    'training': {'iterations': 6},
                    'surfel': {'mode': 'deferred', 'specular_start_iteration': 0,
                               'normal_propagation': {'enabled': True, 'interval': 1, 'until_iteration': 5,
                                                      'densification_interval': 1},
                               'refinement': {'start_iteration': 3, 'prune_interval': prune_interval}},
                    'densification': {'interval': 1, 'fraction': .5, 'opacity_reset_interval': 2},
                    'multiview_adc': {'enabled': False}, 'convergence': {'enabled': False},
                    'reporting': {'eval_every': 1, 'snapshot_every': 6, 'log_every': 1},
                }))
                config.validate()
                trainer.surfel_options = deepcopy(config['surfel'])
                fractions = []
                original_density = trainer.densify_and_prune
                def density(*args, **kwargs):
                    fractions.append((trainer.current_iteration, args[1]))
                    return original_density(*args, **kwargs)
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(trainer, 'densify_and_prune', side_effect=density), \
                     patch.object(trainer, 'reset_opacity', wraps=trainer.reset_opacity) as reset, \
                     patch.object(trainer, 'propagate_normals', wraps=trainer.propagate_normals) as propagate:
                    run_training(trainer, config, 1, [0])
                self.assertEqual(reset.call_count, 1)
                self.assertEqual([c.args[0] for c in propagate.call_args_list], [1, 2])
                self.assertEqual([i for i, f in fractions if f > 0], [1, 2])
                self.assertEqual([i for i, f in fractions if f == 0], [3, 4, 5] if prune_interval else [])
                self.assertEqual(trainer.optimizers.opacity_optimizer.lr, trainer.learning_rates['opacity'])

    def test_invalid_refinement_settings_are_rejected(self):
        for key, value in [('start_iteration', True), ('start_iteration', 0), ('start_iteration', 2001),
                           ('prune_interval', -1), ('prune_interval', True)]:
            config = deepcopy(SurfelTrainingConfig.DEFAULTS)
            config['surfel']['refinement'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                SurfelTrainingConfig(config).validate()

    def test_gaussian_loop_leaves_final_trained_topology_and_opacity_intact(self):
        trainer = base.WarpImageTrainer(
            np.full((1, 16, 16, 3), .3, np.float32), np.eye(4, dtype=np.float32)[None],
            16., 1., 8, 1, DEVICE, 7, init_scene=reference.one_splat_scene(),
            training_options={'sparse': {'enabled': False}, 'multiview_adc': {'enabled': False}})
        config = base.TrainingConfig(base.TrainingConfig._merge(base.TrainingConfig.DEFAULTS, {
            'model': {'width': 16, 'height': 16, 'capacity': 8, 'initial_splats': 1},
            'training': {'iterations': 4},
            'densification': {'interval': 1, 'fraction': .5, 'opacity_reset_interval': 2},
            'multiview_adc': {'enabled': False}, 'convergence': {'enabled': False},
            'reporting': {'eval_every': 1, 'snapshot_every': 4, 'log_every': 1, 'save_ply': False},
        }))
        config.validate()
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(trainer, 'densify_and_prune', wraps=trainer.densify_and_prune) as density, \
             patch.object(trainer, 'reset_opacity', wraps=trainer.reset_opacity) as reset:
            base.run_training(trainer, config, trainer.targets_host, [0], self.directory, self.directory / 'model.png')
        self.assertEqual(density.call_count, 3)
        self.assertEqual(reset.call_count, 1)


if __name__ == '__main__':
    unittest.main()
