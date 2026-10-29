"""Opaque background rejection, silhouette safety, training-only data and lifecycle."""

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
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src')]
from gaussian_set import GaussianSet
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels.foreground_pruning import ForegroundPruner
from cp4282_gs.surfels.trainer import SurfelTrainer, GaussianRGBControl
from cp4282_gs.surfels.workflow import run_training

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


def fixture():
    cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    cameras[:, :2, 3] = [[.06, 0], [0, .06], [-.06, 0]]
    yy, xx = np.mgrid[:32, :32]
    alpha = []
    for camera in cameras:
        center = camera[:2, 3] * 32 / 1.2 + 16
        alpha.append(((xx + .5 - center[0]) ** 2 + (yy + .5 - center[1]) ** 2 <= 3 ** 2).astype(np.float32))
    means = np.array([[0, 0, 1.2], [.30, .30, 1.2]], np.float32)
    rotations = np.tile([1., 0, 0, 0], (2, 1)).astype(np.float32)
    scales = np.full((2, 3), .012, np.float32)
    settings = deepcopy(SurfelTrainingConfig.DEFAULTS['surfel']['foreground_pruning'])
    settings.update(enabled=True, views=3, min_background_views=3, margin_pixels=1)
    return cameras, np.asarray(alpha), means, scales, rotations, settings


class ForegroundPruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def test_multiview_background_and_persistence_keep_white_foreground(self):
        cameras, alpha, means, scales, rotations, settings = fixture()
        pruner = ForegroundPruner(alpha, cameras, 32, 2, settings)
        background, foreground, observed = pruner.votes(means, np.log(scales), rotations, np.ones(2, bool))
        np.testing.assert_array_equal(background, [0, 3])
        np.testing.assert_array_equal(foreground, [3, 0])
        np.testing.assert_array_equal(observed, [3, 3])
        first, _ = pruner.update(means, np.log(scales), rotations, np.ones(2, bool))
        second, event = pruner.update(means, np.log(scales), rotations, np.ones(2, bool))
        np.testing.assert_array_equal(first, [False, False])
        np.testing.assert_array_equal(second, [False, True])
        self.assertEqual(event['foreground_pruned'], 1)

    def test_partial_foreground_support_out_of_frame_and_behind_camera_are_retained(self):
        cameras, alpha, means, scales, rotations, settings = fixture()
        for point, size in (([0, 0, 1.2], .2), ([9, 9, 1.2], .012), ([0, 0, -1.2], .012)):
            means[1] = point
            scales[1] = size
            pruner = ForegroundPruner(alpha, cameras, 32, 2, settings)
            for _ in range(3):
                prune, _ = pruner.update(means, np.log(scales), rotations, np.ones(2, bool))
            self.assertFalse(prune.any())

    def test_soft_alpha_protects_support_and_recycled_children_reset_evidence(self):
        cameras, alpha, means, scales, rotations, settings = fixture()
        pruner = ForegroundPruner(alpha, cameras, 32, 2, settings)
        pruner.update(means, np.log(scales), rotations, np.ones(2, bool))
        pruner.reset_children([1])
        prune, _ = pruner.update(means, np.log(scales), rotations, np.ones(2, bool))
        self.assertFalse(prune.any())
        alpha[:, 20:31, 20:31] = .1
        protected = ForegroundPruner(alpha, cameras, 32, 2, settings)
        for _ in range(2):
            prune, _ = protected.update(means, np.log(scales), rotations, np.ones(2, bool))
        self.assertFalse(prune.any())

    def test_gaussian_and_surfel_trainers_prune_opaque_white_background(self):
        cameras, alpha, means, scales, rotations, settings = fixture()
        initial = GaussianSet(means, scales, rotations, np.full(2, .9, np.float32), np.ones((2, 3), np.float32))
        targets = np.ones((3, 32, 32, 3), np.float32)
        for cls in (GaussianRGBControl, SurfelTrainer):
            trainer = cls(targets, cameras, 32., 1., 2, 2, DEVICE, 7, init_scene=initial,
                training_view_count=3, training_alpha=alpha,
                surfel_options={'foreground_pruning': settings},
                training_options={'sparse': {'enabled': False}})
            trainer.prune_foreground()
            event = trainer.prune_foreground()
            np.testing.assert_array_equal(trainer.active, [True, False])
            np.testing.assert_array_equal(trainer.active_device.numpy(), [1, 0])
            self.assertEqual(event['foreground_pruned'], 1)

    def test_foreground_pruning_continues_in_quiet_refinement_without_growth(self):
        cameras, alpha, means, scales, rotations, settings = fixture()
        settings.update(start_iteration=1, interval=1)
        initial = GaussianSet(means, scales, rotations, np.full(2, .9, np.float32), np.ones((2, 3), np.float32))
        trainer = SurfelTrainer(np.ones((3, 32, 32, 3), np.float32), cameras, 32., 1., 2, 2,
            DEVICE, 7, init_scene=initial, training_view_count=3, training_alpha=alpha,
            surfel_options={'foreground_pruning': settings},
            training_options={'multiview_adc': {'enabled': False}})
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            values = SurfelTrainingConfig._merge(SurfelTrainingConfig.DEFAULTS, {
                'paths': {'output': str(Path(directory) / 'model.png')},
                'model': {'width': 32, 'height': 32, 'capacity': 2, 'initial_splats': 2},
                'training': {'iterations': 4}, 'densification': {'interval': 0},
                'multiview_adc': {'enabled': False}, 'convergence': {'enabled': False},
                'surfel': {'foreground_pruning': settings,
                           'refinement': {'start_iteration': 1, 'prune_interval': 0}},
                'reporting': {'eval_every': 1, 'snapshot_every': 4, 'log_every': 1}})
            config = SurfelTrainingConfig(values)
            config.validate()
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                run_training(trainer, config, 3, [0])
            self.assertEqual(int(trainer.active.sum()), 1)
            self.assertIn('foreground pruning: candidates=1, pruned=1', log.getvalue())

    def test_missing_or_held_out_alpha_duplicate_cameras_and_invalid_threshold_are_rejected(self):
        cameras, alpha, means, scales, rotations, settings = fixture()
        with self.assertRaises(ValueError):
            ForegroundPruner(alpha[:2], cameras, 32, 2, settings)
        with self.assertRaises(ValueError):
            ForegroundPruner(alpha, np.repeat(cameras[:1], 3, axis=0), 32, 2, settings)
        initial = GaussianSet(means, scales, rotations, np.full(2, .9, np.float32), np.ones((2, 3), np.float32))
        with self.assertRaises(ValueError):
            SurfelTrainer(np.ones((4, 32, 32, 3), np.float32), np.concatenate((cameras, cameras[:1])),
                32., 1., 2, 2, DEVICE, 7, init_scene=initial, training_view_count=3,
                training_alpha=np.concatenate((alpha, alpha[:1])), surfel_options={'foreground_pruning': settings})
        values = deepcopy(SurfelTrainingConfig.DEFAULTS)
        values['surfel']['foreground_pruning']['min_background_views'] = 1
        with self.assertRaises(ValueError):
            SurfelTrainingConfig(values).validate()


if __name__ == '__main__':
    unittest.main()
