"""Structural changes must not corrupt sampling or population bookkeeping."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_foreground_pruning import DEVICE, ROOT, fixture, GaussianSet, SurfelTrainer, wp
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels.workflow import base, run_training


class WorkflowBookkeepingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def run_fixture(self, directory, *, interval, foreground=False, iterations=4, adc=False):
        cameras, alpha, means, scales, rotations, settings = fixture()
        settings.update(enabled=foreground, start_iteration=1, interval=1)
        initial = GaussianSet(means, scales, rotations, np.full(2, .9, np.float32),
                              np.ones((2, 3), np.float32))
        values = SurfelTrainingConfig._merge(deepcopy(SurfelTrainingConfig.DEFAULTS), {
            'paths': {'output': str(Path(directory) / 'model.png')},
            'model': {'width': 32, 'height': 32, 'capacity': 2, 'initial_splats': 2},
            'training': {'iterations': iterations},
            'densification': {'interval': interval, 'fraction': .2},
            'multiview_adc': {'enabled': adc}, 'convergence': {'enabled': False},
            'surfel': {'foreground_pruning': settings},
            'reporting': {'eval_every': 1, 'snapshot_every': iterations, 'log_every': 1}})
        config = SurfelTrainingConfig(values)
        config.validate()
        trainer = SurfelTrainer(np.ones((3, 32, 32, 3), np.float32), cameras, 32., 1.,
            2, 2, DEVICE, 7, init_scene=initial, training_view_count=3, training_alpha=alpha,
            surfel_options=values['surfel'], training_options={'multiview_adc': values['multiview_adc']})
        training_views = []
        original_step = trainer.step

        def record_step(view_ids, **kwargs):
            training_views.extend(view_ids)
            return original_step(view_ids, **kwargs)

        trainer.step = record_step
        # Keep density behavior inert, isolating scheduling from pruning quality.
        trainer.densify_and_prune = lambda *args, **kwargs: []
        with contextlib.redirect_stdout(io.StringIO()):
            run_training(trainer, config, 3, [0])
        return training_views

    def test_growth_tracker_observes_population_after_foreground_deletion(self):
        counts = []

        class GrowthSpy:
            def __init__(self, *args):
                pass

            def update(self, count):
                counts.append(count)
                return False

        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            with patch.object(base, 'GrowthPlateauTracker', GrowthSpy):
                self.run_fixture(directory, interval=1, foreground=True)
        self.assertEqual(counts, [2, 1, 1])

    def test_structure_schedule_preserves_training_camera_sequence(self):
        sequences = []
        for interval in (0, 2):
            with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
                sequences.append(self.run_fixture(directory, interval=interval, iterations=8, adc=True))
        expected = np.random.default_rng(7).integers(3, size=8).tolist()
        self.assertEqual(sequences[0], expected)
        self.assertEqual(sequences[1], expected)


if __name__ == '__main__':
    unittest.main()
