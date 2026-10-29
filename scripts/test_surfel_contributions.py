"""Visibility conservation, opaque occlusion and foreground/background partition."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from gaussian_set import GaussianSet
from cp4282_gs.surfels.trainer import SurfelTrainer, GaussianRGBControl
from measure_surfel_contributions import measure

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


def trainer(cls, opacity):
    scene = GaussianSet(np.array([[0, 0, z] for z in (1., 1.3, 1.6, 1.9)], np.float32),
        np.full((4, 3), .1, np.float32), np.tile([1., 0, 0, 0], (4, 1)).astype(np.float32),
        np.full(4, opacity, np.float32), np.ones((4, 3), np.float32))
    return cls(np.ones((1, 1, 1, 3), np.float32), np.eye(4, dtype=np.float32)[None],
        1., 1., 4, 4, DEVICE, 7, init_scene=scene,
        training_options={'compact_box': {'enabled': False}, 'sparse': {'enabled': False}})


class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/contribution-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def test_weights_match_front_to_back_compositing_for_gaussians_and_surfels(self):
        for cls in (GaussianRGBControl, SurfelTrainer):
            model = trainer(cls, .8)
            values, coverage = measure(model, np.full((1, 1, 1), .25, np.float32), is_surfel=cls is SurfelTrainer)
            expected = .8 * .2 ** np.arange(4)
            np.testing.assert_allclose(values[:, 0], expected, atol=1e-6)
            np.testing.assert_allclose(values[:, 1], .25 * expected, atol=1e-6)
            np.testing.assert_allclose(values[:, 2], .75 * expected, atol=1e-6)
            self.assertAlmostEqual(coverage, 1 - .2 ** 4, places=6)

    def test_opaque_occluded_splat_can_have_zero_visibility(self):
        model = trainer(SurfelTrainer, .999)
        values, coverage = measure(model, np.ones((1, 1, 1), np.float32))
        self.assertEqual(float(values[-1, 0]), 0.)
        self.assertGreater(float(model.gaussian_set().opacities[-1]), .99)
        self.assertGreater(coverage, .999)

    def test_white_background_splats_can_have_high_total_but_zero_foreground_weight(self):
        model = trainer(SurfelTrainer, .8)
        image, _ = model.render_single_view_image(0)
        np.testing.assert_allclose(image, 1., atol=1e-6)
        values, _ = measure(model, np.zeros((1, 1, 1), np.float32))
        self.assertGreater(float(values[0, 0]), .79)
        np.testing.assert_array_equal(values[:, 1], 0.)
        np.testing.assert_array_equal(values[:, 2], values[:, 0])


if __name__ == '__main__':
    unittest.main()
