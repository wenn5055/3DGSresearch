"""Analytic screen-footprint checks without rendering or GPU work."""

from types import SimpleNamespace
import unittest

import numpy as np

from diagnose_sphere_footprints import footprint, base


class FootprintTests(unittest.TestCase):
    def test_frontal_disc_has_expected_perspective_sigma(self):
        scene = SimpleNamespace(means=np.array([[0., 0., 2.]]),
                                rotations=np.array([[1., 0., 0., 0.]]),
                                scales=np.array([[.1, .2, 0.]]), opacities=np.array([.8]))
        sigma, diameter = footprint(scene, np.eye(4), 100., 800, 800, {'enabled': False})
        expected = np.sqrt((100*.2/2)**2 + base.FILTER_VARIANCE)
        np.testing.assert_allclose(sigma, [expected], atol=1e-6)
        np.testing.assert_allclose(diameter, [2*expected*np.sqrt(base.SUPPORT_RADIUS_SQUARED)], atol=1e-6)

    def test_behind_camera_and_offscreen_centers_are_excluded(self):
        scene = SimpleNamespace(means=np.array([[0., 0., 2.], [0., 0., -2.], [100., 0., 2.]]),
                                rotations=np.tile([1., 0., 0., 0.], (3, 1)),
                                scales=np.full((3, 3), .1), opacities=np.full(3, .8))
        sigma, diameter = footprint(scene, np.eye(4), 100., 800, 800,
                                    {'enabled': True, 'beta': .5, 'alpha_min': 1/255})
        self.assertEqual(len(sigma), 1)
        self.assertEqual(len(diameter), 1)


if __name__ == '__main__':
    unittest.main()
