"""Analytic coordinate checks for the diagnostic, without Blender or training."""

import unittest
import numpy as np

from compare_environment import blender_uv, renderer_directions, sample_bottom_up


class EnvironmentTests(unittest.TestCase):
    def test_blender_cardinal_directions(self):
        d = np.array([[1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
        u, v = blender_uv(d)
        np.testing.assert_allclose(u[:3], [.5, .25, .75])
        np.testing.assert_allclose(v, [.5, .5, .5, 1., 0.])

    def test_renderer_coordinate_round_trip(self):
        d = renderer_directions(64, 32)
        y, x = np.mgrid[:32, :64]
        np.testing.assert_allclose(np.arccos(np.clip(d[..., 1], -1, 1))/np.pi, (y+.5)/32, atol=1e-12)
        np.testing.assert_allclose(np.arctan2(d[..., 0], d[..., 2])/(2*np.pi)+.5, (x+.5)/64, atol=1e-12)

    def test_bottom_up_pixels_and_horizontal_wrap(self):
        image = np.arange(12).reshape(2, 2, 3).astype(float)
        np.testing.assert_allclose(sample_bottom_up(image, np.array([.25, .75]), np.array([.25, .75])), image[[0, 1], [0, 1]])
        np.testing.assert_allclose(sample_bottom_up(image, np.array([0., 1.]), np.array([.25, .25])), [image[0].mean(axis=0)]*2)


if __name__ == '__main__':
    unittest.main()
