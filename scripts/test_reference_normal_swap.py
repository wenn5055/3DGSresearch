"""Check that an evaluation normal intervention cannot alter coverage or materials."""

import unittest
import numpy as np
from evaluate_reference_normals import substitute_normals


class NormalSwapTests(unittest.TestCase):
    def test_only_normal_direction_changes_and_invalid_pixels_stay_fixed(self):
        original = np.arange(36, dtype=np.float32).reshape(2, 2, 9)/36
        original[..., 3:6] = [0, 0, .7]
        original[1, 0, 7] = 0  # Model hole must not be filled.
        original[1, 1, 3:6] = 0  # No learned normal must not enable specular shading.
        normals = np.zeros((2, 2, 3), np.float32)
        normals[..., 0] = 1
        ids = np.ones((2, 2), np.int16)
        ids[0, 1] = 0  # No reference mesh hit must retain its original raster.
        swapped, valid = substitute_normals(original, normals, ids)
        np.testing.assert_array_equal(valid, [[True, False], [False, False]])
        np.testing.assert_array_equal(swapped[0, 0, 3:6], np.array([.7, 0, 0], np.float32))
        np.testing.assert_array_equal(swapped[..., [0, 1, 2, 6, 7, 8]], original[..., [0, 1, 2, 6, 7, 8]])
        np.testing.assert_array_equal(swapped[~valid], original[~valid])
        np.testing.assert_array_equal(original[0, 0, 3:6], [0, 0, np.float32(.7)])

    def test_nonunit_reference_normals_rejected(self):
        with self.assertRaises(ValueError):
            substitute_normals(np.ones((1, 1, 9)), np.ones((1, 1, 3)), np.ones((1, 1)))


if __name__ == '__main__':
    unittest.main()
