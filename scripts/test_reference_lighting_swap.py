"""Numerical and isolation checks for the evaluation-only lighting swap."""

from types import SimpleNamespace
import unittest
import numpy as np

from evaluate_reference_lighting import inverse_softplus, replace_environment, fixed_parameter_hashes


class Array:
    def __init__(self, value):
        self.value = np.asarray(value).copy()
    def numpy(self):
        return self.value.copy()
    def assign(self, value):
        self.value = np.asarray(value).copy()


class LightingSwapTests(unittest.TestCase):
    def test_inverse_softplus_preserves_linear_radiance(self):
        values = np.array([0., 1e-8, .001, .1, 1., 3.8, 100.], np.float32)
        logits = inverse_softplus(values)
        self.assertTrue(np.isfinite(logits).all())
        np.testing.assert_allclose(np.logaddexp(0, logits), np.maximum(values, 1e-8), rtol=3e-6, atol=1e-8)

    def test_only_environment_changes_and_bad_shape_is_rejected(self):
        trainer = SimpleNamespace(hdr_angular=True, env_width=4, env_height=2,
                                  environment=Array(np.zeros((8, 3))))
        for name in ('means', 'log_scales', 'quaternions', 'opacity', 'color', 'reflection_logits', 'angular_gain'):
            setattr(trainer, name, Array(np.arange(12)))
        initial = fixed_parameter_hashes(trainer)
        replace_environment(trainer, np.full((2, 4, 3), .5))
        self.assertEqual(initial, fixed_parameter_hashes(trainer))
        np.testing.assert_allclose(np.logaddexp(0, trainer.environment.numpy()), .5, rtol=1e-6)
        previous = trainer.environment.numpy()
        with self.assertRaises(ValueError):
            replace_environment(trainer, np.ones((3, 4, 3)))
        np.testing.assert_array_equal(previous, trainer.environment.numpy())


if __name__ == '__main__':
    unittest.main()
