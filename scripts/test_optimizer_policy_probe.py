"""The diagnostic must manipulate only the intended Adam histories."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import warp as wp
from warp.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]
from probe_optimizer_policy import base, optimizer_policy

DEVICE = os.environ.get('SURFEL_TEST_DEVICE', 'cpu')


class OptimizerPolicyProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = str(ROOT / 'outputs/surfel-test-warp-cache')
        wp.config.use_precompiled_headers = False
        wp.init()

    def bundle(self):
        bundle = object.__new__(base.SplatOptimizers)
        bundle.device = DEVICE
        bundle.optimizers = (Adam([wp.ones(4, dtype=wp.float32, device=DEVICE)]),
                             Adam([wp.ones(4, dtype=wp.vec3, device=DEVICE)]))
        bundle.quaternion_m = wp.ones(4, dtype=wp.vec4, device=DEVICE)
        bundle.quaternion_v = wp.ones(4, dtype=wp.vec4, device=DEVICE)
        bundle.quaternion_adam_t = 37
        for optimizer in bundle.optimizers:
            optimizer.t = 37
        arrays = [array for optimizer in bundle.optimizers for array in (*optimizer.m, *optimizer.v)]
        arrays += [bundle.quaternion_m, bundle.quaternion_v]
        for array in arrays:
            values = array.numpy()
            values[:] = np.arange(1, 5, dtype=np.float32).reshape((4,) + (1,) * (values.ndim - 1))
            array.assign(values)
        return bundle, arrays

    def test_inherit_fresh_and_legacy_histories_and_clocks(self):
        original = base.SplatOptimizers.inherit_children
        for policy in ('inherit', 'fresh_children', 'legacy_reset'):
            with self.subTest(policy=policy):
                bundle, arrays = self.bundle()
                before = [array.numpy().copy() for array in arrays]
                with optimizer_policy(policy):
                    bundle.inherit_children(np.array([0]), np.array([2]))
                self.assertIs(base.SplatOptimizers.inherit_children, original)
                for array, old in zip(arrays, before):
                    expected = old.copy()
                    if policy == 'legacy_reset':
                        expected[:] = 0
                    else:
                        expected[2] = old[0] if policy == 'inherit' else 0
                    np.testing.assert_array_equal(array.numpy(), expected)
                expected_clock = 0 if policy == 'legacy_reset' else 37
                self.assertEqual(bundle.quaternion_adam_t, expected_clock)
                self.assertTrue(all(optimizer.t == expected_clock for optimizer in bundle.optimizers))

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            with optimizer_policy('unknown'):
                pass


if __name__ == '__main__':
    unittest.main()
