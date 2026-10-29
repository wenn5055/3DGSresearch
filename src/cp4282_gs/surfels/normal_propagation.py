"""Normal-propagation interventions from 3DGS-DR section 3.3.

These are scheduled parameter updates, not an invented normal-loss equation.
Two tangent scales are enlarged because this renderer fixes local z to zero.
Paper: https://arxiv.org/html/2404.18454v1#S3.SS3
"""

import warp as wp


@wp.kernel(enable_backward=False)
def propagate(active: wp.array(dtype=wp.int32), opacity: wp.array(dtype=wp.float32),
              reflection: wp.array(dtype=wp.float32), scales: wp.array(dtype=wp.vec3),
              colour: wp.array(dtype=wp.vec3), seed: int, opacity_limit: float,
              reflection_limit: float, scale_change: float, max_scale: float,
              scale_threshold: float, colour_threshold: float, noise: float,
              counts: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if active[i] != 0:
        strength = 1.0 / (1.0 + wp.exp(-reflection[i]))
        opacity[i] = wp.max(opacity[i], opacity_limit)
        reflection[i] = wp.max(reflection[i], reflection_limit)
        if strength >= scale_threshold:
            value = scales[i]
            # Keep normal-axis thickness and orientation exactly unchanged.
            scales[i] = wp.vec3(wp.min(value[0] + scale_change, max_scale),
                                wp.min(value[1] + scale_change, max_scale), value[2])
            wp.atomic_add(counts, 0, 1)
        if strength <= colour_threshold:
            state = wp.rand_init(seed, i)
            value = colour[i]
            for axis in range(3):
                value[axis] = wp.clamp(value[axis] + noise * (2.0 * wp.randf(state) - 1.0), 0.0, 1.0)
            colour[i] = value
            wp.atomic_add(counts, 1, 1)


@wp.kernel(enable_backward=False)
def raise_reflection_floor(active: wp.array(dtype=wp.int32), reflection: wp.array(dtype=wp.float32),
                           limit: float):
    i = wp.tid()
    if active[i] != 0:
        reflection[i] = wp.max(reflection[i], limit)
