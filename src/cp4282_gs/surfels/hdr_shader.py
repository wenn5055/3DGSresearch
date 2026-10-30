"""Optional linear-light deferred shading with a learned angular response.

The 16 shared knots are learned from training RGB. No reference material, mesh,
normal or lighting parameters enter this model. Legacy shading remains separate.
"""

import warp as wp
import numpy as np

from .kernels import GBuffer


def effective_blend(buffers, camera, focal, knots):
    """CPU diagnostic matching the optional shader's angular blend."""
    height, width = buffers.shape[:2]
    y, x = np.mgrid[:height, :width]
    ray = np.stack(((x + .5 - width * .5) / focal, (y + .5 - height * .5) / focal,
                    np.ones_like(x)), axis=-1) @ np.asarray(camera)[:3, :3]
    ray /= np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), 1e-12)
    nsum = buffers[..., 3:6]
    length = np.linalg.norm(nsum, axis=-1)
    normal = nsum / np.sqrt(length[..., None] ** 2 + 1e-12)
    cosine = np.clip(np.abs(np.sum(ray * normal, axis=-1)), 0, 1)
    gain = np.interp(cosine, np.linspace(0, 1, 16), knots)
    strength = np.clip(buffers[..., 8] / np.maximum(buffers[..., 7], 1e-8), 1e-6, 1 - 1e-6)
    logit = np.log(strength / (1 - strength)) + gain
    blend = 1 / (1 + np.exp(-np.clip(logit, -60, 60)))
    return np.where((buffers[..., 7] > 1e-6) & (length > 1e-6), blend, 0)


@wp.func
def softplus(x: float):
    return wp.max(x, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(x)))


@wp.func
def to_linear(x: float):
    x = wp.clamp(x, 0.0, 1.0)
    if x <= 0.04045:
        return x / 12.92
    return wp.pow((x + 0.055) / 1.055, 2.4)


@wp.func
def to_display(x: float):
    x = wp.clamp(x, 0.0, 1.0)
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * wp.pow(x, 1.0 / 2.4) - 0.055


@wp.func
def environment_radiance(direction: wp.vec3, environment: wp.array(dtype=wp.vec3), width: int, height: int):
    d = direction / wp.sqrt(wp.dot(direction, direction) + 1.0e-12)
    u = wp.atan2(d[0], d[2]) / (2.0 * wp.pi) + 0.5
    v = wp.acos(wp.clamp(d[1], -0.999999, 0.999999)) / wp.pi
    x = u * float(width) - 0.5
    y = wp.clamp(v * float(height) - 0.5, 0.0, float(height - 1))
    ix, iy = int(wp.floor(x)), int(wp.floor(y))
    tx, ty = x - float(ix), y - float(iy)
    x0, x1 = ((ix % width) + width) % width, (((ix + 1) % width) + width) % width
    y1 = wp.min(iy + 1, height - 1)
    top = (1.0 - tx) * environment[iy * width + x0] + tx * environment[iy * width + x1]
    bottom = (1.0 - tx) * environment[y1 * width + x0] + tx * environment[y1 * width + x1]
    raw = (1.0 - ty) * top + ty * bottom
    return wp.vec3(softplus(raw[0]), softplus(raw[1]), softplus(raw[2]))


@wp.kernel
def deferred_hdr_shader(buffers: wp.array(dtype=GBuffer), environment: wp.array(dtype=wp.vec3),
                        env_width: int, env_height: int, cameras: wp.array(dtype=wp.mat44), view: int,
                        width: int, height: int, focal: float, enabled: int, background: wp.vec3,
                        angular_gain: wp.array(dtype=wp.float32), image: wp.array(dtype=wp.vec3)):
    pixel = wp.tid()
    value = buffers[pixel]
    coverage = value[7]
    colour = wp.vec3(value[0], value[1], value[2])
    result = colour + (1.0 - coverage) * background
    nsum = wp.vec3(value[3], value[4], value[5])
    if enabled != 0 and coverage > 1.0e-6 and wp.dot(nsum, nsum) > 1.0e-12:
        normal = nsum / wp.sqrt(wp.dot(nsum, nsum) + 1.0e-12)
        ray = wp.vec3((float(pixel % width) + 0.5 - float(width) * 0.5) / focal,
                      (float(pixel // width) + 0.5 - float(height) * 0.5) / focal, 1.0)
        camera = cameras[view]
        world_ray = wp.vec3(wp.dot(wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0]), ray),
                            wp.dot(wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1]), ray),
                            wp.dot(wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2]), ray))
        world_ray /= wp.sqrt(wp.dot(world_ray, world_ray) + 1.0e-12)
        reflected = world_ray - 2.0 * wp.dot(world_ray, normal) * normal
        light = environment_radiance(reflected, environment, env_width, env_height)
        coordinate = wp.clamp(wp.abs(wp.dot(world_ray, normal)), 0.0, 1.0) * 15.0
        index = wp.min(int(wp.floor(coordinate)), 14)
        fraction = coordinate - float(index)
        gain = (1.0 - fraction) * angular_gain[index] + fraction * angular_gain[index + 1]
        strength = wp.clamp(value[8] / coverage, 1.0e-6, 1.0 - 1.0e-6)
        logit = wp.log(strength / (1.0 - strength)) + gain
        strength = 1.0 / (1.0 + wp.exp(-logit))
        base = colour / coverage
        linear_base = wp.vec3(to_linear(base[0]), to_linear(base[1]), to_linear(base[2]))
        linear = (1.0 - strength) * linear_base + strength * light
        displayed = wp.vec3(to_display(linear[0]), to_display(linear[1]), to_display(linear[2]))
        result = coverage * displayed + (1.0 - coverage) * background
    image[pixel] = result


@wp.kernel(enable_backward=False)
def clamp_hdr_environment(values: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    value = values[i]
    values[i] = wp.vec3(wp.clamp(value[0], -16.0, 100.0), wp.clamp(value[1], -16.0, 100.0),
                        wp.clamp(value[2], -16.0, 100.0))
