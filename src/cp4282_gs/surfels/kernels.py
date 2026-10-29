"""G-buffer rasterization, deferred environment shading, and explicit raster adjoints."""

import warp as wp

from .geometry import alpha_at_pixel, base, camera_position, normal_objective, surfel_normal

# RGB (0:3), normal sum (3:6), camera-z depth sum (6), coverage (7), reflection sum (8).
GBuffer = wp.types.vector(length=9, dtype=wp.float32)


# Forward adjoints keep nested wp.grad() dependencies available to the explicit backward.
# Only deferred_shader is recorded on the training tape.
@wp.kernel
def build_attributes(means: wp.array(dtype=wp.vec3), quaternions: wp.array(dtype=wp.vec4),
                     colours: wp.array(dtype=wp.vec3), reflection_logits: wp.array(dtype=wp.float32),
                     cameras: wp.array(dtype=wp.mat44), view: int, attributes: wp.array(dtype=GBuffer)):
    i = wp.tid()
    camera = cameras[view]
    normal = surfel_normal(quaternions[i], means[i], camera)
    point = camera * wp.vec4(means[i][0], means[i][1], means[i][2], 1.0)
    colour = colours[i]
    strength = 1.0 / (1.0 + wp.exp(-reflection_logits[i]))
    attributes[i] = GBuffer(wp.clamp(colour[0], 0.0, 1.0), wp.clamp(colour[1], 0.0, 1.0),
                            wp.clamp(colour[2], 0.0, 1.0), normal[0], normal[1], normal[2],
                            point[2], 1.0, strength)


@wp.kernel
def raster_forward(means: wp.array(dtype=wp.vec3), log_scales: wp.array(dtype=wp.vec3),
                   quaternions: wp.array(dtype=wp.vec4), opacity: wp.array(dtype=wp.float32),
                   attributes: wp.array(dtype=GBuffer), cameras: wp.array(dtype=wp.mat44), view: int,
                   offsets: wp.array(dtype=wp.int32), pairs: wp.array(dtype=wp.uint64),
                   width: int, height: int, tiles_x: int, focal: float,
                   compact: int, beta: float, alpha_min: float, buffers: wp.array(dtype=GBuffer)):
    pixel = wp.tid()
    px, py = pixel % width, pixel // width
    record = (py // base.TILE) * tiles_x + px // base.TILE
    value = GBuffer(0.0)
    transmittance = float(1.0)
    for entry in range(offsets[record], offsets[record + 1]):
        i = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        alpha = alpha_at_pixel(means[i], log_scales[i], quaternions[i], opacity[i], cameras[view],
                               float(px) + 0.5, float(py) + 0.5, float(width), float(height), focal, compact, beta, alpha_min)
        if alpha > 0.0:
            value += transmittance * alpha * attributes[i]
            transmittance *= 1.0 - alpha
            if transmittance < base.TRANSMITTANCE_CUTOFF:
                break
    buffers[pixel] = value


@wp.func
def sample_environment(direction: wp.vec3, environment: wp.array(dtype=wp.vec3), width: int, height: int):
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
    value = (1.0 - ty) * top + ty * bottom
    return wp.vec3(1.0 / (1.0 + wp.exp(-value[0])), 1.0 / (1.0 + wp.exp(-value[1])),
                   1.0 / (1.0 + wp.exp(-value[2])))


@wp.kernel
def deferred_shader(buffers: wp.array(dtype=GBuffer), environment: wp.array(dtype=wp.vec3),
                    env_width: int, env_height: int, cameras: wp.array(dtype=wp.mat44), view: int,
                    width: int, height: int, focal: float, enabled: int, background: wp.vec3,
                    image: wp.array(dtype=wp.vec3)):
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
        environment_colour = sample_environment(reflected, environment, env_width, env_height)
        strength = wp.clamp(value[8] / coverage, 0.0, 1.0)
        result = (1.0 - strength) * colour + strength * coverage * environment_colour + (1.0 - coverage) * background
    image[pixel] = result


@wp.kernel(enable_backward=False)
def mse_loss(image: wp.array(dtype=wp.vec3), targets: wp.array(dtype=wp.vec3), view: int, pixels: int,
             loss: wp.array(dtype=wp.float32), pixel_gradient: wp.array(dtype=wp.vec3)):
    pixel = wp.tid()
    delta = image[pixel] - targets[view * pixels + pixel]
    wp.atomic_add(loss, 0, wp.dot(delta, delta) / float(3 * pixels))
    pixel_gradient[pixel] = 2.0 * delta / float(3 * pixels)


@wp.kernel(enable_backward=False)
def raster_backward(means: wp.array(dtype=wp.vec3), log_scales: wp.array(dtype=wp.vec3),
                    quaternions: wp.array(dtype=wp.vec4), opacity: wp.array(dtype=wp.float32),
                    colours: wp.array(dtype=wp.vec3), attributes: wp.array(dtype=GBuffer),
                    cameras: wp.array(dtype=wp.mat44), view: int, offsets: wp.array(dtype=wp.int32),
                    pairs: wp.array(dtype=wp.uint64), width: int, height: int, tiles_x: int, focal: float,
                    compact: int, beta: float, alpha_min: float, buffers: wp.array(dtype=GBuffer),
                    buffer_gradient: wp.array(dtype=GBuffer), mean_grad: wp.array(dtype=wp.float32),
                    scale_grad: wp.array(dtype=wp.float32), quaternion_grad: wp.array(dtype=wp.float32),
                    opacity_grad: wp.array(dtype=wp.float32), colour_grad: wp.array(dtype=wp.float32),
                    normal_grad: wp.array(dtype=wp.float32), reflection_grad: wp.array(dtype=wp.float32)):
    pixel = wp.tid()
    px, py = pixel % width, pixel // width
    record = (py // base.TILE) * tiles_x + px // base.TILE
    final, gradient = buffers[pixel], buffer_gradient[pixel]
    prefix = GBuffer(0.0)
    transmittance = float(1.0)
    camera = cameras[view]
    for entry in range(offsets[record], offsets[record + 1]):
        i = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        alpha = alpha_at_pixel(means[i], log_scales[i], quaternions[i], opacity[i], camera,
                               float(px) + 0.5, float(py) + 0.5, float(width), float(height), focal, compact, beta, alpha_min)
        if alpha > 0.0:
            weight = transmittance * alpha
            next_transmittance = transmittance * (1.0 - alpha)
            behind = (final - prefix - weight * attributes[i]) / wp.max(next_transmittance, 1.0e-8)
            alpha_gradient = transmittance * wp.dot(gradient, attributes[i] - behind)
            dm, ds, dq, do, dc, dpx, dpy, dw, dh, df, de, db, da = wp.grad(alpha_at_pixel)(
                means[i], log_scales[i], quaternions[i], opacity[i], camera,
                float(px) + 0.5, float(py) + 0.5, float(width), float(height), focal, compact, beta, alpha_min)
            for axis in range(3):
                depth_gradient = weight * gradient[6] * camera[2, axis]
                wp.atomic_add(mean_grad, 3 * i + axis, alpha_gradient * dm[axis] + depth_gradient)
                wp.atomic_add(scale_grad, 3 * i + axis, alpha_gradient * ds[axis])
                wp.atomic_add(normal_grad, 3 * i + axis, weight * gradient[3 + axis])
                if colours[i][axis] > 0.0 and colours[i][axis] < 1.0:
                    wp.atomic_add(colour_grad, 3 * i + axis, weight * gradient[axis])
            for axis in range(4):
                wp.atomic_add(quaternion_grad, 4 * i + axis, alpha_gradient * dq[axis])
            wp.atomic_add(opacity_grad, i, alpha_gradient * do)
            strength = attributes[i][8]
            wp.atomic_add(reflection_grad, i, weight * gradient[8] * strength * (1.0 - strength))
            prefix += weight * attributes[i]
            transmittance = next_transmittance
            if transmittance < base.TRANSMITTANCE_CUTOFF:
                break


@wp.kernel(enable_backward=False)
def normal_backward(means: wp.array(dtype=wp.vec3), quaternions: wp.array(dtype=wp.vec4),
                    cameras: wp.array(dtype=wp.mat44), view: int, normal_grad: wp.array(dtype=wp.float32),
                    mean_grad: wp.array(dtype=wp.float32), quaternion_grad: wp.array(dtype=wp.float32)):
    i = wp.tid()
    gradient = wp.vec3(normal_grad[3 * i], normal_grad[3 * i + 1], normal_grad[3 * i + 2])
    if wp.dot(gradient, gradient) > 0.0:
        dq, dm, dc, dg = wp.grad(normal_objective)(quaternions[i], means[i], cameras[view], gradient)
        for axis in range(4):
            quaternion_grad[4 * i + axis] += dq[axis]
        for axis in range(3):
            mean_grad[3 * i + axis] += dm[axis]


@wp.kernel(enable_backward=False)
def clamp_logits(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = wp.clamp(values[i], -12.0, 12.0)


@wp.kernel(enable_backward=False)
def clamp_environment(values: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    v = values[i]
    values[i] = wp.vec3(wp.clamp(v[0], -12.0, 12.0), wp.clamp(v[1], -12.0, 12.0), wp.clamp(v[2], -12.0, 12.0))
