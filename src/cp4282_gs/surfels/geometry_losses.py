"""Image and mask losses, plus the optional historical depth-normal penalty.

The paper-aligned deferred preset uses L1 + D-SSIM and mask BCE. Its normal
gradients come from reflection shading; it does not enable normal_depth_loss.
"""

import json

import numpy as np
from PIL import Image
import warp as wp

from .geometry import base
from .kernels import GBuffer


@wp.kernel(enable_backward=False)
def l1_image_loss(image: wp.array(dtype=wp.vec3), targets: wp.array(dtype=wp.vec3),
                  view: int, pixels: int, compute_gradient: int,
                  loss: wp.array(dtype=wp.float32), gradient: wp.array(dtype=wp.vec3)):
    pixel = wp.tid()
    delta = image[pixel] - targets[view * pixels + pixel]
    normalization = 1.0 / float(3 * pixels)
    wp.atomic_add(loss, 0, (wp.abs(delta[0]) + wp.abs(delta[1]) + wp.abs(delta[2])) * normalization)
    if compute_gradient != 0:
        # Warp's sign(0) is +1; L1 uses the zero subgradient at an exact match.
        value = wp.vec3(0.0)
        for axis in range(3):
            if delta[axis] > 0.0:
                value[axis] = normalization
            elif delta[axis] < 0.0:
                value[axis] = -normalization
        gradient[pixel] = value


def load_training_alpha(data, width, height):
    """Read original alpha, never infer masks from white RGB or held-out images."""
    manifest = json.loads((data / 'transforms_train.json').read_text(encoding='utf-8'))
    masks = []
    for frame in manifest['frames']:
        path = base.resolve_frame_image(data, frame['file_path'])
        with Image.open(path) as image:
            if 'A' not in image.getbands() and 'transparency' not in image.info:
                raise ValueError(f'Geometry supervision requires source alpha masks: {path}')
            alpha = image.convert('RGBA').getchannel('A').resize((width, height), Image.Resampling.LANCZOS)
            masks.append(np.asarray(alpha, np.float32) / 255.0)
    return np.stack(masks)


@wp.func
def camera_point(buffers: wp.array(dtype=GBuffer), pixel: int, width: int, height: int, focal: float):
    value = buffers[pixel]
    depth = value[6] / wp.max(value[7], 1.0e-8)
    return depth * wp.vec3((float(pixel % width) + 0.5 - float(width) * 0.5) / focal,
                           (float(pixel // width) + 0.5 - float(height) * 0.5) / focal, 1.0)


@wp.kernel(enable_backward=False)
def select_normal_pixels(buffers: wp.array(dtype=GBuffer), alpha: wp.array(dtype=wp.float32),
                         view: int, width: int, height: int, focal: float, min_coverage: float,
                         foreground: float, max_jump: float,
                         valid: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.float32)):
    pixel = wp.tid()
    x, y = pixel % width, pixel // width
    valid[pixel] = 0
    if x > 0 and x < width - 1 and y > 0 and y < height - 1:
        centre = buffers[pixel]
        depth = centre[6] / wp.max(centre[7], 1.0e-8)
        good = int(depth > 0.0)
        normal = wp.vec3(centre[3], centre[4], centre[5])
        if wp.dot(normal, normal) < 1.0e-12:
            good = 0
        for index in range(5):
            neighbour = pixel
            if index == 1:
                neighbour = pixel - 1
            elif index == 2:
                neighbour = pixel + 1
            elif index == 3:
                neighbour = pixel - width
            elif index == 4:
                neighbour = pixel + width
            value = buffers[neighbour]
            z = value[6] / wp.max(value[7], 1.0e-8)
            if (value[7] < min_coverage or alpha[view * width * height + neighbour] < foreground
                    or z <= 0.0 or wp.abs(z - depth) > max_jump * depth):
                good = 0
        tangent_x = camera_point(buffers, pixel + 1, width, height, focal) - camera_point(buffers, pixel - 1, width, height, focal)
        tangent_y = camera_point(buffers, pixel + width, width, height, focal) - camera_point(buffers, pixel - width, width, height, focal)
        cross = wp.cross(tangent_x, tangent_y)
        if wp.dot(cross, cross) < 1.0e-20:
            good = 0
        if good != 0:
            valid[pixel] = 1
            wp.atomic_add(count, 0, 1.0)


@wp.kernel
def mask_loss(buffers: wp.array(dtype=GBuffer), alpha: wp.array(dtype=wp.float32), view: int,
              pixels: int, terms: wp.array(dtype=wp.float32)):
    pixel = wp.tid()
    # Linear epsilon contraction keeps endpoint gradients, unlike a hard clamp.
    probability = 1.0e-6 + (1.0 - 2.0e-6) * buffers[pixel][7]
    target = alpha[view * pixels + pixel]
    value = -target * wp.log(probability) - (1.0 - target) * wp.log(1.0 - probability)
    wp.atomic_add(terms, 0, value / float(pixels))


@wp.kernel
def normal_depth_loss(buffers: wp.array(dtype=GBuffer), cameras: wp.array(dtype=wp.mat44), view: int,
                      width: int, height: int, focal: float, valid: wp.array(dtype=wp.int32),
                      count: wp.array(dtype=wp.float32), terms: wp.array(dtype=wp.float32)):
    pixel = wp.tid()
    if valid[pixel] != 0:
        tangent_x = camera_point(buffers, pixel + 1, width, height, focal) - camera_point(buffers, pixel - 1, width, height, focal)
        tangent_y = camera_point(buffers, pixel + width, width, height, focal) - camera_point(buffers, pixel - width, width, height, focal)
        cross = wp.cross(tangent_x, tangent_y)
        depth_normal = cross / wp.sqrt(wp.dot(cross, cross) + 1.0e-20)
        # Camera-space normals point towards the observing camera, matching surfel_normal.
        if wp.dot(depth_normal, camera_point(buffers, pixel, width, height, focal)) > 0.0:
            depth_normal = -depth_normal
        value = buffers[pixel]
        normal = wp.vec3(value[3], value[4], value[5])
        normal /= wp.sqrt(wp.dot(normal, normal) + 1.0e-12)
        camera = cameras[view]
        camera_normal = wp.vec3(camera[0, 0] * normal[0] + camera[0, 1] * normal[1] + camera[0, 2] * normal[2],
                                camera[1, 0] * normal[0] + camera[1, 1] * normal[1] + camera[1, 2] * normal[2],
                                camera[2, 0] * normal[0] + camera[2, 1] * normal[1] + camera[2, 2] * normal[2])
        value_loss = 1.0 - wp.clamp(wp.dot(camera_normal, depth_normal), -1.0, 1.0)
        wp.atomic_add(terms, 1, value_loss / wp.max(count[0], 1.0))


@wp.kernel
def combine_terms(terms: wp.array(dtype=wp.float32), mask_weight: float, normal_weight: float,
                  total: wp.array(dtype=wp.float32)):
    total[0] = mask_weight * terms[0] + normal_weight * terms[1]


class GeometryLoss:
    def __init__(self, width, height, device, options, alpha):
        self.width, self.height, self.device = width, height, device
        self.options = options
        self.alpha = wp.array(np.asarray(alpha, np.float32).reshape(-1), dtype=wp.float32, device=device)
        self.views = len(alpha)
        self.valid = wp.zeros(width * height, dtype=wp.int32, device=device)
        self.count = wp.zeros(1, dtype=wp.float32, device=device)
        self.terms = wp.zeros(2, dtype=wp.float32, device=device, requires_grad=True)
        self.total = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    def normal_weight(self, iteration):
        elapsed = iteration - self.options['normal_start_iteration']
        ramp = self.options['normal_ramp_iterations']
        factor = float(elapsed >= 0) if ramp == 0 else min(1.0, max(0.0, elapsed / ramp))
        return self.options['normal_depth_weight'] * factor

    def compute(self, buffers, cameras, view, focal, iteration, backward=True, download=True):
        if not 0 <= view < self.views:
            raise ValueError('Geometry-loss gradients may only use the supplied training views.')
        options = self.options
        mask_weight, normal_weight = options['mask_weight'], self.normal_weight(iteration)
        self.count.zero_()
        self.terms.zero_()
        self.total.zero_()
        self.terms.grad.zero_()
        self.total.grad.zero_()
        if normal_weight > 0.0:
            wp.launch(select_normal_pixels, self.width * self.height,
                      inputs=[buffers, self.alpha, view, self.width, self.height, focal,
                              options['min_coverage'], options['foreground_threshold'], options['max_relative_depth_jump']],
                      outputs=[self.valid, self.count], device=self.device)
        tape = wp.Tape()
        with tape:
            if mask_weight > 0.0:
                wp.launch(mask_loss, self.width * self.height,
                          inputs=[buffers, self.alpha, view, self.width * self.height], outputs=[self.terms], device=self.device)
            if normal_weight > 0.0:
                wp.launch(normal_depth_loss, self.width * self.height,
                          inputs=[buffers, cameras, view, self.width, self.height, focal, self.valid, self.count],
                          outputs=[self.terms], device=self.device)
            wp.launch(combine_terms, 1, inputs=[self.terms, mask_weight, normal_weight], outputs=[self.total], device=self.device)
        if backward:
            # Add to the shader's existing G-buffer gradients; do not clear buffers.grad.
            tape.backward(self.total)
        if download:
            terms = self.terms.numpy()
            return {'mask': float(terms[0]), 'normal_depth': float(terms[1]),
                    'normal_weight': normal_weight, 'normal_pixels': int(self.count.numpy()[0]),
                    'weighted': mask_weight * float(terms[0]) + normal_weight * float(terms[1])}
        return None
