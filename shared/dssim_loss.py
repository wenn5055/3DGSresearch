"""Windowed D-SSIM and its image-space gradient, computed entirely in Warp.

Uses uniform, reflected windows and population variances, matching image_metrics.ssim_rgb.
The backward filters apply the transpose of reflection: a border pixel can occur several
times in one window. Simply reflecting the derivative image would give the wrong adjoint.
"""

from __future__ import annotations

import warp as wp
from warp.utils import array_sum


@wp.func
def reflected_index(index: int, size: int):
    result = int(0)
    if size > 1:
        period = 2 * (size - 1)
        result = ((index % period) + period) % period
        if result >= size:
            result = period - result
    return result


@wp.func
def reflection_count(centre: int, pixel: int, size: int, radius: int):
    """Number of entries of a 1D reflected window that refer to this pixel."""
    count = 2 * radius + 1
    if size > 1:
        period = float(2 * (size - 1))
        lower = centre - radius
        upper = centre + radius
        count = int(wp.floor(float(upper - pixel) / period))
        count += int(wp.floor(float(pixel - lower) / period)) + 1
        # Interior pixels have two distinct preimages per reflection period.
        if pixel > 0 and pixel < size - 1:
            count += int(wp.floor(float(upper + pixel) / period))
            count += int(wp.floor(float(-pixel - lower) / period)) + 1
    return count


@wp.kernel
def box_filter_horizontal(
    image: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    view_ids: wp.array(dtype=wp.int32),
    width: int,
    height: int,
    radius: int,
    mean_x: wp.array(dtype=wp.vec3),
    mean_y: wp.array(dtype=wp.vec3),
    second_x: wp.array(dtype=wp.vec3),
    second_y: wp.array(dtype=wp.vec3),
    cross: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    pixels = width * height
    batch = index // pixels
    pixel = index % pixels
    row = pixel // width
    column = pixel % width
    sx = wp.vec3(0.0)
    sy = wp.vec3(0.0)
    sxx = wp.vec3(0.0)
    syy = wp.vec3(0.0)
    sxy = wp.vec3(0.0)
    for offset in range(-radius, radius + 1):
        neighbour = row * width + reflected_index(column + offset, width)
        x = image[batch * pixels + neighbour]
        y = targets[view_ids[batch] * pixels + neighbour]
        sx += x
        sy += y
        sxx += wp.cw_mul(x, x)
        syy += wp.cw_mul(y, y)
        sxy += wp.cw_mul(x, y)
    factor = 1.0 / float(2 * radius + 1)
    mean_x[index] = factor * sx
    mean_y[index] = factor * sy
    second_x[index] = factor * sxx
    second_y[index] = factor * syy
    cross[index] = factor * sxy


@wp.kernel
def dssim_window_terms(
    mean_x: wp.array(dtype=wp.vec3),
    mean_y: wp.array(dtype=wp.vec3),
    second_x: wp.array(dtype=wp.vec3),
    second_y: wp.array(dtype=wp.vec3),
    cross: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    radius: int,
    view_count: int,
    terms: wp.array(dtype=wp.float32),
    mean_adjoint: wp.array(dtype=wp.vec3),
    second_adjoint: wp.array(dtype=wp.vec3),
    cross_adjoint: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    pixels = width * height
    batch = index // pixels
    pixel = index % pixels
    row = pixel // width
    column = pixel % width
    mx = wp.vec3(0.0)
    my = wp.vec3(0.0)
    exx = wp.vec3(0.0)
    eyy = wp.vec3(0.0)
    exy = wp.vec3(0.0)
    for offset in range(-radius, radius + 1):
        neighbour = batch * pixels + reflected_index(row + offset, height) * width + column
        mx += mean_x[neighbour]
        my += mean_y[neighbour]
        exx += second_x[neighbour]
        eyy += second_y[neighbour]
        exy += cross[neighbour]
    factor = 1.0 / float(2 * radius + 1)
    mx *= factor
    my *= factor
    exx *= factor
    eyy *= factor
    exy *= factor
    # D-SSIM = mean((1 - SSIM) / 2), averaging over views, pixels and channels.
    normalization = 1.0 / float(6 * pixels * view_count)
    value = float(0.0)
    gmean = wp.vec3(0.0)
    gsecond = wp.vec3(0.0)
    gcross = wp.vec3(0.0)
    for channel in range(3):
        u = mx[channel]
        v = my[channel]
        raw_variance = exx[channel] - u * u
        variance_x = wp.max(raw_variance, 0.0)
        variance_y = wp.max(eyy[channel] - v * v, 0.0)
        covariance = exy[channel] - u * v
        a = 2.0 * u * v + 0.0001
        b = 2.0 * covariance + 0.0009
        c = u * u + v * v + 0.0001
        d = variance_x + variance_y + 0.0009
        inverse_cd = 1.0 / (c * d)
        ssim = a * b * inverse_cd
        value += normalization * (1.0 - ssim)
        variance_derivative = float(0.0)
        if raw_variance > 0.0:
            variance_derivative = 1.0
        # Differentiate the moments, without dividing by a possibly zero covariance.
        gmean[channel] = -normalization * (
            2.0 * v * b * inverse_cd - 2.0 * u * ssim / c
            - 2.0 * v * a * inverse_cd + 2.0 * u * ssim / d * variance_derivative
        )
        gsecond[channel] = normalization * ssim / d * variance_derivative
        gcross[channel] = -normalization * 2.0 * a * inverse_cd
    terms[index] = value
    mean_adjoint[index] = gmean
    second_adjoint[index] = gsecond
    cross_adjoint[index] = gcross


@wp.kernel
def box_adjoint_vertical(
    mean_adjoint: wp.array(dtype=wp.vec3),
    second_adjoint: wp.array(dtype=wp.vec3),
    cross_adjoint: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    radius: int,
    mean_rows: wp.array(dtype=wp.vec3),
    second_rows: wp.array(dtype=wp.vec3),
    cross_rows: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    pixels = width * height
    batch = index // pixels
    pixel = index % pixels
    row = pixel // width
    column = pixel % width
    gm = wp.vec3(0.0)
    gs = wp.vec3(0.0)
    gc = wp.vec3(0.0)
    for centre in range(wp.max(0, row - radius), wp.min(height, row + radius + 1)):
        neighbour = batch * pixels + centre * width + column
        count = float(reflection_count(centre, row, height, radius))
        gm += count * mean_adjoint[neighbour]
        gs += count * second_adjoint[neighbour]
        gc += count * cross_adjoint[neighbour]
    factor = 1.0 / float(2 * radius + 1)
    mean_rows[index] = factor * gm
    second_rows[index] = factor * gs
    cross_rows[index] = factor * gc


@wp.kernel
def blend_loss_gradients(
    image: wp.array(dtype=wp.vec3),
    targets: wp.array(dtype=wp.vec3),
    view_ids: wp.array(dtype=wp.int32),
    mean_rows: wp.array(dtype=wp.vec3),
    second_rows: wp.array(dtype=wp.vec3),
    cross_rows: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    radius: int,
    weight: float,
    pixel_gradient: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    pixels = width * height
    batch = index // pixels
    pixel = index % pixels
    row = pixel // width
    column = pixel % width
    gm = wp.vec3(0.0)
    gs = wp.vec3(0.0)
    gc = wp.vec3(0.0)
    for centre in range(wp.max(0, column - radius), wp.min(width, column + radius + 1)):
        neighbour = batch * pixels + row * width + centre
        count = float(reflection_count(centre, column, width, radius))
        gm += count * mean_rows[neighbour]
        gs += count * second_rows[neighbour]
        gc += count * cross_rows[neighbour]
    x = image[index]
    y = targets[view_ids[batch] * pixels + pixel]
    gradient = (gm + 2.0 * wp.cw_mul(x, gs) + wp.cw_mul(y, gc)) / float(2 * radius + 1)
    pixel_gradient[index] = (1.0 - weight) * pixel_gradient[index] + weight * gradient


class DSSIMComputer:
    """Reusable image buffers; no per-step host image transfers or autograd dependency."""

    def __init__(self, width: int, height: int, max_views: int, device):
        self.width, self.height = width, height
        self.max_views, self.device = max_views, wp.get_device(device)
        count = width * height * max_views
        self.horizontal = [wp.empty(count, dtype=wp.vec3, device=self.device) for _ in range(5)]
        self.adjoints = [wp.empty(count, dtype=wp.vec3, device=self.device) for _ in range(3)]
        self.vertical = [wp.empty(count, dtype=wp.vec3, device=self.device) for _ in range(3)]
        self.terms = wp.empty(count, dtype=wp.float32, device=self.device)
        self.total = wp.empty(1, dtype=wp.float32, device=self.device)

    def compute(self, image, targets, view_ids, pixel_gradient, window: int,
                view_count: int, weight: float, download_loss=True, compute_gradient=True):
        if not 1 <= view_count <= self.max_views:
            raise ValueError("D-SSIM view count exceeds its image-buffer capacity.")
        if not isinstance(window, int) or isinstance(window, bool) or not 1 <= window <= 31 or window % 2 == 0:
            raise ValueError("D-SSIM window must be an odd integer in [1, 31].")
        count = self.width * self.height * view_count
        shape = [self.width, self.height, window // 2]
        wp.launch(box_filter_horizontal, dim=count,
                  inputs=[image, targets, view_ids] + shape,
                  outputs=self.horizontal, device=self.device)
        wp.launch(dssim_window_terms, dim=count,
                  inputs=self.horizontal + shape + [view_count],
                  outputs=[self.terms] + self.adjoints, device=self.device)
        if compute_gradient:
            wp.launch(box_adjoint_vertical, dim=count,
                      inputs=self.adjoints + shape, outputs=self.vertical, device=self.device)
            wp.launch(blend_loss_gradients, dim=count,
                      inputs=[image, targets, view_ids] + self.vertical + shape + [weight],
                      outputs=[pixel_gradient], device=self.device)
        if download_loss:
            array_sum(self.terms, out=self.total, value_count=count)
            return float(self.total.numpy()[0])
        return None
