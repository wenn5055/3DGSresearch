"""Real spherical harmonics through degree 3, evaluated once per splat and view.

Convention follows the course's positive-m polynomial signs and splat-to-camera direction.
The DC term is the existing RGB directly; sh_rest holds indices 1..(degree+1)^2-1 in
degree/order order (l=1..3, m=-l..l). These sidecars are not Graphdeco SH checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

SH_BASIS = "course-real-sh-positive-m-v1"
SH_DIRECTION = "splat_to_camera"


@wp.func
def sh_basis(index: int, direction: wp.vec3):
    x, y, z = direction[0], direction[1], direction[2]
    value = float(0.0)
    if index == 1:
        value = 0.4886025119029199 * y
    elif index == 2:
        value = 0.4886025119029199 * z
    elif index == 3:
        value = 0.4886025119029199 * x
    elif index == 4:
        value = 1.0925484305920792 * x * y
    elif index == 5:
        value = 1.0925484305920792 * y * z
    elif index == 6:
        value = 0.31539156525252005 * (2.0 * z * z - x * x - y * y)
    elif index == 7:
        value = 1.0925484305920792 * x * z
    elif index == 8:
        value = 0.5462742152960396 * (x * x - y * y)
    elif index == 9:
        value = 0.5900435899266435 * y * (3.0 * x * x - y * y)
    elif index == 10:
        value = 2.890611442640554 * x * y * z
    elif index == 11:
        value = 0.4570457994644658 * y * (4.0 * z * z - x * x - y * y)
    elif index == 12:
        value = 0.3731763325901154 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
    elif index == 13:
        value = 0.4570457994644658 * x * (4.0 * z * z - x * x - y * y)
    elif index == 14:
        value = 1.445305721320277 * z * (x * x - y * y)
    elif index == 15:
        value = 0.5900435899266435 * x * (x * x - 3.0 * y * y)
    return value


@wp.func
def camera_position(camera: wp.mat44):
    translation = wp.vec3(camera[0, 3], camera[1, 3], camera[2, 3])
    return -wp.vec3(
        wp.dot(wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0]), translation),
        wp.dot(wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1]), translation),
        wp.dot(wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2]), translation),
    )


@wp.kernel
def shade_splats(
    means: wp.array(dtype=wp.vec3),
    base_rgb: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),
    cameras: wp.array(dtype=wp.mat44),
    view: int,
    rest_count: int,
    live_count: int,
    colours: wp.array(dtype=wp.vec3),
):
    splat = wp.tid()
    colour = base_rgb[splat]
    if active[splat] != 0:
        offset = camera_position(cameras[view]) - means[splat]
        direction = offset / wp.sqrt(wp.dot(offset, offset) + 1.0e-8)
        for index in range(live_count):
            colour += sh_basis(index + 1, direction) * coefficients[splat * rest_count + index]
    # Rasterization applies the original [0,1] clamp and supplies its masked colour adjoint.
    colours[splat] = colour


@wp.kernel(enable_backward=False)
def shade_splats_backward(
    means: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),
    cameras: wp.array(dtype=wp.mat44),
    view: int,
    rest_count: int,
    live_count: int,
    colour_gradient: wp.array(dtype=wp.float32),
    mean_gradient: wp.array(dtype=wp.float32),
    coefficient_gradient: wp.array(dtype=wp.vec3),
):
    splat = wp.tid()
    g = wp.vec3(colour_gradient[3 * splat], colour_gradient[3 * splat + 1], colour_gradient[3 * splat + 2])
    direction_gradient = wp.vec3(0.0)
    offset = camera_position(cameras[view]) - means[splat]
    distance = wp.sqrt(wp.dot(offset, offset) + 1.0e-8)
    direction = offset / distance
    for index in range(rest_count):
        gradient = wp.vec3(0.0)
        if active[splat] != 0 and index < live_count:
            gradient = sh_basis(index + 1, direction) * g
            index_adjoint, direction_adjoint = wp.grad(sh_basis)(index + 1, direction)
            direction_gradient += wp.dot(g, coefficients[splat * rest_count + index]) * direction_adjoint
        coefficient_gradient[splat * rest_count + index] = gradient
    # d normalize(camera - mean) / d mean. This is the appearance path to geometry.
    appearance_gradient = -(direction_gradient - direction * wp.dot(direction, direction_gradient)) / distance
    mean_gradient[3 * splat] += appearance_gradient[0]
    mean_gradient[3 * splat + 1] += appearance_gradient[1]
    mean_gradient[3 * splat + 2] += appearance_gradient[2]


@wp.kernel
def adam_sh_step(
    coefficients: wp.array(dtype=wp.vec3),
    gradients: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),
    moment: wp.array(dtype=wp.vec3),
    variance: wp.array(dtype=wp.vec3),
    rest_count: int,
    live_count: int,
    steps: wp.vec3,
    learning_rate: float,
):
    index = wp.tid()
    basis = index % rest_count
    if active[index // rest_count] != 0 and basis < live_count:
        band = int(0)
        if basis >= 8:
            band = 2
        elif basis >= 3:
            band = 1
        gradient = gradients[index]
        moment[index] = 0.9 * moment[index] + 0.1 * gradient
        variance[index] = 0.999 * variance[index] + 0.001 * wp.cw_mul(gradient, gradient)
        mhat = moment[index] / (1.0 - wp.pow(0.9, steps[band]))
        vhat = variance[index] / (1.0 - wp.pow(0.999, steps[band]))
        denominator = wp.vec3(wp.sqrt(vhat[0]) + 1.0e-8, wp.sqrt(vhat[1]) + 1.0e-8, wp.sqrt(vhat[2]) + 1.0e-8)
        coefficients[index] -= learning_rate * wp.cw_div(mhat, denominator)


@wp.kernel
def copy_sh_children(
    parents: wp.array(dtype=wp.int32),
    children: wp.array(dtype=wp.int32),
    rest_count: int,
    coefficients: wp.array(dtype=wp.vec3),
):
    pair, basis = wp.tid()
    coefficients[children[pair] * rest_count + basis] = coefficients[parents[pair] * rest_count + basis]


@wp.kernel
def clear_sh_child_state(children: wp.array(dtype=wp.int32), rest_count: int,
                         moment: wp.array(dtype=wp.vec3), variance: wp.array(dtype=wp.vec3)):
    pair, basis = wp.tid()
    index = children[pair] * rest_count + basis
    moment[index] = wp.vec3(0.0)
    variance[index] = wp.vec3(0.0)


class SHAppearance:
    def __init__(self, capacity: int, max_degree: int, learning_rate: float, device):
        if not isinstance(max_degree, int) or isinstance(max_degree, bool) or not 1 <= max_degree <= 3:
            raise ValueError("SH appearance requires an integer maximum degree in [1, 3].")
        self.capacity, self.max_degree = capacity, max_degree
        self.degree, self.learning_rate = 0, learning_rate
        self.device = wp.get_device(device)
        self.rest_count = (max_degree + 1) ** 2 - 1
        self.coefficients = wp.zeros(capacity * self.rest_count, dtype=wp.vec3, device=self.device)
        self.gradient = wp.zeros_like(self.coefficients)
        self.moment = wp.zeros_like(self.coefficients)
        self.variance = wp.zeros_like(self.coefficients)
        self.colours = wp.empty(capacity, dtype=wp.vec3, device=self.device)
        self.steps = [0, 0, 0]

    @property
    def live_count(self):
        return (self.degree + 1) ** 2 - 1

    def forward(self, means, rgb, active, cameras, view):
        wp.launch(shade_splats, dim=self.capacity,
                  inputs=[means, rgb, self.coefficients, active, cameras, int(view),
                          self.rest_count, self.live_count], outputs=[self.colours], device=self.device)
        return self.colours

    def backward(self, means, active, cameras, view, colour_gradient, mean_gradient):
        wp.launch(shade_splats_backward, dim=self.capacity,
                  inputs=[means, self.coefficients, active, cameras, int(view), self.rest_count,
                          self.live_count, colour_gradient],
                  outputs=[mean_gradient, self.gradient], device=self.device)

    def step(self, active):
        if self.degree == 0 or self.learning_rate == 0.0:
            return
        for band in range(self.degree):
            self.steps[band] += 1
        wp.launch(adam_sh_step, dim=self.capacity * self.rest_count,
                  inputs=[self.coefficients, self.gradient, active, self.moment, self.variance,
                          self.rest_count, self.live_count, wp.vec3(*self.steps), self.learning_rate],
                  device=self.device)

    def copy_children(self, parents, children, optimizer_state="inherit"):
        if optimizer_state not in ("inherit", "fresh_children", "legacy_reset"):
            raise ValueError("Unknown SH child optimizer-state policy.")
        if len(children):
            parent_ids = wp.array(np.asarray(parents, np.int32), dtype=wp.int32, device=self.device)
            child_ids = wp.array(np.asarray(children, np.int32), dtype=wp.int32, device=self.device)
            wp.launch(copy_sh_children, dim=(len(children), self.rest_count),
                      inputs=[parent_ids, child_ids, self.rest_count], outputs=[self.coefficients], device=self.device)
            if optimizer_state == "legacy_reset":
                self.reset_optimizer()
            elif optimizer_state == "fresh_children":
                wp.launch(clear_sh_child_state, dim=(len(children), self.rest_count),
                          inputs=[child_ids, self.rest_count], outputs=[self.moment, self.variance], device=self.device)
            else:
                for values in (self.moment, self.variance):
                    wp.launch(copy_sh_children, dim=(len(children), self.rest_count),
                              inputs=[parent_ids, child_ids, self.rest_count], outputs=[values], device=self.device)

    def reset_optimizer(self):
        self.moment.zero_()
        self.variance.zero_()
        self.gradient.zero_()
        self.steps = [0, 0, 0]

    def save(self, path, active):
        path = Path(path).with_suffix(".sh.npz")
        values = self.coefficients.numpy().reshape(self.capacity, self.rest_count, 3)[active]
        np.savez_compressed(path, sh_rest=values, sh_degree=np.int32(self.degree),
                            max_sh_degree=np.int32(self.max_degree), basis=SH_BASIS,
                            direction=SH_DIRECTION, format_version=np.int32(1))
        return path


def load_sh_sidecar(path, count: int):
    """Validate row count, basis convention and saved active degree before GPU allocation."""
    with np.load(path, allow_pickle=False) as saved:
        values = np.asarray(saved["sh_rest"], dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != count or values.shape[2] != 3 or values.shape[1] not in (3, 8, 15):
            raise ValueError(f"Invalid SH shape {values.shape} in {path}; expected ({count}, 3/8/15, 3).")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite SH coefficients in {path}.")
        max_degree = {3: 1, 8: 2, 15: 3}[values.shape[1]]
        if "max_sh_degree" in saved and int(saved["max_sh_degree"]) != max_degree:
            raise ValueError(f"SH maximum degree does not match coefficient shape in {path}.")
        degree = int(saved["sh_degree"]) if "sh_degree" in saved else max_degree
        if not 0 <= degree <= max_degree:
            raise ValueError(f"Invalid active SH degree {degree} in {path}.")
        if "format_version" in saved and int(saved["format_version"]) != 1:
            raise ValueError(f"Unsupported SH sidecar version in {path}.")
        if "basis" in saved and str(saved["basis"].item()) != SH_BASIS:
            raise ValueError(f"Unsupported SH basis in {path}.")
        if "direction" in saved and str(saved["direction"].item()) != SH_DIRECTION:
            raise ValueError(f"Unsupported SH direction convention in {path}.")
    return values, max_degree, degree
