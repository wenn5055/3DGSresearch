# ARCHIVE CANDIDATE (2026-10-29): One-splat synthetic teaching exercise.
# Next teaching version: 3dgs_k_syn_trainer.py; current full trainer: 3dgs_trainer.py.
# Archive with the teaching renderer, multi-splat trainer and their exclusive helpers.

"""Train one anisotropic 3D Gaussian against synthetic target images with Warp.

Usage:
    python 3dgs_1_syn_trainer.py training.png --device cpu --iterations 800

This is the Unit 8 teaching trainer. It keeps the scene deliberately tiny: one Gaussian,
three known cameras, synthetic target images, and no tiling or densification. The goal is to
show the training loop before Unit 9 adds the full renderer and adaptive density control.

Synthetic targets use the implemented `CpuRenderer.render` in `3dgs_renderer_v1.py`.
The full trainer in `3dgs_trainer.py` does not depend on this exercise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import warp as wp

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# `shared/` sits beside this file in the assignment repo, and one level up in the course repo.
for _candidate in (_here / "shared", _here.parent / "shared"):
    if _candidate.is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from trainable_gaussian import TrainableGaussianSet
_reference_renderer = importlib.import_module("3dgs_renderer_v1")
Camera = _reference_renderer.Camera
CpuRenderer = _reference_renderer.CpuRenderer
GaussianSet = _reference_renderer.GaussianSet


WIDTH = 96
HEIGHT = 96
VIEWS = 3
FOCAL_LENGTH = 90.0
NEAR_PLANE = 0.1
FILTER_VARIANCE = 0.3
UNIT9_LEARNING_RATES = {
    "position_lr": 20.0,
    "scale_lr": 8.0,
    "rotation_lr": 2.0,
    "opacity_lr": 10.0,
    "color_lr": 20.0,
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def psnr_from_mse(mse: float) -> float:
    """Return PSNR for images whose channel values are in [0, 1]."""
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def camera_matrices() -> np.ndarray:
    positions = [(-2.4, -2.4, 2.8), (2.6, -1.5, 2.3), (0.2, 3.0, 2.1)]
    target = (0.0, 0.0, 0.0)
    up = (0.0, 0.0, -1.0)
    cameras = [
        Camera.from_look_at(WIDTH, HEIGHT, FOCAL_LENGTH, position, target, up)
        for position in positions
    ]
    return np.stack([camera.world_to_camera for camera in cameras])


def render_targets(
    means: np.ndarray,
    log_scales: np.ndarray,
    quaternions: np.ndarray,
    opacity_logits: np.ndarray,
    colors: np.ndarray,
    cameras: np.ndarray,
) -> np.ndarray:
    hidden = GaussianSet(
        means=means,
        scales=np.exp(log_scales),
        rotations=quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True),
        opacities=sigmoid(opacity_logits),
        colors=colors,
    )
    images = []
    for view in range(VIEWS):
        camera = Camera(
            WIDTH, HEIGHT, FOCAL_LENGTH, FOCAL_LENGTH,
            WIDTH / 2.0, HEIGHT / 2.0, cameras[view],
        )
        images.append(
            CpuRenderer(
                camera,
                near=NEAR_PLANE,
                filter_variance=FILTER_VARIANCE,
                qmax=9.0,
                ).render(hidden, background=(0.0, 0.0, 0.0))
        )
    return np.stack(images)


@dataclass
class SyntheticScene:
    """Three known cameras and the multi-view images they see of one hidden ground-truth
    Gaussian. Training never touches the hidden parameters -- only these images and cameras,
    exactly as if they came from real photographs."""

    cameras: np.ndarray  # (VIEWS, 4, 4) world-to-camera matrices
    targets: np.ndarray  # (VIEWS * HEIGHT * WIDTH, 3) flattened target RGB in [0, 1]

    @classmethod
    def build(cls) -> "SyntheticScene":
        cameras = camera_matrices()
        hidden_means = np.array([[0.18, -0.10, 0.05]], dtype=np.float32)
        hidden_log_scales = np.log(np.array([[0.20, 0.12, 0.26]], dtype=np.float32))
        hidden_quaternions = np.array([[0.96, 0.18, -0.08, 0.20]], dtype=np.float32)
        hidden_opacity_logits = np.array([1.4], dtype=np.float32)
        hidden_colors = np.array([[0.95, 0.55, 0.18]], dtype=np.float32)
        targets = render_targets(
            hidden_means, hidden_log_scales, hidden_quaternions,
            hidden_opacity_logits, hidden_colors, cameras,
        ).reshape(-1, 3)
        return cls(cameras=cameras, targets=targets)


@wp.func
def projected_covariance(
    log_scale: wp.vec3,
    quaternion: wp.vec4,
    camera: wp.mat44,
    x: float,
    y: float,
    z: float,
    focal: float,
):
    quaternion = quaternion / wp.sqrt(wp.dot(quaternion, quaternion) + 1.0e-8)
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    r0 = wp.vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy))
    r1 = wp.vec3(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx))
    r2 = wp.vec3(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
    c0 = wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0])
    c1 = wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1])
    c2 = wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2])
    j0 = wp.vec3(focal / z, 0.0, -focal * x / (z * z))
    j1 = wp.vec3(0.0, focal / z, -focal * y / (z * z))
    p0 = wp.vec3(wp.dot(j0, c0), wp.dot(j0, c1), wp.dot(j0, c2))
    p1 = wp.vec3(wp.dot(j1, c0), wp.dot(j1, c1), wp.dot(j1, c2))
    b0 = wp.vec3(wp.dot(p0, r0), wp.dot(p0, r1), wp.dot(p0, r2))
    b1 = wp.vec3(wp.dot(p1, r0), wp.dot(p1, r1), wp.dot(p1, r2))
    scales_squared = wp.vec3(wp.exp(2.0 * log_scale[0]), wp.exp(2.0 * log_scale[1]), wp.exp(2.0 * log_scale[2]))
    a = wp.dot(scales_squared, wp.vec3(b0[0] * b0[0], b0[1] * b0[1], b0[2] * b0[2])) + FILTER_VARIANCE
    b = wp.dot(scales_squared, wp.vec3(b0[0] * b1[0], b0[1] * b1[1], b0[2] * b1[2]))
    c = wp.dot(scales_squared, wp.vec3(b1[0] * b1[0], b1[1] * b1[1], b1[2] * b1[2])) + FILTER_VARIANCE
    return wp.vec3(a, b, c)


@wp.func
def alpha_at_pixel(
    mean: wp.vec3,
    log_scale: wp.vec3,
    quaternion: wp.vec4,
    opacity_logit: float,
    camera: wp.mat44,
    px: float,
    py: float,
    width: float,
    height: float,
    focal: float,
):
    point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)
    z = wp.max(point[2], NEAR_PLANE)
    centre_x = focal * point[0] / z + 0.5 * width
    centre_y = focal * point[1] / z + 0.5 * height
    covariance = projected_covariance(log_scale, quaternion, camera, point[0], point[1], z, focal)
    determinant = wp.max(covariance[0] * covariance[2] - covariance[1] * covariance[1], 1.0e-8)
    conic = wp.vec3(covariance[2] / determinant, -covariance[1] / determinant, covariance[0] / determinant)
    dx, dy = px - centre_x, py - centre_y
    q = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy
    alpha = float(0.0)
    if q <= 9.0:
        alpha = wp.min(1.0 / (1.0 + wp.exp(-opacity_logit)) * wp.exp(-0.5 * q), 0.99)
    return alpha


@wp.kernel
def render_loss(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3),
    cameras: wp.array(dtype=wp.mat44),
    targets: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    focal: float,
    image: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):
    thread = wp.tid()
    pixels = width * height
    view = thread // pixels
    pixel = thread - view * pixels
    px = float(pixel % width) + 0.5
    py = float(pixel // width) + 0.5
    alpha = alpha_at_pixel(
        means[0], log_scales[0], quaternions[0], opacity_logits[0],
        cameras[view], px, py, float(width), float(height), focal,
    )
    rgb = alpha * colors[0]
    image[thread] = rgb
    difference = rgb - targets[thread]
    wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * pixels * VIEWS))


class SyntheticTrainer:
    """Owns the one trainable Gaussian and the Warp buffers needed to fit it to a
    :class:`SyntheticScene`. Splits one training step into a forward half (render every view
    under a differentiation tape) and a backward half (propagate gradients and apply one SGD
    update), so the outer loop can report each iteration's loss before deciding whether to
    update -- the same shape as Unit 9's trainer, at a much smaller scale.
    """

    def __init__(self, scene: SyntheticScene, seed: int, device):
        self.device = wp.get_device(device)
        rng = np.random.default_rng(seed)
        self.trainable = TrainableGaussianSet.random_init(count=1, rng=rng, device=self.device)
        if self.trainable.count != 1:
            raise ValueError("Unit 8's renderer supports exactly one Gaussian.")
        self.cameras = wp.array(scene.cameras, dtype=wp.mat44, device=self.device)
        self.targets = wp.array(scene.targets, dtype=wp.vec3, device=self.device)
        self.image = wp.zeros(VIEWS * WIDTH * HEIGHT, dtype=wp.vec3, device=self.device)
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._tape = None

    def render_and_compute_loss(self) -> float:
        """Forward pass: render every view under a fresh differentiation tape, return the loss."""
        self.loss.zero_()
        self._tape = wp.Tape()
        with self._tape:
            wp.launch(
                render_loss,
                dim=VIEWS * WIDTH * HEIGHT,
                inputs=[
                    self.trainable.means, self.trainable.log_scales, self.trainable.quaternions,
                    self.trainable.opacity_logits, self.trainable.colors, self.cameras, self.targets,
                    WIDTH, HEIGHT, FOCAL_LENGTH,
                ],
                outputs=[self.image, self.loss],
                device=self.device,
            )
        return float(self.loss.numpy()[0])

    def backward_and_update(self) -> None:
        """Backpropagate through the last forward pass and apply one SGD update."""
        self._tape.backward(loss=self.loss)
        self.trainable.sgd_step(self._tape, **UNIT9_LEARNING_RATES)
        self._tape.zero()


def save_panel(path: Path, targets: np.ndarray, rendered: np.ndarray) -> None:
    top = targets.reshape(VIEWS, HEIGHT, WIDTH, 3)
    bottom = rendered.reshape(VIEWS, HEIGHT, WIDTH, 3)
    rows = [np.concatenate([top[i] for i in range(VIEWS)], axis=1),
            np.concatenate([bottom[i] for i in range(VIEWS)], axis=1)]
    panel = np.concatenate(rows, axis=0)
    Image.fromarray(np.uint8(np.clip(panel, 0.0, 1.0) * 255.0)).save(path)


def train(args: argparse.Namespace) -> None:
    wp.init()
    device = wp.get_device(args.device)
    scene = SyntheticScene.build()
    trainer = SyntheticTrainer(scene, args.seed, device)

    snapshots = {0, args.iterations}
    if args.iterations >= 300:
        snapshots.add(300)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(args.iterations + 1):
        current_loss = trainer.render_and_compute_loss()
        if iteration in snapshots:
            suffix = "" if iteration == args.iterations else f"-{iteration:04d}"
            save_panel(output.with_name(output.stem + suffix + output.suffix), scene.targets, trainer.image.numpy())
        if iteration % args.log_every == 0:
            print(
                f"{iteration:5d}: loss={current_loss:.6f}, "
                f"psnr={psnr_from_mse(current_loss):.2f} dB",
                flush=True,
            )
        if iteration == args.iterations:
            break
        trainer.backward_and_update()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu", help="Warp device, such as cpu or cuda:0")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
