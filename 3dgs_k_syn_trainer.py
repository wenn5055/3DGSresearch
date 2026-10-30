# ARCHIVE CANDIDATE (2026-10-29): Small synthetic teaching exercise, superseded by full multi-view training.
# Successor: 3dgs_trainer.py; research extensions: src/cp4282_gs/surfels/.
# Archive with 3dgs_1_syn_trainer.py, 3dgs_renderer_v1.py and their exclusive helpers.

"""Train three overlapping coloured Gaussians against synthetic views with Warp.

Usage:
    python 3dgs_k_syn_trainer.py multi.png --device cpu --iterations 800

This is the Unit 8 extension after the one-Gaussian program. It keeps the same small synthetic
scene and Warp tape, but alpha-composites several trainable splats for every pixel, and introduces
adaptive density control -- gradient-ranked clone/split, growing a fixed-capacity pool -- so that a
random initialization that leaves one target under-covered can still recover. Unit 9's full
trainer reuses this same densify_and_prune, adding multi-view scoring and late pruning on top.
Tile assignment, Adam, and the full multi-splat trainer belong to Unit 9.

Synthetic targets use the implemented `CpuRenderer.render` in `3dgs_renderer_v1.py`,
via `3dgs_1_syn_trainer.py`.
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
_one_gaussian = importlib.import_module("3dgs_1_syn_trainer")
FOCAL_LENGTH = _one_gaussian.FOCAL_LENGTH
HEIGHT = _one_gaussian.HEIGHT
VIEWS = _one_gaussian.VIEWS
WIDTH = _one_gaussian.WIDTH
alpha_at_pixel = _one_gaussian.alpha_at_pixel
camera_matrices = _one_gaussian.camera_matrices
psnr_from_mse = _one_gaussian.psnr_from_mse
render_targets = _one_gaussian.render_targets
sigmoid = _one_gaussian.sigmoid


INITIAL_SPLATS = 3
CAPACITY = 16
LEARNING_RATES = {
    # Tuned for this small synthetic problem: the original rates were stable but
    # still improving noticeably after iteration 800.
    "position_lr": 28.0,
    "scale_lr": 10.0,
    "rotation_lr": 3.0,
    "opacity_lr": 12.0,
    "color_lr": 24.0,
}

# Same reference values as Unit 9's densification config: check every 100 iterations, add
# splats equal to half the currently active count, stop halfway through the run, and treat a
# splat as "large" once its biggest scale exceeds 1% of the scene radius.
DENSIFY_INTERVAL = 100
DENSIFY_FRACTION = 0.5
PERCENT_DENSE = 0.01


def depth_order(means: np.ndarray, cameras: np.ndarray) -> np.ndarray:
    """Return near-to-far splat IDs for each view, for the splats currently at `means`.

    Depth order is a discrete, non-differentiable event -- see "Depth order, refreshed when the
    splat population changes" in Unit 8. This program holds the order fixed between refreshes and
    recomputes it (from the trainable splats' own positions) each time densify_and_prune changes
    which splats exist, the same moments Unit 9 rebuilds its tile-local depth lists.
    """
    orders = []
    for camera in cameras:
        camera_means = (camera[:3, :3] @ means.T + camera[:3, 3:4]).T
        orders.append(np.argsort(camera_means[:, 2]).astype(np.int32))
    return np.stack(orders)


def camera_world_position(world_to_camera: np.ndarray) -> np.ndarray:
    """Invert a world-to-camera matrix's rigid part to recover the camera's world position."""
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    return -rotation.T @ translation


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix for one (w, x, y, z) quaternion, columns matching the
    renderer's local frame -- same construction as Unit 9, used to sample split offsets."""
    qw, qx, qy, qz = quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-8)
    return np.stack((
        np.array([1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy)]),
        np.array([2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx)]),
        np.array([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]),
    ), axis=1).astype(np.float32)


@dataclass
class SyntheticMultiScene:
    cameras: np.ndarray
    targets: np.ndarray

    @classmethod
    def build(cls) -> "SyntheticMultiScene":
        cameras = camera_matrices()
        # The three centres overlap in the images but have distinct colours.
        means = np.array([
            [-0.16, 0.02, 0.03],
            [0.00, -0.02, 0.00],
            [0.15, 0.04, -0.02],
        ], dtype=np.float32)
        log_scales = np.log(np.array([
            [0.18, 0.11, 0.20],
            [0.16, 0.13, 0.18],
            [0.17, 0.10, 0.22],
        ], dtype=np.float32))
        quaternions = np.array([
            [0.98, 0.08, 0.02, 0.10],
            [0.96, -0.10, 0.12, 0.08],
            [0.97, 0.04, -0.12, 0.14],
        ], dtype=np.float32)
        opacity_logits = np.array([1.0, 0.8, 0.9], dtype=np.float32)
        colors = np.array([
            [0.95, 0.10, 0.08],  # red
            [0.08, 0.85, 0.16],  # green
            [0.10, 0.25, 0.95],  # blue
        ], dtype=np.float32)
        targets = render_targets(
            means, log_scales, quaternions, opacity_logits, colors, cameras,
        ).reshape(-1, 3)
        return cls(cameras, targets)


def initial_population(rng: np.random.Generator, device) -> tuple[TrainableGaussianSet, np.ndarray]:
    """CAPACITY trainable splats: the first INITIAL_SPLATS start active and randomly placed,
    like Unit 8's one-splat program; the rest start inactive until densify_and_prune activates
    them, the same fixed-capacity, active-mask shape Unit 9 uses for its full splat pool.

    Active and dormant slots draw from the same distribution, so one `random_init(CAPACITY, ...)`
    call is equivalent to the two smaller calls it replaces -- without the extra device-to-host
    downloads and re-upload that concatenating two separately-uploaded sets would need.
    """
    trainable = TrainableGaussianSet.random_init(CAPACITY, rng, device)
    active = np.array([True] * INITIAL_SPLATS + [False] * (CAPACITY - INITIAL_SPLATS))
    return trainable, active


@wp.kernel
def render_multi_loss(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),
    cameras: wp.array(dtype=wp.mat44),
    orders: wp.array(dtype=wp.int32),
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
    colour = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)
    for rank in range(CAPACITY):
        splat = orders[view * CAPACITY + rank]
        if active[splat] != 0:
            alpha = alpha_at_pixel(
                means[splat], log_scales[splat], quaternions[splat], opacity_logits[splat],
                cameras[view], px, py, float(width), float(height), focal,
            )
            colour = colour + transmittance * alpha * colors[splat]
            transmittance = transmittance * (1.0 - alpha)
    image[thread] = colour
    difference = colour - targets[thread]
    wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * pixels * VIEWS))


class MultiGaussianTrainer:
    def __init__(self, scene: SyntheticMultiScene, seed: int, device, iterations: int):
        self.device = wp.get_device(device)
        self.rng = np.random.default_rng(seed)
        self.trainable, self.active = initial_population(self.rng, self.device)
        self.active_device = wp.array(self.active.astype(np.int32), dtype=wp.int32, device=self.device)
        self.cameras = wp.array(scene.cameras, dtype=wp.mat44, device=self.device)
        self.scene_cameras = scene.cameras
        orders = depth_order(self.trainable.means.numpy(), scene.cameras)
        self.orders = wp.array(orders.reshape(-1), dtype=wp.int32, device=self.device)
        self.targets = wp.array(scene.targets, dtype=wp.vec3, device=self.device)
        self.image = wp.zeros(VIEWS * WIDTH * HEIGHT, dtype=wp.vec3, device=self.device)
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        self._tape = None
        # Same reference formula as Unit 9: 35% of the mean camera distance from the origin.
        camera_positions = np.stack([camera_world_position(c) for c in scene.cameras])
        self.scene_radius = float(np.linalg.norm(camera_positions, axis=1).mean()) * 0.35
        self.densify_until = iterations // 2

    def render_and_compute_loss(self) -> float:
        self.loss.zero_()
        self._tape = wp.Tape()
        with self._tape:
            wp.launch(
                render_multi_loss,
                dim=VIEWS * WIDTH * HEIGHT,
                inputs=[
                    self.trainable.means, self.trainable.log_scales,
                    self.trainable.quaternions, self.trainable.opacity_logits,
                    self.trainable.colors, self.active_device,
                    self.cameras, self.orders, self.targets,
                    WIDTH, HEIGHT, FOCAL_LENGTH,
                ],
                outputs=[self.image, self.loss],
                device=self.device,
            )
        return float(self.loss.numpy()[0])

    def densify_and_prune(self, mean_gradients: np.ndarray) -> None:
        """Rank active splats by position-gradient norm, prune faded ones, then split large
        parents (offset sampled from the parent's own covariance) or clone small ones into freed
        or never-used capacity slots. Unit 9's full trainer reuses this exact structure and adds
        multi-view scoring and late pruning on top -- see Unit 9, Stage 7-8."""
        opacity = sigmoid(self.trainable.opacity_logits.numpy())
        means = self.trainable.means.numpy()
        log_scales = self.trainable.log_scales.numpy()
        quaternions = self.trainable.quaternions.numpy()
        opacity_logits = self.trainable.opacity_logits.numpy()
        colors = self.trainable.colors.numpy()

        prune_mask = (opacity < 0.005) & self.active
        scores = np.linalg.norm(mean_gradients, axis=1)
        parents = [i for i in np.argsort(scores)[::-1] if self.active[i]]
        self.active[prune_mask] = False
        # A pruned splat must not stay a clone source: it now sits in `free`, so reusing it as a
        # parent would let a clone read a slot already overwritten as another clone's
        # destination this same call. Keep only still-active parents.
        parents = [parent for parent in parents if self.active[parent]]
        free = np.flatnonzero(~self.active)
        children_to_add = min(len(free), max(1, int(self.active.sum() * DENSIFY_FRACTION)))

        split_size = PERCENT_DENSE * self.scene_radius
        for parent, child in zip(parents[:children_to_add], free[:children_to_add]):
            self.active[child] = True
            parent_scales = np.exp(log_scales[parent])
            if float(parent_scales.max()) > split_size:
                rotation = quaternion_to_rotation(quaternions[parent])
                origin = means[parent].copy()
                means[parent] = origin + rotation @ self.rng.normal(0.0, parent_scales)
                means[child] = origin + rotation @ self.rng.normal(0.0, parent_scales)
                log_scales[parent] = log_scales[parent] - np.log(1.6)
                log_scales[child] = log_scales[parent]
            else:
                means[child] = means[parent]
                log_scales[child] = log_scales[parent]
            quaternions[child] = quaternions[parent]
            opacity_logits[child] = opacity_logits[parent]
            colors[child] = colors[parent]

        self.trainable.means.assign(means.astype(np.float32))
        self.trainable.log_scales.assign(log_scales.astype(np.float32))
        self.trainable.quaternions.assign(quaternions.astype(np.float32))
        self.trainable.opacity_logits.assign(opacity_logits.astype(np.float32))
        self.trainable.colors.assign(colors.astype(np.float32))
        self.active_device.assign(self.active.astype(np.int32))
        self.orders.assign(depth_order(means, self.scene_cameras).reshape(-1))

    def backward_and_update(self, iteration: int) -> None:
        self._tape.backward(loss=self.loss)
        if iteration > 0 and iteration % DENSIFY_INTERVAL == 0 and iteration <= self.densify_until:
            self.densify_and_prune(self._tape.gradients[self.trainable.means].numpy())
        self.trainable.sgd_step(self._tape, **LEARNING_RATES)
        self._tape.zero()


def save_panel(
    path: Path,
    targets: np.ndarray,
    rendered: np.ndarray,
) -> None:
    top = targets.reshape(VIEWS, HEIGHT, WIDTH, 3)
    bottom = rendered.reshape(VIEWS, HEIGHT, WIDTH, 3)
    rows = [np.concatenate([top[i] for i in range(VIEWS)], axis=1),
            np.concatenate([bottom[i] for i in range(VIEWS)], axis=1)]
    panel = np.concatenate(rows, axis=0)
    Image.fromarray(np.uint8(np.clip(panel, 0.0, 1.0) * 255.0)).save(path)


def train(args: argparse.Namespace) -> None:
    wp.init()
    device = wp.get_device(args.device)
    scene = SyntheticMultiScene.build()
    trainer = MultiGaussianTrainer(scene, args.seed, device, args.iterations)
    snapshots = {0, 100, 300, args.iterations}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(args.iterations + 1):
        current_loss = trainer.render_and_compute_loss()
        if iteration in snapshots:
            suffix = "" if iteration == args.iterations else f"-{iteration:04d}"
            save_panel(output.with_name(output.stem + suffix + output.suffix),
                       scene.targets, trainer.image.numpy())
        if iteration % args.log_every == 0:
            print(f"{iteration:5d}: loss={current_loss:.6f}, psnr={psnr_from_mse(current_loss):.2f} dB, "
                  f"active={int(trainer.active.sum())}", flush=True)
        if iteration == args.iterations:
            break
        trainer.backward_and_update(iteration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--device", default="cpu", help="Warp device, such as cpu or cuda:0")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
