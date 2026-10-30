"""Train anisotropic 3D Gaussian splats from NeRF-synthetic posed images with Warp.

Explicit dense and sampled backward kernels optimize position, scale, rotation, opacity and
fixed RGB. The optional Unit 10 loss curriculum starts with MSE, then enables dense MSE +
D-SSIM when held-out MSE PSNR reaches adaptive.dssim_psnr_threshold. Without a loss block,
the trainer keeps its MSE-only objective.

Check the dense kernel with the gradient checker, which compares your analytic gradients against
central finite differences:

    python 3dgs_gradient_check_gpu.py --device cpu

Check the sparse kernel against the dense one yourself. Set `sparse.samples_per_tile` to 256
(`TILE * TILE`), which makes the sampler fall back to the exact raster grid: every pixel is then
sampled exactly once, so the sparse path is doing the dense path's work and the loss and every
gradient buffer must match a dense run to floating-point noise. If they do not, one of the two
kernels is wrong.

Usage:
    python 3dgs_trainer.py config/3dgs_training_gpu.yaml

The data directory contains transforms_train.json and its RGBA image files. Warp owns the
parallel raster, explicit backward pass, and Adam updates. Python owns image loading, camera
selection, discrete tile records, densification, pruning, PLY snapshots, and validation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from time import perf_counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import warp as wp
from warp.optim import Adam
import yaml

# Shared data and parameter classes are kept outside the renderer/trainer modules.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# Shared modules sit beside this file in the assignment repository.
_shared = _here / "shared"
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

from trainable_gaussian import TrainableGaussianSet
from dssim_loss import DSSIMComputer
from sh_appearance import SHAppearance
from optimizer_state import CHILD_STATE_POLICIES, copy_state_rows, clear_state_rows, inherit_adam_rows, initialize_adam_rows
from gaussian_set import GaussianSet
from profiling import profile_stage
from splat_math import (
    ALPHA_CUTOFF,
    SUPPORT_RADIUS_SQUARED,
    TILE,
    TRANSMITTANCE_CUTOFF,
    quaternion_to_matrix,
)
# Bound here as well as imported: the later trainers and the evaluation scripts reach these
# through this module, and moving them out from under those names would be a change to every
# caller rather than to where the code lives.
from training_config import (
    DEFAULT_INITIAL_SPLATS,
    DEFAULT_RESOLUTION,
    DEFAULT_SPLATS,
    TrainingConfig,
)


NEAR_PLANE = 0.1
FILTER_VARIANCE = 0.3
INACTIVE_LOGIT = -12.0
# Splats with opacity below this are pruned outright, independent of the ADC opacity_cutoff
# config knob above it in severity -- this is the unconditional floor every splat must clear.
PRUNE_OPACITY_FLOOR = 0.005
# constrain_parameters clamps log-scale to [log(scene_radius * MIN), log(scene_radius * MAX)].
SCALE_CLAMP_MIN_FRACTION = 1.0 / 500.0
SCALE_CLAMP_MAX_FRACTION = 1.0 / 3.0
# warp.optim.Adam's own defaults. The quaternion needs a matching hand-written Adam step because
# Adam rejects vec4 parameters; plain SGD at an Adam-tuned rate leaves gradients this small
# (~1e-9, since the loss averages over every pixel and splat) unable to move the parameter at
# all, freezing every splat's orientation at initialization.
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1.0e-8

def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def exponential_learning_rate(
    step: int,
    initial: float,
    final: float,
    max_steps: int,
    delay_steps: int = 0,
    delay_multiplier: float = 1.0,
) -> float:
    interpolation = np.clip(step / max_steps, 0.0, 1.0)
    learning_rate = np.exp(
        np.log(initial) * (1.0 - interpolation) + np.log(final) * interpolation
    )
    if delay_steps > 0:
        delay_progress = np.clip(step / delay_steps, 0.0, 1.0)
        delay = delay_multiplier + (1.0 - delay_multiplier) * np.sin(
            0.5 * np.pi * delay_progress
        )
        learning_rate *= delay
    return float(learning_rate)


def blender_pose_to_world_to_camera(transform: list[list[float]]) -> np.ndarray:
    camera_to_world = np.array(transform, dtype=np.float64)
    camera_to_world[:3, 1:3] *= -1.0
    return np.linalg.inv(camera_to_world).astype(np.float32)


def projected_covariance_diagonal(
    scales: np.ndarray,
    quaternions: np.ndarray,
    camera: np.ndarray,
    camera_points: np.ndarray,
    focal: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the x/y diagonal of each projected 2D covariance."""
    quaternions = quaternions / np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-8
    )
    rotation = quaternion_to_matrix(quaternions)
    x, y, z = camera_points.T
    jacobian_x = np.stack((focal / z, np.zeros_like(z), -focal * x / (z * z)), axis=1)
    jacobian_y = np.stack((np.zeros_like(z), focal / z, -focal * y / (z * z)), axis=1)
    projected_x = jacobian_x @ camera[:3, :3]
    projected_y = jacobian_y @ camera[:3, :3]
    local_x = np.einsum("ni,nij->nj", projected_x, rotation)
    local_y = np.einsum("ni,nij->nj", projected_y, rotation)
    scales_squared = scales * scales
    variance_x = np.sum(scales_squared * local_x * local_x, axis=1) + FILTER_VARIANCE
    variance_y = np.sum(scales_squared * local_y * local_y, axis=1) + FILTER_VARIANCE
    return variance_x, variance_y


def resolve_frame_image(data: Path, frame_path: str) -> Path:
    """Resolve a NeRF-synthetic frame path that may point to PNG, JPG, or JPEG."""
    base = data / frame_path
    candidates = []
    if base.suffix:
        candidates.append(base)
        candidates.extend(base.with_suffix(suffix) for suffix in (".png", ".jpg", ".jpeg"))
    else:
        candidates.extend(base.with_suffix(suffix) for suffix in (".png", ".jpg", ".jpeg"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find image for frame '{frame_path}'. Tried: {tried}")


def load_views(
    data: Path,
    width: int,
    height: int,
    background: np.ndarray,
    manifest_name: str = "transforms_train.json",
    frame_ids=None,
):
    """Load a manifest's views, or only the frames named by `frame_ids`.

    `frame_ids` exists so the held-out evaluation set costs eight image decodes rather than the
    whole validation manifest: the frames are chosen from the poses, which are in the JSON, before
    any pixels are read.
    """
    manifest = json.loads((data / manifest_name).read_text())
    focal = np.float32(0.5 * width / np.tan(0.5 * manifest["camera_angle_x"]))
    frames = manifest["frames"]
    if frame_ids is not None:
        frames = [frames[int(frame_id)] for frame_id in frame_ids]
    images, cameras, distances, frame_paths = [], [], [], []
    for frame in frames:
        image = Image.open(resolve_frame_image(data, frame["file_path"])).convert("RGBA")
        rgba = np.asarray(image.resize((width, height), Image.LANCZOS), np.float32) / 255.0
        alpha = rgba[..., 3:4]
        images.append(rgba[..., :3] * alpha + (1.0 - alpha) * background)
        cameras.append(blender_pose_to_world_to_camera(frame["transform_matrix"]))
        distances.append(np.linalg.norm(np.asarray(frame["transform_matrix"], np.float32)[:3, 3]))
        frame_paths.append(frame["file_path"])
    return np.stack(images), np.stack(cameras), focal, float(np.mean(distances)), frame_paths


def read_manifest(data: Path, manifest_name: str) -> dict:
    """Read one transforms manifest without decoding any image."""
    path = data / manifest_name
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation manifest '{manifest_name}' not found in {data}. Point "
            "reporting.eval_manifest at the held-out manifest, or set it to null to evaluate "
            "on training views."
        )
    return json.loads(path.read_text())


def select_diverse_frame_ids(transform_matrices, count: int) -> list:
    """Choose `count` frames whose cameras are as spread out over the sphere as possible.

    Farthest-point sampling on the unit camera directions, seeded at frame 0. Evenly *striding*
    the manifest is not the same thing: the NeRF-synthetic validation cameras walk a spiral rather
    than an ordered sweep, so on lego's 100 validation frames eight evenly spaced indices include a
    pair only 2.4 degrees apart -- effectively the same view measured twice -- while these eight
    are at least 41.4 degrees apart. Sampling on direction rather than position ignores the small
    radius differences between frames, which are not what makes a view informative.

    Deterministic: `argmax` breaks ties toward the lowest index, and the result is returned in
    manifest order so a run's evaluation set can be read straight out of the log.
    """
    positions = np.asarray(
        [np.asarray(matrix, np.float64)[:3, 3] for matrix in transform_matrices], np.float64,
    )
    if len(positions) == 0:
        raise ValueError("The evaluation manifest contains no frames.")
    directions = positions / np.maximum(
        np.linalg.norm(positions, axis=1, keepdims=True), 1.0e-12,
    )
    count = max(1, min(count, len(directions)))
    chosen = [0]
    distance = np.sum((directions - directions[0]) ** 2, axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.sum((directions - directions[nxt]) ** 2, axis=1))
    return sorted(chosen)


def load_evaluation_views(
    data: Path, width: int, height: int, background, reporting: dict,
    train_manifest_name: str = "transforms_train.json",
):
    """Resolve the held-out evaluation views, or `None` when evaluation stays on training views.

    Returns `(targets, cameras, frame_ids)` for the frames selected out of
    `reporting.eval_manifest`.
    """
    manifest_name = reporting["eval_manifest"]
    if manifest_name is None:
        return None
    manifest = read_manifest(data, manifest_name)
    # The trainer holds one focal length for every view, taken from the training manifest, so a
    # held-out manifest shot with a different field of view would be rendered with the wrong
    # camera and read as a quality problem rather than a configuration one.
    train_angle = float(read_manifest(data, train_manifest_name)["camera_angle_x"])
    eval_angle = float(manifest["camera_angle_x"])
    if abs(eval_angle - train_angle) > 1.0e-6:
        raise ValueError(
            f"'{manifest_name}' has camera_angle_x={eval_angle:.8f}, but the training manifest "
            f"'{train_manifest_name}' has {train_angle:.8f}. Held-out views must share the "
            "training field of view."
        )
    frames = manifest["frames"]
    frame_ids = reporting["eval_view_ids"]
    if frame_ids is None:
        frame_ids = select_diverse_frame_ids(
            [frame["transform_matrix"] for frame in frames], reporting["eval_views"],
        )
    else:
        out_of_range = [i for i in frame_ids if i >= len(frames)]
        if out_of_range:
            raise ValueError(
                f"reporting.eval_view_ids {out_of_range} are past the end of "
                f"'{manifest_name}', which has {len(frames)} frames."
            )
        frame_ids = sorted(frame_ids)
    targets, cameras, _, _, _ = load_views(
        data, width, height, background, manifest_name=manifest_name, frame_ids=frame_ids,
    )
    return targets, cameras, frame_ids


def append_evaluation_views(targets, cameras, held_out):
    """Place the held-out views after the training views in one shared view-id space.

    The trainer indexes cameras and ground truth by a single view id, so held-out views have to
    live in the same arrays. Keeping them at the end means the training loop's random view draw,
    `rng.integers(len(targets))` over the *training* array it is still handed, can never reach
    them -- which is the whole point of holding them out.
    """
    if held_out is None:
        return targets, cameras, None
    held_targets, held_cameras, _ = held_out
    eval_view_ids = np.arange(
        len(targets), len(targets) + len(held_targets), dtype=np.int32,
    )
    return (
        np.concatenate((targets, held_targets)),
        np.concatenate((cameras, held_cameras)),
        eval_view_ids,
    )


def training_view_eval_ids(targets, reporting: dict) -> np.ndarray:
    """Fallback evaluation set: evenly spaced *training* views, the pre-held-out behaviour."""
    eval_count = min(reporting["eval_views"], len(targets))
    return np.linspace(0, len(targets) - 1, eval_count, dtype=np.int32)


def resolve_evaluation_views(
    targets, cameras, data: Path, background, reporting: dict,
    train_manifest_name: str = "transforms_train.json",
):
    """Build the trainer's view arrays and the fixed-evaluation view ids from one config.

    All three trainers call this, so the held-out set is chosen the same way in each. `targets`
    and `cameras` are the *training* views as loaded; the returned arrays are those plus the
    held-out views appended, and the caller must keep passing the training-only `targets` to the
    training loop so the random view draw stays inside the training set.

    Returns `(all_targets, all_cameras, eval_view_ids, held_out_frame_ids)`, where the last is
    `None` when evaluation falls back to evenly spaced training views.
    """
    height, width = targets.shape[1], targets.shape[2]
    held_out = load_evaluation_views(
        data, width, height, background, reporting, train_manifest_name,
    )
    all_targets, all_cameras, eval_view_ids = append_evaluation_views(targets, cameras, held_out)
    if eval_view_ids is None:
        return all_targets, all_cameras, training_view_eval_ids(targets, reporting), None
    return all_targets, all_cameras, eval_view_ids, held_out[2]


def describe_evaluation_views(eval_view_ids, held_out_frame_ids, manifest_name) -> str:
    """One log line saying exactly which cameras `fixed_eval` is measured on."""
    if held_out_frame_ids is None:
        return f"eval_views={list(map(int, eval_view_ids))} (training views)"
    return (
        f"eval_views={list(map(int, eval_view_ids))} "
        f"(held out: {manifest_name} frames {list(map(int, held_out_frame_ids))})"
    )


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
    b0_squared = wp.vec3(b0[0] * b0[0], b0[1] * b0[1], b0[2] * b0[2])
    b0_b1 = wp.vec3(b0[0] * b1[0], b0[1] * b1[1], b0[2] * b1[2])
    b1_squared = wp.vec3(b1[0] * b1[0], b1[1] * b1[1], b1[2] * b1[2])
    a = wp.dot(scales_squared, b0_squared) + FILTER_VARIANCE
    b = wp.dot(scales_squared, b0_b1)
    c = wp.dot(scales_squared, b1_squared) + FILTER_VARIANCE
    return wp.vec3(a, b, c)


@wp.func
def conic_at_centre(log_scale: wp.vec3, quaternion: wp.vec4, camera: wp.mat44, x: float, y: float, z: float, focal: float):
    covariance = projected_covariance(log_scale, quaternion, camera, x, y, z, focal)
    a, b, c = covariance[0], covariance[1], covariance[2]
    determinant = wp.max(a * c - b * b, 1.0e-8)
    return wp.vec3(c / determinant, -b / determinant, a / determinant)


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
    compact_enabled: int,
    compact_beta: float,
    compact_alpha_min: float,
):
    """Alpha of one splat at one pixel.

    The cutoff is the splat's own compact-support radius, not a fixed 3-sigma disc: a faint splat
    is truncated sooner because it reaches ALPHA_CUTOFF sooner. This is the same rule the tile
    workspace already uses to decide which tiles a splat touches, and the same rule the validation
    rasterizer shades with, so training, tiling, and validation now agree. Passing the compact-box
    settings in (rather than reading a module constant) keeps `compact_box.enabled: false` working,
    where the cutoff falls back to the fixed SUPPORT_RADIUS_SQUARED disc.
    """
    point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)
    z = wp.max(point[2], NEAR_PLANE)
    centre_x = focal * point[0] / z + 0.5 * width
    centre_y = focal * point[1] / z + 0.5 * height
    conic = conic_at_centre(log_scale, quaternion, camera, point[0], point[1], z, focal)
    dx, dy = px - centre_x, py - centre_y
    q = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy
    opacity = 1.0 / (1.0 + wp.exp(-opacity_logit))
    support = float(SUPPORT_RADIUS_SQUARED)
    if compact_enabled != 0:
        support = 0.0
        if opacity > compact_alpha_min:
            support = wp.min(
                float(SUPPORT_RADIUS_SQUARED),
                compact_beta * 2.0 * wp.log(opacity / compact_alpha_min),
            )
    alpha = float(0.0)
    if support > 0.0 and q <= support:
        candidate = wp.min(opacity * wp.exp(-0.5 * q), 0.99)
        if candidate >= ALPHA_CUTOFF:
            alpha = candidate
    return alpha


@wp.func
def colour_at_view(
    color: wp.vec3,
):
    return wp.vec3(
        wp.clamp(color[0], 0.0, 1.0),
        wp.clamp(color[1], 0.0, 1.0),
        wp.clamp(color[2], 0.0, 1.0),
    )


@wp.kernel
def render_forward(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    color: wp.array(dtype=wp.vec3),
    cameras: wp.array(dtype=wp.mat44),
    targets: wp.array(dtype=wp.vec3),
    view_ids: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    pairs: wp.array(dtype=wp.uint64),
    width: int,
    height: int,
    tiles_x: int,
    tiles_y: int,
    tile: int,
    focal: float,
    view_count: int,
    background: wp.vec3,
    compact_enabled: int,
    compact_beta: float,
    compact_alpha_min: float,
    image: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):
    thread = wp.tid()
    pixels = width * height
    batch_view = thread // pixels
    view = view_ids[batch_view]
    pixel = thread - batch_view * pixels
    px = float(pixel % width) + 0.5
    py = float(pixel // width) + 0.5
    record = batch_view * tiles_x * tiles_y + (pixel // width // tile) * tiles_x + (pixel % width // tile)
    rgb = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)
    for entry in range(offsets[record], offsets[record + 1]):
        splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        alpha = alpha_at_pixel(means[splat], log_scales[splat], quaternions[splat],
                               opacity_logits[splat], cameras[view], px, py,
                               float(width), float(height), focal, compact_enabled, compact_beta, compact_alpha_min)
        if alpha > 0.0:
            colour = colour_at_view(color[splat])
            rgb = rgb + transmittance * alpha * colour
            transmittance = transmittance * (1.0 - alpha)
            if transmittance < TRANSMITTANCE_CUTOFF:
                break
    rgb = rgb + transmittance * background
    image[thread] = rgb
    difference = rgb - targets[view * pixels + pixel]
    wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * pixels * view_count))


@wp.kernel(enable_backward=False)
def render_backward(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    color: wp.array(dtype=wp.vec3),
    cameras: wp.array(dtype=wp.mat44),
    view_ids: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    pairs: wp.array(dtype=wp.uint64),
    width: int,
    height: int,
    tiles_x: int,
    tiles_y: int,
    tile: int,
    focal: float,
    image: wp.array(dtype=wp.vec3),
    pixel_gradient: wp.array(dtype=wp.vec3),
    compact_enabled: int,
    compact_beta: float,
    compact_alpha_min: float,
    mean_grad_flat: wp.array(dtype=wp.float32),
    scale_grad_flat: wp.array(dtype=wp.float32),
    quaternion_grad_flat: wp.array(dtype=wp.float32),
    opacity_grad: wp.array(dtype=wp.float32),
    color_grad_flat: wp.array(dtype=wp.float32),
):
    thread = wp.tid()
    pixels = width * height
    batch_view = thread // pixels
    view = view_ids[batch_view]
    pixel = thread - batch_view * pixels
    px = float(pixel % width) + 0.5
    py = float(pixel // width) + 0.5
    record = batch_view * tiles_x * tiles_y + (pixel // width // tile) * tiles_x + (pixel % width // tile)
    final_rgb = image[thread]
    pixel_grad = pixel_gradient[thread]
    prefix_rgb = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)

    # This is an explicit adjoint of front-to-back compositing. It is an ordinary forward
    # execution of a runtime loop, so Warp never has to reverse a variable-length loop.
    for entry in range(offsets[record], offsets[record + 1]):
        splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        mean = means[splat]
        log_scale = log_scales[splat]
        quaternion = quaternions[splat]
        opacity_logit = opacity_logits[splat]
        camera = cameras[view]
        alpha = alpha_at_pixel(mean, log_scale, quaternion, opacity_logit, camera,
                               px, py, float(width), float(height), focal, compact_enabled, compact_beta, compact_alpha_min)
        if alpha > 0.0:
            colour = colour_at_view(color[splat])
            next_transmittance = transmittance * (1.0 - alpha)

            # TODO: accumulate this splat's share of the gradient.
            #
            # The forward pass composited front to back:
            #     image = sum over splats of (transmittance * alpha * colour)
            #             + background * final transmittance
            # and `pixel_grad` holds d(loss)/d(this pixel). Walk the same splats in the same
            # order and push that pixel gradient back onto the parameters.
            #
            # Two adjoints to derive:
            #   colour  -- how much of the pixel gradient this splat's colour receives, given
            #              the transmittance in front of it and its own alpha. Note the forward
            #              clamps colour to [0, 1]. At an endpoint use the interior
            #              derivative so projected updates can recover inward.
            #   alpha   -- raising alpha adds this splat's colour and removes whatever the
            #              splats behind it contribute. `remaining_rgb` below is that trailing
            #              contribution, recovered from the final pixel value rather than by a
            #              second pass.
            #
            # From the alpha adjoint, `wp.grad(alpha_at_pixel)(...)` gives the adjoints of the
            # splat parameters. It returns one value per argument of `alpha_at_pixel`, and a
            # Warp kernel cannot star-unpack, so every returned adjoint needs a name even where
            # it is unused. Scale each by the alpha adjoint before accumulating.
            #
            # Write results with `wp.atomic_add` into mean_grad_flat, scale_grad_flat,
            # quaternion_grad_flat, opacity_grad and color_grad_flat -- many pixels touch the
            # same splat concurrently. The flat buffers are component-major: splat i's mean
            # occupies indices 3*i, 3*i+1, 3*i+2, and its quaternion 4*i .. 4*i+3.
            
            remaining_rgb = (final_rgb - prefix_rgb - transmittance * alpha * colour) / wp.max(next_transmittance, 1.0e-8) #R1
            alpha_adjoint  = transmittance * wp.dot(pixel_grad, colour - remaining_rgb)   # scalar, # dL/d(alpha): scalar
            colour_adjoint = transmittance * alpha * pixel_grad                            # dL/d(colour): vec3
            (
                mean_adjoint,
                scale_adjoint,
                quaternion_adjoint,
                opacity_adjoint,
                camera_adjoint,
                px_adjoint,
                py_adjoint,
                width_adjoint,
                height_adjoint,
                focal_adjoint,
                compact_enabled_adjoint,
                compact_beta_adjoint,
                compact_alpha_min_adjoint,
            ) = wp.grad(alpha_at_pixel)(mean, log_scale, quaternion, opacity_logit, camera, px, py, float(width), float(height), focal, compact_enabled, compact_beta, compact_alpha_min)

            wp.atomic_add(mean_grad_flat, splat * 3 + 0, alpha_adjoint * mean_adjoint[0] )
            wp.atomic_add(mean_grad_flat, splat * 3 + 1, alpha_adjoint * mean_adjoint[1])
            wp.atomic_add(mean_grad_flat, splat * 3 + 2, alpha_adjoint * mean_adjoint[2])

            wp.atomic_add(scale_grad_flat, splat * 3 + 0, alpha_adjoint * scale_adjoint[0])
            wp.atomic_add(scale_grad_flat, splat * 3 + 1, alpha_adjoint * scale_adjoint[1])
            wp.atomic_add(scale_grad_flat, splat * 3 + 2, alpha_adjoint * scale_adjoint[2])

            wp.atomic_add(
                quaternion_grad_flat, splat * 4 + 0, alpha_adjoint * quaternion_adjoint[0]
            )
            wp.atomic_add(
                quaternion_grad_flat, splat * 4 + 1, alpha_adjoint *  quaternion_adjoint[1]
            )
            wp.atomic_add(
                quaternion_grad_flat, splat * 4 + 2, alpha_adjoint * quaternion_adjoint[2]
            )
            wp.atomic_add(
                quaternion_grad_flat, splat * 4 + 3, alpha_adjoint * quaternion_adjoint[3]
            )

            wp.atomic_add(opacity_grad, splat, alpha_adjoint * opacity_adjoint)

            raw_colour = color[splat]

            if raw_colour[0] >= 0.0 and raw_colour[0] <= 1.0:
                wp.atomic_add(
                    color_grad_flat, splat * 3 + 0, colour_adjoint[0]
                )

            if raw_colour[1] >= 0.0 and raw_colour[1] <= 1.0:
                wp.atomic_add(
                    color_grad_flat, splat * 3 + 1, colour_adjoint[1]
                )

            if raw_colour[2] >= 0.0 and raw_colour[2] <= 1.0:
                wp.atomic_add(
                    color_grad_flat, splat * 3 + 2, colour_adjoint[2]
                )

            prefix_rgb = prefix_rgb + transmittance * alpha * colour
            transmittance = next_transmittance
            if transmittance < TRANSMITTANCE_CUTOFF:
                break


@wp.kernel
def mse_pixel_gradient(
    image: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    view_ids: wp.array(dtype=wp.int32),
    width: int,
    height: int,
    view_count: int,
    pixel_gradient: wp.array(dtype=wp.vec3),
):
    """d(MSE)/d(pixel) for the dense path, the counterpart of `sparse_mse_pixel_gradient`.

    Computing this into a buffer rather than inline in the backward kernel is what lets a later
    version replace the gradient with one from a different objective before the backward runs.
    """
    thread = wp.tid()
    pixels = width * height
    batch_view = thread // pixels
    pixel = thread - batch_view * pixels
    target_pixel = view_ids[batch_view] * pixels + pixel
    difference = image[thread] - target[target_pixel]
    pixel_gradient[thread] = difference * (2.0 / float(3 * pixels * view_count))


@wp.kernel
def gather_sample_targets(
    targets: wp.array(dtype=wp.vec3),
    sample_xy: wp.array(dtype=wp.vec2i),
    view: int,
    width: int,
    height: int,
    sample_gt: wp.array(dtype=wp.vec3),
):
    sample = wp.tid()
    xy = sample_xy[sample]
    sample_gt[sample] = targets[view * width * height + xy[1] * width + xy[0]]


@wp.kernel
def sparse_mse_pixel_gradient(
    rendered_rgb: wp.array(dtype=wp.vec3),
    sample_gt: wp.array(dtype=wp.vec3),
    num_samples: int,
    pixel_gradient: wp.array(dtype=wp.vec3),
):
    sample = wp.tid()
    pixel_gradient[sample] = 2.0 * (rendered_rgb[sample] - sample_gt[sample]) / float(
        3 * num_samples
    )


@wp.kernel
def render_sparse_forward(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    color: wp.array(dtype=wp.vec3),
    cameras: wp.array(dtype=wp.mat44),
    view: int,
    sample_xy: wp.array(dtype=wp.vec2i),
    offsets: wp.array(dtype=wp.int32),
    pairs: wp.array(dtype=wp.uint64),
    samples_per_tile: int,
    width: int,
    height: int,
    focal: float,
    sample_gt: wp.array(dtype=wp.vec3),
    background: wp.vec3,
    num_samples: int,
    compact_enabled: int,
    compact_beta: float,
    compact_alpha_min: float,
    rendered_rgb: wp.array(dtype=wp.vec3),
    last_contributor: wp.array(dtype=wp.int32),
    loss: wp.array(dtype=wp.float32),
):
    thread = wp.tid()
    record = thread // samples_per_tile
    xy = sample_xy[thread]
    px = float(xy[0]) + 0.5
    py = float(xy[1]) + 0.5
    camera = cameras[view]
    rgb = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)
    last_entry = offsets[record] - 1
    for entry in range(offsets[record], offsets[record + 1]):
        splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        alpha = alpha_at_pixel(
            means[splat], log_scales[splat], quaternions[splat], opacity_logits[splat],
            camera, px, py, float(width), float(height), focal,
            compact_enabled, compact_beta, compact_alpha_min,
        )
        if alpha > 0.0:
            colour = colour_at_view(color[splat])
            rgb = rgb + transmittance * alpha * colour
            transmittance = transmittance * (1.0 - alpha)
            last_entry = entry
            if transmittance < TRANSMITTANCE_CUTOFF:
                break
    rgb = rgb + transmittance * background
    rendered_rgb[thread] = rgb
    last_contributor[thread] = last_entry
    difference = rgb - sample_gt[thread]
    wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * num_samples))


@wp.kernel(enable_backward=False)
def render_sparse_backward(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    color: wp.array(dtype=wp.vec3),
    cameras: wp.array(dtype=wp.mat44),
    view: int,
    sample_xy: wp.array(dtype=wp.vec2i),
    offsets: wp.array(dtype=wp.int32),
    pairs: wp.array(dtype=wp.uint64),
    samples_per_tile: int,
    width: int,
    height: int,
    focal: float,
    rendered_rgb: wp.array(dtype=wp.vec3),
    last_contributor: wp.array(dtype=wp.int32),
    pixel_gradient: wp.array(dtype=wp.vec3),
    compact_enabled: int,
    compact_beta: float,
    compact_alpha_min: float,
    mean_grad_flat: wp.array(dtype=wp.float32),
    scale_grad_flat: wp.array(dtype=wp.float32),
    quaternion_grad_flat: wp.array(dtype=wp.float32),
    opacity_grad: wp.array(dtype=wp.float32),
    color_grad_flat: wp.array(dtype=wp.float32),
):
    thread = wp.tid()
    record = thread // samples_per_tile
    xy = sample_xy[thread]
    px = float(xy[0]) + 0.5
    py = float(xy[1]) + 0.5
    camera = cameras[view]
    final_rgb = rendered_rgb[thread]
    pixel_grad = pixel_gradient[thread]
    prefix_rgb = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)
    stop = last_contributor[thread]
    for entry in range(offsets[record], stop + 1):
        splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        mean = means[splat]
        log_scale = log_scales[splat]
        quaternion = quaternions[splat]
        opacity_logit = opacity_logits[splat]
        alpha = alpha_at_pixel(
            mean, log_scale, quaternion, opacity_logit, camera,
            px, py, float(width), float(height), focal,
            compact_enabled, compact_beta, compact_alpha_min,
        )
        if alpha > 0.0:
            colour = colour_at_view(color[splat])
            next_transmittance = transmittance * (1.0 - alpha)

            # TODO: accumulate this splat's share of the gradient, exactly as in
            # `render_backward` above. The compositing algebra is identical; only the addressing
            # differs, because one thread here owns one sampled pixel rather than one dense
            # pixel, and the walk stops at `last_contributor[thread]` -- the entry where the
            # forward pass stopped -- instead of at the end of the tile's splat list.
            #
            # Getting this to agree with the dense kernel matters, and full coverage is how you
            # check it: at `sparse.samples_per_tile == TILE * TILE` every pixel in every tile is
            # sampled exactly once, so the sparse loss and every sparse gradient buffer must match
            # the dense ones.
            remaining_rgb = (final_rgb - prefix_rgb - transmittance * alpha * colour) / wp.max(next_transmittance, 1.0e-8) #R1
            alpha_adjoint  = transmittance * wp.dot(pixel_grad, colour - remaining_rgb)   # scalar, # dL/d(alpha): scalar
            colour_adjoint = transmittance * alpha * pixel_grad                            # dL/d(colour): vec3
            (
                mean_adjoint,
                scale_adjoint,
                quaternion_adjoint,
                opacity_adjoint,
                camera_adjoint,
                px_adjoint,
                py_adjoint,
                width_adjoint,
                height_adjoint,
                focal_adjoint,
                compact_enabled_adjoint,
                compact_beta_adjoint,
                compact_alpha_min_adjoint,
            ) = wp.grad(alpha_at_pixel)(mean, log_scale, quaternion, opacity_logit, camera, px, py, float(width), float(height), focal, compact_enabled, compact_beta, compact_alpha_min)

            wp.atomic_add(mean_grad_flat, splat * 3 + 0, alpha_adjoint * mean_adjoint[0] )
            wp.atomic_add(mean_grad_flat, splat * 3 + 1, alpha_adjoint * mean_adjoint[1])
            wp.atomic_add(mean_grad_flat, splat * 3 + 2, alpha_adjoint * mean_adjoint[2])

            wp.atomic_add(scale_grad_flat, splat * 3 + 0, alpha_adjoint * scale_adjoint[0])
            wp.atomic_add(scale_grad_flat, splat * 3 + 1, alpha_adjoint * scale_adjoint[1])
            wp.atomic_add(scale_grad_flat, splat * 3 + 2, alpha_adjoint * scale_adjoint[2])

            wp.atomic_add(
                quaternion_grad_flat, splat * 4 + 0, alpha_adjoint * quaternion_adjoint[0]
            )
            wp.atomic_add(
                quaternion_grad_flat, splat * 4 + 1, alpha_adjoint *  quaternion_adjoint[1]
            )
            wp.atomic_add(
                quaternion_grad_flat, splat * 4 + 2, alpha_adjoint * quaternion_adjoint[2]
            )
            wp.atomic_add(
                quaternion_grad_flat, splat * 4 + 3, alpha_adjoint * quaternion_adjoint[3]
            )

            wp.atomic_add(opacity_grad, splat, alpha_adjoint * opacity_adjoint)

            raw_colour = color[splat]

            if raw_colour[0] >= 0.0 and raw_colour[0] <= 1.0:
                wp.atomic_add(
                    color_grad_flat, splat * 3 + 0, colour_adjoint[0]
                )

            if raw_colour[1] >= 0.0 and raw_colour[1] <= 1.0:
                wp.atomic_add(
                    color_grad_flat, splat * 3 + 1, colour_adjoint[1]
                )

            if raw_colour[2] >= 0.0 and raw_colour[2] <= 1.0:
                wp.atomic_add(
                    color_grad_flat, splat * 3 + 2, colour_adjoint[2]
                )
            

            prefix_rgb = prefix_rgb + transmittance * alpha * colour
            transmittance = next_transmittance


@wp.kernel
def pack_vec3_gradient(flat: wp.array(dtype=wp.float32), packed: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    packed[i] = wp.vec3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2])


@wp.kernel
def adam_quaternion_step(
    quaternions: wp.array(dtype=wp.vec4),
    gradients: wp.array(dtype=wp.float32),
    m: wp.array(dtype=wp.vec4),
    v: wp.array(dtype=wp.vec4),
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
):
    # Same update as warp.optim.Adam's own float/vec3 kernels, generalized to four components
    # because Adam does not support wp.vec4 parameters directly.
    i = wp.tid()
    g = wp.vec4(gradients[i * 4], gradients[i * 4 + 1], gradients[i * 4 + 2], gradients[i * 4 + 3])
    m[i] = beta1 * m[i] + (1.0 - beta1) * g
    v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g, g)
    mhat = m[i] / (1.0 - wp.pow(beta1, t + 1.0))
    vhat = v[i] / (1.0 - wp.pow(beta2, t + 1.0))
    sqrt_vhat = wp.vec4(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]), wp.sqrt(vhat[3]))
    eps_vec4 = wp.vec4(eps, eps, eps, eps)
    q = quaternions[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec4))
    quaternions[i] = q / wp.sqrt(wp.dot(q, q) + 1.0e-8)


@wp.kernel
def constrain_parameters(
    log_scales: wp.array(dtype=wp.vec3),
    color: wp.array(dtype=wp.vec3),
    minimum_log_scale: float,
    maximum_log_scale: float,
):
    i = wp.tid()
    scale = log_scales[i]
    log_scales[i] = wp.vec3(
        wp.clamp(scale[0], minimum_log_scale, maximum_log_scale),
        wp.clamp(scale[1], minimum_log_scale, maximum_log_scale),
        wp.clamp(scale[2], minimum_log_scale, maximum_log_scale),
    )
    colour = color[i]
    color[i] = wp.vec3(
        wp.clamp(colour[0], 0.0, 1.0),
        wp.clamp(colour[1], 0.0, 1.0),
        wp.clamp(colour[2], 0.0, 1.0),
    )


class SplatOptimizers:
    """Adam state for every trainable splat parameter, plus the explicit gradient buffers the
    hand-written backward pass writes into.

    The quaternion gets its own hand-written Adam step (`adam_quaternion_step`) with its own
    moment buffers and step count, since `warp.optim.Adam` does not accept `vec4` parameters.
    Every gradient produced by the backward pass is consumed by Adam in the same iteration.
    """

    def __init__(self, trainable, capacity, learning_rates, device, child_state_policy="inherit"):
        if child_state_policy not in CHILD_STATE_POLICIES:
            raise ValueError("Unknown child optimizer-state policy.")
        self.child_state_policy = child_state_policy
        self.device = device
        self.learning_rates = learning_rates
        self.quaternions = trainable.quaternions

        self.mean_optimizer = Adam([trainable.means], lr=learning_rates["position_initial"])
        self.scale_optimizer = Adam([trainable.log_scales], lr=learning_rates["scale"])
        self.opacity_optimizer = Adam([trainable.opacity_logits], lr=learning_rates["opacity"])
        self.color_optimizer = Adam([trainable.colors], lr=learning_rates["feature_dc"])
        self.optimizers = (
            self.mean_optimizer,
            self.scale_optimizer,
            self.opacity_optimizer,
            self.color_optimizer,
        )
        self.quaternion_m = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.quaternion_v = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.quaternion_adam_t = 0

        self.mean_grad_flat = wp.zeros(capacity * 3, dtype=wp.float32, device=device)
        self.scale_grad_flat = wp.zeros(capacity * 3, dtype=wp.float32, device=device)
        self.quaternion_grad_flat = wp.zeros(capacity * 4, dtype=wp.float32, device=device)
        self.opacity_grad = wp.zeros(capacity, dtype=wp.float32, device=device)
        self.color_grad_flat = wp.zeros(capacity * 3, dtype=wp.float32, device=device)
        self.mean_grad = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.scale_grad = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.color_grad = wp.zeros(capacity, dtype=wp.vec3, device=device)

    def zero_gradients(self):
        for gradient in (self.mean_grad_flat, self.scale_grad_flat, self.quaternion_grad_flat,
                         self.opacity_grad, self.color_grad_flat):
            gradient.zero_()

    def pack_gradients(self, capacity):
        """Repack the flat per-component gradient buffers `render_backward` writes into
        vec3 arrays, for use as Adam inputs and as the densification statistic."""
        for flat, packed in ((self.mean_grad_flat, self.mean_grad),
                             (self.scale_grad_flat, self.scale_grad),
                             (self.color_grad_flat, self.color_grad)):
            wp.launch(pack_vec3_gradient, dim=capacity, inputs=[flat], outputs=[packed],
                      device=self.device)

    def apply(self, position_lr, capacity):
        """Apply one Adam update using the gradients from the current render."""
        self.mean_optimizer.lr = position_lr
        self.mean_optimizer.step([self.mean_grad])
        self.scale_optimizer.step([self.scale_grad])
        self.opacity_optimizer.step([self.opacity_grad])
        self.color_optimizer.step([self.color_grad])
        wp.launch(
            adam_quaternion_step,
            dim=capacity,
            inputs=[self.quaternions, self.quaternion_grad_flat,
                    self.quaternion_m, self.quaternion_v,
                    self.learning_rates["rotation"], ADAM_BETA1, ADAM_BETA2,
                    float(self.quaternion_adam_t), ADAM_EPS],
            device=self.device,
        )
        self.quaternion_adam_t += 1

    def reset(self):
        """Clear all Adam moments after densify/prune changes which parameters exist. A clone
        or split child needs fresh optimizer state -- see Unit 9, Stage 7."""
        for optimizer in self.optimizers:
            optimizer.reset_internal_state()
        self.quaternion_m.zero_()
        self.quaternion_v.zero_()
        self.quaternion_adam_t = 0

    def inherit_children(self, parents, children):
        """Initialize new slots under the configured optimizer-state policy.

        The historical method name remains for callers/probes. Fresh state does
        not copy parent moments; existing slots and clocks remain unchanged.
        """
        parent_ids = wp.array(np.asarray(parents, np.int32), dtype=wp.int32, device=self.device)
        child_ids = wp.array(np.asarray(children, np.int32), dtype=wp.int32, device=self.device)
        policy = getattr(self, 'child_state_policy', 'inherit')
        if not len(children):
            return parent_ids, child_ids
        if policy == 'legacy_reset':
            self.reset()
        else:
            for optimizer in self.optimizers:
                initialize_adam_rows(optimizer, parent_ids, child_ids, policy)
            for values in (self.quaternion_m, self.quaternion_v):
                if policy == 'inherit':
                    wp.launch(copy_state_rows, len(children), inputs=[values, parent_ids, child_ids], device=self.device)
                else:
                    wp.launch(clear_state_rows, len(children), inputs=[values, child_ids], device=self.device)
        return parent_ids, child_ids

    def reset_opacity(self):
        """Clear only the opacity optimizer's moments, for a periodic opacity ceiling reset."""
        self.opacity_optimizer.reset_internal_state()


class WarpImageTrainer:
    """Warp GPU tiler and differentiable renderer plus a Python training schedule."""

    def __init__(self, targets, cameras, focal, scene_radius, capacity, initial_splats,
                 device, seed, init_scene=None, background=(1.0, 1.0, 1.0),
                 learning_rates=None, tile_pair_capacity=None, training_options=None, sh_degree=0,
                 child_optimizer_state="inherit"):
        if child_optimizer_state not in CHILD_STATE_POLICIES:
            raise ValueError("Unknown child optimizer-state policy.")
        self.child_optimizer_state = child_optimizer_state
        self.focal = float(focal)
        self.height, self.width = targets.shape[1], targets.shape[2]
        self.capacity, self.device = capacity, wp.get_device(device)
        # Training always uses one randomly selected camera. Evaluation also renders one camera
        # at a time, so every image buffer can have a simple, fixed batch dimension of one.
        self.max_views_per_batch = 1
        self.learning_rates = deepcopy(TrainingConfig.DEFAULTS["learning_rates"])
        if learning_rates is not None:
            self.learning_rates.update(learning_rates)
        self.training_options = {
            "compact_box": deepcopy(TrainingConfig.DEFAULTS["compact_box"]),
            "multiview_adc": deepcopy(TrainingConfig.DEFAULTS["multiview_adc"]),
            "sparse": deepcopy(TrainingConfig.DEFAULTS["sparse"]),
            "loss": deepcopy(TrainingConfig.DEFAULTS["loss"]),
        }
        if training_options is not None:
            for section, values in training_options.items():
                self.training_options[section].update(values)
        self.sparse_enabled = self.training_options["sparse"]["enabled"]
        self.samples_per_tile = self.training_options["sparse"]["samples_per_tile"]
        self.last_position_learning_rate = self.learning_rates["position_initial"]
        self.loss_config = self.training_options["loss"]
        self.dssim_stage = 0
        self._dssim = None  # Allocate the image-loss workspace only after the stage transition.
        self.last_loss_phase = "sparse-mse" if self.sparse_enabled else "mse"
        self.background = wp.vec3(float(background[0]), float(background[1]), float(background[2]))
        self.tiles_x = (self.width + TILE - 1) // TILE
        self.tiles_y = (self.height + TILE - 1) // TILE
        self.tiles_per_view = self.tiles_x * self.tiles_y
        self.tile_pair_capacity = int(
            tile_pair_capacity
            if tile_pair_capacity is not None
            else 16 * capacity * self.max_views_per_batch
        )
        if not 1 <= self.tile_pair_capacity <= np.iinfo(np.int32).max:
            raise ValueError(
                "Tile-pair capacity must be positive and fit in a signed 32-bit prefix sum."
            )
        self.rng = np.random.default_rng(seed)
        self.scene_radius = scene_radius

        self._init_splats(capacity, initial_splats, init_scene, targets)
        self._init_buffers(cameras, targets, capacity)
        self.sh = SHAppearance(capacity, sh_degree, self.learning_rates["feature_rest"], self.device) if sh_degree else None

    @property
    def active_sh_degree(self):
        return self.sh.degree if self.sh is not None else 0

    def advance_sh_stage(self):
        if self.sh is None or self.sh.degree >= self.sh.max_degree:
            raise ValueError("No higher SH band is configured.")
        self.sh.degree += 1

    def _render_colours(self, view):
        if not self.active_sh_degree:
            return self.color
        return self.sh.forward(self.means, self.color, self.active_device, self.cameras, view)

    def _appearance_backward(self, view):
        if self.active_sh_degree:
            self.sh.backward(self.means, self.active_device, self.cameras, view,
                             self.optimizers.color_grad_flat, self.optimizers.mean_grad_flat)

    def _init_splats(self, capacity, initial_splats, init_scene, targets):
        """Allocate and populate every trainable per-splat parameter array."""
        means = np.zeros((capacity, 3), np.float32)
        log_scales = np.full((capacity, 3), np.log(self.scene_radius / 80.0), np.float32)
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (capacity, 1))
        opacity = np.full(capacity, INACTIVE_LOGIT, np.float32)
        colours = np.zeros((capacity, 3), np.float32)
        count = initial_splats
        if init_scene is not None:
            count = min(len(init_scene.means), capacity)
            means[:count], log_scales[:count] = init_scene.means[:count], np.log(np.maximum(init_scene.scales[:count], 1e-7))
            quaternions[:count], colours[:count] = init_scene.rotations[:count], init_scene.colors[:count]
            opacity[:count] = np.log(np.clip(init_scene.opacities[:count], 1e-5, 1 - 1e-5) / np.clip(1 - init_scene.opacities[:count], 1e-5, 1))
        else:
            means[:count] = self.rng.uniform(-self.scene_radius, self.scene_radius, (count, 3))
            colours[:count] = targets.mean(axis=(0, 1, 2))
            opacity[:count] = -4.0
        self.active = np.zeros(capacity, bool)
        self.active[:count] = True
        self.active_device = wp.array(self.active.astype(np.int32), dtype=wp.int32, device=self.device)
        self.trainable = TrainableGaussianSet(
            means, log_scales, quaternions, opacity, colours,
            self.device, requires_grad=False,
        )
        self.means = self.trainable.means
        self.log_scales = self.trainable.log_scales
        self.quaternions = self.trainable.quaternions
        self.opacity = self.trainable.opacity_logits
        self.color = self.trainable.colors

    def _init_buffers(self, cameras, targets, capacity):
        """Allocate camera/target uploads, the tile workspace, optimizers, and render buffers."""
        self.cameras = wp.array(cameras, dtype=wp.mat44, device=self.device)
        self.targets = wp.array(targets.reshape(-1, 3), dtype=wp.vec3, device=self.device)
        # Ground truth never changes after upload, but `multiview_adc_scores` needs a few views of
        # it on the host at every densify/late-prune event. Keep one host-side copy rather than
        # calling `self.targets.numpy()` there, which pulls the *entire* image set back across PCIe
        # (~700 MB at 100 views of 768x768) to read the handful of views ADC actually scores.
        self.targets_host = np.ascontiguousarray(
            targets, dtype=np.float32,
        ).reshape(-1, self.height, self.width, 3)
        workspace_module = importlib.import_module("gaussian_first_tile_workspace_gpu")
        self.tiles = workspace_module.GaussianFirstTileWorkspace(
            capacity, self.max_views_per_batch, self.tiles_x, self.tiles_y,
            self.tile_pair_capacity, self.device,
        )
        self.optimizers = SplatOptimizers(
            self.trainable, capacity, self.learning_rates, self.device,
            child_state_policy=self.child_optimizer_state,
        )
        self.last_pair_count = 0
        self.image = wp.empty(
            self.max_views_per_batch * self.width * self.height,
            dtype=wp.vec3,
            device=self.device,
        )
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.last_timings = {"tile": 0.0, "device": 0.0}
        # Only this trainer's own sparse path reads these. A wrapper that adds its own sparse
        # objective allocates its own set and drives this object dense, so allocating them
        # unconditionally reserved five per-sample buffers that such a run never touches.
        self.num_sparse_samples = (
            self.tiles_per_view * self.samples_per_tile if self.sparse_enabled else 0
        )
        self._sample_rng = np.random.default_rng(self.rng.integers(0, np.iinfo(np.int32).max))
        self.pixel_gradient = wp.zeros(
            self.max_views_per_batch * self.width * self.height, dtype=wp.vec3, device=self.device,
        )
        self.sample_xy = wp.zeros(self.num_sparse_samples, dtype=wp.vec2i, device=self.device)
        self.sample_gt = wp.zeros(self.num_sparse_samples, dtype=wp.vec3, device=self.device)
        self.sparse_rendered_rgb = wp.zeros(
            self.num_sparse_samples, dtype=wp.vec3, device=self.device,
        )
        self.sparse_last_contributor = wp.zeros(
            self.num_sparse_samples, dtype=wp.int32, device=self.device,
        )
        self.sparse_pixel_gradient = wp.zeros(
            self.num_sparse_samples, dtype=wp.vec3, device=self.device,
        )

    def build_tiles(self, view_ids):
        """Build compact, depth-ordered tile records on the selected Warp device."""
        return self.tiles.build(
            self.means, self.log_scales, self.quaternions, self.opacity, self.active_device,
            self.cameras, view_ids, self.width, self.height,
            self.focal, self.training_options["compact_box"],
        )

    def _draw_sparse_samples(self, samples_per_tile=None, rng=None) -> np.ndarray:
        """Host-side, NumPy-only: cheap relative to a kernel launch, and it is what keeps the
        per-tile grouping exact and free -- sample `s` belongs to tile `s // samples_per_tile` by
        construction, no sort needed. Re-drawn fresh every call (stochastic pixel sampling).

        At `samples_per_tile == TILE * TILE` ("full coverage"), offsets are the exact raster grid
        instead of a random draw -- every pixel in every tile is sampled exactly once, so the
        sparse path reduces bit-for-bit to the dense one. The sparse-versus-dense equivalence
        check relies on that, so do not make the full-coverage case random.

        The SH trainer samples through this method with its own count and its own generator,
        rather than keeping a second copy that has to be changed in step with this one. The
        generator has to be passed: that trainer seeds a separate stream, and drawing from this
        one instead silently changes which pixels every sparse iteration sees.
        """
        if samples_per_tile is None:
            samples_per_tile = self.samples_per_tile
        if rng is None:
            rng = self._sample_rng
        tile_index = np.repeat(
            np.arange(self.tiles_per_view, dtype=np.int32), samples_per_tile,
        )
        tile_x = tile_index % self.tiles_x
        tile_y = tile_index // self.tiles_x
        if samples_per_tile == TILE * TILE:
            raster = np.tile(np.arange(TILE * TILE, dtype=np.int32), self.tiles_per_view)
            offset_x = raster % TILE
            offset_y = raster // TILE
        else:
            offset_x = rng.integers(0, TILE, size=tile_index.shape, dtype=np.int32)
            offset_y = rng.integers(0, TILE, size=tile_index.shape, dtype=np.int32)
        px = np.minimum(tile_x * TILE + offset_x, self.width - 1)
        py = np.minimum(tile_y * TILE + offset_y, self.height - 1)
        return np.stack([px, py], axis=1).astype(np.int32)

    def dense_loss_and_gradients(self, view_ids, download_loss=True):
        if len(view_ids) > self.max_views_per_batch:
            raise ValueError("View batch exceeds the reusable image-buffer capacity.")
        started = perf_counter()
        offsets, pairs, pair_count = self.build_tiles(view_ids)
        self.last_pair_count = pair_count
        tile_seconds = perf_counter() - started
        device_started = perf_counter()
        self.loss.zero_()
        self.optimizers.zero_gradients()
        render_colours = self._render_colours(int(view_ids[0]))
        wp.launch(
            render_forward,
            dim=len(view_ids) * self.width * self.height,
            inputs=[self.means, self.log_scales, self.quaternions, self.opacity, render_colours,
                    self.cameras, self.targets, self.tiles.view_ids, offsets, pairs,
                    self.width, self.height, self.tiles_x, self.tiles_y,
                    TILE, self.focal, len(view_ids), self.background] + self.compact_box_args,
            outputs=[self.image, self.loss],
            device=self.device,
        )
        mse = float(self.loss.numpy()[0]) if download_loss else None
        wp.launch(
            mse_pixel_gradient,
            dim=len(view_ids) * self.width * self.height,
            inputs=[self.image, self.targets, self.tiles.view_ids,
                    self.width, self.height, len(view_ids)],
            outputs=[self.pixel_gradient],
            device=self.device,
        )
        objective = self._auxiliary_loss(mse, view_ids, download_loss)
        wp.launch(
            render_backward,
            dim=len(view_ids) * self.width * self.height,
            inputs=[self.means, self.log_scales, self.quaternions, self.opacity, render_colours,
                    self.cameras, self.tiles.view_ids, offsets, pairs,
                    self.width, self.height, self.tiles_x, self.tiles_y,
                    TILE, self.focal, self.image, self.pixel_gradient] + self.compact_box_args,
            outputs=[self.optimizers.mean_grad_flat, self.optimizers.scale_grad_flat,
                     self.optimizers.quaternion_grad_flat, self.optimizers.opacity_grad,
                     self.optimizers.color_grad_flat],
            device=self.device,
        )
        self._appearance_backward(int(view_ids[0]))
        self.optimizers.pack_gradients(self.capacity)
        self.last_timings = {"tile": tile_seconds, "device": perf_counter() - device_started}
        return objective

    def sparse_loss_and_gradients(self, view_ids, download_loss=True):
        if len(view_ids) != 1:
            raise ValueError("Sparse training expects exactly one view per step.")
        started = perf_counter()
        offsets, pairs, pair_count = self.build_tiles(view_ids)
        self.last_pair_count = pair_count
        tile_seconds = perf_counter() - started
        device_started = perf_counter()
        self.loss.zero_()
        self.optimizers.zero_gradients()
        samples = self._draw_sparse_samples()
        self.sample_xy.assign(samples)
        view = int(view_ids[0])
        render_colours = self._render_colours(view)
        wp.launch(
            gather_sample_targets,
            dim=self.num_sparse_samples,
            inputs=[self.targets, self.sample_xy, view, self.width, self.height],
            outputs=[self.sample_gt],
            device=self.device,
        )
        shared_inputs = [
            self.means, self.log_scales, self.quaternions, self.opacity, render_colours,
            self.cameras, view, self.sample_xy, offsets, pairs, self.samples_per_tile,
            self.width, self.height, self.focal,
        ]
        wp.launch(
            render_sparse_forward,
            dim=self.num_sparse_samples,
            inputs=shared_inputs + [
                self.sample_gt, self.background, self.num_sparse_samples,
            ] + self.compact_box_args,
            outputs=[self.sparse_rendered_rgb, self.sparse_last_contributor, self.loss],
            device=self.device,
        )
        mse = float(self.loss.numpy()[0]) if download_loss else None
        wp.launch(
            sparse_mse_pixel_gradient,
            dim=self.num_sparse_samples,
            inputs=[self.sparse_rendered_rgb, self.sample_gt, self.num_sparse_samples],
            outputs=[self.sparse_pixel_gradient],
            device=self.device,
        )
        wp.launch(
            render_sparse_backward,
            dim=self.num_sparse_samples,
            inputs=shared_inputs + [
                self.sparse_rendered_rgb, self.sparse_last_contributor,
                self.sparse_pixel_gradient,
            ] + self.compact_box_args,
            outputs=[
                self.optimizers.mean_grad_flat, self.optimizers.scale_grad_flat,
                self.optimizers.quaternion_grad_flat, self.optimizers.opacity_grad,
                self.optimizers.color_grad_flat,
            ],
            device=self.device,
        )
        self._appearance_backward(view)
        self.optimizers.pack_gradients(self.capacity)
        self.last_timings = {"tile": tile_seconds, "device": perf_counter() - device_started}
        return mse

    def loss_and_gradients(self, view_ids, download_loss=True):
        """Use sampled MSE during warm-up and whole images for the windowed objective."""
        if self.sparse_enabled and not self._dssim_active():
            self.last_loss_phase = "sparse-mse"
            return self.sparse_loss_and_gradients(view_ids, download_loss)
        return self.dense_loss_and_gradients(view_ids, download_loss)

    def _dssim_active(self):
        return self.loss_config["mode"] == "mse_dssim" and self.dssim_stage == 1

    @property
    def loss_stage(self):
        return "dssim" if self._dssim_active() else "mse"

    def advance_dssim_stage(self):
        """Latch the loss curriculum; the controller calls this only on a held-out eval tick."""
        if self.loss_config["mode"] != "mse_dssim":
            raise ValueError("D-SSIM stage requires loss.mode: mse_dssim.")
        self.dssim_stage = 1

    def _dssim_computer(self):
        if self._dssim is None:
            self._dssim = DSSIMComputer(
                self.width, self.height, self.max_views_per_batch, self.device,
            )
        return self._dssim

    def _auxiliary_loss(self, mse, view_ids, download_loss):
        """Rewrite the image gradient before the renderer propagates it to the splats."""
        self.last_loss_phase = self.loss_stage
        if not self._dssim_active():
            return mse
        weight = self.loss_config["dssim_weight"]
        dssim = self._dssim_computer().compute(
            self.image, self.targets, self.tiles.view_ids, self.pixel_gradient,
            self.loss_config["dssim_window"], len(view_ids), weight,
            download_loss=download_loss,
        )
        return (1.0 - weight) * mse + weight * dssim if download_loss else None

    def _render_view(self, view_id):
        """Forward-render one view into `self.image` and `self.loss`, both left on the device.

        The launch is here rather than written out at each call site because `evaluate` and
        `render_single_view_image` need the same eighteen arguments and differ only in what they
        read back afterwards -- the loss alone, or the loss and the image. Keeping two copies of
        the argument list meant either could drift out of step with the kernel signature.
        """
        offsets, pairs, _ = self.build_tiles(np.asarray([view_id], np.int32))
        self.loss.zero_()
        render_colours = self._render_colours(int(view_id))
        wp.launch(
            render_forward,
            dim=self.width * self.height,
            inputs=[self.means, self.log_scales, self.quaternions, self.opacity, render_colours,
                    self.cameras, self.targets, self.tiles.view_ids, offsets, pairs,
                    self.width, self.height, self.tiles_x, self.tiles_y, TILE, self.focal, 1,
                    self.background] + self.compact_box_args,
            outputs=[self.image, self.loss],
            device=self.device,
        )

    def evaluate(self, view_ids):
        """Evaluate the current training objective; image metrics still use plain MSE."""
        return self.evaluate_components(view_ids)[0]

    def evaluate_components(self, view_ids):
        """Return mean (objective, MSE, D-SSIM), preserving the first view for snapshots."""
        view_ids = np.asarray(view_ids, np.int32)
        if len(view_ids) < 1:
            raise ValueError("Evaluation requires at least one view.")

        # Render the first fixed camera last so self.image retains the snapshot view.
        ordered_view_ids = np.concatenate((view_ids[1:], view_ids[:1]))
        total_mse = 0.0
        total_dssim = 0.0
        for view_id in ordered_view_ids:
            self._render_view(view_id)
            total_mse += float(self.loss.numpy()[0])
            if self._dssim_active():
                total_dssim += self._dssim_computer().compute(
                    self.image, self.targets, self.tiles.view_ids, self.pixel_gradient,
                    self.loss_config["dssim_window"], 1, self.loss_config["dssim_weight"],
                    compute_gradient=False,
                )
        mse = total_mse / len(view_ids)
        dssim = total_dssim / len(view_ids)
        weight = self.loss_config["dssim_weight"] if self._dssim_active() else 0.0
        return (1.0 - weight) * mse + weight * dssim, mse, dssim

    def render_single_view_image(self, view_id):
        """Render one view and bring the pixels back, for a snapshot or an ADC error map."""
        self._render_view(view_id)
        return (
            self.image.numpy()[: self.width * self.height].reshape(self.height, self.width, 3),
            float(self.loss.numpy()[0]),
        )

    @property
    def compact_box_args(self):
        """The compact-support settings every render kernel needs, in launch order.

        `alpha_at_pixel` truncates each splat at its own compact-support radius, so these three
        scalars are appended to the inputs of every kernel that shades a pixel. Exposed here (and
        reached through `__getattr__` by the SH trainers) so the values come from one place.
        """
        compact = self.training_options["compact_box"]
        return [int(compact["enabled"]), float(compact["beta"]), float(compact["alpha_min"])]

    def multiview_adc_scores(self, candidate_indices, view_ids, render_view=None):
        """Reference multi-view ADC scorer for a bounded set of candidate splats.

        `render_view` supplies the renderer the error maps are built from. It is a parameter
        rather than `self.render_single_view_image` because the later versions wrap this trainer
        by composition instead of subclassing it: calling the method on `self` bound `self` to
        this object, so a view-dependent model was scored against a view-independent picture of
        itself, and every densify and prune decision it drove was made on the wrong image.
        """
        if render_view is None:
            render_view = self.render_single_view_image
        candidate_indices = np.asarray(candidate_indices, np.int32)
        if len(candidate_indices) == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)

        adc = self.training_options["multiview_adc"]
        means = self.means.numpy()[candidate_indices]
        scales = np.exp(self.log_scales.numpy()[candidate_indices])
        quaternions = self.quaternions.numpy()[candidate_indices]
        cameras = self.cameras.numpy()
        targets = self.targets_host
        compact = self.training_options["compact_box"]
        opacity = sigmoid(self.opacity.numpy()[candidate_indices])
        support = np.full(len(candidate_indices), 9.0, np.float32)
        if compact["enabled"]:
            support.fill(0.0)
            visible_opacity = opacity > compact["alpha_min"]
            support[visible_opacity] = (
                compact["beta"] * 2.0 * np.log(opacity[visible_opacity] / compact["alpha_min"])
            )

        densify_score = np.zeros(len(candidate_indices), np.float32)
        low_error_score = np.zeros(len(candidate_indices), np.float32)
        for view_id in view_ids:
            image, _ = render_view(int(view_id))
            error = np.mean(np.abs(image - targets[int(view_id)]), axis=2)
            error_min = float(error.min())
            error_range = float(error.max() - error_min)
            normalized_error = np.zeros_like(error) if error_range < 1.0e-8 else (error - error_min) / error_range
            high_error = normalized_error > adc["loss_threshold"]
            high_error_integral = high_error.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
            low_error_integral = (~high_error).astype(np.int32).cumsum(axis=0).cumsum(axis=1)

            camera = cameras[int(view_id)]
            homo = np.concatenate((means, np.ones((len(means), 1), np.float32)), axis=1)
            camera_points = homo @ camera.T
            visible = (camera_points[:, 2] > NEAR_PLANE) & (support > 0.0)
            if not np.any(visible):
                continue
            variance_x, variance_y = projected_covariance_diagonal(
                scales[visible], quaternions[visible], camera, camera_points[visible, :3],
                self.focal,
            )
            centres = np.empty((int(np.count_nonzero(visible)), 2), np.float32)
            centres[:, 0] = self.focal * camera_points[visible, 0] / camera_points[visible, 2] + 0.5 * self.width
            centres[:, 1] = self.focal * camera_points[visible, 1] / camera_points[visible, 2] + 0.5 * self.height
            radius_scale = np.sqrt(support[visible])
            min_x = np.floor(centres[:, 0] - radius_scale * np.sqrt(np.maximum(variance_x, 0.0))).astype(np.int32)
            max_x = np.floor(centres[:, 0] + radius_scale * np.sqrt(np.maximum(variance_x, 0.0))).astype(np.int32)
            min_y = np.floor(centres[:, 1] - radius_scale * np.sqrt(np.maximum(variance_y, 0.0))).astype(np.int32)
            max_y = np.floor(centres[:, 1] + radius_scale * np.sqrt(np.maximum(variance_y, 0.0))).astype(np.int32)
            in_image = (max_x >= 0) & (max_y >= 0) & (min_x < self.width) & (min_y < self.height)
            visible_indices = np.flatnonzero(visible)
            for local, x0, x1, y0, y1 in zip(
                visible_indices[in_image],
                np.clip(min_x[in_image], 0, self.width - 1),
                np.clip(max_x[in_image], 0, self.width - 1),
                np.clip(min_y[in_image], 0, self.height - 1),
                np.clip(max_y[in_image], 0, self.height - 1),
            ):
                total = high_error_integral[y1, x1]
                if x0 > 0:
                    total -= high_error_integral[y1, x0 - 1]
                if y0 > 0:
                    total -= high_error_integral[y0 - 1, x1]
                if x0 > 0 and y0 > 0:
                    total += high_error_integral[y0 - 1, x0 - 1]
                low_total = low_error_integral[y1, x1]
                if x0 > 0:
                    low_total -= low_error_integral[y1, x0 - 1]
                if y0 > 0:
                    low_total -= low_error_integral[y0 - 1, x1]
                if x0 > 0 and y0 > 0:
                    low_total += low_error_integral[y0 - 1, x0 - 1]
                densify_score[local] += total
                # A high residual identifies a place where more geometry may be needed, so it
                # must not increase the pruning score. The low-error score is only a simple
                # redundancy heuristic; opacity remains the primary pruning signal.
                low_error_score[local] += low_total

        densify_score /= max(1, len(view_ids))
        if low_error_score.max() > low_error_score.min():
            prune_score = (low_error_score - low_error_score.min()) / (
                low_error_score.max() - low_error_score.min()
            )
        else:
            prune_score = np.zeros_like(low_error_score)
        return densify_score, prune_score

    def position_learning_rate(self, iteration):
        rates = self.learning_rates
        return exponential_learning_rate(
            iteration,
            rates["position_initial"],
            rates["position_final"],
            rates["position_max_steps"],
            rates["position_delay_steps"],
            rates["position_delay_multiplier"],
        )

    def step(self, view_ids, download_gradients=False, download_loss=True, iteration=1):
        loss = self.loss_and_gradients(view_ids, download_loss)
        device_started = perf_counter()
        self.last_position_learning_rate = self.position_learning_rate(iteration)
        self.optimizers.apply(self.last_position_learning_rate, self.capacity)
        if self.sh is not None:
            self.sh.step(self.active_device)
        wp.launch(
            constrain_parameters,
            dim=self.capacity,
            inputs=[
                self.log_scales, self.color,
                np.log(self.scene_radius * SCALE_CLAMP_MIN_FRACTION),
                np.log(self.scene_radius * SCALE_CLAMP_MAX_FRACTION),
            ],
            device=self.device,
        )
        mean_gradients = self.optimizers.mean_grad.numpy() if download_gradients else None
        wp.synchronize_device(self.device)
        self.last_timings["device"] += perf_counter() - device_started
        return loss, mean_gradients

    def reset_opacity(self, ceiling=0.01):
        """Cap every active splat's opacity so faded splats must re-earn it or be pruned.

        Splats only ever lose opacity through the loss, so without this reset a low-opacity
        floater can sit forever just above the prune threshold. Adam's opacity state is reset
        because the parameter it was tracking has changed discontinuously.
        """
        opacity_logits = self.opacity.numpy()
        limit = float(np.log(ceiling / (1.0 - ceiling)))
        opacity_logits[self.active] = np.minimum(opacity_logits[self.active], limit)
        self.opacity.assign(
            wp.array(opacity_logits.astype(np.float32), dtype=wp.float32, device=self.device)
        )
        self.optimizers.reset_opacity()

    def densify_and_prune(self, gradients, fraction, view_ids=None, late_prune=False,
                          percent_dense=0.01, render_view=None):
        opacity = sigmoid(self.opacity.numpy())
        means, log_scales = self.means.numpy(), self.log_scales.numpy()
        quaternions, opacity_logits = self.quaternions.numpy(), self.opacity.numpy()
        color = self.color.numpy()
        adc = self.training_options["multiview_adc"]
        prune_mask = (opacity < PRUNE_OPACITY_FLOOR) & self.active
        scores = np.linalg.norm(gradients, axis=1)
        parents = [i for i in np.argsort(scores)[::-1] if self.active[i]]
        if adc["enabled"] and view_ids is not None:
            limit = min(len(parents), int(adc["candidate_limit"]))
            scored = np.asarray(parents[:limit], np.int32)
            densify_score, prune_score = self.multiview_adc_scores(
                scored, view_ids, render_view=render_view,
            )
            keep_for_densify = densify_score > adc["densify_score_threshold"]
            parents = scored[keep_for_densify].tolist()
            if late_prune:
                prune_mask[scored[prune_score > adc["prune_score_threshold"]]] = True
                prune_mask[(opacity < adc["opacity_cutoff"]) & self.active] = True
        elif late_prune:
            prune_mask[(opacity < adc["opacity_cutoff"]) & self.active] = True
        self.active[prune_mask] = False
        # A pruned splat must not stay a clone source: it now sits in `free`, so reusing it as
        # a parent would let a later clone read a slot already overwritten as a clone
        # destination in this same loop (means/etc. are mutated in place). Keep only still-active
        # parents so sources are always active and destinations always inactive -- no overlap.
        parents = [parent for parent in parents if self.active[parent]]
        free = np.flatnonzero(~self.active)
        children_to_add = 0 if fraction <= 0.0 else min(
            len(free), max(1, int(self.active.sum() * fraction))
        )
        # Split large splats, clone small ones -- the two distinct reference operations. A single
        # rule that always offsets by a fixed distance and always shrinks is wrong for both: the
        # offset stops being related to the splat's own extent, so each generation lands further
        # from the surface measured in its own radii, and the shrink makes that worse without
        # bound. Displacement here is sampled from the parent's own covariance instead.
        split_size = percent_dense * self.scene_radius
        # Vectorised over all children at once. The per-splat Python loop this replaces cost about
        # 56 microseconds per child -- dominated by numpy call overhead on single splats (one
        # `np.exp`, one single-quaternion rotation build, one `rng.normal`), not by real work --
        # which at tens of thousands of children per event was seconds per densify tick and a
        # double-digit percentage of total training time. Parents are distinct (they come from
        # `argsort`) and destinations are drawn from `free`, so parent and child slots never
        # overlap and these whole-array writes cannot alias.
        # `parents` is ADC-filtered and can be shorter than `children_to_add`; the `zip` this
        # replaced truncated to the shorter side, so pair explicitly rather than relying on two
        # slices happening to match.
        pair_count = min(children_to_add, len(parents))
        chosen_parents = np.asarray(parents[:pair_count], dtype=np.intp)
        chosen_children = np.asarray(free[:pair_count], dtype=np.intp)
        if len(chosen_parents):
            if self.sh is not None:
                self.sh.copy_children(chosen_parents, chosen_children, self.optimizers.child_state_policy)
            self.active[chosen_children] = True
            parent_scales = np.exp(log_scales[chosen_parents])
            is_split = parent_scales.max(axis=1) > split_size
            # Every child inherits these regardless of branch.
            quaternions[chosen_children] = quaternions[chosen_parents]
            opacity_logits[chosen_children] = opacity_logits[chosen_parents]
            color[chosen_children] = color[chosen_parents]
            # Clone is the default: a small splat is copied exactly. The gradient separates the
            # pair, so no displacement is invented and no opacity is thrown away.
            means[chosen_children] = means[chosen_parents]
            log_scales[chosen_children] = log_scales[chosen_parents]
            # Split overrides it for large splats: two smaller children replace one large parent,
            # the parent's slot becoming the first child, so the population still grows by exactly
            # one splat. Displacement is sampled from the parent's own covariance.
            split_parents = chosen_parents[is_split]
            if len(split_parents):
                split_children = chosen_children[is_split]
                split_scales = parent_scales[is_split]
                unit = quaternions[split_parents] / np.maximum(
                    np.linalg.norm(quaternions[split_parents], axis=1, keepdims=True), 1.0e-8,
                )
                rotation = quaternion_to_matrix(unit)
                # Fancy indexing copies, so this keeps the pre-split origin even though the
                # parent's own mean is overwritten below.
                origin = means[split_parents]
                offsets = self.rng.normal(size=(2,) + split_scales.shape) * split_scales
                means[split_parents] = origin + np.einsum("nij,nj->ni", rotation, offsets[0])
                means[split_children] = origin + np.einsum("nij,nj->ni", rotation, offsets[1])
                shrunk = log_scales[split_parents] - np.log(1.6)
                log_scales[split_parents] = shrunk
                log_scales[split_children] = shrunk
        self.active_device.assign(self.active.astype(np.int32))
        for array, values, dtype in (
            (self.means, means, wp.vec3),
            (self.log_scales, log_scales, wp.vec3),
            (self.quaternions, quaternions, wp.vec4),
            (self.opacity, opacity_logits, wp.float32),
            (self.color, color, wp.vec3),
        ):
            array.assign(wp.array(values.astype(np.float32), dtype=dtype, device=self.device))
        if pair_count:
            self.optimizers.inherit_children(chosen_parents, chosen_children)
        # Report the slots this call filled. Nothing here needs it -- every parameter a version-1
        # splat owns is overwritten above -- but a subclass carrying extra per-splat state has to
        # reset that state on exactly these slots, and it cannot recover the list afterwards: a
        # slot pruned and refilled within this same call is active before and after, so
        # differencing the active mask misses it and the child silently inherits the dead splat's
        # state. See Unit 9, Stage 7, and `SHTrainer.densify_and_prune` in version 2.
        return chosen_children

    def save_sidecars(self, path):
        """Save SH in the exact active-splat order exported by gaussian_set()."""
        return [self.sh.save(path, self.active)] if self.sh is not None else []

    def gaussian_set(self):
        active = np.flatnonzero(self.active)
        return GaussianSet(
            self.means.numpy()[active],
            np.exp(self.log_scales.numpy()[active]),
            self.quaternions.numpy()[active],
            sigmoid(self.opacity.numpy()[active]),
            np.clip(self.color.numpy()[active], 0, 1),
        )


class ConvergenceTracker:
    """Tracks whether the fixed-evaluation loss has plateaued, for early stopping.

    Fed the *smoothed* fixed_eval series, not the raw one. A single fixed evaluation moves with
    every densification and prune event, so a raw plateau test spends its `min_delta` budget on
    that jitter; smoothing first means the threshold measures trend rather than noise. The cost is
    lag -- the average keeps improving for a while after the raw series flattens -- so a run stops
    somewhat later than a raw-fed detector would.

    Only a true `eval_every` tick advances this state -- a snapshot-only iteration recomputes
    `fixed_eval` for logging and rendering, but must not affect when training stops, or
    `snapshot_every` would silently reshape the stopping schedule. See Unit 9, Stage 9.
    """

    def __init__(self, config, initial_loss, enabled=None):
        # `enabled` overrides the config, for a detector that is always on: a curriculum stage
        # boundary is detected, never scheduled, so the section driving it carries no such flag.
        # Defaulted rather than indexed for that reason -- reading `config["enabled"]` here made
        # an unrelated section without the key a KeyError waiting on the next caller.
        self.enabled = enabled if enabled is not None else config.get("enabled", True)
        self.min_iterations = config["min_iterations"]
        self.patience = config["patience"]
        self.min_delta = config["min_delta"]
        self.best_loss = initial_loss
        self.previous_loss = initial_loss
        self.stale_count = 0

    def update(self, fixed_eval_loss, iteration):
        """Record one eval tick and count only a plateau toward early stopping.

        Any worsening resets patience so a temporary regression cannot be mistaken for convergence.
        """
        worsening = fixed_eval_loss > self.previous_loss
        if fixed_eval_loss < self.best_loss - self.min_delta:
            self.best_loss = fixed_eval_loss
            self.stale_count = 0
        elif worsening:
            self.stale_count = 0
        elif self.enabled and iteration >= self.min_iterations:
            self.stale_count += 1
        else:
            self.stale_count = 0
        self.previous_loss = fixed_eval_loss

    def rebaseline(self, fixed_eval_loss):
        """Restart from this reading, crediting nothing to the stale count.

        For the first tick after a deliberate disturbance the model has recovered to. Calling
        `update` there instead is a bug: recovery is defined as returning to the pre-disturbance
        value, and `best_loss` is at most that value, so the reading can never clear
        `best_loss - min_delta` and always counts stale. Every reset then donates one stale tick,
        and enough resets declare a plateau on their own -- measured on the cluster, two v3 runs
        with identical settings advanced stage at 6800 and 13200, each on a reset-recovery tick.
        """
        self.best_loss = fixed_eval_loss
        self.previous_loss = fixed_eval_loss
        self.stale_count = 0

    def should_stop(self, iteration):
        return (
            self.enabled
            and iteration >= self.min_iterations
            and self.stale_count >= self.patience
        )


class OpacityResetWindow:
    """Hides an opacity reset's transient from the convergence detectors.

    Reads the same smoothed fixed_eval series the convergence tracker does. `open` and `accepts`
    must be given the same series as each other: `accepts` decides recovery by comparing against
    the level `open` recorded, and mixing a raw reading with a smoothed one would compare two
    different quantities.

    An opacity reset deliberately caps every splat's opacity, so the next `fixed_eval` measures a
    model that was just knocked down. The detectors compare against the best value ever seen, so
    those readings can never win, and a reset dropped into a run reads as a plateau: measured on
    the cluster, a reset every 200 iterations stopped v1 and v2 at iteration 2400 with the loss
    still falling steeply, and an aggressive prune did the same at 6200. Training was progressing
    in both cases; only the yardstick was broken.

    So convergence bookkeeping is suspended from a reset until the loss has climbed back to where
    it stood just before it. Recovery is measured, not configured -- the interval that is healthy
    depends on the scene and on how far the reset set opacity back.
    """

    def __init__(self):
        self.target = None
        self.since = None
        self.resumed_this_tick = False

    def open(self, fixed_eval_loss, iteration):
        """Called when a reset fires: remember the quality it is about to disturb."""
        self.target = fixed_eval_loss
        self.since = iteration

    def accepts(self, fixed_eval_loss, iteration):
        """Whether this eval tick may advance convergence state.

        Sets `resumed_this_tick` when it is the tick that ended a suspension, so the caller
        rebaselines rather than updates. The model improved throughout the suspension and none of
        that progress reached the tracker; charging the resume tick as stale on top of that is
        what let the reset cadence drive the stage transition.
        """
        self.resumed_this_tick = False
        if self.target is None:
            return True
        if fixed_eval_loss <= self.target:
            self.resumed_this_tick = True
            print(
                f"{iteration:6d}: recovered from the opacity reset at {self.since} "
                f"({fixed_eval_loss:.6f} <= {self.target:.6f}); convergence checks resume.",
                flush=True,
            )
            self.target = None
            return True
        return False

    def warn_if_unrecovered(self, iteration):
        """Called when the next reset fires while the previous one is still being absorbed."""
        if self.target is not None:
            print(
                f"{iteration:6d}: opacity reset fired before the one at {self.since} had "
                "recovered -- densification.opacity_reset_interval is too short for this scene, "
                "and convergence checks have been suspended throughout.",
                flush=True,
            )


class ExponentialMovingAverage:
    """Decaying average of a scalar series, seeded by its first sample.

    Both reported series -- the training loss and the fixed-evaluation loss -- are noisy enough
    per sample that a single reading says little. The training loss is measured on one randomly
    chosen view, and the fixed evaluation, though deterministic, moves with every densification
    and prune event. Reporting and stage decisions therefore read the smoothed series; seeding
    from the first sample rather than from zero avoids the warm-up bias that would otherwise make
    early values meaningless.
    """

    def __init__(self, decay: float):
        self.decay = decay
        self.value = None

    def update(self, sample: float) -> float:
        self.value = sample if self.value is None else (
            self.decay * self.value + (1.0 - self.decay) * sample
        )
        return self.value


class GrowthPlateauTracker:
    """Latch when active-splat growth has become small over recent densification ticks.

    A plateau means growth has slowed, not that the population is collapsing. A run that starts
    from a large random cloud prunes most of it away in the first few hundred iterations, and that
    collapse reads as a large *negative* mean growth -- far below any sensible threshold. Testing
    only `mean < threshold` therefore latched on the startup transient at the first opportunity,
    before the pool had begun growing at all, which is the opposite of the intended signal.
    Requiring the mean to be non-negative as well distinguishes "growth has slowed" from
    "growth has not started".
    """

    def __init__(self, window: int, threshold_percent: float):
        self.window = window
        self.threshold_percent = threshold_percent
        self.previous_active = None
        self.recent_growth = []
        self.triggered = False
        self.mean_growth_at_trigger = None

    def update(self, active_count: int) -> bool:
        if self.triggered:
            return False
        if self.previous_active is not None and self.previous_active > 0:
            self.recent_growth.append(
                100.0 * (active_count - self.previous_active) / self.previous_active
            )
            if len(self.recent_growth) > self.window:
                self.recent_growth.pop(0)
            if len(self.recent_growth) == self.window:
                mean_growth = sum(self.recent_growth) / self.window
                if 0.0 <= mean_growth < self.threshold_percent:
                    self.triggered = True
                    self.mean_growth_at_trigger = mean_growth
        self.previous_active = active_count
        return self.triggered


def png_output_path(path: Path) -> Path:
    """Return the path used for rendered output; rendered images are always PNG."""
    return path if path.suffix.lower() == ".png" else path.with_suffix(".png")


def save_rendered_image(trainer: WarpImageTrainer, path: Path) -> Path:
    """Save the trainer's most recently rendered image (its first buffered view) as a PNG."""
    path = png_output_path(path)
    pixels = trainer.image.numpy()[: trainer.width * trainer.height].reshape(
        trainer.height, trainer.width, 3
    )
    Image.fromarray(np.uint8(np.clip(pixels, 0.0, 1.0) * 255.0)).save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian splats from one YAML configuration file."
    )
    parser.add_argument("config", type=Path, help="Path to the training YAML file.")
    args = parser.parse_args()
    try:
        config = TrainingConfig.load(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Resolved training configuration ({args.config.expanduser().resolve()}):")
    print(yaml.safe_dump(config.printable(), sort_keys=False).rstrip(), flush=True)

    paths = config["paths"]
    runtime = config["runtime"]
    model = config["model"]
    training = config["training"]
    rates = config["learning_rates"]
    compact = config["compact_box"]
    adc = config["multiview_adc"]
    sparse = config["sparse"]
    reporting = config["reporting"]
    wp.init()
    device = "cpu" if runtime["arch"] == "cpu" else "cuda:0"
    if runtime["arch"] == "gpu" and not wp.is_cuda_available():
        raise RuntimeError("runtime.arch=gpu requires a CUDA-enabled Warp installation.")
    background = (
        np.ones(3, np.float32) if model["background"] == "white"
        else np.zeros(3, np.float32)
    )
    targets, cameras, focal, distance, _ = load_views(
        paths["data"], model["width"], model["height"], background
    )
    scene_radius = model["scene_radius"] or distance * 0.35
    initial = GaussianSet.from_ply(paths["init_ply"]) if paths["init_ply"] else None
    if initial is not None and len(initial.means) > model["capacity"]:
        parser.error(
            f"Checkpoint has {len(initial.means):,} splats, exceeding "
            f"model.capacity={model['capacity']:,}."
        )
    # `targets`/`cameras` stay the training views; `all_targets`/`all_cameras` are those with the
    # held-out evaluation views appended. The trainer indexes one view-id space, so it gets the
    # combined arrays; `run_training` gets the training-only ones and can therefore never draw a
    # held-out camera to optimise against.
    all_targets, all_cameras, eval_view_ids, held_out_frame_ids = resolve_evaluation_views(
        targets, cameras, paths["data"], background, reporting,
    )
    trainer = WarpImageTrainer(
        all_targets, all_cameras, focal, scene_radius, model["capacity"],
        model["initial_splats"], device, runtime["seed"],
        init_scene=initial,
        background=background,
        learning_rates=rates,
        sh_degree=model["sh_degree"],
        child_optimizer_state=config["densification"]["child_optimizer_state"],
        tile_pair_capacity=model["tile_pair_capacity"],
        training_options={
            "compact_box": compact,
            "multiview_adc": adc,
            "sparse": sparse,
            "loss": config["loss"],
        },
    )
    output = png_output_path(paths["output"])
    snapshots = Path(f"{output.with_suffix('')}_snapshots")
    snapshots.mkdir(parents=True, exist_ok=True)
    run_training(
        trainer, config, targets, eval_view_ids, snapshots, output,
        held_out_frame_ids=held_out_frame_ids,
    )


def run_training(
    trainer, config, targets, eval_view_ids, snapshots, output, description=None,
    held_out_frame_ids=None,
) -> None:
    """Run the training loop: logging, snapshots, densification, and the final save.

    The optional loss curriculum latches on held-out MSE PSNR. After the switch, convergence
    reads the blended objective with a fresh baseline and counts iterations within that stage.
    """
    runtime = config["runtime"]
    training = config["training"]
    densification = config["densification"]
    adc = config["multiview_adc"]
    adaptive = config["adaptive"]
    convergence = config["convergence"]
    reporting = config["reporting"]
    model = config["model"]
    curriculum = config["loss"]["mode"] == "mse_dssim"
    target_sh_degree = model["sh_degree"]
    dssim_threshold = 10.0 ** (-adaptive["dssim_psnr_threshold"] / 10.0)
    device = trainer.device
    print(f"Density child optimizer state: {trainer.optimizers.child_state_policy}", flush=True)

    # Densification runs for the whole run. The active-splat growth plateau arms late pruning,
    # but does not stop densification: the two are separate decisions, and stopping on that
    # signal cut the final splat count to roughly a third with a measurable quality cost.
    print(
        f"Training {trainer.active.sum():,} initial splats (capacity {model['capacity']:,}) at "
        f"{model['width']}x{model['height']} on Warp {device}; "
        f"{description or 'one random training view per iteration'}, "
        f"tile_pair_capacity={trainer.tile_pair_capacity:,}, "
        f"densify_every={densification['interval']}, no densification cutoff. "
        "Building tiles for the first step...",
        flush=True,
    )
    if trainer.sparse_enabled:
        print(
            f"Sparse MSE training is enabled with {trainer.samples_per_tile} sampled pixels per "
            f"{TILE}x{TILE} tile. Fixed evaluation and snapshots remain dense.",
            flush=True,
        )
    if curriculum:
        print(
            f"Loss curriculum: MSE -> dense MSE + D-SSIM at held-out MSE EMA PSNR >= "
            f"{adaptive['dssim_psnr_threshold']:.2f} dB; "
            f"weight={config['loss']['dssim_weight']}, window={config['loss']['dssim_window']}. "
            "Early stopping is armed only after the transition.",
            flush=True,
        )
    if target_sh_degree:
        gates = ", ".join(f"sh{band}: {adaptive[f'sh{band}_psnr_threshold']:.2f} dB"
                          for band in range(1, target_sh_degree + 1))
        print(
            f"SH curriculum: {gates}; gates read held-out MSE EMA after the loss warm-up. "
            "Final convergence is armed only after the configured maximum SH degree unlocks.",
            flush=True,
        )
    if adc["enabled"]:
        print(
            f"multiview_adc late-prune fires every {adc['late_prune_interval']} iterations after "
            f"active-splat growth over {adaptive['late_prune_growth_window']} densify ticks falls "
            f"below {adaptive['late_prune_growth_percent']:.2f}%; each event removes every "
            f"active splat with opacity below opacity_cutoff={adc['opacity_cutoff']} across the "
            "whole population, not only multi-view-scored candidates.",
            flush=True,
        )
    print(
        "Reported loss and fixed_eval are exponential moving averages over the raw series "
        f"(decay {reporting['loss_ema_decay']}). Convergence checks and the opacity-reset "
        "recovery test read the smoothed fixed_eval, so what is printed is what they decide on. "
        + (
            "fixed_eval is measured on held-out views the optimiser never sees, so it is a "
            "generalisation measurement and not a second reading of the training loss."
            if held_out_frame_ids is not None
            else "fixed_eval is measured on training views (reporting.eval_manifest is null)."
        ),
        flush=True,
    )

    # Every trainer this loop drives reports which loss term is active from construction onward:
    # a fixed one here, a changing one in the versions that blend terms.
    fixed_eval_loss, fixed_eval_mse, _ = trainer.evaluate_components(eval_view_ids)
    fixed_eval_iteration = 0
    loss_ema = ExponentialMovingAverage(reporting["loss_ema_decay"])
    fixed_eval_ema = ExponentialMovingAverage(reporting["loss_ema_decay"])
    fixed_eval_mse_ema = ExponentialMovingAverage(reporting["loss_ema_decay"])
    fixed_eval_ema.update(fixed_eval_loss)
    fixed_eval_mse_ema.update(fixed_eval_mse)
    print(
        f"{0:6d}: fixed_eval_ema={fixed_eval_ema.value:.6f}, "
        f"loss_phase={trainer.last_loss_phase}, "
        f"stage={trainer.loss_stage}, fixed_eval_mse_ema={fixed_eval_mse_ema.value:.6f}, "
        f"sh_stage=sh{trainer.active_sh_degree}, "
        + describe_evaluation_views(
            eval_view_ids, held_out_frame_ids, reporting["eval_manifest"],
        ),
        flush=True,
    )
    rng = np.random.default_rng(runtime["seed"])
    convergence_tracker = None if curriculum or target_sh_degree else ConvergenceTracker(convergence, fixed_eval_ema.value)
    stage_start_iteration = 0
    reset_loss_ema = False
    growth_tracker = GrowthPlateauTracker(
        adaptive["late_prune_growth_window"], adaptive["late_prune_growth_percent"],
    )
    late_prune_armed = False
    reset_window = OpacityResetWindow()
    completed_iteration = 0
    # Snapshot 0: the initial (pre-optimisation) model, so an animation can open on the
    # random/checkpoint starting point before any gradient step has run.
    save_rendered_image(trainer, snapshots / "out_000000.png")
    if reporting["save_ply"]:
        initial_ply = snapshots / "out_000000.ply"
        trainer.gaussian_set().to_ply(initial_ply)
        trainer.save_sidecars(initial_ply)
    training_started = perf_counter()
    for iteration in range(1, training["iterations"] + 1):
        completed_iteration = iteration
        view_ids = np.asarray([rng.integers(len(targets))], dtype=np.int32)
        should_densify = bool(
            iteration < training["iterations"]
            and densification["interval"] and iteration % densification["interval"] == 0
        )
        should_late_prune = bool(
            iteration < training["iterations"] and adc["enabled"]
            and late_prune_armed
            and iteration % adc["late_prune_interval"] == 0
        )
        should_structure_update = should_densify or should_late_prune
        should_snapshot = iteration % reporting["snapshot_every"] == 0
        is_eval_tick = iteration % reporting["eval_every"] == 0
        should_log = iteration % reporting["log_every"] == 0
        loss, gradients = trainer.step(
            view_ids,
            download_gradients=should_structure_update,
            iteration=iteration,
            download_loss=should_log or is_eval_tick or should_snapshot or reset_loss_ema,
        )
        if should_structure_update:
            adc_views = None
            if adc["enabled"]:
                sample_count = min(adc["views"], len(targets))
                adc_views = rng.choice(len(targets), size=sample_count, replace=False)
            trainer.densify_and_prune(
                gradients,
                densification["fraction"] if should_densify else 0.0,
                view_ids=adc_views,
                late_prune=should_late_prune,
                percent_dense=densification["percent_dense"],
            )
            if should_densify and not late_prune_armed:
                if growth_tracker.update(int(trainer.active.sum())):
                    late_prune_armed = True
                    print(
                        f"{iteration:6d}: active-splat growth plateaued "
                        f"({growth_tracker.mean_growth_at_trigger:.2f}% mean over last "
                        f"{growth_tracker.window} densify ticks) -> arming late-prune "
                        "(densification continues).",
                        flush=True,
                    )
        if (
            iteration < training["iterations"] and densification["opacity_reset_interval"]
            and iteration % densification["opacity_reset_interval"] == 0
        ):
            reset_window.warn_if_unrecovered(iteration)
            reset_window.open(fixed_eval_ema.value, iteration)
            trainer.reset_opacity()
        # A version that skips the loss download on quiet iterations reports None for them.
        if loss is not None:
            if reset_loss_ema:
                loss_ema = ExponentialMovingAverage(reporting["loss_ema_decay"])
                reset_loss_ema = False
            loss_ema.update(loss)
        if is_eval_tick or should_snapshot:
            fixed_eval_loss, fixed_eval_mse, _ = trainer.evaluate_components(eval_view_ids)
            fixed_eval_iteration = iteration
            fixed_eval_ema.update(fixed_eval_loss)
            fixed_eval_mse_ema.update(fixed_eval_mse)
            # Convergence bookkeeping only advances on a true eval_every tick. A snapshot alone
            # still needs a fresh fixed_eval to render and log, but it must not change when
            # training stops -- otherwise snapshot_every silently reshapes the stopping schedule.
            # An opacity reset's transient is not evidence about convergence; see
            # OpacityResetWindow.
            if is_eval_tick and reset_window.accepts(fixed_eval_ema.value, iteration):
                if curriculum and trainer.dssim_stage == 0:
                    if fixed_eval_mse_ema.value <= dssim_threshold:
                        psnr = -10.0 * np.log10(max(fixed_eval_mse_ema.value, 1.0e-12))
                        print(
                            f"{iteration:6d}: fixed_eval MSE EMA reached {fixed_eval_mse_ema.value:.6f} "
                            f"({psnr:.2f} dB >= {adaptive['dssim_psnr_threshold']:.2f} dB) "
                            "-> advancing loss stage 'mse' -> 'dssim'.",
                            flush=True,
                        )
                        trainer.advance_dssim_stage()
                        stage_start_iteration = iteration
                        # A blended objective cannot be compared with the old MSE baseline.
                        fixed_eval_loss, _, _ = trainer.evaluate_components(eval_view_ids)
                        fixed_eval_ema = ExponentialMovingAverage(reporting["loss_ema_decay"])
                        fixed_eval_ema.update(fixed_eval_loss)
                        convergence_tracker = (None if trainer.active_sh_degree < target_sh_degree
                                               else ConvergenceTracker(convergence, float("inf")))
                        reset_window = OpacityResetWindow()
                        reset_loss_ema = True
                else:
                    sh_advanced = False
                    if trainer.active_sh_degree < target_sh_degree:
                        next_degree = trainer.active_sh_degree + 1
                        threshold = adaptive[f"sh{next_degree}_psnr_threshold"]
                        if fixed_eval_mse_ema.value <= 10.0 ** (-threshold / 10.0):
                            trainer.advance_sh_stage()
                            stage_start_iteration = iteration
                            sh_advanced = True
                            print(
                                f"{iteration:6d}: held-out MSE EMA "
                                f"{-10.0 * np.log10(max(fixed_eval_mse_ema.value, 1.0e-12)):.2f} dB "
                                f">= {threshold:.2f} dB -> advancing SH stage to sh{next_degree}.",
                                flush=True,
                            )
                            if next_degree == target_sh_degree:
                                convergence_tracker = ConvergenceTracker(convergence, float("inf"))
                    if not sh_advanced and convergence_tracker is not None:
                        if reset_window.resumed_this_tick:
                            convergence_tracker.rebaseline(fixed_eval_ema.value)
                        else:
                            convergence_tracker.update(fixed_eval_ema.value, iteration - stage_start_iteration)
        if should_log:
            print(
                f"{iteration:6d}: ema={loss_ema.value:.6f}, "
                f"loss_phase={trainer.last_loss_phase}, "
                f"stage={trainer.loss_stage}, "
                f"sh_stage=sh{trainer.active_sh_degree}, "
                f"fixed_eval_ema={fixed_eval_ema.value:.6f}@{fixed_eval_iteration}, "
                f"fixed_eval_mse_ema={fixed_eval_mse_ema.value:.6f}, "
                f"active={trainer.active.sum()}, "
                f"pairs={trainer.last_pair_count}, "
                f"lr_xyz={trainer.last_position_learning_rate:.8f}, "
                f"tiles={trainer.last_timings['tile']:.2f}s, device={trainer.last_timings['device']:.2f}s",
                flush=True,
            )
        if should_snapshot:
            save_rendered_image(trainer, snapshots / f"out_{iteration:06d}.png")
            if reporting["save_ply"]:
                snapshot_ply = snapshots / f"out_{iteration:06d}.ply"
                trainer.gaussian_set().to_ply(snapshot_ply)
                trainer.save_sidecars(snapshot_ply)
        if convergence_tracker is not None and convergence_tracker.should_stop(iteration - stage_start_iteration):
            print(
                f"Stopping at iteration {iteration}: the fixed_eval EMA has not improved by "
                f"at least {convergence['min_delta']:.3g} for {convergence_tracker.stale_count} "
                "evaluation checks.",
                flush=True,
            )
            break
    print(f"Final loss stage: {trainer.loss_stage}", flush=True)
    print(f"Final SH degree: {trainer.active_sh_degree} (configured maximum: {target_sh_degree})", flush=True)
    if trainer.active_sh_degree < target_sh_degree:
        print("SH gates were not all reached; saved checkpoints render at the reported active degree.", flush=True)
    if curriculum and trainer.dssim_stage == 0:
        print(
            f"D-SSIM was not activated: held-out MSE EMA PSNR did not reach "
            f"{adaptive['dssim_psnr_threshold']:.2f} dB. This run trained with MSE only.",
            flush=True,
        )
    trainer.evaluate(eval_view_ids)
    output = save_rendered_image(trainer, output)
    if reporting["save_ply"]:
        final_ply = output.with_suffix(".ply")
        trainer.gaussian_set().to_ply(final_ply)
        print(f"Saved final trained splats to {final_ply}", flush=True)
    for sidecar in trainer.save_sidecars(output):
        print(f"Saved {sidecar}", flush=True)
    elapsed = perf_counter() - training_started
    print(
        f"Training finished after {completed_iteration:,} iterations in "
        f"{elapsed:.2f}s ({elapsed / max(1, completed_iteration):.3f}s/iter).",
        flush=True,
    )


if __name__ == "__main__":
    main()
