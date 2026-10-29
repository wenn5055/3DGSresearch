"""Flattened covariance surfels, with optional deferred environment reflection.

The baseline trainer remains unchanged. This subclass replaces projection/rasterization,
reuses its optimizers and D-SSIM objective, and carries project-specific material state.
"""

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import warp as wp
from warp.optim import Adam

from gaussian_set import GaussianSet
from splat_math import quaternion_to_matrix

from .config import SurfelTrainingConfig
from .geometry import SurfelTileWorkspace, base
from . import kernels as k


class TrainingOnlyInitialization:
    def _init_splats(self, capacity, initial_splats, init_scene, targets):
        count = len(targets) if self.training_view_count is None else self.training_view_count
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= len(targets):
            raise ValueError('Invalid training_view_count for initialization.')
        if count == 0 and init_scene is None:
            raise ValueError('Random initialization requires training images.')
        self.initial_training_colour = (targets[:count].mean(axis=(0, 1, 2)) if count
                                        else np.full(3, 0.5, np.float32))
        super()._init_splats(capacity, initial_splats, init_scene, targets[:count])


class GaussianRGBControl(TrainingOnlyInitialization, base.WarpImageTrainer):
    """Original 3D Gaussian renderer/optimizer with training-only colour statistics."""
    def __init__(self, *args, training_view_count=None, surfel_options=None, **kwargs):
        self.training_view_count = training_view_count
        self.surfel_options = surfel_options or {}
        self.specular_active = False
        super().__init__(*args, **kwargs)

    @property
    def surfel_stage(self):
        return 'gaussian_rgb'


class SurfelTrainer(TrainingOnlyInitialization, base.WarpImageTrainer):
    def __init__(self, *args, surfel_options=None, training_view_count=None, **kwargs):
        options = deepcopy(SurfelTrainingConfig.DEFAULTS["surfel"])
        if surfel_options:
            options.update(surfel_options)
        self.surfel_options = options
        self.training_view_count = training_view_count
        training = deepcopy(kwargs.pop("training_options", {}))
        training.setdefault("sparse", {})["enabled"] = False
        super().__init__(*args, training_options=training, **kwargs)
        if self.sh is not None:
            raise ValueError("The surfel passes require sh_degree=0.")
        self.tiles = SurfelTileWorkspace(self.capacity, 1, self.tiles_x, self.tiles_y, self.tile_pair_capacity, self.device)
        pixels = self.width * self.height
        self.attributes = wp.empty(self.capacity, dtype=k.GBuffer, device=self.device)
        self.buffers = wp.zeros(pixels, dtype=k.GBuffer, device=self.device, requires_grad=True)
        self.image = wp.zeros(pixels, dtype=wp.vec3, device=self.device, requires_grad=True)
        self.normal_gradient = wp.zeros(self.capacity * 3, dtype=wp.float32, device=self.device)
        reflection = options["initial_reflection"]
        logits = np.full(self.capacity, np.log(reflection / (1.0 - reflection)), np.float32)
        self.reflection_logits = wp.array(logits, dtype=wp.float32, device=self.device)
        self.reflection_gradient = wp.zeros(self.capacity, dtype=wp.float32, device=self.device)
        self.reflection_optimizer = Adam([self.reflection_logits], lr=options["reflection_lr"])
        self.env_width = options["environment_width"] if options["mode"] == "deferred" else 1
        self.env_height = options["environment_height"] if options["mode"] == "deferred" else 1
        colour = np.clip(self.initial_training_colour, 0.001, 0.999)
        environment = np.broadcast_to(np.log(colour / (1.0 - colour)), (self.env_width * self.env_height, 3)).copy()
        self.environment = wp.array(environment.astype(np.float32), dtype=wp.vec3, device=self.device, requires_grad=True)
        self.environment_optimizer = Adam([self.environment], lr=options["environment_lr"])
        self.specular_active = options["mode"] == "deferred" and options["specular_start_iteration"] == 0
        self._shader_tape = None

    @property
    def surfel_stage(self):
        return "deferred" if self.specular_active else "rgb"

    def activate_specular(self):
        if self.surfel_options["mode"] != "deferred":
            raise ValueError("Deferred shading requires surfel.mode: deferred.")
        self.specular_active = True

    def _forward(self, view, record=False):
        if self._shader_tape is not None:
            self._shader_tape.zero()
        started = perf_counter()
        offsets, pairs, count = self.build_tiles(np.asarray([view], np.int32))
        self.last_pair_count = count
        tile_seconds = perf_counter() - started
        device_started = perf_counter()
        wp.launch(k.build_attributes, dim=self.capacity,
                  inputs=[self.means, self.quaternions, self.color, self.reflection_logits, self.cameras, view],
                  outputs=[self.attributes], device=self.device)
        wp.launch(k.raster_forward, dim=self.width * self.height,
                  inputs=[self.means, self.log_scales, self.quaternions, self.opacity, self.attributes,
                          self.cameras, view, offsets, pairs, self.width, self.height, self.tiles_x,
                          self.focal] + self.compact_box_args, outputs=[self.buffers], device=self.device)
        inputs = [self.buffers, self.environment, self.env_width, self.env_height, self.cameras, view,
                  self.width, self.height, self.focal, int(self.specular_active), self.background]
        if record:
            self._shader_tape = wp.Tape()
            with self._shader_tape:
                wp.launch(k.deferred_shader, dim=self.width * self.height, inputs=inputs, outputs=[self.image], device=self.device)
        else:
            wp.launch(k.deferred_shader, dim=self.width * self.height, inputs=inputs, outputs=[self.image], device=self.device)
        self.loss.zero_()
        wp.launch(k.mse_loss, dim=self.width * self.height,
                  inputs=[self.image, self.targets, view, self.width * self.height],
                  outputs=[self.loss, self.pixel_gradient], device=self.device)
        self.last_timings = {"tile": tile_seconds, "device": perf_counter() - device_started}
        return offsets, pairs

    def _render_view(self, view_id):
        self._forward(int(view_id), record=False)

    def dense_loss_and_gradients(self, view_ids, download_loss=True):
        if len(view_ids) != 1:
            raise ValueError("Surfel training expects one view per step.")
        view = int(view_ids[0])
        self.optimizers.zero_gradients()
        self.normal_gradient.zero_()
        self.reflection_gradient.zero_()
        offsets, pairs = self._forward(view, record=True)
        started = perf_counter()
        mse = float(self.loss.numpy()[0]) if download_loss else None
        objective = self._auxiliary_loss(mse, view_ids, download_loss)
        self._shader_tape.backward(grads={self.image: self.pixel_gradient})
        wp.launch(k.raster_backward, dim=self.width * self.height,
                  inputs=[self.means, self.log_scales, self.quaternions, self.opacity, self.color,
                          self.attributes, self.cameras, view, offsets, pairs, self.width, self.height,
                          self.tiles_x, self.focal] + self.compact_box_args + [self.buffers, self.buffers.grad],
                  outputs=[self.optimizers.mean_grad_flat, self.optimizers.scale_grad_flat,
                           self.optimizers.quaternion_grad_flat, self.optimizers.opacity_grad,
                           self.optimizers.color_grad_flat, self.normal_gradient, self.reflection_gradient], device=self.device)
        wp.launch(k.normal_backward, dim=self.capacity,
                  inputs=[self.means, self.quaternions, self.cameras, view, self.normal_gradient],
                  outputs=[self.optimizers.mean_grad_flat, self.optimizers.quaternion_grad_flat], device=self.device)
        self.optimizers.pack_gradients(self.capacity)
        self.last_timings["device"] += perf_counter() - started
        return objective

    def step(self, view_ids, download_gradients=False, download_loss=True, iteration=1):
        loss, gradient = super().step(view_ids, download_gradients, download_loss, iteration)
        if self.specular_active:
            started = perf_counter()
            self.reflection_optimizer.step([self.reflection_gradient])
            self.environment_optimizer.step([self.environment.grad])
            wp.launch(k.clamp_logits, dim=self.capacity, inputs=[self.reflection_logits], device=self.device)
            wp.launch(k.clamp_environment, dim=self.env_width * self.env_height, inputs=[self.environment], device=self.device)
            wp.synchronize_device(self.device)
            self.last_timings["device"] += perf_counter() - started
        return loss, gradient

    def multiview_adc_scores(self, candidate_indices, view_ids, render_view=None):
        """Same error-footprint criterion as the baseline, using zero normal-axis variance."""
        indices = np.asarray(candidate_indices, np.int32)
        if not len(indices):
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        means = self.means.numpy()[indices]
        scales = np.exp(self.log_scales.numpy()[indices])
        scales[:, 2] = 0.0
        rotations = self.quaternions.numpy()[indices]
        cameras = self.cameras.numpy()
        opacity = base.sigmoid(self.opacity.numpy()[indices])
        compact = self.training_options["compact_box"]
        support = np.full(len(indices), base.SUPPORT_RADIUS_SQUARED, np.float32)
        if compact["enabled"]:
            support.fill(0)
            visible = opacity > compact["alpha_min"]
            support[visible] = np.minimum(base.SUPPORT_RADIUS_SQUARED,
                compact["beta"] * 2 * np.log(opacity[visible] / compact["alpha_min"]))
        high_score, low_score = np.zeros(len(indices), np.float32), np.zeros(len(indices), np.float32)
        render = render_view or self.render_single_view_image
        for view in view_ids:
            image, _ = render(int(view))
            error = np.mean(np.abs(image - self.targets_host[int(view)]), axis=2)
            spread = float(error.max() - error.min())
            normalized = (error - error.min()) / spread if spread >= 1e-8 else np.zeros_like(error)
            high = normalized > self.training_options["multiview_adc"]["loss_threshold"]
            points = np.column_stack((means, np.ones(len(means), np.float32))) @ cameras[view].T
            visible = np.flatnonzero((points[:, 2] > base.NEAR_PLANE) & (support > 0))
            if not len(visible):
                continue
            vx, vy = base.projected_covariance_diagonal(scales[visible], rotations[visible], cameras[view], points[visible, :3], self.focal)
            cx = self.focal * points[visible, 0] / points[visible, 2] + self.width * 0.5
            cy = self.focal * points[visible, 1] / points[visible, 2] + self.height * 0.5
            rx, ry = np.sqrt(support[visible] * vx), np.sqrt(support[visible] * vy)
            x0, x1 = np.floor(cx - rx).astype(int), np.floor(cx + rx).astype(int)
            y0, y1 = np.floor(cy - ry).astype(int), np.floor(cy + ry).astype(int)
            keep = (x1 >= 0) & (y1 >= 0) & (x0 < self.width) & (y0 < self.height)
            local = visible[keep]
            x0, x1 = np.clip(x0[keep], 0, self.width - 1), np.clip(x1[keep], 0, self.width - 1)
            y0, y1 = np.clip(y0[keep], 0, self.height - 1), np.clip(y1[keep], 0, self.height - 1)
            for mask, score in ((high, high_score), (~high, low_score)):
                integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                score[local] += integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1] - integral[y1 + 1, x0] + integral[y0, x0]
        high_score /= max(1, len(view_ids))
        spread = float(low_score.max() - low_score.min())
        prune = (low_score - low_score.min()) / spread if spread > 0 else np.zeros_like(low_score)
        return high_score, prune

    def densify_and_prune(self, gradients, fraction, view_ids=None, late_prune=False, percent_dense=0.01, render_view=None):
        """Split only in the tangent plane and preserve material state when slots are reused."""
        means, logs = self.means.numpy(), self.log_scales.numpy()
        rotations, logits, colour = self.quaternions.numpy(), self.opacity.numpy(), self.color.numpy()
        reflection = self.reflection_logits.numpy()
        opacity = base.sigmoid(logits)
        prune = (opacity < base.PRUNE_OPACITY_FLOOR) & self.active
        parents = np.asarray([i for i in np.argsort(np.linalg.norm(gradients, axis=1))[::-1] if self.active[i]], np.intp)
        adc = self.training_options["multiview_adc"]
        if adc["enabled"] and view_ids is not None:
            scored = parents[:adc["candidate_limit"]]
            add_score, prune_score = self.multiview_adc_scores(scored, view_ids, render_view)
            parents = scored[add_score > adc["densify_score_threshold"]]
            if late_prune:
                prune[scored[prune_score > adc["prune_score_threshold"]]] = True
        if late_prune:
            prune |= (opacity < adc["opacity_cutoff"]) & self.active
        self.active[prune] = False
        parents = parents[self.active[parents]]
        free = np.flatnonzero(~self.active)
        requested = 0 if fraction <= 0 else min(len(free), max(1, int(self.active.sum() * fraction)))
        count = min(requested, len(parents))
        parents, children = parents[:count], free[:count]
        if count:
            self.active[children] = True
            means[children], logs[children] = means[parents], logs[parents]
            rotations[children], logits[children], colour[children] = rotations[parents], logits[parents], colour[parents]
            reflection[children] = reflection[parents]
            scales = np.exp(logs[parents])
            scales[:, 2] = 0.0
            large = scales[:, :2].max(axis=1) > percent_dense * self.scene_radius
            p, c = parents[large], children[large]
            if len(p):
                rotation = quaternion_to_matrix(rotations[p] / np.maximum(np.linalg.norm(rotations[p], axis=1, keepdims=True), 1e-8))
                offset = self.rng.normal(size=(2, len(p), 3)) * scales[large]
                origin = means[p].copy()
                means[p] = origin + np.einsum('nij,nj->ni', rotation, offset[0])
                means[c] = origin + np.einsum('nij,nj->ni', rotation, offset[1])
                logs[p, :2] -= np.log(1.6)
                logs[c, :2] = logs[p, :2]
        for field, values in ((self.means, means), (self.log_scales, logs), (self.quaternions, rotations),
                              (self.opacity, logits), (self.color, colour), (self.reflection_logits, reflection)):
            field.assign(values.astype(np.float32))
        self.active_device.assign(self.active.astype(np.int32))
        if requested:
            self.optimizers.reset()
            self.reflection_optimizer.reset_internal_state()
        return children

    def gaussian_set(self):
        model = super().gaussian_set()
        model.scales[:, 2] = 0.0
        return model

    def save_sidecars(self, path):
        path = Path(path)
        sidecar = path.with_suffix('.surfel.npz')
        np.savez_compressed(sidecar, format_version=np.int32(1), geometry='rank2_local_z_zero',
                            mode=self.surfel_options['mode'], specular_active=np.bool_(self.specular_active),
                            reflection_logits=self.reflection_logits.numpy()[self.active],
                            environment_logits=self.environment.numpy().reshape(self.env_height, self.env_width, 3),
                            scene_radius=np.float32(self.scene_radius))
        if self.surfel_options['save_debug']:
            self.save_debug(path)
            if self.surfel_options['mode'] == 'deferred':
                environment = base.sigmoid(self.environment.numpy()).reshape(self.env_height, self.env_width, 3)
                Image.fromarray(np.uint8(np.clip(environment, 0, 1) * 255)).save(path.with_name(path.stem + '.environment.png'))
        return [sidecar]

    def save_debug(self, path):
        values = self.buffers.numpy().reshape(self.height, self.width, 9)
        coverage = values[:, :, 7]
        normal = values[:, :, 3:6] / np.maximum(np.linalg.norm(values[:, :, 3:6], axis=2, keepdims=True), 1e-8)
        depth = values[:, :, 6] / np.maximum(coverage, 1e-8)
        visible = coverage > 1e-6
        normal_vis = np.where(visible[:, :, None], normal * 0.5 + 0.5, 0)
        depth_vis = np.zeros_like(depth)
        if np.any(visible):
            lo, hi = np.percentile(depth[visible], [1, 99])
            depth_vis[visible] = np.clip((depth[visible] - lo) / max(hi - lo, 1e-6), 0, 1)
        path = Path(path)
        for suffix, image in (('normals', normal_vis), ('depth', depth_vis), ('coverage', coverage)):
            Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).save(path.with_name(path.stem + f'.{suffix}.png'))
        if self.surfel_options['mode'] == 'deferred':
            strength = values[:, :, 8] / np.maximum(coverage, 1e-8)
            if not self.specular_active:
                strength = np.zeros_like(strength)
            Image.fromarray(np.uint8(np.clip(strength * coverage, 0, 1) * 255)).save(path.with_name(path.stem + '.reflection.png'))
        np.savez_compressed(path.with_name(path.stem + '.buffers.npz'), normal=normal, depth=depth, coverage=coverage)

    def load_materials(self, path, count, for_evaluation=False):
        values = read_surfel_sidecar(path, count)
        logits = self.reflection_logits.numpy()
        logits[:count] = values['reflection_logits']
        self.reflection_logits.assign(logits)
        if self.surfel_options['mode'] == 'deferred' and values['mode'] == 'deferred':
            env = values['environment_logits']
            if env.shape != (self.env_height, self.env_width, 3):
                raise ValueError('Environment resolution differs from the source checkpoint.')
            self.environment.assign(env.reshape(-1, 3))
            self.specular_active = values['specular_active']
        if for_evaluation:
            self.specular_active = values['specular_active']


def read_surfel_sidecar(path, count):
    with np.load(path, allow_pickle=False) as data:
        if int(data['format_version']) != 1 or str(data['geometry'].item()) != 'rank2_local_z_zero':
            raise ValueError('Unsupported surfel checkpoint format or geometry.')
        mode = str(data['mode'].item())
        reflection = np.asarray(data['reflection_logits'], np.float32)
        environment = np.asarray(data['environment_logits'], np.float32)
        if mode not in ('rgb', 'deferred') or reflection.shape != (count,):
            raise ValueError('Surfel sidecar and PLY rows do not match.')
        if environment.ndim != 3 or environment.shape[2] != 3 or min(environment.shape[:2]) < 1:
            raise ValueError('Invalid environment-map shape.')
        if not np.all(np.isfinite(reflection)) or not np.all(np.isfinite(environment)):
            raise ValueError('Non-finite surfel material values.')
        radius = float(data['scene_radius'])
        activated = bool(data['specular_active'])
        if not np.isfinite(radius) or radius <= 0 or (mode == 'rgb' and activated):
            raise ValueError('Invalid surfel radius or shading-stage metadata.')
        return {'mode': mode, 'reflection_logits': reflection, 'environment_logits': environment,
                'specular_active': activated, 'scene_radius': radius}
