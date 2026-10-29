"""Rank-two covariance surfels: two tangent scales and the rotated local-z normal.

The third stored log-scale is a compatibility placeholder, never exponentiated here.
Screen projection keeps the baseline's EWA approximation and pixel filter; it is not
the ray/plane intersection renderer of 2DGS.
"""

import importlib

import numpy as np
import warp as wp

base = importlib.import_module("3dgs_trainer")
workspace_module = importlib.import_module("gaussian_first_tile_workspace_gpu")


@wp.func
def camera_position(camera: wp.mat44):
    t = wp.vec3(camera[0, 3], camera[1, 3], camera[2, 3])
    return -wp.vec3(wp.dot(wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0]), t),
                    wp.dot(wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1]), t),
                    wp.dot(wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2]), t))


@wp.func
def surfel_normal(quaternion: wp.vec4, mean: wp.vec3, camera: wp.mat44):
    q = quaternion / wp.sqrt(wp.dot(quaternion, quaternion) + 1.0e-8)
    w, x, y, z = q[0], q[1], q[2], q[3]
    normal = wp.vec3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))
    if wp.dot(normal, camera_position(camera) - mean) < 0.0:
        normal = -normal
    return normal


@wp.func
def normal_objective(quaternion: wp.vec4, mean: wp.vec3, camera: wp.mat44, gradient: wp.vec3):
    return wp.dot(surfel_normal(quaternion, mean, camera), gradient)


@wp.func
def projected_covariance(log_scale: wp.vec3, quaternion: wp.vec4, camera: wp.mat44,
                         x: float, y: float, z: float, focal: float):
    q = quaternion / wp.sqrt(wp.dot(quaternion, quaternion) + 1.0e-8)
    w, qx, qy, qz = q[0], q[1], q[2], q[3]
    tangent_x = wp.vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + w * qz), 2.0 * (qx * qz - w * qy))
    tangent_y = wp.vec3(2.0 * (qx * qy - w * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + w * qx))
    jx = wp.vec3(focal / z, 0.0, -focal * x / (z * z))
    jy = wp.vec3(0.0, focal / z, -focal * y / (z * z))
    c0 = wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0])
    c1 = wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1])
    c2 = wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2])
    px = wp.vec3(wp.dot(jx, c0), wp.dot(jx, c1), wp.dot(jx, c2))
    py = wp.vec3(wp.dot(jy, c0), wp.dot(jy, c1), wp.dot(jy, c2))
    ax, ay = wp.dot(px, tangent_x), wp.dot(px, tangent_y)
    bx, by = wp.dot(py, tangent_x), wp.dot(py, tangent_y)
    sx2, sy2 = wp.exp(2.0 * log_scale[0]), wp.exp(2.0 * log_scale[1])
    return wp.vec3(sx2 * ax * ax + sy2 * ay * ay + base.FILTER_VARIANCE,
                   sx2 * ax * bx + sy2 * ay * by,
                   sx2 * bx * bx + sy2 * by * by + base.FILTER_VARIANCE)


@wp.func
def alpha_at_pixel(mean: wp.vec3, log_scale: wp.vec3, quaternion: wp.vec4, opacity_logit: float,
                   camera: wp.mat44, px: float, py: float, width: float, height: float,
                   focal: float, compact_enabled: int, beta: float, alpha_min: float):
    point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)
    z = wp.max(point[2], base.NEAR_PLANE)
    covariance = projected_covariance(log_scale, quaternion, camera, point[0], point[1], z, focal)
    determinant = wp.max(covariance[0] * covariance[2] - covariance[1] * covariance[1], 1.0e-8)
    dx = px - (focal * point[0] / z + width * 0.5)
    dy = py - (focal * point[1] / z + height * 0.5)
    q = (covariance[2] * dx * dx - 2.0 * covariance[1] * dx * dy + covariance[0] * dy * dy) / determinant
    opacity = 1.0 / (1.0 + wp.exp(-opacity_logit))
    support = float(base.SUPPORT_RADIUS_SQUARED)
    if compact_enabled != 0:
        support = 0.0
        if opacity > alpha_min:
            support = wp.min(float(base.SUPPORT_RADIUS_SQUARED), beta * 2.0 * wp.log(opacity / alpha_min))
    alpha = float(0.0)
    if support > 0.0 and q <= support:
        candidate = wp.min(opacity * wp.exp(-0.5 * q), 0.99)
        if candidate >= base.ALPHA_CUTOFF:
            alpha = candidate
    return alpha


@wp.kernel(enable_backward=False)
def project_records(means: wp.array(dtype=wp.vec3), log_scales: wp.array(dtype=wp.vec3),
                    quaternions: wp.array(dtype=wp.vec4), opacity_logits: wp.array(dtype=wp.float32),
                    active: wp.array(dtype=wp.int32), cameras: wp.array(dtype=wp.mat44),
                    view_ids: wp.array(dtype=wp.int32), capacity: int, width: int, height: int,
                    focal: float, compact: int, beta: float, alpha_min: float,
                    centres: wp.array(dtype=wp.vec2), conics: wp.array(dtype=wp.vec3),
                    supports: wp.array(dtype=wp.float32), depths: wp.array(dtype=wp.float32),
                    groups: wp.array(dtype=wp.int32), ids: wp.array(dtype=wp.uint32)):
    splat = wp.tid()
    centres[splat] = wp.vec2(0.0)
    conics[splat] = wp.vec3(0.0)
    supports[splat] = 0.0
    depths[splat] = 0.0
    groups[splat] = 0
    ids[splat] = wp.uint32(splat)
    if active[splat] != 0:
        camera = cameras[view_ids[0]]
        point = camera * wp.vec4(means[splat][0], means[splat][1], means[splat][2], 1.0)
        if point[2] > base.NEAR_PLANE:
            cov = projected_covariance(log_scales[splat], quaternions[splat], camera,
                                       point[0], point[1], point[2], focal)
            determinant = cov[0] * cov[2] - cov[1] * cov[1]
            support = float(base.SUPPORT_RADIUS_SQUARED)
            if compact != 0:
                opacity = 1.0 / (1.0 + wp.exp(-opacity_logits[splat]))
                support = 0.0
                if opacity > alpha_min:
                    support = wp.min(float(base.SUPPORT_RADIUS_SQUARED), beta * 2.0 * wp.log(opacity / alpha_min))
            if determinant > 1.0e-12 and support > 0.0:
                centres[splat] = wp.vec2(focal * point[0] / point[2] + float(width) * 0.5,
                                         focal * point[1] / point[2] + float(height) * 0.5)
                conics[splat] = wp.vec3(cov[2] / determinant, -cov[1] / determinant, cov[0] / determinant)
                depths[splat] = point[2]
                supports[splat] = support


class SurfelTileWorkspace(workspace_module.GaussianFirstTileWorkspace):
    def build(self, means, log_scales, quaternions, opacity, active_device, cameras,
              view_ids, width, height, focal, compact_box):
        if len(view_ids) != 1:
            raise ValueError("Surfel rendering expects exactly one camera at a time.")
        self.view_ids.assign(np.asarray(view_ids, np.int32))
        wp.launch(project_records, dim=self.capacity,
                  inputs=[means, log_scales, quaternions, opacity, active_device, cameras, self.view_ids,
                          self.capacity, width, height, focal, int(compact_box["enabled"]),
                          compact_box["beta"], compact_box["alpha_min"]],
                  outputs=[self.projected_centres, self.projected_conics, self.projected_supports,
                           self.projected_depths, self.projected_group_ids, self.projected_splat_ids], device=self.device)
        return self.builder.build(self.projected_centres, self.projected_conics, self.projected_supports,
                                  self.projected_depths, self.projected_group_ids, self.projected_splat_ids,
                                  self.capacity, self.tiles_per_view, width, height)
