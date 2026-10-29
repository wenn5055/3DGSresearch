"""Conservative silhouette contradictions from training alpha, independent of RGB.

A complete projected support box must lie in observed background in multiple
training cameras, repeatedly, before deletion. Occluded/foreground projections,
out-of-frame supports and behind-camera points supply no background vote.
Foreground overlap in a different camera may be another object's occlusion and
does not negate an observed background contradiction.
"""

import numpy as np

from .geometry import base


class ForegroundPruner:
    def __init__(self, alpha, cameras, focal, capacity, settings):
        self.settings = dict(settings)
        self.alpha = np.asarray(alpha, np.float32)
        self.cameras = np.asarray(cameras, np.float32)
        if (self.alpha.ndim != 3 or self.cameras.shape != (len(self.alpha), 4, 4)
                or not np.isfinite(self.alpha).all() or np.any(self.alpha < 0) or np.any(self.alpha > 1)):
            raise ValueError('Foreground pruning needs finite training-only alpha masks and matching cameras.')
        if len(self.alpha) < settings['min_background_views']:
            raise ValueError('Not enough training cameras for the configured background evidence threshold.')
        self.height, self.width = self.alpha.shape[1:]
        self.focal = float(focal)
        self.capacity = capacity
        _, unique = np.unique(self.cameras.reshape(len(self.cameras), -1).round(6), axis=0, return_index=True)
        unique = np.sort(unique)
        if len(unique) < settings['min_background_views']:
            raise ValueError('Foreground pruning needs distinct training camera poses.')
        poses = np.linalg.inv(self.cameras[unique])
        positions = poses[:, :3, 3]
        directions = positions / np.maximum(np.linalg.norm(positions, axis=1, keepdims=True), 1e-12)
        chosen = [0]
        distance = np.sum((directions - directions[0]) ** 2, axis=1)
        for _ in range(1, min(settings['views'], len(poses))):
            distance[chosen] = -1
            nxt = int(np.argmax(distance))
            chosen.append(nxt)
            distance = np.minimum(distance, np.sum((directions - directions[nxt]) ** 2, axis=1))
        self.views = sorted(int(unique[index]) for index in chosen)
        # Any non-background pixel protects the support box, including soft alpha.
        self.integrals = {}
        for view in self.views:
            possible_foreground = self.alpha[view] > settings['background_alpha_threshold']
            self.integrals[view] = np.pad(possible_foreground.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.strikes = np.zeros(capacity, np.int32)

    def reset_children(self, children):
        self.strikes[np.asarray(children, np.intp)] = 0

    def votes(self, means, log_scales, rotations, active, geometry='surfel'):
        indices = np.flatnonzero(active)
        background, foreground, observed = (np.zeros(self.capacity, np.int32) for _ in range(3))
        if not len(indices):
            return background, foreground, observed
        means = np.asarray(means)[indices]
        scales = np.exp(np.asarray(log_scales)[indices])
        if geometry == 'surfel':
            scales[:, 2] = 0.
        rotations = np.asarray(rotations)[indices]
        margin = self.settings['margin_pixels']
        for view in self.views:
            camera = self.cameras[view]
            points = means @ camera[:3, :3].T + camera[:3, 3]
            visible = np.flatnonzero(points[:, 2] > base.NEAR_PLANE)
            if not len(visible):
                continue
            vx, vy = base.projected_covariance_diagonal(
                scales[visible], rotations[visible], camera, points[visible], self.focal)
            cx = self.focal * points[visible, 0] / points[visible, 2] + self.width * .5
            cy = self.focal * points[visible, 1] / points[visible, 2] + self.height * .5
            # Full 3-sigma support is conservative even when compact support is smaller.
            rx = np.sqrt(base.SUPPORT_RADIUS_SQUARED * np.maximum(vx, 0))
            ry = np.sqrt(base.SUPPORT_RADIUS_SQUARED * np.maximum(vy, 0))
            x0 = np.floor(cx - rx - margin).astype(np.int64)
            y0 = np.floor(cy - ry - margin).astype(np.int64)
            x1 = np.ceil(cx + rx + margin).astype(np.int64) + 1
            y1 = np.ceil(cy + ry + margin).astype(np.int64) + 1
            inside = (x0 >= 0) & (y0 >= 0) & (x1 <= self.width) & (y1 <= self.height)
            local = indices[visible[inside]]
            if not len(local):
                continue
            x0, y0, x1, y1 = x0[inside], y0[inside], x1[inside], y1[inside]
            integral = self.integrals[view]
            count = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            observed[local] += 1
            foreground[local[count > 0]] += 1
            background[local[count == 0]] += 1
        return background, foreground, observed

    def update(self, means, log_scales, rotations, active, geometry='surfel'):
        background, foreground, observed = self.votes(means, log_scales, rotations, active, geometry)
        contradicted = active & (background >= self.settings['min_background_views'])
        self.strikes[~contradicted] = 0
        self.strikes[contradicted] = np.minimum(self.strikes[contradicted] + 1, self.settings['persistence'])
        prune = contradicted & (self.strikes >= self.settings['persistence'])
        return prune, {'active_before': int(np.count_nonzero(active)),
            'background_candidates': int(np.count_nonzero(contradicted)),
            'foreground_pruned': int(np.count_nonzero(prune)),
            'pruned_with_other_foreground_votes': int(np.count_nonzero(prune & (foreground > 0))),
            'unobserved': int(np.count_nonzero(active & (observed == 0))),
            'max_background_votes': int(background.max(initial=0)),
            'training_views': [int(view) for view in self.views]}
