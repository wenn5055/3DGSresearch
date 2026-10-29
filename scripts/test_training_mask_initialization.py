"""Analytic shape recovery and training-only evidence for silhouette initialization."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from initialize_from_training_masks import carve, surface_from_occupancy, normal_quaternions, initialize
from splat_math import quaternion_to_matrix
from cp4282_gs.surfels.geometry import base


def sphere_views(size=96):
    poses, masks = [], []
    focal = .5 * size / np.tan(np.pi / 6)
    yy, xx = np.mgrid[:size, :size]
    for elevation in (-.7, 0., .7):
        for angle in np.arange(8) * np.pi / 4:
            eye = np.array([np.cos(elevation) * np.cos(angle), np.cos(elevation) * np.sin(angle), np.sin(elevation)]) * 2
            forward = -eye / np.linalg.norm(eye)
            right = np.cross(forward, [0, 0, 1.]); right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            pose = np.eye(4); pose[:3, :3] = np.column_stack((right, up, -forward)); pose[:3, 3] = eye
            local = np.stack(((xx + .5 - size * .5) / focal, -(yy + .5 - size * .5) / focal, -np.ones_like(xx)), axis=-1)
            rays = local @ pose[:3, :3].T
            rays /= np.linalg.norm(rays, axis=2, keepdims=True)
            dot = np.sum(rays * eye, axis=2)
            hit = (dot * dot - (np.dot(eye, eye) - .3 ** 2) >= 0) & (dot < 0)
            poses.append(pose); masks.append(hit)
    return np.asarray(poses), np.asarray(masks), focal


class MaskInitializationTests(unittest.TestCase):
    def test_reconstructed_sphere_surface_and_outward_normals(self):
        poses, masks, focal = sphere_views()
        cameras = np.asarray([base.blender_pose_to_world_to_camera(pose) for pose in poses])
        n, lower, step = 48, -.6, 1.2 / 48
        zz, yy, xx = np.meshgrid(*(lower + (np.arange(n) + .5) * step for _ in range(3)), indexing='ij')
        points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
        occupied = carve(points, cameras, masks, focal).reshape(n, n, n)
        surface, normals = surface_from_occupancy(occupied, lower, step)
        radius = np.linalg.norm(surface, axis=1)
        self.assertLess(float(np.mean(np.abs(radius - .3))), .025)
        expected = surface / radius[:, None]
        angle = np.degrees(np.arccos(np.clip(np.sum(expected * normals, axis=1), -1, 1)))
        self.assertLess(float(angle.mean()), 15.)
        self.assertGreater(float(np.mean(np.sum(expected * normals, axis=1) > 0)), .99)

    def test_outside_image_or_behind_one_camera_is_unknown_not_background(self):
        cameras = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        cameras[1, 0, 3] = -10
        masks = np.stack((np.zeros((16, 16), bool), np.ones((16, 16), bool)))
        np.testing.assert_array_equal(carve(np.array([[10., 0, 1]]), cameras, masks, 16), [True])
        cameras[1] = np.diag([-1., 1., -1., 1.])
        np.testing.assert_array_equal(carve(np.array([[0., 0, -1]]), cameras, masks, 16), [True])
        cameras[1] = np.eye(4)
        np.testing.assert_array_equal(carve(np.array([[0., 0, 1]]), cameras, masks, 16), [False])

    def test_quaternions_map_local_z_to_surface_normal_including_antipode(self):
        normals = np.array([[0, 0, 1], [0, 0, -1], [.001, .001, -1], [1, 0, 0], [0, 1, 0]], np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        actual = quaternion_to_matrix(normal_quaternions(normals))[:, :, 2]
        np.testing.assert_allclose(actual, normals, atol=2e-6)

    def test_initializer_never_reads_validation_test_or_reference_geometry(self):
        poses, masks, _ = sphere_views(size=48)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            root = Path(directory); data = root / 'data'; (data / 'train').mkdir(parents=True)
            frames = []
            for i, (pose, mask) in enumerate(zip(poses, masks)):
                rgba = np.full((48, 48, 4), 120, np.uint8); rgba[:, :, 3] = mask * 255
                Image.fromarray(rgba).save(data / 'train' / f'r_{i}.png')
                frames.append({'file_path': f'./train/r_{i}', 'transform_matrix': pose.tolist()})
            (data / 'transforms_train.json').write_text(json.dumps({'camera_angle_x': np.pi / 3, 'frames': frames}))
            for name in ('transforms_val.json', 'transforms_test.json', 'reference_geometry.npz'):
                (data / name).write_text('Invalid sentinel: must never be read.')
            args = argparse.Namespace(data=data, output=root / 'initial', grid_resolution=32,
                mask_resolution=48, max_points=1000, bound=.6, alpha_threshold=.5,
                mask_margin=0, smoothing=1., scale_multiplier=1., seed=7)
            with contextlib.redirect_stdout(io.StringIO()):
                initialize(args)
            result = json.loads((args.output / 'initialization.json').read_text())
            self.assertFalse(result['uses_held_out_images'])
            self.assertFalse(result['uses_reference_geometry'])
            self.assertEqual(len(result['inputs']), len(frames) + 1)
            self.assertTrue((args.output / 'initial.ply').exists())


if __name__ == '__main__':
    unittest.main()
