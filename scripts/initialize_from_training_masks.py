"""Build a coarse surface hypothesis from posed training silhouettes only.

No validation/test images, depth, normals, Blender assets or material identities.
This is a visual hull: unseen concavities cannot be recovered from silhouettes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src')]
from gaussian_set import GaussianSet
from cp4282_gs.surfels.geometry import base


def carve(points, cameras, masks, focal):
    """Remove points observed in background; outside the image is unknown."""
    alive = np.ones(len(points), bool)
    seen = np.zeros(len(points), bool)
    height, width = masks.shape[1:]
    for camera, mask in zip(cameras, masks):
        indices = np.flatnonzero(alive)
        if not len(indices):
            break
        p = points[indices] @ camera[:3, :3].T + camera[:3, 3]
        in_front = p[:, 2] > base.NEAR_PLANE
        uv = np.zeros((len(p), 2), np.float32)
        uv[in_front] = focal * p[in_front, :2] / p[in_front, 2:3] + [width * .5, height * .5]
        inside = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        local = indices[inside]
        xy = uv[inside].astype(np.int32)
        seen[local] = True
        alive[local[~mask[xy[:, 1], xy[:, 0]]]] = False
    return alive & seen


def surface_from_occupancy(occupied, lower, step, smoothing=1.):
    nonzero = np.argwhere(occupied)
    if not len(nonzero):
        raise ValueError('Training silhouettes have no occupied intersection in the initialization bounds.')
    padding = max(4, int(math.ceil(3 * smoothing)))
    lo = np.maximum(nonzero.min(axis=0) - padding, 0)
    hi = np.minimum(nonzero.max(axis=0) + padding + 1, occupied.shape)
    cropped = occupied[tuple(slice(a, b) for a, b in zip(lo, hi))]
    field = (ndimage.distance_transform_edt(cropped) - ndimage.distance_transform_edt(~cropped)).astype(np.float32)
    field = ndimage.gaussian_filter(field, smoothing) if smoothing else field
    gradient = np.stack(np.gradient(field), axis=-1)
    points, normals = [], []
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        start, end = field[a], field[b]
        crossing = (start >= 0) != (end >= 0)
        ijk = np.argwhere(crossing)
        if not len(ijk):
            continue
        t = start[crossing] / (start[crossing] - end[crossing])
        index = ijk.astype(np.float32)
        index[:, axis] += t
        xyz = lower + (index[:, ::-1] + lo[::-1] + .5) * step
        normal = -((1 - t[:, None]) * gradient[a][crossing] + t[:, None] * gradient[b][crossing])[:, ::-1]
        length = np.linalg.norm(normal, axis=1)
        valid = length > 1e-8
        points.append(xyz[valid])
        normals.append(normal[valid] / length[valid, None])
    if not points:
        raise ValueError('No zero-crossing surface in the silhouette intersection.')
    return np.concatenate(points).astype(np.float32), np.concatenate(normals).astype(np.float32)


def normal_quaternions(normals):
    normals = np.asarray(normals, np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    q = np.column_stack((1 + normals[:, 2], -normals[:, 1], normals[:, 0], np.zeros(len(normals))))
    reverse = np.linalg.norm(q, axis=1) < 1e-10
    q[reverse] = [0., 1., 0., 0.]
    return (q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def initialize(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory.')
    started = perf_counter()
    manifest_path = args.data / 'transforms_train.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    cameras = np.asarray([base.blender_pose_to_world_to_camera(frame['transform_matrix']) for frame in manifest['frames']])
    poses = np.asarray([frame['transform_matrix'] for frame in manifest['frames']])
    bound = args.bound or float(np.linalg.norm(poses[:, :3, 3], axis=1).mean() * .35)
    lower, step = -bound, 2 * bound / args.grid_resolution
    masks, inputs, colour_sum, colour_count = [], [], np.zeros(3, np.float64), 0
    paths = [manifest_path]
    for frame in manifest['frames']:
        path = base.resolve_frame_image(args.data, frame['file_path'])
        paths.append(path)
        with Image.open(path) as source:
            if 'A' not in source.getbands():
                raise ValueError(f'Training alpha is required: {path}')
            rgba = np.asarray(source.convert('RGBA').resize((args.mask_resolution, args.mask_resolution), Image.Resampling.LANCZOS), np.float32) / 255
        mask = rgba[:, :, 3] >= args.alpha_threshold
        masks.append(ndimage.binary_dilation(mask, iterations=args.mask_margin) if args.mask_margin else mask)
        colour_sum += (rgba[:, :, :3] * rgba[:, :, 3:] + 1 - rgba[:, :, 3:]).sum(axis=(0, 1), dtype=np.float64)
        colour_count += args.mask_resolution ** 2
    masks = np.asarray(masks, bool)
    focal = .5 * args.mask_resolution / math.tan(.5 * manifest['camera_angle_x'])
    n = args.grid_resolution
    axis = lower + (np.arange(n, dtype=np.float32) + .5) * step
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    occupied = np.zeros((n, n, n), bool)
    for z0 in range(0, n, 4):
        z1 = min(z0 + 4, n)
        slab = np.empty((z1 - z0, n, n, 3), np.float32)
        slab[:, :, :, 0], slab[:, :, :, 1] = xx, yy
        slab[:, :, :, 2] = axis[z0:z1, None, None]
        occupied[z0:z1] = carve(slab.reshape(-1, 3), cameras, masks, focal).reshape(z1 - z0, n, n)
        if z0 % 64 == 0:
            print(f'Carved z={z0}/{n}; {perf_counter() - started:.1f}s', flush=True)
    points, normals = surface_from_occupancy(occupied, lower, step, args.smoothing)
    raw_count = len(points)
    if raw_count > args.max_points:
        selected = np.sort(np.random.default_rng(args.seed).choice(raw_count, args.max_points, replace=False))
        points, normals = points[selected], normals[selected]
    scale = step * args.scale_multiplier * math.sqrt(raw_count / len(points))
    scales = np.tile([scale, scale, scale * .3], (len(points), 1)).astype(np.float32)
    colour = np.broadcast_to((colour_sum / colour_count).astype(np.float32), (len(points), 3)).copy()
    scene = GaussianSet(points, scales, normal_quaternions(normals), np.full(len(points), .8, np.float32), colour)
    args.output.mkdir(parents=True, exist_ok=True)
    scene.to_ply(args.output / 'initial.ply')
    np.savez_compressed(args.output / 'surface.npz', points=points, normals=normals)
    metadata = {'method': 'training-alpha visual hull, smoothed signed-distance zero crossings',
        'inputs': [{'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        'training_views': len(masks), 'uses_held_out_images': False, 'uses_reference_geometry': False,
        'bounds_xyz': [[lower] * 3, [bound] * 3], 'grid_resolution': n, 'voxel_step': step,
        'mask_resolution': args.mask_resolution, 'alpha_threshold': args.alpha_threshold,
        'mask_margin_pixels': args.mask_margin, 'sdf_smoothing_voxels': args.smoothing,
        'occupied_voxels': int(occupied.sum()), 'raw_surface_points': raw_count,
        'exported_points': len(points), 'initial_tangent_scale': scale, 'seed': args.seed,
        'seconds': perf_counter() - started,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitation': 'Silhouettes cannot recover unseen concavities; this is an inferred initializer, not ground-truth geometry.'}
    (args.output / 'initialization.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in metadata.items() if key != 'inputs'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--grid-resolution', type=int, default=256)
    parser.add_argument('--mask-resolution', type=int, default=800)
    parser.add_argument('--max-points', type=int, default=100000)
    parser.add_argument('--bound', type=float)
    parser.add_argument('--alpha-threshold', type=float, default=.5)
    parser.add_argument('--mask-margin', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=1.)
    parser.add_argument('--scale-multiplier', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if (args.grid_resolution < 8 or args.mask_resolution < 1 or args.max_points < 1
            or args.mask_margin < 0 or args.smoothing < 0 or args.scale_multiplier <= 0
            or not 0 < args.alpha_threshold < 1 or (args.bound is not None and args.bound <= 0)):
        parser.error('Invalid grid, mask, scale or bounds settings.')
    initialize(args)


if __name__ == '__main__':
    main()
