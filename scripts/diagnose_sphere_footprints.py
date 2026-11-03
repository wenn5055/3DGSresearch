"""Inspect saved sphere runs: projected footprint distributions and normal/lighting maps.

Geometry references, when supplied, are evaluation-only Blender exports.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from gaussian_set import GaussianSet
from splat_math import quaternion_to_matrix
from cp4282_gs.surfels.geometry import base
from evaluate_surfels import build_trainer
from compare_surfel_runs import sheet, fingerprint


def footprint(scene, camera, focal, width, height, compact):
    points = scene.means @ camera[:3, :3].T + camera[:3, 3]
    positive = points[:, 2] > base.NEAR_PLANE
    z = np.maximum(points[:, 2], base.NEAR_PLANE)
    uv = focal * points[:, :2] / z[:, None] + [width/2, height/2]
    keep = positive & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    rotations = quaternion_to_matrix(scene.rotations / np.sqrt((scene.rotations**2).sum(axis=1, keepdims=True)+1e-8))
    tangent = np.einsum('ij,njk->nik', camera[:3, :3], rotations[:, :, :2]) * scene.scales[:, None, :2]
    jacobian = np.zeros((len(points), 2, 3))
    jacobian[:, 0, 0], jacobian[:, 1, 1] = focal/z, focal/z
    jacobian[:, 0, 2], jacobian[:, 1, 2] = -focal*points[:, 0]/z**2, -focal*points[:, 1]/z**2
    projected = np.einsum('nij,njk->nik', jacobian, tangent)
    covariance = projected @ projected.transpose(0, 2, 1) + np.eye(2)[None]*base.FILTER_VARIANCE
    sigma = np.sqrt(np.linalg.eigvalsh(covariance)[:, 1])
    support = np.full(len(points), base.SUPPORT_RADIUS_SQUARED, np.float64)
    if compact['enabled']:
        support = np.minimum(support, compact['beta']*2*np.log(np.maximum(scene.opacities, 1e-12)/compact['alpha_min']))
    keep &= support > 0
    return sigma[keep], 2*sigma[keep]*np.sqrt(support[keep])


def save_rgb(path, value):
    Image.fromarray(np.uint8(np.clip(value, 0, 1)*255)).save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry-reference', type=Path)
    parser.add_argument('--views', nargs='+', type=int, default=[0, 13, 73, 88, 51])
    args = parser.parse_args()
    if len(args.runs) != len(args.labels):
        parser.error('One label per run is required.')
    args.output.mkdir(parents=True, exist_ok=False)
    configs = [json.loads((run/'run.json').read_text())['config'] for run in args.runs]
    config = configs[0]
    size = config['model']['width']
    if any((c['paths']['data'], c['model']['width'], c['model']['height']) !=
           (config['paths']['data'], size, size) for c in configs):
        raise ValueError('Dataset and square resolution must match.')
    data = Path(config['paths']['data'])
    targets, cameras, focal, _, _ = base.load_views(data, size, size, np.ones(3, np.float32), manifest_name='transforms_val.json')
    selected_targets, selected_cameras = targets[args.views], cameras[args.views]
    reference = {}
    metadata = None
    if args.geometry_reference:
        metadata = json.loads((args.geometry_reference/'geometry.json').read_text())
        assert metadata['status'] == 'completed' and metadata['resolution'] == [size, size]
        assert metadata['manifest_sha256'] == fingerprint(data/'transforms_val.json')['sha256']
    normal_images, error_images, rgb_images = {}, {}, {}
    rows, source_records, distributions = [], [], []
    for view in args.views:
        with Image.open(data/f'val/r_{view}.png') as im:
            reference[view] = np.asarray(im.convert('RGBA'))[:, :, 3]/255 >= .95
        rgb_images[view] = [save_rgb(args.output/f'reference-{view}.png', targets[view])]
        normal_images[view], error_images[view] = [], []
        if metadata:
            with np.load(args.geometry_reference/f'r_{view}.npz') as truth:
                normal_images[view].append(save_rgb(args.output/f'mesh-normal-{view}.png',
                    np.where((truth['object_id']>0)[:, :, None], truth['normal']*.5+.5, 0)))
    wp.init()
    for index, (run, label, config) in enumerate(zip(args.runs, args.labels, configs)):
        scene = GaussianSet.from_ply(run/'model.ply')
        trainer = build_trainer(argparse.Namespace(ply=run/'model.ply', background='white', tile_pair_capacity=None,
                                  sh_degree=None), scene, selected_targets, selected_cameras, focal, 'cuda:0')
        source_records.append({'run': str(run.resolve()), 'label': label,
                               'checkpoints': [fingerprint(run/file) for file in ('model.ply', 'model.surfel.npz')]})
        for local, view in enumerate(args.views):
            image, _ = trainer.render_single_view_image(local)
            rgb_images[view].append(save_rgb(args.output/f'model-{index}-{view}.png', image))
            values = trainer.buffers.numpy().reshape(size, size, 9)
            length = np.linalg.norm(values[:, :, 3:6], axis=2)
            normals = values[:, :, 3:6]/np.maximum(length[:, :, None], 1e-8)
            coverage = values[:, :, 7]
            valid = reference[view] & (coverage >= .5) & (length > 1e-8)
            normal_images[view].append(save_rgb(args.output/f'normal-{index}-{view}.png',
                np.where(valid[:, :, None], normals*.5+.5, 0)))
            sigma, diameter = footprint(scene, cameras[view], focal, size, size, config['compact_box'])
            row = {'model': label, 'view': view,
                   'foreground_coverage_fraction': float(((coverage >= .5)&reference[view]).sum()/reference[view].sum()),
                   'normal_coherence': float((length[valid]/np.maximum(coverage[valid], 1e-8)).mean()),
                   'candidate_count': len(sigma)}
            for key, values_array in [('sigma_major_px', sigma), ('support_major_diameter_px', diameter)]:
                for percentile, value in zip((5, 50, 95), np.percentile(values_array, [5, 50, 95])):
                    row[f'{key}_p{percentile}'] = float(value)
            if metadata:
                with np.load(args.geometry_reference/f'r_{view}.npz') as truth:
                    truth_valid = valid & (truth['object_id']>0)
                    angles = np.degrees(np.arccos(np.clip((normals*truth['normal']).sum(axis=2), -1, 1)))
                    row['mesh_normal_mae_deg'] = float(angles[truth_valid].mean())
                    ray_mask = truth['object_id']>0
                    row['reference_alpha_mesh_iou'] = float((ray_mask&reference[view]).sum()/(ray_mask|reference[view]).sum())
                    heat = plt.get_cmap('magma')(np.clip(angles/30, 0, 1))[:, :, :3]
                    error_images[view].append(save_rgb(args.output/f'normal-error-{index}-{view}.png',
                        np.where(truth_valid[:, :, None], heat, 0)))
            rows.append(row)
            if local == 0:
                distributions.append(diameter)
        del trainer
    with (args.output/'diagnostics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    sheet(args.output/'normal-comparison.png', [(f'Validation view {v}', normal_images[v]) for v in args.views],
          (['Reference mesh'] if metadata else [])+args.labels)
    sheet(args.output/'rgb-comparison.png', [(f'Validation view {v}', rgb_images[v]) for v in args.views], ['Reference']+args.labels)
    if metadata:
        sheet(args.output/'normal-errors.png', [(f'Validation view {v}: error 0..30 degrees, clipped', error_images[v]) for v in args.views], args.labels)
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    for label, values_array in zip(args.labels, distributions):
        ax.hist(values_array, bins=80, density=True, histtype='step', label=label)
    ax.set(xlabel='Projected support major-axis diameter (pixels)', ylabel='Density',
           title='justChromeBall | validation view 0 | primitive candidates, not visibility weighted', xlim=(0, 35))
    ax.legend(); fig.savefig(args.output/'footprints.png', dpi=170); plt.close(fig)
    hdr_runs = [(label, run/'model.environment.png') for label, run, c in zip(args.labels, args.runs, configs)
                if c['surfel']['mode']=='deferred']
    if hdr_runs:
        sheet(args.output/'learned-environments.png', [('Learned HDR environment, clipped sRGB display', [p for _, p in hdr_runs])],
              [label for label, _ in hdr_runs])
    protocol = {'status': 'completed', 'runs': source_records, 'views': args.views,
                'footprint_scope': 'positive-depth primitives whose projected centers are inside the image; includes occluded/back-side primitives; not contribution weighted',
                'normal_error': 'reference mesh normals; valid reference foreground with coverage>=0.5; heatmap clipped at 30 degrees',
                'geometry_reference': metadata, 'script': fingerprint(Path(__file__))}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('Diagnostic output:', args.output, flush=True)


if __name__ == '__main__':
    main()
