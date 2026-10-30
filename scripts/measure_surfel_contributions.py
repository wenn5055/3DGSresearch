"""Measure actual T*alpha visibility, split by training foreground/background.

Read-only Unit 14 diagnostic. Visibility is not a deletion-error bound for a
nonlinear deferred shader. This tool neither prunes nor uses held-out images.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from gaussian_set import GaussianSet
from cp4282_gs.surfels import geometry as g
from cp4282_gs.surfels.geometry_losses import load_training_alpha
from evaluate_surfels import build_trainer as build_surfel

evaluator = importlib.import_module('3dgs_evaluator')
base = g.base


@wp.kernel(enable_backward=False)
def accumulate(means: wp.array(dtype=wp.vec3), scales: wp.array(dtype=wp.vec3),
               rotations: wp.array(dtype=wp.vec4), opacity: wp.array(dtype=wp.float32),
               cameras: wp.array(dtype=wp.mat44), view: int,
               offsets: wp.array(dtype=wp.int32), pairs: wp.array(dtype=wp.uint64),
               width: int, height: int, tiles_x: int, focal: float, surfel: int,
               compact: int, beta: float, alpha_min: float, masks: wp.array(dtype=wp.float32),
               total: wp.array(dtype=wp.float32), foreground: wp.array(dtype=wp.float32),
               background: wp.array(dtype=wp.float32), coverage: wp.array(dtype=wp.float32)):
    pixel = wp.tid()
    x, y = pixel % width, pixel // width
    tile = (y // base.TILE) * tiles_x + x // base.TILE
    transmittance = float(1.)
    label = masks[view * width * height + pixel]
    for entry in range(offsets[tile], offsets[tile + 1]):
        i = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        alpha = float(0.)
        if surfel != 0:
            alpha = g.alpha_at_pixel(means[i], scales[i], rotations[i], opacity[i], cameras[view],
                float(x) + .5, float(y) + .5, float(width), float(height), focal, compact, beta, alpha_min)
        else:
            alpha = base.alpha_at_pixel(means[i], scales[i], rotations[i], opacity[i], cameras[view],
                float(x) + .5, float(y) + .5, float(width), float(height), focal, compact, beta, alpha_min)
        if alpha > 0.:
            weight = transmittance * alpha
            wp.atomic_add(total, i, weight)
            wp.atomic_add(foreground, i, weight * label)
            wp.atomic_add(background, i, weight * (1. - label))
            transmittance *= 1. - alpha
            if transmittance < base.TRANSMITTANCE_CUTOFF:
                break
    coverage[pixel] = 1. - transmittance


def measure(trainer, alpha, views=None, is_surfel=True):
    alpha = np.asarray(alpha, np.float32)
    if alpha.shape != (len(trainer.targets_host), trainer.height, trainer.width):
        raise ValueError('Contribution masks must match the supplied training-only image set.')
    views = list(range(len(alpha))) if views is None else list(views)
    if not views or len(set(views)) != len(views) or any(view < 0 or view >= len(alpha) for view in views):
        raise ValueError('Contribution cameras must be unique indices in the supplied training set.')
    masks = wp.array(alpha.reshape(-1), dtype=wp.float32, device=trainer.device)
    arrays = [wp.zeros(trainer.capacity, dtype=wp.float32, device=trainer.device) for _ in range(3)]
    coverage = wp.zeros(trainer.width * trainer.height, dtype=wp.float32, device=trainer.device)
    coverage_sum = 0.
    for view in views:
        offsets, pairs, _ = trainer.build_tiles(np.asarray([view], np.int32))
        wp.launch(accumulate, trainer.width * trainer.height,
            inputs=[trainer.means, trainer.log_scales, trainer.quaternions, trainer.opacity,
                    trainer.cameras, int(view), offsets, pairs, trainer.width, trainer.height,
                    trainer.tiles_x, trainer.focal, int(is_surfel)] + trainer.compact_box_args + [masks],
            outputs=arrays + [coverage], device=trainer.device)
        coverage_sum += float(coverage.numpy().sum(dtype=np.float64))
    values = np.stack([array.numpy() for array in arrays], axis=1)
    np.testing.assert_allclose(values[:, 0], values[:, 1] + values[:, 2], rtol=3e-4, atol=1e-4)
    if not np.isclose(values[:, 0].sum(dtype=np.float64), coverage_sum, rtol=3e-4, atol=1e-4):
        raise AssertionError('Per-splat contributions do not sum to projected coverage.')
    return values, coverage_sum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arch', choices=('auto', 'cpu', 'gpu'), default='gpu')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory.')
    record = json.loads((args.run / 'run.json').read_text(encoding='utf-8'))
    if record['status'] != 'completed':
        raise ValueError('Measure a completed frozen checkpoint.')
    config = record['config']
    data = Path(config['paths']['data'])
    width, height = config['model']['width'], config['model']['height']
    background = np.ones(3, np.float32) if config['model']['background'] == 'white' else np.zeros(3, np.float32)
    targets, cameras, focal, distance, frames = base.load_views(data, width, height, background,
                                                              manifest_name='transforms_train.json')
    alpha = load_training_alpha(data, width, height)
    wp.init()
    device = evaluator.resolve_device(args.arch)
    scene = GaussianSet.from_ply(args.run / 'model.ply')
    model_args = argparse.Namespace(ply=args.run / 'model.ply', background=config['model']['background'],
                                   tile_pair_capacity=None, sh_degree=0, sh_npz=None)
    is_surfel = (args.run / 'model.surfel.npz').exists()
    if is_surfel:
        trainer = build_surfel(model_args, scene, targets, cameras, focal, device)
    else:
        trainer, _ = evaluator.build_trainer(model_args, scene, targets, cameras, focal, distance, device)
    values, coverage = measure(trainer, alpha, is_surfel=is_surfel)
    total, foreground, background = values.T
    order = np.argsort(total)
    summary = {'run': str(args.run.resolve()), 'training_views': len(frames),
        'manifest': 'transforms_train.json', 'resolution': [width, height], 'active_splats': len(total),
        'total_contribution': float(total.sum(dtype=np.float64)), 'coverage_sum': coverage,
        'zero_total_splats': int(np.sum(total == 0)), 'zero_foreground_splats': int(np.sum(foreground == 0)),
        'background_dominated_99_percent': int(np.sum((total > 0) & (background >= .99 * total))),
        'lowest_40_percent_contribution_fraction': float(total[order[:int(.4 * len(order))]].sum(dtype=np.float64) / max(coverage, 1e-12)),
        'scope': 'All training pixels/cameras; foreground uses original soft alpha. Read-only visibility measurement, no pruning.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': [{'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in [data / 'transforms_train.json'] + [base.resolve_frame_image(data, frame) for frame in frames]],
        'checkpoints': [{'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in [args.run / 'model.ply', args.run / 'model.surfel.npz', args.run / 'model.sh.npz'] if path.exists()]}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'weights.npz', total=total, foreground=foreground,
                        background=background, opacity=scene.opacities)
    summary['weights_sha256'] = hashlib.sha256((args.output / 'weights.npz').read_bytes()).hexdigest()
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key not in ('inputs', 'checkpoints')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
