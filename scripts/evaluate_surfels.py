"""Evaluate surfel checkpoints using their saved geometry and material sidecar."""

import argparse
import csv
import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src')]
from gaussian_set import GaussianSet
from image_metrics import ssim_rgb
from cp4282_gs.surfels.trainer import SurfelTrainer, read_surfel_sidecar
from cp4282_gs.surfels.geometry import base

metrics_module = importlib.import_module('3dgs_evaluator')


def build_trainer(args, scene, targets, cameras, focal, device):
    metadata_path = args.ply.with_suffix('.surfel.npz')
    metadata = read_surfel_sidecar(metadata_path, len(scene.means))
    environment = metadata['environment_logits']
    requested_degree = getattr(args, 'sh_degree', None)
    if requested_degree is not None and requested_degree > metadata['sh_degree']:
        raise ValueError('Requested SH degree was not trained in this checkpoint.')
    background = np.ones(3, np.float32) if args.background == 'white' else np.zeros(3, np.float32)
    trainer = SurfelTrainer(targets, cameras, focal, metadata['scene_radius'], len(scene.means), len(scene.means), device, 0,
        init_scene=scene, background=background, tile_pair_capacity=args.tile_pair_capacity,
        training_view_count=0,
        sh_degree=metadata['max_sh_degree'],
        surfel_options={'mode': metadata['mode'], 'environment_width': environment.shape[1],
                        'environment_height': environment.shape[0], 'save_debug': False})
    trainer.load_materials(metadata_path, len(scene.means), for_evaluation=True)
    if requested_degree is not None and trainer.sh is not None:
        trainer.sh.degree = requested_degree
    return trainer


def evaluate(args):
    wp.init()
    device = metrics_module.resolve_device(args.arch)
    width, height = args.width or args.resolution, args.height or args.resolution
    background = np.ones(3, np.float32) if args.background == 'white' else np.zeros(3, np.float32)
    targets, cameras, focal, _, frames = base.load_views(args.data, width, height, background, manifest_name=args.manifest)
    scene = GaussianSet.from_ply(args.ply)
    trainer = build_trainer(args, scene, targets, cameras, focal, device)
    print(f'Evaluating {len(scene.means):,} surfels: stage={trainer.surfel_stage}, manifest={args.manifest}, resolution={width}x{height}.', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    lpips = metrics_module.LPIPSMetric(args.lpips_net) if 'lpips' in args.metrics else None
    fields = ['view', 'file', 'mse'] + [{'psnr': 'psnr_db', 'ssim': 'ssim', 'lpips': 'lpips'}[name] for name in dict.fromkeys(args.metrics)]
    rows = []
    for view, frame in enumerate(frames):
        image, mse = trainer.render_single_view_image(view)
        image = np.clip(image, 0, 1)
        row = {'view': view, 'file': frame, 'mse': mse}
        if 'psnr' in args.metrics:
            row['psnr_db'] = metrics_module.psnr_from_mse(mse)
        if 'ssim' in args.metrics:
            row['ssim'] = ssim_rgb(image, targets[view])
        if lpips is not None:
            row['lpips'] = lpips(image, targets[view])
        Image.fromarray(np.uint8(image * 255)).save(args.output_dir / f'{Path(frame).stem}.png')
        if args.save_buffers:
            trainer.save_debug(args.output_dir / f'{Path(frame).stem}.png')
        rows.append(row)
        print(f'[{view:4d}] {frame}: mse={mse:.6f}', flush=True)
    with args.csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Mean metrics: ' + ', '.join(f'{field}={np.mean([row[field] for row in rows]):.6f}' for field in fields[2:]), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ply', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('--manifest', default='transforms_test.json')
    parser.add_argument('--arch', choices=('auto', 'cpu', 'gpu'), default='auto')
    parser.add_argument('--resolution', type=int, default=base.DEFAULT_RESOLUTION)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--background', choices=('white', 'black'), default='white')
    parser.add_argument('--sh-degree', type=int, choices=(0, 1, 2, 3), default=None,
                        help='Truncate saved base SH; default restores the checkpoint degree.')
    parser.add_argument('--metrics', nargs='+', choices=('psnr', 'ssim', 'lpips'), default=['psnr', 'ssim', 'lpips'])
    parser.add_argument('--lpips-net', choices=('alex', 'vgg', 'squeeze'), default='alex')
    parser.add_argument('--tile-pair-capacity', type=int)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/surfel-eval'))
    parser.add_argument('--csv', type=Path, default=Path('outputs/surfel-eval/metrics.csv'))
    parser.add_argument('--save-buffers', action='store_true')
    return parser


if __name__ == '__main__':
    evaluate(build_parser().parse_args())
