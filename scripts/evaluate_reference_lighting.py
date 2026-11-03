"""Swap reference illumination into a frozen saved model, without optimization or file mutation."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'shared'), str(ROOT/'src'), str(ROOT/'scripts')]
from gaussian_set import GaussianSet
from image_metrics import ssim_rgb
from cp4282_gs.surfels.geometry import base
from evaluate_surfels import build_trainer, metrics_module
from compare_surfel_runs import fingerprint, sheet, verify_sources


def inverse_softplus(linear):
    value = np.maximum(np.asarray(linear, np.float32), 1e-8)
    return (value+np.log(-np.expm1(-value))).astype(np.float32)


def fixed_parameter_hashes(trainer):
    names = ('means', 'log_scales', 'quaternions', 'opacity', 'color', 'reflection_logits', 'angular_gain')
    return {name: hashlib.sha256(getattr(trainer, name).numpy().tobytes()).hexdigest()
            for name in names if getattr(trainer, name, None) is not None}


def replace_environment(trainer, radiance):
    values = np.asarray(radiance)
    if not trainer.hdr_angular or values.shape != (trainer.env_height, trainer.env_width, 3):
        raise ValueError('Reference lighting must match the saved HDR environment dimensions.')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Reference linear radiance must be finite and nonnegative.')
    before = fixed_parameter_hashes(trainer)
    trainer.environment.assign(inverse_softplus(values).reshape(-1, 3))
    assert fixed_parameter_hashes(trainer) == before


def save_image(path, value):
    Image.fromarray(np.uint8(np.clip(value, 0, 1)*255)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference-comparison', type=Path, required=True,
                        help='Existing compare_environment.py output containing aligned-linear.npz and comparison.json.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--views', nargs='+', type=int, default=[0, 13, 73, 88, 51])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.run.resolve()
    record = json.loads((run/'run.json').read_text())
    assert record['status'] == 'completed'
    config = record['config']
    if config['surfel']['shading_model'] != 'hdr_angular':
        raise ValueError('This test requires an HDR/angular checkpoint.')
    sources = verify_sources(run)
    data = Path(config['paths']['data'])
    capture = json.loads((data/'capture.json').read_text())
    reference_metadata = json.loads((args.reference_comparison/'comparison.json').read_text())
    assert capture['specification']['scene_sha256'] == reference_metadata['reference']['scene_sha256']
    with np.load(args.reference_comparison/'aligned-linear.npz') as values:
        reference = values['reference'].astype(np.float32)
    if not np.isfinite(reference).all():
        raise ValueError('Reference map contains nonfinite radiance.')
    reference_preparation = {
        'source_min': float(reference.min()), 'source_max': float(reference.max()),
        'negative_channels': int((reference < 0).sum()), 'total_channels': int(reference.size),
        'policy': 'Clamp negative EXR channels to zero; softplus encoding floors zero to 1e-8. No gain fitting.'}
    reference = np.maximum(reference, 0)
    artifacts = [fingerprint(run/name) for name in ('model.ply', 'model.surfel.npz')]
    manifest_path = data/'transforms_val.json'
    manifest = json.loads(manifest_path.read_text())
    width, height = config['model']['width'], config['model']['height']
    targets, cameras, focal, _, frames = base.load_views(data, width, height, np.ones(3, np.float32), manifest_name='transforms_val.json')
    assert all(0 <= view < len(frames) for view in args.views)
    masks = []
    for frame in manifest['frames']:
        with Image.open(base.resolve_frame_image(data, frame['file_path'])) as image:
            masks.append(np.asarray(image.convert('RGBA').getchannel('A')) >= 243)
    protocol = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'run': str(run), 'evaluation_only': True, 'manifest': fingerprint(manifest_path),
                'views': args.views, 'all_validation_views': len(frames), 'resolution': [width, height],
                'checkpoint_files': artifacts, 'renderer_sources': sources,
                'reference_comparison': fingerprint(args.reference_comparison/'comparison.json'),
                'reference_linear_map': fingerprint(args.reference_comparison/'aligned-linear.npz'),
                'reference_preparation': reference_preparation,
                'intervention': 'only environment logits replaced in memory; original map resolution, strength and exposure; no fitted brightness gain',
                'fixed': 'means, tangent scales, quaternions/normals, opacity, base colors, reflection logits, angular response',
                'script': fingerprint(Path(__file__))}
    protocol_path = args.output/'protocol.json'
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    import torch
    torch.set_num_threads(2)
    lpips = metrics_module.LPIPSMetric('alex')
    wp.init()
    scene = GaussianSet.from_ply(run/'model.ply')
    trainer = build_trainer(argparse.Namespace(ply=run/'model.ply', background='white', tile_pair_capacity=None, sh_degree=None),
                            scene, targets, cameras, focal, 'cuda:0')
    parameters = fixed_parameter_hashes(trainer)
    initial_environment = trainer.environment.numpy().copy()
    raster_hashes = {}
    rows, summary = [], []
    image_paths = {view: [] for view in args.views}
    boxes = {}
    for view in args.views:
        path = args.output/f'reference-{view}.png'
        save_image(path, targets[view]); image_paths[view].append(path)
        with Image.open(base.resolve_frame_image(data, manifest['frames'][view]['file_path'])) as image:
            x0, y0, x1, y1 = image.getchannel('A').getbbox()
        cx, cy = (x0+x1)//2, (y0+y1)//2
        half = round(min(x1-x0, y1-y0)*.24)
        boxes[view] = (cx-half, cy-half, cx+half, cy+half)
    for label in ('learned-light', 'reference-light'):
        if label == 'reference-light':
            replace_environment(trainer, reference)
        model_rows = []
        for view in range(len(frames)):
            image, _ = trainer.render_single_view_image(view)
            image = np.clip(image, 0, 1)
            buffers = trainer.buffers.numpy().reshape(height, width, 9)
            digest = hashlib.sha256(buffers.tobytes()).hexdigest()
            if label == 'learned-light': raster_hashes[view] = digest
            else: assert digest == raster_hashes[view], f'Rasterized geometry/material buffers changed at view {view}'
            error = (image-targets[view])**2
            mask = masks[view]
            row = {'lighting': label, 'view': view, 'file': frames[view],
                   'psnr_db': float(-10*np.log10(max(float(error.mean()), 1e-12))),
                   'ssim': ssim_rgb(image, targets[view]), 'lpips_alex': lpips(image, targets[view]),
                   'foreground_psnr_db': float(-10*np.log10(max(float(error[mask].mean()), 1e-12))),
                   'foreground_coverage_fraction': float((buffers[:, :, 7][mask] >= .5).mean())}
            rows.append(row); model_rows.append(row)
            if view in args.views:
                path = args.output/f'{label}-{view}.png'
                save_image(path, image); image_paths[view].append(path)
            if (view+1)%20 == 0: print(label, view+1, '/', len(frames), flush=True)
        result = {'lighting': label, **{key: float(np.mean([r[key] for r in model_rows]))
                  for key in ('psnr_db', 'ssim', 'lpips_alex', 'foreground_psnr_db', 'foreground_coverage_fraction')}}
        summary.append(result)
        assert fixed_parameter_hashes(trainer) == parameters
        print(result, flush=True)
    trainer.environment.assign(initial_environment)
    assert np.array_equal(trainer.environment.numpy(), initial_environment)
    for name, values in [('metrics.csv', rows), ('summary.csv', summary)]:
        with (args.output/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    sheet(args.output/'rgb-comparison.png', [(f'justChromeBall | validation r_{view}', image_paths[view]) for view in args.views],
          ['Blender reference', 'Learned lighting', 'Reference lighting; fixed model'])
    sheet(args.output/'reflection-crops.png',
          [(f'justChromeBall | validation r_{view} | identical crop {boxes[view]}',
            [Image.open(path).crop(boxes[view]) for path in image_paths[view]]) for view in args.views],
          ['Blender reference', 'Learned lighting', 'Reference lighting; fixed model'], panel=300)
    for item in artifacts: assert fingerprint(item['path'])['sha256'] == item['sha256']
    protocol.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat(),
                    fixed_parameter_hashes=parameters, all_raster_buffers_identical=True,
                    original_environment_restored=True, checkpoint_files_unchanged=True,
                    crop_boxes_xyxy={str(view): box for view, box in boxes.items()}, summary=summary)
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    print('Completed:', args.output, flush=True)


if __name__ == '__main__':
    main()
