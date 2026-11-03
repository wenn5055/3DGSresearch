"""Evaluation-only 2x2 lighting/normal intervention using exported Blender mesh normals."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
import warp as wp

from evaluate_reference_lighting import (
    ROOT, GaussianSet, base, build_trainer, fingerprint, fixed_parameter_hashes,
    metrics_module, replace_environment, save_image, sheet, ssim_rgb, verify_sources)
from cp4282_gs.surfels.hdr_shader import deferred_hdr_shader


def substitute_normals(buffers, normals, object_ids):
    """Replace only the normal-sum direction at existing, valid mesh/model pixels."""
    if buffers.shape[:-1] != object_ids.shape or normals.shape != (*object_ids.shape, 3):
        raise ValueError('Reference and raster dimensions must match.')
    if not np.isfinite(normals).all():
        raise ValueError('Nonfinite reference normals.')
    hit = object_ids > 0
    if not np.allclose(np.linalg.norm(normals[hit], axis=-1), 1, atol=1e-4):
        raise ValueError('Reference hit normals must have unit length.')
    length = np.linalg.norm(buffers[..., 3:6], axis=-1)
    valid = hit & (buffers[..., 7] > 1e-6) & (length > 1e-6)
    result = buffers.copy()
    result[..., 3:6][valid] = normals[valid] * length[valid, None]
    assert np.array_equal(result[..., [0, 1, 2, 6, 7, 8]], buffers[..., [0, 1, 2, 6, 7, 8]])
    assert np.array_equal(result[~valid], buffers[~valid])
    return result, valid


def shade(trainer, view):
    wp.launch(deferred_hdr_shader, dim=trainer.width*trainer.height,
              inputs=[trainer.buffers, trainer.environment, trainer.env_width, trainer.env_height,
                      trainer.cameras, view, trainer.width, trainer.height, trainer.focal,
                      int(trainer.specular_active), trainer.background, trainer.angular_gain],
              outputs=[trainer.image], device=trainer.device)
    return np.clip(trainer.image.numpy().reshape(trainer.height, trainer.width, 3), 0, 1)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--geometry-reference', type=Path, required=True)
    parser.add_argument('--reference-comparison', type=Path, required=True)
    parser.add_argument('--lighting-evaluation', type=Path, required=True,
                        help='Completed prior lighting-swap output, for baseline reproduction checks.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--views', type=int, nargs='+', default=[0, 13, 73, 88, 51])
    args = parser.parse_args()
    config = json.loads((args.run/'run.json').read_text())['config']
    if config['surfel']['shading_model'] != 'hdr_angular':
        raise ValueError('HDR/angular checkpoint required.')
    sources = verify_sources(args.run)
    data = Path(config['paths']['data'])
    manifest_path = data/'transforms_val.json'
    geometry = json.loads((args.geometry_reference/'geometry.json').read_text())
    capture = json.loads((data/'capture.json').read_text())
    lighting = json.loads((args.reference_comparison/'comparison.json').read_text())
    width, height = config['model']['width'], config['model']['height']
    assert geometry['status'] == 'completed' and geometry['resolution'] == [width, height]
    assert geometry['manifest_sha256'] == fingerprint(manifest_path)['sha256']
    assert geometry['scene_sha256'] == capture['specification']['scene_sha256'] == lighting['reference']['scene_sha256']
    assert all(view in geometry['views'] for view in args.views)
    with np.load(args.reference_comparison/'aligned-linear.npz') as values:
        reference_light = np.maximum(values['reference'].astype(np.float32), 0)
    previous_protocol = json.loads((args.lighting_evaluation/'protocol.json').read_text())
    assert previous_protocol['status'] == 'completed'
    for item in previous_protocol['checkpoint_files']:
        assert fingerprint(args.run/Path(item['path']).name)['sha256'] == item['sha256']
    with (args.lighting_evaluation/'metrics.csv').open() as handle:
        previous = {(r['lighting'], int(r['view'])): r for r in csv.DictReader(handle)}
    targets, cameras, focal, _, frames = base.load_views(
        data, width, height, np.ones(3, np.float32), manifest_name='transforms_val.json')
    assert abs(focal-geometry['focal']) < 1e-4
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = [fingerprint(args.run/name) for name in ('model.ply', 'model.surfel.npz')]
    protocol = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'evaluation_only': True, 'views': args.views, 'split': 'validation',
                'resolution': [width, height], 'run': str(args.run.resolve()),
                'manifest': fingerprint(manifest_path), 'geometry_reference': geometry,
                'geometry_files': [fingerprint(args.geometry_reference/f'r_{v}.npz') for v in args.views],
                'lighting_reference': fingerprint(args.reference_comparison/'aligned-linear.npz'),
                'previous_lighting_evaluation': fingerprint(args.lighting_evaluation/'protocol.json'),
                'checkpoints': artifacts, 'renderer_sources': sources, 'script': fingerprint(Path(__file__)),
                'helper_script': fingerprint(ROOT/'scripts/evaluate_reference_lighting.py'),
                'intervention': 'Only normal-sum direction substituted after rasterization; magnitude preserved; fallback to original on no mesh hit or no model normal/coverage.',
                'normal_effects': 'The production shader uses the substituted normal for both reflection direction and angular response. Material parameters remain fixed.',
                'lighting_preparation': 'Original aligned 512x256 forest radiance, negatives clamped to zero, inverse-softplus encoding; no gain/rotation fitting.'}
    protocol_path = args.output/'protocol.json'
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    import torch
    from matplotlib import colormaps
    torch.set_num_threads(2)
    lpips = metrics_module.LPIPSMetric('alex')
    wp.init()
    trainer = build_trainer(argparse.Namespace(ply=args.run/'model.ply', background='white',
                            tile_pair_capacity=None, sh_degree=None), GaussianSet.from_ply(args.run/'model.ply'),
                            targets, cameras, focal, 'cuda:0')
    initial_parameters = fixed_parameter_hashes(trainer)
    learned_environment = trainer.environment.numpy().copy()
    cases = [('learned-light', 'learned-normal'), ('reference-light', 'learned-normal'),
             ('learned-light', 'reference-normal'), ('reference-light', 'reference-normal')]
    titles = ['Blender target', 'Learned light + learned normals', 'Ref light + learned normals',
              'Learned light + ref normals', 'Ref light + ref normals']
    rows, geometry_rows, rgb_rows, crop_rows, normal_rows = [], [], [], [], []
    boxes = {}
    max_baseline_difference = 0.
    for view in args.views:
        trainer.environment.assign(learned_environment)
        original_image, _ = trainer.render_single_view_image(view)
        original = trainer.buffers.numpy().reshape(height, width, 9)
        with np.load(args.geometry_reference/f'r_{view}.npz') as reference:
            normals, ids = reference['normal'], reference['object_id']
        substituted, swapped = substitute_normals(original, normals, ids)
        with Image.open(base.resolve_frame_image(data, frames[view])) as im:
            alpha = np.asarray(im.convert('RGBA').getchannel('A'))
            x0, y0, x1, y1 = im.convert('RGBA').getchannel('A').getbbox()
        mask = alpha >= 243
        valid = swapped & mask & (original[..., 7] >= .5)
        length = np.linalg.norm(original[..., 3:6], axis=-1)
        predicted_normal = original[..., 3:6]/np.maximum(length[..., None], 1e-12)
        angles = np.degrees(np.arccos(np.clip((predicted_normal*normals).sum(axis=-1), -1, 1)))
        geometry_rows.append({'view': view, 'normal_mae_deg': float(angles[valid].mean()),
                              'normal_p95_deg': float(np.percentile(angles[valid], 95)),
                              'reference_fg_normal_swap_fraction': float(swapped[mask].mean()),
                              'mesh_alpha_iou': float(((ids>0)&(alpha>=128)).sum()/((ids>0)|(alpha>=128)).sum())})
        cx, cy = (x0+x1)//2, (y0+y1)//2
        half = round(min(x1-x0, y1-y0)*.24)
        boxes[str(view)] = box = (cx-half, cy-half, cx+half, cy+half)
        target_path = args.output/f'target-{view}.png'
        save_image(target_path, targets[view])
        paths = [target_path]
        for illumination, normal_case in cases:
            if illumination == 'learned-light': trainer.environment.assign(learned_environment)
            else: replace_environment(trainer, reference_light)
            buffers = original if normal_case == 'learned-normal' else substituted
            trainer.buffers.assign(buffers.reshape(-1, 9))
            image = shade(trainer, view)
            assert np.array_equal(trainer.buffers.numpy().reshape(height, width, 9), buffers)
            if illumination == 'learned-light' and normal_case == 'learned-normal':
                np.testing.assert_array_equal(image, np.clip(original_image, 0, 1))
            error = (image-targets[view])**2
            row = {'lighting': illumination, 'normals': normal_case, 'view': view,
                   'psnr_db': float(-10*np.log10(max(float(error.mean()), 1e-12))),
                   'ssim': ssim_rgb(image, targets[view]), 'lpips_alex': lpips(image, targets[view]),
                   'foreground_psnr_db': float(-10*np.log10(max(float(error[mask].mean()), 1e-12))),
                   'foreground_coverage_fraction': float((buffers[..., 7][mask] >= .5).mean())}
            if normal_case == 'learned-normal':
                for key in ('psnr_db', 'ssim', 'lpips_alex', 'foreground_psnr_db'):
                    difference = abs(row[key]-float(previous[(illumination, view)][key]))
                    max_baseline_difference = max(max_baseline_difference, difference)
                    assert difference < 1e-6, (view, illumination, key, difference)
            rows.append(row)
            path = args.output/f'{illumination}-{normal_case}-{view}.png'
            save_image(path, image); paths.append(path)
        trainer.buffers.assign(original.reshape(-1, 9))
        assert fixed_parameter_hashes(trainer) == initial_parameters
        title = f'justChromeBall | validation r_{view}'
        rgb_rows.append((title, paths))
        crop_rows.append((title+f' | identical crop {box}', [Image.open(p).crop(box) for p in paths]))
        normal_rows.append((title, [Image.fromarray(np.uint8(np.clip(value, 0, 1)*255)) for value in (
            np.where(valid[..., None], normals*.5+.5, 0),
            np.where(valid[..., None], predicted_normal*.5+.5, 0),
            np.where(valid[..., None], colormaps['magma'](np.clip(angles/30, 0, 1))[..., :3], 0))]))
        print('Completed view', view, flush=True)
    summary = []
    for illumination, normal_case in cases:
        selected = [r for r in rows if (r['lighting'], r['normals']) == (illumination, normal_case)]
        summary.append({'lighting': illumination, 'normals': normal_case,
                        **{key: float(np.mean([r[key] for r in selected])) for key in
                           ('psnr_db', 'ssim', 'lpips_alex', 'foreground_psnr_db', 'foreground_coverage_fraction')}})
    for name, values in [('metrics.csv', rows), ('summary.csv', summary), ('normal-metrics.csv', geometry_rows)]:
        write_csv(args.output/name, values)
    sheet(args.output/'rgb-comparison.png', rgb_rows, titles, panel=300)
    sheet(args.output/'reflection-crops.png', crop_rows, titles, panel=300)
    sheet(args.output/'normal-comparison.png', normal_rows,
          ['Reference mesh normal', 'Learned shading normal', 'Error: 0 dark to 30 deg yellow'], panel=350)
    trainer.environment.assign(learned_environment)
    assert np.array_equal(trainer.environment.numpy(), learned_environment)
    for item in artifacts: assert fingerprint(item['path'])['sha256'] == item['sha256']
    protocol.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat(),
                    fixed_parameter_hashes=initial_parameters, checkpoint_files_unchanged=True,
                    non_normal_raster_fields_unchanged=True, original_environment_restored=True,
                    baseline_metric_max_abs_difference=max_baseline_difference,
                    crop_boxes_xyxy=boxes, summary=summary)
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
