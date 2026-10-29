"""Compare frozen Gaussian/surfel runs, foreground scores, crops and render cost.

Camera selection and crop bounds use poses/reference alpha before model rendering.
Scores use float renders, not saved 8-bit PNGs.
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import html
import importlib
import json
import re
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from cp4282_gs.surfels.geometry import base
from gaussian_set import GaussianSet
from image_metrics import ssim_rgb
from evaluate_surfels import build_trainer as build_surfel
from compare_surfel_runs import crop_boxes, sheet

gaussian_evaluator = importlib.import_module('3dgs_evaluator')


def fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checkpoint_paths(run):
    return [path for name in ('model.ply', 'model.surfel.npz', 'model.sh.npz')
            if (path := run / name).exists()]


def renderer_source_differences(run):
    saved = {item['path']: item['sha256'] for item in json.loads((run / 'source-files.json').read_text(encoding='utf-8'))}
    paths = [ROOT / name for name in ('3dgs_trainer.py', '3dgs_evaluator.py',
             'gaussian_first_tile_workspace_gpu.py', 'image_metrics.py', 'scripts/evaluate_surfels.py')]
    paths += [path for folder in ('shared', 'src') for path in (ROOT / folder).rglob('*.py')]
    return [path.relative_to(ROOT).as_posix() for path in paths
            if saved.get(path.relative_to(ROOT).as_posix()) != fingerprint(path)['sha256']]


def psnr(error):
    return float(-10 * np.log10(max(float(error), 1e-12)))


def save_image(path, values):
    Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255)).save(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + '.'))
        else:
            result[name] = item
    return result


def projected_material_boxes(path, materials, cameras, focal, width, height, views):
    inspection = json.loads(path.read_text(encoding='utf-8'))
    boxes = {}
    for material in materials:
        objects = [obj for obj in inspection['objects']
                   if any(item and item['name'] == material for item in obj['materials'])]
        if not objects:
            raise ValueError(f'Material {material!r} not found in inspection.')
        for view in views:
            bounds = np.concatenate([np.asarray(obj['world_bounds'], np.float64) for obj in objects])
            points = np.column_stack((bounds, np.ones(len(bounds)))) @ cameras[view].T
            if np.any(points[:, 2] <= base.NEAR_PLANE):
                continue
            xy = focal * points[:, :2] / points[:, 2:3] + [width * .5, height * .5]
            lo = np.maximum(np.floor(xy.min(axis=0)).astype(int) - 3, 0)
            hi = np.minimum(np.ceil(xy.max(axis=0)).astype(int) + 3, [width, height])
            if np.all(hi > lo):
                boxes[(int(view), material)] = [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]
    return boxes


def evaluate(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory; preserve existing evidence.')
    runs = [path.resolve() for path in args.runs]
    records = [json.loads((path / 'run.json').read_text(encoding='utf-8')) for path in runs]
    if any(record['status'] != 'completed' for record in records):
        raise ValueError('All training/evaluation runs must be completed.')
    labels = args.labels or [path.name for path in runs]
    if len(labels) != len(runs) or len(set(labels)) != len(labels):
        raise ValueError('Provide one unique label per run.')
    configs = [record['config'] for record in records]
    common = lambda config: (config['paths']['data'], config['model']['width'],
                             config['model']['height'], config['model']['background'])
    if any(common(config) != common(configs[0]) for config in configs[1:]):
        raise ValueError('Dataset, resolution and background must match.')
    flattened = [flatten(config) for config in configs]
    differences = {key: [item.get(key) for item in flattened]
                   for key in sorted(set().union(*(item.keys() for item in flattened)))
                   if any(item.get(key) != flattened[0].get(key) for item in flattened[1:])}
    config = configs[0]
    width, height, background_name = config['model']['width'], config['model']['height'], config['model']['background']
    data = Path(config['paths']['data'])
    background = np.ones(3, np.float32) if background_name == 'white' else np.zeros(3, np.float32)
    manifest_path = data / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    targets, cameras, focal, distance, frames = base.load_views(data, width, height, background, manifest_name=args.manifest)
    selected = args.views if args.views is not None else base.select_diverse_frame_ids(
        [frame['transform_matrix'] for frame in manifest['frames']], 4)
    if not selected or len(set(selected)) != len(selected) or any(view < 0 or view >= len(frames) for view in selected):
        raise ValueError('Illustration view IDs must be unique indices in the manifest.')
    masks, silhouettes, inputs, boxes = [], [], [fingerprint(manifest_path)], {}
    args.output.mkdir(parents=True, exist_ok=True)
    for view, frame in enumerate(manifest['frames']):
        image_path = base.resolve_frame_image(data, frame['file_path'])
        inputs.append(fingerprint(image_path))
        with Image.open(image_path) as image:
            alpha = np.asarray(image.convert('RGBA').getchannel('A').resize((width, height), Image.Resampling.LANCZOS), np.float32) / 255
        masks.append(alpha >= .95)
        silhouettes.append(alpha >= .5)
        if view in selected:
            for index, bounds in enumerate(crop_boxes(alpha, limit=12)):
                boxes[(int(view), f'component-{index + 1}')] = bounds
            save_image(args.output / f'reference-{view}.png', targets[view])
    if args.material_inspection:
        boxes.update(projected_material_boxes(args.material_inspection, args.materials, cameras, focal, width, height, selected))
    geometry_metadata = None
    material_ids = {}
    if args.geometry_reference:
        geometry_metadata = json.loads((args.geometry_reference / 'geometry.json').read_text(encoding='utf-8'))
        if (geometry_metadata['status'] != 'completed' or geometry_metadata['resolution'] != [width, height]
                or geometry_metadata['manifest_sha256'] != fingerprint(manifest_path)['sha256']
                or set(geometry_metadata['views']) != set(range(len(frames)))):
            raise ValueError('Geometry reference must match the full manifest and resolution.')
        for material in args.materials:
            material_ids[material] = [item['id'] for item in geometry_metadata['objects'] if material in item['materials']]
            if not material_ids[material]:
                raise ValueError(f'Material {material!r} not present in the geometry reference.')
        for view in selected:
            with np.load(args.geometry_reference / f'r_{view}.npz', allow_pickle=False) as reference:
                mask = reference['object_id'] > 0
                save_image(args.output / f'reference-normals-{view}.png',
                           np.where(mask[:, :, None], reference['normal'] * .5 + .5, 0))
    protocol = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'runs': [str(run) for run in runs], 'labels': labels, 'manifest': args.manifest,
        'resolution': [width, height], 'selected_views': [int(view) for view in selected],
        'crop_boxes_xyxy_half_open': [{'view': view, 'region': region, 'bounds': bounds} for (view, region), bounds in boxes.items()],
        'material_box_definition': 'Projected mesh bounding boxes include background, pedestals and possible occlusion; not a visible-material segmentation.',
        'config_differences': differences, 'inputs': inputs,
        'evaluation_sources': [fingerprint(ROOT / path) for path in
            ('scripts/evaluate_reflection_suite.py', 'scripts/evaluate_surfels.py', 'scripts/compare_surfel_runs.py', '3dgs_trainer.py',
             '3dgs_evaluator.py', 'image_metrics.py')]
            + [fingerprint(path) for directory in ('shared', 'src') for path in (ROOT / directory).rglob('*.py')],
        'training_sources': [fingerprint(run / 'source-files.json') for run in runs],
        'training_vs_evaluation_source_differences': [renderer_source_differences(run) for run in runs],
        'checkpoints': [[fingerprint(path) for path in checkpoint_paths(run)] for run in runs]}
    if args.material_inspection:
        protocol['material_inspection'] = fingerprint(args.material_inspection)
    if geometry_metadata:
        protocol['geometry_reference'] = geometry_metadata
        protocol['geometry_files'] = [fingerprint(args.geometry_reference / f'r_{view}.npz') for view in range(len(frames))]
        protocol['material_metric_definition'] = 'First-hit evaluated-mesh pixels for exact material assignment; require reference image alpha >= 0.95. Mesh normals exclude shader bump. Normal errors require prediction coverage >= 0.5; report valid fraction.'
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    wp.init()
    device = gaussian_evaluator.resolve_device(args.arch)
    protocol['evaluation_runtime'] = {'device': device, 'device_name': wp.get_device(device).name,
                                      'warp_version': wp.__version__, 'python_version': sys.version}
    lpips = None
    if 'lpips' in args.metrics:
        import torch
        torch.set_num_threads(args.cpu_threads)
        lpips = gaussian_evaluator.LPIPSMetric('alex')
    rows, crop_rows, summaries, timings, geometry_rows, normal_labels = [], [], [], [], [], []
    image_paths = {int(view): [args.output / f'reference-{view}.png'] for view in selected}
    normal_paths = {int(view): [] for view in selected}
    for model_index, (run, record, label) in enumerate(zip(runs, records, labels)):
        scene = GaussianSet.from_ply(run / 'model.ply')
        build_args = argparse.Namespace(ply=run / 'model.ply', background=background_name,
                                       tile_pair_capacity=None, sh_degree=None, sh_npz=None)
        is_surfel = (run / 'model.surfel.npz').exists()
        if is_surfel:
            normal_labels.append(label)
        if is_surfel:
            trainer = build_surfel(build_args, scene, targets, cameras, focal, device)
        else:
            trainer, _ = gaussian_evaluator.build_trainer(build_args, scene, targets, cameras, focal, distance, device)
        model_rows = []
        for view in range(len(frames)):
            image, _ = trainer.render_single_view_image(view)
            image = np.clip(image, 0, 1)
            error = (image - targets[view]) ** 2
            row = {'model': label, 'view': view, 'file': frames[view], 'mse': float(error.mean()),
                   'psnr_db': psnr(error.mean()), 'ssim': ssim_rgb(image, targets[view]),
                   'foreground_psnr_db': psnr(error[masks[view]].mean()) if masks[view].any() else None,
                   'lpips_alex': lpips(image, targets[view]) if lpips is not None else None}
            rows.append(row)
            model_rows.append(row)
            if geometry_metadata:
                with np.load(args.geometry_reference / f'r_{view}.npz', allow_pickle=False) as reference:
                    ids = reference['object_id']
                    truth_normal = reference['normal']
                    truth_depth = reference['depth']
                ray_foreground = ids > 0
                alpha_foreground = masks[view]
                intersection = int(np.count_nonzero(ray_foreground & silhouettes[view]))
                union = int(np.count_nonzero(ray_foreground | silhouettes[view]))
                diagnostic = {'model': label, 'view': view, 'mesh_alpha_iou': intersection / max(union, 1)}
                for material, object_values in material_ids.items():
                    mask = np.isin(ids, object_values) & alpha_foreground
                    row[f'{material}_psnr_db'] = psnr(error[mask].mean()) if mask.any() else None
                    diagnostic[f'{material}_pixels'] = int(mask.sum())
                if is_surfel:
                    buffers = trainer.buffers.numpy().reshape(height, width, 9)
                    length = np.linalg.norm(buffers[:, :, 3:6], axis=2)
                    normals = buffers[:, :, 3:6] / np.maximum(length[:, :, None], 1e-8)
                    valid = ray_foreground & alpha_foreground & (buffers[:, :, 7] >= .5) & (length > 1e-6)
                    depth = buffers[:, :, 6] / np.maximum(buffers[:, :, 7], 1e-8)
                    diagnostic['normal_valid_fraction'] = float(valid.sum() / max((ray_foreground & alpha_foreground).sum(), 1))
                    diagnostic['mesh_normal_mae_deg'] = float(np.degrees(np.arccos(np.clip(np.sum(normals[valid] * truth_normal[valid], axis=1), -1, 1))).mean()) if valid.any() else None
                    diagnostic['mesh_depth_mae'] = float(np.abs(depth[valid] - truth_depth[valid]).mean()) if valid.any() else None
                geometry_rows.append(diagnostic)
            if view in selected:
                path = args.output / f'model-{model_index}-{view}.png'
                save_image(path, image)
                image_paths[view].append(path)
                if is_surfel:
                    trainer.save_debug(path)
                    normal_paths[view].append(path.with_name(path.stem + '.normals.png'))
                for (crop_view, region), bounds in boxes.items():
                    if crop_view != view:
                        continue
                    x0, y0, x1, y1 = bounds
                    crop, reference = image[y0:y1, x0:x1], targets[view, y0:y1, x0:x1]
                    crop_rows.append({'model': label, 'view': view, 'region': region,
                        'psnr_db': psnr(np.mean((crop - reference) ** 2)),
                        'ssim': ssim_rgb(crop, reference) if min(crop.shape[:2]) > 5 else None})
            print(f'{label} [{view + 1}/{len(frames)}] PSNR={row["psnr_db"]:.4f}', flush=True)
        # Time only forward device rendering/tiling/MSE, without image download,
        # file I/O, LPIPS or parameter allocation. Host tile preparation is included.
        measurements = []
        if not args.skip_timing:
            for view in selected:
                trainer._render_view(int(view))
            wp.synchronize_device(device)
            for repeat in range(args.timing_repeats):
                started = perf_counter()
                for view in selected:
                    trainer._render_view(int(view))
                wp.synchronize_device(device)
                measurements.append((perf_counter() - started) / len(selected))
        timing = {'model': label, 'median_ms_per_view': 1000 * float(np.median(measurements)) if measurements else None,
                  'fps': 1 / float(np.median(measurements)) if measurements else None,
                  'timing_repeats': len(measurements), 'timing_skipped': args.skip_timing,
                  'scope': 'forward rendering + host tile preparation + MSE; no image download/file I/O/metrics',
                  'active_primitives': len(scene.means),
                  'checkpoint_bytes': sum(path.stat().st_size for path in checkpoint_paths(run)),
                  'warp_pool_current_bytes': int(wp.get_mempool_used_mem_current(device)) if device != 'cpu' else None,
                  'warp_pool_process_high_bytes': int(wp.get_mempool_used_mem_high(device)) if device != 'cpu' else None,
                  'memory_scope': 'evaluation process allocation pool; cumulative high watermark across models, not peak training VRAM',
                  'trainer_seconds': record.get('training_result', {}).get('trainer_seconds')}
        memory_match = re.search(r'^Training Warp pool high bytes: (\d+)$',
                                 (run / 'train.log').read_text(encoding='utf-8'), re.MULTILINE)
        timing['training_warp_pool_high_bytes'] = int(memory_match[1]) if memory_match else None
        timing['training_memory_scope'] = 'Fresh training process Warp allocation peak; excludes driver/context and external allocations. Missing for older runs.'
        timings.append(timing)
        summary = {'model': label}
        for field in ('psnr_db', 'ssim', 'foreground_psnr_db', 'lpips_alex',
                      *(f'{material}_psnr_db' for material in material_ids)):
            values = [row[field] for row in model_rows if row[field] is not None]
            summary[field] = float(np.mean(values)) if values else None
        summaries.append(summary)
        write_csv(args.output / 'metrics.csv', rows)
        write_csv(args.output / 'crop-metrics.csv', crop_rows)
        write_csv(args.output / 'summary.csv', summaries)
        write_csv(args.output / 'geometry-metrics.csv', geometry_rows)
        (args.output / 'performance.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
        del trainer, scene
        gc.collect()
    sheet(args.output / 'rgb-comparison.png',
          [(f'{args.manifest} view {view}', image_paths[view]) for view in selected], ['Reference'] + labels)
    crop_panels = []
    for (view, region), bounds in boxes.items():
        crop_panels.append((f'View {view}, {region}, box {bounds}',
            [Image.open(path).crop(bounds) for path in image_paths[view]]))
    sheet(args.output / 'crop-comparison.png', crop_panels, ['Reference'] + labels, panel=240)
    material_panels = [(f'View {view}, {region}, box {bounds}',
        [Image.open(path).crop(bounds) for path in image_paths[view]])
        for (view, region), bounds in boxes.items() if region in args.materials]
    if material_panels:
        sheet(args.output / 'material-crops.png', material_panels, ['Reference'] + labels, panel=300)
    if normal_labels:
        sheet(args.output / 'normals-comparison.png',
              [(f'View {view}', ([args.output / f'reference-normals-{view}.png'] if geometry_metadata else [])
                + normal_paths[view]) for view in selected], (['Reference mesh'] if geometry_metadata else []) + normal_labels)
    fields = ('model', 'psnr_db', 'foreground_psnr_db', 'ssim', 'lpips_alex',
              *(f'{material}_psnr_db' for material in material_ids))
    table = '<table><tr>' + ''.join(f'<th>{field}</th>' for field in fields) + '</tr>'
    for row in summaries:
        table += '<tr>' + ''.join(f'<td>{html.escape(str(row[field]) if isinstance(row[field], str) or row[field] is None else format(row[field], ".6f"))}</td>' for field in fields) + '</tr>'
    table += '</table>'
    normals_html = '<h2>Normals</h2><img src="normals-comparison.png">' if normal_labels else ''
    material_html = '<h2>Target material crops</h2><img src="material-crops.png">' if material_panels else ''
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Reflection comparison</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccc}}img{{max-width:100%}}a{{color:#075ea8}}</style>
<h1>Reflection comparison</h1><p>{html.escape(args.manifest)}, {width} x {height}; means of per-view scores; LPIPS AlexNet. Foreground: reference alpha at least 0.95.</p>
{table}<p>Camera and component crops selected from poses/alpha before rendering. Material boxes use mesh bounds and can include occlusion. See <a href="protocol.json">protocol and config differences</a>, <a href="metrics.csv">per-view scores</a>, <a href="crop-metrics.csv">crop scores</a>, <a href="performance.json">performance</a>.</p>
{material_html}<h2>RGB</h2><img src="rgb-comparison.png"><h2>Fixed crops</h2><img src="crop-comparison.png">
{normals_html}</html>'''
    (args.output / 'index.html').write_text(body, encoding='utf-8')
    protocol['status'] = 'completed'
    protocol['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('Summary:', json.dumps(summaries), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', choices=('transforms_val.json', 'transforms_test.json'), default='transforms_val.json')
    parser.add_argument('--views', type=int, nargs='+', help='Preselected illustration cameras; default uses pose diversity.')
    parser.add_argument('--metrics', nargs='+', choices=('psnr', 'ssim', 'lpips'), default=['psnr', 'ssim', 'lpips'])
    parser.add_argument('--arch', choices=('auto', 'cpu', 'gpu'), default='gpu')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--timing-repeats', type=int, default=10)
    parser.add_argument('--skip-timing', action='store_true', help='Skip speed measurement, e.g. while other GPU work runs.')
    parser.add_argument('--material-inspection', type=Path)
    parser.add_argument('--geometry-reference', type=Path)
    parser.add_argument('--materials', nargs='+', default=['Chrome'])
    args = parser.parse_args()
    if args.timing_repeats < 1 or args.cpu_threads < 1:
        parser.error('Timing repeats and CPU threads must be positive.')
    evaluate(args)


if __name__ == '__main__':
    main()
