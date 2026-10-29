"""Finalize two frozen surfel runs on test views, with matched diagnostic figures.

No training or reference geometry. Cameras are selected by pose and object crops
by connected components of the reference alpha, before either model is rendered.
Metrics use float predictions; display PNGs are quantized separately.
"""

import argparse
from collections import deque
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src')]
from gaussian_set import GaussianSet
from image_metrics import ssim_rgb
from cp4282_gs.surfels.geometry import base
from evaluate_surfels import build_trainer


def fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def psnr(mse):
    return float(-10 * np.log10(max(float(mse), 1e-12)))


def crop_boxes(alpha, limit=3, margin=6):
    """Largest alpha components, tie-broken spatially; half-open pixel bounds."""
    mask = alpha >= .5
    seen = np.zeros_like(mask)
    height, width = mask.shape
    components = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        area, x0, x1, y0, y1 = 0, int(x), int(x), int(y), int(y)
        while queue:
            cy, cx = queue.popleft()
            area += 1
            x0, x1, y0, y1 = min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy)
            for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if area >= 64:
            components.append((area, x0, y0, x1 + 1, y1 + 1))
    components.sort(key=lambda c: (-c[0], c[1], c[2]))
    return [[max(0, x0-margin), max(0, y0-margin), min(width, x1+margin), min(height, y1+margin)]
            for _, x0, y0, x1, y1 in components[:limit]]


def matched_config(first, second):
    values = []
    for config in (first, second):
        value = deepcopy(config)
        value['paths'].pop('output', None)
        value['geometry_loss'].pop('normal_depth_weight')
        values.append(value)
    if values[0] != values[1]:
        raise ValueError('Runs differ in more than normal-depth weight/output destination.')


def verify_sources(run):
    records = {r['path']: r for r in json.loads((run / 'source-files.json').read_text())}
    paths = [ROOT / '3dgs_trainer.py', ROOT / '3dgs_evaluator.py', ROOT / 'gaussian_first_tile_workspace_gpu.py',
             ROOT / 'image_metrics.py', ROOT / 'scripts/evaluate_surfels.py']
    paths += list((ROOT / 'shared').glob('*.py')) + list((ROOT / 'src').rglob('*.py'))
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if relative not in records or fingerprint(path)['sha256'] != records[relative]['sha256']:
            raise ValueError(f'Evaluation dependency differs from training snapshot: {relative}')
    return [fingerprint(p) for p in paths]


def save_png(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(value, 0, 1) * 255)).save(path)


def sheet(path, rows, headings, panel=400):
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    columns, top, label, gap = len(headings), 38, 30, 12
    canvas = Image.new('RGB', (gap + columns*(panel+gap), top + len(rows)*(panel+label+gap)), '#ededed')
    draw = ImageDraw.Draw(canvas)
    for i, heading in enumerate(headings):
        draw.text((gap+i*(panel+gap), 9), heading, fill='#111111', font=font)
    for r, (title, images) in enumerate(rows):
        y = top + r*(panel+label+gap)
        draw.text((gap, y), title, fill='#111111', font=font)
        for c, source in enumerate(images):
            tile = Image.open(source).convert('RGB') if isinstance(source, Path) else source.convert('RGB')
            scale = min(panel/tile.width, panel/tile.height)
            tile = tile.resize((max(1, round(tile.width*scale)), max(1, round(tile.height*scale))), Image.Resampling.LANCZOS)
            x = gap + c*(panel+gap) + (panel-tile.width)//2
            canvas.paste(tile, (x, y+label+(panel-tile.height)//2))
    canvas.save(path)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory to preserve previous evaluation artifacts.')
    runs = [args.control.resolve(), args.regularized.resolve()]
    records = [json.loads((p / 'run.json').read_text()) for p in runs]
    configs = [r['config'] for r in records]
    if any(r['status'] != 'completed' for r in records):
        raise ValueError('Both runs must be completed.')
    matched_config(*configs)
    sources = [verify_sources(p) for p in runs]
    config = configs[0]
    width, height = config['model']['width'], config['model']['height']
    data = Path(config['paths']['data'])
    manifest = data / 'transforms_test.json'
    test = json.loads(manifest.read_text())
    selected = base.select_diverse_frame_ids([f['transform_matrix'] for f in test['frames']], 4)
    background = np.ones(3, np.float32) if config['model']['background'] == 'white' else np.zeros(3, np.float32)
    targets, cameras, focal, _, frames = base.load_views(data, width, height, background, manifest_name=manifest.name)
    alphas, inputs, boxes = [], [fingerprint(manifest)], {}
    for view, frame in enumerate(test['frames']):
        path = base.resolve_frame_image(data, frame['file_path'])
        inputs.append(fingerprint(path))
        with Image.open(path) as image:
            alpha = np.asarray(image.convert('RGBA').getchannel('A').resize((width, height), Image.Resampling.LANCZOS), np.float32)/255
        alphas.append(alpha)
        if view in selected:
            boxes[view] = crop_boxes(alpha)
    # Selection is fixed from poses/reference alpha before viewing model output.
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'split': manifest.name, 'resolution': [width, height], 'selected_views': selected,
                'crop_bounds_xyxy_half_open': boxes, 'runs': [str(p) for p in runs],
                'normal_depth_weights': [c['geometry_loss']['normal_depth_weight'] for c in configs],
                'configs': configs, 'sources': sources, 'inputs': inputs,
                'checkpoints': [[fingerprint(run/'model.ply'), fingerprint(run/'model.surfel.npz')] for run in runs]}
    (output/'comparison.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    wp.config.kernel_cache_dir = str(ROOT/'outputs/surfel-test-warp-cache')
    wp.config.use_precompiled_headers = False
    wp.init()
    trainers = []
    for run in runs:
        options = argparse.Namespace(ply=run/'model.ply', background=config['model']['background'], tile_pair_capacity=None)
        trainers.append(build_trainer(options, GaussianSet.from_ply(options.ply), targets, cameras, focal, args.device))
    rows, crop_rows, diagnostics = [], [], {}
    for view, frame in enumerate(frames):
        stem = Path(frame).stem
        save_png(output/'reference'/f'{stem}.png', targets[view])
        row = {'view': view, 'file': frame, 'foreground_fraction': float(np.mean(alphas[view] >= .95))}
        for index, trainer in enumerate(trainers):
            tag = 'control' if index == 0 else 'regularized'
            image, mse = trainer.render_single_view_image(view)
            foreground, background_mask = alphas[view] >= .95, alphas[view] == 0
            buffers = trainer.buffers.numpy().reshape(height, width, 9)
            row[f'{tag}_mse'] = mse
            row[f'{tag}_psnr_db'] = psnr(mse)
            row[f'{tag}_ssim'] = ssim_rgb(image, targets[view])
            row[f'{tag}_foreground_mse'] = float(np.mean((image[foreground]-targets[view][foreground])**2)) if foreground.any() else None
            row[f'{tag}_foreground_psnr_db'] = psnr(row[f'{tag}_foreground_mse']) if foreground.any() else None
            row[f'{tag}_background_coverage'] = float(buffers[:, :, 7][background_mask].mean()) if background_mask.any() else None
            save_png(output/tag/f'{stem}.png', image)
            if view in selected:
                diagnostics[(view, tag)] = buffers.copy()
                np.savez_compressed(output/tag/f'{stem}.buffers.npz', buffers=buffers)
                for number, (x0, y0, x1, y1) in enumerate(boxes[view]):
                    crop = image[y0:y1, x0:x1]
                    reference = targets[view, y0:y1, x0:x1]
                    crop_rows.append({'view': view, 'crop': number, 'model': tag,
                                      'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                                      'psnr_db': psnr(np.mean((crop-reference)**2)), 'ssim': ssim_rgb(crop, reference)})
                    save_png(output/'crops'/f'{stem}-{number}-{tag}.png', crop)
                    if index == 0:
                        save_png(output/'crops'/f'{stem}-{number}-reference.png', reference)
        rows.append(row)
        if view % 20 == 0:
            print(f'{view+1}/{len(frames)} test views complete', flush=True)
    write_csv(output/'test-metrics.csv', rows)
    write_csv(output/'crop-metrics.csv', crop_rows)
    full_rows, normal_rows, depth_rows, close_rows = [], [], [], []
    for view in selected:
        stem = Path(frames[view]).stem
        full_rows.append((f'Test {stem}', [output/tag/f'{stem}.png' for tag in ('reference', 'control', 'regularized')]))
        values = [diagnostics[(view, tag)] for tag in ('control', 'regularized')]
        depths = [v[:, :, 6]/np.maximum(v[:, :, 7], 1e-8) for v in values]
        valid = [(alphas[view] >= .95) & (v[:, :, 7] >= .5) for v in values]
        samples = np.concatenate([d[m] for d, m in zip(depths, valid)])
        lo, hi = np.percentile(samples, [1, 99]) if len(samples) else (0., 1.)
        for tag, v, depth, mask in zip(('control', 'regularized'), values, depths, valid):
            normal = v[:, :, 3:6]/np.maximum(np.linalg.norm(v[:, :, 3:6], axis=2, keepdims=True), 1e-8)
            save_png(output/tag/f'{stem}.normals.png', np.where(mask[:, :, None], .5*normal+.5, 0))
            gray = np.where(mask, np.clip((depth-lo)/max(float(hi-lo), 1e-8), 0, 1), 0)
            save_png(output/tag/f'{stem}.depth.png', gray)
            save_png(output/tag/f'{stem}.coverage.png', v[:, :, 7])
            save_png(output/tag/f'{stem}.reflection.png', v[:, :, 8])
        normal_rows.append((f'Test {stem}; world-space normals', [output/tag/f'{stem}.normals.png' for tag in ('control', 'regularized')]))
        depth_rows.append((f'Test {stem}; shared depth range [{lo:.3f}, {hi:.3f}]', [output/tag/f'{stem}.depth.png' for tag in ('control', 'regularized')]))
        for number, bounds in enumerate(boxes[view]):
            close_rows.append((f'Test {stem}, alpha component {number}: {bounds}',
                               [output/'crops'/f'{stem}-{number}-{tag}.png' for tag in ('reference', 'control', 'regularized')]))
    headings = ['Reference RGB', 'Normal-depth weight 0', 'Normal-depth weight 0.01']
    sheet(output/'rgb-comparison.png', full_rows, headings)
    sheet(output/'reflection-crops.png', close_rows, headings, panel=220)
    sheet(output/'normal-comparison.png', normal_rows, headings[1:])
    sheet(output/'depth-comparison.png', depth_rows, headings[1:])
    summary = {}
    for tag in ('control', 'regularized'):
        summary[tag] = {metric: float(np.mean([r[f'{tag}_{metric}'] for r in rows if r[f'{tag}_{metric}'] is not None]))
                        for metric in ('psnr_db', 'ssim', 'foreground_psnr_db', 'background_coverage')}
    summary['regularized_minus_control'] = {metric: summary['regularized'][metric]-summary['control'][metric]
                                           for metric in summary['control']}
    summary['regularized_wins'] = {metric: sum(r[f'regularized_{metric}'] > r[f'control_{metric}'] for r in rows)
                                   for metric in ('psnr_db', 'ssim', 'foreground_psnr_db')}
    metadata.update(status='completed', views=len(rows), summary=summary,
                    depth_preview='Shared 1st/99th percentile range per selected view across both models.',
                    diagnostic_mask='Reference alpha >= 0.95 and model coverage >= 0.5.',
                    metric_conventions='Mean per-view float-image PSNR/SSIM; foreground PSNR uses alpha >= 0.95. PSNR capped at 120 dB for zero MSE. Crops scored before display resizing.')
    (output/'comparison.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with zipfile.ZipFile(output/'evaluation-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in [Path(__file__), *[Path(r['path']) for r in sources[0]]]:
            archive.write(source, source.resolve().relative_to(ROOT).as_posix())
    table = '<tr><th>Model</th><th>Test PSNR</th><th>SSIM</th><th>Foreground PSNR</th></tr>'
    for tag in ('control', 'regularized'):
        s = summary[tag]
        table += f"<tr><td>{html.escape(tag)}</td><td>{s['psnr_db']:.4f}</td><td>{s['ssim']:.6f}</td><td>{s['foreground_psnr_db']:.4f}</td></tr>"
    (output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Frozen surfel comparison</title>
<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#202020}img{max-width:100%;height:auto}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}details{margin:24px 0}a{color:#155ab0}</style>
<h1>Frozen surfel comparison</h1><p>200 held-out test views, 400 × 400. Both models trained for 30k steps; normal-depth weights 0 and 0.01. No retraining or reference geometry.</p>
<table>'''+table+'''</table><p>Metrics use float RGB; foreground means reference alpha ≥ 0.95. Four cameras were selected by pose diversity; crops use the largest reference-alpha components. Image scores do not establish normal accuracy.</p>
<p><a href="test-metrics.csv">Per-view metrics</a> · <a href="crop-metrics.csv">Crop metrics and bounds</a> · <a href="comparison.json">Provenance and protocol</a></p>
<h2>Matched RGB views</h2><img src="rgb-comparison.png">
<h2>Object/reflection crops</h2><p>Identical bounds across reference and both models; crops are display-scaled only. These are object regions, not ground-truth specular masks.</p><img src="reflection-crops.png">
<details open><summary>Normal maps</summary><p>World-space normals; reference alpha ≥ 0.95 and model coverage ≥ 0.5.</p><img src="normal-comparison.png"></details>
<details><summary>Depth maps</summary><p>Shared depth range for both models within each camera; units vary only with scene coordinates, not independently chosen display scales.</p><img src="depth-comparison.png"></details>''', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Comparison saved: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('regularized', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    compare(parser.parse_args())
