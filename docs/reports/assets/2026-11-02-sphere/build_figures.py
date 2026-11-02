"""Rebuild the sphere closeout figures from saved captures and evaluation PNGs.

Run from the repository root with .venv/Scripts/python.exe and this file's path.
No model training, image synthesis, or new evaluation is performed.
"""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parents[3]
DATA = ROOT / 'data/justChromeBall'
PIPELINE = ROOT / 'outputs/Nov02-202000-justChromeBall-staged-hdr-pipeline'
RUNS = [ROOT / 'outputs/Nov02-2021-justChromeBall-frozen-hdr-10k',
        ROOT / 'outputs/Nov02-2023-justChromeBall-staged-hdr-refine-5k']
VIEWS = [0, 13, 73, 88]  # Reuse prior reports' cameras, independently of new scores.
SOURCES = {}


def fingerprint(path):
    value = {'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    SOURCES[value['path']] = value
    return value


def load_json(path):
    fingerprint(path)
    return json.loads(path.read_text(encoding='utf-8'))


def font(size):
    path = Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def picture(path):
    fingerprint(path)
    with Image.open(path) as source:
        rgba = source.convert('RGBA')
        white = Image.new('RGBA', rgba.size, 'white')
        return Image.alpha_composite(white, rgba).convert('RGB')


def board(rows, columns, size, filename):
    cell, label = size, 42
    canvas = Image.new('RGB', (len(columns) * cell, 46 + len(rows) * (cell + label)), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(columns):
        draw.text((column * cell + 10, 12), title, fill='#182431', font=font(20))
    for row, cells in enumerate(rows):
        for column, (im, text) in enumerate(cells):
            x, y = column * cell, 46 + row * (cell + label)
            canvas.paste(im.resize((cell, cell), Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 10, y + cell + 9), text, fill='#182431', font=font(18))
    canvas.save(ASSETS / filename)


def main():
    capture = load_json(DATA / 'capture.json')
    assert capture['complete'] and len(capture['images']) == 400
    assert fingerprint(Path(capture['specification']['scene']))['sha256'] == capture['specification']['scene_sha256']
    for name, digest in capture['images'].items():
        path = DATA / f'{name}.png'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    records = [load_json(run / 'run.json') for run in RUNS]
    assert all(r['status'] == 'completed' and r['evaluation']['manifest'] == 'transforms_val.json' for r in records)
    assert [r['training_result']['completed_iterations'] for r in records] == [10000, 5000]
    for key, path in [('initialization', RUNS[0] / 'model.ply'), ('initial_materials', RUNS[0] / 'model.surfel.npz')]:
        assert records[1]['inputs'][key]['sha256'] == fingerprint(path)['sha256']
    split_sets, split_counts = {}, {}
    for split in ('train', 'val', 'test'):
        frames = load_json(DATA / f'transforms_{split}.json')['frames']
        split_counts[split] = len(frames)
        split_sets[split] = {tuple(np.round(f['transform_matrix'], 7).ravel()) for f in frames}
    overlaps = {f'{a}_{b}': len(split_sets[a] & split_sets[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    assert not any(overlaps.values())
    rows = []
    for run in RUNS:
        fingerprint(run / 'test-metrics.csv')
        with (run / 'test-metrics.csv').open() as handle:
            rows.append(list(csv.DictReader(handle)))
    assert len(rows[0]) == len(rows[1]) == 100
    expected = [f'./val/r_{i}' for i in range(100)]
    assert all([r['file'] for r in arm] == expected for arm in rows)
    metrics = {key: [np.array([float(r[key]) for r in arm]) for arm in rows] for key in ('psnr_db', 'ssim')}
    delta = metrics['psnr_db'][1] - metrics['psnr_db'][0]
    worst = int(np.argmin(delta))
    refs = {view: picture(DATA / f'val/r_{view}.png') for view in VIEWS + [worst]}
    board([[(refs[v], f'justChromeBall | val r_{v}') for v in VIEWS]],
          ['Blender reference'] * 4, 360, 'dataset-views.png')
    comparisons, crops, bounds = [], [], {}
    for view in VIEWS + [worst]:
        images = [refs[view]] + [picture(run / f'test-renders/r_{view}.png') for run in RUNS]
        labels = [f'val r_{view} | reference'] + [f'val r_{view} | {metrics["psnr_db"][arm][view]:.2f} dB' for arm in range(2)]
        if view in VIEWS:
            comparisons.append(list(zip(images, labels)))
        with Image.open(DATA / f'val/r_{view}.png') as rgba:
            x0, y0, x1, y1 = rgba.getchannel('A').getbbox()
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = round(min(x1 - x0, y1 - y0) * .24)
        box = (cx - half, cy - half, cx + half, cy + half)
        bounds[str(view)] = list(box)
        crops.append([(im.crop(box), text) for im, text in zip(images, labels)])
    columns = ['Blender reference', 'Stage 1 | frozen HDR, 10k', 'Stage 2 | refined HDR, 15k total']
    board(comparisons, columns, 420, 'stage-comparison.png')
    board(crops[:4], columns, 360, 'reflection-crops.png')
    board([crops[-1]], columns, 360, 'regression-view-51.png')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    for ax, (key, label) in zip(axes, [('psnr_db', 'PSNR (dB)'), ('ssim', 'SSIM')]):
        x, y = metrics[key]
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], '--', color='#7b8794', linewidth=1)
        ax.scatter(x, y, s=18, alpha=.8, color='#087e8b')
        ax.set(xlabel=f'Stage 1: {label}', ylabel=f'Stage 2: {label}',
               title=f'justChromeBall | all 100 validation views\n{int((y > x).sum())}/100 improve')
        ax.grid(alpha=.15)
    fig.savefig(ASSETS / 'validation-plot.png', dpi=180)
    plt.close(fig)
    for stage, run in zip(('frozen', 'refine'), RUNS):
        for source, suffix in [('config.yaml', 'yaml'), ('run.json', 'run.json'), ('test-metrics.csv', 'metrics.csv')]:
            shutil.copyfile(run / source, ASSETS / f'{stage}.{suffix}')
    for source, destination in [(DATA / 'capture.json', 'capture.json'),
                                (PIPELINE / 'pipeline.json', 'pipeline.json'),
                                (PIPELINE / 'material-inspection.json', 'material-inspection.json'),
                                (PIPELINE / 'initializer/initialization.json', 'initialization.json')]:
        fingerprint(source)
        shutil.copyfile(source, ASSETS / destination)
    fingerprint(Path(__file__).resolve())
    summary = {'scene': 'justChromeBall', 'split_counts': split_counts, 'camera_pose_overlaps_rounded_7dp': overlaps,
               'capture_hashes_verified': 400, 'source_blend_hash_verified': True,
               'warm_start_geometry_and_material_hashes_verified': True,
               'metrics': {key: {'stage_1': float(x.mean()), 'stage_2': float(y.mean()),
                                 'delta': float((y-x).mean()), 'improved_views': int((y > x).sum())}
                           for key, (x, y) in metrics.items()},
               'psnr_regression_views': [int(i) for i in np.flatnonzero(delta < 0)],
               'worst_psnr_change': {'view': worst, 'delta_db': float(delta[worst])},
               'metrics_source': 'existing float-render CSVs; displayed images are saved 8-bit PNGs',
               'illustration_views': VIEWS, 'illustration_selection': 'reused earlier reports camera indices, independent of sphere metric gains',
               'failure_selection': 'largest PSNR regression selected after reading all validation scores',
               'crop_xyxy': bounds, 'crop_policy': 'central square; half-size=0.24*minimum reference-alpha bbox side; identical between variants',
               'display': 'reference alpha composited onto white; full-view panels downsampled; crop panels enlarged uniformly; no retouching',
               'sources': list(SOURCES.values())}
    (ASSETS / 'figure-provenance.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
