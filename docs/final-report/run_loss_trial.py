"""Run one matched, validation-only refinement-loss trial and archive evidence.

No test manifest or hyperparameter sweep is launched. Existing renderer/trainer
sources and original checkpoints are preserved.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from statistics import mean

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT / 'shared')]
from run_experiment import build_parser, run_experiment
from cp4282_gs.surfels.config import SurfelTrainingConfig

BASE = ROOT / 'outputs/Nov02-2023-justChromeBall-staged-hdr-refine-5k'
CONFIG = ROOT / 'config/experiments/justChromeBall_l1dssim_100k.yaml'
SESSION = ROOT / 'outputs/Nov07-refinement-loss-session'
DEST = ROOT / 'docs/reports/assets/2026-11-07-refinement-loss'


def fingerprint(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + '.'))
        else:
            result[name] = item
    return result


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    SESSION.mkdir(parents=True, exist_ok=False)
    state_path = SESSION / 'state.json'
    state = {'status': 'preflight', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'question': 'One 5k L1/D-SSIM continuation from the same frozen HDR parent versus MSE.',
             'baseline': str(BASE), 'test_evaluation': False}
    def save():
        state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        original = json.loads((BASE / 'run.json').read_text(encoding='utf-8'))
        values = original['config']
        values['surfel']['photometric_loss'] = 'l1_dssim'
        values['paths']['output'] = str(ROOT / 'outputs/justChromeBall-l1dssim-100k/model.png')
        CONFIG.write_text('# Same frozen 100k parent and 5k refinement; explicit L1/D-SSIM override from step 1.\n'
                          + yaml.safe_dump(values, sort_keys=False), encoding='utf-8')
        resolved = SurfelTrainingConfig.load(CONFIG).printable()
        baseline = json.loads((BASE / 'run.json').read_text(encoding='utf-8'))['config']
        flat = [flatten(c) for c in (baseline, resolved)]
        changes = {key: [f.get(key) for f in flat] for key in set(flat[0]) | set(flat[1])
                   if flat[0].get(key) != flat[1].get(key)}
        assert set(changes) == {'paths.output', 'surfel.photometric_loss'}, changes
        assert resolved['loss']['dssim_weight'] == .2 and resolved['loss']['dssim_window'] == 11
        saved_sources = json.loads((BASE / 'source-files.json').read_text(encoding='utf-8'))
        checked_sources = []
        for item in saved_sources:
            path = item['path']
            if path.startswith(('src/', 'shared/')) or path in ('3dgs_trainer.py', 'gaussian_first_tile_workspace_gpu.py',
                                                              'scripts/train_surfels.py', 'scripts/evaluate_surfels.py', 'image_metrics.py'):
                current = fingerprint(ROOT / path)
                assert current['sha256'] == item['sha256'], path
                checked_sources.append(current)
        input_records = json.loads((BASE / 'input-files.json').read_text(encoding='utf-8'))
        for item in input_records:
            assert fingerprint(Path(item['path']))['sha256'] == item['sha256'], item['path']
        protected = [fingerprint(BASE / name) for name in ('model.ply', 'model.surfel.npz')]
        parent = Path(resolved['paths']['init_ply'])
        protected += [fingerprint(parent), fingerprint(parent.with_suffix('.surfel.npz'))]
        state.update(status='training', changes=changes, checked_sources=checked_sources,
                     input_fingerprints_matched=len(input_records), protected_checkpoints=protected,
                     config=fingerprint(CONFIG))
        save()
        args = build_parser().parse_args([str(CONFIG), '--pipeline', 'surfel', '--name', 'l1dssim-100k-refine-5k',
              '--eval-manifest', 'transforms_val.json', '--metrics', 'psnr', 'ssim', 'lpips',
              '--note', 'Matched loss trial: same frozen HDR parent; replace MSE refinement with 0.8 L1 + 0.2 D-SSIM; validation only.'])
        trial = run_experiment(args)
        state.update(status='paired-validation', trial=str(trial))
        save()
        paired, normals = SESSION / 'paired-validation', SESSION / 'normal-diagnostics'
        commands = [
            [sys.executable, str(ROOT / 'scripts/evaluate_reflection_suite.py'), str(BASE), str(trial),
             '--labels', 'MSE', 'L1-DSSIM', '--output', str(paired), '--manifest', 'transforms_val.json',
             '--views', '0', '13', '73', '88', '51', '--metrics', 'psnr', 'ssim', 'lpips', '--cpu-threads', '2', '--skip-timing'],
            [sys.executable, str(ROOT / 'scripts/diagnose_sphere_footprints.py'), str(BASE), str(trial),
             '--labels', 'MSE', 'L1-DSSIM', '--output', str(normals), '--geometry-reference', str(ROOT / 'outputs/Nov03-sphere-geometry-val'),
             '--views', '0', '13', '73', '88', '51']]
        state['commands'] = commands
        save()
        for name, command in zip(('paired-validation', 'normal-diagnostics'), commands):
            with (SESSION / f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for item in protected:
            assert fingerprint(ROOT / item['path'])['sha256'] == item['sha256']
        assert json.loads((trial / 'run.json').read_text())['training_result']['loss_stage'] == 'l1_dssim'
        summary = read_rows(paired / 'summary.csv')
        metrics = read_rows(paired / 'metrics.csv')
        geometry = read_rows(normals / 'diagnostics.csv')
        assert len(metrics) == 200 and len(geometry) == 10
        geometry_means = {label: {key: mean(float(r[key]) for r in geometry if r['model'] == label)
                                for key in ('mesh_normal_mae_deg', 'normal_coherence', 'foreground_coverage_fraction')}
                          for label in ('MSE', 'L1-DSSIM')}
        grouped = {label: sorted([r for r in metrics if r['model'] == label], key=lambda r: int(r['view']))
                   for label in ('MSE', 'L1-DSSIM')}
        improved = {key: sum((float(b[key]) > float(a[key])) if key != 'lpips_alex' else (float(b[key]) < float(a[key]))
                             for a, b in zip(grouped['MSE'], grouped['L1-DSSIM']))
                    for key in ('psnr_db', 'ssim', 'lpips_alex', 'foreground_psnr_db')}
        DEST.mkdir(parents=True, exist_ok=True)
        sources = []
        for directory, filenames, prefix in [(paired, ['summary.csv', 'metrics.csv', 'protocol.json', 'crop-metrics.csv', 'rgb-comparison.png'], 'validation-'),
                (normals, ['diagnostics.csv', 'protocol.json', 'normal-comparison.png', 'normal-errors.png', 'learned-environments.png'], 'normal-'),
                (trial, ['run.json', 'config.yaml', 'train.log'], 'trial-')]:
            for name in filenames:
                source = directory / name
                sources.append(fingerprint(source))
                shutil.copyfile(source, DEST / (prefix + name))
        # Actual central reflection crops; view order fixed before training.
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
        cell, top, label_height = 360, 42, 36
        views = [0, 13, 73, 88, 51]
        canvas = Image.new('RGB', (cell * 3, top + len(views) * (cell + label_height)), 'white')
        draw = ImageDraw.Draw(canvas)
        for col, title in enumerate(('Blender reference', 'MSE refinement', 'L1 + D-SSIM refinement')):
            draw.text((col * cell + 10, 10), title, fill='#182431', font=font)
        boxes = {}
        for row, view in enumerate(views):
            target = ROOT / f'data/justChromeBall/val/r_{view}.png'
            with Image.open(target) as im:
                x0, y0, x1, y1 = im.getchannel('A').getbbox()
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            half = round(min(x1 - x0, y1 - y0) * .24)
            boxes[str(view)] = box = (cx - half, cy - half, cx + half, cy + half)
            paths = [paired / f'reference-{view}.png'] + [paired / f'model-{i}-{view}.png' for i in (0, 1)]
            for col, path in enumerate(paths):
                sources.append(fingerprint(path))
                with Image.open(path) as im:
                    crop = im.convert('RGB').crop(box).resize((cell, cell), Image.Resampling.LANCZOS)
                y = top + row * (cell + label_height)
                canvas.paste(crop, (col * cell, y))
                draw.text((col * cell + 10, y + cell + 7), f'justChromeBall | val r_{view}', fill='#182431', font=font)
        canvas.save(DEST / 'reflection-crops.png')
        canvas.crop((0, 0, cell * 3, top + cell + label_height)).save(DEST / 'reflection-highlight-view-0.png')
        state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat(),
                     summary=summary, five_view_normal_means=geometry_means,
                     improved_views_out_of_100=improved, protected_checkpoints_unchanged=True,
                     crop_boxes_xyxy=boxes, artifact_sources=sources, controller=fingerprint(Path(__file__)))
        save()
        shutil.copyfile(state_path, DEST / 'provenance.json')
        print(json.dumps({'trial': str(trial), 'summary': summary, 'normal_means': geometry_means,
                          'improved_views': improved}, indent=2), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
