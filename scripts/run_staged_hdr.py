"""Build a training-mask surface, fit frozen HDR appearance, then refine geometry.

Example: python scripts/run_staged_hdr.py data/justChromeBall
Uses the 10k + 5k schedule of Oct30-1733-materials_chrome-staged-hdr-refine-5k.
"""

import argparse
from datetime import datetime
import json
import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from run_experiment import ROOT, SINGAPORE_TIMEZONE, build_parser, run_experiment

sys.path.insert(0, str(ROOT / 'src'))
from cp4282_gs.surfels.config import SurfelTrainingConfig


def copy_initializer(source, destination, multiplier):
    """Change only the two stored tangent log-scales, preserving all other fields."""
    from plyfile import PlyData
    ply = PlyData.read(source, mmap=False)
    count = len(ply['vertex'].data)
    if count != 100000:
        raise ValueError('This matched preset expects exactly 100,000 initial points.')
    if multiplier == 1:
        shutil.copyfile(source, destination)
    else:
        for field in ('scale_0', 'scale_1'):
            ply['vertex'].data[field] += math.log(multiplier)
        ply.write(destination)
    return count


def write_config(template, destination, data, initial, output, appearance='hdr'):
    values = SurfelTrainingConfig.load(template).printable()
    values['paths'] = {'data': str(data), 'init_ply': str(initial), 'output': str(output)}
    if appearance == 'sh3':
        values['model']['sh_degree'] = 3
        values['surfel'].update(mode='rgb', shading_model='ldr', sh_start_iteration=1000, sh_band_interval=1000)
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding='utf-8')
    SurfelTrainingConfig.load(destination)


def train_stage(config, name, metrics):
    args = build_parser().parse_args([
        str(config), '--pipeline', 'surfel', '--name', name,
        '--eval-manifest', 'transforms_val.json', '--metrics', *metrics,
        '--note', 'Staged sphere: training-mask initializer, 10k frozen appearance, 5k geometry refinement; Adam restarts at handoff.',
    ])
    return run_experiment(args)


def execute(work, state, metrics):
    record = work / 'pipeline.json'

    def save():
        record.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')

    try:
        state['status'] = 'initializing'
        save()
        if state['initializer_command']:
            subprocess.run(state['initializer_command'], cwd=ROOT, check=True)
        state['status'] = 'frozen-appearance'
        save()
        appearance = state.get('appearance', 'hdr')
        frozen = train_stage(work / 'frozen.yaml', f'frozen-{appearance}-10k', metrics)
        # Geometry and learned HDR/material parameters are both required.
        required = ['model.ply', 'model.surfel.npz'] + (['model.sh.npz'] if appearance == 'sh3' else [])
        for name in required:
            if not (frozen / name).is_file():
                raise FileNotFoundError(f'Missing warm-start artifact: {frozen / name}')
        values = yaml.safe_load((work / 'refine.yaml').read_text(encoding='utf-8'))
        values['paths']['init_ply'] = str(frozen / 'model.ply')
        (work / 'refine.yaml').write_text(yaml.safe_dump(values, sort_keys=False), encoding='utf-8')
        state.update(status='geometry-refinement', frozen_run=str(frozen))
        save()
        refined = train_stage(work / 'refine.yaml', f'staged-{appearance}-refine-5k', metrics)
        state.update(status='completed', refined_run=str(refined))
        save()
        print(f'Final model: {refined / "model.ply"}', flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='Captured RGBA dataset directory.')
    parser.add_argument('--appearance', choices=('hdr', 'sh3'), default='hdr', help='Appearance model; identical geometry schedules.')
    parser.add_argument('--init-ply', type=Path, help='Reuse a training-derived geometry initializer instead of carving again.')
    parser.add_argument('--scale-multiplier', type=float, default=1., help='Multiply initial tangent scales only; preserves point count and schedules.')
    parser.add_argument('--metrics', nargs='+', choices=('psnr', 'ssim', 'lpips'), default=['psnr', 'ssim'])
    parser.add_argument('--prepare-only', action='store_true', help='Write and validate the plan/configs without initialization or training.')
    args = parser.parse_args()
    if not math.isfinite(args.scale_multiplier) or args.scale_multiplier <= 0:
        parser.error('--scale-multiplier must be finite and positive.')
    if args.init_ply and not args.init_ply.is_file():
        parser.error('--init-ply must name an existing PLY.')
    data = args.data.expanduser().resolve()
    for split in ('train', 'val'):
        manifest = data / f'transforms_{split}.json'
        if not json.loads(manifest.read_text(encoding='utf-8-sig'))['frames']:
            raise ValueError(f'Empty manifest: {manifest}')
    capture = data / 'capture.json'
    if capture.exists() and not json.loads(capture.read_text(encoding='utf-8'))['complete']:
        raise ValueError('Finish dataset capture before training.')
    scene = re.sub(r'[^A-Za-z0-9_-]', '_', data.name)
    stamp = datetime.now(SINGAPORE_TIMEZONE).strftime('%b%d-%H%M%S')
    scale_label = '' if args.scale_multiplier == 1 else f'-scale{args.scale_multiplier:g}'
    base = ROOT / 'outputs' / f'{stamp}-{scene}-staged-{args.appearance}{scale_label}-pipeline'
    work = base
    suffix = 2
    work.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            work.mkdir()
            break
        except FileExistsError:
            work = base.with_name(f'{base.name}-{suffix}')
            suffix += 1
    initial = work / 'initializer' / 'initial.ply'
    write_config(ROOT / 'config/materials_chrome_frozen_hdr.yaml', work / 'frozen.yaml',
                 data, initial, work / 'frozen/model.png', args.appearance)
    write_config(ROOT / 'config/materials_chrome_staged_hdr_refine.yaml', work / 'refine.yaml',
                 data, work / 'frozen/model.ply', work / 'refine/model.png', args.appearance)
    state = {'status': 'prepared', 'data': str(data), 'cumulative_training_iterations': 15000,
             'initializer_command': [sys.executable, '-u', str(ROOT / 'scripts/initialize_from_training_masks.py'),
                 str(data), '--output', str(initial.parent), '--grid-resolution', '512',
                 '--mask-resolution', '800', '--max-points', '100000', '--smoothing', '3', '--seed', '7',
                 '--scale-multiplier', str(args.scale_multiplier)],
             'appearance': args.appearance, 'initial_scale_multiplier': args.scale_multiplier}
    if args.init_ply:
        source = args.init_ply.resolve()
        if source.with_suffix('.surfel.npz').exists() or source.with_suffix('.sh.npz').exists():
            parser.error('--init-ply expects the geometry-only initializer, not a trained checkpoint.')
        initial.parent.mkdir()
        count = copy_initializer(source, initial, args.scale_multiplier)
        state.update(initializer_command=None, initializer_source=str(source),
                     initializer_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     initializer_sha256=hashlib.sha256(initial.read_bytes()).hexdigest())
        (initial.parent / 'initialization.json').write_text(json.dumps({
            'method': 'reuse training-derived geometry; multiply two tangent scales only',
            'source': str(source), 'source_sha256': state['initializer_source_sha256'],
            'scale_multiplier': args.scale_multiplier, 'exported_points': count,
            'source_metadata': str(source.parent / 'initialization.json'),
        }, indent=2) + '\n', encoding='utf-8')
    (work / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print(f'Pipeline directory: {work}', flush=True)
    if args.prepare_only:
        print(f'Prepared: initializer -> 10k frozen {args.appearance} -> 5k refinement. Refine input is wired after the frozen run completes.')
        return
    execute(work, state, args.metrics)


if __name__ == '__main__':
    main()
