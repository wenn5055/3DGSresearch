"""Build a training-mask surface, fit frozen HDR appearance, then refine geometry.

Example: python scripts/run_staged_hdr.py data/justChromeBall
Uses the 10k + 5k schedule of Oct30-1733-materials_chrome-staged-hdr-refine-5k.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

from run_experiment import ROOT, SINGAPORE_TIMEZONE, build_parser, run_experiment

sys.path.insert(0, str(ROOT / 'src'))
from cp4282_gs.surfels.config import SurfelTrainingConfig


def write_config(template, destination, data, initial, output):
    values = SurfelTrainingConfig.load(template).printable()
    values['paths'] = {'data': str(data), 'init_ply': str(initial), 'output': str(output)}
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding='utf-8')
    SurfelTrainingConfig.load(destination)


def train_stage(config, name, metrics):
    args = build_parser().parse_args([
        str(config), '--pipeline', 'surfel', '--name', name,
        '--eval-manifest', 'transforms_val.json', '--metrics', *metrics,
        '--note', 'Staged HDR: training-mask initializer, 10k frozen appearance, 5k geometry refinement.',
    ])
    return run_experiment(args)


def execute(work, state, metrics):
    record = work / 'pipeline.json'

    def save():
        record.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')

    try:
        state['status'] = 'initializing'
        save()
        subprocess.run(state['initializer_command'], cwd=ROOT, check=True)
        state['status'] = 'frozen-appearance'
        save()
        frozen = train_stage(work / 'frozen.yaml', 'frozen-hdr-10k', metrics)
        # Geometry and learned HDR/material parameters are both required.
        for name in ('model.ply', 'model.surfel.npz'):
            if not (frozen / name).is_file():
                raise FileNotFoundError(f'Missing warm-start artifact: {frozen / name}')
        values = yaml.safe_load((work / 'refine.yaml').read_text(encoding='utf-8'))
        values['paths']['init_ply'] = str(frozen / 'model.ply')
        (work / 'refine.yaml').write_text(yaml.safe_dump(values, sort_keys=False), encoding='utf-8')
        state.update(status='geometry-refinement', frozen_run=str(frozen))
        save()
        refined = train_stage(work / 'refine.yaml', 'staged-hdr-refine-5k', metrics)
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
    parser.add_argument('--metrics', nargs='+', choices=('psnr', 'ssim', 'lpips'), default=['psnr', 'ssim'])
    parser.add_argument('--prepare-only', action='store_true', help='Write and validate the plan/configs without initialization or training.')
    args = parser.parse_args()
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
    base = ROOT / 'outputs' / f'{stamp}-{scene}-staged-hdr-pipeline'
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
                 data, initial, work / 'frozen/model.png')
    write_config(ROOT / 'config/materials_chrome_staged_hdr_refine.yaml', work / 'refine.yaml',
                 data, work / 'frozen/model.ply', work / 'refine/model.png')
    state = {'status': 'prepared', 'data': str(data), 'cumulative_training_iterations': 15000,
             'initializer_command': [sys.executable, '-u', str(ROOT / 'scripts/initialize_from_training_masks.py'),
                 str(data), '--output', str(initial.parent), '--grid-resolution', '512',
                 '--mask-resolution', '800', '--max-points', '100000', '--smoothing', '3', '--seed', '7']}
    (work / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print(f'Pipeline directory: {work}', flush=True)
    if args.prepare_only:
        print('Prepared: initializer -> 10k frozen HDR -> 5k refinement. Refine input is wired to the actual frozen run after it completes.')
        return
    execute(work, state, args.metrics)


if __name__ == '__main__':
    main()
