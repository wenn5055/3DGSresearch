"""Prepare portable 100k study configs; train only when --run is explicit.

Rebases archived configs into the current checkout, retaining every non-path
setting. Validation only; final test evaluation remains a separate decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT / 'shared')]
from run_staged_hdr import copy_initializer, execute
from cp4282_gs.surfels.config import SurfelTrainingConfig

ARCHIVE = ROOT / 'docs/reports/assets/2026-11-03-sphere-study'
ARMS = [
    ('SH3', 'sh3', 1., 'Nov03-1354-justChromeBall-frozen-sh3-10k', 'Nov03-1357-justChromeBall-staged-sh3-refine-5k'),
    ('HDR-1.0', 'hdr', 1., 'Nov02-2021-justChromeBall-frozen-hdr-10k', 'Nov02-2023-justChromeBall-staged-hdr-refine-5k'),
    ('HDR-0.75', 'hdr', .75, 'Nov03-1400-justChromeBall-frozen-hdr-10k', 'Nov03-1402-justChromeBall-staged-hdr-refine-5k')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory for this reproduction plan.')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/justChromeBall')
    parser.add_argument('--run', action='store_true', help='Actually initialize and train all six stages; otherwise prepare only.')
    args = parser.parse_args()
    data = args.data.resolve()
    for split, count in [('train', 100), ('val', 100), ('test', 200)]:
        frames = json.loads((data / f'transforms_{split}.json').read_text(encoding='utf-8-sig'))['frames']
        assert len(frames) == count, (split, len(frames))
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    initializer = work / 'shared-initializer'
    init_command = [sys.executable, str(ROOT / 'scripts/initialize_from_training_masks.py'), str(data),
                    '--output', str(initializer), '--grid-resolution', '512', '--mask-resolution', '800',
                    '--max-points', '100000', '--smoothing', '3', '--seed', '7', '--scale-multiplier', '1']
    plans = []
    for label, appearance, multiplier, frozen_name, refine_name in ARMS:
        arm = work / label
        (arm / 'initializer').mkdir(parents=True)
        initial = arm / 'initializer/initial.ply'
        sources = []
        for phase, recorded_name in [('frozen', frozen_name), ('refine', refine_name)]:
            source = ARCHIVE / recorded_name / 'config.yaml'
            values = yaml.safe_load(source.read_text(encoding='utf-8'))
            assert values['model']['capacity'] == values['model']['initial_splats'] == 100000
            assert values['training']['iterations'] == (10000 if phase == 'frozen' else 5000)
            values['paths'] = {'data': str(data), 'init_ply': str(initial if phase == 'frozen' else arm / 'frozen/model.ply'),
                               'output': str(arm / phase / 'model.png')}
            destination = arm / f'{phase}.yaml'
            destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding='utf-8')
            SurfelTrainingConfig.load(destination)
            original = yaml.safe_load(source.read_text(encoding='utf-8'))
            assert {k: v for k, v in original.items() if k != 'paths'} == {k: v for k, v in values.items() if k != 'paths'}
            sources.append({'path': source.relative_to(ROOT).as_posix(),
                            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        state = {'status': 'prepared', 'data': str(data), 'appearance': appearance,
                 'initializer_command': None, 'initial_scale_multiplier': multiplier,
                 'cumulative_training_iterations': 15000, 'archived_configs': sources,
                 'preparation_only': not args.run}
        (arm / 'pipeline.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        plans.append((arm, initial, multiplier, state))
    (work / 'plan.json').write_text(json.dumps({'status': 'prepared', 'initializer_command': init_command,
        'arms': [str(arm) for arm, *_ in plans], 'test_evaluation': 'not launched',
        'uses_current_200k_presets': False}, indent=2) + '\n', encoding='utf-8')
    if args.run:
        subprocess.run(init_command, cwd=ROOT, check=True)
        for arm, initial, multiplier, state in plans:
            copy_initializer(initializer / 'initial.ply', initial, multiplier)
            execute(arm, state, ['psnr', 'ssim', 'lpips'])
    print(json.dumps({'directory': str(work), 'stages_prepared': 6, 'training_executed': args.run,
                      'test_evaluation_executed': False}, indent=2))


if __name__ == '__main__':
    main()
