"""Tune deferred normal-depth weights with full-budget, validation-only runs."""

from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import yaml

import run_experiment as runner
from tune_sh_learning_rate import read_metrics

sys.path.insert(0, str(runner.ROOT / 'src'))
from cp4282_gs.surfels.config import SurfelTrainingConfig


def training_sources(root):
    paths = set(root.glob('*.py'))
    for folder in ('shared', 'src', 'scripts'):
        paths.update((root / folder).rglob('*.py'))
    return {path.relative_to(root).as_posix(): runner.fingerprint(path)['sha256']
            for path in sorted(paths)}


def comparable(config):
    config = deepcopy(config)
    config['paths'].pop('output', None)
    config['geometry_loss'].pop('normal_depth_weight', None)
    return config


def read_record(directory):
    return json.loads((Path(directory) / 'run.json').read_text(encoding='utf-8'))


def active_gpu_runs(root):
    return [str(path.parent.resolve()) for path in sorted((root / 'outputs').glob('*/run.json'))
            if (record := read_record(path.parent))['status'] in ('preparing', 'running')
            and record['config']['runtime']['arch'] == 'gpu']


def write_progress(directory, plan):
    plan['updated_utc'] = runner.utc_now()
    temporary = directory / 'sweep.json.tmp'
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(directory / 'sweep.json')
    columns = ['weight', 'status', 'iterations', 'surfel_stage', 'max_normal_pixels',
               'psnr_db', 'ssim', 'eligible', 'reason', 'run_dir', 'metrics_csv']
    with (directory / 'validation-results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(plan['candidates'])
    lines = ['# Deferred normal-loss validation sweep', '',
             f"Status: **{plan['status']}**. Only `geometry_loss.normal_depth_weight` varies.", '',
             'Select the highest mean validation PSNR; validation SSIM breaks exact ties. '
             'Test images/scores are not used. Every candidate starts from scratch.', '',
             f"Seed: {plan['base_config']['runtime']['seed']}; "
             f"steps per candidate: {plan['base_config']['training']['iterations']}; "
             f"mask weight: {plan['base_config']['geometry_loss']['mask_weight']}.", '',
             '| Normal weight | Status | Steps | Stage | Valid normal pixels (max logged) | Validation PSNR | Validation SSIM | Eligible |',
             '| ---: | --- | ---: | --- | ---: | ---: | ---: | --- |']
    for row in plan['candidates']:
        lines.append(f"| {row['weight']:g} | {row['status']} | {row.get('iterations', '-')} | "
                     f"{row.get('surfel_stage', '-')} | {row.get('max_normal_pixels', '-')} | "
                     f"{row.get('psnr_db', '-')} | {row.get('ssim', '-')} | {row.get('eligible', '-')} |")
        if row.get('reason'):
            lines += ['', f"Weight {row['weight']:g}: {row['reason']}"]
    if 'best_weight' in plan:
        lines += ['', f"Selected normal weight: **{plan['best_weight']:g}**.", '',
                  'Frozen settings: [best-config.yaml](best-config.yaml). '
                  'A zero winner means disabling the normal term under this validation protocol.']
    if plan.get('error'):
        lines += ['', f"Error: {plan['error']}"]
    lines += ['', 'Eligibility requires the full budget and deferred shading. Positive-weight '
              'runs must finish the normal ramp and log at least one valid normal pixel '
              'while its weight is active. Image-quality selection does not establish normal '
              'accuracy; inspect the saved normal/depth/coverage maps.', '',
              'Completed candidates are retained on resume; failed/interrupted candidates '
              'restart from scratch in new run folders. One-seed selection is provisional.', '',
              'Exact settings, input/source fingerprints, waiting jobs and logs: [sweep.json](sweep.json).']
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    runner.refresh_output_index(directory.parent.parent)


def prepare(args, root=runner.ROOT):
    config = SurfelTrainingConfig.load(args.config).printable()
    weights = args.weights
    if (len(weights) < 2 or len(set(weights)) != len(weights)
            or any(not math.isfinite(value) or value < 0 for value in weights)
            or not any(value > 0 for value in weights)):
        raise ValueError('Supply at least two unique, finite, nonnegative weights, including a positive weight.')
    if config['model']['geometry'] != 'surfel' or config['surfel']['mode'] != 'deferred':
        raise ValueError('This sweep requires deferred surfels.')
    if config['paths']['init_ply'] is not None or config['convergence']['enabled']:
        raise ValueError('Use init_ply: null and convergence.enabled: false for matched, full-budget fresh runs.')
    if config['reporting']['eval_manifest'] != 'transforms_val.json':
        raise ValueError('Training gates must use transforms_val.json.')
    normal = config['geometry_loss']
    budget = config['training']['iterations']
    if (budget <= normal['normal_start_iteration'] + normal['normal_ramp_iterations']
            or budget <= config['surfel']['specular_start_iteration']):
        raise ValueError('The budget must extend beyond the normal ramp and deferred activation.')
    now = datetime.now(runner.SINGAPORE_TIMEZONE)
    stamp = f"{runner.MONTHS[now.month - 1]}{now.day:02d}-{now:%H%M}"
    scene = re.sub(r'[^A-Za-z0-9_-]', '_', Path(config['paths']['data']).name)
    name = f'{scene}-{stamp}-deferred-normal-sweep'
    outputs = (root / 'outputs').resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    directory, suffix = outputs / name, 2
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            directory = outputs / f'{name}-{suffix}'
            suffix += 1
    # Freeze train/validation inputs; no test manifest or test image is read.
    runner.inspect_inputs(config, 'transforms_val.json', directory)
    (directory / 'configs').mkdir()
    waiting = [str(path.resolve()) for path in args.wait_for]
    if config['runtime']['arch'] == 'gpu':
        waiting += active_gpu_runs(root)
    plan = {'version': 1, 'kind': 'deferred_normal_weight', 'status': 'prepared',
            'created_utc': runner.utc_now(), 'base_config': config,
            'training_sources': training_sources(root),
            'input_files': json.loads((directory / 'input-files.json').read_text(encoding='utf-8')),
            'wait_for': list(dict.fromkeys(waiting)), 'candidates': []}
    for index, weight in enumerate(sorted(weights)):
        label = f'normal-{index}-{weight:g}'.replace('.', 'p').replace('+', '')
        candidate_config = deepcopy(config)
        candidate_config['geometry_loss']['normal_depth_weight'] = weight
        path = directory / 'configs' / f'{label}.yaml'
        path.write_text(yaml.safe_dump(candidate_config, sort_keys=False), encoding='utf-8')
        plan['candidates'].append({'weight': weight, 'label': label, 'config': str(path),
                                   'status': 'pending',
                                   # Existing running jobs imported an older index writer
                                   # that reads this field unconditionally for every sweep.
                                   'rate': weight})
    write_progress(directory, plan)
    return directory


def verify_frozen(plan, root):
    if training_sources(root) != plan['training_sources']:
        raise ValueError('Training/evaluation sources changed; prepare a new matched sweep.')
    for saved in plan['input_files']:
        if runner.fingerprint(Path(saved['path']))['sha256'] != saved['sha256']:
            raise ValueError(f"Training/validation input changed: {saved['path']}")


def wait_for_runs(directory, plan, root, timeout=21600):
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        waiting = list(plan['wait_for'])
        if plan['base_config']['runtime']['arch'] == 'gpu':
            waiting += active_gpu_runs(root)
        running = [path for path in dict.fromkeys(waiting)
                   if read_record(path)['status'] in ('preparing', 'running')]
        if not running:
            return
        if running != previous:
            print('Waiting for GPU jobs: ' + ', '.join(Path(path).name for path in running), flush=True)
            previous = running
            plan['wait_for'] = list(dict.fromkeys(plan['wait_for'] + running))
            plan['status'] = 'waiting'
            write_progress(directory, plan)
        if time.monotonic() >= deadline:
            raise TimeoutError('Jobs still marked active after six hours; inspect run.json and resume later.')
        time.sleep(10)


def finish_candidate(candidate, run_dir, config):
    record = read_record(run_dir)
    if (record['status'] != 'completed' or record.get('pipeline') != 'surfel'
            or record['evaluation']['manifest'] != 'transforms_val.json'):
        raise ValueError('A completed surfel run evaluated on validation is required.')
    if (comparable(record['config']) != comparable(config)
            or record['config']['geometry_loss']['normal_depth_weight'] != candidate['weight']):
        raise ValueError('Candidate settings differ from the frozen sweep.')
    metrics = run_dir / 'test-metrics.csv'  # Historical filename; manifest verified above.
    candidate.update(read_metrics(metrics, config))
    result = record['training_result']
    log = (run_dir / 'train.log').read_text(encoding='utf-8')
    signal = [int(pixels) for weight, pixels in re.findall(
        r'normal_weight=([\d.eE+-]+), normal_pixels=(\d+)', log) if float(weight) > 0]
    maximum = max(signal, default=0)
    candidate.update(status='completed', iterations=result.get('completed_iterations'),
                     surfel_stage=result.get('surfel_stage'), max_normal_pixels=maximum,
                     run_dir=str(run_dir), metrics_csv=str(metrics))
    reasons = []
    if candidate['iterations'] != config['training']['iterations']:
        reasons.append('Full iteration budget was not completed.')
    if candidate['surfel_stage'] != 'deferred':
        reasons.append('Deferred shading was not activated.')
    if candidate['weight'] > 0 and maximum == 0:
        reasons.append('No valid normal pixels were logged while the normal weight was active.')
    candidate['eligible'] = not reasons
    candidate['reason'] = ' '.join(reasons)


def execute(directory, root=runner.ROOT):
    directory = directory.resolve()
    if not directory.is_relative_to((root / 'outputs').resolve()):
        raise ValueError('Sweep directory must be inside outputs/.')
    plan = json.loads((directory / 'sweep.json').read_text(encoding='utf-8'))
    if plan.get('kind') != 'deferred_normal_weight' or plan['version'] != 1:
        raise ValueError('Unsupported deferred normal-loss sweep.')
    if plan['status'] == 'completed':
        print(f"Sweep already completed: {directory / 'summary.md'}", flush=True)
        return
    os.environ.setdefault('WARP_CACHE_PATH', str(root / 'outputs/warp-cache'))
    try:
        plan.pop('error', None)
        verify_frozen(plan, root)
        for candidate in plan['candidates']:
            config = SurfelTrainingConfig.load(Path(candidate['config'])).printable()
            if (comparable(config) != comparable(plan['base_config'])
                    or config['geometry_loss']['normal_depth_weight'] != candidate['weight']):
                raise ValueError('Candidate configuration changed; prepare a new sweep.')
            if candidate['status'] == 'completed':
                finish_candidate(candidate, Path(candidate['run_dir']), config)
                continue
            wait_for_runs(directory, plan, root)
            verify_frozen(plan, root)
            candidate.update(status='running')
            candidate.pop('error', None)
            plan['status'] = 'running'
            attempt = len(candidate.setdefault('attempt_logs', [])) + 1
            log_path = directory / f"{candidate['label']}-attempt{attempt}.log"
            candidate['attempt_logs'].append(str(log_path))
            write_progress(directory, plan)
            print(f"Starting normal weight {candidate['weight']:g}; log: {log_path}", flush=True)
            args = runner.build_parser().parse_args([
                candidate['config'], '--pipeline', 'surfel', '--eval-manifest', 'transforms_val.json',
                '--metrics', 'psnr', 'ssim', '--name', f"deferred-tune-{candidate['label']}",
                '--note', f'Validation-only deferred normal-weight sweep: {directory.name}'])
            with log_path.open('w', encoding='utf-8') as handle, \
                    contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                run_dir = runner.run_experiment(args, root)
            verify_frozen(plan, root)
            finish_candidate(candidate, run_dir, config)
            write_progress(directory, plan)
            print(f"Weight {candidate['weight']:g}: PSNR={candidate['psnr_db']}, "
                  f"SSIM={candidate['ssim']}, eligible={candidate['eligible']}", flush=True)
        eligible = [row for row in plan['candidates'] if row.get('eligible')]
        if not eligible:
            raise ValueError('No eligible candidates. Inspect summary.md and training logs; no winner saved.')
        best = max(eligible, key=lambda row: (float(row['psnr_db']), float(row['ssim'])))
        plan['best_weight'] = best['weight']
        plan['best_run'] = best['run_dir']
        (directory / 'best-config.yaml').write_bytes(Path(best['config']).read_bytes())
        plan['status'] = 'completed'
        write_progress(directory, plan)
        print(f"Selected normal weight {best['weight']:g}: {directory / 'best-config.yaml'}", flush=True)
    except BaseException as error:
        plan['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        plan['error'] = str(error) or type(error).__name__
        for candidate in plan['candidates']:
            if candidate['status'] == 'running':
                candidate.update(status=plan['status'], error=plan['error'])
        write_progress(directory, plan)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, nargs='?')
    parser.add_argument('--weights', type=float, nargs='+', default=[0.0, 0.005, 0.01, 0.02])
    parser.add_argument('--wait-for', type=Path, nargs='+', default=[])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error('Supply a config or --resume SWEEP_DIRECTORY.')
    directory = args.resume.resolve() if args.resume else prepare(args)
    print(f'Sweep directory: {directory}', flush=True)
    if not args.prepare_only:
        execute(directory)


if __name__ == '__main__':
    main()
