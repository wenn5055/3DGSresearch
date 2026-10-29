"""Controlled Gaussian RGB backbone probes, with normal experiment provenance.

Policies are scoped to the training subprocess. Production trainers are unchanged.
Fresh children retain the shared Adam clock, as in ordinary 3DGS tensor extension;
this is distinct from creating per-child clocks. Legacy resets all live histories.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import warp as wp
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'shared'), str(ROOT / 'src'), str(ROOT / 'scripts')]
from cp4282_gs.surfels.config import SurfelTrainingConfig
from cp4282_gs.surfels.workflow import base, train
import run_experiment as experiment

POLICIES = ('inherit', 'fresh_children', 'legacy_reset')


@wp.kernel
def clear_finite_rows(values: wp.array(dtype=Any), children: wp.array(dtype=wp.int32)):
    child = children[wp.tid()]
    # Optimizer histories are finite; multiplication preserves scalar/vector dtype.
    values[child] = values[child] * 0.0


@contextmanager
def optimizer_policy(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown optimizer policy.')
    if policy == 'inherit':
        yield
        return

    def initialize_children(bundle, parents, children):
        parent_ids = wp.array(np.asarray(parents, np.int32), dtype=wp.int32, device=bundle.device)
        child_ids = wp.array(np.asarray(children, np.int32), dtype=wp.int32, device=bundle.device)
        if policy == 'legacy_reset':
            bundle.reset()
        else:
            arrays = [array for optimizer in bundle.optimizers for array in (*optimizer.m, *optimizer.v)]
            arrays += [bundle.quaternion_m, bundle.quaternion_v]
            for array in arrays:
                wp.launch(clear_finite_rows, len(children), inputs=[array, child_ids], device=bundle.device)
        return parent_ids, child_ids

    with patch.object(base.SplatOptimizers, 'inherit_children', initialize_children):
        yield


def validate_config(path):
    config = SurfelTrainingConfig.load(path)
    if config['model']['geometry'] != 'gaussian' or config['model']['sh_degree'] != 0:
        raise ValueError('Backbone probes require Gaussian RGB, without SH or surfel state.')
    if config['loss']['mode'] != 'mse' or config['surfel']['mode'] != 'rgb':
        raise ValueError('Use fixed MSE and RGB to isolate backbone learning.')
    return config


def run_probe(args):
    validate_config(args.config)
    parser = experiment.build_parser()
    prepared_args = parser.parse_args([str(args.config), '--pipeline', 'surfel', '--prepare-only',
        '--eval-manifest', 'transforms_val.json', '--metrics', 'psnr', 'ssim', '--name', args.name,
        '--note', 'Controlled Gaussian RGB/MSE backbone audit; optimizer policy ' + args.policy])
    started = perf_counter()
    directory = experiment.run_experiment(prepared_args)
    record = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    record['optimizer_probe'] = {'policy': args.policy,
        'fresh_children_clock': 'existing shared clock retained; only child moments zeroed',
        'scope': 'Gaussian RGB geometry/opacity/colour optimizers only; production code unchanged'}
    record['planned_commands']['train'] = [sys.executable, '-u', str(Path(__file__).resolve()),
        'train', str(directory / 'config.yaml'), '--policy', args.policy]
    record['status'] = 'running'
    record.pop('finished_utc', None)
    experiment.write_record(directory, record)
    try:
        for stage in ('train', 'eval'):
            experiment.run_stage(stage, record['planned_commands'][stage], ROOT, directory, record)
            if stage == 'train':
                experiment.collect_results(directory, record)
        experiment.collect_metrics(directory, record)
        record['status'] = 'completed'
    except BaseException as error:
        record['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        record['error'] = str(error) or type(error).__name__
        raise
    finally:
        record['finished_utc'] = experiment.utc_now()
        record['total_wall_seconds'] = perf_counter() - started
        experiment.write_record(directory, record)
    print(f'Probe completed: {directory}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('run', 'train'):
        subparser = subparsers.add_parser(command)
        subparser.add_argument('config', type=Path)
        subparser.add_argument('--policy', choices=POLICIES, required=True)
        if command == 'run':
            subparser.add_argument('--name', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run_probe(args)
    else:
        config = validate_config(args.config)
        print(f'Optimizer probe policy: {args.policy}', flush=True)
        with optimizer_policy(args.policy):
            train(args.config)
        if config['runtime']['arch'] != 'cpu':
            wp.synchronize_device('cuda:0')
            print(f'Training Warp pool high bytes: {wp.get_mempool_used_mem_high("cuda:0")}', flush=True)


if __name__ == '__main__':
    main()
