"""Evaluate the two saved sphere stages on validation, without changing checkpoints.

Run: python THIS_FILE --output outputs/Nov03-sphere-lpips
"""

import argparse
import csv
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_surfel_runs import fingerprint, verify_sources

RUNS = [('frozen', ROOT / 'outputs/Nov02-2021-justChromeBall-frozen-hdr-10k'),
        ('refine', ROOT / 'outputs/Nov02-2023-justChromeBall-staged-hdr-refine-5k')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'preflight', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'evaluation_only': True, 'manifest': 'transforms_val.json', 'resolution': [800, 800],
             'lpips': {'network': 'alex', 'version': '0.1', 'input': 'float RGB, clipped to [0,1], mapped to [-1,1]',
                       'scope': 'whole image, white background, no resize'},
             'packages': {name: metadata.version(name) for name in ('torch', 'torchvision', 'lpips', 'numpy', 'warp-lang')},
             'script': fingerprint(Path(__file__)), 'stages': {}}
    # GPU arithmetic / equal-depth ordering need not reproduce float CSVs bitwise.
    # Keep tolerances far below the precision used to report model differences.
    tolerances = {'mse': 1e-6, 'psnr_db': 1e-3, 'ssim': 1e-5}
    state['original_metric_absolute_tolerances'] = tolerances
    record = output / 'evaluation.json'

    def save():
        record.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')

    try:
        for name, run in RUNS:
            sources = verify_sources(run)
            inputs = json.loads((run / 'input-files.json').read_text(encoding='utf-8'))
            for entry in inputs:
                assert fingerprint(entry['path'])['sha256'] == entry['sha256'], entry['path']
            artifacts = [fingerprint(run / file) for file in ('model.ply', 'model.surfel.npz')]
            state['stages'][name] = {'run': str(run), 'artifacts': artifacts,
                                      'source_verification': sources, 'verified_training_input_files': len(inputs)}
        save()
        arrays = {}
        for name, run in RUNS:
            state['status'] = f'evaluating-{name}'
            command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate_surfels.py'),
                       str(run / 'model.ply'), str(ROOT / 'data/justChromeBall'),
                       '--manifest', 'transforms_val.json', '--arch', 'gpu', '--width', '800', '--height', '800',
                       '--background', 'white', '--sh-degree', '0', '--metrics', 'psnr', 'ssim', 'lpips',
                       '--lpips-net', 'alex', '--output-dir', str(output / name / 'renders'),
                       '--csv', str(output / f'{name}.csv')]
            state['stages'][name]['command'] = command
            save()
            print(f'Evaluating {name}: {run.name}', flush=True)
            with (output / f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (output / f'{name}.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            with (run / 'test-metrics.csv').open() as handle:
                old = list(csv.DictReader(handle))
            assert len(rows) == len(old) == 100
            assert [r['file'] for r in rows] == [r['file'] for r in old]
            differences = {}
            for key in ('mse', 'psnr_db', 'ssim'):
                differences[key] = max(abs(float(a[key])-float(b[key])) for a, b in zip(rows, old))
                assert differences[key] < tolerances[key], (name, key, differences[key])
            arrays[name] = {key: np.array([float(r[key]) for r in rows]) for key in ('psnr_db', 'ssim', 'lpips')}
            assert all(np.isfinite(value).all() for value in arrays[name].values())
            state['stages'][name].update(means={key: float(value.mean()) for key, value in arrays[name].items()},
                                         max_absolute_change_from_original=differences,
                                         csv=fingerprint(output / f'{name}.csv'))
            for artifact in state['stages'][name]['artifacts']:
                assert fingerprint(artifact['path'])['sha256'] == artifact['sha256']
            save()
            print(name, state['stages'][name]['means'], flush=True)
        delta = arrays['refine']['lpips'] - arrays['frozen']['lpips']
        state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat(),
                     lpips_delta_refine_minus_frozen=float(delta.mean()),
                     lpips_improved_views=int((delta < 0).sum()), lpips_worse_views=int((delta > 0).sum()))
        save()
        print('Completed:', record, flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
