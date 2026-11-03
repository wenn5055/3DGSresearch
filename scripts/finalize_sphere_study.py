"""Freeze the validation-only sphere choice, then evaluate both selected models on test."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def choose_hdr(summaries):
    baseline, smaller = summaries['HDR-1.0'], summaries['HDR-0.75']
    checks = {'lpips_improves': float(smaller['lpips_alex']) < float(baseline['lpips_alex']),
              'foreground_psnr_not_lower': float(smaller['foreground_psnr_db']) >= float(baseline['foreground_psnr_db']),
              'foreground_coverage_at_least_99_percent': float(smaller['foreground_coverage_fraction']) >= .99}
    return ('HDR-0.75' if all(checks.values()) else 'HDR-1.0'), checks


def fingerprint(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('validation', type=Path)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--test-output', type=Path, required=True)
    parser.add_argument('--select-only', action='store_true', help='Freeze and print the final command without evaluating test.')
    args = parser.parse_args()
    protocol = json.loads((args.validation/'protocol.json').read_text())
    assert protocol['status'] == 'completed' and protocol['manifest'] == 'transforms_val.json'
    with (args.validation/'summary.csv').open() as handle:
        summary = {row['model']: row for row in csv.DictReader(handle)}
    assert set(summary) == {'SH3', 'HDR-1.0', 'HDR-0.75'}
    assert not any(protocol['training_vs_evaluation_source_differences'])
    selected, checks = choose_hdr(summary)
    lookup = dict(zip(protocol['labels'], protocol['runs']))
    # Retain the same-size primary contrast; the selected size variant is additional.
    test_labels = ['SH3', 'HDR-1.0'] + (['HDR-0.75'] if selected == 'HDR-0.75' else [])
    runs = [Path(lookup[label]) for label in test_labels]
    artifacts = [[fingerprint(run/name) for name in ('model.ply', 'model.surfel.npz', 'model.sh.npz')
                  if (run/name).is_file()] for run in runs]
    command = [sys.executable, '-u', str(ROOT/'scripts/evaluate_reflection_suite.py'),
               *(str(run) for run in runs), '--labels', *test_labels,
               '--output', str(args.test_output.resolve()), '--manifest', 'transforms_test.json',
               '--views', '0', '13', '73', '88', '--metrics', 'psnr', 'ssim', 'lpips',
               '--cpu-threads', '2']
    record = {'status': 'frozen-before-test', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'validation_protocol': fingerprint(args.validation/'protocol.json'),
              'validation_summary': summary, 'chosen_hdr_label': selected, 'checks': checks,
              'selection_rule': 'smaller size must improve mean validation LPIPS, not lower foreground PSNR, and retain mean foreground coverage >=0.99',
              'all_validation_runs': protocol['runs'], 'selected_labels': test_labels,
              'primary_matched_comparison': ['SH3', 'HDR-1.0'],
              'selected_checkpoints': artifacts, 'test_command': command, 'script': fingerprint(__file__)}
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    with args.selection.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    print(f'Frozen test models: {test_labels}; selected HDR={selected}; checks={checks}', flush=True)
    if not args.select_only:
        subprocess.run(command, cwd=ROOT, check=True)
        for group in artifacts:
            for item in group:
                assert fingerprint(item['path'])['sha256'] == item['sha256']


if __name__ == '__main__':
    main()
