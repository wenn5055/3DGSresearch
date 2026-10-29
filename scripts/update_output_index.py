"""Refresh a concise, tracked guide to outputs/ from recorded settings."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]

# Historical descriptions checked against saved configs/logs; no causal claims.
GOALS = {
    'lego-20261028T043116Z-8a700f36': 'Lego baseline: 800px, 100k capacity, 2k steps.',
    'lego-20261028T043530Z-5299e5c7': 'vs Lego 043116: steps 2k → 10k; position decay stays 2k.',
    'materials-20261027T081123Z-e0ca4ce2': 'Prepare Materials 256px/100k/2k setup; no training.',
    'materials-20261027T082508Z-51f0952e': 'Materials baseline: 256px, 100k capacity, 2k steps.',
    'materials-20261027T095514Z-332633f0': 'Prepare 5k-step/5k-decay variant; no training.',
    'materials-20261027T101525Z-47cd07b6': 'vs Oct27 082508: steps AND position decay 2k → 5k.',
    'materials-20261027T102215Z-0b52e390': 'vs Oct27 101525: steps AND position decay 5k → 10k.',
    'materials-20261027T112424Z-c556467e': 'vs Oct27 102215: resolution 256 → 512px.',
    'materials-20261027T113646Z-58a47e63': 'vs Oct27 112424: resolution 512 → 800px; 100k capacity.',
    'materials-20261028T045111Z-98b4397b': 'vs Oct27 113646: capacity 100k → 200k; 10k steps.',
    'materials-20261028T045722Z-9fe24bef': 'vs Oct28 045111: capacity 200k → 500k; 10k steps.',
    'materials-20261028T050531Z-a8f4529f': 'vs Oct28 045722: capacity 500k → 1M AND steps 10k → 60k; decay stays 10k.',
    'materials-Oct28-1801': 'vs Oct28 050531: position decay 10k → 2k; requested 60k, stopped at 45k.',
    'materials-Oct28-1833': 'vs 1801: position decay 2k → 60k AND budget 60k → 30k; MSE reference; test re-eval in eval-test/.',
    'materials-Oct28-1945-dssim': 'vs 1833: add staged 20% D-SSIM; matched 30k-step comparison.',
    'materials-Oct28-2151-sh3-smoke': '12-step/64px execution check: D-SSIM → SH1 → SH2 → SH3, checkpoints/evaluation.',
    'materials-Oct28-2157-sh2': 'vs 1945-dssim: add SH through degree 2; same 30k budget.',
    'materials-Oct28-2157-sh3': 'vs 2157-sh2: maximum SH degree 2 → 3.',
    'materials-Oct28-2252-sh3-tune-lr-6p25e-05': 'vs 2157-sh3: SH coefficient LR 0.000125 → 0.0000625.',
    'materials-Oct28-2327-sh3-tune-lr-0p00025': 'vs 2157-sh3: SH coefficient LR 0.000125 → 0.00025.',
    'materials-Oct28-2353-gaussian_control-smoke': '6-step/32px check: Gaussian RGB + training-only initialization.',
    'materials-Oct28-2353-surfel_rgb-smoke': 'vs Gaussian smoke: flatten local z; check normals/gradients/densification/checkpoints.',
    'materials-Oct28-2353-surfel_deferred-smoke': 'vs surfel RGB smoke: add deferred environment reflection; check shading gradients/checkpoints.',
    'materials-Oct29-0003-surfel-rgb-prepare': 'Check full surfel-RGB preset/default validation split; no training.',
}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def flatten(values, prefix=''):
    result = {}
    for key, value in values.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def settings(record):
    values = flatten(record['config'])
    values.pop('paths.output', None)
    return values


def future_goal(record, previous):
    config = record['config']
    model = config['model']
    description = ' '.join(record.get('note', '').split()) or (
        f"{model.get('geometry', 'gaussian')}/{config.get('surfel', {}).get('mode', 'rgb')}/"
        f"SH{model.get('sh_degree', 0)}; {config['training']['iterations']} steps; {model['capacity']} capacity.")
    current = settings(record)
    candidates = [r for r in previous if r.get('status') == 'completed'
                  and r['config']['paths']['data'] == config['paths']['data']
                  and 'smoke' not in r['run_id'].lower()]
    if candidates:
        def differences(other):
            earlier = settings(other)
            return [(key, earlier.get(key, '(unset)'), current.get(key, '(unset)'))
                    for key in sorted(earlier.keys() | current.keys())
                    if earlier.get(key) != current.get(key)]
        reference = min(reversed(candidates), key=lambda r: len(differences(r)))
        changes = differences(reference)
        compact = ', '.join(f'{key}: {old} → {new}' for key, old, new in changes[:4])
        if len(changes) > 4:
            compact += f', +{len(changes) - 4} other settings (see config)'
        description += f" Closest prior: {reference['run_id']}; {compact or 'same recorded settings'}."
    return description


def refresh_output_index(root=ROOT):
    root = Path(root)
    outputs = root / 'outputs'
    if not outputs.is_dir():
        return
    runs, sweeps, support = [], [], []
    for directory in sorted(p for p in outputs.iterdir() if p.is_dir()):
        record = read_json(directory / 'run.json')
        sweep = read_json(directory / 'sweep.json')
        if record and 'config' in record:
            runs.append(dict(record, run_id=directory.name))
        elif sweep:
            sweeps.append((directory.name, sweep))
        else:
            support.append(directory)
    runs.sort(key=lambda r: (r.get('started_utc', ''), r['run_id']))
    date = datetime.now(timezone(timedelta(hours=8))).strftime('%d %b %Y')
    lines = ['# Output folders — quick guide', '', f'Updated {date} (Singapore).', '',
             '- Each bullet says **what changed**; multiple changes are not isolated causal tests.',
             '- `prepared` = no training; `smoke` = tiny execution check, not a quality experiment.',
             '- Split/status at the end; full evidence in each folder. `decay` means position LR decay length.',
             '- Auto-refreshed by runners; future `--note "goal / change"` supplies the purpose.',
             '- Manual refresh: `python scripts/update_output_index.py`.', '', '## Training runs', '']
    previous = []
    for record in runs:
        name = record['run_id']
        goal = GOALS.get(name) or future_goal(record, previous)
        manifest = record.get('evaluation', {}).get('manifest', '')
        split = manifest.removeprefix('transforms_').removesuffix('.json')
        lines.append(f"- [{name}](../outputs/{name}/) — {' '.join(goal.split())} [{record.get('status', 'unknown')}; {split}]")
        previous.append(record)
    lines += ['', '## Tuning', '']
    for name, sweep in sweeps:
        rates = ' / '.join(f"{c['rate']:g}" for c in sweep['candidates'])
        winner = f" Selected {sweep['best_rate']:g} by validation PSNR." if sweep.get('best_rate') is not None else ''
        lines.append(f'- [{name}](../outputs/{name}/) — SH coefficient LR: {rates}; validation only.{winner} [{sweep["status"]}]')
    lines += ['', '## Supporting folders', '']
    caches = []
    purposes = {'dssim-smoke-input-': 'tiny D-SSIM check config.',
                'sh-smoke-input-': 'tiny SH check config/launcher log.',
                'project-smoke-input-': 'tiny project-pass configs and launcher logs.',
                'runner-validation-inputs-': 'small manifests/config for runner checks.',
                'eval': 'legacy evaluation images/CSV; source checkpoint unrecorded here.',
                'warp-training_snapshots': 'legacy direct-trainer snapshots; source run unrecorded here.'}
    for directory in support:
        name = directory.name
        if 'cache' in name:
            caches.append(f'`{name}`')
            continue
        purpose = next((v for k, v in purposes.items() if name == k or (k.endswith('-') and name.startswith(k))),
                       'unrecorded folder; purpose needs review (no run.json/sweep.json).')
        lines.append(f'- [{name}](../outputs/{name}/) — {purpose}')
    if caches:
        lines.append(f"- {', '.join(caches)} — compiled Warp kernels reused by checks; not experiments.")
    lines += ['', 'Scores: [findings](findings.md). Commands: [project passes](project-passes.md).', '']
    destination = root / 'docs' / 'output-index.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.output-index-', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write('\n'.join(lines))
        # Windows can briefly hold the destination during another terminal's refresh.
        for attempt in range(10):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.02)
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    refresh_output_index()
