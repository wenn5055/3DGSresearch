"""Run only the predeclared protected SH3 control and its paired validation."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
PYTHON = str(ROOT / '.venv/Scripts/python.exe')
DEFERRED = ROOT / 'outputs/Oct30-1717-materials-hull-protected-deferred-sh3'
CONFIG = ROOT / 'config/materials_reflection_hull_regularized_surfel_sh3.yaml'
STATE = FOLDER / 'state.json'
sys.path[:0] = [str(ROOT / 'shared'), str(ROOT / 'src')]
from cp4282_gs.surfels.config import SurfelTrainingConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result.update(flatten(item, prefix + key + '.'))
        else:
            result[prefix + key] = item
    return result


def preflight():
    old = json.loads((DEFERRED / 'run.json').read_text())
    assert old['status'] == 'completed'
    assert old['training_result']['completed_iterations'] == 30000
    previous = flatten(old['config'])
    current = flatten(SurfelTrainingConfig.load(CONFIG).printable())
    differences = {key: [previous.get(key), current.get(key)]
                   for key in previous.keys() | current.keys()
                   if previous.get(key) != current.get(key)}
    assert set(differences) == {'paths.output', 'surfel.mode'}, differences
    assert differences['surfel.mode'] == ['deferred', 'rgb']
    saved_sources = json.loads((DEFERRED / 'source-files.json').read_text())
    changed_sources = [item['path'] for item in saved_sources
                       if (ROOT / item['path']).suffix == '.py'
                       and sha(ROOT / item['path']) != item['sha256']]
    assert set(changed_sources) <= {'scripts/evaluate_reflection_suite.py', 'scripts/test_surfel_hdr.py'}, changed_sources
    inputs = json.loads((DEFERRED / 'input-files.json').read_text())
    assert all(sha(item['path']) == item['sha256'] for item in inputs), 'Training/validation inputs changed'
    assert sha(current['paths.init_ply']) == '4f9acea4c85b93fcca19101ca4dffc35fd4de815c16263018b702cd51a736a64'
    previous_packages = json.loads((DEFERRED / 'packages.json').read_text())
    installed = {dist.metadata['Name']: dist.version for dist in metadata.distributions() if dist.metadata['Name']}
    package_differences = {key: [previous_packages.get(key), installed.get(key)]
                           for key in previous_packages.keys() | installed.keys()
                           if previous_packages.get(key) != installed.get(key)}
    assert not package_differences, package_differences
    for artifact in old['model_artifacts']:
        if Path(artifact['path']).suffix in ('.ply', '.npz'):
            assert sha(artifact['path']) == artifact['sha256'], artifact['path']
    audit = {'status': 'passed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
             'config_differences': differences, 'changed_python_since_deferred': changed_sources,
             'identical_input_files': len(inputs), 'package_differences': package_differences,
             'deferred_run': str(DEFERRED), 'control_config': str(CONFIG),
             'controller_sha256': sha(__file__)}
    (FOLDER / 'preflight.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)


def main():
    preflight()
    if '--preflight-only' in sys.argv:
        return
    if STATE.exists():
        raise RuntimeError('Preserve the previous session; choose a new folder.')
    state = {'status': 'starting', 'controller_pid': os.getpid(),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'stages': []}

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = STATE.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(STATE)

    def execute(name, command):
        state['status'] = name
        stage = {'name': name, 'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()}
        state['stages'].append(stage)
        save()
        with (FOLDER / (name + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            stage['pid'] = process.pid
            save()
            stage['exit_code'] = process.wait()
        stage['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if stage['exit_code']:
            raise RuntimeError(name + ' failed; inspect its log')

    os.environ.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
                      WARP_CACHE_PATH=str(ROOT / 'outputs/warp-cache'),
                      CUDA_CACHE_PATH=str(ROOT / 'outputs/cuda-cache'))
    save()
    try:
        execute('train-and-native-validation', [PYTHON, '-u', 'scripts/run_project.py', str(CONFIG),
                '--name', 'hull-protected-surfel-sh3', '--eval-manifest', 'transforms_val.json',
                '--metrics', 'psnr', 'ssim', 'lpips', '--lpips-net', 'alex',
                '--note', 'Nov 2: complete missing original Materials protected SH-only control. Fresh 30k, 800px, 1M, seed 7; matched to Oct30-1717 deferred except appearance mode. No test evaluation.'])
        candidates = []
        for path in (ROOT / 'outputs').glob('*/run.json'):
            record = json.loads(path.read_text(encoding='utf-8'))
            if (record.get('source_config') == str(CONFIG) and record['status'] == 'completed'
                    and record['started_utc'] >= state['started_utc']):
                candidates.append(path.parent)
        assert len(candidates) == 1, candidates
        control = candidates[0]
        state['control_run'] = str(control)
        comparison = FOLDER / 'paired-validation'
        state['comparison'] = str(comparison)
        save()
        execute('paired-validation', [PYTHON, '-u', 'scripts/evaluate_reflection_suite.py',
                str(control), str(DEFERRED), '--labels', 'protected-surfel-SH3', 'protected-deferred-SH3',
                '--output', str(comparison), '--manifest', 'transforms_val.json',
                '--views', '0', '13', '73', '88', '--metrics', 'psnr', 'ssim', 'lpips',
                '--geometry-reference', 'outputs/first-principles-2026-10-29/materials-geometry-val-800',
                '--material-inspection', 'outputs/first-principles-2026-10-29/materials-inspection.json',
                '--materials', 'Chrome', '--cpu-threads', '2'])
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
