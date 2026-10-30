"""Export a compact checkpoint using measured training-view blending weights.

The parent stays intact. This is inference compression, with no retraining or
claim of reduced training cost. Validate image/material quality after pruning.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from time import perf_counter

import numpy as np
from plyfile import PlyData, PlyElement
import yaml

from run_experiment import MONTHS, ROOT, SINGAPORE_TIMEZONE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_pruned(run, importance, remove_fraction, output=None):
    started = perf_counter()
    run, importance = Path(run).resolve(), Path(importance).resolve()
    if not 0 < remove_fraction < 1:
        raise ValueError('remove_fraction must be between zero and one.')
    record = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    if record['status'] not in ('completed', 'derived'):
        raise ValueError('Use a completed parent checkpoint.')
    config = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    if config != record['config']:
        raise ValueError('Parent config differs from the saved run record.')
    metadata = json.loads((importance / 'summary.json').read_text(encoding='utf-8'))
    data = Path(config['paths']['data'])
    manifest = data / 'transforms_train.json'
    if (Path(metadata['run']).resolve() != run or metadata['manifest'] != manifest.name
            or metadata['training_views'] != len(json.loads(manifest.read_text())['frames'])):
        raise ValueError('Importance must cover this checkpoint and all training views.')
    checkpoints = [run / name for name in ('model.ply', 'model.sh.npz', 'model.surfel.npz')
                   if (run / name).exists()]
    if {Path(item['path']).resolve() for item in metadata['checkpoints']} != set(checkpoints):
        raise ValueError('Importance checkpoint inventory differs from the parent.')
    for item in metadata['checkpoints'] + metadata['inputs']:
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('Stale importance input/checkpoint: ' + item['path'])
    if not any(Path(item['path']).resolve() == manifest.resolve() for item in metadata['inputs']):
        raise ValueError('Importance report lacks the training manifest fingerprint.')
    weights_path = importance / 'weights.npz'
    weights_sha = sha(weights_path)
    if metadata.get('weights_sha256', weights_sha) != weights_sha:
        raise ValueError('Weights changed after measurement.')
    ply = PlyData.read(run / 'model.ply')
    if len(ply.elements) != 1 or ply.elements[0].name != 'vertex':
        raise ValueError('Expected a point-only Gaussian PLY.')
    vertices = ply['vertex'].data
    with np.load(weights_path, allow_pickle=False) as archive:
        weights = archive['total'].copy()
    if (weights.shape != (len(vertices),) or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isclose(weights.sum(dtype=np.float64), metadata['coverage_sum'], rtol=1e-3)):
        raise ValueError('Invalid weights, point count or blending-weight conservation.')
    count = int(remove_fraction * len(vertices))
    if not 1 <= count < len(vertices):
        raise ValueError('Requested fraction must remove and retain at least one point.')
    removed = np.argsort(weights, kind='stable')[:count]
    keep_mask = np.ones(len(vertices), bool)
    keep_mask[removed] = False
    keep = np.flatnonzero(keep_mask)
    # Prepare all sidecars before creating output, including the tiny-scene case
    # where environment height happens to equal the number of points.
    sidecars = {}
    for path in checkpoints:
        if path.suffix != '.npz':
            continue
        point_key = 'sh_rest' if path.name == 'model.sh.npz' else 'reflection_logits'
        with np.load(path, allow_pickle=False) as archive:
            allowed_versions = (1, 2) if point_key == 'reflection_logits' else (1,)
            if int(archive['format_version']) not in allowed_versions or archive[point_key].shape[0] != len(vertices):
                raise ValueError('Unsupported sidecar format or inconsistent point count.')
            sidecars[path.name] = {key: value[keep] if key == point_key else value.copy()
                                  for key, value in archive.items()}
    for name in ('source-files.json', 'train.log'):
        if not (run / name).is_file():
            raise ValueError('Missing parent provenance: ' + name)
    if output is None:
        now = datetime.now(SINGAPORE_TIMEZONE)
        stamp = f'{MONTHS[now.month - 1]}{now.day:02d}-{now:%H%M}'
        scene = re.sub(r'[^A-Za-z0-9_-]', '_', data.name)
        base = ROOT / 'outputs' / f'{stamp}-{scene}-pruned-{remove_fraction:g}'
        output, suffix = base, 2
        while True:
            try:
                output.mkdir()
                break
            except FileExistsError:
                output, suffix = base.with_name(base.name + f'-{suffix}'), suffix + 1
    else:
        output = Path(output).resolve()
        output.mkdir()  # Never overwrite/reuse an existing artifact.
    PlyData([PlyElement.describe(vertices[keep], 'vertex')], text=ply.text, byte_order=ply.byte_order,
            comments=ply.comments, obj_info=ply.obj_info).write(output / 'model.ply')
    for name, values in sidecars.items():
        np.savez_compressed(output / name, **values)
    np.save(output / 'retained-parent-indices.npy', keep)
    for name in ('source-files.json', 'train.log'):
        shutil.copyfile(run / name, output / name)
    shutil.copyfile(Path(__file__), output / 'pruning-source.py')
    parent_config = record.get('source_config')
    config['paths']['output'] = str(output / 'model.png')
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    derived = deepcopy(record)
    derived.update(status='derived', run_id=output.name, config=config, source_config=str(output / 'config.yaml'),
        artifact_kind='frozen training-visibility pruning; no post-pruning training', stages=[],
        started_utc=datetime.now(timezone.utc).isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(),
        note='Compact inference checkpoint. Parent training costs are inherited; quality needs validation.')
    derived.pop('mean_metrics', None)
    derived['evaluation'] = {'status': 'pending', 'metrics': ['psnr', 'ssim', 'lpips']}
    derived['derivation'] = {'parent_run': str(run), 'parent_run_sha256': sha(run / 'run.json'),
        'parent_source_config': parent_config, 'importance_summary_sha256': sha(importance / 'summary.json'),
        'weights_sha256': weights_sha, 'weight_digest_verified_against_report': 'weights_sha256' in metadata,
        'script_sha256': sha(Path(__file__)), 'removed_points': count, 'retained_points': len(keep),
        'removed_training_blending_fraction': float(weights[removed].sum(dtype=np.float64) / max(metadata['coverage_sum'], 1e-12)),
        'seconds': perf_counter() - started, 'optimization_steps_after_pruning': 0,
        'selection': 'Lowest total T*alpha over all training cameras/pixels; stable ties. No validation/material labels.',
        'training_cost_scope': 'Parent full training; no post-pruning optimization or training-memory saving.'}
    (output / 'run.json').write_text(json.dumps(derived, indent=2), encoding='utf-8')
    (output / 'summary.md').write_text(f'# Frozen pruning export\n\nParent: `{run}`.\n\n'
        f'Retained {len(keep):,}/{len(vertices):,} points. No retraining. Validate PSNR, SSIM, LPIPS and target material quality.\n', encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--importance', type=Path, required=True)
    parser.add_argument('--remove-fraction', type=float, required=True)
    parser.add_argument('--output', type=Path, help='New directory; default begins MonDD-HHMM in Singapore time.')
    args = parser.parse_args()
    print(export_pruned(args.run, args.importance, args.remove_fraction, args.output))


if __name__ == '__main__':
    main()
