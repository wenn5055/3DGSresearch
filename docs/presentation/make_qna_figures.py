"""Plot Q&A diagnostics from existing saved images/buffers; no model evaluation.

--archive-inputs prepares portable arrays from the completed validation renders.
Default rebuilding uses only the archived arrays and their recorded metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs/reports/assets'
DEST = ROOT / 'docs/presentation/assets/2026-11-12-qna'
VIEWS = [0, 51]
MODELS = ['SH3', 'HDR-1.0']


def fingerprint(path):
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-inputs', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    inputs = DEST / 'inputs'
    if args.archive_inputs:
        inputs.mkdir(exist_ok=True)
        source = ROOT / 'outputs/Nov03-sphere-validation'
        reference = ROOT / 'outputs/Nov03-sphere-geometry-val'
        sources = []

        def record(path):
            sources.append(fingerprint(path))
            return path

        protocol = json.loads(record(source / 'protocol.json').read_text(encoding='utf-8'))
        geometry = json.loads(record(reference / 'geometry.json').read_text(encoding='utf-8'))
        assert protocol['status'] == geometry['status'] == 'completed'
        assert protocol['manifest'] == 'transforms_val.json'
        assert protocol['labels'][:2] == MODELS
        assert protocol['resolution'] == geometry['resolution'] == [800, 800]
        manifest = ROOT / 'data/justChromeBall/transforms_val.json'
        assert fingerprint(record(manifest))['sha256'] == geometry['manifest_sha256']
        capture = json.loads(record(ROOT / 'data/justChromeBall/capture.json').read_text(encoding='utf-8'))
        assert capture['specification']['scene_sha256'] == geometry['scene_sha256']
        assert Path(protocol['runs'][1]).name == 'Nov02-2023-justChromeBall-staged-hdr-refine-5k'
        archive_files = []
        for view in VIEWS:
            assert view in protocol['selected_views'] and view in geometry['views']
            with np.load(record(reference / f'r_{view}.npz')) as truth:
                arrays = {'reference_normal': truth['normal'], 'reference_depth': truth['depth'],
                          'reference_hit': truth['object_id'] > 0}
            with Image.open(record(ROOT / f'data/justChromeBall/val/r_{view}.png')) as image:
                arrays['foreground'] = np.asarray(image.convert('RGBA'))[..., 3] / 255 >= .95
            names = ['reference'] + MODELS
            files = [f'reference-{view}.png', f'model-0-{view}.png', f'model-1-{view}.png']
            for name, filename in zip(names, files):
                with Image.open(record(source / filename)) as image:
                    arrays[f'{name}_rgb'] = np.asarray(image.convert('RGB'))
            for index, model in enumerate(MODELS):
                with np.load(record(source / f'model-{index}-{view}.buffers.npz')) as saved:
                    for key in ('normal', 'depth', 'coverage'):
                        arrays[f'{model}_{key}'] = saved[key]
            path = inputs / f'validation-{view}.npz'
            np.savez_compressed(path, **arrays)
            archive_files.append(fingerprint(path))
        archive = {'scene': 'justChromeBall', 'split': 'validation', 'views': VIEWS,
                   'models': MODELS, 'files': archive_files, 'original_sources': sources,
                   'geometry_reference': geometry,
                   'selection': 'Existing illustration view 0 and previously preserved failure view 51.',
                   'note': 'Saved RGB PNGs and float normal/depth/coverage arrays only; no training or new rendering.'}
        (DEST / 'archive.json').write_text(json.dumps(archive, indent=2) + '\n', encoding='utf-8')

    archive = json.loads((DEST / 'archive.json').read_text(encoding='utf-8'))
    assert archive['views'] == VIEWS and archive['models'] == MODELS
    for item in archive['files']:
        assert fingerprint(ROOT / item['path'])['sha256'] == item['sha256']
    data, samples, statistics = {}, [], []
    for view in VIEWS:
        with np.load(inputs / f'validation-{view}.npz') as saved:
            row = {key: saved[key] for key in saved.files}
        valid = row['reference_hit'] & row['foreground']
        for model in MODELS:
            valid &= row[f'{model}_coverage'] >= .5
            valid &= np.linalg.norm(row[f'{model}_normal'], axis=-1) > 1e-8
            valid &= np.isfinite(row[f'{model}_depth'])
        assert valid.any()
        row['valid'] = valid
        for model in MODELS:
            normal = row[f'{model}_normal'].astype(np.float64)
            normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
            truth = row['reference_normal'].astype(np.float64)
            truth /= np.maximum(np.linalg.norm(truth, axis=-1, keepdims=True), 1e-8)
            row[f'{model}_angle'] = np.degrees(np.arccos(np.clip((normal * truth).sum(-1), -1, 1)))
            row[f'{model}_depth_error'] = np.abs(row[f'{model}_depth'] - row['reference_depth'])
            row[f'{model}_rgb_error'] = np.abs(row[f'{model}_rgb'].astype(float) - row['reference_rgb']).mean(-1) / 255
            statistics.append({'view': view, 'model': model, 'joint_valid_pixels': int(valid.sum()),
                               'normal_error_mean_deg': float(row[f'{model}_angle'][valid].mean()),
                               'depth_error_mean_scene_units': float(row[f'{model}_depth_error'][valid].mean()),
                               'normal_fraction_clipped_above_5_deg': float((row[f'{model}_angle'][valid] > 5).mean()),
                               'depth_fraction_clipped_above_0_02': float((row[f'{model}_depth_error'][valid] > .02).mean()),
                               'rgb_fraction_clipped_above_0_25': float((row[f'{model}_rgb_error'] > .25).mean())})
        samples += [row[key][valid] for key in ('reference_depth', 'SH3_depth', 'HDR-1.0_depth')]
        data[view] = row
    depth_low = min(float(value.min()) for value in samples)
    depth_high = max(float(value.max()) for value in samples)
    plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white'})
    output_files = []

    def board(kind, titles):
        fig, axes = plt.subplots(2, len(titles), figsize=(3.15 * len(titles), 6.6),
                                 constrained_layout=True, squeeze=False)
        for col, title in enumerate(titles):
            axes[0, col].set_title(title, fontsize=12)
        for row_index, view in enumerate(VIEWS):
            axes[row_index, 0].text(-.05, .5, f'Validation view {view}', rotation=90,
                                     va='center', ha='right', transform=axes[row_index, 0].transAxes)
            for ax in axes[row_index]:
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        fig.suptitle(f'Chrome sphere | {kind} | same saved SH3 and HDR-1.0 models', fontsize=15)
        return fig, axes

    fig, axes = board('RGB and absolute differences',
                      ['Blender target', 'SH3', 'HDR-1.0', 'SH3 RGB difference', 'HDR-1.0 RGB difference'])
    for r, view in enumerate(VIEWS):
        row = data[view]
        for col, name in enumerate(['reference'] + MODELS):
            axes[r, col].imshow(row[f'{name}_rgb'])
        for col, model in enumerate(MODELS, 3):
            rgb_map = axes[r, col].imshow(row[f'{model}_rgb_error'], cmap='magma', vmin=0, vmax=.25)
    fig.colorbar(rgb_map, ax=axes[:, 3:].ravel().tolist(), shrink=.75,
                 label='Mean absolute RGB error [0, 1]; clipped at 0.25')
    fig.supxlabel('Saved 8-bit PNG differences for visual inspection; not the float-render scores used in the report.', fontsize=11)
    path = DEST / 'rgb-differences.png'; fig.savefig(path, dpi=155); plt.close(fig); output_files.append(path)

    fig, axes = board('surface normals and angular error',
                      ['Blender mesh normal', 'SH3 normal', 'HDR-1.0 normal', 'HDR-1.0 angular error'])
    for r, view in enumerate(VIEWS):
        row, valid = data[view], data[view]['valid']
        for col, key in enumerate(['reference_normal', 'SH3_normal', 'HDR-1.0_normal']):
            axes[r, col].imshow(np.where(valid[..., None], row[key] * .5 + .5, 1).clip(0, 1))
        angle_map = axes[r, 3].imshow(np.ma.masked_where(~valid, row['HDR-1.0_angle']),
                                      cmap='magma', vmin=0, vmax=5)
    fig.colorbar(angle_map, ax=axes[:, 3].tolist(), shrink=.75, label='Angular error (degrees); clipped at 5')
    fig.supxlabel('Normals: world-space XYZ encoded as RGB. White outside jointly valid foreground; colour brightness is not accuracy.', fontsize=11)
    path = DEST / 'normal-comparison-errors.png'; fig.savefig(path, dpi=155); plt.close(fig); output_files.append(path)

    fig, axes = board('camera depth and absolute depth error',
                      ['Blender mesh depth', 'SH3 depth', 'HDR-1.0 depth', 'HDR-1.0 depth error'])
    for r, view in enumerate(VIEWS):
        row, valid = data[view], data[view]['valid']
        for col, key in enumerate(['reference_depth', 'SH3_depth', 'HDR-1.0_depth']):
            depth_map = axes[r, col].imshow(np.ma.masked_where(~valid, row[key]),
                                            cmap='viridis', vmin=depth_low, vmax=depth_high)
        depth_error = axes[r, 3].imshow(np.ma.masked_where(~valid, row['HDR-1.0_depth_error']),
                                        cmap='magma', vmin=0, vmax=.02)
    fig.colorbar(depth_map, ax=axes[:, :3].ravel().tolist(), shrink=.75, label='Camera-z depth (scene units); shared scale')
    fig.colorbar(depth_error, ax=axes[:, 3].tolist(), shrink=.75, label='Absolute depth error (scene units); clipped at 0.02')
    fig.supxlabel('Same depth range across both views and all three surfaces. White pixels are excluded; scene units are not metres.', fontsize=11)
    path = DEST / 'depth-comparison-errors.png'; fig.savefig(path, dpi=155); plt.close(fig); output_files.append(path)

    provenance = {'scene': archive['scene'], 'split': archive['split'], 'views': VIEWS, 'models': MODELS,
                  'kind': 'New visualisations of existing saved experimental arrays; no model rendering or optimisation.',
                  'validity': 'Intersection: reference alpha >= 0.95, mesh hit, both model coverages >= 0.5, nonzero normals and finite depths.',
                  'normal_encoding': 'world-space XYZ in [-1,1] mapped to RGB in [0,1]',
                  'angular_error': 'acos of normalised dot product against the Blender mesh normal, in degrees',
                  'depth_convention': archive['geometry_reference']['depth_convention'],
                  'depth_shared_range_scene_units': [depth_low, depth_high],
                  'rgb_error': 'Mean over RGB of absolute saved uint8 PNG differences / 255. Full-frame, including background; not report PSNR input.',
                  'statistics_scope': 'Two selected validation views, using a common covered-foreground mask; not the five-view or full-split means.',
                  'statistics': statistics,
                  'sources': [fingerprint(DEST / 'archive.json'), *archive['files'], fingerprint(Path(__file__).resolve())],
                  'figures': [fingerprint(path) for path in output_files]}
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'figures': [p.name for p in output_files], 'statistics': statistics,
                      'depth_range_scene_units': [depth_low, depth_high]}, indent=2))


if __name__ == '__main__':
    main()
