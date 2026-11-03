"""Archive completed sphere-study evidence and compose actual matched reflection crops."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parents[3]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'shared'), str(ROOT/'src')]
from compare_surfel_runs import sheet, fingerprint
from evaluate_reflection_suite import renderer_source_differences

VIEWS = [0, 13, 73, 88]


def read_csv(path):
    with path.open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def crop_figure(source, split, name):
    protocol = json.loads((source/'protocol.json').read_text())
    rows, bounds, sources = [], {}, []
    for view in VIEWS:
        with Image.open(ROOT/f'data/justChromeBall/{split}/r_{view}.png') as rgba:
            x0, y0, x1, y1 = rgba.getchannel('A').getbbox()
        cx, cy = (x0+x1)//2, (y0+y1)//2
        half = round(min(x1-x0, y1-y0)*.24)
        box = (cx-half, cy-half, cx+half, cy+half)
        bounds[str(view)] = box
        paths = [source/f'reference-{view}.png'] + [source/f'model-{index}-{view}.png' for index in range(len(protocol['labels']))]
        panels = []
        for path in paths:
            sources.append(fingerprint(path))
            with Image.open(path) as image:
                panels.append(image.convert('RGB').crop(box))
        rows.append((f'justChromeBall | {split} r_{view} | identical central reflection crop', panels))
    sheet(ASSETS/name, rows, ['Reference']+protocol['labels'], panel=300)
    return {'bounds_xyxy': bounds, 'sources': sources}


def main():
    study = ROOT/'outputs/Nov03-sphere-study'
    selection = json.loads((study/'selection.json').read_text())
    assert selection['status'] == 'frozen-before-test'
    evidence = {'selection': selection, 'figures': {}, 'sources': []}
    for phase, source in [('validation', ROOT/'outputs/Nov03-sphere-validation'),
                          ('test', ROOT/'outputs/Nov03-sphere-final-test')]:
        protocol = json.loads((source/'protocol.json').read_text())
        assert protocol['status'] == 'completed'
        assert not any(protocol['training_vs_evaluation_source_differences'])
        if phase == 'test':
            assert selection['created_utc'] < protocol['created_utc']
        destination = ASSETS/phase
        destination.mkdir(exist_ok=True)
        for filename in ('summary.csv', 'metrics.csv', 'crop-metrics.csv', 'protocol.json', 'performance.json'):
            evidence['sources'].append(fingerprint(source/filename))
            shutil.copyfile(source/filename, destination/filename)
        evidence['figures'][phase] = crop_figure(source, 'val' if phase=='validation' else 'test', f'{phase}-reflection-crops.png')
    diagnostics = ROOT/'outputs/Nov03-sphere-footprints-size'
    for filename in ('diagnostics.csv', 'protocol.json', 'normal-comparison.png', 'normal-errors.png',
                     'learned-environments.png', 'footprints.png'):
        evidence['sources'].append(fingerprint(diagnostics/filename))
        shutil.copyfile(diagnostics/filename, ASSETS/filename)
    for filename in ('preflight.json', 'size-preflight.json', 'selection.json'):
        shutil.copyfile(study/filename, ASSETS/filename)
    training_runs = {Path(run) for run in selection['all_validation_runs']}
    for run in list(training_runs):
        record = json.loads((run/'run.json').read_text())
        parent = Path(record['config']['paths']['init_ply']).parent
        if (parent/'run.json').exists():
            training_runs.add(parent)
    for path in sorted(training_runs):
        assert not renderer_source_differences(path)
        dest = ASSETS/path.name
        dest.mkdir(exist_ok=True)
        for filename in ('config.yaml', 'run.json'):
            shutil.copyfile(path/filename, dest/filename)
    rows = read_csv(ROOT/'outputs/Nov03-sphere-validation/metrics.csv')
    labels = list(dict.fromkeys(row['model'] for row in rows))
    values = {label: np.array([float(row['lpips_alex']) for row in rows if row['model']==label]) for label in labels}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for label, data in values.items():
        axes[0].plot(np.arange(len(data)), data, linewidth=.9, label=label)
    axes[0].set(xlabel='Validation view', ylabel='LPIPS AlexNet (lower better)', title='justChromeBall | all 100 validation views')
    axes[0].legend(fontsize=8)
    delta = values['HDR-0.75']-values['HDR-1.0']
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].plot(np.arange(100), delta, color='#087e8b', linewidth=1)
    axes[1].set(xlabel='Validation view', ylabel='LPIPS: size 0.75 minus 1.0', title=f'Smaller initial splats: {int((delta<0).sum())}/100 improve')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(ASSETS/'validation-lpips.png', dpi=180)
    plt.close(fig)
    evidence['script'] = fingerprint(Path(__file__))
    evidence['note'] = 'All figures use saved actual experiment images/scalars. Central crops are identical across variants. No retouching.'
    (ASSETS/'figure-provenance.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print('Archived sphere study:', ASSETS)


if __name__ == '__main__':
    main()
