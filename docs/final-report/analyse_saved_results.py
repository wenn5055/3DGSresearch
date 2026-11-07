"""Descriptive paired analysis of frozen saved metrics; no model evaluation.

Viewwise variation is camera dependence within one seed/scene, not uncertainty
over training repetitions. No hypothesis tests or independent-view CIs are used.
"""
import csv
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
DEST = ASSETS / '2026-11-07-report-analysis'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()), 'min': float(a.min()),
            'p10': float(np.percentile(a, 10)), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'max': float(a.max()),
            'improved': int((a > 0).sum()), 'worsened': int((a < 0).sum()),
            'unchanged': int((a == 0).sum())}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    sources = [ASSETS / '2026-11-03-sphere-study/test/metrics.csv',
               ASSETS / '2026-11-03-reference-normal-swap/metrics.csv']
    rows = read_csv(sources[0])
    models = {label: {int(r['view']): r for r in rows if r['model'] == label}
              for label in ('SH3', 'HDR-1.0', 'HDR-0.75')}
    views = sorted(models['SH3'])
    assert len(views) == 200 and all(sorted(m) == views for m in models.values())
    keys = ['psnr_db', 'ssim', 'lpips_alex', 'foreground_psnr_db']
    result = {'scope': 'Descriptive camera variation in one saved seed/scene; not independent training repetitions.',
              'comparisons': {}, 'normal_intervention': {}}
    all_deltas = {}
    for baseline, candidate in [('SH3', 'HDR-1.0'), ('HDR-1.0', 'HDR-0.75')]:
        label = candidate + '_vs_' + baseline
        deltas = {key: [(float(models[candidate][v][key]) - float(models[baseline][v][key]))
                        * (-1 if key == 'lpips_alex' else 1) for v in views] for key in keys}
        all_deltas[label] = deltas
        result['comparisons'][label] = {key: stats(values) for key, values in deltas.items()}
    oracle = read_csv(sources[1])
    oracle_views = sorted({int(r['view']) for r in oracle})
    assert oracle_views == [0, 13, 51, 73, 88]
    indexed = {(r['lighting'], r['normals'], int(r['view'])): r for r in oracle}
    conditional = {}
    for light in ('learned-light', 'reference-light'):
        conditional[light] = {}
        for key in ('psnr_db', 'ssim', 'lpips_alex'):
            values = [(float(indexed[light, 'reference-normal', v][key])
                       - float(indexed[light, 'learned-normal', v][key]))
                      * (-1 if key == 'lpips_alex' else 1) for v in oracle_views]
            conditional[light][key] = stats(values)
    result['normal_intervention']['normal_substitution_effect'] = conditional
    result['normal_intervention']['difference_of_conditional_mean_effects'] = {
        key: conditional['reference-light'][key]['mean'] - conditional['learned-light'][key]['mean']
        for key in ('psnr_db', 'ssim', 'lpips_alex')}
    result['normal_intervention']['interpretation'] = (
        'Effect of substituting normal direction in the existing shader, conditional on lighting; '
        'changes reflection direction and angular blend, not geometry or material parameters. '
        'Difference is descriptive on five cameras; not a factorial retraining experiment.')
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.0), constrained_layout=True)
    labels = [('psnr_db', 'Whole-image PSNR gain (dB)'),
              ('foreground_psnr_db', 'Foreground PSNR gain (dB)'),
              ('lpips_alex', 'LPIPS reduction')]
    for ax, (key, label) in zip(axes, labels):
        values = np.sort(all_deltas['HDR-1.0_vs_SH3'][key])
        ax.step(values, np.arange(1, len(values) + 1) / len(values) * 100, where='post', color='#246482', linewidth=2)
        ax.axvline(0, color='#8f9497', linestyle='--', linewidth=1)
        ax.set(xlabel=label, ylabel='Test views at or below (%)', ylim=(0, 102))
        ax.grid(alpha=.2)
    fig.suptitle('justChromeBall | 200 test views | HDR-1.0 versus matched SH3 | positive = better', fontsize=11)
    fig.savefig(DEST / 'paired-test-effects.png', dpi=190)
    plt.close(fig)
    # Compact selections from tracked actual sheets, preserving original labels.
    # Sheet layout is recorded in compare_surfel_runs.sheet: top=38, gap=12,
    # row label=30; test panels=300 and normal panels=350.
    test_sheet = ASSETS / '2026-11-03-sphere-study/test-reflection-crops.png'
    normal_sheet = ASSETS / '2026-11-03-reference-normal-swap/normal-comparison.png'
    sources += [test_sheet, normal_sheet]
    with Image.open(test_sheet) as im:
        assert im.size == (1260, 1406)
        compact = Image.new('RGB', (1260, 722), '#ededed')
        compact.paste(im.crop((0, 0, 1260, 380)), (0, 0))
        compact.paste(im.crop((0, 722, 1260, 1064)), (0, 380))
        compact.save(DEST / 'test-crops-views-0-73.png')
    with Image.open(normal_sheet) as im:
        assert im.size == (1098, 1998)
        im.crop((0, 0, 1098, 430)).save(DEST / 'normal-example-view-0.png')
    result['figure_selections'] = {
        'test-crops-views-0-73.png': {'views': [0, 73], 'source_rectangles_xyxy': [[0, 0, 1260, 380], [0, 722, 1260, 1064]]},
        'normal-example-view-0.png': {'view': 0, 'source_rectangle_xyxy': [0, 0, 1098, 430]},
        'note': 'Layout crops only; full original sheets remain tracked. No experimental pixel retouching.'}
    result['sources'] = [{'path': p.relative_to(ROOT).as_posix(),
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
    (DEST / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
