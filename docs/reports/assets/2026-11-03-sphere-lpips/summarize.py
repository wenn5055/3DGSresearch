"""Archive the completed evaluation and plot LPIPS from its recorded CSVs."""

import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parents[3]
SOURCE = ROOT / 'outputs/Nov03-sphere-lpips-verified'
sys.path.insert(0, str(ROOT / 'shared'))
from gaussian_set import GaussianSet


def main():
    state = json.loads((SOURCE / 'evaluation.json').read_text(encoding='utf-8'))
    assert state['status'] == 'completed'
    values = []
    for name in ('frozen', 'refine'):
        with (SOURCE / f'{name}.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert [r['file'] for r in rows] == [f'./val/r_{i}' for i in range(100)]
        values.append(np.array([float(r['lpips']) for r in rows]))
        shutil.copyfile(SOURCE / f'{name}.csv', ASSETS / f'{name}.csv')
    shutil.copyfile(SOURCE / 'evaluation.json', ASSETS / 'evaluation.json')
    scales = {}
    for name in ('frozen', 'refine'):
        scene = GaussianSet.from_ply(Path(state['stages'][name]['run']) / 'model.ply')
        scales[name] = {'tangent_sigma_scene_units_p5_p50_p95': np.percentile(scene.scales[:, :2], [5, 50, 95]).tolist(),
                        'scope': 'both tangent axes across all stored primitives; not screen footprint or visibility weighted'}
    (ASSETS / 'scale-statistics.json').write_text(json.dumps(scales, indent=2) + '\n', encoding='utf-8')
    x, y = values
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    axes[0].plot([lo, hi], [lo, hi], '--', color='#7b8794', linewidth=1)
    axes[0].scatter(x, y, color='#087e8b', s=20, alpha=.8)
    axes[0].set(xlabel='Frozen HDR, 10k: LPIPS', ylabel='Refined HDR, 15k total: LPIPS',
                title='justChromeBall | all 100 validation views\nLower is better; all 100 improve')
    axes[1].axhline(0, color='#7b8794', linestyle='--', linewidth=1)
    axes[1].plot(np.arange(100), y-x, color='#087e8b', linewidth=1)
    axes[1].set(xlabel='Validation view index', ylabel='LPIPS change: refined minus frozen',
                title=f'Mean {x.mean():.6f} -> {y.mean():.6f}\n{100*(1-y.mean()/x.mean()):.2f}% lower')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(ASSETS / 'lpips-validation.png', dpi=180)
    plt.close(fig)
    with (ASSETS / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['stage', 'cumulative_steps', 'views', 'psnr_db', 'ssim', 'lpips'])
        writer.writeheader()
        for name, steps in [('frozen', 10000), ('refine', 15000)]:
            writer.writerow({'stage': name, 'cumulative_steps': steps, 'views': 100, **state['stages'][name]['means']})
    print(json.dumps({key: state[key] for key in ('status', 'lpips_delta_refine_minus_frozen',
                                                'lpips_improved_views', 'lpips_worse_views')}, indent=2))


if __name__ == '__main__':
    main()
