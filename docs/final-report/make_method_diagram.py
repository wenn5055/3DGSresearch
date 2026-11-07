"""Draw a schematic of the implemented pipeline, not an experimental render."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'docs/reports/assets/2026-11-07-report-readability'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 2.9))
    fig.subplots_adjust(left=.015, right=.985, top=.96, bottom=.06)
    ax.set(xlim=(0, 12), ylim=(0, 3))
    ax.axis('off')

    def box(x, y, width, height, label, fill='#edf4f8'):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.07',
                                   linewidth=1.1, edgecolor='#7392a3', facecolor=fill))
        ax.text(x + width/2, y + height/2, label, ha='center', va='center',
                fontsize=12, color='#183449', linespacing=1.3)

    def arrow(a, b, color='#386783'):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
                                     linewidth=1.3, color=color))

    positions = [.12, 3.15, 6.18, 9.21]
    labels = ['Learned scene\npatches + lighting',
              'First rendering pass\nblend surface attributes',
              'Second rendering pass\ncompute pixel reflections',
              'Rendered image\npredicted RGB colours']
    for x, label in zip(positions, labels):
        box(x, 1.82, 2.65, .88, label)
    for x in positions[:-1]:
        arrow((x + 2.74, 2.26), (x + 2.93, 2.26))
    ax.text(6.03, 2.93, 'FORWARD RENDERING', ha='center', fontsize=10.5, color='#466172')
    box(5.2, .18, 5.9, .78, 'Compute losses\nimage / mask matching\n+ normal–depth consistency', '#fbf4e7')
    box(.52, .18, 3.65, .78, 'Training feedback\nupdate learned parameters', '#edf5ef')
    arrow((4.48, 1.70), (6.0, 1.02))
    ax.text(5.13, 1.39, 'surface maps', ha='center', fontsize=10, color='#466172',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    arrow((10.54, 1.70), (9.8, 1.02))
    ax.text(10.32, 1.39, 'RGB', ha='center', fontsize=10, color='#466172',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    arrow((5.06, .57), (4.33, .57))
    arrow((1.45, 1.08), (1.45, 1.67))
    fig.savefig(DEST / 'method-pipeline.svg')
    fig.savefig(DEST / 'method-pipeline.png', dpi=180)
    plt.close(fig)
    sources = [ROOT / 'src/cp4282_gs/surfels/trainer.py',
               ROOT / 'src/cp4282_gs/surfels/hdr_shader.py',
               ROOT / 'src/cp4282_gs/surfels/geometry_losses.py']
    record = {'kind': 'Schematic of implemented method; not experiment data or a planned result.',
              'note': 'Top: forward computation. Bottom: training feedback. Geometry is fixed in the first training stage.',
              'sources': [{'path': p.relative_to(ROOT).as_posix(),
                           'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
              'script': {'path': Path(__file__).relative_to(ROOT).as_posix(),
                         'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (DEST / 'diagram-provenance.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Saved implemented-method schematic (SVG and PNG).')


if __name__ == '__main__':
    main()
