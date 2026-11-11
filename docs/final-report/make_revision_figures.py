"""Compose report revisions from saved experiment images; no rendering/training.

Default inputs are tracked assets. --archive-materials copies the two historical
Materials runs' evidence from local outputs once, preserving exact file hashes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs/reports/assets'
DEST = ASSETS / '2026-11-11-report-revision'
RUNS = {'rgb': 'materials-Oct28-1945-dssim', 'sh3': 'materials-Oct28-2157-sh3'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    path = Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-materials', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / 'materials-evidence'
    if args.archive_materials:
        archive.mkdir(exist_ok=True)
        copies = []
        for label, run in RUNS.items():
            for filename, suffix in [('run.json', 'run.json'), ('test-metrics.csv', 'metrics.csv'),
                                     ('test-renders/r_13.png', 'view-13.png')]:
                source = ROOT / 'outputs' / run / filename
                target = archive / f'{label}.{suffix}'
                if target.exists():
                    assert digest(target) == digest(source), target
                else:
                    shutil.copyfile(source, target)
                copies.append({'original': source.relative_to(ROOT).as_posix(),
                               'copy': target.relative_to(ROOT).as_posix(), 'sha256': digest(source)})
        (archive / 'archive.json').write_text(json.dumps(copies, indent=2) + '\n', encoding='utf-8')

    sources = {}

    def record(path):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
        return path

    def picture(path):
        with Image.open(record(path)) as im:
            rgba = im.convert('RGBA')
            return Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')

    def metadata(path):
        return json.loads(record(path).read_text(encoding='utf-8'))

    summary = []
    for label in RUNS:
        run = metadata(archive / f'{label}.run.json')
        with record(archive / f'{label}.metrics.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        assert run['status'] == 'completed'
        assert run['evaluation']['manifest'] == 'transforms_test.json'
        assert run['training_result']['completed_iterations'] == 30000
        assert run['config']['runtime']['seed'] == 7
        assert run['config']['model']['width'] == run['config']['model']['height'] == 800
        assert run['config']['model']['capacity'] == 1000000
        assert run['config']['model']['initial_splats'] == 1000
        assert run['config']['loss']['mode'] == 'mse_dssim'
        assert len(rows) == 200 and [int(row['view']) for row in rows] == list(range(200))
        assert all(row['file'] == f'./test/r_{index}' for index, row in enumerate(rows))
        if label == 'sh3':
            assert run['training_result']['sh_degree'] == 3
        means = {key: sum(float(row[key]) for row in rows) / len(rows) for key in ('psnr_db', 'ssim')}
        assert all(abs(value - run['mean_metrics'][key]) < 1e-10 for key, value in means.items())
        summary.append({'model': label, 'views': len(rows), **means})
    with (DEST / 'materials-summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    images = [picture(ROOT / 'data/materials/test/r_13.png')]
    images += [picture(archive / f'{label}.view-13.png') for label in RUNS]
    assert all(im.size == (800, 800) for im in images)
    canvas = Image.new('RGB', (1080, 810), 'white')
    draw = ImageDraw.Draw(canvas)
    box = (200, 505, 380, 685)
    for col, (im, title) in enumerate(zip(images, ['Blender target', 'Early fixed RGB + D-SSIM', 'Early SH3 + D-SSIM'])):
        x = col * 360
        draw.text((x + 10, 12), title, fill='#182431', font=font(21))
        canvas.paste(im.resize((360, 360), Image.Resampling.LANCZOS), (x, 46))
        draw.text((x + 10, 420), 'Test view 13 | Chrome crop', fill='#182431', font=font(19))
        canvas.paste(im.crop(box).resize((330, 330), Image.Resampling.LANCZOS), (x + 15, 458))
    canvas.save(DEST / 'materials-initial-exploration.png')

    # Preserve complete labelled rows from the original stage sheet.
    stage = picture(ASSETS / '2026-11-02-sphere/reflection-crops.png')
    assert stage.size == (1080, 1654)
    stage.crop((0, 0, 1080, 850)).save(DEST / 'refinement-improvements.png')

    environment = picture(ASSETS / '2026-11-03-environment-check/comparison.png')
    assert environment.size == (1024, 596)
    environment.crop((0, 0, 1024, 298)).save(DEST / 'environment-comparison.png')

    sheet = picture(ASSETS / '2026-11-03-reference-normal-swap/reflection-crops.png')
    assert sheet.size == (1572, 1748)
    # Original sheet: 12px gaps, 300px panels, 38px heading + 30px row label.
    rectangles = [(12 + col * 312, 68, 312 + col * 312, 368) for col in range(5)]
    tiles = [sheet.crop(rectangle) for rectangle in rectangles]
    canvas = Image.new('RGB', (1030, 780), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), 'Chrome sphere | validation 0 | HDR-0.75 | same crop in all cases',
              fill='#182431', font=font(22))
    layout = [(0, 20, 275, 'Blender target'),
              (1, 370, 100, 'Learned light + learned normals'),
              (3, 710, 100, 'Learned light + reference normals'),
              (2, 370, 460, 'Reference light + learned normals'),
              (4, 710, 460, 'Reference light + reference normals')]
    for index, x, y, title in layout:
        draw.text((x, y - 28), title, fill='#182431', font=font(17))
        canvas.paste(tiles[index], (x, y))
    canvas.save(DEST / 'lighting-normal-permutations.png')

    record(Path(__file__).resolve())
    provenance = {
        'date_singapore': '2026-11-11',
        'scope': 'Existing experimental images and saved metrics only; no new training or evaluation.',
        'materials': {'scene': 'original Materials', 'split': 'test', 'view': 13,
                      'selection': 'Reuse existing illustration camera 13, without ranking historical view scores.',
                      'crop_xyxy': list(box), 'means': summary,
                      'historical_protocol': 'Initial exploration; validation contributed to initial mean colour. Separate from later training-only surfel study.'},
        'figures': {
            'materials-initial-exploration.png': 'Full frames downsampled; identical Chrome crops enlarged with Lanczos; reference alpha composited on white.',
            'refinement-improvements.png': {'source_rectangle_xyxy': [0, 0, 1080, 850],
                                            'views': [0, 13], 'model': 'HDR-1.0'},
            'environment-comparison.png': {'source_rectangle_xyxy': [0, 0, 1024, 298], 'display_ev': 0},
            'lighting-normal-permutations.png': {'view': 0, 'model': 'HDR-0.75',
                                                 'source_rectangles_xyxy': rectangles,
                                                 'source_order': ['target', 'learned/learned', 'reference/learned',
                                                                  'learned/reference', 'reference/reference'],
                                                 'grid': 'Rows: learned/reference lighting; columns: learned/reference normals.'}},
        'display': 'Layout crops, labels and uniform resizing only; no experimental pixel retouching.',
        'sources': [{'path': path, 'sha256': sha} for path, sha in sources.items()]}
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Saved four revised figures and verified early Materials means from 400 archived metric rows.')


if __name__ == '__main__':
    main()
