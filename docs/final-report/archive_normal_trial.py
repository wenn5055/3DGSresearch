"""Archive the completed normal-weight diagnostic; compose only real saved renders.

Run with the research environment after the two evaluation commands in the
normal-weight closeout. This does not train, shade, or invent experimental data.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'docs/reports/assets/2026-11-07-normal-weight-closeout'
VALIDATION = ROOT / 'outputs/Nov07-report-normal003-validation'
NORMALS = ROOT / 'outputs/Nov07-report-normal003-normals'
RUNS = [ROOT / 'outputs/Nov02-2023-justChromeBall-staged-hdr-refine-5k',
        ROOT / 'outputs/Nov03-1753-justChromeBall-normal003-100k']
SOURCES = {}


def fingerprint(path):
    record = {'path': path.relative_to(ROOT).as_posix(),
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'bytes': path.stat().st_size}
    SOURCES[record['path']] = record
    return record


def read_json(path):
    fingerprint(path)
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    fingerprint(path)
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + '.'))
        else:
            result[name] = item
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = [read_json(run / 'run.json') for run in RUNS]
    assert all(record['status'] == 'completed' for record in records)
    configs = [flatten(record['config']) for record in records]
    differences = {key: [c.get(key) for c in configs] for key in configs[0]
                   if configs[0].get(key) != configs[1].get(key)}
    assert set(differences) == {'geometry_loss.normal_depth_weight', 'paths.output'}, differences
    assert differences['geometry_loss.normal_depth_weight'] == [0.01, 0.03]
    for name in ('model.ply', 'model.surfel.npz'):
        fingerprint(RUNS[0].parent / 'Nov02-2021-justChromeBall-frozen-hdr-10k' / name)
    source_records = [read_json(run / 'source-files.json') for run in RUNS]
    core = lambda records: {r['path']: r['sha256'] for r in records
                            if r['path'].startswith(('src/', 'shared/'))
                            or r['path'] in ('3dgs_trainer.py', 'gaussian_first_tile_workspace_gpu.py')}
    assert core(source_records[0]) == core(source_records[1])
    validation_protocol = read_json(VALIDATION / 'protocol.json')
    normal_protocol = read_json(NORMALS / 'protocol.json')
    assert validation_protocol['status'] == normal_protocol['status'] == 'completed'
    summaries = read_csv(VALIDATION / 'summary.csv')
    rows = read_csv(VALIDATION / 'metrics.csv')
    diagnostics = read_csv(NORMALS / 'diagnostics.csv')
    assert len(rows) == 200 and len(diagnostics) == 10
    paired = {label: [r for r in rows if r['model'] == label] for label in ('normal001', 'normal003')}
    assert all(len(value) == 100 for value in paired.values())
    assert [r['view'] for r in paired['normal001']] == [r['view'] for r in paired['normal003']]
    means = {label: {key: mean(float(r[key]) for r in diagnostics if r['model'] == label)
                    for key in ('mesh_normal_mae_deg', 'normal_coherence', 'foreground_coverage_fraction')}
             for label in paired}
    counts = {key: sum((float(b[key]) > float(a[key])) if key != 'lpips_alex'
                      else (float(b[key]) < float(a[key]))
                      for a, b in zip(paired['normal001'], paired['normal003']))
              for key in ('psnr_db', 'ssim', 'lpips_alex')}
    files = [(VALIDATION / 'summary.csv', 'validation-summary.csv'),
             (VALIDATION / 'metrics.csv', 'validation-metrics.csv'),
             (VALIDATION / 'protocol.json', 'validation-protocol.json'),
             (VALIDATION / 'crop-metrics.csv', 'crop-metrics.csv'),
             (NORMALS / 'diagnostics.csv', 'normal-diagnostics.csv'),
             (NORMALS / 'protocol.json', 'normal-protocol.json'),
             (NORMALS / 'normal-comparison.png', 'normal-comparison.png'),
             (NORMALS / 'normal-errors.png', 'normal-errors.png'),
             (NORMALS / 'footprints.png', 'footprints.png'),
             (RUNS[1] / 'run.json', 'trial-run.json'),
             (RUNS[1] / 'config.yaml', 'trial-config.yaml')]
    for source, name in files:
        fingerprint(source)
        shutil.copyfile(source, DEST / name)
    # Reference-defined central reflection crops, unrelated to candidate scores.
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    cell, header, label_height = 360, 42, 36
    canvas = Image.new('RGB', (cell * 3, header + 2 * (cell + label_height)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(('Blender reference', 'Normal weight 0.01', 'Normal weight 0.03')):
        draw.text((index * cell + 10, 10), title, fill='#182431', font=font)
    boxes = {}
    for row, view in enumerate((0, 51)):
        target = ROOT / f'data/justChromeBall/val/r_{view}.png'
        fingerprint(target)
        with Image.open(target) as im:
            x0, y0, x1, y1 = im.getchannel('A').getbbox()
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = round(min(x1 - x0, y1 - y0) * .24)
        box = (cx - half, cy - half, cx + half, cy + half)
        boxes[str(view)] = box
        paths = [VALIDATION / f'reference-{view}.png'] + [VALIDATION / f'model-{i}-{view}.png' for i in (0, 1)]
        for col, path in enumerate(paths):
            fingerprint(path)
            with Image.open(path) as im:
                crop = im.convert('RGB').crop(box).resize((cell, cell), Image.Resampling.LANCZOS)
            y = header + row * (cell + label_height)
            canvas.paste(crop, (col * cell, y))
            draw.text((col * cell + 10, y + cell + 7), f'justChromeBall | val r_{view}', fill='#182431', font=font)
    canvas.save(DEST / 'reflection-crops.png')
    result = {'status': 'completed', 'analysis_date_singapore': '2026-11-07',
              'training_date_singapore': '2026-11-03', 'evaluation_only': True,
              'config_differences': differences, 'training_core_sources_match': True,
              'validation_summary': summaries, 'five_view_normal_means': means,
              'improved_views_out_of_100': counts, 'crop_boxes_xyxy': boxes,
              'normal_scope': normal_protocol['normal_error'],
              'conclusion': 'Higher weight slightly worsens mean image and reference-normal scores despite higher normal coherence; no successful detail recovery established.',
              'sources': list(SOURCES.values())}
    (DEST / 'provenance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('five_view_normal_means', 'improved_views_out_of_100')}, indent=2))


if __name__ == '__main__':
    main()
