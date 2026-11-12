"""Compose full test views from an archived experimental sheet; no model rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs/reports/assets'
DEST = ASSETS / '2026-11-12-full-views'
STUDY = ASSETS / '2026-11-03-sphere-study'
VIEWS = [0, 13, 73, 88]


def fingerprint(path):
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-source', action='store_true',
                        help='Verify and copy the original local saved RGB sheet once.')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((STUDY / 'test/protocol.json').read_text(encoding='utf-8'))
    assert protocol['manifest'] == 'transforms_test.json'
    assert protocol['selected_views'] == VIEWS
    assert protocol['labels'][1] == 'HDR-1.0'
    assert Path(protocol['runs'][1]).name == 'Nov02-2023-justChromeBall-staged-hdr-refine-5k'
    source = DEST / 'source-test-rgb.png'
    if args.archive_source:
        original = ROOT / 'outputs/Nov03-sphere-final-test/rgb-comparison.png'
        evidence = json.loads((STUDY / 'figure-provenance.json').read_text(encoding='utf-8'))
        expected = {Path(item['path']).name: item['sha256']
                    for item in evidence['figures']['test']['sources']}
        verified = []
        with Image.open(original) as sheet:
            assert sheet.size == (1660, 1806)
            for row, view in enumerate(VIEWS):
                for column, name in [(0, f'reference-{view}.png'), (1, f'model-0-{view}.png'),
                                     (2, f'model-1-{view}.png')]:
                    path = original.parent / name
                    item = fingerprint(path)
                    assert item['sha256'] == expected[name], name
                    with Image.open(path) as image:
                        panel = image.convert('RGB').resize((400, 400), Image.Resampling.LANCZOS)
                    x, y = 12 + column * 412, 68 + row * 442
                    assert panel.tobytes() == sheet.crop((x, y, x + 400, y + 400)).tobytes()
                    verified.append(item)
        if source.exists():
            assert fingerprint(source)['sha256'] == fingerprint(original)['sha256']
        else:
            shutil.copyfile(original, source)
        archive = {'original': fingerprint(original), 'copy': fingerprint(source),
                   'verified_original_panels': verified,
                   'check': 'Selected panels match previously archived raw image hashes and downsampled sheet pixels exactly.'}
        (DEST / 'archive.json').write_text(json.dumps(archive, indent=2) + '\n', encoding='utf-8')

    archive = json.loads((DEST / 'archive.json').read_text(encoding='utf-8'))
    assert fingerprint(source)['sha256'] == archive['copy']['sha256']
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default(size=20)
    canvas = Image.new('RGB', (1248, 1806), '#ededed')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), 'Blender reference', fill='#111111', font=font)
    draw.text((424, 9), 'SH3 | 15k training steps', fill='#111111', font=font)
    draw.text((836, 9), 'HDR-1.0 | 15k training steps', fill='#111111', font=font)
    rectangles = []
    with Image.open(source) as sheet:
        assert sheet.size == (1660, 1806)
        for row, view in enumerate(VIEWS):
            label_y = 38 + row * 442
            draw.text((12, label_y), f'Chrome sphere | test view {view}', fill='#111111', font=font)
            for column, source_x in enumerate((12, 424, 836)):
                box = (source_x, label_y + 30, source_x + 400, label_y + 430)
                canvas.paste(sheet.crop(box), (12 + column * 412, label_y + 30))
                rectangles.append({'view': view, 'model': ['reference', 'SH3', 'HDR-1.0'][column],
                                   'source_rectangle_xyxy': box})
    output = DEST / 'reference-sh3-hdr1-full-views.png'
    canvas.save(output)
    provenance = {'scene': 'justChromeBall', 'split': 'test', 'views': VIEWS,
                  'columns': ['Blender reference', 'SH3', 'HDR-1.0'],
                  'selection': 'Reuse the four original illustration cameras, without ranking scores.',
                  'display': 'Complete image panels copied pixel-for-pixel from saved 400px displays of 800px renders. Labels/layout only; no scene crop, contrast adjustment or retouching.',
                  'panels': rectangles, 'archive': 'archive.json', 'figure': fingerprint(output),
                  'sources': [fingerprint(source), fingerprint(STUDY / 'test/protocol.json'),
                              fingerprint(Path(__file__).resolve())]}
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Saved four matched full test views: reference / SH3 / HDR-1.0. No new evaluation.')


if __name__ == '__main__':
    main()
