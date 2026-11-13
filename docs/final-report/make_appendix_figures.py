"""Select labelled appendix rows from tracked experimental sheets; no rendering."""
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'docs/reports/assets/2026-11-13-appendix'


def fingerprint(path):
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'docs/reports/assets/2026-11-03-reference-normal-swap/reflection-crops.png'
    output = DEST / 'lighting-normal-views-13-51.png'
    # Original sheet: 38px column heading, then 342px per complete labelled row.
    rectangles = [(0, 0, 1572, 38), (0, 380, 1572, 722), (0, 1406, 1572, 1748)]
    with Image.open(source) as original:
        assert original.size == (1572, 1748)
        canvas = Image.new('RGB', (1572, 722), '#ededed')
        for rectangle, y in zip(rectangles, [0, 38, 380]):
            canvas.paste(original.crop(rectangle), (0, y))
        canvas.save(output)
    provenance = {'scene': 'justChromeBall', 'split': 'validation', 'views': [13, 51],
                  'model': 'HDR-0.75',
                  'columns': ['target', 'learned light / learned normals',
                              'reference light / learned normals', 'learned light / reference normals',
                              'reference light / reference normals'],
                  'selection': 'Additional views from the existing five-camera diagnostic; includes preserved failure view 51.',
                  'source_rectangles_xyxy': rectangles,
                  'display': 'Complete column headings and labelled rows copied pixel-for-pixel; no retouching or rescaling.',
                  'figure': fingerprint(output),
                  'sources': [fingerprint(source), fingerprint(Path(__file__).resolve())]}
    (DEST / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Saved appendix lighting/normal comparisons for validation views 13 and 51.')


if __name__ == '__main__':
    main()
