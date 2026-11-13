"""Build an offline HTML/PDF draft from report.md; check assets and PDF layout.

No experiments are executed. Uses an installed Chrome/Edge through Playwright.
Local figure bytes are embedded so the HTML is independently readable.
"""
import argparse
import base64
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import re

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CSS = """
@page { size: A4; margin: 18mm 18mm 19mm; }
* { box-sizing: border-box; }
html { color: #17202a; background: #e9edf0; }
body { max-width: 850px; margin: 28px auto; padding: 48px 58px;
  background: white; font: 15px/1.55 Georgia, 'Times New Roman', serif; }
h1,h2,h3 { font-family: Calibri, Arial, sans-serif; line-height: 1.2; color: #162f43; }
h1 { font-size: 28px; margin: 0 0 22px; }
h2 { font-size: 21px; margin: 29px 0 12px; border-bottom: 1px solid #ccd5dc; padding-bottom: 6px; }
h3 { font-size: 17px; margin: 20px 0 10px; }
p { margin: 10px 0; orphans: 3; widows: 3; }
a { color: #145c83; text-decoration: none; overflow-wrap: anywhere; }
blockquote { margin: 16px 0; padding: 10px 15px; border-left: 3px solid #819db0; background: #f3f6f8; font-size: .92em; }
table { width: 100%; border-collapse: collapse; margin: 12px 0 17px; font: 12px/1.4 Calibri, Arial, sans-serif; }
table { break-inside: avoid; }
th,td { border-bottom: 1px solid #d9e0e5; text-align: left; padding: 7px 6px; vertical-align: top; overflow-wrap: anywhere; }
th { background: #edf2f5; color: #162f43; }
tr { break-inside: avoid; }
figure { margin: 19px 0; break-inside: avoid; }
figure img { width: 100%; height: auto; max-height: 650px; object-fit: contain; display: block; }
figcaption { margin-top: 8px; font: 12px/1.42 Calibri, Arial, sans-serif; color: #344654; }
.equation { text-align: center; font-family: Cambria, 'Times New Roman', serif; padding: 10px 4px; margin: 12px 0; background: #f7f9fa; break-inside: avoid; }
code { font: .82em Consolas, monospace; overflow-wrap: anywhere; }
pre { padding: 10px; background: #f3f5f7; white-space: pre-wrap; font-size: 11px; break-inside: auto; }
.review { color: #823b10; background: #fff4df; }
.screen-tools { font: 13px Calibri, Arial, sans-serif; background: #eaf1f5; padding: 10px; margin-bottom: 20px; }
@media print {
  html { background: white; }
  body { max-width: none; margin: 0; padding: 0; font-size: 10.6pt; line-height: 1.42; }
  h1 { font-size: 22pt; }
  h2 { font-size: 15pt; margin-top: 18pt; }
  h3 { font-size: 11.8pt; margin-top: 13pt; }
  h1,h2,h3 { break-after: avoid; }
  p { break-inside: avoid; }
  p:has(+ .equation) { break-after: avoid; }
  table { font-size: 8.4pt; }
  th,td { padding: 5px; }
  thead { display: table-header-group; }
  figure img { max-height: 175mm; }
  figure[data-figure="D1"] img { max-height: 85mm; }
  figcaption { font-size: 8.4pt; }
  pre { font-size: 8pt; }
  .screen-tools { display: none; }
}
"""


def browser_path(requested):
    if requested:
        return Path(requested).resolve(strict=True)
    candidates = []
    for env_name in ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA'):
        base = Path(os.environ.get(env_name, 'C:/'))
        candidates += [base / 'Google/Chrome/Application/chrome.exe',
                       base / 'Microsoft/Edge/Application/msedge.exe']
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Provide --browser with the installed Chrome/Chromium/Edge executable.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/final-report-draft')
    parser.add_argument('--browser')
    args = parser.parse_args()
    source = HERE / 'report.md'
    text = source.read_text(encoding='utf-8')
    if '[BUILD:' in text:
        raise ValueError('Unresolved BUILD marker in report source.')
    body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'], output_format='html')
    assets = []

    def embed(match):
        relative = html.unescape(match.group(1))
        path = (HERE / relative).resolve(strict=True)
        path.relative_to(ROOT)
        data = path.read_bytes()
        assets.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
                       'sha256': hashlib.sha256(data).hexdigest()})
        mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml'}[path.suffix]
        return 'src="data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii') + '"'

    body = re.sub(r'src="([^"]+)"', embed, body)
    def figure(match):
        number = re.match(r'Figure ([A-Za-z0-9]+)\.', match.group(2))
        tag = number.group(1) if number else ''
        return f'<figure data-figure="{tag}">{match.group(1)}<figcaption>{match.group(2)}</figcaption></figure>'

    body = re.sub(r'<p>(<img\b[^>]*>)</p>\s*<p><em>(Figure.*?)</em></p>',
                  figure, body, flags=re.S)
    body = re.sub(r'(\[AUTHOR REVIEW:.*?\])', r'<span class="review">\1</span>', body, flags=re.S)
    title = html.escape(text.splitlines()[0].lstrip('# '))
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{title}</title><style>{CSS}</style></head><body>'
                '<div class="screen-tools">CP4282 · Reviewed draft · 7 November 2026 · '
                'Offline reading copy. Edit docs/final-report/report.md and rebuild.</div>'
                + body + '</body></html>')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / 'report.html'
    pdf_path = output / 'report.pdf'
    html_path.write_text(document, encoding='utf-8')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(browser_path(args.browser)), headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 960}, device_scale_factor=1)
        page.goto(html_path.as_uri(), wait_until='load')
        page.evaluate('async () => { await document.fonts.ready; await Promise.all([...document.images].map(i => i.decode())); }')
        image_state = page.evaluate('() => [...document.images].map(i => ({loaded:i.complete && i.naturalWidth>0, alt:i.alt}))')
        if not all(item['loaded'] for item in image_state):
            raise RuntimeError(f'Broken image: {image_state}')
        overflow = page.evaluate('() => document.documentElement.scrollWidth > innerWidth')
        if overflow:
            raise RuntimeError('HTML horizontal overflow.')
        page.screenshot(path=str(output / 'preview.png'), full_page=False)
        page.pdf(path=str(pdf_path), format='A4', print_background=True, prefer_css_page_size=True,
                 display_header_footer=True, header_template='<span></span>',
                 footer_template='<div style="font:8px Arial;width:100%;text-align:center;color:#65727c;">'
                                 '<span class="pageNumber"></span></div>')
        browser.close()
    import pymupdf
    pdf = pymupdf.open(pdf_path)
    page_records = []
    for index, pdf_page in enumerate(pdf):
        page_text = pdf_page.get_text()
        if not page_text.strip():
            raise RuntimeError(f'Blank PDF page {index + 1}')
        page_records.append({'page': index + 1, 'characters': len(page_text),
                             'images': len(pdf_page.get_images()), 'first_text': page_text[:100]})
        pdf_page.get_pixmap(matrix=pymupdf.Matrix(0.85, 0.85)).save(output / f'page-{index+1:02}.png')
    pdf_text = '\n'.join(p.get_text() for p in pdf)
    for needle in ('24.5732', '0.083517', '1.5943', 'AI-use declaration', 'Reproduce the results'):
        if needle not in pdf_text:
            raise RuntimeError(f'Missing expected PDF text: {needle}')
    (output / 'report-text.txt').write_text(pdf_text, encoding='utf-8')
    for start in range(0, len(pdf), 8):
        contact = pymupdf.open()
        contact_page = contact.new_page(width=720, height=530)
        for local, number in enumerate(range(start, min(start + 8, len(pdf)))):
            x, y = (local % 4) * 180, (local // 4) * 265
            contact_page.insert_text((x + 5, y + 12), f'Page {number + 1}', fontsize=9)
            contact_page.insert_image(pymupdf.Rect(x + 3, y + 18, x + 177, y + 263),
                                      filename=str(output / f'page-{number+1:02}.png'))
        contact_page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(output / f'contact-{start//8+1}.png')
        contact.close()
    manifest = {'source': source.relative_to(ROOT).as_posix(),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'html_sha256': hashlib.sha256(html_path.read_bytes()).hexdigest(),
                'pdf_sha256': hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                'pages': page_records, 'embedded_images': assets,
                'author_review_markers': text.count('[AUTHOR REVIEW:'),
                'build_packages': {name: importlib.metadata.version(name)
                                   for name in ('Markdown', 'playwright', 'PyMuPDF')}}
    (output / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pdf': str(pdf_path), 'pages': len(pdf), 'embedded_images': len(assets),
                      'author_review_markers': manifest['author_review_markers']}, indent=2))


if __name__ == '__main__':
    main()
