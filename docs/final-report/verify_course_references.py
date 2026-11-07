"""Verify report course links against recorded primary-page headings.

--capture builds the compact manifest from the downloaded HTML/text snapshots.
Normal verification is offline and uses the tracked manifest, not the network.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / 'course-references.json'
SNAPSHOTS = ROOT / 'outputs/Nov07-report-course-notes'


def course_links():
    report = (HERE / 'report.md').read_text(encoding='utf-8')
    links = re.findall(r'https://weitsang\.github\.io/gaussian-splatting-notes/unit-\d+/[^)\s]*', report)
    return sorted(set(links))


def capture(links):
    records = []
    for unit in (5, 6, 8, 9, 10, 11, 12):
        path = SNAPSHOTS / f'unit-{unit}.html'
        source = path.read_text(encoding='utf-8')
        headings = {}
        for anchor, raw in re.findall(r'<h[1-3][^>]*id="([^"]+)"[^>]*>(.*?)</h[1-3]>', source, re.S):
            headings[anchor] = html.unescape(re.sub(r'<[^>]+>', '', raw)).replace('¶', '').strip()
        checked = {urlparse(link).fragment for link in links
                   if urlparse(link).path.endswith(f'/unit-{unit}/') and urlparse(link).fragment}
        assert checked <= set(headings), (unit, checked - set(headings))
        title = next(title for anchor, title in headings.items() if anchor.startswith(f'unit-{unit}-'))
        records.append({'unit': unit, 'title': title,
            'url': f'https://weitsang.github.io/gaussian-splatting-notes/unit-{unit}/',
            'html_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'text_sha256': hashlib.sha256((SNAPSHOTS / f'unit-{unit}.txt').read_bytes()).hexdigest(),
            'checked_sections': {anchor: headings[anchor] for anchor in sorted(checked)}})
    manifest = {'retrieved_date_singapore': '2026-11-07',
                'retrieval': 'Direct HTTPS GET; all seven pages returned HTTP 200. Web search returned no results.',
                'local_snapshots': 'outputs/Nov07-report-course-notes (ignored, complete primary pages)',
                'scope': 'Course concepts and links; experiment results come from repository evidence.',
                'pages': records}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    links = course_links()
    if args.capture:
        capture(links)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    pages = {item['url']: item for item in manifest['pages']}
    anchors = 0
    for link in links:
        parsed = urlparse(link)
        page_url = parsed._replace(fragment='').geturl()
        assert page_url in pages, link
        if parsed.fragment:
            assert parsed.fragment in pages[page_url]['checked_sections'], link
            anchors += 1
    print(json.dumps({'course_pages': len(pages), 'unique_report_course_links': len(links),
                      'section_anchors_verified': anchors, 'offline_verification': True}, indent=2))


if __name__ == '__main__':
    main()
