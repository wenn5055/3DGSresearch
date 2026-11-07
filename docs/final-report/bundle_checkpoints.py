"""Package existing main checkpoints for sharing, without modifying originals.

Only data/output paths in packaged run metadata are rebased for commands run
from the repository root. Original run.json is retained beside each copy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / 'docs/reports/assets/2026-11-07-report-analysis'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/final-report-reproduction/checkpoints.zip')
    args = parser.parse_args()
    selection = json.loads((ROOT / 'docs/reports/assets/2026-11-03-sphere-study/selection.json').read_text())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    readme = ('Extract this archive into the root of a FRESH CLONE, preserving original workspace run metadata.\n'
              'All 400 sphere images and manifests are tracked in Git.\n'
              'Run the Appendix A.1 evaluation command from the repository root. Packaged data/output paths are relative.\n'
              'Original metadata is preserved as original-run.json; model bytes match the frozen selection hashes.\n'
              'Use repository code/evidence commit 59c553f plus the report helpers. No training is needed to evaluate.\n')
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        archive.writestr('CHECKPOINTS-README.txt', readme)
        for group in selection['selected_checkpoints']:
            run_name = Path(group[0]['path']).parent.name
            run = ROOT / 'outputs' / run_name
            for entry in group:
                source = run / Path(entry['path']).name
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                assert digest == entry['sha256'], source
                archive_name = source.relative_to(ROOT).as_posix()
                archive.write(source, archive_name)
                files.append({'path': archive_name, 'bytes': source.stat().st_size, 'sha256': digest})
            metadata = json.loads((run / 'run.json').read_text(encoding='utf-8'))
            archive.write(run / 'run.json', f'outputs/{run_name}/original-run.json')
            metadata['config']['paths']['data'] = 'data/justChromeBall'
            metadata['config']['paths']['output'] = f'outputs/{run_name}/model.png'
            metadata['report_bundle_portability'] = 'Only config data/output paths rebased; original-run.json is exact source.'
            archive.writestr(f'outputs/{run_name}/run.json', json.dumps(metadata, indent=2) + '\n')
            for name in ('source-files.json', 'train.log', 'packages.json'):
                archive.write(run / name, f'outputs/{run_name}/{name}')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for entry in files:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest() == entry['sha256']
    manifest = {'status': 'prepared-locally-not-published', 'bundle_name': output.name,
                'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'model_files': files, 'dataset': '400 tracked justChromeBall images plus tracked split manifests',
                'metadata_rebasing': 'Packaged run.json data/output paths only; original metadata included.',
                'publication': 'Author must provide the persistent download/submission location.'}
    ASSET.mkdir(parents=True, exist_ok=True)
    (ASSET / 'checkpoint-bundle-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(output), 'bytes': manifest['bytes'], 'verified_model_files': len(files)}, indent=2))


if __name__ == '__main__':
    main()
