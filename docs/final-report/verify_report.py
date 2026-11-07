"""Check first-draft table values, local links and portable evidence integrity.

Uses only Python's standard library. Does not run training or image evaluation.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / 'docs/reports/assets'


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main():
    text = (HERE / 'report.md').read_text(encoding='utf-8')
    assert '[BUILD:' not in text
    checked = []
    for row in csv_rows(ASSETS / '2026-11-03-sphere-study/test/summary.csv'):
        name = {'SH3': 'SH3', 'HDR-1.0': 'HDR-1.0, matched primary',
                'HDR-0.75': 'HDR-0.75, additional variant'}[row['model']]
        expected = (f"| {name} | {float(row['psnr_db']):.4f} | {float(row['ssim']):.6f} | "
                    f"{float(row['lpips_alex']):.6f} | {float(row['foreground_psnr_db']):.4f} |")
        assert expected in text, expected
        checked.append('test:' + row['model'])
    for row in csv_rows(ASSETS / '2026-11-03-reference-normal-swap/summary.csv'):
        lighting = 'Learned' if row['lighting'] == 'learned-light' else 'Reference'
        normals = 'Learned' if row['normals'] == 'learned-normal' else 'Reference'
        expected = (f"| {lighting} | {normals} | {float(row['psnr_db']):.4f} | "
                    f"{float(row['ssim']):.6f} | {float(row['lpips_alex']):.6f} |")
        assert expected in text, expected
        checked.append('oracle:' + lighting + '/' + normals)
    normal_asset = ASSETS / '2026-11-07-normal-weight-closeout'
    provenance = json.loads((normal_asset / 'provenance.json').read_text(encoding='utf-8'))
    for row in csv_rows(normal_asset / 'validation-summary.csv'):
        label = row['model']
        weight = {'normal001': '0.01', 'normal003': '0.03'}[label]
        error = provenance['five_view_normal_means'][label]['mesh_normal_mae_deg']
        expected = (f"| {weight} | {float(row['psnr_db']):.4f} | {float(row['ssim']):.6f} | "
                    f"{float(row['lpips_alex']):.6f} | {error:.4f} |")
        assert expected in text, expected
        checked.append('normal-weight:' + label)
    loss_asset = ASSETS / '2026-11-07-refinement-loss'
    loss_provenance = json.loads((loss_asset / 'provenance.json').read_text(encoding='utf-8'))
    assert loss_provenance['status'] == 'completed' and not loss_provenance['test_evaluation']
    assert loss_provenance['protected_checkpoints_unchanged']
    for row in csv_rows(loss_asset / 'validation-summary.csv'):
        label = {'MSE': 'MSE', 'L1-DSSIM': 'L1 + D-SSIM'}[row['model']]
        error = loss_provenance['five_view_normal_means'][row['model']]['mesh_normal_mae_deg']
        expected = (f"| {label} | {float(row['psnr_db']):.4f} | {float(row['ssim']):.6f} | "
                    f"{float(row['lpips_alex']):.6f} | {error:.4f} |")
        assert expected in text, expected
        checked.append('refinement-loss:' + label)
    assert loss_provenance['improved_views_out_of_100']['lpips_alex'] == 100
    assert len(csv_rows(loss_asset / 'validation-metrics.csv')) == 200
    assert len(csv_rows(loss_asset / 'normal-diagnostics.csv')) == 10
    for item in loss_provenance['protected_checkpoints']:
        source = ROOT / item['path']
        if source.is_file():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    stage_pairs = [
        ('SH3', 'Nov03-1354-justChromeBall-frozen-sh3-10k', 'Nov03-1357-justChromeBall-staged-sh3-refine-5k'),
        ('HDR-1.0', 'Nov02-2021-justChromeBall-frozen-hdr-10k', 'Nov02-2023-justChromeBall-staged-hdr-refine-5k'),
        ('HDR-0.75', 'Nov03-1400-justChromeBall-frozen-hdr-10k', 'Nov03-1402-justChromeBall-staged-hdr-refine-5k')]
    performance = json.loads((ASSETS / '2026-11-03-sphere-study/test/performance.json').read_text())
    for label, frozen, refined in stage_pairs:
        times = [json.loads((ASSETS / '2026-11-03-sphere-study' / name / 'run.json').read_text())
                 ['training_result']['trainer_seconds'] for name in (frozen, refined)]
        row = next(r for r in performance if r['model'] == label)
        # Both stage logs were checked during drafting; the refinement peak is larger.
        expected = (f"| {label} | {row['checkpoint_bytes']/1e6:.2f} | {row['fps']:.2f} | "
                    f"{sum(times):.2f} | {row['training_warp_pool_high_bytes']/1e9:.3f} |")
        assert expected in text, expected
        checked.append('cost:' + label)
    local_links = 0
    documents = list(HERE.glob('*.md')) + [ROOT / 'docs/experiments/2026-11-07-normal-weight-closeout.md',
                                         ROOT / 'docs/experiments/2026-11-07-refinement-loss.md']
    for document in documents:
        value = document.read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)', value):
            if re.match(r'\w+://', target) or target.startswith('#'):
                continue
            path = (document.parent / target.split('#')[0]).resolve()
            assert path.exists(), f'{document.name}: missing link {target}'
            local_links += 1
    assert len(re.findall(r'^!\[', text, flags=re.M)) == 11
    analysis_path = ASSETS / '2026-11-07-report-analysis/analysis.json'
    analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
    main_effect = analysis['comparisons']['HDR-1.0_vs_SH3']
    assert all(r['improved'] == 200 for r in main_effect.values())
    assert f"median {main_effect['psnr_db']['median']:.4f} dB" in text
    assert f"{main_effect['psnr_db']['min']:.4f}–{main_effect['psnr_db']['max']:.4f} dB" in text
    assert '198/200' in text and '189/200' in text
    conditional = analysis['normal_intervention']['normal_substitution_effect']
    assert f"{abs(conditional['learned-light']['lpips_alex']['mean']):.4f}" in text
    assert f"{conditional['reference-light']['lpips_alex']['mean']:.4f}" in text
    for source in analysis['sources']:
        assert hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() == source['sha256']
    assert len(csv_rows(normal_asset / 'validation-metrics.csv')) == 200
    assert len(csv_rows(normal_asset / 'normal-diagnostics.csv')) == 10
    checked_copies = 0
    for item in provenance['sources']:
        source = ROOT / item['path']
        # Local input archives may be absent on another clone; check every one available.
        if source.is_file():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256'], item['path']
            checked_copies += 1
    print(json.dumps({'table_rows_checked': len(checked), 'local_links_checked': local_links,
                      'available_original_input_hashes_checked': checked_copies,
                      'report_figures': 11, 'saved_data_analysis_sources_checked': len(analysis['sources']),
                      'loss_trial_paired_validation_rows': 200, 'loss_trial_normal_rows': 10,
                      'validation_rows': 200, 'normal_rows': 10,
                      'author_review_items': text.count('[AUTHOR REVIEW:')}, indent=2))


if __name__ == '__main__':
    main()
