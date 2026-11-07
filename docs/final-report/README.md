# Final report reviewed draft

- [Editable report](report.md): reviewed draft, including figures, method equations, literature, paired-view analysis, results, discussion, references, proposed AI declaration and reproduction appendix.
- [Evidence map](evidence-map.md): claim-to-artifact links and figure provenance.
- [Review checklist](review-checklist.md): remaining author decisions and submission gaps.
- [Normal-weight closeout](../experiments/2026-11-07-normal-weight-closeout.md): evaluation-only closure of the completed 3 November trial.
- [Refinement-loss trial](../experiments/2026-11-07-refinement-loss.md): user-requested 5k L1/D-SSIM continuation, validation only; 10.15% lower LPIPS with a PSNR cost and unchanged normal accuracy.
- Independent agent reviews: [grading/argument](reviews/rubric-review.md), [technical accuracy](reviews/technical-review.md), [structure/readability](reviews/editorial-review.md).
- [Review changes and rationale](reviews/revision-summary.md): what was incorporated and what remains an author decision.

The primary evidence is the recorded **100k-point / seed-7 / 10k + 5k** study. The main models remain unchanged. After drafting, the user requested one additional 5k loss trial; it is a separate validation-only diagnostic. Later 200k exploration is also separate from the main comparison.

## Build a PDF and offline reading copy

Use an isolated report environment so document packages do not alter the research environment:

```powershell
py -3.12 -m venv outputs/report-tools
./outputs/report-tools/Scripts/python.exe -m pip install -r docs/final-report/requirements.txt
./outputs/report-tools/Scripts/python.exe docs/final-report/build_report.py
```

Outputs are `outputs/final-report-draft/report.pdf` and `report.html`. The HTML embeds all figures and needs no network connection. The builder uses installed Chrome or Edge; pass `--browser PATH` for another Chromium executable. It verifies all ten images load, checks expected PDF text, exports page previews, and records asset hashes/package versions in `build-manifest.json`.

Check numeric tables, links and available evidence hashes independently with `python docs/final-report/verify_report.py` (standard library only).

Generated artifacts and document-tool packages stay under ignored `outputs/`. Source prose, archive scripts and actual experiment assets are tracked. Edit `report.md`, then rebuild. Figure captions identify scene, split/view, variants and diagnostic scope.

## Reproduce the study and derived analysis

`python docs/final-report/reproduce_study.py --output outputs/NEW_PLAN` prepares all six historical 100k configs with rebased local paths. Add `--run` only when intentionally launching initialization and training; the default prepares only. This was checked without retraining. It evaluates validation only, leaving the final test decision separate.

`python docs/final-report/analyse_saved_results.py` uses the research environment to regenerate paired descriptive statistics and plot/crop layouts from tracked saved evidence. It runs no model evaluation. Camera variation is not treated as uncertainty over training seeds.

The locally prepared `outputs/final-report-reproduction/checkpoints.zip` contains the seven required files for the three main final models plus original/rebased run metadata, source fingerprints, package records and training logs. `bundle_checkpoints.py` rebuilds it and verifies frozen selection hashes. Its [manifest](../reports/assets/2026-11-07-report-analysis/checkpoint-bundle-manifest.json) is tracked; the author still needs a persistent submission/download location. All 400 sphere images and split manifests are already tracked in Git.

Extract the ZIP into a **fresh clone**, preserving the original workspace's run metadata. Then run Appendix A.1 from that clone's root.

## Review order

Start with Sections 5–6: verify what the evidence says. Next review the Method attribution table, Introduction/Conclusion and AI declaration. Add author details and resolve every `AUTHOR REVIEW` marker. Confirm full checkpoint/data availability before treating the reproduction appendix as submission-ready.

**Good commit point:** this draft, evidence map, review checklist and the portable normal-weight evidence. Full checkpoints need separate preservation.
