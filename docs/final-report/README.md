# Final report reviewed draft

- [Editable report](report.md): reviewed draft, including figures, method equations, literature, paired-view analysis, results, discussion, references, proposed AI declaration and reproduction appendix.
- [Evidence map](evidence-map.md): claim-to-artifact links and figure provenance.
- [Review checklist](review-checklist.md): remaining author decisions and submission gaps.
- [Normal-weight closeout](../experiments/2026-11-07-normal-weight-closeout.md): evaluation-only closure of the completed 3 November trial.
- [Refinement-loss trial](../experiments/2026-11-07-refinement-loss.md): user-requested 5k L1/D-SSIM continuation, validation only; 10.15% lower LPIPS with a PSNR cost and unchanged normal accuracy.
- Independent agent reviews: [grading/argument](reviews/rubric-review.md), [technical accuracy](reviews/technical-review.md), [structure/readability](reviews/editorial-review.md).
- [Review changes and rationale](reviews/revision-summary.md): what was incorporated and what remains an author decision.
- [Course connections](course-connections.md): verified Unit 11 motivation, Unit 10 loss discussion and relevant Units 5, 6, 8, 9 and 12.

The primary evidence is the recorded **100k-surfel / seed-7 / 10k + 5k** study. The main models remain unchanged. After drafting, the user requested one additional 5k loss trial; it is a separate validation-only diagnostic. Later 200k exploration is also separate from the main comparison.

The 11 November revision adds early Materials RGB/SH3 results as initial exploration, defines appearance/footprint/NeRF terminology, makes the surfel count and training budget explicit, replaces Figure 2 with view-count histograms, shows refinement improvements before the preserved failure, simplifies the environment display to one exposure, and shows all four lighting/normal combinations. HDR-0.75 remains clearly labelled as the additional diagnostic model. The Markdown and figures are current; the PDF/HTML were not rebuilt for this revision, as requested.

## Build a PDF and offline reading copy

Use an isolated report environment so document packages do not alter the research environment:

```powershell
py -3.12 -m venv outputs/report-tools
./outputs/report-tools/Scripts/python.exe -m pip install -r docs/final-report/requirements.txt
./outputs/report-tools/Scripts/python.exe docs/final-report/build_report.py
```

Outputs are `outputs/final-report-draft/report.pdf` and `report.html`. The HTML embeds all figures and needs no network connection. The builder uses installed Chrome or Edge; pass `--browser PATH` for another Chromium executable. It verifies all thirteen images load (twelve experiment figures and one labelled method schematic), checks expected PDF text, exports page previews, and records asset hashes/package versions in `build-manifest.json`.

Check numeric tables, links and available evidence hashes independently with `python docs/final-report/verify_report.py` (standard library only).

`python docs/final-report/verify_course_references.py` checks the report's course links and section anchors against the recorded source manifest, offline.

Generated artifacts and document-tool packages stay under ignored `outputs/`. Source prose, archive scripts and actual experiment assets are tracked. Edit `report.md`, then rebuild. Figure captions identify scene, split/view, variants and diagnostic scope.

## Reproduce the study and derived analysis

`python docs/final-report/reproduce_study.py --output outputs/NEW_PLAN` prepares all six historical 100k configs with rebased local paths. Add `--run` only when intentionally launching initialization and training; the default prepares only. This was checked without retraining. It evaluates validation only, leaving the final test decision separate.

`python docs/final-report/analyse_saved_results.py` uses the research environment to regenerate paired descriptive statistics and plot/crop layouts from tracked saved evidence. It runs no model evaluation. Camera variation is not treated as uncertainty over training seeds.

`python docs/final-report/make_revision_figures.py` rebuilds the early Materials, refinement, single-exposure environment and full lighting/normal figures from tracked images and metric archives. It runs no experiments. The optional `--archive-materials` flag is only for checking/copying the original historical local outputs into that archive.

The locally prepared `outputs/final-report-reproduction/checkpoints.zip` contains the seven required files for the three main final models plus original/rebased run metadata, source fingerprints, package records and training logs. `bundle_checkpoints.py` rebuilds it and verifies frozen selection hashes. Its [manifest](../reports/assets/2026-11-07-report-analysis/checkpoint-bundle-manifest.json) is tracked; the author still needs a persistent submission/download location. All 400 sphere images and split manifests are already tracked in Git.

Extract the ZIP into a **fresh clone**, preserving the original workspace's run metadata. Then run Appendix A.1 from that clone's root.

## Review order

The main report now explains the project for readers who have not followed development: background → method → dedicated loss section → setup → results → discussion. Start with Section 4 for the objective choices and Section 6 for their measured consequences. Detailed shader equations/attribution, reproduction, costs and supporting scenes are in Appendices A–D. Add author details, confirm the AI declaration and resolve the `AUTHOR REVIEW` markers before submission.

**Good commit point:** this draft, evidence map, review checklist and the portable normal-weight evidence. Full checkpoints need separate preservation.
