# First-draft review and remaining gaps

The full prose draft is [report.md](report.md); the claim/figure inventory is [evidence-map.md](evidence-map.md). The main study is frozen. One subsequently user-requested loss trial is preserved separately; no broader method-improvement sweep is queued.

## Essential before submission

- [ ] Add author name/student number and confirm final submission format/portal requirements.
- [ ] Read the research question, scope changes, explanations and conclusions; revise them into wording you can explain and defend.
- [ ] Confirm file-level attribution: assignment starting code, external code copied verbatim (if any), independently implemented paper ideas, and project additions. Add precise citations/license acknowledgements where needed.
- [ ] Confirm the AI-use declaration, add other tools used, and describe checks/research decisions personally performed. The old root `AI_Use.md` is narrower than the assistance used for this draft and needs reconciliation before submission.
- [ ] Upload/attach the prepared `outputs/final-report-reproduction/checkpoints.zip` and add its persistent location. Seven main-model files were checked against frozen selection hashes; all 400 sphere images are already tracked. The portable six-stage launcher was preparation-checked, without retraining.
- [ ] Review figure readability at PDF size. The main test sheet now shows two of the preselected cameras; the appendix uses one normal-map example. Full original sheets remain tracked.
- [ ] Decide whether to shorten supporting Materials/logo history and Appendix B for the final length.
- [ ] Check every final citation and remove all `AUTHOR REVIEW` markers after resolving them.
- [ ] Freeze the final report source and required artifact/bundle versions, then prepare the 20-minute presentation.

## Evidence closure performed for this draft

- [x] Preserve the completed 100k/15k sphere study as the primary result; keep later exploration separate.
- [x] Use the recorded main test means and actual experiment figures.
- [x] Include PSNR/SSIM/LPIPS, foreground quality, storage, timing and measurement limitations.
- [x] Sum both stages for training time; do not mislabel refinement-only time as full training.
- [x] Distinguish 100-view validation, 200-view test and five-view oracle diagnostics.
- [x] Re-evaluate the existing 0.01/0.03 normal-weight checkpoints on validation only and compare reference-normal errors; archive evidence and a matched crop sheet.
- [x] Distinguish normal coherence from reference-normal accuracy; neither smoother maps nor consistency alone establish correctness.
- [x] Disclose historical test inspection and later development after main test visibility.
- [x] Obtain independent grading, technical and editorial reviews; incorporate saved-data analysis, design rationale, gradient/correctness evidence and a question-to-inference synthesis.
- [x] Compute paired-view effects and conditional normal-substitution effects from existing CSVs; no test re-rendering or parameter selection.
- [x] Prepare a locally portable main-checkpoint bundle and historical 100k reproduction configs; no publishing or new training.
- [x] Complete the later user-requested L1/D-SSIM refinement trial: one 5k continuation, 100-view validation and five-view normals, actual crops and a separately labelled report result. No test evaluation.
- [x] Rewrite for interested peers, define terms before using them, add dedicated loss explanations and cite verified course material (especially Units 10–11). Preserve result tables and move detailed math/metadata to appendices.
- [x] Address 11 November author comments: add clearly separated early Materials RGB/SH3 results and actual images; clarify terminology/count/budget; replace cumulative plots with histograms; lead with visible refinement improvements while retaining the failure; use one environment display exposure; show all four lighting/normal cases and identify the HDR-0.75 diagnostic. Verify source/figure evidence without rebuilding the PDF.

## Optional strengthening; not required to finish this scoped draft

- Additional seeds or another controlled reflective scene.
- Independent controls for HDR, angular response, fixed final geometry, and forward/deferred evaluation.
- A controlled full-training timing benchmark if claiming training-speed differences.

Run these only if they answer a specific necessary report claim or the supervisor requires them. Perfecting the method is not a prerequisite for reporting the measured improvement and limitations.

**Good commit point:** report source, evidence map, build/verification scripts, corrected status notes, and portable normal-weight closeout. Preserve full checkpoints separately. The generated PDF/HTML live in ignored `outputs/` and can be rebuilt.
