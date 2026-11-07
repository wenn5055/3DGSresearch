# Review changes and rationale

Three independent agents reviewed the first draft on 7 November: [grading and argument](rubric-review.md), [technical accuracy](technical-review.md), and [editorial structure](editorial-review.md). The root agent checked their recommendations against the code and saved evidence before revising the report. No grade is predicted by these reviews.

## Incorporated

- **Stronger analytical structure:** Introduction names the main comparison and two diagnostic questions. Section 6.4 maps each question to its evidence, supported inference and remaining explanation.
- **Paired-view evidence:** HDR-1.0 improves PSNR, SSIM, LPIPS and foreground PSNR on 200/200 saved test cameras. Median PSNR gain is 4.4439 dB, range 2.9349–5.5425. The smaller-size gain is weaker and nonuniform: PSNR 198/200, LPIPS 189/200. A descriptive effect-distribution plot replaces reliance on means alone. This is saved-data analysis, not another test evaluation or independent-seed confidence estimate.
- **Conditional normal/light interpretation:** Section 6.2 quantifies the reversal in normal-substitution effects. LPIPS worsens by 0.005782 under learned light but improves by 0.028657 under reference light. The report explains what this supports and why it does not uniquely separate material compensation, sampling or shader differences.
- **Mechanism and design judgment:** Section 3 explains the shared composited-normal gradient and why scene-material inspection motivated positive HDR lighting plus angular modulation. The illustrative mirror-angle calculation is explicitly separate from measured lookup errors.
- **Implementation evidence:** Added recorded finite-difference, alpha/display-order, checkpoint reload and intervention-invariant checks. It does not claim they were newly rerun for this revision.
- **Coherence versus accuracy:** Defined coherence as ‖N‖/A and distinguished contributor alignment from spatial smoothness and reference-normal correctness.
- **Less peripheral main-text history:** Moved detailed Materials/logo discussion and the preserved failure image to Appendix B. Main crops use two preselected views and a compact actual normal-map example; original full sheets remain tracked.
- **Concrete reproducibility:** Prepared a 35.23 MB local checkpoint ZIP whose seven model files match the frozen selection hashes. Packaged metadata uses relative dataset/output paths and keeps originals. Added a six-stage historical 100k launcher, tested in preparation-only mode. It does not inherit current 200k defaults or launch test evaluation.

The technical reviewer found no central table/equation errors or camera-index/mask unfairness. The revisions therefore strengthen explanation, analysis and reproducibility rather than changing measured conclusions.

## Deliberately not added

No broad training sweep, extra paper catalogue, unsupported novelty claim, or significance claim based on correlated cameras. Additional seeds/scenes are explicitly future work unless required by the supervisor. Full training reproduction and public release were not performed in this review pass.

## Author work still required

Confirm identity, exact assignment/project/copied-code boundaries and personal AI-use wording. Upload or attach the prepared checkpoint ZIP, add its persistent location, and freeze the final source revision. Read and revise the explanations so they can be defended in the presentation. See the [review checklist](../review-checklist.md).

![Central matched normal diagnostic](../../reports/assets/2026-11-03-reference-normal-swap/normal-swap-highlight.png)

*Actual `justChromeBall`, validation view 0: target / reference lighting with learned normals / reference lighting with reference normals. The revised discussion uses this evidence with the full four-case numerical comparison.*

**Good commit point:** the reviewed report, reviewer memos, derived-analysis assets, reproducibility helpers and updated evidence map. Keep the generated PDF and checkpoint ZIP separately from Git.
