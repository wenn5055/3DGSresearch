# Editorial review of the first report draft

Reviewed 7 November 2026 against `report.md` and the 16-page PDF. No report source was changed by this reviewer.

## Overall recommendation

The draft already contains a credible study. Improve its analytical focus and reader guidance rather than making it longer or adding a catalogue of experiments. The strongest narrative is: **HDR improves image fitting; normal/light interaction still limits sharp detail; making normals more coherent does not necessarily make them more accurate.**

## Priority 1: turn the discussion into explicit answers

- **Introduction and Sections 5–6:** state three analytical questions: (1) does the complete HDR treatment improve the matched SH3 comparison? (2) what do normal/light interventions reveal about the remaining error? (3) does stronger normal–depth consistency improve normal accuracy? These are an organising device, not a claim that all questions were preregistered.
- **Section 6.2:** interpret Table 3 through conditional effects. Correcting normals helps substantially with reference light but worsens scores with learned light. That asymmetry is more revealing than repeating the four absolute scores. Give the relevant deltas and explicitly explain the learned pair's compensation. Reserve claims of unique causal attribution because materials remain fitted and fixed.
- **Section 6.3:** elevate the coherence/accuracy divergence into a clear finding. Explain what coherence measures and why two learned quantities can agree while both remain biased. The negative trial is useful reasoning, not a disappointing aside.
- **Section 5.1:** add one compact paired-view summary or plot from existing CSVs, such as median/range of per-view improvement and improved-view counts. Describe variation across cameras without treating cameras as independent training seeds. No retraining is needed.

## Priority 2: make implementation work assessable

- **Section 3:** keep the current core equations, but briefly explain *why* each design was chosen: planar normals, per-pixel rather than per-splat evaluation, staged fitting, and angular response. Correct equations alone do not show design judgment.
- Add a short **implementation verification** paragraph: what gradients were checked, one nontrivial failure mode caught, and what reload/consistency checks validate. Cite the actual recorded checks; do not imply every old test was rerun for the report.
- The attribution table is worth keeping. Resolve its author-review marker before submission. Distinguish inherited code, independently implemented ideas and copied code explicitly.
- An optional small schematic of the actual pipeline could replace some buffer prose. Label it a method schematic; keep experimental images as actual renders.

## Priority 3: cut repetition and peripheral history

- State the revised scope and the main comparison boundary once in the Introduction/Setup. Remove repeated references to newer 200k defaults from the report body; retain them in reproducibility notes only.
- Consolidate repeated “not a full implementation / not a physical model / not a universal claim” wording. Keep precise comparison limits in Setup and one final limitations subsection. Positive descriptions of what the implementation does should carry most paragraphs.
- **Section 6.4:** keep the Materials result and metric disagreement in approximately one paragraph. Move the logo learning-rate detail and Figure 7 to an appendix, preserving the failure and its role in scene simplification. This is useful supporting evidence, but it competes with the main diagnostic story.
- **Section 2:** the current short literature review is sufficient. Do not add NeRO/GGX material unless actually needed to explain an implemented choice or limitation.
- Use fewer decimal places in narrative prose; reserve detailed precision for tables/evidence CSVs.

## Figures and layout

- Figure 2 and the normal/light intervention are the core visual evidence. Keep them prominent and discuss particular visible structures rather than merely saying “blur remains.”
- Figure 1's four small spheres add little beyond establishing scene simplicity. Reduce it, or combine one full scene view with a central reflection crop.
- Retain the preserved refinement regression; shorten its discussion rather than hiding it.
- PDF pages 6 and 15 have substantial unused space. Adjust figure placement/page breaks after substantive cuts. Aim for roughly 11–13 pages including appendices if feasible, without shrinking text or crops.
- A short main-text normal/error example may serve readers better than the entire five-row Appendix B sheet. Keep the complete sheet as supplementary evidence.

![Core diagnostic worth foregrounding](../../reports/assets/2026-11-03-reference-normal-swap/normal-swap-highlight.png)

*Existing actual diagnostic: `justChromeBall`, validation view 0. Target / reference light with learned normals / reference light with reference normals. This directly supports the report's central interpretation.*

## Suggested framing

Shorter title: **Reconstructing Specular Reflections with Deferred Gaussian Surfels**.

Trim the abstract to about 160–180 words: problem → implemented comparison → +4.31 dB and 35.1% LPIPS reduction → unresolved detail → normal/light interaction and failed stronger-consistency trial. End on the practical lesson: image fit and self-consistency do not establish accurate geometry/light recovery.

**Good commit point:** revised argument, one saved-data analysis, resolved attribution, and a rebuilt PDF with readable figures. More adjectives or broader claims will not improve this report.
