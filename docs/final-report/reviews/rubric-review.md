# Rubric review: strengthening the final report

Reviewed 7 November 2026 against the [course requirements](https://weitsang.github.io/gaussian-splatting-notes/project/), the draft, evidence map and review checklist. Analysis receives 20 marks and execution 15; a new algorithm or publication-quality result is not required. The strongest route is a clear, defensible investigation, not more parameter tuning. No grade prediction follows from this review.

## Highest-priority improvements

1. **Make the argument explicit (Introduction and Discussion).** State two questions: does the complete HDR treatment improve matched reconstruction, and what limits its detail? Add a compact synthesis table: question / experiment / observation / inference / unresolved alternative. This lets the reader see the reasoning rather than reconstruct it from caveats. End Sections 5–6 with direct answers to those questions.

2. **Analyse consistency across cameras, not only means (Section 5.1).** Existing per-view CSVs support matched HDR-minus-SH3 deltas, median and interquartile range, and the fraction of views improving PSNR/LPIPS. A small paired plot would show whether the main gain is broad or concentrated. Label these descriptive variation across views, not training-seed uncertainty; do not bootstrap cameras into an unjustified claim about training robustness. No new test rendering or model selection is needed.

3. **Explain the mechanism (Sections 3.3 and 6.2).** The report says small normal errors matter but does not demonstrate why. A short reflection-vector derivation or clearly labelled illustrative calculation would connect course geometry to the observations. In a coplanar small-angle example, rotating a mirror normal by θ rotates the reflected direction by 2θ; about 1.5° can therefore correspond to about 3°, or four longitude texels in a 512-wide environment. This is explanatory geometry, not the measured average lookup displacement of these runs. Mention that the implemented angular response also changes with the normal.

4. **Extract the interaction from Table 3 (Section 6.2).** Explicitly contrast the normal substitution effect under learned lighting with its effect under reference lighting. For LPIPS it worsens by approximately 0.00578 in the former case but improves by 0.02866 in the latter. This conditional reversal is stronger analysis than saying both inputs matter. It is consistent with coupled fitting, without proving a unique compensation mechanism. The joint-reference case's better structure but lower PSNR also explains why one metric cannot stand in for reflection fidelity.

5. **Show implementation verification (Section 3 or appendix).** Execution currently lists substantial engineering but provides little concrete correctness evidence in the report itself. Add a small validation summary grounded in existing records: finite-difference checks for the relevant surfel/HDR backward paths, checkpoint reload/sidecar checks, and invariants in the reference intervention. Cite actual logs or documented counts; do not imply all final settings underwent a new comprehensive audit. Explain one gradient path in plain language so the author can defend it orally.

6. **Close the reproducibility gap (Appendix A).** A Git link plus ignored checkpoints and machine-specific archived paths is not yet a complete reproduction package. Prepare a portable configuration/launcher and checkpoint bundle, then provide its stable location before submission. The existing honest review marker is appropriate during drafting but must become a usable instruction. The exact course requirement is a repo link with reproduction instructions.

## Tighten rather than expand

- Reduce Section 6.4 to one decision paragraph plus a compact supporting-result table; move the large logo figure/history to the appendix. It explains the sphere scope but is not the main research question.
- Keep related work close to the decisions: Gaussian Surfels and 3DGS-DR as direct influences, RGS-DR as close related work, 2DGS/Ref-NeRF as brief context. Avoid adding papers merely to lengthen the bibliography.
- Consolidate repeated disclaimers about what each model is not. Keep specific causal limits at the relevant result and one final limitations subsection.
- Round most reporting to 2–3 PSNR decimals and 4 LPIPS decimals unless fine differences are being discussed; retain exact CSVs. This should follow the project's verifier update, not hide negative results.
- Explain the measured storage reduction through shared-environment versus per-surfel SH parameterization. Keep allocator/timing details in the appendix and avoid a training-speed claim from overlapping runs.

## Preserve the strongest evidence

![Normal diagnostic worth retaining prominently](../../reports/assets/2026-11-03-reference-normal-swap/normal-swap-highlight.png)

*Actual `justChromeBall` validation view 0. Target / reference lighting with learned normals / same lighting with reference normals. This should remain a central explanatory figure, alongside the full four-case numerical comparison.*

The main matched improvement, the negative normal-weight trial, and the normal/light interaction are enough for a coherent investigation. Extra scenes/seeds strengthen generalisation but are optional unless the supervisor requires them. The author must resolve code attribution and personally revise/defend the interpretation and AI declaration.

**Good commit point:** after the revised argument, paired-result summary and correctness evidence are incorporated and the PDF is rebuilt.
