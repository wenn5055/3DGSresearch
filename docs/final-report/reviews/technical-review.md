# Technical review of the first report draft

Reviewed 7 November 2026 against shader/loss/evaluator code, archived configs and portable CSV/protocol evidence. No training or rendering was launched. The central numerical tables and shader equations are consistent with the implementation. The draft already handles the largest confounds responsibly: SH3 is a surfel control, refinement adds optimization steps, the main treatment combines multiple appearance changes, oracle diagnostics are evaluation-only, and later exploration does not replace the recorded main test.

## Priority 1: show the reasoning behind the implementation

1. **Explain why HDR/angular response was introduced (Section 3.3).** The current method appears arbitrary. The [angular-response record](../../experiments/2026-10-30-angular-response.md) documents read-only inspection of a Facing → ColorRamp → glossy-color graph. Its artistic grazing response and HDR illumination motivated positive linear-light lighting plus a learned angular modulation. State that this inspection informed the model family, while reference ramp values/environment pixels/normals were not used as optimization targets. This is more precise than suggesting generic material recovery from RGB alone.

2. **Add implementation-correctness evidence (end of Section 3).** The report describes substantial differentiable code but supplies no concrete verification beyond the AI declaration. A compact paragraph should name finite-difference checks of shader/normal/ray-plane paths, the nonconstant angular-response case, and checkpoint-sidecar reload checks. The dated record reports six HDR tests on CPU/CUDA, 22 finite-difference cases per device, 12 existing CUDA trainer regressions and an 800px save/reload integration smoke. Label these as recorded checks, not new runs; cite the test files and evidence. This directly supports execution quality without more experiments.

3. **Explain the deferred normal-gradient mechanism.** Write N = Σwᵢnᵢ and n = N/||N||, so the normal-only derivative is proportional to wᵢ(I − nnᵀ)/||N||. It couples contributors to the same pixel's shading objective; other weight/position paths also exist. Avoid claiming deferred shading guarantees correct or globally smoother normals. The current normal-propagation-disabled final experiment tests the complete representation/appearance treatment, not the published propagation mechanism.

## Priority 2: extract stronger analysis from existing CSVs

4. **Add paired improvement distributions, not more training.** On the 200 test cameras HDR-1.0 beats SH3 on PSNR, SSIM, LPIPS and foreground PSNR in all 200 views. PSNR gain median is 4.4439 dB, range 2.9349–5.5425; mean gain is 4.3077. The additional size-0.75 variant improves PSNR on 198/200 and LPIPS on 189/200. These distributions show a broad main benefit and a smaller, nonuniform size benefit. Camera samples are correlated and do not replace multi-seed uncertainty; do not attach training-population significance claims.

5. **Make the oracle interaction explicit.** Correcting normals changes mean LPIPS by +0.005782 under learned lighting (worse) versus −0.028657 under reference lighting (better). The difference-in-differences is −0.034439 and has the same sign in all five views. Present conditional effects as descriptive evidence of compensation/interaction, not a unique causal decomposition or an oracle performance upper bound. The substitution changes both reflection direction and angular response, which the current draft correctly discloses.

6. **Define normal coherence.** The 0.998552 → 0.998927 statistic is ||N||/A, measuring alignment among weighted contributors. It is not spatial smoothness or reference accuracy. The mean reference-normal error rises by only 0.02146°; describe this as a small negative observation under one seed, not evidence of a generally harmful regularizer.

## Precision and scope checks

- **Normal cameras/masks:** no camera-index mismatch found. Footprint diagnostics subset the camera tensor and use local indices; the oracle uses the full tensor and original indices. Both use reference alpha ≥0.95 and predicted coverage ≥0.5. Different normal-length guards are immaterial for these covered pixels. Their 1.5728° and 1.4678° means belong to different size-1.0/0.75 models and must remain separate.
- **Timing:** the 172.92 s HDR-1.0 sum excludes the recorded 68.79 s initializer, evaluation and startup. Keep the existing label; optionally give the initializer separately. The 4.9% observed FPS difference is not strong evidence of a shader speed advantage.
- **Storage:** measured file totals are sound. Explain that common geometry plus compressed sidecars makes the 63.7% saving format-specific; it is not simply 48 SH numbers versus four material values.
- **Reproducibility:** exact copied-code attribution and a usable checkpoint/data bundle remain essential author tasks. Machine-specific archived paths require rewriting; current 200k defaults cannot reproduce the historical experiment directly.

![Evidence supporting the diagnostic discussion](../../reports/assets/2026-11-03-reference-normal-swap/normal-swap-highlight.png)

*Existing actual experiment: `justChromeBall`, validation view 0; target / reference light with learned normals / reference light with reference normals. Keep this matched comparison in the main discussion. Extensive logo history and extra normal maps can move to appendices if shortening.*

**Recommendation:** strengthen mechanism, design rationale and paired analysis first. No broad new experiment campaign is necessary to make this scoped report technically defensible.
