# Chrome material response and optional shader experiment

Started 30 October, 16:53 Singapore. This is a model-capacity hypothesis, not a
positive rendering result. Original Materials remains the primary benchmark.

## Evidence

The completed original hull pair gives surfel SH3 27.7921 / 0.942775 / 0.037801
PSNR / SSIM / LPIPS, versus deferred 27.7503 / 0.942333 / 0.038428. Both use the
same training-only initializer, 800px, 1M capacity, 30k iterations and seed 7.
Report: `outputs/Oct30-1615-materials-initializer-controls-val/index.html`.

Read-only inspection of the user-owned chrome scene found a linked Layer Weight
Facing -> ColorRamp -> Glossy Color graph. The ramp spans linear RGB 0.8 to
0.0308957, with EASE interpolation and Layer Weight blend 0.7. Socket defaults
alone therefore misdescribe the rendered material. Full graph, scene hash and
inspection command are retained in
`outputs/first-principles-2026-10-30/chrome-response.json` and its controller log.

The official [Cycles implementation](https://github.com/blender/blender/blob/main/intern/cycles/kernel/svm/fresnel.h)
shows that Facing increases toward grazing incidence. This scene's artistic ramp
darkens there; imposing an ordinary increasing Schlick term would be inappropriate.
The existing constant per-surfel blend and bounded LDR environment cannot directly
express a shared angular modulation of high-frequency HDR illumination. SH can
absorb some residual but does not make that multiplicative model exact.

## Proposed separate variant

Learn a small shared angular lookup table from training RGB, use positive HDR
environment radiance and perform foreground shading in linear light, then apply
the dataset's Standard display transfer before white alpha composition. Do not
copy Blender ramp parameters, ground-truth normals, mesh geometry or environment
textures into training. Keep the current controls and default shader intact.
This does not address nearby-object parallax or remaining normal error.

## Evaluation tooling

`evaluate_reflection_suite.py --save-all-renders` optionally retains all RGB frames;
metric computation still uses float images. `video_reflection_comparison.py`
encodes completed reports to labelled side-by-side H.264 clips and writes source
provenance. A two-frame synthetic encode/decode smoke passed in
`outputs/Oct30-1616-video-encoder-smoke`; it is not a research rendering result.
The optional encoder/plotting packages were installed under constraints preserving
all seven pinned core package versions.

An additional Gaussian SH3 reference with bands at 1k/2k/3k is queued only if time
permits, following the original implementation's early SH schedule. It changes
the full schedule, not merely the final degree; do not call it a matched late-SH
control. Final model selection must use validation before test rendering.
