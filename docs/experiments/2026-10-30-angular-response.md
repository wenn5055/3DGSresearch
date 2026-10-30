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

## Implemented, 16:58 Singapore

Optional `surfel.shading_model: hdr_angular` uses 16 shared learned logit offsets
interpolated by absolute N dot V, added to the rasterized base reflection logit.
All knots start at zero. The environment uses positive softplus radiance, with
parameter bounds [-16, 100]. Initial radiance is converted from the training-only
mean display color, avoiding a clipped-white initialization. Shading converts base
display RGB to linear light, blends with HDR reflection, clips to Standard's display
range and applies the sRGB transfer, then composites coverage over the background.
SH remains an optional base-color residual. No material labels enter optimization.

The original scene's read-only inspection also confirms Standard / None / exposure
0 / gamma 1 and the same linked Chrome ramp, retained in `original-chrome-response.json`.
Saved scene settings do not independently prove the historical PNG capture pipeline;
the new Chrome capture pipeline is recorded and verified.

The default LDR kernel and optimization arithmetic remain unchanged. HDR sidecars
use format 2 so older readers reject them instead of silently interpreting HDR
parameters as LDR. Current evaluation routes from saved shader metadata. Contribution
compaction preserves the global angular knots and HDR environment.

Checks: 6 HDR tests on CPU and CUDA, including 22 finite-difference cases per
device with a nonconstant angular response; 12 existing CUDA trainer regressions;
frozen-compaction regressions. Final-format eight-step 800px integration smoke
completed with SH3, growth 100k -> 160k, sidecar reload and all three metrics:
`outputs/Oct30-1654-materials_chrome-hdr-angular-smoke-v2`. Its scores are not a
quality result. The earlier smoke is preserved as a format-1 prototype, readable
with its source snapshot rather than the final reader.

Full presets change the shading model relative to each protected LDR control;
initializer, geometry loss, reflection prior, resolution, capacity, seed, optimizer
policy, SH schedule and 30k budget stay fixed within each scene. The original
Materials variant has priority. Runs may be skipped to retain at least one hour
for a final training job plus validation/test/video work before the hard stop.

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

## Final selection rule, fixed at 17:01 Singapore

Within each original-scene representation family (Gaussian SH3, surfel SH3,
deferred SH3), select the maximum mean validation Chrome PSNR. Break exact ties
by lower LPIPS, then higher global PSNR, then run path. Retain every candidate's
full global PSNR/SSIM/LPIPS and Chrome scores. Eligible runs must complete 30k,
800px, 1M capacity, SH3 and the corrected fresh-child optimizer policy in this
session. This selects complete family treatments, not matched causal ablations;
the initializer and SH schedule may differ. The matched reports remain separate.

The auxiliary Chrome report will evaluate all completed protected surfel, LDR
deferred and HDR/angular members separately. Selection is frozen to JSON with
checkpoint and validation-source hashes before rendering any final test report.
All 200 test frames, the three metrics, geometry diagnostics and idle-GPU timing
are scheduled, followed by labelled videos. No test-based retuning. The earlier
frozen-pruning test remains an independently completed derivative comparison.

Machine-readable plan and dry-run-checked controller:
`outputs/first-principles-2026-10-30/final-evaluation-plan.json` and
`final_evaluation_queue.py`. They wait for the training queues to end and share
the same absolute deadline.

## Objective check, 17:10 Singapore

The completed Chrome LDR log explicitly says D-SSIM never activated: validation
PSNR remained below the preset's 21 dB gate throughout 30k. Its matched surfel
control is still running. This is an additional limitation of transferring a
global-PSNR curriculum from the original, mostly background scene to the enlarged
mirror. Keep the current pair and the shader-only HDR comparison unchanged;
otherwise the shading contrast would also change its objective. If the shading
correction does not cross the gate, a separately matched objective experiment
is warranted, rather than silently calling the run an MSE+D-SSIM fit.

## 17:22 Singapore: pre-training mask revision

The enlarged Chrome matched pair is complete. Surfel SH3 gives **19.8175 /
0.792973 / 0.130209**, Chrome **13.9886 dB**; LDR deferred gives **19.6178 /
0.783578 / 0.145693**, Chrome **14.4575 dB**. Deferred improves the target material
by **0.4689 dB**, with worse global/perceptual scores. Chrome normal errors are
15.97 versus 12.65 degrees, and deferred's mean effective Chrome blend is 0.640.
The image still lacks sharp reflection detail.

Recovering pixel-weighted error sums from the existing float metrics reveals
foreground MSE **0.036532 -> 0.033027**, but nonforeground MSE **0.000488 ->
0.002281** (4.67x). Here nonforeground includes alpha boundaries and background;
these cannot be separated from the archived scalar errors. The diagnostic is
`outputs/first-principles-2026-10-30/chrome-error-partition.json`, with per-view
counts, source hashes and no new GPU rendering.

Before either HDR quality run starts, revise **both full HDR presets** to use
the already implemented training-alpha mask BCE with weight **0.1**. This supplies
direct per-pixel silhouette supervision, whereas the existing mask pruner only
rejects repeatedly empty full support boxes. Keep the original protected pair
unchanged. This supersedes the earlier shader-only design: the upcoming HDR run
is a **combined shading and mask-supervision treatment**, so its quality cannot
be attributed to HDR/angular response alone. Other fields stay fixed. Frozen
appearance diagnostics have fixed coverage and retain mask weight zero.

The original HDR queue was verified waiting with zero model jobs, archived and
superseded. Its owned process tree was stopped; no active training was stopped.
The revised queue has a fresh controller identity and the same absolute deadline.
The final selector continues to wait on the canonical queue state. Gradient
checks now also cover HDR shading with mask and normal/depth losses jointly.
Future performance JSON explicitly labels training times as observed wall time,
including possible diagnostic interference, rather than controlled speed ratios.

The revision passes 10 HDR/freeze tests on both CPU and CUDA, including 33
finite-difference cases per device. Its full eight-step integration smoke,
`outputs/Oct30-1722-materials_chrome-hdr-mask-smoke`, completes with growth,
SH3, mask/normal losses, format-2 save/reload and all three validation metrics.
Smoke scores are not quality evidence. The earlier no-mask source/presets remain
available in Git history and the archived waiting-queue source/state.
