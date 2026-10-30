# Training-mask initialization and mirror diagnostic

The corrected shared optimizer restores the Gaussian baseline, but chrome
reflections remain blurred. Finish the matched random-initialized surfel SH3 and
deferred SH3 controls before attributing a benefit to deferred shading.

## Next controlled treatment

Initialize oriented surfels from the existing training-alpha visual hull, then
retain the full 800px / 1M-capacity / 30k-step / seed-7 / late-SH3 policy. Compare
against the random-initialized control. This treats the complete initializer,
including its 100k starting points, as the intervention; it does not isolate
normal orientation from initial population, locations, scales or opacity.

Presets: `materials_reflection_hull_surfel_sh3.yaml` and
`materials_reflection_hull_deferred_sh3.yaml`. Their original Materials PLY is
`outputs/first-principles-2026-10-29/materials-mask-initializer/initial.ply`
(SHA256 `4f9acea4c85b93fcca19101ca4dffc35fd4de815c16263018b702cd51a736a64`).
It uses 100 training silhouettes and poses, a 512-cubed occupancy grid and a
smoothed signed-distance surface. No Blender mesh, reference depth/normal,
validation or test image initializes or supervises the model. Silhouettes cannot
recover hidden concavities. The prior untrained 400px/100-view geometry audit
measures 18.30 degrees mean normal error; a trained quality advantage is pending.

Measure PSNR, SSIM, LPIPS AlexNet, actual visible Chrome pixels, independent mesh
normal/depth errors, silhouette coverage and model/render costs. If initialization
helps, retain its SH-only control to identify what the reflection pass adds.
Longer training is authorized, but these 30k controls remain the common reference.

## Newly authorized auxiliary scene

The user subsequently authorized the new `materials_chrome` dataset, parameter
tweaks and longer training. Original Materials remains the primary benchmark.
The new scene contains only `CoreSolidMaterial.008` and `SolidModel.008`, enlarged
and recentered: Chrome spans about 1.59 units versus 0.46 in the original scene.
It is not merely the original scene with other objects hidden. Its larger mirror
is useful for inspecting reflected tree detail. The Chrome glossy roughness is
zero; the pedestal uses MidGray diffuse shading.

Audit: `outputs/Oct30-1435-materials-chrome-audit/`. All 400 image hashes match the
capture record (100 train, 100 validation, 200 test); parsed camera manifests equal
the original Materials templates, although JSON file hashes differ by formatting.
Image-hash verification does not use held-out pixels as training targets or for
model selection. Scene SHA256:
`91155c6a847a959e377863010f09575ff5d2a5d7ab4bcdbd1ccfc2f21bbc97af`.

Training-only initializer:
`Oct30-1436-training-mask-init/initial.ply` under the audit folder; 100k exported
points. Independent all-100-view 800px Blender validation reference:
`Oct30-1440-geometry-all-val-800/`, evaluation only. The first four-view pilot and
initializer completed with intact metadata/artifacts, but nested Windows process
stdout was not inherited. The full reference explicitly captures stdout/stderr
in `all-validation-geometry.log`; preserve the pilot rather than overwriting it.

Auxiliary presets `materials_chrome_hull_surfel_sh3.yaml` and
`materials_chrome_hull_deferred_sh3.yaml` hold the other settings fixed within
their pair. Their results cannot replace or be pooled with original Materials
scores. The before-training revision below adds geometry protection based on
observed drift in original Materials, rather than an appearance-parameter sweep.

### Untrained chrome initializer audit, 14:44 Singapore

All 100 validation views at 800px, no optimization: **9.2869 degrees** mean
camera-facing mesh normal error, **0.0343243** camera-z depth MAE, **0.977980**
silhouette IoU, **1.0** valid reference-foreground normal coverage. These are
per-view means over both meshes, excluding shader bump; they are geometry
diagnostics, not image-quality scores. Initializer SHA256:
`2108278f4e8ed7e508a15cbbed83f5cf308949c8b438e2aadb51577f0532854e`.
CPU-only evaluation leaves the training GPU free. Metrics, debug buffers and
source/reference fingerprints: `Oct30-1444-initial-geometry/` under the audit.

The sequential follow-up queue waits for the random controls, measures those on
an idle GPU, then trains the original Materials hull deferred/SH-only pair and
the separate chrome hull deferred/SH-only pair. All quality reports include
PSNR, SSIM, LPIPS and independent Chrome geometry. It checks the shared stop
switch and original deadline; each child job has its own bounded controller.
State: `outputs/first-principles-2026-10-30/initializer-controls-state.json`.

## First-principles interpretation

The current implementation already shades reflections **per pixel** from the
normalized alpha-composited normal and camera ray. Its learned 128x64 latitude-
longitude environment and scalar reflection blend are simpler than roughness-
aware physically based shading. Base SH can absorb low-frequency residual
appearance, but SH3 cannot by itself represent arbitrarily sharp reflected detail.
Sharper environment sampling helps only if normals and surface coverage are
accurate enough to address the correct directions.

[3DGS-DR's released renderer](https://github.com/gapszju/3DGS-DR/blob/main/gaussian_renderer/__init__.py)
also interpolates normals before reflected environment lookup; normal propagation
is a separate optimization mechanism, not an automatic consequence of flattening.
[Gaussian Surfels](https://arxiv.org/abs/2404.17774) adds depth-normal consistency
and normal priors to guide alignment. Our earlier consistency result improved
geometry but lost image quality under the previously weak optimizer; do not
assume that tradeoff is unchanged after the optimizer correction.

[RGS-DR](https://github.com/gkouros/RGS-DR) directly demonstrates a surfel/deferred
approach with roughness-aware environment lighting and a directional residual
pass. This supports feasibility of the architectural direction, not the claim
that our simpler renderer already implements it or achieves its results.

## Geometry drift and pretraining protocol revision, 15:23 Singapore

Saved floating-point snapshot buffers locate a concrete failure in the ongoing
original Materials hull-deferred run, **before its reflection stage**. Training
`evaluate_components` explicitly leaves the first fixed validation camera last;
the saved buffers therefore correspond to validation frame 0, not an arbitrary
training view. Its assignment is checked in the run log.

| Fixed validation view 0 | Initial hull | After 5k RGB-only steps |
| --- | ---: | ---: |
| Mean mesh normal error | 15.92 degrees | 35.29 degrees |
| Chrome normal error | 15.40 degrees | 38.25 degrees |
| Camera-z depth MAE | 0.02633 | 0.02257 |
| Valid reference-foreground fraction | 1.00000 | 0.99971 |

Depth error improves while the normals degrade. Thus good initial geometry is
not sufficient when RGB fitting can freely distort the tangent field. This is a
single predefined camera diagnostic, not an all-view quality result or proof that
every scene drifts identically. It supports protecting alignment rather than
merely extending the existing unconstrained schedule. Read-only source, snapshot
fingerprints and CSV: `outputs/Oct30-1519-materials-geometry-trajectories/`.

Keep the current unregularized original Materials pair unchanged as controls.
Add `materials_reflection_hull_regularized_{surfel,deferred}_sh3.yaml`: ray-plane
depth, self-supervised depth-normal weight **0.01**, starting at **step 1** with a
**2k ramp**, and deferred activation at step 1. Training alpha supplies only valid
foreground/mask information; mesh normals/depth are never optimization targets.
Both modes share the geometry policy and late SH3 schedule, 800px, 1M capacity,
100k inferred initial points, seed 7 and 30k steps. The original 128x64 environment
remains. This tests the combined protected training policy, not each component
in isolation. Its pair is queued after the earlier initializer experiments.

**Before any chrome quality training**, revise both auxiliary presets to the
same protected geometry policy and immediate deferred activation. Their deferred
environment becomes **512x256**: a 128-wide longitude grid samples every 2.8125
degrees; 512 samples every 0.7031 degrees, appropriate to the enlarged mirror.
The SH-only mode does not use that environment. The paired budget stays 800px,
1M capacity, 100k initial points and 30k steps. These are planned improvements,
not measured wins. The earlier auxiliary presets are retained in Git history.

An eight-step GPU smoke (`Oct30-1523-materials_chrome-protected-smoke`) activates
reflection, ray-plane consistency and all three SH bands, saves/reloads the
512x256 environment and evaluates all 100 views with PSNR/SSIM/LPIPS. It completes
with finite scores; its untrained quality is not a comparison result. The first
smoke was rejected before training because snapshot interval zero is invalid;
the corrected interval is 1000. Both controller logs and the invalid YAML remain.

### Reflection-branch diagnostic and final chrome prior, 15:49 Singapore

All-100-view diagnostic reports now measure the effective reflection blend and
normal coherence (length of the blended normal sum divided by coverage), including
the true visible Chrome region. These are renderer diagnostics, not physical
reflectance estimates. Random deferred Chrome has mean blend **0.09393** and
coherence **0.74409**; hull-deferred Chrome has **0.11510** and **0.75979**. The
environment itself varies, but its contribution to Chrome remains weak and
overlapping normals disagree. Both reports reproduce the existing image metrics.

Before any chrome quality training, set its pair's initial reflection strength
to **0.5**, a neutral blend prior for the predominantly mirror scene. All point
strengths remain learnable; no mesh/material labels enter training. The SH-only
mode ignores reflection strength. Keep the original mixed-material protected
pair at **0.1**. This is part of the combined auxiliary method, not evidence that
the initialization value alone improves quality. The 0.5 presets pass validation;
the earlier eight-step integration smoke used 0.1.

The unregularized original hull-deferred model is complete: **27.7503 dB**,
**0.942333 SSIM**, **0.038428 LPIPS**, Chrome **16.0601 dB**. Mean normal error
improves to **32.88 degrees** overall and **42.05 on Chrome**, but sharp reflections
remain unresolved. Its paired SH-only control is required before attributing the
LPIPS gain to deferred shading.

[Hull report](../../outputs/Oct30-1546-materials-hull-deferred-val/index.html) ·
[Random-deferred branch audit](../../outputs/Oct30-1544-deferred-branch-audit/index.html) ·
[Standalone normal-error plot](../../outputs/Oct30-1519-materials-geometry-trajectories/normal-error-trajectory.png).
Matplotlib 3.11.2 was added for the scientific PNG/SVG figure; seven core numeric,
rendering and metric package versions were pinned and verified unchanged.

## 16:58 Singapore: completed pair and auxiliary diagnostics

The matched hull surfel SH3 result is **27.7921 / 0.942775 / 0.037801**, Chrome
16.0594 dB; normal errors are 33.75 degrees overall and 42.28 on Chrome. This
slightly beats hull deferred globally, which remains 27.7503 / 0.942333 / 0.038428.
The common five-model report is `outputs/Oct30-1615-materials-initializer-controls-val`.

The enlarged Chrome protected deferred run has finished at 19.6184 / 0.783578 /
0.145693 on all 100 validation views. Its matched surfel control is running.
The predefined camera's normal error starts at 6.82 degrees overall / 5.99 on
Chrome, reaches 12.63 / 10.90 at 5k, then 11.92 / 10.25 at 15k. Depth MAE also
worsens from initialization. This trajectory uses one predefined camera and
cannot establish an all-view or cross-dataset causal comparison.

A CPU-only smoothing check changes the initializer's grid-gradient sigma from
1 to 3 while retaining the same training masks, 512 grid, 100k points and seed.
All-100-view untrained geometry error changes **9.2869 -> 8.8174 degrees**, depth
MAE **0.034324 -> 0.031937**, silhouette IoU **0.977980 -> 0.985816**. This is a
modest initializer gain, not a trained reflection result. Neither paired quality
preset was changed to this initializer. Artifacts:
`outputs/Oct30-1435-materials-chrome-audit/Oct30-1630-training-mask-smooth3` and
`Oct30-1632-smooth3-initial-geometry` under the same audit folder.

The separate [angular-response variant](2026-10-30-angular-response.md) addresses
another verified model mismatch while holding each protected preset's geometry
settings and initializer fixed. It remains untested at quality-training scale.

## User-requested wrap-up: original protected deferred completed

The 30k protected deferred run gives **28.4721 / 0.946323 / 0.038978**, Chrome
**16.1883 dB**, normal errors **13.19 degrees overall / 15.71 Chrome**, Chrome
coherence 0.9072 and effective blend 0.5740. Full 100-view validation is in
`outputs/Oct30-1753-materials-protected-deferred-wrapup-val`.
Its forward timing is invalidated because an accidentally started queued control
overlapped it; image quality and geometry results are unaffected.

The protected SH-only control briefly started during shutdown and was interrupted;
it has no completed result. The full HDR/angular/mask runs were cancelled before
starting. An independent original sigma3 CPU initializer audit finishes at
18.307-degree Chrome normal error, so smoothing alone does not solve its geometry.
See the [illustrated wrap-up](../reports/2026-10-30-research-wrap-up.md).
