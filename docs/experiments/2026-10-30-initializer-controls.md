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
`materials_chrome_hull_deferred_sh3.yaml` hold the other settings fixed. Their
results cannot replace or be pooled with original Materials scores. Additional
shading changes require evidence from these controls, rather than a blind sweep.

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
