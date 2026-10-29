# Return to the proposal: reflection quality

Started 29 October 2026, 20:38 Singapore. Hard stop: 30 October, 06:38 Singapore.
Status: active investigation. This record supersedes the earlier decision to stop
research only for the newly authorized investigation; archived runs stay intact.

**Current status:** the pruning correction is committed and the revised
`pruning_queue.py` is running. Corrected edited-scene deferred SH3 finished at
37.0101 dB with 133,700 surfels; consistency + SH3 finished at 35.3783 dB.
The completed SH-only control scores 36.7667 dB: the deferred gain is small and
material/perceptual results are mixed. Original Materials/Ship runs are next. The
[complete course-notes audit](../course-notes-audit-2026-10-29.md) records why
opacity cleanup and foreground detection are separate. Earlier queue/protocol
sections below are historical; preserved run configs identify the policy used.

## Objective and evidence

The private proposal asks whether deferred surfel shading improves highly specular
novel-view reconstruction over degree-3 SH on Materials, with Ship as a secondary
scene, while measuring quality, time and memory. Smooth normals are a means, not
the primary outcome. Current image gains over fixed RGB do not answer the SH
comparison. A clean environment panorama or recovered ground-truth geometry is
not a prerequisite for a useful novel-view synthesis result.

The edited three-ball scene is a diagnostic, not a substitute for Materials.
Recent mask/consistency/propagation/depth experiments are preserved. The ray-plane
run lost 0.8790 validation dB and did not visibly fix normal patches; another
normal-weight sweep is not justified by that evidence.

## First-principles diagnosis

1. A mirror lookup can explain sharp distant reflection if normals and the learned
   radiance map are good. It cannot alone express roughness, local occlusion,
   interreflection or view-dependent Fresnel effects. Current base RGB must absorb
   all of these while remaining constant with view.
2. Normal/depth self-consistency admits a consistently wrong surface. A low loss
   is not independent evidence of correct geometry. More regularization can trade
   image quality for coherence without fixing the inverse problem.
3. Deferred pixel shading couples overlapping surfels but does not guarantee the
   global optimum. Bilinear environment lookup supplies only local angular
   gradients. Neither zero thickness nor an intersection-depth correction alone
   guarantees a single coherent surface.
4. The implementation forbids SH in surfel passes. Ye et al. explicitly retain
   base SH for rough/global effects and unlock higher orders after reflection
   discovery. This omitted capacity is a concrete, testable difference.

## Bounded next experiment

Implement optional late SH using the existing checked basis/optimizer. Preserve
SH0 behavior, differentiate the full appearance-to-position path, inherit SH state
on splits, and restore appearance at evaluation. Verify CPU/CUDA gradients and
checkpoint round trips before training. Do not mix new geometry regularizers or
pruning changes into this experiment.

First use the edited scene to isolate deferred SH0 versus late SH3 under matched
budgets and frozen geometry-loss policy. Compare SH-only surfels and a fresh
Gaussian SH baseline on the same protocol before claiming a deferred advantage.
Evaluate validation during development; choose illustrative crops from camera
poses/object location rather than favorable metric changes. Once settings are
frozen, evaluate the selected models on test and report foreground/crop quality
alongside whole-image metrics. Existing test results have already been seen, so
fresh results are development evidence, not an untouched benchmark.

## Sources checked

- [Unit 11](https://weitsang.github.io/gaussian-splatting-notes/unit-11): downloaded
  directly after web-tool fetch failures; local HTML/text in
  `outputs/first-principles-2026-10-29/`. Its appearance-project guidance separates
  high-frequency reflection from directional residuals and calls for chrome crops.
- [Ye et al., 3DGS-DR](https://arxiv.org/html/2404.18454v1), sections 1 and 3.3:
  mirror environment shading plus base SH, with late higher-order activation.
- [Dai et al., Gaussian Surfels](https://arxiv.org/abs/2404.17774): rank-two EWA
  surfels and normal/depth consistency. This is the proposal's geometry reference.
- [Huang et al., 2DGS](https://surfsplatting.github.io/): a distinct method with
  perspective-correct footprints and depth distortion. Current EWA surfels are
  not a complete reproduction of this renderer.
- [Kouros et al., RGS-DR](https://arxiv.org/abs/2504.18468): deferred 2D surfels and
  a late directional residual support the same general decomposition. This does
  not establish that a simple SH residual will succeed here.

The proposal also overstates SH storage: degree 3 has 48 RGB scalar coefficients
in total, including the three DC/base scalars; it adds 45 scalars to fixed RGB.
Current deferred shading is a learned reflection blend, not a full roughness/GGX
physical BRDF. State this accurately in the report.

## Work log

- 20:38: recorded start/deadline; read private proposal and current experiment
  records; GPU idle. Preserved pre-existing staged/unstaged patches under outputs.
- 20:40: identified missing late base-SH capacity and selected a controlled
  appearance experiment. No new normal-weight sweeps authorized by this protocol.
- 20:44: optional late SH implemented. All 96 repository tests passed on CPU;
  all 50 surfel tests passed on CUDA. Six new regressions cover full image
  gradients (including appearance-to-position), locked bands, split inheritance,
  schedule, sidecar restoration and missing/untrained appearance. A 16-step
  32px GPU runner smoke reached SH3 and evaluated all 100 validation cameras:
  `materials_red-rose-silver-Oct29-2043-late-sh-smoke`. Its metrics are not a
  quality result. Committing the checked implementation before full training.
- 20:46: checked effective Blender object material slots (not shared mesh
  datablock materials). The edited silver object is `Aluminium`, with glossy
  roughness 0.4472 and a connected procedural bump normal. The red object uses
  `AnodizedMetal`; gold uses `CooperYellowish`. The actual `Chrome` material is
  on `SolidModel.008` in the original scene and has zero glossy roughness. The
  edited scene does not contain this chrome object. Expecting a crisp forest
  reflection from the rough silver reference was the wrong visual target.
- 20:54: added repeatable material inspection and evaluation-only Blender
  ray-cast geometry export. Camera/pixel alignment checked against edited-scene
  alpha at 400px: silhouette IoU 0.99955, 0.99945, 0.99916 and 0.99963 for views
  0, 13, 21 and 25. Exported all 100 validation views for original and edited
  scenes. Mesh normals intentionally exclude shader bump; they are independent
  geometric references, not exact shading normals or training supervision.
- 20:56: comparison tooling passed same-checkpoint equality, mixed Gaussian/
  surfel checkpoint, and per-material mesh-mask integration smokes on CPU. It
  records whole-image/foreground/material PSNR, SSIM, optional AlexNet LPIPS,
  fixed crops, geometry diagnostics, checkpoint size, synchronized render cost,
  and explicitly scoped evaluation-pool memory. Prepared four original-Materials
  400px/200k/30k presets with common seed, loss policy, learning rates, density
  schedule, no mask/normal regularization, and SH1/2/3 at 20k/22k/24k. Only
  geometry/appearance differ. The edited SH3 run is now in its SH1 phase.
- 20:58: all-view mesh alignment audit: mean/minimum silhouette IoU is
  0.999478/0.998302 on edited Materials and 0.999594/0.999061 on original
  Materials (100 validation views each, 400px, alpha threshold 0.5). Added
  per-process training Warp-pool peak logging; a GPU smoke recorded 2,600,788
  bytes. This excludes driver/context/external allocations. The already-running
  first SH3 job predates this logging and will have unknown training memory.

## Interpretation rules for this study

The original-Materials four-arm study uses MSE then D-SSIM under a shared
validation-MSE gate. Actual gate times can differ; report them. The edited-scene
study instead retains its existing L1/D-SSIM/mask objective. These are separate
protocols and their scene means must not be ranked together.

Report exact visible-material pixel metrics from Blender ray hits separately
from projected bounding-box crop metrics. Boxes can include pedestals, background
and occlusion; they are useful figures but not a material segmentation.

Performance timings include host tile preparation, device forward rendering and
MSE, with warm-up and synchronization; exclude image download, image writing and
perceptual metrics. Evaluation Warp allocator watermarks do not measure total or
peak training VRAM. Do not label them as such. Compressed model size includes PLY
and required material/SH sidecars, excluding diagnostic buffer files.

## First completed result: edited-scene appearance capacity

All 100 validation cameras, 400x400, seed 7, 30k steps, 200k surfels, identical
L1/D-SSIM/mask policy, no normal-depth penalty or propagation. New run:
`materials_red-rose-silver-Oct29-2044-late-sh3`. Historical comparison:
`materials_red-rose-silver-Oct29-1852-deferred-tune-normal-0-0`.

| Metric, mean per view | Deferred SH0 | Deferred + late SH3 | Change |
| --- | ---: | ---: | ---: |
| Whole-image PSNR | 34.5821 | 37.1227 | +2.5405 dB |
| Foreground PSNR | 23.4441 | 26.0282 | +2.5841 dB |
| SSIM | 0.987773 | 0.991101 | +0.003328 |
| LPIPS, AlexNet | 0.011082 | 0.005744 | -0.005337 |
| Visible aluminium pixels PSNR | 22.6782 | 26.9912 | +4.3131 dB |
| Visible gold pixels PSNR | 22.1731 | 23.8271 | +1.6540 dB |
| Visible red pixels PSNR | 23.7010 | 26.3415 | +2.6405 dB |

Every view improved in PSNR, foreground PSNR, SSIM and LPIPS. Every view also
improved for each of the three visible-material pixel metrics. Material masks
come from independent Blender first-hit geometry and reference alpha >=0.95;
they were not used to train these models. This supports the missing-appearance
capacity diagnosis, particularly for the rough silver material.

It does **not** show recovered geometry. Mean per-view mesh-normal angular error
is 45.9639 versus 45.6769 degrees; mean per-view absolute mesh-depth error is
0.03737 versus 0.03348 scene units. At least 99.96% of reference foreground
pixels meet the prediction-coverage/normal-validity criteria on average.
These are geometric mesh normals, excluding procedural shading bump. Improved
RGB fitting with almost unchanged normal error is evidence against interpreting
the image gain as normal recovery.

The hybrid also forfeits the proposal's four-material-scalar memory argument:
SH3 adds 45 scalar coefficients per surfel. Required compressed checkpoint size
increased from 14,364,057 to 48,307,710 bytes. At 400px the measured forward
pipeline median was 9.92 versus 10.49 ms/view (100.8 versus 95.3 FPS), including
host tile preparation and MSE, excluding image download and metric computation.
This is a small fixed-camera microbenchmark, not a real-time application claim.
Trainer-reported times were 836.80 and 890.18 seconds under different session
workloads; do not treat their ratio as a controlled training-speed result.

The legacy comparison has different source snapshots and snapshot cadence (1k
versus 5k); SH0 rendering behavior is covered by the new regressions. Current
renderer re-evaluation reproduces its archived mean within 0.00002 dB. A fresh
SH0 control is queued to check reproducibility under current source. Original
Materials controls remain necessary before claiming a deferred advantage over SH.

Evidence: [comparison gallery](../../outputs/first-principles-2026-10-29/edited-appearance-val/index.html),
[per-view metrics](../../outputs/first-principles-2026-10-29/edited-appearance-val/metrics.csv),
[geometry metrics](../../outputs/first-principles-2026-10-29/edited-appearance-val/geometry-metrics.csv),
[performance](../../outputs/first-principles-2026-10-29/edited-appearance-val/performance.json),
and [protocol/source differences](../../outputs/first-principles-2026-10-29/edited-appearance-val/protocol.json).

## Independent reassessment: consistency did improve geometry

Re-rendered the three frozen geometry checkpoints on all 100 validation views
with the same independent Blender mesh references. This uses no new training or
reference supervision. Render timings were explicitly skipped because another
GPU training job was running.

| Historical variant | PSNR | Mean mesh-normal error | Mean mesh-depth absolute error |
| --- | ---: | ---: | ---: |
| No consistency | 34.5821 dB | 45.9639 degrees | 0.037371 |
| Centre-depth consistency, weight 0.01 | 34.0690 dB | **22.7586 degrees** | **0.025738** |
| Ray-plane-depth consistency, weight 0.01 | 33.1900 dB | 23.3974 degrees | 0.033560 |

Means are computed per view then averaged. Normal-valid foreground fractions
are 0.99961, 0.99914 and 0.99827. Mesh normals exclude material bump, and depth
uses scene units. The large consistency benefit is not explained by dropping
most reference pixels. Ray-plane depth did not improve mean geometry accuracy
over the simpler consistency model in this evaluation.

This corrects the earlier qualitative-only conclusion: normal consistency was
useful geometry work, despite imperfect previews and slightly worse RGB scores.
It does not establish perfect surfaces. The most justified next test is
**centre-depth consistency plus late SH**, changing only the consistency weight
relative to the completed late-SH model. A new normal-weight sweep, distortion
loss, or roughness implementation is not needed to test that combination.

Preset: `config/materials_red-rose-silver_deferred_consistency_sh3.yaml`.
Evidence: [geometry reassessment](../../outputs/first-principles-2026-10-29/geometry-reassessment/index.html)
and [per-view geometry metrics](../../outputs/first-principles-2026-10-29/geometry-reassessment/geometry-metrics.csv).

## Original scene and proposal-scale follow-up

The 400px/200k/30k deferred-SH3 scout completed with 21.2251 dB / 0.877096 SSIM
and visible floaters. It first reached the D-SSIM gate late in training. This is
not a good reflection result and should not be used as a substitute for the
proposal-scale benchmark. The remaining matched scout controls will be retained.

Prepared full-resolution Gaussian-SH3 / surfel-SH3 / deferred-SH3 presets by
copying the **already measured original Materials baseline policy**: 800px, 1M
capacity, 30k steps, position decay over 60k, reflection from 10k, initial strength
0.1 and environment/reflection learning rates 0.01. SH bands unlock at
20,001/22,001/24,001. They do not inherit the edited-scene settings used by the
cheap scout. These settings are a separate, explicitly recorded protocol; do not
attribute cross-protocol changes to any one parameter.

Ship was downloaded with pinned revision and per-file hashes, with 100/100/200
disjoint train/validation/test image paths and valid camera matrices. Its raw
259,074,427-byte dataset is excluded from commits. The same full-resolution
Gaussian-SH3 / deferred-SH3 pair is prepared for the proposal's secondary scene.
See [Ship data provenance](../ship-data-source.md).

## Execution and stopping rules

Two session-local sequential controllers keep independent runs off the GPU while
render timing is measured. `research_queue.py` runs the cheap controls;
`proposal_queue.py` waits for it, then runs the consistency+SH combination,
matched reports, full-resolution Materials and Ship models, and frozen-model
test reports. Their machine-readable state/logs live in
`outputs/first-principles-2026-10-29/`. Both enforce the original absolute
06:38 Singapore hard stop, including terminating only their own active child
process trees if necessary. No queued full-resolution settings depend on test
results. Ship test illustrations 20 and 120 are predetermined by Unit 11.

Further architecture changes are held until the consistency+SH result and
proposal-scale controls are measured. There is already a concrete appearance
improvement and independently verified geometry improvement to combine. This
prevents every imperfect preview from triggering a different loss or renderer.

If reflection fitting still fails with reasonable geometry, a grounded subsequent
test is progressive environment resolution: a coarse map gives a wider angular
gradient neighborhood than the current 128x64 bilinear lookup. This is a
continuation hypothesis, not a proven fix or an original algorithm claim.
[SpecGloss-GS, WACV 2026](https://github.com/gkouros/SpecGloss-GS) uses progressive
environment-map upsampling alongside a richer BRDF and normal/diffuse priors.
Its [training loop](https://github.com/gkouros/SpecGloss-GS/blob/main/train.py)
replaces the environment optimizer parameter when resolution doubles. That
supports the mechanism, not an expectation that upsampling alone reproduces
its results. HDR lighting, full roughness/Fresnel and indirect reflection remain
different changes and must not be bundled into that test.

## Pruning correction: first full result

Run `materials_red-rose-silver-Oct29-2139-pruned-deferred-sh3` completed 30k
steps at 400px. This tests the revised complete pruning policy (training-alpha
contradictions plus continuing late ADC/opacity cleanup), not each route in
isolation. The archived pre-correction run remains the comparison.

| Quantity | Earlier late SH3 | Pruning correction |
| --- | ---: | ---: |
| Active surfels | 200,000 | **133,700** |
| Validation PSNR | 37.1227 dB | 37.0101 dB |
| Validation SSIM | 0.991101 | 0.990965 |
| Required checkpoint bytes | 48,307,710 | **32,342,148** |

The count fell 33.15%, for a 0.1126 dB PSNR difference. Do not select a model on
count alone; geometry, reflection crops and LPIPS are still to be compared.
The corrected run reported 841.44 training seconds. Session workloads differ,
so that is not a controlled training-speed ratio.

The log now exposes turnover: at step 4,800, 12,866 surfels were pruned and
12,866 created, leaving 200k active. At step 20,500, 8,891 were pruned and zero
created, reducing 200k to 191,109. Cleanup continued throughout refinement.
The revised sequential controller is `pruning_queue.py`; the older two
controllers remain held. The original 06:38 Singapore hard stop is unchanged.

## Unit 12 diagnostic: geometry from training silhouettes

Implemented a visual-hull initializer using only the 100 posed training PNG
alpha channels. It intersects silhouette cones in a 512-cubed grid, extracts
a smoothed signed-distance surface, and creates 100k oriented surfels. Bounds
come from training camera distances. It reads no validation/test files, Blender
scene, depth or reference normals. Four tests cover analytic-sphere geometry,
unknown off-frame regions, quaternion orientation and forbidden-file sentinels.

Independent evaluation on all 100 validation cameras, **before any training**:

- Mean mesh-normal error: **14.3395 degrees** (earlier trained consistency model:
  22.7586 degrees; earlier unregularized late-SH model: 45.6769 degrees).
- Mean absolute mesh-depth error: 0.022986 scene units.
- Normal-valid reference foreground fraction: 1.0000.
- Silhouette IoU: 0.925866. The inferred surface/support leaks beyond silhouettes
  and fills unseen concavities; it still needs photometric/mask refinement.
- Preprocessing: 20.71 seconds on CPU. This is a coarse initializer, not recovered
  ground-truth geometry or a finished novel-view synthesis result.

This passes the stated geometry-only go/no-go check for a bounded follow-up.
Prepared `materials_red-rose-silver_deferred_hull_sh3.yaml`: same consistency,
late-SH and pruning settings, with the derived initial PLY. Actual initial count
is 100k rather than 1k random points, so it tests the initialization method as a
whole, not placement or orientation alone. Do not present any future advantage
as an isolated normal-loss or deferred-shader effect.

Reproduce the initializer:

```powershell
.venv/Scripts/python.exe scripts/initialize_from_training_masks.py `
  data/materials_red-rose-silver `
  --output outputs/first-principles-2026-10-29/edited-mask-initializer `
  --grid-resolution 512 --max-points 100000
```

Source/provenance: `edited-mask-initializer/initialization.json`; independent
metrics: `edited-mask-initializer-evaluation/summary.json`, both under the session
output directory. The initializer itself never consumes these evaluation files.

## Sampling and growth bookkeeping correction, 22:08 Singapore

Two regression tests reproduced additional orchestration defects:

1. Growth history was updated before the same-step foreground deletion. It now
   sees the final population after both density and foreground operations.
   Otherwise a net deletion could be mistaken for a growth plateau.
2. Density view selection consumed the training-camera RNG. A changed pruning
   schedule consequently changed the camera sequence. ADC now has an independent
   deterministic stream; camera sampling depends only on the run seed.

Both regressions passed on CPU and CUDA; all 111 CPU tests passed. The three
corrected edited-scene runs had already loaded the earlier common workflow and
retain that source in their archives. Subsequent proposal-scale runs will all
use the corrected workflow. Do not treat comparisons across that boundary as
isolating only a configuration change. Renderer equations are unchanged.

The corrected consistency + SH3 run completed at 35.378275 dB / 0.988175 SSIM.
It improves on the earlier no-SH consistency model (34.0690 dB), but trails the
corrected unregularized model (37.0101 dB). Independent geometry, material scores
and perceptual scores are pending; this alone does not select the winner.

## Stronger historical SH3 reference, 22:17 Singapore

Re-evaluated the earlier validation-selected SH3 learning-rate winner
`materials-Oct28-2327-sh3-tune-lr-0p00025` on all 100 original Materials
validation cameras at 800px, with the same evaluation-only chrome mesh masks.
This is a stronger historical reference alongside the newly matched late-SH
controls, not a claim of matching training settings/source/pruning policies.

| Metric | Earlier tuned Gaussian SH3 |
| --- | ---: |
| Whole-image PSNR | 27.337226 dB |
| Foreground PSNR | 21.428579 dB |
| Chrome-pixel PSNR | 16.163473 dB |
| SSIM | 0.933825 |
| LPIPS AlexNet | 0.048299 |
| Required checkpoint bytes | 170,285,798 |

The repeated whole-image PSNR agrees with its original validation result.
Fixed chrome crops show broad scene/landscape colours but blurred fine reflected
detail; do not describe SH as producing no reflection at all. Higher scene PSNR
alone will not establish that the deferred model fixes this limitation.
Report: `outputs/first-principles-2026-10-29/materials-strong-baseline-val/`.
This read-only quality audit ran while training continued, so GPU timing was
explicitly skipped. Its brief GPU/CPU overlap also rules out interpreting the
concurrent edited control's training time as an isolated speed benchmark.

## Corrected edited-scene comparison, 22:23 Singapore

All 100 validation views at 400px; common training-mask pruning, late SH schedule,
200k capacity and 30k steps. RGB/deferred models share the same mask objective;
the consistency treatment additionally uses normal-depth weight 0.01. All three
loaded the same workflow before the later sampling/bookkeeping fix.

| Quantity | SH-only surfels | Deferred SH3 | Deferred + consistency + SH3 |
| --- | ---: | ---: | ---: |
| Whole-image PSNR, dB | 36.7667 | **37.0101** | 35.3783 |
| Foreground PSNR, dB | 25.6518 | **25.9521** | 24.4407 |
| SSIM | 0.990424 | **0.990965** | 0.988175 |
| LPIPS AlexNet, lower better | **0.005906** | 0.006156 | 0.007531 |
| Aluminium PSNR, dB | **27.0947** | 26.4041 | 26.2124 |
| Gold PSNR, dB | 23.1157 | **24.0276** | 21.8595 |
| Red PSNR, dB | **26.1631** | 26.1406 | 25.1055 |
| Mesh-normal error, degrees | 44.9912 | 46.3633 | **22.7840** |
| Mesh-depth MAE, scene units | 0.034234 | 0.033787 | **0.025940** |
| Active surfels | 132,394 | 133,700 | 133,146 |
| Required checkpoint bytes | **31,493,594** | 32,342,148 | 32,052,130 |
| Scoped forward FPS, idle GPU | 121.85 | 132.01 | 139.85 |

Deferred versus SH-only improves whole-image PSNR on 67/100 views, foreground
PSNR on 71/100, SSIM on 76/100, and LPIPS on only 39/100. Gold pixels improve
on 91/100 views; aluminium improves on 18/100 and red on 46/100. The gold gain
is 0.9119 dB, while aluminium loses 0.6905 dB. This is a mixed result, not a
general reflection-quality win. Fixed crops still show grain and smoothed gold
detail. SH capacity solved part of the appearance deficit, not the whole problem.

Consistency lowers independent normal error in all 100 views. Per-material
normal errors are 21.18/21.24/21.81 degrees on aluminium/gold/red, versus
48.84/52.36/47.04 for unregularized deferred SH3. At least 99.89% of each material's
reference foreground remains valid in the consistency model. The geometry gain
is real but still insufficient for accurate sharp reflection; its image penalty
is also real. This supports the bounded geometry-initialization test, not more
normal-weight sweeps.

Fresh-process training Warp allocation-pool peaks were 639,216,940 bytes for all
three. These are allocator peaks, not total VRAM. Training seconds were
877.74/841.44/792.61; session workload differences prevent controlled speed-ratio
claims. Render timing was measured while no training job was active, with the
previously documented four-view/ten-repeat forward scope. FPS ordering reflects
the different learned geometry/work, not just shader overhead.

Evidence: [comparison](../../outputs/first-principles-2026-10-29/edited-pruning-val/index.html),
[per-view scores](../../outputs/first-principles-2026-10-29/edited-pruning-val/metrics.csv),
[material geometry](../../outputs/first-principles-2026-10-29/edited-pruning-val/geometry-summary.csv),
and [performance](../../outputs/first-principles-2026-10-29/edited-pruning-val/performance.json).

## Remaining frozen checks

Evaluation now also records mesh-normal/depth error and valid-coverage fraction
for each visible target material, with means in `geometry-summary.csv`. This
prevents a whole-scene geometry score dominated by pedestals from being mistaken
for accuracy on reflective balls. A mixed Gaussian/surfel, 100-view CPU smoke
verified material/empty-coverage handling; its tiny untrained models are not
quality evidence. The upcoming frozen-model reports use these added columns.

One bounded initializer follow-up is queued in `initialization_queue.py`, waiting
for the entire proposal study and its timing reports to finish. It retrains the
random-initialized consistency + SH3 control with the corrected sampling code,
then trains the prepared mask-initialized variant. Both use the same code,
settings, seed and 30k-step/200k-cap budget. Initial count, placement and orientation
are explicitly one method treatment. Their all-view validation report includes
independent mesh normals/depth, material/foreground metrics, LPIPS and idle-GPU
render timing. No additional architecture or hyperparameter sweep is queued.
The same original absolute deadline applies while waiting and during training.

## Commit policy

Commit the reviewed protocol, then verified implementation, then measured
findings. Preserve pre-existing staged changes; explicitly commit only work from
this investigation. Large runs and source/input fingerprints remain in outputs.
