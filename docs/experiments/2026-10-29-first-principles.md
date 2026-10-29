# Return to the proposal: reflection quality

Started 29 October 2026, 20:38 Singapore. Hard stop: 30 October, 06:38 Singapore.
Status: active investigation. This record supersedes the earlier decision to stop
research only for the newly authorized investigation; archived runs stay intact.

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

## Commit policy

Commit the reviewed protocol, then verified implementation, then measured
findings. Preserve pre-existing staged changes; explicitly commit only work from
this investigation. Large runs and source/input fingerprints remain in outputs.
