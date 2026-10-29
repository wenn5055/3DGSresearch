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

## Commit policy

Commit the reviewed protocol, then verified implementation, then measured
findings. Preserve pre-existing staged changes; explicitly commit only work from
this investigation. Large runs and source/input fingerprints remain in outputs.
