# Deferred Specular Shading with 2D Gaussian Surfels

Working documents: [research stages](researchStages.md), [todo list](TODO.md),
[run-record template](run-record-template.md), and [commands](CommandList.md).

Historical rationale and literature checks:
[research design record](research-design-notes.md).
Current measured evidence: [findings](findings.md).
Latest design and implementation decisions: [29 October training update](#design-update-log).

## Purpose and status

Extend the custom Python/NVIDIA Warp 3D Gaussian Splatting trainer in
`3dgs_trainer.py` with oriented 2D surfels and deferred specular shading.
**Updated 29 October 2026 (Singapore).** The RGB, staged D-SSIM, and degree-2/3 SH
baselines are implemented and have recorded Materials results. The validation-only
SH learning-rate sweep selected 0.00025 by validation PSNR (single seed). Consolidated evidence is in
[findings.md](findings.md) and [the experiment archive](experiment-results-2026-10-28.csv).

The project now has three independent passes, with commands and model details in
[project-passes.md](project-passes.md): a training-only Gaussian RGB control,
flattened surfels with fixed RGB, and surfels with deferred environment reflection.
The first full Materials comparison is recorded: deferred surfels improve validation
PSNR by 1.1374 dB over RGB surfels for seed 7, but normals remain noisy. Separate
training-alpha mask and normal-depth loss presets now target geometry. The edited
three-ball scene has completed RGB/normal and deferred propagation runs, but a
matched geometry comparison remains pending. The latest optimizer fixes, quiet
refinement schedule and matched consistency presets are recorded in the design log
below. Project code lives under `src/cp4282_gs/surfels/` and shares optimizer code
with the Gaussian baseline; archived runs retain their original source snapshots.

## Research question and baseline

Does deferred specular shading using explicit 2D surfel normals improve held-out
image reconstruction on the NeRF-synthetic materials and ship scenes relative to
the existing Unit 9 trainer, and what training-time and VRAM costs does it add?

The repository now supports fixed RGB and SH through degree 3. On the matched
200-view Materials test set, fixed-RGB D-SSIM, SH2, and SH3 reached 24.9365,
25.6693, and 26.0690 dB respectively at 30k steps. These are archived single-seed
baselines, not claims about the new surfel/shading passes.

An initialization audit found the historical baseline averaged combined
train/validation images for its initial RGB. The independent project passes use
training-only initialization statistics. Use the new Gaussian control when
isolating the geometry change; historical scores remain useful context but carry
this initialization difference. No validation/test images provide training gradients.

## Implemented project iterations

0. **Gaussian RGB control:** the existing 3D renderer/optimizer, with training-only
   initialization and the same project budget/settings.
1. **Flattened surfels + fixed RGB:** quaternion orientation and two positive tangent
   scales; exactly zero local-z variance, with a rotated-z normal. Screen projection
   retains the baseline's filtered EWA approximation, rather than implementing the
   perspective-correct ray/plane rasterizer of 2DGS.
2. **Surfels + deferred specular:** composite RGB, world-normal, depth, coverage and
   reflection-strength G-buffers. Normalize the per-pixel normal, reflect its camera
   ray, and bilinearly sample a learned latitude/longitude environment map. Blend
   base RGB and environment reflection with coverage-aware learned strength.

The reflection prototype has no GGX roughness or Fresnel term. It is an
environment-reflection model; optional normal-depth consistency now supplies a
geometry constraint. Richer material models remain later ablations.
Gradients flow through the shader, environment lookup,
G-buffer compositing, surfel normals, and geometry; both CPU and CUDA checks cover them.

## Datasets and evaluation

- Primary scene: materials, including a fixed chrome-sphere crop for focused
  reflection evaluation and full-image metrics.
- real-world NeRO evaluation. Its
  scope, scene selection, data access, and camera conversion remain to be confirmed.
- Report held-out test-view PSNR, SSIM, and LPIPS, plus model size, rendering FPS,
  GPU VRAM usage, and training wall-clock time.
- Use matched train/validation/test splits, image resolution, training budget
  (30,000 iterations in the proposal), and shared optimization settings. Document
  settings introduced by the new model and any departures from this protocol.
- Record baseline results before changes. Published results from other systems
  require matched evaluation conditions before being treated as direct comparisons.

## Literature named in the proposal

- Ye et al., *3D Gaussian Splatting with Deferred Reflection* (SIGGRAPH 2024).
- Dai et al., *High-quality Surface Reconstruction using Gaussian Surfels*
  (SIGGRAPH 2024).
- https://github.com/liuyuan-pal/nero
- https://github.com/gapszju/3DGS-DR

## Design update log

### 2026-10-29 — Stabilize optimization before tuning surface constraints

**Reason for the update:** deferred RGB scores improved while object normals stayed
noisy. A full training-path audit identified optimizer problems and insufficient
surface constraints. The [audit snapshot](training-code-audit-2026-10-29.md) records
the pre-fix behavior, source locations and reproduction evidence.

**Verified current model:** per-pixel deferred environment reflection is implemented;
normals derive from surfel rotations. The active image objective is
`0.8*L1 + 0.2*D-SSIM + 0.1*mask_BCE`, with D-SSIM defined as `(1-SSIM)/2`.
`photometric_loss: l1_dssim` overrides the inherited MSE curriculum. The numeric
0.2 structural weight therefore differs from the paper code's `1-SSIM` convention;
retain the project's convention for this ablation and record it in run summaries.
Mask supervision is present and reduces surrounding floaters, but does not constrain
hidden layers or establish normal accuracy. The recent propagation presets have
normal-depth weight zero.

**Implemented following the audit:**

- RGB updates can recover inward from exactly 0 or 1. The explicit dense, sparse
  and surfel backward paths use the interior derivative at the endpoints, while
  parameter projection retains the [0,1] bounds.
- Densification preserves live splats' optimizer moments and clocks. New clone/split
  slots inherit their parent's complete Adam trajectory, including quaternion,
  reflection and SH state. Split history is an explicit warm-start choice, rather
  than zero moments with an old clock. Rejected candidates do not reset state.
- `surfel.refinement.start_iteration` begins a quiet final phase: no growth,
  opacity resets or propagation/noise. `prune_interval: 0` keeps topology fixed;
  a positive interval explicitly enables pruning without growth. Any outstanding
  propagation opacity freeze ends at entry. Structure updates are skipped on the
  final training step.
- Current 30k paper-inspired presets enter quiet refinement at 24,001; the 10k
  preset at 8,001. Saved historical configurations remain unchanged.
- Optional normal-depth consistency can coexist with propagation. The first
  matched comparison disables propagation in both arms to isolate consistency.

**Next controlled experiment:** use the new 400px/200k-capacity
[control](../config/materials_red-rose-silver_deferred_consistency_control.yaml)
and [normal-consistency](../config/materials_red-rose-silver_deferred_consistency.yaml)
presets. Both start fresh with seed 7, run 30k steps, finish position decay at 30k,
keep the final 10k steps fixed in topology, and retain image/mask supervision.
Only normal-depth coefficient (0 versus 0.01) and output destination differ.
The positive coefficient ramps from 2k to 4k. The existing sweep can run this pair:

```bash
python scripts/tune_deferred_normal_loss.py config/materials_red-rose-silver_deferred_consistency.yaml --weights 0 0.01
```

**Acceptance criteria:** boundary recovery, unchanged parent state, inherited child
state, no-op density events, quiet-phase scheduling, independent pruning and hybrid
gradients must pass regression checks on CPU/CUDA. Short integration runs validate
training/evaluation and serialization; full-budget reconstruction gains remain to
be measured. Compare foreground RGB plus normal/depth accuracy on matched held-out
cameras; export Blender normal/depth references for evaluation only. PSNR-based
sweep selection does not establish the best geometry.

**Implementation verification:** 84 tests passed on CPU and with surfel/SH tests
on CUDA; a 64-step GPU run completed checkpoint/evaluation integration. Detailed
evidence and the smoke run's limits are in the
[dated findings](findings.md#2026-10-29--training-audit-fixes-and-geometry-experiment-design).

**Deferred decisions:** contribution-aware pruning, accumulated visibility-aware
growth scores, ray/plane depth or depth-distortion constraints, and richer materials
remain separate work. The current footprint-based pruning heuristic and averaged
centre-depth limitations are documented, not fixed by these optimizer changes.
Keep photometric loss alongside geometry consistency; do not replace the entire
objective with a normal-only loss.

### 2026-10-29 — Close normal-consistency tuning and move on

**User decision:** stop this ablation here and move on; additional normal-weight
sweeps and Blender reference-normal/depth evaluation are not required to proceed.
Preserve both completed runs rather than presenting one as the best geometry.

The [matched validation sweep](../outputs/materials_red-rose-silver-Oct29-1852-deferred-normal-sweep/summary.md)
completed both 30k-step runs. Weight 0 achieved **34.5822 dB / 0.987773 SSIM**;
weight 0.01 achieved **34.0690 dB / 0.985603 SSIM**. Weight 0 is the image-score
selection. Weight 0.01 is retained as the geometry-regularized comparison: its
saved normals/depth look smoother, with residual patches. This is qualitative
coherence evidence, not measured normal accuracy. The image-score difference is
0.5131 dB in favor of zero weight.

**Next phase:** final held-out renders/reflection comparisons and write-up using
the preserved runs. Explain the appearance/coherence tradeoff, retain the loss
ablation, and avoid claims of recovered ground-truth geometry. Further geometry
and material changes remain future experiments rather than prerequisites.

### 2026-10-29 — Final test renders and ablation write-up completed

Evaluated both frozen checkpoints on all **200 test views at 400x400**, verifying
matched settings and renderer source fingerprints. Weight 0 achieved
**34.5007 dB / 0.987244 SSIM**; weight 0.01 achieved **34.1435 dB / 0.984849 SSIM**.
Foreground PSNR was **23.4455 versus 23.2170 dB**. The validation-selected control
therefore retains higher mean test RGB scores; the regularized model has visibly
smoother normal maps across the four pose-selected illustration cameras.

The [final experiment write-up](experiments/2026-10-29-normal-consistency-final.md)
records protocol, exact metrics, interpretation and limitations. The
[comparison report](../outputs/materials-red-rose-silver-final-comparison-2026-10-29/index.html)
contains RGB, eight matched object/reflection crop triplets, normals and depth
figures, plus all per-view scores and source/input/checkpoint fingerprints.
No retraining or reference geometry was used; tuning remains closed.
