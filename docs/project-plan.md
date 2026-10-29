# Deferred Specular Shading with 2D Gaussian Surfels

Working documents: [research stages](researchStages.md), [todo list](TODO.md),
[run-record template](run-record-template.md), and [commands](CommandList.md).

Detailed rationale, literature checks, and current evidence:
[research design record](research-design-notes.md).

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
training-alpha mask and normal-depth loss presets now target geometry; their full
quality comparison remains to be run. Project code lives under `src/cp4282_gs/surfels/`, leaving the
baseline trainer and the completed sweep's training dependencies unchanged.

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

The reflection prototype has no GGX roughness, Fresnel term, or independent normal
priors yet. It is an environment-reflection model, not a complete physically based
BRDF implementation. Normal-depth consistency and richer material models remain
separate later ablations. Gradients flow through the shader, environment lookup,
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
