# Deferred Specular Shading with 2D Gaussian Surfels

Working documents: [research stages](researchStages.md), [todo list](TODO.md),
[run-record template](run-record-template.md), and [commands](CommandList.md).

Detailed rationale, literature checks, and current evidence:
[research design record](research-design-notes.md).

## Purpose and status

Extend the custom Python/NVIDIA Warp 3D Gaussian Splatting trainer in
`3dgs_trainer.py` with oriented 2D surfels and deferred specular shading.
This document records the intended project scope; it does not report completed
implementation or experimental results.

## Research question and baseline

Does deferred specular shading using explicit 2D surfel normals improve held-out
image reconstruction on the NeRF-synthetic materials and ship scenes relative to
the existing Unit 9 trainer, and what training-time and VRAM costs does it add?

The current trainer uses trainable, view-independent RGB through
`colour_at_view`, rather than degree-3 spherical harmonics (SH). The proposal's
baseline table assumes degree-3 SH and needs to be reconciled with the code.
An RGB-baseline comparison measures the overall extension; a claim about
outperforming SH would require an explicitly implemented or selected SH baseline.

## Intended implementation

1. Introduce oriented 2D Gaussian surfels with explicit unit normals. The proposal
   suggests two tangent vectors; finalize the parameterization and projection
   before implementation.
2. Extend rasterization to alpha-composite normal, depth, and diffuse-color
   G-buffers, along with the attributes required by shading.
3. Add a per-pixel deferred specular pass using camera rays, normals, and material
   parameters. Define the illumination representation and BRDF, including how
   roughness enters the model.
4. Propagate image-loss gradients through shading and G-buffer compositing to
   surfel geometry and appearance. Validate new gradients with finite differences
   before relying on training results.

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
