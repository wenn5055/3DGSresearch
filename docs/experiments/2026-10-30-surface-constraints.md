# Surface constraints: a remaining gap in the deferred model

Reviewed 30 October, 17:07 Singapore. This is source analysis and a hypothesis;
it is not a new trained result or an implementation of the cited method.

The proposal's surfel/deferred combination is feasible in published work.
[RGS-DR](https://github.com/gkouros/RGS-DR) implements pixel-deferred 2DGS with
roughness-aware environmental lighting and a directional residual. Our current
renderer is a simpler rank-two EWA surfel adaptation. The HDR/angular option is
still a mirror-oriented appearance model, not the proposal's complete roughness
BRDF. Adding iterations cannot supply missing shading or surface constraints.

| Proposal item | Current implementation and evidence |
| --- | --- |
| Explicit surface normals | Rank-two covariance surfels; normals can still drift and overlap incoherently. |
| Per-pixel deferred shading | Implemented, with verified geometry/material adjoints. |
| Roughness BRDF | Not implemented: the learned scalar is reflection blend, not roughness. HDR/angular shading remains a mirror-focused variant. |
| Replace SH appearance | Current quality controls retain SH3 as a residual; they test a hybrid rather than the proposed four-scalar appearance budget. |
| Materials and Ship | Original Materials is primary; enlarged Chrome is auxiliary. Fresh Ship controls remain unrun. |
| Metrics and cost | PSNR/SSIM/LPIPS, file size and rendering timing; reported training memory is the Warp allocation peak, not total device VRAM. |

## A concrete loss difference

Let A be accumulated alpha, S the weighted sum of unit surfel normals, and n_d
the unit normal inferred from rendered depth. Our normal/depth loss normalizes
S, then penalizes `1 - dot(S / |S|, n_d)`. It measures mean normal alignment but
does not directly penalize dispersion among contributors to the same pixel.

RGS-DR retains the unnormalized rendered normal and multiplies its depth-derived
normal by detached alpha. See its
[normal preparation](https://github.com/gkouros/RGS-DR/blob/main/gaussian_renderer/__init__.py#L242)
and [normal/distortion losses](https://github.com/gkouros/RGS-DR/blob/main/train.py#L129).
At opaque pixels, its normal term consequently also discourages cancellation
among contributing normals. This is an inference from the source, not a claim
that the two implementations have identical normalization or gradients.

RGS-DR additionally uses the rasterizer's depth-distortion statistic to encourage
thin surfaces. We currently retain expected depth and lack an equivalent
distortion penalty. Better expected depth alone does not imply a coherent thin
surface or accurate reflection normals; our saved original-hull trajectory
already shows depth improving while normal error worsens.

The reference's raw loss weights are not portable: normalization, rasterization,
valid masks and image objectives differ. Copying its numeric constants would
not constitute a controlled fix. Before adding another training variant, inspect
the protected pair's measured normal coherence, depth and Chrome crops.

## What SH can and cannot fix

[Unit 11](https://weitsang.github.io/gaussian-splatting-notes/unit-11/) explains
smooth directional appearance and progressive SH bands. Degree 3 has 16 basis
functions, hence 48 RGB scalars total, or 45 beyond RGB. SH is useful for broad
residual view dependence. It does not repair geometric normals, and low-order
SH cannot directly represent arbitrary sharp reflected illumination.

Reflection is sensitive to normal direction: for coplanar angular perturbations,
a normal rotation produces twice that angular change in the reflected ray.
Our measured errors of tens of degrees are therefore a substantial obstacle to
sharp environment details. Even the auxiliary Chrome run's smaller errors remain
material for this scene. This is a sensitivity argument, not a prediction of a
specific PSNR gain from reducing normal error.

## Ordered next decisions

1. Finish the protected matched controls and inspect their geometry and images.
2. Test the separate HDR/angular shader with those geometry settings fixed.
3. If geometry remains incoherent, test a direct coherence/thin-surface constraint
   before spending more iterations on appearance capacity.
4. Roughness-aware filtering is needed for the rough materials; position-dependent
   transport/residuals are needed for nearby reflected objects. These solve
   different limitations and should follow a stable far-field mirror result.

Source URLs and relevant line ranges were inspected directly from the official
repository on this date. No external renderer or copied implementation was
installed into the training environment. The course notes were also rechecked
against the previously retained HTML/text snapshots after today's web fetch failed.

The proposal's Ship scene is also not a purely diffuse control: direct inspection
of its reference images shows reflective water around the vessel. If evaluated,
treat whole-scene scores as a mixed-appearance secondary benchmark rather than
evidence restricted to diffuse geometry.
