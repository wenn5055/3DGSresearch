# Run the project in separate passes

Folder goals and changes: [output index](output-index.md), automatically refreshed
by the runners. Add `--note "goal / change"` to describe a new experiment.

Updated **29 October 2026, Asia/Singapore**. Each command creates a fresh dated run
with its own config, source/input snapshot, logs, checkpoint and validation results.
The project modules are separate from the Gaussian/SH baseline used by the completed
learning-rate sweep.

The project runner defaults to the **validation** split. Use
`--eval-manifest transforms_test.json` only for evaluation after settings are frozen.

## Pass 0 — Matched Gaussian RGB control

```powershell
python scripts/run_project.py config/materials_gaussian_control.yaml --name gaussian-control --eval-manifest transforms_val.json --metrics psnr ssim
```

This uses the original 3D Gaussian renderer/optimizer, with initialization colour
statistics computed from training images only. The archived baseline initialized
colour from the combined training/validation tensor; this control removes that
confound from the new geometry comparison. Its PLY is an ordinary Gaussian model.

## Pass 1 — Flattened surfels with fixed RGB

```powershell
python scripts/run_project.py config/materials_surfel_rgb.yaml --name surfel-rgb --eval-manifest transforms_val.json --metrics psnr ssim
```

Each surfel has two positive tangent scales and a quaternion. Its world covariance
is `R diag(sx², sy², 0) Rᵀ`, and its normal is the rotated local z-axis, oriented
toward the current camera before compositing. Local-z scale never enters projection
and has exactly zero gradient. Splitting displaces children within the tangent plane.

The baseline's filtered screen-space EWA projection is retained. This is the
flattened-covariance surfel construction; it does not implement the perspective-correct
ray/plane rasterizer of 2DGS. A screen-space pixel filter keeps edge-on projections finite.

## Pass 2 — Surfels with deferred specular reflection

```powershell
python scripts/run_project.py config/materials_surfel_deferred.yaml --name surfel-deferred --eval-manifest transforms_val.json --metrics psnr ssim
```

This starts independently with the same random seed and training-only initialization.
It warms up RGB geometry for 10,000 steps, then enables per-pixel reflection. The loss
still follows the same validation-MSE-controlled MSE/D-SSIM curriculum as the other passes.

For alpha-compositing weights `wᵢ`, the first pass accumulates RGB `C = Σwᵢcᵢ`, normal
sum `N = Σwᵢnᵢ`, camera-z depth, coverage `A = Σwᵢ`, and reflection sum
`S = Σwᵢ sigmoid(material_logitᵢ)`. The shader uses:

```text
n = normalize(N)
r = incoming_camera_ray - 2 dot(incoming_camera_ray, n) n
s = clamp(S / A, 0, 1)
pixel = (1-s) C + s A environment(r) + (1-A) background
```

The environment is a learned 128×64 latitude/longitude logit texture with periodic
longitude, clamped latitude and bilinear sampling; a sigmoid produces bounded RGB.
It initializes from training-only mean colour. Both environment and reflection strength
are optimized only after the deferred stage activates. Image-loss gradients pass through
the shader and normalized normal into the weighted G-buffers, surfel rotations, alpha,
positions/scales/opacity, material strengths and environment pixels.

This is a learned environment-reflection blend. It has no GGX roughness, Fresnel,
or visibility-aware secondary rays. The three original presets disable geometry
regularization; the additional presets below enable it. This design can
test the project's deferred-normal/shading idea without claiming a complete BRDF model.
Accurate normals and improved full-scene quality remain empirical questions.

## Shared settings and fair comparison

All three presets use 30k steps, capacity 1M, 1,000 initial primitives, seed 7,
800×800 white-background images, position decay over 60k steps, and density fraction
0.2. Early stopping is disabled. All project passes render densely; the archived
baseline's 256-sample 16×16 tiles also covered every pixel at this resolution.
Debug snapshots occur every 1,000 steps, a subset of the 100-step validation ticks.

Compare Pass 0 versus Pass 1 for geometry, then Pass 1 versus Pass 2 for the complete
deferred extension. The latter includes additional lighting/material parameters as
well as deferred computation; it does not isolate deferred versus forward shading.
Use validation to select settings and matched test evaluation after freezing them.
The first full-budget comparison is recorded in [findings](findings.md): deferred
surfels improve mean validation PSNR over RGB surfels by 1.1374 dB for seed 7.
Geometry remains noisy; short smoke checks are not quality comparisons.

## Geometry iterations — foreground masks and normal–depth consistency

Keep the completed three runs as the unregularized reference. These new presets
start independently from scratch with the same 30k budget, capacity, seed and RGB
loss settings. Run one at a time:

```powershell
python scripts/run_project.py config/materials_surfel_rgb_mask.yaml --name surfel-mask --metrics psnr ssim --note "RGB surfels + alpha mask loss"
python scripts/run_project.py config/materials_surfel_rgb_geometry.yaml --name surfel-geometry --metrics psnr ssim --note "RGB surfels + mask and normal-depth losses"
python scripts/run_project.py config/materials_surfel_deferred_geometry.yaml --name deferred-geometry --metrics psnr ssim --note "Deferred surfels + mask and normal-depth losses"
```

- **Mask only:** compare rendered coverage against original training-image alpha,
  using mean binary cross-entropy. This penalizes background opacity even if the
  primitive colour matches white. The loader requires real alpha channels and never
  infers masks from white RGB. Validation/test masks do not supply training gradients.
- **Both losses:** also compare the normalized composited surfel normal with the
  normal from neighbouring rendered depth points, using mean `1 - dot(n, n_depth)`.
  Camera-z depth is back-projected with the camera intrinsics; normal vectors are
  compared in camera coordinates and oriented towards the camera. Gradients flow
  through both the depth and surfel-normal paths, then through alpha compositing
  into positions, rotations, scales and opacity.
- **Reliable neighbourhoods:** the centre and all four neighbours must have target
  alpha ≥0.95 and rendered coverage ≥0.1. Exclude image borders, invalid/degenerate
  depths, and relative neighbour depth jumps >5%. Valid-pixel selection/counting is
  not differentiated. Normal loss averages over valid pixels; no valid pixels gives
  zero normal loss. Mask loss still uses the whole image, including boundaries.
- **Initial settings:** mask weight **0.1** from step 1; normal-depth weight ramps
  from zero at step **2,000** to **0.01** at step **4,000**. These are starting
  hypotheses, not tuned values. The deferred preset still activates reflection at
  step 10,000. Adjust values in the `geometry_loss` config section; both weights
  default to zero so the original presets retain their previous objective.
- **Logs:** `train_photo` is image loss; `mask` and `normal_depth` are unweighted
  terms; `normal_weight` is the effective scheduled weight; `normal_pixels` counts
  valid normals. Training `ema` includes weighted geometry terms. Held-out gates,
  convergence checks and final PSNR/SSIM stay photometric. Normal-loss warm-up must
  finish before early stopping is allowed; these presets disable early stopping.

Current depth is the opacity-normalized blend of surfel-centre camera-z depths,
not exact ray/plane intersection depth. Consistency constrains this approximation;
it does not establish correct geometry or guarantee smooth normals. Preserve this
renderer when comparing loss ablations. Check foreground coverage, depth/normal
maps and validation image scores before selecting loss weights.

The geometry terms follow the mask and consistency ideas in
[Gaussian Surfels §3.3](https://arxiv.org/html/2404.17774v2#S3.SS3), with the explicit
normalization, stencil guards and ramp described above. This is not a full
reproduction of that paper's renderer or loss suite.

### Why the environment map can look plain

The map starts at a constant training-image mean colour, is learned from image loss,
and uses a bounded 128×64 RGB texture. No garden panorama supervises it. Base RGB,
reflection strengths and imperfect normals can explain the observations in multiple
ways, so a recognizable recovered garden is not guaranteed. In the first deferred
run the map does vary with direction (per-channel spatial standard deviations
0.0667 / 0.0726 / 0.0902), while median primitive reflection strength is 0.1009.
It is being learned; inspect lighting/material separation after geometry improves.

## Checkpoints and diagnostics

The surfel passes save `model.ply` plus `model.surfel.npz`. The sidecar records exact
zero-thickness geometry, active shading stage, reflection logits and the environment.
The PLY's third log-scale uses the existing writer's finite compatibility floor;
the surfel renderer always removes that axis exactly. Keep both files together.

Snapshots/final outputs include `.normals.png` (world normals), `.depth.png`
(per-image normalized camera-z depth), `.coverage.png`, and `.buffers.npz` (float
normal/depth/coverage). Deferred outputs add `.reflection.png` (coverage-weighted
reflection strength) and `.environment.png` (learned lighting). Uncovered pixels
retain the background and receive no environment reflection.

```powershell
python scripts/check_surfel_gradients.py --device cpu
python scripts/check_surfel_gradients.py --device cuda:0
```

Add `--check-gradients` to a project run to record the applicable checks before training.
`stage=mse/dssim` identifies the loss; `surfel_stage=rgb/deferred` identifies shading.
The Gaussian control reports `surfel_stage=gaussian_rgb`.

For independent evaluation, use `scripts/evaluate_surfels.py MODEL.ply data/materials`
with `--width 800 --height 800 --manifest transforms_val.json --metrics psnr ssim`.
It restores the saved surfel/material stage. Add `--save-buffers` for per-view diagnostics.
The Gaussian control uses the ordinary `3dgs_evaluator.py`.

For a development warm start, copy the deferred config and set `paths.init_ply` to a
Pass-1 PLY; its adjacent surfel sidecar is loaded automatically. Set
`surfel.specular_start_iteration: 0` to refine immediately. Optimizer state starts
fresh. Record the source checkpoint and total prior training budget; do not compare
that result as a fresh 30k-step run against the from-scratch passes above.

## References

- [Gaussian Surfels](https://arxiv.org/abs/2404.17774): zero local-z scale and surface normals.
- [3DGS with Deferred Reflection](https://arxiv.org/abs/2404.18454): per-pixel reflection gradients.
- [Official deferred renderer](https://github.com/gapszju/3DGS-DR/blob/main/gaussian_renderer/__init__.py): normal compositing and environment reflection.
