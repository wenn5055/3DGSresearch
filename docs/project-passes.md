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
visibility-aware secondary rays, or normal-depth consistency regularizer yet. It can
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
No full-resolution quality improvement is established by the short smoke checks.

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
