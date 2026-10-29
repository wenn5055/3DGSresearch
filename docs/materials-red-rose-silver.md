# Capture and train the edited Materials scene

Run these Git Bash commands from the repository root. The saved scene contains
three balls (six mesh parts); capture uses the file as saved.

## 0. Activate the project environment

```bash
source .venv/Scripts/activate
export WARP_CACHE_PATH="$(pwd -W)/outputs/warp-cache"
```

This selects the project's Python 3.12.10 environment. Use `python` in the commands below.

## 1. Capture the dataset

```bash
"/c/Program Files/Blender Foundation/Blender 5.0/blender.exe" \
  --background --disable-autoexec \
  data/blender/materials_red-rose-silver.blend \
  --python-exit-code 1 \
  --python scripts/capture_blender_views.py \
  -- --output data/materials_red-rose-silver
```

This renders 100 training, 100 validation and 200 test views at 800 × 800,
128 Cycles samples with GPU rendering and denoising. It replays the original
Materials camera poses/FOV, preserves object positions and scene colour management,
and exports RGBA PNGs plus `transforms_train.json`, `transforms_val.json` and
`transforms_test.json`. Camera matrices use Blender camera-to-world coordinates.
Transparent pixels become white in the trainer; alpha also supplies mask supervision.

The Blender file is never saved by the script. `capture.json` records the source,
settings, template hashes and image hashes. To continue an interrupted capture,
repeat the command with `--resume` at the end. The scene, script and capture
settings must match. Wait for `Capture complete` before training.

The normal loss derives its depth target from the renderer during training;
Blender ground-truth normal/depth passes are not needed by these presets.

## 2. Train RGB surfels with mask and normal losses

### Completed consistency comparison — tuning closed

The [matched sweep](../outputs/materials_red-rose-silver-Oct29-1852-deferred-normal-sweep/summary.md)
selected weight 0 for validation RGB quality (34.5822 dB versus 34.0690 for 0.01).
The 0.01 run has smoother normal/depth previews and remains the regularized
comparison. At the user's request, stop tuning here and move on to final renders
and write-up; Blender reference-normal/depth evaluation is deferred.
The command below is retained to reproduce the comparison.

Final test evaluation and write-up are complete:
[experiment record](experiments/2026-10-29-normal-consistency-final.md) and
[matched figure gallery](../outputs/materials-red-rose-silver-final-comparison-2026-10-29/index.html).
Weight 0 achieved 34.5007 test dB versus 34.1435 for 0.01; the latter has smoother
normal diagnostics. All 200 test cameras were evaluated at 400x400.

The optimizer fixes and quiet final refinement are implemented; see the
[design update log](project-plan.md#design-update-log) and
[pre-fix audit](training-code-audit-2026-10-29.md).
Reproduce the matched **normal-depth weight 0 versus 0.01** experiment:

```bash
python scripts/tune_deferred_normal_loss.py config/materials_red-rose-silver_deferred_consistency.yaml --weights 0 0.01
```

Both candidates use 400px images, capacity 200k, seed 7, 30k steps, L1 + D-SSIM +
mask BCE, propagation disabled and fixed topology for the final 10k steps.
Position decay ends at 30k. Only normal-depth coefficient varies; the 0.01 arm
ramps from 2k to 4k. Standalone presets:
[control](../config/materials_red-rose-silver_deferred_consistency_control.yaml) and
[consistency](../config/materials_red-rose-silver_deferred_consistency.yaml).
The D-SSIM definition stays `(1-SSIM)/2`. Inspect normal/depth and foreground
quality alongside the validation image score; final quality is not established
by the implementation checks.

### Earlier RGB geometry comparison

```bash
python scripts/run_project.py \
  config/materials_red-rose-silver_rgb_geometry.yaml \
  --name rgb-normal --metrics psnr ssim
```

## 3. Deferred normals: paper method plus mask

Smaller preset: [deferred mini](../config/materials_red-rose-silver_deferred_mini.yaml).
It uses 400x400 images, a 200,000-splat cap and all 30,000 steps, preserving the
3,000-step reflection activation and 24,000-step propagation cutoff. Position
learning-rate decay finishes at 30,000 steps. It is prepared and schema-validated;
reconstruction quality has not yet been measured for this preset.

```bash
python scripts/run_project.py config/materials_red-rose-silver_deferred_mini.yaml --name mini-paper-mask --metrics psnr ssim
```

Completed propagation sweep:
[summary](../outputs/materials_red-rose-silver-Oct29-1656-deferred-propagation-sweep/summary.md).
Factor 1.5 won on validation PSNR; this selection does not establish normal accuracy.
That archived sweep predates the optimizer fixes and quiet refinement. New full/mini
30k propagation presets keep topology fixed after 24k, and the 10k preset after 8k.

The deferred preset now uses **0.8 L1 + 0.2 D-SSIM + 0.1 mask BCE** from step 1.
Depth-normal consistency is disabled in these propagation presets. It is enabled
in the new consistency preset above. The RGB command remains an earlier comparison.

[3DGS-DR sections 3.2-3.3](https://arxiv.org/html/2404.18454v1#S3.SS2) has no separate
normal-loss equation. Normals receive image gradients through reflected environment
lookups. Its normal-propagation procedure increases overlap so these gradients
reach neighbouring primitives. The implementation follows that procedure:

- RGB bootstrap until step 3,000, then reflection/environment optimization.
- Propagation every 1,000 steps until step 24,000: opacity floor 0.9, reflection
  floor 0.001, tangent scale factor 1.5 for reflection strength at least 0.02.
- Add uniform +/-0.1 RGB noise to primitives with reflection strength at most 0.1,
  following the paper's color-sabotage idea. Reflection gradients optimize normals;
  normals are not directly copied or fitted to a depth-map target.
- Interleave opacity clamps every 3,000 steps with propagation. Never apply both
  on the same step, and never clamp opacity on the final step.
- Freeze opacity updates for 200 steps after propagation, matching the released
  code's stabilization strategy. Density updates occur every 500 steps during
  propagation, with the ordinary interval used outside that stage.
- Keep alpha-mask supervision active throughout; refine without propagation for
  the last 6,000 steps.

Run one fresh 30,000-step experiment:

```bash
python scripts/run_project.py \
  config/materials_red-rose-silver_deferred_geometry.yaml \
  --name paper-normal-mask --metrics psnr ssim
```

Or automatically compare propagation scale factors **1.5 / 1.0 / 1.25**, with the
paper's 1.5 tried first and every other setting fixed:

```bash
python scripts/tune_deferred_normal_loss.py \
  config/materials_red-rose-silver_deferred_geometry.yaml
```

That is three fresh 30,000-step runs. The factor 1 control retains opacity and
color interventions but does not enlarge tangents. The sweep waits for recorded
GPU jobs, evaluates every validation view, and selects mean validation PSNR with
SSIM breaking exact ties. It does not use test scores. Inspect normal, depth,
reflection and coverage maps alongside image scores before drawing conclusions.

The printed sweep directory contains `summary.md`, `validation-results.csv`,
`sweep.json` and, after selection, `best-config.yaml` plus the winning run path.
Positive scale-expansion candidates must actually log enlarged reflective
primitives. Incomplete runs and runs that never activate deferred shading cannot win.

Add `--prepare-only` to inspect without starting. To continue a prepared or
interrupted sweep:

```bash
python scripts/tune_deferred_normal_loss.py --resume outputs/YOUR_SWEEP_DIRECTORY
```

Completed candidates are kept; an interrupted candidate restarts from scratch.
Keep training/evaluation sources and train/validation inputs unchanged during a
sweep. Override factors with `--scales 1 1.3 1.5`. Use `--weights` for configurations
with propagation disabled, including the new consistency comparison above.

The previous depth-normal sweep was interrupted at the user's request. Its
snapshots and recorded RGB result remain under `outputs/`; do not resume that
old sweep with the replacement source.

The renderer still uses rank-two projected surfels and a latitude-longitude learned
environment, rather than the paper's complete 3DGS/cubemap system. This is an
adaptation of its normal optimization. No ground-truth sphere geometry is used.
The scheduled cutoff, interleaving and opacity freeze follow the
[released training loop](https://github.com/gapszju/3DGS-DR/blob/main/train.py);
reflection thresholds and tangent expansion also follow the
[released model updates](https://github.com/gapszju/3DGS-DR/blob/main/scene/gaussian_model.py).

## Ray-plane depth experiment

The new preset changes only depth mode and output destination relative to
`materials_red-rose-silver_deferred_consistency.yaml`:

```bash
python scripts/run_project.py \
  config/materials_red-rose-silver_deferred_ray_plane.yaml \
  --name ray-plane-depth --metrics psnr ssim
```

It computes each surfel's per-pixel plane-intersection depth and differentiates
through it. The existing normal-depth weight remains 0.01. Growth/pruning settings
are unchanged: 1k initial surfels, 200k maximum, the same density schedule, and no
growth, pruning or opacity resets for the final 10k steps. No additional normal
smoothing or depth-distortion penalty is enabled. EWA footprints and centre-based
sorting remain; degenerate plane intersections fall back to centre depth. This
isolates a depth-model change; improved crater recovery still needs evaluation.

Verification includes analytical/finite-difference image, mask, quaternion and
reflection gradients; propagation thresholds, inactive slots and normal-axis
preservation; opacity freeze/interleaving; and prevention of last-step opacity
clamping. A short full-runner sweep also exercised training, validation and winner
selection. These short checks verify operation, not final reconstruction quality.
