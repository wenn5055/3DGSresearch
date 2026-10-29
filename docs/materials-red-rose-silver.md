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

Current full sweep: [live summary](../outputs/materials_red-rose-silver-Oct29-1656-deferred-propagation-sweep/summary.md).
The 1.5 factor is running first. Follow
`outputs/materials_red-rose-silver-Oct29-1656-deferred-propagation-sweep/launcher.log`
for candidate progress; each candidate's run folder contains its live `train.log`.

The deferred preset now uses **0.8 L1 + 0.2 D-SSIM + 0.1 mask BCE** from step 1.
The former depth-normal penalty is disabled. The RGB command above remains the
historical comparison; use deferred training for the new normal method.

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
sweep. Override factors with `--scales 1 1.3 1.5`. The old `--weights` grid applies
only to historical configs without paper normal propagation.

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

Verification includes analytical/finite-difference image, mask, quaternion and
reflection gradients; propagation thresholds, inactive slots and normal-axis
preservation; opacity freeze/interleaving; and prevention of last-step opacity
clamping. A short full-runner sweep also exercised training, validation and winner
selection. These short checks verify operation, not final reconstruction quality.
