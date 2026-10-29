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

## 3. Train deferred surfels with the same losses

```bash
python scripts/run_project.py \
  config/materials_red-rose-silver_deferred_geometry.yaml \
  --name deferred-normal --metrics psnr ssim
```

Run training commands sequentially. Both start from scratch for 30,000 steps,
use alpha-mask weight 0.1 and ramp normal–depth weight to 0.01 over steps
2,000–4,000. Deferred reflection starts at step 10,000. The runner evaluates on
validation views and saves dated run folders under `outputs/`, with model,
metrics, and normal/depth/coverage diagnostics. Test views remain available for
final evaluation after choosing settings on validation.

The original RGB/deferred geometry presets supply all other settings. These are
starting configurations; they have not been tuned for this smaller scene.

Checked locally: three 128 × 128 GPU preview renders, exact pose/FOV metadata,
RGBA foreground, capture resume, and three GPU training steps for each preset.
Preview data/logs are under `outputs/red-rose-silver-capture-verified/`.
These checks verify the workflow; full capture and 30,000-step training have not run.
