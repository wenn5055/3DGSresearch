# Materials: commands to run

Use **Git Bash on Windows**, from the repository root.

## Project passes (28 October 2026)

These are independent runs; the project runner evaluates validation by default.
Keep the shared settings matched when comparing them. Details, warm starts and
diagnostics: [project-passes.md](project-passes.md).

```bash
python scripts/run_project.py config/materials_gaussian_control.yaml --name gaussian-control --metrics psnr ssim
python scripts/run_project.py config/materials_surfel_rgb.yaml --name surfel-rgb --metrics psnr ssim
python scripts/run_project.py config/materials_surfel_deferred.yaml --name surfel-deferred --metrics psnr ssim
```

Run them one at a time for interpretable timing. The deferred pass fits RGB geometry
for 10k steps before enabling learned environment reflection. All three presets use
30k steps and disable early stopping. Use `--prepare-only` to inspect a run's resolved
settings, or `--check-gradients` to record checks before training.

The commands below remain the original Gaussian/RGB baseline workflow.

## Geometry-loss iterations (29 October 2026)

```bash
python scripts/run_project.py config/materials_surfel_rgb_mask.yaml --name surfel-mask --metrics psnr ssim
python scripts/run_project.py config/materials_surfel_rgb_geometry.yaml --name surfel-geometry --metrics psnr ssim
python scripts/run_project.py config/materials_surfel_deferred_geometry.yaml --name deferred-geometry --metrics psnr ssim
```

These add mask supervision, then normal–depth consistency, then deferred reflection.
Each is a fresh 30k-step validation run. Weights and diagnostics:
[project-passes.md](project-passes.md).

## 1. Activate Python

```bash
source .venv/Scripts/activate
```

## 2. Check your dataset and config

```bash
python scripts/run_experiment.py config/materials_rgb.yaml --prepare-only
```

Look for **Run prepared**. Open the printed run folder's `config.yaml` if you
want to check the settings. This step checks files and saves the setup.

## 3. Run your first experiment

For a short first test, set `training.iterations: 200` in
[materials_rgb.yaml](../config/materials_rgb.yaml) before running:

```bash
python scripts/run_experiment.py config/materials_rgb.yaml \
  --check-gradients --metrics psnr ssim \
  --eval-manifest transforms_val.json
```

This checks gradients, trains, then evaluates validation views. Logs are
saved automatically. Stop and inspect the stage's log if it fails.

Each invocation creates a new folder under `outputs/`; its path is printed
in the terminal. Preparation and training create separate folders.

## 4. Find your results

Open the experiment folder printed by step 3:

| Open | To see |
| --- | --- |
| `summary.md` | Status, completed iterations, and average scores |
| `train.log` / `eval.log` | Terminal output and errors |
| `model.png` / `test-renders/` | Rendered images |
| `test-metrics.csv` | Per-view scores for the selected split (validation here) |

The run folder also keeps the exact config, source snapshot, and provenance.
Add your observations to the summary after looking at the images.

## Individual commands for debugging

Run only the stage you need. Direct training uses the output path in your YAML,
currently `outputs/materials-rgb/model.png`, and saves a matching `model.ply`.

**Check gradients** (choose GPU or CPU):

```bash
python 3dgs_gradient_check_gpu.py --device cuda:0
python 3dgs_gradient_check_gpu.py --device cpu
```

**Train only:**

```bash
python 3dgs_trainer.py config/materials_rgb.yaml
```

**Evaluate that model only:**

```bash
python 3dgs_evaluator.py outputs/materials-rgb/model.ply data/materials \
  --manifest transforms_val.json --arch gpu \
  --width 256 --height 256 --background white --sh-degree 0 \
  --metrics psnr ssim \
  --output-dir outputs/materials-rgb/eval-val \
  --csv outputs/materials-rgb/eval-val/metrics.csv
```

For a runner-created model, replace the PLY path with that run's `model.ply`.
Match resolution and background to the saved config. The runner handles logging;
direct commands print to the terminal unless you capture their output.

## When you need more

See the [full command reference](command-reference.md) for manual logging,
LPIPS, final test-set evaluation, other resolutions, setup, and older assignment
commands. Use `--help` to list runner options:

```bash
python scripts/run_experiment.py --help
```
