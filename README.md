# CP4282 Gaussian Splatting — Assignment 2 (Training)

Starter code and data for the CP4282 Gaussian Splatting **training** assignment.

This repository intentionally contains incomplete teaching implementations. The missing sections
are the work: read the corresponding unit in the course notes, implement the marked functions,
and use the supplied checks before moving on.

## Setup

```bash
git clone https://github.com/wenn5055/3DGSresearch.git
cd 3DGSresearch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`torch` and `lpips` are in that list only for the evaluator's LPIPS metric. If you would rather not
pull PyTorch down, install everything above them and run the evaluator with `--metrics psnr ssim`.

The repository contains a small Lego dataset under `data/lego/`. The three manifests sit at the
top of that directory and name the images in the split directories beside them:

- `train/` — 100 posed training images, listed by `transforms_train.json`.
- `val/` — 100 validation images, listed by `transforms_val.json`, for training-time held-out checks.
- `test/` — 200 held-out images, listed by `transforms_test.json`, for final evaluation.
- `init.ply` — the 45,733-splat point cloud training starts from.

The three splits share no images.

The trainer reads **both** manifests. It optimises only against `train/`; it renders eight
held-out views from `val/` every `reporting.eval_every` iterations to report `fixed_eval`, so
that number measures generalisation rather than a second reading of the training loss. The
validation views are never used to compute a gradient. The trainer prints exactly which frames it
picked on its first line, and `reporting.eval_manifest: null` reverts to evaluating on training
views if you want the old behaviour.

## The assignment

1. `3dgs_trainer.py`: implement the backward rendering pass. Two kernels are marked TODO,
   `render_backward` for the dense path and `render_sparse_backward` for the sampled one.

```bash
python 3dgs_trainer.py config/3dgs_training_gpu.yaml
```

Check your gradients against finite differences before judging a training run:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
```

As shipped this check fails, reporting an analytic gradient of zero: nothing accumulates into the
gradient buffers until you write the kernels. Training also runs to completion without improving,
for the same reason.

The annotated walkthrough explains the whole file without giving the TODO solution:

- `3dgs_trainer_annotated.md`

Shared and support files:

- `shared/trainable_gaussian.py`: trainable splat parameter storage used by the trainer
  (documented in `shared_annotated.md`)
- `shared/training_config.py`: the YAML schema, defaults and validation
- `gaussian_first_tile_workspace_gpu.py`: tile record construction used by the trainer
- `config/3dgs_training_gpu.yaml`: the starter YAML for training runs. Its paths are resolved
  relative to the config file, and `shared/training_config.py` lists every key it may contain.
- `3dgs_renderer_v1.py`, `3dgs_1_syn_trainer.py`, `3dgs_k_syn_trainer.py`: Assignment 1's renderer
  and the two small Unit 8 synthetic trainers. **These carry Assignment 1's own unfinished
  `render` loop.** The synthetic trainers build their ground truth by calling it, so until you
  paste your Assignment 1 solution into `3dgs_renderer_v1.py` they train against an all-black
  target and their PSNR figures are meaningless. Assignment 2's task does not depend on any of
  these three files.

## Evaluating your output

To train and evaluate in a new run folder:

```bash
python scripts/run_experiment.py config/materials_rgb.yaml --metrics psnr ssim
```

Folders use the dataset name and Singapore time, e.g. `outputs/materials-Oct28-1305/`.
Add `--name cap1k` to get `materials-Oct28-1305-cap1k`. Repeated names receive `-2`,
`-3`, etc. Exact dates and UTC timestamps are saved in the run metadata.

The Materials config enables the Unit 10 loss curriculum: MSE warm-up, then dense
`0.8 * MSE + 0.2 * D-SSIM` with reflected 11x11 windows. It switches when the smoothed
validation MSE reaches 21 dB PSNR (`adaptive.dssim_psnr_threshold`), chosen below the
previous Materials MSE plateau. Configs without a `loss` block keep MSE-only training;
set `loss.mode: mse` for a baseline comparison. The full settings are:

```yaml
loss:
  mode: mse_dssim
  dssim_weight: 0.2
  dssim_window: 11
adaptive:
  dssim_psnr_threshold: 21.0
```

The gate uses held-out MSE, and early stopping starts only after D-SSIM activates.
Logs show `stage`, `loss_phase`, the blended `fixed_eval_ema`, and the separate
`fixed_eval_mse_ema`. A transition resets the objective baseline so the two loss scales
are not compared. The final stage is saved in the run summary; if the gate is never
reached, the trainer explicitly reports that the run used MSE only. Final PSNR and SSIM
evaluation still uses the rendered images and ordinary MSE, independent of training loss.

`3dgs_evaluator.py` renders every camera in a manifest, saves the images, and reports the metrics
you ask for:

```bash
python 3dgs_evaluator.py outputs/warp-training.ply data/lego
```

It decides three things for you rather than making you pick a script:

- **Device.** `--arch auto` (the default) uses CUDA when Warp reports it and CPU otherwise. Force
  it with `--arch cpu` or `--arch gpu`; asking for gpu without CUDA is an error, not a slow
  fallback.
- **Appearance.** If a `<ply-stem>.sh.npz` sidecar sits beside the PLY, the render uses degree-2
  view-dependent colour. `--sh-degree 0` ignores it, `--sh-degree 2` requires it, and `--sh-npz`
  points at one stored elsewhere. This assignment produces no sidecar, so this stays off.
- **Metrics.** `--metrics` takes any of `psnr`, `ssim`, `lpips`; all three run by default. LPIPS
  needs PyTorch, so drop it with `--metrics psnr ssim` if you have not installed the optional
  requirements.

Use `--manifest transforms_train.json` or `--manifest transforms_val.json` to score against those
splits instead of the held-out test split, and `--width`/`--height` to render at a different
resolution than the default. The test manifest has 200 views; on CPU, evaluate a smaller split
first.

## Running checks

```bash
python -m compileall src scripts shared 3dgs_trainer.py 3dgs_gradient_check_gpu.py gaussian_first_tile_workspace_gpu.py 3dgs_1_syn_trainer.py 3dgs_k_syn_trainer.py image_metrics.py 3dgs_evaluator.py 3dgs_renderer_v1.py
python scripts/check_setup.py
python -m unittest discover -s scripts -p "test_*.py"
```

After implementing `3dgs_trainer.py`'s backward pass, verify it with a finite-difference gradient
check:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
python 3dgs_gradient_check_gpu.py --device cuda:0
```

Use `--help` on each script for its command-line arguments. Start with low resolution and a small
iteration count while debugging.
