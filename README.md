# CP4282 Gaussian Splatting — Assignment 2 (Training)

Starter code and data for the CP4282 Gaussian Splatting **training** assignment.

Research extensions now include staged D-SSIM, SH degrees 0–3, and independent
Gaussian-control / RGB-surfel / deferred-surfel passes. Start with
[project passes](docs/project-passes.md) for the new experiments and
[findings](docs/findings.md) for measured baseline results. The surfel passes are
tested implementations; their full Materials quality comparisons remain to be run.
For a one-line explanation of each output folder, see the auto-updated
[output index](docs/output-index.md).

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

### Spherical-harmonic baselines

`model.sh_degree` selects the maximum appearance degree: `0` (the default) keeps
fixed RGB; `1`, `2`, or `3` enable directional colour. Materials presets are provided:

```bash
python scripts/run_experiment.py config/materials_sh2.yaml --name sh2 --metrics psnr ssim
python scripts/run_experiment.py config/materials_sh3.yaml --name sh3 --metrics psnr ssim
```

Both presets copy the current RGB experiment settings and change the appearance
settings and output path. Compare saved configs when changing any preset later.
They first fit RGB, switch to D-SSIM at 21 dB, then unlock SH bands at 22, 23, and
24 dB of smoothed held-out MSE PSNR. Degree 2 stops unlocking after the second band.
The gates are `adaptive.sh1_psnr_threshold`, `sh2_psnr_threshold`, and
`sh3_psnr_threshold`; each can advance only on an evaluation tick, after the loss
warm-up. Early stopping waits until the configured maximum degree unlocks. If a gate
is never reached, the log and summary report the actual degree that was trained.

The rest-coefficient learning rate is `learning_rates.feature_rest` (0.000125 by
default). Locked bands have zero gradients and untouched optimizer state. Each band
has its own Adam step count; density control copies the parent's coefficients to
its children and resets SH optimizer state with the geometry optimizer.

The existing per-splat RGB is the DC term. The higher bands add 3, 8, or 15 RGB
coefficient vectors for degrees 1, 2, or 3. Colour is evaluated once per splat/view
and reused by both dense and sparse rendering. Gradients include the appearance
path through the splat-to-camera direction back to position.

Every SH checkpoint consists of the ordinary RGB/geometric `.ply` plus a `.sh.npz`
sidecar with coefficients in the same active-splat order. The sidecar records the
active and maximum degrees and the basis/direction conventions. These follow the
course's positive polynomial signs and splat-to-camera direction, with direct RGB
as DC; they are not interchangeable with Graphdeco's SH checkpoint format.
Keep both files together. Evaluation automatically loads the saved active degree;
`--sh-degree 0` previews base RGB, and `--sh-degree 1/2/3` can truncate a checkpoint
to a lower trained degree. Asking for an untrained degree or a missing explicitly
requested sidecar is an error.

### Tune the SH learning rate on validation

The sweep compares half/default/double `learning_rates.feature_rest`
(`0.0000625`, `0.000125`, `0.00025`), changing no other training settings:

```bash
python scripts/tune_sh_learning_rate.py config/materials_sh3.yaml
```

Candidates run sequentially and evaluate all validation views. Selection uses mean
validation PSNR, with SSIM breaking exact ties. A candidate must finish the configured
iteration budget and reach the configured SH degree to be eligible. The sweep saves
`summary.md`, `validation-results.csv`, `sweep.json`, and the selected `best-config.yaml`
under a dated `outputs/*-lr-sweep/` directory. Selection is provisional for one seed;
test scores are not used to choose parameters.

Use `--baseline-run outputs/EXISTING_RUN` to reuse a matching baseline model and
re-evaluate it on validation. Configs and trainer/evaluator source hashes must match.
Add `--wait-for outputs/RUN_A outputs/RUN_B` to wait for existing jobs before starting
new GPU work. Add `--prepare-only` to save a plan without launching experiments.
Continue a prepared/interrupted sweep with:

```bash
python scripts/tune_sh_learning_rate.py --resume outputs/SWEEP_DIRECTORY
```

Keep training/evaluation source files unchanged while a sweep runs; it stops if they
change, preserving evidence instead of ranking different implementations together.

`3dgs_evaluator.py` renders every camera in a manifest, saves the images, and reports the metrics
you ask for:

```bash
python 3dgs_evaluator.py outputs/warp-training.ply data/lego
```

It decides three things for you rather than making you pick a script:

- **Device.** `--arch auto` (the default) uses CUDA when Warp reports it and CPU otherwise. Force
  it with `--arch cpu` or `--arch gpu`; asking for gpu without CUDA is an error, not a slow
  fallback.
- **Appearance.** If a `<ply-stem>.sh.npz` sidecar sits beside the PLY, the render uses its
  saved active SH degree. `--sh-degree 0` ignores it, `--sh-degree 1/2/3` selects a trained
  degree, and `--sh-npz` points at a sidecar stored elsewhere.
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
