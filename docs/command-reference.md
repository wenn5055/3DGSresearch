# Full command reference

For your next run, start with the [short command list](CommandList.md).
This page keeps the detailed setup, logging, evaluation, and older examples.

Run all commands from the repository root in **Git Bash on Windows** using the
Windows Python virtual environment. The individual commands remain available for debugging;
the experiment runner below automates recording, training, and evaluation.

## 1. Activate the virtual environment

```bash
source .venv/Scripts/activate
```

If the virtual environment has not been created yet:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

WSL is a separate Linux environment. A virtual environment created with Linux
Python uses this activation path instead; keep it separate from a Windows-created
virtual environment:

```bash
source .venv/bin/activate
```

## 2. Install dependencies

Only required during initial setup:

```bash
python -m pip install -r requirements.txt
```

## Recorded experiments: single-command runner

With the environment activated, check gradients, train, and evaluate using:

```bash
python scripts/run_experiment.py config/materials_rgb.yaml --check-gradients --note "materials RGB baseline"
```

Each run gets a unique directory under outputs/. The runner preserves the input
YAML, saves fully merged defaults and absolute input paths in its run config,
redirects model outputs into that directory, captures stdout/stderr live, and
stops if a stage fails. It uses the current RGB trainer and evaluator.

```bash
# Optional preparation: check inputs and save the setup without executing kernels.
python scripts/run_experiment.py config/materials_rgb.yaml --prepare-only

# Evaluate on validation views during development.
python scripts/run_experiment.py config/materials_rgb.yaml --eval-manifest transforms_val.json

# Run training and evaluation with PSNR/SSIM only, omitting LPIPS.
python scripts/run_experiment.py config/materials_rgb.yaml --metrics psnr ssim

# Set an unused directory name; otherwise a unique name is generated.
python scripts/run_experiment.py config/materials_rgb.yaml --name materials-rgb-r001
```

Preparation-only mode catches missing manifests/images and lets you inspect the
resolved config before a long job. It does not validate CUDA, render, train, or
check gradients. A normal invocation creates a separate new run; existing run
directories are never reused or overwritten.

The runner matches evaluation resolution, background, device, and explicit tile
capacity to training. It evaluates the test manifest by default with PSNR, SSIM,
and LPIPS (alex backbone, adjustable with --lpips-net). Gradient checking is
optional with --check-gradients and uses the configured device.

| Saved artifact | Contents |
| --- | --- |
| original-config.yaml / config.yaml | Supplied YAML and resolved run configuration |
| run.json / summary.md | Status, commands, environment/Git details, timings, actual completed iterations, and mean per-view metrics |
| source.zip / source-files.json | Python/config source snapshot, including new files, with hashes |
| packages.json / working-tree.patch | Package versions and tracked changes against HEAD |
| input-files.json | Hashes of manifests, their images, and any initialization PLY |
| gradients.log / train.log / eval.log | Stage output; gradient log exists only when requested |
| model.ply / model.png / model_snapshots/ | Trained model and rendered outputs |
| test-renders/ / test-metrics.csv | Evaluation images and per-view metrics for the selected manifest |

Provide the research question with --note, then add your observations, dataset
source/preprocessing details, and interpretation to summary.md. The
[run-record template](run-record-template.md) provides a fuller checklist.
Raw run artifacts remain ignored under outputs/. Copy a compact summary to
docs/experiments/ for Git, adjusting its artifact links, and back up raw artifacts
separately if needed. FPS, peak VRAM, and reflection-crop metrics require separate
measurement. The GPU information is a hardware snapshot, not peak memory usage.

The individual commands below can be used without the runner. Their default
paths may overwrite previous outputs when repeated; use separate configs/output
paths when retaining experiments. To debug a runner-created run, its run.json
contains the exact command for each stage and its config.yaml is runnable with
absolute input paths. Evaluation can be rerun independently of training.

## 3. Check the dataset and setup

```bash
python scripts/check_setup.py
```

This legacy setup script checks Lego only. For materials, use the dataset/config
check in the [short command list](CommandList.md).

## 4. Compile-check the source files

```bash
python -m compileall src scripts shared 3dgs_trainer.py 3dgs_gradient_check_gpu.py gaussian_first_tile_workspace_gpu.py 3dgs_1_syn_trainer.py 3dgs_k_syn_trainer.py image_metrics.py 3dgs_evaluator.py 3dgs_renderer_v1.py
```

## 5. Configure GPU training

Open `config/materials_rgb.yaml` and set:

```yaml
runtime:
  arch: gpu
```

The materials configuration uses random initialization with init_ply: null and
256 x 256 resolution. Set its training.iterations to your intended budget. Early
stopping is enabled in the merged defaults. For a fixed 30,000-step experiment,
set training.iterations to 30000, convergence.enabled to false, and choose the
learning-rate schedule deliberately.

## 6. Run the gradient checks

Run the CPU check:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
```

Then run the GPU check:

```bash
python 3dgs_gradient_check_gpu.py --device cuda:0
```

The analytic gradients should agree with the finite-difference gradients within
the reported tolerance.

## 7. Train the 3DGS model

```bash
python 3dgs_trainer.py config/materials_rgb.yaml
```

Expected outputs include:

```text
outputs/materials-rgb/model.png
outputs/materials-rgb/model.ply
outputs/materials-rgb/model_snapshots/
```

The snapshot directory may contain intermediate PNG and PLY files, for example:

```text
outputs/materials-rgb/model_snapshots/out_000000.png
outputs/materials-rgb/model_snapshots/out_000000.ply
outputs/materials-rgb/model_snapshots/out_000200.png
outputs/materials-rgb/model_snapshots/out_000200.ply
```

## 8. Evaluate the trained model

Run all available metrics:

```bash
python 3dgs_evaluator.py outputs/materials-rgb/model.ply data/materials --arch gpu --width 256 --height 256 --background white --sh-degree 0 --metrics psnr ssim lpips
```

Evaluation results are written to:

```text
outputs/eval/metrics.csv
outputs/eval/
```

## 9. Evaluate without LPIPS

If LPIPS is unavailable or not required:

```bash
python 3dgs_evaluator.py outputs/materials-rgb/model.ply data/materials --arch gpu --width 256 --height 256 --background white --sh-degree 0 --metrics psnr ssim
```

## 10. Evaluate different dataset splits

Evaluate the training views:

```bash
python 3dgs_evaluator.py outputs/materials-rgb/model.ply data/materials --arch gpu --manifest transforms_train.json --metrics psnr ssim lpips --output-dir outputs/eval-train --csv outputs/eval-train/metrics.csv
```

Evaluate the validation views:

```bash
python 3dgs_evaluator.py outputs/materials-rgb/model.ply data/materials --arch gpu --manifest transforms_val.json --metrics psnr ssim lpips --output-dir outputs/eval-val --csv outputs/eval-val/metrics.csv
```

The default evaluation split is:

```text
data/materials/transforms_test.json
```

Training-view scores measure reconstruction of images used for optimization;
they must be labelled separately from held-out validation/test results.

## 11. Evaluate at a different resolution

For example, evaluate at 800 x 800:

```bash
python 3dgs_evaluator.py outputs/materials-rgb/model.ply data/materials --arch gpu --width 800 --height 800 --metrics psnr ssim lpips --output-dir outputs/eval-800 --csv outputs/eval-800/metrics.csv
```

Record evaluation resolution and background for every comparison. The runner
uses the training configuration's width/height and background automatically.

## 12. Run evaluation on the CPU

```bash
python 3dgs_evaluator.py outputs/materials-rgb/model.ply data/materials --arch cpu --metrics psnr ssim --output-dir outputs/eval-cpu --csv outputs/eval-cpu/metrics.csv
```

## 13. View command-line help

```bash
python scripts/run_experiment.py --help
python 3dgs_trainer.py --help
python 3dgs_gradient_check_gpu.py --help
python 3dgs_evaluator.py --help
python 3dgs_renderer_v1.py --help
python 3dgs_1_syn_trainer.py --help
python 3dgs_k_syn_trainer.py --help
```

## 14. Capture logs and provenance for a manual debugging run

This is the step-by-step alternative when you want to inspect each stage.
Choose a fresh run ID, then edit the copied config before recording or training.

```bash
run_id='materials-rgb-debug-r001'
run_dir="outputs/$run_id"
manual_config="config/$run_id.yaml"
prepare_manual_run() {
  if [[ -e "$run_dir" || -e "$manual_config" ]]; then
    printf '%s\n' 'Choose an unused run ID.' >&2
    return 1
  fi
  mkdir -p "$run_dir" && cp config/materials_rgb.yaml "$manual_config"
}
prepare_manual_run
```

Proceed only if preparation succeeds. If the ID is already used, choose a new
run_id and rerun that block before editing configs or writing any logs.

Set paths.output in that config to
`../outputs/materials-rgb-debug-r001/model.png`. Adjust the device, resolution, budget,
and schedule as needed. Its relative paths are resolved against config/, so
keep the runnable YAML there. The materials config selects data/materials and
init_ply: null for random initialization; it has no supplied init.ply to hash.

Save the config and environment after finishing those edits:

```bash
git rev-parse HEAD > "$run_dir/commit.txt"
git status --short > "$run_dir/git-status.txt"
git diff HEAD --output="$run_dir/working-tree.patch"
cp "$manual_config" "$run_dir/config-source.yaml"
cp shared/training_config.py "$run_dir/training_config_source.py"
python --version > "$run_dir/python-version.txt" 2>&1
python -m pip freeze > "$run_dir/packages.txt"
nvidia-smi > "$run_dir/gpu.txt"
printf '%s\n' 'Random initialization; seed is recorded in the saved YAML.' > "$run_dir/initialization.txt"
```

For a future scene-specific initialization PLY, record its SHA-256 hash as well.
The config copy is for reference; its relative paths are not relocated with it.
Git diff excludes untracked files, so preserve new source files separately too.

Run each of the following commands individually. Inspect the log and saved exit
code before proceeding; a nonzero exit code means that stage failed. The setup
script checks Lego specifically; preparation-only mode checks a configured scene. Bash's PIPESTATUS
must be captured immediately after each pipeline to retain Python's exit code
rather than tee's exit code.

```bash
python scripts/check_setup.py 2>&1 | tee "$run_dir/setup.log"
stage_exit=${PIPESTATUS[0]}
printf '%s\n' "$stage_exit" > "$run_dir/setup-exit-code.txt"

python 3dgs_gradient_check_gpu.py --device cpu 2>&1 | tee "$run_dir/gradient-cpu.log"
stage_exit=${PIPESTATUS[0]}
printf '%s\n' "$stage_exit" > "$run_dir/gradient-cpu-exit-code.txt"

python 3dgs_gradient_check_gpu.py --device cuda:0 2>&1 | tee "$run_dir/gradient-gpu.log"
stage_exit=${PIPESTATUS[0]}
printf '%s\n' "$stage_exit" > "$run_dir/gradient-gpu-exit-code.txt"

start_seconds=$SECONDS
python -u 3dgs_trainer.py "$manual_config" 2>&1 | tee "$run_dir/train.log"
stage_exit=${PIPESTATUS[0]}
printf '%s\n' "$stage_exit" > "$run_dir/train-exit-code.txt"
printf '%s\n' "$((SECONDS - start_seconds))" > "$run_dir/train-process-seconds.txt"
```

Use the device check relevant to your work, or both for CPU/GPU comparisons.
Evaluate separately, matching the run's data, resolution, device, and background
(the example below assumes the starter's 256x256, white background, and GPU):

```bash
python -u 3dgs_evaluator.py "$run_dir/model.ply" data/materials --manifest transforms_test.json --arch gpu --width 256 --height 256 --background white --sh-degree 0 --metrics psnr ssim lpips --output-dir "$run_dir/test-renders" --csv "$run_dir/test-metrics.csv" 2>&1 | tee "$run_dir/eval.log"
stage_exit=${PIPESTATUS[0]}
printf '%s\n' "$stage_exit" > "$run_dir/eval-exit-code.txt"
```

Use new log/output names to retain results when rerunning a stage. The runner's
run.json records its exact commands, so you can also debug a particular stage
from an existing recorded run. New kernels need additional gradient checks.

For a tracked manual summary:

```bash
mkdir -p docs/experiments
cp -n docs/run-record-template.md "docs/experiments/$run_id.md"
```

## 15. Check the experiment runner

```bash
python -m unittest discover -s scripts -p test_run_experiment.py -v
```

# Optional Assignment 1 Commands

These scripts belong to the earlier synthetic-rendering exercises and are
separate from the main Assignment 2 Lego training workflow.

## Render a PLY with the Assignment 1 renderer

```bash
mkdir -p outputs
python 3dgs_renderer_v1.py input.ply outputs/render.png
```

Optional camera and image settings:

```bash
python 3dgs_renderer_v1.py input.ply outputs/render.png --width 800 --height 800 --focal-length 700 --camera-position 0 0 0 --look-at 0 0 1 --up 0 1 0
```

## Run the one-Gaussian synthetic trainer

```bash
python 3dgs_1_syn_trainer.py outputs/synthetic-training.png --device cuda:0 --iterations 800
```

## Run the multi-Gaussian synthetic trainer

```bash
python 3dgs_k_syn_trainer.py outputs/multi-synthetic-training.png --device cuda:0 --iterations 800
```

The Assignment 1 synthetic trainers depend on the implementation in
`3dgs_renderer_v1.py`. Their results are separate from the main Assignment 2
Lego training pipeline.

# Essential GPU Workflow

Run each stage separately and resolve failures before proceeding. For automatic
failure handling and experiment records, use the single-command runner above.

```bash
source .venv/Scripts/activate
python scripts/run_experiment.py config/materials_rgb.yaml --prepare-only
python 3dgs_gradient_check_gpu.py --device cuda:0
python 3dgs_trainer.py config/materials_rgb.yaml
python 3dgs_evaluator.py outputs/materials-rgb/model.ply data/materials --arch gpu --width 256 --height 256 --background white --sh-degree 0 --metrics psnr ssim lpips
```
