# Experiment record template

Copy this file to `docs/experiments/<run-id>.md` for a tracked summary of a real
run. Store its raw artifacts under the already gitignored `outputs/<run-id>/`.
Replace placeholders using evidence; use "unknown" for unavailable information.

Terminal logs are useful, but they do not replace the exact configuration, code
revision, dataset/splits, and environment. Preserve those before running.

## Identity and purpose

- Run ID:
- Date/time with timezone:
- Stage / model variant:
- Question being tested; comparison run ID(s):
- Status: planned / running / completed / failed / exploratory

## Code, environment, and inputs

- Git commit:
- Working tree clean? If not, saved patch and any new source/config files:
- Command(s) and repository working directory:
- GPU/model/VRAM, driver, OS, Python, Warp and dependency snapshot:
- Dataset source/version, local path, and any preprocessing:
- Train / validation / test manifests and selected frame IDs:
- Initialization point cloud/source/hash or random initialization policy:
- Input and evaluation image resolution; background/alpha/color handling:

## Configuration and execution

- Original config path and saved copy:
- Defaults snapshot or fully resolved config (omitted YAML keys still matter):
- Seed(s):
- Planned / completed iterations; early-stopping policy and actual stop reason:
- Learning rates/schedules, loss terms/weights, densification/pruning, capacity:
- Geometry, appearance, lighting, and model-specific settings:
- Start/end times, total process wall time, trainer-reported time, and what each includes:
- Exit code(s); log paths:

## Evaluation

- Evaluation command, code revision, device, and checkpoint/model hash:
- Split: validation / test / training (label training-view results explicitly):
- Metric implementation/settings and aggregation (e.g. mean per-view PSNR):
- LPIPS backbone; omitted metrics and reasons:
- PSNR (higher is better):
- SSIM (higher is better):
- LPIPS (lower is better):
- Rendered image and per-view metric CSV paths:
- Crop definitions: view IDs, image dimensions, and pixel bounds [x0, y0, x1, y1),
  identical for reference and every compared method; do not resize for scoring:
- Full-image versus crop results (report separately):
- FPS/resolution/device, warmup, timed region, and synchronization procedure:
- Peak GPU memory, sampling/profiling method, and measurement scope:
- Model size (include lighting/material sidecars), active splat count:

## Observations and interpretation

- Reference/prediction/crop/normal/depth artifact paths:
- What improved or failed? Any numerical issues or capacity limits?
- Comparison validity and known confounders:
- Next action based on the evidence:

## Legacy outputs

If this record describes an older run, distinguish recovered evidence from
unknown settings. The current YAML and file timestamps alone do not establish
the original configuration, budget, code revision, or runtime.
