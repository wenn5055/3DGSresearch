# Project todo list

Unchecked items are planned, not completed. Follow [researchStages.md](researchStages.md)
for rationale and stage completion criteria. Immediate milestone: a measured Materials
comparison of the three project passes. Current implementation details and commands:
[project-passes.md](project-passes.md); measured results: [findings.md](findings.md).

## First: decisions and experiment records

- [ ] Add any course rubric, relevant lecture notes, and professor feedback to
  `local-context/`; extract requirements into the tracked plan where appropriate.
- [x] Reconcile the RGB baseline with the proposal's SH comparison language;
  degree-2 and degree-3 SH baselines are implemented and measured.
- [ ] Confirm primary materials + real NeRO scope and whether ship remains required.
- [ ] Read the surfel parameterization/normal loss and deferred-reflection sections
  of the two primary papers alongside their official implementations.
- [x] Document the current geometry, lighting, material parameters and losses in
  [project-passes.md](project-passes.md); full BRDF/geometry priors remain future work.
- [x] Adopt unique dated run IDs, isolated outputs and saved config/source provenance.
- [x] Implement a single-command RGB experiment runner for isolated outputs,
  configuration/source/input snapshots, logs, training/evaluation, and failure records.
  Usage: [CommandList.md](CommandList.md).
- [ ] Label old Lego outputs as exploratory unless their original settings can
  be recovered from evidence.

## Baseline and data

- [x] Record GPU/driver, Python, dependencies, Git revision, and local changes per run.
- [x] Run baseline and project gradient checks; save logs and exit codes.
- [x] Run and evaluate recorded Lego baselines with dedicated configs/outputs.
- [x] Set the data policy: keep Lego and materials in Git; generated results stay in outputs/.
- [x] Obtain materials train/validation/test images and manifests; all referenced RGB images are present.
- [ ] Verify data loading, cameras, resizing, background, scale, and initialization.
- [x] Create a materials config with a suitable initialization policy, output path,
  seed, training budget, learning-rate schedule, and stopping policy.
- [ ] Fix validation views and the test protocol, including any reflection crops.
- [x] Run the RGB baseline on materials; save logs, config, trained
  splats, rendered test images, per-view CSV, and an experiment summary.
- [x] Add staged MSE/D-SSIM, SH 0–3, checkpoint sidecars and matched test evaluation.
- [x] Start validation-only SH learning-rate tuning with matched-baseline reuse.
- [x] Finish the sweep and inspect eligible candidates; 0.00025 selected by validation PSNR.
- [ ] Repeat promising settings with additional matched seeds.
- [x] Identify validation-colour initialization in the historical baseline and add
  a training-only Gaussian control for the new project comparison.

## Implementation

- [x] Add rank-two RGB surfels with explicit normals and adapt projection, density control and checkpoints.
- [x] Check surfel projection and geometry gradients on CPU/CUDA.
- [ ] Record the full Gaussian-control versus RGB-surfel quality comparison.
- [x] Add normal/depth/base-color/coverage/reflection-strength G-buffers.
- [x] Inspect controlled overlap, uncovered pixels, normal flips and G-buffer gradients.
- [x] Implement a deferred learned-environment reflection prototype.
- [x] Check gradients through lighting, shading, normal normalization, and compositing.
- [x] Scope this iteration as environment-reflection blending; no full GGX/roughness claim.
- [ ] Evaluate whether normal-depth regularization and a richer roughness model are needed.
- [ ] Run recorded deferred-surfel training/evaluation on materials.

## Evaluation and reporting

- [ ] Choose useful additional ablations: 3D Gaussians with deferred shading,
  forward versus deferred shading, and/or normal regularization.
- [ ] Fix matched final settings, including 30,000 completed steps if retaining
  that protocol; disable early stopping and align the learning-rate schedule.
- [ ] Define timing, warmup, memory, model-size, metric, and crop procedures.
- [ ] Repeat key comparisons with multiple seeds where practical.
- [ ] Inspect one real NeRO scene's format and conversion requirements; establish
  its splits, camera support, initialization, and RGB baseline.
- [ ] Evaluate the extension on the real scene and ship if retained in scope.
- [ ] Assemble full-image metrics, reflection crops, normal/depth maps, costs,
  failures, and limitations in the report.
- [ ] Update project scope, decisions, and AI-use documentation to reflect actual work.
