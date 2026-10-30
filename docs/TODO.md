# Project todo list

Unchecked items are planned, not completed. Follow [researchStages.md](researchStages.md)
for rationale and stage completion criteria. The final held-out comparison and
[ablation write-up](experiments/2026-10-29-normal-consistency-final.md) are complete;
normal-consistency tuning is closed. Details:
[design update log](project-plan.md#design-update-log). Current implementation and commands:
[project-passes.md](project-passes.md); measured results: [findings.md](findings.md).

## Optimizer and geometry follow-up — 29 October

Design reasoning and configuration: [update log](project-plan.md#design-update-log).

- [x] Audit losses, per-step geometry updates, pruning and densification.
- [x] Fix RGB endpoint recovery in dense, sparse and surfel paths.
- [x] Preserve existing optimizer trajectories and inherit parent state for children;
  avoid resets when no children are created, including SH/reflection state.
- [x] Add quiet refinement with independently configurable pruning and no final-step growth.
- [x] Allow optional propagation plus normal-depth consistency; add matched propagation-off
  coefficient 0/0.01 presets with 10k final fixed-topology steps.
- [x] Complete and inspect the fresh full-budget matched consistency pair: weight 0
  wins image scores; 0.01 produces smoother normal/depth previews. Preserve both.
- [x] Close normal-weight tuning at the user's request; move on to final renders/write-up.
- [x] Prepare final matched held-out renders/reflection crops and describe the tradeoff:
  200 test views, four pose-selected figure cameras, eight alpha-defined crops,
  normal/depth figures and [write-up](experiments/2026-10-29-normal-consistency-final.md).
- [ ] Optional future evaluation: Blender normal/depth references and foreground metrics;
  reference geometry evaluation is deferred by user preference.
- [ ] Evaluate contribution-aware pruning and accumulated growth statistics separately.
- [ ] If geometry remains inaccurate, investigate ray/plane depth and layer concentration.

## Consultation follow-up — 29 October

Details and verified sources: [consultation notes](consultation-2026-10-29.md).

- [x] User saves edited `materials_red-rose-silver.blend` (three balls remain).
- [x] Obtain a Materials Blender scene copy and view-capture source; scene opens in Blender 5.0.1.
- [x] Prepare pose-replay capture script and RGB/deferred training presets; [commands](materials-red-rose-silver.md).
- [x] Render the edited red/rose/silver dataset with NeRF camera metadata and disjoint splits.
- [x] Automate validation-only deferred normal-weight tuning with resumption and saved winner.
- [x] Stop the old normal-weight sweep and replace the active deferred preset with
  paper-style reflection normal gradients/propagation, retaining alpha-mask BCE.
- [x] Complete the [replacement propagation-factor sweep](../outputs/materials_red-rose-silver-Oct29-1656-deferred-propagation-sweep/summary.md)
  and inspect the saved normals. Factor 1.5 won validation PSNR; normals remain noisy.
- [ ] Compare normal-loss training on the subset; inspect normal/depth maps and validation scores.

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
- [x] Record the full Gaussian-control versus RGB-surfel quality comparison.
- [x] Add normal/depth/base-color/coverage/reflection-strength G-buffers.
- [x] Inspect controlled overlap, uncovered pixels, normal flips and G-buffer gradients.
- [x] Implement a deferred learned-environment reflection prototype.
- [x] Check gradients through lighting, shading, normal normalization, and compositing.
- [x] Scope this iteration as environment-reflection blending; no full GGX/roughness claim.
- [x] Run recorded deferred-surfel training/evaluation on materials; interim findings saved.
- [x] Add training-alpha mask loss and normal-depth consistency with CPU/CUDA gradient checks.
- [x] Provide separate mask-only, RGB geometry-loss, and deferred geometry-loss presets.
- [ ] Measure the new losses' effect on geometry and image reconstruction at full budget.
- [ ] Evaluate richer roughness/material models after the geometry comparison.

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
