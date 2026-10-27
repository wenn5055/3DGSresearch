# Project todo list

Unchecked items are planned, not completed. Follow [researchStages.md](researchStages.md)
for rationale and stage completion criteria. Immediate milestone: a checked
implementation design and a reproducible RGB baseline on materials.

## First: decisions and experiment records

- [ ] Add any course rubric, relevant lecture notes, and professor feedback to
  `local-context/`; extract requirements into the tracked plan where appropriate.
- [ ] Reconcile the RGB baseline with the proposal's SH comparison language.
- [ ] Confirm primary materials + real NeRO scope and whether ship remains required.
- [ ] Read the surfel parameterization/normal loss and deferred-reflection sections
  of the two primary papers alongside their official implementations.
- [ ] Write down the geometry parameterization, lighting representation, material
  parameters, and losses. Start from the recommendations in researchStages.md.
- [ ] Adopt unique run IDs and [run-record-template.md](run-record-template.md).
- [x] Implement a single-command RGB experiment runner for isolated outputs,
  configuration/source/input snapshots, logs, training/evaluation, and failure records.
  Usage: [CommandList.md](CommandList.md).
- [ ] Label old Lego outputs as exploratory unless their original settings can
  be recovered from evidence.

## Baseline and data

- [ ] Record GPU/driver, Python, dependencies, Git revision, and local changes.
- [ ] Run setup and baseline gradient checks; save logs and exit codes.
- [ ] Run and evaluate a recorded Lego baseline using a dedicated config/output.
- [x] Set the data policy: keep Lego and materials in Git; generated results stay in outputs/.
- [x] Obtain materials train/validation/test images and manifests; all referenced RGB images are present.
- [ ] Verify data loading, cameras, resizing, background, scale, and initialization.
- [ ] Create a materials config with a suitable initialization policy, output path,
  seed, training budget, learning-rate schedule, and stopping policy.
- [ ] Fix validation views and the test protocol, including any reflection crops.
- [ ] Run the unchanged RGB baseline on materials; save logs, config, trained
  splats, rendered test images, per-view CSV, and an experiment summary.

## Implementation

- [ ] Add RGB surfels with explicit normals and adapt all affected training paths.
- [ ] Check surfel projection and geometry gradients; save the RGB-surfel comparison.
- [ ] Add normal/depth/base-color/coverage and necessary material G-buffers.
- [ ] Inspect controlled overlap and edge cases; check G-buffer gradients.
- [ ] Implement a deferred reflection prototype under controlled lighting.
- [ ] Check gradients through lighting, shading, normal normalization, and compositing.
- [ ] Decide whether to extend the prototype to GGX environment shading or
  explicitly narrow the proposal's physically based shading claim.
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
