# Training and geometry audit — 29 October 2026

**Pre-implementation snapshot.** The optimizer fixes, quiet refinement and matched
presets were subsequently implemented; see the
[design update log](project-plan.md#design-update-log). Findings below preserve the
original diagnosis and measurements rather than describing the patched behavior.

The active deferred pipeline works, but appearance accuracy currently has stronger
support than surface accuracy. Fix optimization behavior before tuning more loss
weights. This audit changes no training implementation or existing experiment.

## Scope and evidence

Traced the project entry points, baseline trainer and optimizer, complete surfel
package, shared projection/compositing/tiling and appearance dependencies,
configuration, checkpoint/evaluation paths, experiment runners, and sweep selection.
The Python inventory contains 46 files and 10,262 lines; historical teaching
renderers/trainers are separate from the active surfel path. The detailed findings
below concern the active training paths, not a claim that every historical utility
has been exhaustively verified.

Reviewed current presets and saved configurations/logs/checkpoints, including:

- `outputs/materials_red-rose-silver-Oct29-1656-deferred-tune-propagation-0-1p5`
- `outputs/materials_red-rose-silver-Oct29-1755-mini-paper-mask`
- `outputs/materials_red-rose-silver-Oct29-1557-rgb-normal`

Validation: 76 tests passed on CPU; all 31 `test_surfel*.py` tests passed with
`SURFEL_TEST_DEVICE=cuda:0`. Logs are `outputs/training-audit-tests.log` and
`outputs/training-audit-cuda-tests.log`. Separate small probes reproduced the two
implementation problems below; those cases are not caught by the existing suite.
The previous finite-difference check also passed on CPU, away from discontinuities.

## Actual current loss

The current full and 30k mini deferred presets set `photometric_loss: l1_dssim`.
This overrides the inherited MSE curriculum and its validation-PSNR transition gate:

`L = 0.8 * mean(abs(prediction - target)) + 0.2 * D + 0.1 * mask_BCE`

Here `D = mean((1 - SSIM) / 2)`, using reflected uniform 11x11 windows. Therefore
the structural contribution is `0.1 * mean(1 - SSIM)`. The project's definition is
internally consistent, but the same numeric 0.2 weight does not reproduce the
official 3DGS-DR code's `0.2 * (1 - SSIM)` convention. Record this distinction before
changing it; old experiment scores describe the objective they actually used.

The mask term compares accumulated coverage to original training PNG alpha over
all pixels. It supplies gradients even when white splats hide against white RGB.
It is silhouette supervision, not a direct 3D floater detector or the paper's
optional spherical foreground restriction for real scenes.

Current `normal_depth_weight` is zero. Its logged `normal_pixels=0` follows from
the disabled computation; it does not mean no foreground pixels exist.

The earlier RGB geometry preset used MSE, then `0.8*MSE + 0.2*D` after its 21 dB
validation gate, plus mask BCE and normal-depth consistency. The consistency
coefficient ramps from zero at 2k to 0.01 at 4k. It compares the normalized blended
surfel normal with the normal from finite differences of back-projected rendered
depth, differentiating through both. Selection excludes low coverage, silhouette
edges, and relative depth jumps. This is self-consistency, not ground-truth normals.

Source: `src/cp4282_gs/surfels/trainer.py:108`,
`src/cp4282_gs/surfels/geometry_losses.py:65`, `shared/dssim_loss.py:124`.

## What happens each iteration

1. Choose one random training camera with replacement. Camera poses and targets are
   fixed. Held-out views supply evaluation and curriculum decisions, not gradients.
2. At the configured activation step (3k in the current 30k presets), enable
   reflection/environment optimization. There is no measured geometry-readiness
   test at this transition.
3. Project active primitives, build tile lists, and sort by primitive centre depth.
   Composite RGB, normals, centre-depth, coverage and reflection strength.
4. Before reflection activation, composite base RGB over the background. Afterwards,
   normalize the blended normal, reflect the pixel ray, and query the learned map.
5. Compute the actual image objective and enabled geometry terms. Backpropagate
   shader gradients into the G-buffer, add geometry gradients, then differentiate
   compositing/projection and the quaternion-derived normals.
6. Apply Adam to position, scale, opacity and base RGB; apply the custom Adam update
   to quaternion rotation. Normalize quaternions and constrain scales/RGB. The third
   stored scale is unused by surfel rendering. After activation, also update
   reflection logits and environment logits, then constrain those logits.
7. On structure-update steps, run pruning/densification. Afterwards run any opacity
   reset or propagation intervention, then evaluation/logging/snapshot work.

Normals have no separate trainable vector: `n = R(q) * local_z`, flipped towards
the camera. Updating `q` changes both the normal and the projected surfel footprint.
Depth is computed from primitive positions; there is no separately trained depth map.

| Variable | Current learning rate / schedule |
| --- | --- |
| Position | Exponential: 0.00016 to 0.0000016 over `position_max_steps` |
| Two tangent log-scales | 0.005 |
| Quaternion | 0.001 |
| Opacity logit | 0.05; zero LR during the 200-step post-propagation freeze |
| Base RGB | 0.0025 |
| Reflection logit | 0.006 after reflection activation |
| Environment logits | 0.05 after reflection activation |

These are optimizer updates; visibility determines which splats receive a new
image gradient, and existing Adam momentum may still move parameters with zero
current gradient. Current surfels use SH0 only. Gaussian SH baselines additionally
train unlocked directional coefficients and include their appearance-to-position
gradient path.

The full preset's position decay spans 60k steps: at its 30k endpoint LR is still
0.000016, ten times the configured final LR. The 30k mini finishes its decay at 30k.

## How pruning and densification work

Current ordinary density interval: 100 steps. During active propagation through
24k: 500 steps. After 24k: back to 100. There is no growth cutoff, including at the
last iteration. Growth and late pruning are separate decisions.

- Rank active parents by the norm of their position gradient from the current
  training view. This is not accumulated screen-space gradient magnitude.
- With multiview ADC enabled, score only the top 20,000 candidates using 10 sampled
  training views. Normalize each view's absolute RGB error into [0,1]; pixels above
  0.1 are high-error. Count high-error pixels inside each candidate's projected
  bounding rectangle; average across views. Keep candidates whose score exceeds 5.
- At every structure update, prune active primitives with opacity below 0.005.
- Once mean population growth over three density updates is nonnegative and below
  2%, latch late pruning. Every 400 steps thereafter, also prune opacity below 0.02
  and candidates whose normalized low-error-footprint score exceeds 0.9.
- After pruning, request up to 20% of the surviving active count as new children,
  bounded by free slots and eligible parents. This is a maximum addition, not a
  promise that every event adds exactly 20%.
- Small parents are cloned at the same position/scale. Large parents, with maximum
  tangent scale above `0.01*scene_radius`, become two children: reuse the parent's
  slot and one free slot, sample displacements from its tangent-plane Gaussian,
  and divide both tangent scales by 1.6. Each selected parent adds one net primitive.
- Children inherit rotation, opacity, RGB and reflection. Sources and destinations
  are disjoint; inactive slots are excluded from rendering.

Periodic opacity resets cap active opacities at 0.01, allowing unsupported splats
to fade and be pruned. During propagation, reset steps skip expansion/noise. Other
1k propagation steps raise opacity to at least 0.9 and reflection to at least 0.001,
expand tangents 1.5x for reflection >=0.02, and add RGB noise up to +/-0.1 for
reflection <=0.1. Opacity learning freezes for 200 subsequent updates. Propagation
does not directly copy normals; it changes overlap to alter reflection gradients.

Source: `src/cp4282_gs/surfels/trainer.py:271`,
`src/cp4282_gs/surfels/workflow.py:123`, `normal_propagation.py`.

## Findings, ordered by action priority

1. **Confirmed: hard-clamped colours can become unable to recover.**
   `constrain_parameters` clips RGB to exactly 0/1, while explicit raster backward
   passes colour gradients only for values strictly between 0 and 1. A probe with
   green=1 had zero analytic gradient; reducing it to 0.999 lowered loss by
   0.000102818, yet three optimizer updates left it at 1. Existing momentum/noise
   can sometimes move a channel, but its new colour gradient is blocked. In the
   full completed model, 40.58% of channels are within 1e-6 of a boundary; in the
   mini, 52.10%. Those counts show exposure, not that every boundary value is wrong.
   Prefer a consistent bounded parameterization, or a projected update that allows
   inward gradients. Add boundary recovery checks. See `kernels.py:141` and
   `3dgs_trainer.py:915`; the Gaussian path has the same strict-boundary pattern.

2. **Confirmed: density events erase all existing Adam history.**
   The code resets every geometry optimizer and reflection optimizer whenever
   `requested` is nonzero. A probe rejected every candidate, added zero children,
   and still reset the quaternion Adam clock from 1 to 0. Preserve state for
   unchanged slots and initialize changed/reused slots deliberately. Account for
   per-slot step/bias-correction semantics when implementing this; copying only
   moment arrays is not a complete design. Source: `trainer.py:363`.

3. **Structural concern: final refinement still changes topology.**
   Growth continues at increased frequency after propagation ends; it can also
   create final-step children that receive no subsequent optimizer update. Repeated
   opacity clamps further disturb the last phase. Test a distinct fixed-topology
   final phase, with separate growth/pruning/reset controls. Simply setting the
   density interval to zero from startup also prevents ordinary pruning events.

4. **Structural concern: density scores measure footprints, not contribution.**
   A large rectangle in a low-error region is not proof that a splat is redundant.
   The scorer does not account for its actual transmittance/occlusion contribution,
   and large footprints can earn high growth scores by covering someone else's
   residual. There is no absolute gradient eligibility threshold. Test accumulated,
   visibility-aware growth statistics and contribution-aware pruning before making
   the fraction more aggressive. Geometry/mask gradients also affect parent ranking.

5. **Current loss does not demand a coherent surface.**
   Silhouette masks allow internal layers, hidden splats and orientation variation.
   Reflection, base RGB and environment can compensate for each other. The current
   full model has only 7.16% of splats above reflection 0.1; this is consistent with
   incomplete reflection discovery, but is not a per-pixel decomposition measurement.
   The explicit prohibition on combining propagation with normal-depth loss is a
   project policy, not a mathematical incompatibility. For the first clean geometry
   ablation, disable propagation in both arms and vary only normal-depth weight.

6. **Geometry limitation: projected surfels do not supply ray/plane depth.**
   Tangent covariance has exact zero local-z variance, which is correct for this
   representation. Projection still uses the EWA approximation. Each primitive
   contributes its centre depth everywhere; the normalized accumulated depth is an
   average across layers. Consistency with its derivative is helpful but does not
   establish a true surface. There is no depth-distortion penalty or explicit
   neighbour-consistency term. If geometry plateaus after optimization fixes,
   investigate ray/plane depth and layer concentration before stronger smoothing.

7. **Representation and evaluation limits.**
   A single distant environment with scalar reflection strength lacks roughness,
   coloured specular modulation and local interreflection handling. Material
   mismatch can therefore be absorbed by RGB or geometry. Whole-image PSNR/SSIM,
   especially with extensive clean background, cannot rank normal accuracy. Use
   foreground metrics and Blender normal/depth references for evaluation.

## What is working

- Reflection-to-normal-to-quaternion gradient flow and overlapping compositing are
  exercised by numerical checks on CPU/CUDA.
- Tangent-plane splitting preserves zero normal-axis thickness and material state.
- Training PNG alpha drives mask supervision; held-out masks do not train the model.
- Project-pass initialization statistics use training views only. The historical
  direct baseline trainer still initializes colour from its combined target array;
  use the project Gaussian control for strictly training-only initialization.
- Current propagation/reset schedules interleave correctly; final-step opacity
  clamping is skipped. Fixed 24k propagation termination follows the chosen code
  schedule rather than measuring reflection discovery convergence.
- Run folders preserve resolved configuration, source/input hashes, logs and
  checkpoints. Evaluation loads saved surfel material state. Warm start explicitly
  starts optimizer state and iteration numbering afresh; it is not exact resumption.
- Latest full-model saved-view background coverage is small: mean 0.00007458,
  with 0.0850% of alpha-zero pixels above coverage 0.01. Internal floaters remain
  untested by that diagnostic.

## Recommended direction

1. Fix boundary recovery and optimizer-state handling; add targeted regression
   checks. Freeze the loss convention and stop changing several schedules together.
2. Add an explicit final phase without growth, disruptive opacity resets or colour
   sabotage. Keep pruning independently configurable; compare against the preserved
   baseline because earlier whole-run growth reductions had image-quality costs.
3. Run a matched 400px deferred+mask pair: propagation off in both, normal-depth
   coefficient 0 versus 0.01, same initialization/seed/budget/density/LR schedule.
   Ramp consistency after coverage forms. Keep image supervision in both arms.
4. Judge foreground RGB and normal/depth accuracy on the same held-out cameras.
   Export Blender references for evaluation only; do not select solely by PSNR.
5. If the consistency arm helps, retain it and separately ablate propagation.
   If both have poor depth accuracy, improve depth/visibility geometry. Only then
   investigate more environment resolution or material complexity.

The intended research question is whether deferred reflection improves appearance
while explicit surface constraints keep geometry coherent. A normal-only loss
would leave appearance and shape insufficiently anchored.
