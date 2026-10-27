# Research stages

This is a proposed sequence, not a record of completed experiments. See
[project-plan.md](project-plan.md) for scope, [TODO.md](TODO.md) for actions, and
[run-record-template.md](run-record-template.md) for recording each run.

## Stage 0: Define the experiment and preserve the RGB baseline

Goal: establish exactly what is being compared and make runs reproducible.

- Describe the existing trainer as custom Warp 3DGS with view-independent RGB.
- Reconcile the proposal's SH baseline claim; decide whether an SH baseline is
  needed for the final research claims.
- Preserve the source revision and config for each baseline. Git history is
  sufficient; a duplicate baseline code tree is unnecessary.
- Run the existing finite-difference checks and a recorded Lego training/evaluation.
- Record configured and completed iterations, settings, logs, and held-out results.
- Mark existing outputs with unknown provenance as exploratory; do not invent
  their settings from the current config.

Ready to proceed when: the baseline is checked, and a new run can be repeated
from its saved code revision, environment, data, and configuration.

## Stage 1: Establish a materials benchmark

Goal: measure the existing renderer on the primary research scene.

- Obtain materials with train/validation/test manifests and images. The current
  repository policy is to version materials alongside Lego; generated results
  remain under ignored outputs/.
- Verify camera conventions, resize behavior, background/alpha handling, scene
  scale, and initialization. Do not reuse Lego's initialization point cloud.
- Save a scene-specific config and record a baseline run with a unique output path.
- Fix test view IDs and chrome-sphere crop coordinates using the reference images.
- Save full-image test metrics and matching reference/prediction crops.

Ready to proceed when: training and evaluation work on materials and the baseline
metrics and images have known provenance.

## Stage 2: Introduce surfels with RGB appearance

Goal: isolate the geometry change before introducing a new appearance model.

- Recommended starting parameterization: existing rotation plus two positive
  in-plane scales, with the third local scale explicitly fixed to zero.
- Define the normal from the rotated local z-axis. Adapt projection, gradients,
  initialization, densification, pruning, and export where required.
- Do not store log(0); explicitly handle the absent thickness axis. An epsilon
  thickness approximation must be documented as an approximation.
- Keep the RGB appearance and shared training settings matched to the baseline.
- Check finite-difference gradients away from visibility/sorting boundaries and
  inspect front-facing, tilted, and nearly edge-on surfels.
- Evaluate whether normal-depth consistency regularization is needed. If added,
  record its weight and distinguish its effect from parameterization changes.

Ready to proceed when: surfel rendering/training is stable, its changed gradients
are checked, and its RGB results are recorded. Improvement is not a prerequisite.

### When would two tangent vectors be justified?

Let u and v be scaled tangent vectors. A rank-two covariance can be constructed
as Sigma = u u^T + v v^T, and its plane normal as normalize(u cross v).
This can make plane geometry and derivatives through the normal explicit.

However, unconstrained vectors can become nearly parallel, making the normal
unstable. Different tangent bases can also represent the same covariance; more
optimizer freedom does not imply better geometry or simpler optimization.
Orthogonality constraints or a stable basis construction need a design decision.
Direct tangent gradients do not by themselves establish smoother optimization.

Use this alternative if it materially simplifies the Warp implementation or a
controlled comparison demonstrates better stability. Otherwise, rotation plus
two scales reuses more of the current code. Dai et al. flatten local z-scale and
use a depth-normal consistency loss; independent tangent vectors are our possible
implementation choice, not a requirement of their method.

## Stage 3: Add and inspect G-buffers

Goal: make the geometry information used by shading observable and differentiable.

- Composite normals, depth, RGB/base color, opacity coverage, and material
  attributes required by the selected shading model.
- Specify world/camera coordinate conventions, normal orientation, attribute
  normalization, and behavior for uncovered pixels and near-zero normal sums.
- Retain RGB output for comparison and save normal/depth/coverage visualizations.
- Verify simple overlapping-surface cases and gradients into the accumulated
  attributes, their weights, and surfel geometry.

Ready to proceed when: the buffers behave as specified on controlled scenes and
changed backward paths pass meaningful gradient checks.

## Stage 4: Add deferred reflection and its backward pass

Goal: test whether per-pixel reflections improve held-out appearance.

Choose and record both a material response and a lighting representation:

| Option | Role in this project | Design work required |
| --- | --- | --- |
| Phong/Blinn-Phong with a known light | Controlled debugging experiment | Light position/direction, intensity, diffuse/specular coefficients, shininess |
| Environment-map reflection blended with base RGB | Suggested intermediate prototype | Fixed or learned environment, reflection strength, differentiable sampling |
| GGX microfacet shading with environment lighting | Candidate final model for the proposal's physically based claim | Roughness filtering/integration, Fresnel/specular reflectance, diffuse/specular energy handling |

Phong under a small set of lights can test highlight motion, but it does not
by itself model structured environment reflections. A simple environment lookup
is also not automatically a complete physically based BRDF model. If it becomes
the final method, describe the reduced scope explicitly.

- Start with controlled fixed lighting to check normal/reflection derivatives.
- Define which lighting and material parameters are fixed versus learned. In
  particular, decide how specular reflectance is handled for metallic regions.
- Add gradients through shading, G-buffer normalization and compositing, surfel
  normals, material parameters, and any learned illumination.
- Validate important parameter families with finite differences before long runs.
- Save reflection, normal, and material visualizations alongside final RGB.

Ready to proceed when: the full gradient path is checked and a recorded materials
run can be evaluated against the earlier stages under a matched protocol.

## Stage 5: Run comparisons and real-world evaluation

Core comparison: RGB 3D Gaussians -> RGB surfels -> deferred surfels. This measures
the geometry change first, then the appearance/shading extension on those surfels.
The last change may include lighting and extra material capacity as well as the
deferred computation; avoid attributing all gains to deferred evaluation alone.

- If feasible, add deferred shading on 3D Gaussians to isolate the surfel benefit.
- A matched forward-versus-deferred shading experiment is needed to directly
  support claims about deferred shading's normal-optimization advantage.
- Match splits, resolution, initialization policy, shared settings, evaluation,
  and training budget. Record model-specific settings and parameter counts.
- For the proposed fixed 30,000-step comparison, disable early stopping and set
  the learning-rate schedule deliberately. Record actual completed iterations.
- Report PSNR/SSIM/LPIPS, images/crops, training time, peak GPU memory, model size,
  and rendering FPS. Define warmup, timed region, and memory measurement method.
- Use repeated seeds where practical; report variability and failures.
- Select one real NeRO scene, inspect cameras/masks/distortion and define splits,
  then run the RGB baseline before the extension. The 3DGS-DR conversion script
  targets GlossySynthetic and is not assumed to support GlossyReal.
- Resolve whether ship remains a required secondary scene; it appears in the
  research question but is absent from the current plan's dataset bullet list.

Ready to report when: results have known provenance, limitations and comparison
conditions are clear, and conclusions follow the measured evidence.

## Primary references

- [Dai et al.: Gaussian Surfels](https://arxiv.org/abs/2404.17774)
- [Official Gaussian Surfels code](https://github.com/turandai/gaussian_surfels)
- [Official 3DGS-DR renderer](https://github.com/gapszju/3DGS-DR/blob/main/gaussian_renderer/__init__.py)
- [Filament: material and lighting models](https://google.github.io/filament/main/filament.html)
- [NeRO datasets](https://github.com/liuyuan-pal/NeRO)
- [3DGS-DR synthetic converter](https://github.com/gapszju/3DGS-DR/blob/main/nero2blender.py)
