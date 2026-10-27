# Research design record: Deferred Specular Shading with Gaussian Surfels

**Last updated:** 27 October 2026 (Asia/Singapore).

**Purpose:** Document the reasoning, literature checks, implementation choices,
and unresolved questions developed so far, as preparation for the final report.

**Living document:** These notes will be updated as implementation, experiments,
and literature review progress, recording new evidence and changes to decisions.

## 1. Research objective and the claim we can actually test

The project investigates whether explicit surface-oriented Gaussian primitives
and per-pixel specular evaluation improve reconstruction of reflective surfaces
in a custom Gaussian Splatting trainer. The intended contribution is an
implementation and controlled investigation of this combination in NVIDIA Warp.
It should not be presented as an established novel algorithm without a broader
novelty review.

The practical research question is:

> Relative to the existing RGB Warp trainer, does a surfel representation with
> deferred reflection improve held-out image quality on reflective scenes, and
> what additional training, rendering, and memory costs does it introduce?

The original proposal phrases this as replacing spherical-harmonic appearance
with physically based shading. Code inspection revealed two qualifications.
First, the actual baseline trains view-independent RGB, rather than degree-3 SH.
Second, a simple environment-reflection blend and a complete physically based
roughness model are different levels of implementation. The report's final
claims must follow the implemented baseline and shading model.

This leads to separate questions: whether the complete extension improves
images; whether surface primitives improve the geometry used by reflection;
and whether deferred evaluation itself improves normal optimization. One
before/after comparison can address the first question, but cannot isolate all
three explanations.

## 2. What the existing repository provides

The extension starts from a custom educational Python/NVIDIA Warp implementation
of vanilla-style 3DGS, not from the official Graphdeco CUDA/PyTorch implementation.
The existing trainer exposes projection, tile-based rasterization, alpha
compositing, explicit backward kernels, Adam optimization, density control,
validation, and PLY export. Its visibility and gradient code are therefore
accessible for adding intermediate attributes and their adjoints.

The original 3DGS work combines anisotropic Gaussian scene parameters,
optimization with density control, and a visibility-aware splatting renderer.
It supplies the broader representation and rendering background, rather than
evidence that this educational implementation reproduces every detail or
performance characteristic. [Kerbl et al., 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

Several local implementation details determine the project's design:

- In [3dgs_trainer.py](../3dgs_trainer.py), `colour_at_view` clamps a stored RGB
  vector and accepts no viewing direction. This is constant per-primitive color,
  with degree-0-style angular capacity, rather than a trained directional SH model.
- `projected_covariance` computes the full symmetric screen covariance, including
  its off-diagonal term. The separate `projected_covariance_diagonal` helper is
  used for bounds; its name should not be mistaken for a diagonal-only renderer.
- Physical scales are obtained from log-scales. Quaternion parameters encode
  orientation, and the optimization code maintains their normalization.
- The loader reads Blender-style transforms, derives focal length from horizontal
  field of view, resizes images, and composites their alpha against a configured
  background. It assumes a centered pinhole camera with the supported focal model.
- The starter config uses Lego's initialization cloud, 256x256 images, and a
  2,000-iteration budget. Merged defaults enable early stopping and define settings
  absent from the short YAML, including the learning-rate schedule.
- The evaluator can consume certain SH sidecars, but that capability does not
  mean the current RGB trainer produces or optimizes those coefficients.

Per-splat RGB being constant does not make rendered images entirely independent
of viewpoint: projection, visibility, and compositing still change with camera
position. An RGB baseline may partly fit glossy observations through geometry
or opacity changes. That is a reason to inspect geometry and normals alongside
image metrics, rather than equating appearance fit with correct reconstruction.

There is also a parameter-count correction to carry into the final report.
An RGB SH expansion through degree 3 has 3(3+1)^2 = 48 scalar coefficients in
total, including three DC coefficients; it has 45 additional scalars beyond DC,
not 48 additional scalars. The official model allocates channels according to
this basis count. [Official 3DGS Gaussian model](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py)

Consequently, the present experiment cannot claim to outperform degree-3 SH
unless an appropriate SH baseline is implemented or selected and evaluated under
a matched protocol. It also cannot infer memory savings from a comparison of
48 SH values against four proposed material values when its real baseline has
three RGB values.

## 3. Literature informing the design

### 3.1 Gaussian Surfels: making the surface direction explicit

Dai et al. construct planar Gaussian surfels by fixing local thickness to zero.
Their surface-reconstruction system includes depth-normal consistency and other
supervision/regularization, as well as a subsequent meshing pipeline. It also
accounts for the changing depth across a slanted surfel instead of assuming that
every covered pixel has the center's depth. These details matter: introducing
flat primitives alone does not reproduce their complete method or guarantee
surface alignment. We are considering their representation and relevant geometry
constraints, without committing to their complete reconstruction system.
[Dai et al., 2024, Sections 3.2-3.4](https://arxiv.org/html/2404.17774v1#S3.SS2)

### 3.2 Deferred Reflection: changing how normals receive useful gradients

Ye et al. rasterize surface attributes before querying reflected environment
color. Their motivation is an optimization problem: a high-frequency environment
lookup offers useful local normal gradients over a limited range. Their training
method includes a bootstrap and explicit normal-propagation interventions, not
just a second rendering pass. Their discussion also distinguishes the mirror
reflection component from richer appearance handled by base SH and identifies
physically based roughness as further work. Therefore, a four-parameter
diffuse/roughness replacement is our proposed adaptation, not a faithful
description of their original model.
[Ye et al., 2024, Sections 3 and 5](https://arxiv.org/html/2404.18454v1#S3)

For this project, the combination of explicit surfel normals and a deferred
reflection gradient is a research hypothesis. It is plausible that the two
components complement each other, but their published successes do not establish
the outcome of combining them in this renderer.

### 3.3 Related methods checked while preparing this record

Huang et al.'s 2DGS also uses oriented planar disks, but emphasizes perspective-
accurate ray-splat intersection and geometry regularization. It is a separate
method from Dai et al.'s Gaussian Surfels. A flattened covariance added to the
existing renderer should not automatically be called an implementation of the
complete Huang et al. method. [Huang et al., 2024](https://surfsplatting.github.io/)

Ref-NeRF provides relevant context for structuring appearance around reflected
directions and surface properties. Its neural radiance-field architecture is
different from the proposed Warp splatting pipeline, so it is conceptual related
work rather than a drop-in implementation. [Verbin et al., 2022](https://dorverbin.github.io/refnerf/)

NeRO addresses geometry and BRDF reconstruction under unknown illumination,
including direct and indirect lighting concerns. This makes its reflective-object
data relevant to later evaluation. Its inverse-rendering objective is broader
than our immediate novel-view image reconstruction experiment.
[Liu et al., 2023](https://liuyuan-pal.github.io/NeRO/)

## 4. Geometry parameterization: flattening versus independent tangents

Two representations were considered. The current working recommendation is to
retain the existing rotation and optimize two positive in-plane scales, with
the third physical scale fixed to zero. A local expression is

\[
\Sigma_i = R_i\,\mathrm{diag}(s_{x,i}^2,s_{y,i}^2,0)R_i^\top,
\qquad n_i = R_i(0,0,1)^\top.
\]

This is the flattened construction described by Dai et al.; each patch still
has its own position and orientation in 3D. The official source also derives
normals from the rotation representation.
[Gaussian Surfels model implementation](https://github.com/turandai/gaussian_surfels/blob/main/scene/gaussian_model.py)

Our implementation rationale is to minimize simultaneous changes. The trainer
already stores quaternions and scales, and its initialization, optimization,
densification, and export paths are organized around them. Reusing this structure
reduces the number of mechanisms that must change before an interpretable
comparison is possible. Lower implementation risk is the justification here;
better reconstruction quality remains untested.

The alternative uses scaled tangent vectors u and v:

\[
\Sigma = uu^\top+vv^\top,
\qquad n=\frac{u\times v}{\|u\times v\|}.
\]

This algebra makes a surface basis and normal computation explicit, and could
be convenient for kernels naturally expressed in tangent coordinates. However,
independent vectors can approach parallelism, causing unstable normal
normalization. Different bases can also describe the same covariance, adding
redundancy to optimization. Tangents do not inherently make the underlying
ellipse more expressive than rotation plus in-plane scales, nor do direct
derivatives alone establish smoother training. A future tangent implementation
would need a concrete simplification or measured benefit to justify its extra
constraints and maintenance cost.

Flattening still requires deliberate code changes. In the current log-scale
representation, storing zero gives exp(0)=1; storing log(0) is invalid. The missing
thickness contribution must be handled explicitly, including in projection and
backward evaluation. Scale clamps, splitting, cloning, pruning, initialization,
and export must respect that convention. An epsilon-thickness implementation
would be a distinct approximation and should be named and measured accordingly.

A rank-two world covariance also must not be inverted as an ordinary nonsingular
3D covariance. The existing renderer projects it to a filtered screen covariance;
edge-on behavior and that filtering need checking. Flattening is a representation
change, not a one-line guarantee of numerical stability. Similarly, an explicit
normal gives a clear variable to optimize; it does not supply the supervision
needed to recover the correct orientation.

## 5. The proposed deferred pipeline and its gradient paths

The proposed extension separates attribute rasterization from appearance
evaluation. With front-to-back weights w_i(p)=T_i(p) alpha_i(p), a first-pass
prototype would accumulate base color, normals, coverage, and necessary material
attributes at each pixel p. Depth is also useful for geometry inspection and
normal-depth regularization, although a distant environment lookup alone does
not require a depth buffer.

The official 3DGS-DR renderer normalizes its blended normal, reflects a camera
ray, samples an environment representation, and blends the result with base
appearance using a reflection-strength map. It retains SH base appearance.
This code check is the basis for treating a similar RGB-base blend as a feasible
intermediate prototype rather than claiming an existing GGX implementation.
[Official 3DGS-DR renderer](https://github.com/gapszju/3DGS-DR/blob/main/gaussian_renderer/__init__.py)

Our corresponding notation would be

\[
N=\sum_iw_i n_i,\quad \hat n=N/\|N\|,\quad
r=d-2(d\cdot\hat n)\hat n,\quad
C=(1-\rho)C_{\mathrm{base}}+\rho E(r),
\]

where d points from the camera into the scene, E is the environment lookup, and
rho is the composited reflection strength. This is a candidate implementation,
not a finalized material model. Coverage normalization, background treatment,
normal orientation, and near-zero N behavior must be specified consistently.
In particular, background pixels should not receive arbitrary reflection color
from normalizing an undefined vector.

The optimization reasoning can be expressed through the chain

\[
\frac{\partial L}{\partial n_i}
\supset
\frac{\partial L}{\partial C}
\frac{\partial C}{\partial\hat n}
\frac{\partial\hat n}{\partial N}\,w_i.
\]

For nonzero N, the normalization Jacobian is
(I - hat(n)hat(n)^T)/||N||. Overlapping primitives therefore share a pixel-level
normal-dependent shading signal. They also influence the compositing weights,
so a correct backward implementation must include those paths, not merely
distribute a normal gradient with fixed weights.

This interpretation does not mean the existing forward renderer has independent
gradients for every splat: its alpha and transmittance already couple contributors
to a pixel. The proposed difference is where the nonlinear reflection function
is evaluated. Visibility changes, sorting, tile membership, alpha cutoffs, and
normalization degeneracies can still create piecewise behavior. The defensible
claim is a potentially more useful optimization pathway, not globally smooth
optimization or guaranteed escape from local minima.

We also need to distinguish depth at the surfel center from depth at a particular
ray-plane intersection. A center-depth diagnostic may be an early approximation;
a geometry-consistency loss should use depth consistent with the actual rendered
surface. Normal regularization and depth construction are design choices to
record and ablate, not invisible implementation details.

## 6. Shading choices: material response and illumination are separate

### 6.1 What Phong would and would not establish

Phong has a specular term and can produce moving highlights. Under a small set
of point or directional lights, it is a useful controlled test of normal/view
conventions and backward derivatives. Changing its shininess controls highlight
concentration. [Phong lighting reference](https://cs.nyu.edu/~perlin/courses/fall2005ugrad/phong.html)

The earlier suggestion that Phong cannot reproduce detailed reflections needed
qualification. A point light carries direction and color, but no image of a tree
or window; that lighting input cannot produce those structures merely by raising
shininess. A Phong-style response can be evaluated under richer lighting. The
limitation of the simple example is its illumination representation, not the
absence of specular reflection in Phong.

The project should consequently choose two things explicitly: the surface's
directional response and the illumination incident on it. A different BRDF alone
does not provide missing scene illumination.

### 6.2 Why an environment map is the proposed first reflection prototype

An environment map represents illumination as a function of direction, which
provides structured reflection information beyond a few lights. A distant-map
model assumes the lighting distribution can be approximated independently of
surface position. [PBRT, Infinite Area Lights](https://pbr-book.org/4ed/Light_Sources/Infinite_Area_Lights)

For our RGB-image fitting experiment, the environment could be learned rather
than supplied as ground truth. A simple lookup plus a learned reflection strength
offers a manageable first implementation of the deferred gradient path. However,
one shared map is an approximation for nearby-object reflections, occlusion,
concavities, and spatially varying indirect light. It must not be described as
complete global illumination or as a faithful recovery of the physical scene.

Geometry, base color, lighting intensity, and reflectance can compensate for each
other during image fitting. A successful image reconstruction therefore does
not establish that the recovered environment or materials are uniquely correct.
Controlled fixed-light tests can check correctness before these variables are
jointly optimized.

### 6.3 Why GGX remains a candidate, not a completed decision

GGX is a microfacet normal-distribution model commonly used inside a specular
BRDF. Roughness affects reflection spread; a complete response also needs
Fresnel and masking/visibility terms. Walter et al. provide microfacet background,
while Filament documents a practical Cook-Torrance implementation using GGX,
Smith visibility, and Schlick Fresnel.
[Walter et al., 2007](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html),
[Filament material model](https://google.github.io/filament/main/filament.html#materialsystem/specularbrdf)

For the project's final physically based claim, such a model is more directly
aligned than an unrestricted RGB reflection blend. It also requires more work:
roughness-dependent environment integration or filtering, a defined specular
reflectance, appropriate diffuse/specular energy handling, and differentiable
sampling. Roughness is not simply the reflection-blend weight. A material with
weak reflection and one with broad reflection describe different effects.

The proposal's three diffuse channels plus one roughness value leave important
assumptions unspecified, especially for metal-like regions. We must decide which
specular parameters are fixed or learned and justify those choices. Global
lighting parameters also contribute storage and optimizer state. Four per-splat
numbers do not describe the total appearance budget.

Finally, the current loader trains against PNG values scaled to [0,1]. Before
claiming physical light transport, the extension needs a deliberate choice of
linear/display color space, exposure, and output conversion. These are currently
open implementation decisions. The same image-comparison convention must be
used for all evaluated variants.

## 7. Datasets, supplied information, and scope

### 7.1 Materials is present; a materials baseline is still pending

Inspection of the local download on this date found:

| Property | Local observation |
| --- | --- |
| Scene root | data/materials/ |
| Training / validation / test frames | 100 / 100 / 200 |
| Referenced RGB images missing | 0 across all three manifests |
| Sample RGB image from each split | 800x800 RGBA |
| Horizontal field of view | 0.6194058656692505 radians in each manifest |
| Manifest-level fields | camera_angle_x, frames |
| Per-frame fields | file_path, rotation, transform_matrix |
| Additional test-depth PNGs | 200 |
| Files / total bytes | 603 / 144,125,494 |
| init.ply at scene root | Absent |

These are file/manifest observations, not a completed camera or reconstruction
validation. The sample-image check does not establish every image's contents.
The standard Blender loader uses the image paths, poses, and field of view as
inputs; the auxiliary rotation field is not a substitute for the transform
matrix. [Original NeRF Blender loader](https://github.com/bmild/nerf/blob/master/load_blender.py)

The camera focal length follows f_x = W/(2 tan(theta_x/2)). The existing loader
recomputes it at the requested width, so ordinary square resizing is already
accounted for in that model. Cropping, nonuniform resizing, off-center principal
points, and more general intrinsics require separate treatment.

Neither the camera manifests nor the current RGB loader provide ready-to-use
roughness, metallic, light-position, or environment-map parameters. The test-depth
passes are auxiliary files that the RGB loader does not read. Their encoding,
scale, and depth convention must be established before numerical geometry use.
They are held-out data: they should not initialize training geometry or supervise
training losses. A self-supervised normal-depth loss on rendered training views
is a different source of supervision.

The existing config still points to Lego and its point cloud. A materials config
must use an appropriate initialization, such as the trainer's random option
with init_ply: null; Lego's cloud cannot simply be reused. Initialization policy
needs testing and matching across variants, because poor initialization could
dominate the effect attributed to shading.

### 7.2 Lego, ship, and real reflective data have different roles

Lego is useful for checking the existing training/evaluation pipeline with the
provided initialization. Materials is the primary experiment tied to reflective
appearance. The proposal also names ship, but the current plan's dataset list
no longer explicitly includes it; its final role remains to be confirmed.
Diffuse-region quality can be examined without assuming that an entire scene
is non-reflective.

Real-world evaluation would test the model under less idealized camera,
background, illumination, and material conditions. It strengthens an external-
validity argument if successful, but one object cannot prove general real-world
robustness. A limited but carefully documented real scene is more informative
than a broad unsupported claim.

NeRO distinguishes GlossyReal and GlossySynthetic. The official examples classify
bear as real and bell as synthetic; the name alone should not be used to infer
the category. The 3DGS-DR converter expects the synthetic layout and its camera/
depth files, so its existence does not establish real-data compatibility.
[NeRO dataset instructions](https://github.com/liuyuan-pal/NeRO),
[nero2blender.py](https://github.com/gapszju/3DGS-DR/blob/main/nero2blender.py)

A real-data path still needs inspection of intrinsics, distortion, coordinate
conventions, masks/backgrounds, scale, initialization, and evaluation splits.
No real scene has yet been selected or validated in this workspace.

### 7.3 Repository data policy

Lego was already versioned in the starter repository. Keeping new downloads
outside Git was considered, but the current user decision is to include materials
in this repository too. The planned commit therefore includes the dataset files
and manifests. That choice makes these local inputs available with the code;
generated models/logs remain under ignored outputs/, and personal background
references remain under ignored local-context/. No materials ignore rule should
be added under the current policy.

## 8. Evaluation designed to support specific conclusions

### 8.1 An ablation sequence instead of a single unexplained improvement

The proposed core sequence is:

| Variant | Geometry | Appearance evaluation | Main comparison supported |
| --- | --- | --- | --- |
| A | Existing 3D Gaussians | Current per-splat RGB | Actual repository baseline |
| B | Flattened surfels | Matched per-splat RGB | Geometry change, subject to any added regularization |
| C | Flattened surfels | Deferred reflection/shading | Complete appearance extension on fixed primitive type |
| D, if feasible | 3D Gaussians | Same deferred model as C | Surfel contribution under matched shading |
| E, if needed | Same geometry/material/lighting as C | Matched forward specular evaluation | Deferred-versus-forward optimization claim |

A versus C tests the complete extension, but changes geometry, appearance
capacity, lighting representation, and evaluation location together. B versus C
reduces some confounding but still adds materials/illumination. C versus a matched
E is needed to directly attribute an advantage to deferred nonlinear evaluation.
Any geometry regularizer added in B or C should be identified and, where feasible,
tested separately.

### 8.2 Held-out metrics and reflection crops

Training views drive gradient updates. Validation views guide development and
configuration choices. Test views are reserved for the final comparison. Training-
view scores must be labelled accordingly; selecting a model repeatedly on test
results also weakens the intended separation.

PSNR measures pixel fidelity through MSE; higher is better. SSIM adds a structural
similarity perspective, while LPIPS uses learned feature distances; lower LPIPS
is better. Their different emphases motivate reporting all three rather than
assuming any one number completely captures reflection quality.
[Wang et al., 2004](https://www.cns.nyu.edu/~lcv/ssim/),
[Zhang et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html)

The implementation matters as much as the metric name. The local SSIM function
uses an 11-pixel uniform window and averages RGB results; another implementation
may use different filtering or color treatment. The evaluator's LPIPS backbone
is recorded, and its PSNR summary averages per-view PSNR values. This differs
from taking PSNR after averaging MSE across all views. Cross-paper comparisons
must establish matching definitions, resolutions, backgrounds, and splits.

Full images may dilute improvements on a small reflective object. We therefore
consider fixed reflection crops, with identical view IDs and pixel bounds for
reference and every prediction. Crops should be defined from the reference
content before inspecting which method wins. Report crop and full-image results
separately; do not change crop resolution or select only favorable views. Very
small crops also require appropriate SSIM/LPIPS support and interpretation.

PSNR improvements alone do not prove accurate normals. Normal/depth visualizations
and any trustworthy geometric reference should support geometry claims. Smooth
but incorrectly oriented normals can still look plausible in a color map.

### 8.3 Fair training and efficiency measurements

The proposal specifies a 30,000-step comparison, but the current starter requests
2,000 steps and may stop early. Fixed-step experiments need early stopping
disabled and a deliberate learning-rate schedule. Saving a nominal budget is
insufficient; actual completed steps and stop reasons must also be recorded.

Shared settings should be matched where meaningful: data, initialization policy,
resolution, seeds, common losses, common optimization settings, and evaluation.
New material/lighting parameters need their own documented learning rates.
Keeping every number identical despite different parameter meanings is not a
substitute for a fair protocol. Any tuning should use validation data and a
comparable search budget.

The existing trainer optimizes MSE, whereas the cited reflection method uses
a different photometric objective. The first ablations should preserve the
baseline objective, or apply a changed common objective to both methods, rather
than silently attributing loss-function improvements to geometry/shading.

Training time, peak GPU memory, final model size, and rendering FPS address
different costs. Record hardware, image size, warmup, synchronization, and what
is included in each timed region. Include global illumination/material sidecars
in model size. Allocated capacity, tile buffers, gradients, and optimizer moments
can dominate memory, so per-splat appearance counts alone cannot predict VRAM.
Repeated seeds should be used where practical, with variability and failed runs
reported instead of only the best result.

## 9. Why reproducibility infrastructure was implemented first

Existing Lego renders and metric files demonstrated that output artifacts exist,
but their original settings were not adequately recorded. Reconstructing their
provenance from today's YAML or timestamps would risk inventing experimental
conditions. They should remain exploratory unless original evidence is recovered.

The new [experiment runner](../scripts/run_experiment.py) gives each RGB run a
fresh output directory and captures the supplied and resolved configs, Git state,
Python/package/GPU information, source and input fingerprints, commands, logs,
timings, and evaluation CSV. The source archive includes new implementation files
that a tracked Git diff alone would miss. Failures retain their evidence instead
of being recorded as successful experiments.

Preparation-only mode checks inputs and saves the setup before expensive work.
It is useful for finding a wrong data path or unexpected default, but it is not
a renderer, CUDA, or gradient correctness check. The normal runner performs
training/evaluation, with optional baseline gradient checking. It currently
targets RGB export and evaluation; a surfel/material model will require explicit
integration rather than being supported automatically.

The [command reference](CommandList.md) retains individual stages for debugging
alongside the runner. It uses Git Bash on Windows and the Windows virtual
environment activation path. Manual logging captures Python's pipeline exit code
immediately after tee. This preserves the ability to diagnose one stage without
repeating a complete experiment.

Automation captures evidence, but not its interpretation. Research questions,
dataset origin, preprocessing choices, visual observations, confounders, and
decisions still belong in a reviewed record. The runner also does not yet measure
peak VRAM, rendering FPS, or crop metrics.

## 10. Verified progress and its limits

| Item | Status at this record's date |
| --- | --- |
| Existing custom RGB baseline inspected | Confirmed in source; not an official-3DGS equivalence claim |
| Materials download | Present; all manifest-referenced RGB files found |
| Materials initialization/configuration | Still to establish; starter YAML remains Lego-specific |
| Materials baseline training/evaluation | Not yet established as a recorded benchmark |
| Experiment runner | Implemented in the current working tree |
| Runner checks | Nine orchestration checks passed during implementation |
| Actual renderer integration | One CPU step at 16x16 with four initial splats and one view per split completed through training and PSNR/SSIM evaluation |
| Bash documentation | Syntax of 29 Bash blocks, Windows-venv activation, and pipeline exit-code handling checked during documentation work |
| Surfel primitives, G-buffers, deferred reflection | Planned; not implemented by the runner work |
| GGX material shading and real NeRO support | Open design/implementation work |

The tiny integration run is evidence that the wrapper can invoke the existing
trainer and evaluator and collect their artifacts. It is not a meaningful
reconstruction benchmark, a GPU validation, an LPIPS validation, or evidence
supporting the research hypothesis. Logs from that check also record Warp's
fallback after temporary precompiled-header permission warnings.

The current workspace builds on commit 3fec81f plus uncommitted runner,
documentation, and downloaded-data changes. The proposed commit groups those
foundations; it should not be titled as if deferred surfel rendering were already
implemented.

## 11. Current decisions, recommendations, and unresolved work

| Topic | Current position | What would settle it |
| --- | --- | --- |
| Framework | Extend the custom Warp trainer | Preserve and measure the actual baseline |
| Primitive representation | Working recommendation: rotation + two scales, zero local thickness | Correct projection/backward/export behavior and controlled geometry tests |
| Independent tangents | Considered alternative | Demonstrable implementation or optimization benefit |
| First reflection model | Working recommendation: environment-reflection prototype | Checked gradients and a recorded materials comparison |
| Final physically based model | GGX remains a candidate | Final material/lighting assumptions, integration method, scope/time decision |
| SH comparison | Proposal/code mismatch remains | Implement/select an SH baseline or narrow the claim |
| Datasets | Materials primary; real reflective evaluation desired | Materials benchmark first, real-scene selection/conversion next |
| Ship | Scope unresolved | Confirm whether it remains a required control |
| Data in Git | Include both bundled Lego and downloaded materials | Current explicit user preference |
| Normal regularization | Likely useful but not finalized | Geometry diagnostics and a regularization ablation |
| Experimental evidence | Infrastructure checks only so far | Reproducible baselines and matched ablations |

The immediate next milestone is a materials-specific RGB configuration with a
defensible initialization, a checked loading/rendering path, and a recorded
baseline run. Geometry and reflection changes can then be introduced in stages
whose outcomes are attributable and whose failures can be localized. The final
report should rewrite the research question around the method actually completed
and retain unsuccessful or inconclusive findings where they explain its limits.

## References and implementation evidence

The links below identify the primary sources checked for this design record.
Equations and implementation recommendations above are project reasoning unless
explicitly attributed; they are not claims that every referenced method was
reproduced or that all source code was exhaustively audited.

1. Kerbl, Kopanas, Leimkuehler, and Drettakis (2023),
   [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
2. Dai et al. (2024),
   [High-quality Surface Reconstruction using Gaussian Surfels](https://arxiv.org/abs/2404.17774);
   [official implementation](https://github.com/turandai/gaussian_surfels).
3. Ye, Hou, and Zhou (2024),
   [3D Gaussian Splatting with Deferred Reflection](https://arxiv.org/abs/2404.18454);
   [project](https://gapszju.github.io/3DGS-DR/) and
   [official implementation](https://github.com/gapszju/3DGS-DR).
4. Huang et al. (2024),
   [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/).
5. Verbin et al. (2022),
   [Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields](https://dorverbin.github.io/refnerf/).
6. Liu et al. (2023),
   [NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images](https://liuyuan-pal.github.io/NeRO/).
7. Mildenhall et al. (2020), NeRF;
   [official data and loading implementation](https://github.com/bmild/nerf).
8. Walter, Marschner, Li, and Torrance (2007),
   [Microfacet Models for Refraction through Rough Surfaces](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html).
9. Google Filament,
   [Physically Based Rendering in Filament](https://google.github.io/filament/main/filament.html).
10. Wang, Bovik, Sheikh, and Simoncelli (2004),
    [Image Quality Assessment: From Error Visibility to Structural Similarity](https://www.cns.nyu.edu/~lcv/ssim/).
11. Zhang et al. (2018),
    [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html).
12. Ken Perlin's NYU course,
    [Phong lighting reference](https://cs.nyu.edu/~perlin/courses/fall2005ugrad/phong.html).

Local implementation evidence:
[trainer](../3dgs_trainer.py), [training defaults](../shared/training_config.py),
[evaluator](../3dgs_evaluator.py), [metric implementation](../image_metrics.py),
[runner](../scripts/run_experiment.py), [runner checks](../scripts/test_run_experiment.py),
[staged research plan](researchStages.md), and [todo list](TODO.md).
