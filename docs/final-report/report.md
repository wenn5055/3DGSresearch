# Reconstructing Specular Reflections with Deferred Gaussian Surfels

**CP4282 individual project — reviewed draft, 7 November 2026**  
**Author:** [AUTHOR REVIEW: add name and student number]  
**Repository:** [wenn5055/3DGSresearch](https://github.com/wenn5055/3DGSresearch)

> Reviewed draft. Bracketed author-review items identify remaining attribution and submission details. Main study: 100,000 surfels, seed 7, 15,000 cumulative steps.

## Abstract

Sharp reflections are sensitive to small surface-normal errors. This project extends a course-based NVIDIA Warp Gaussian splatting trainer with planar surfels and deferred HDR environment shading. A controlled chrome-sphere study compares the complete HDR/angular treatment with degree-three spherical harmonics (SH3), using the same initialization, 100,000 surfels and 10k frozen-geometry plus 5k refinement steps. HDR improves PSNR, SSIM and LPIPS on all 200 test views: mean PSNR rises from 20.2655 to 24.5732 dB and LPIPS falls by 35.1%. Fine reflected branches and fencing nevertheless remain blurred or warped. Evaluation-only substitutions reveal a conditional effect: correcting normals improves reflection structure under reference lighting but worsens mean image scores under learned lighting. Stronger normal–depth consistency increases contributor alignment without improving reference accuracy. A separate L1/D-SSIM refinement trial lowers validation LPIPS by 10.1% but sacrifices PSNR and retains the fine-detail failure. These observations distinguish image fitting from accurate geometry/light recovery. The contribution is a verified implementation and controlled analysis; broader generalisation requires additional scenes and seeds.

## 1. Introduction

Novel-view synthesis must reproduce how an object changes appearance as the camera moves. A reflective surface makes this demanding: a relatively smooth geometric surface can display a high-frequency image of its surroundings. Fitting broad color changes is therefore different from recovering recognizable reflected detail.

The starting point is the custom Python/NVIDIA Warp trainer developed through the course assignments. The project adds explicit surface normals and per-pixel reflection evaluation, drawing on Gaussian Surfels and 3D Gaussian Splatting with Deferred Reflection. The implementation is an adaptation within this educational framework; the contribution is its construction and experimental investigation rather than a claim of a new rendering algorithm.

The final research question is: **Under a matched initialization and training budget, does deferred HDR/angular surfel shading reconstruct reflections better than SH3 appearance, and what storage and rendering costs accompany it?** The analysis then asks what lighting/normal interventions reveal about missing detail, and whether stronger normal–depth consistency improves normal accuracy. These questions organise the completed investigation; they are not a claim that every diagnostic was specified before development.

The original proposal targeted Materials and Ship and a physically based roughness model. Materials experiments exposed geometry and local-reflection confounds, motivating a plain sphere as the primary scene. The implemented model is a learned HDR reflection blend; Materials provides supporting evidence, while Ship and real-scene expansion were deferred. The resulting contribution is a controlled implementation study, not a new inverse-rendering algorithm.

The report presents three contributions: a differentiable surfel/deferred implementation, a matched SH3/HDR comparison with recorded provenance, and controlled diagnostic interventions explaining the remaining reflection defects.

## 2. Background and related work

### 2.1 Gaussian splatting and surface representations

Kerbl et al.'s 3DGS combines anisotropic Gaussians, density control and visibility-aware splatting. SH models directional color with a finite angular basis: degree three uses 16 coefficients per color channel, or 48 scalars including DC. This supplies the course trainer's representation background. [Kerbl et al., 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

Dai et al.'s Gaussian Surfels flattens each primitive by setting its local thickness to zero and uses the local z-axis as its normal. Its reconstruction pipeline also contains normal–depth consistency and other supervision and reconstruction components. This project adapts the planar representation and consistency idea, without reproducing the full surface-reconstruction system. [Dai et al., 2024](https://arxiv.org/abs/2404.17774)

Huang et al.'s 2DGS uses planar disks with perspective-accurate ray–splat rasterization. The present implementation instead retains filtered EWA footprints, while adapting ray–plane depth for geometry regularization. [Huang et al., 2024](https://surfsplatting.github.io/)

### 2.2 Reflection-based appearance

Ye et al.'s 3DGS-DR first composites surface attributes, then uses the blended normal to query reflected environment color. Its normal-propagation procedure addresses the difficulty of obtaining useful normal gradients from detailed environment lighting. The project adapts the deferred reflection idea and explored propagation during development; propagation is disabled in the final sphere comparison. [Ye et al., 2024](https://arxiv.org/abs/2404.18454)

RGS-DR combines deferred 2D surfel shading with a directional residual pass and specular refinement, making it close related work. Here the main HDR model uses an SH0 base and is evaluated for image reconstruction rather than editable inverse rendering. [Kouros et al., 2025](https://arxiv.org/abs/2504.18468)

Ref-NeRF provides broader context for parameterising appearance through reflected directions and surface properties, using a neural radiance field rather than splats. [Verbin et al., 2022](https://dorverbin.github.io/refnerf/)

## 3. Method and implementation

### 3.1 Starting framework and attribution

The project extends the assignment-based renderer, explicit backward kernels, Adam optimization and checkpoint infrastructure. The project code adds surfel geometry, intermediate surface buffers, deferred shaders, geometry losses, staged experiments and diagnostic tooling. SH degrees zero through three and evaluation tooling also form part of the repository's development.

| Component | Starting point or influence | Use in this project |
| --- | --- | --- |
| Projection, compositing, optimization | Course/assignment Warp trainer | Extended existing framework |
| Planar covariance and normals | Gaussian Surfels | Adapted mathematical representation |
| Deferred normal/environment lookup | 3DGS-DR | Adapted rendering and gradient structure |
| HDR/angular shader and staged sphere study | Project implementation | Learned positive lighting, angular modulation, matched experiment and diagnostics |
| RGS-DR, 2DGS, Ref-NeRF | Related-work references | Context and comparison of design choices; no full reproduction |

**[AUTHOR REVIEW: confirm file-level code provenance and any verbatim external code before submission. The table describes documented conceptual influences, not a completed code-license audit. Clarify the boundary between assignment work and project additions.]**

### 3.2 Surfel geometry and rasterization

Surfel i has center μᵢ, quaternion rotation Rᵢ and positive tangent scales sₓᵢ and sᵧᵢ. Its covariance and normal are

<div class="equation">Σᵢ = Rᵢ diag(sₓᵢ², sᵧᵢ², 0) Rᵢᵀ, &nbsp; nᵢ = Rᵢ(0, 0, 1)ᵀ.</div>

The zero local thickness is enforced by the renderer. A screen-space filter keeps projections finite. Normals are oriented consistently toward the current camera before compositing. The EWA support and center-based ordering are retained from the existing rendering approach.

For front-to-back alpha weights wᵢ, the rasterizer accumulates color C = Σᵢwᵢcᵢ, normal sum N = Σᵢwᵢnᵢ, coverage A = Σᵢwᵢ, reflection sum S = Σᵢwᵢσ(mᵢ), and depth. The shading normal is the normalized N. Depth uses ray–plane intersection per contributor, with guarded fallback to center depth for invalid or near-parallel intersections. This changes depth estimation, not the EWA footprint or ordering.

### 3.3 Deferred HDR/angular appearance

The shader reflects the incoming world-space camera ray d about the composited normal n:

<div class="equation">r = d − 2(d · n)n.</div>

It bilinearly samples a learned 512 × 256 latitude–longitude environment, with periodic longitude and clamped latitude. Interpolated raw environment values are decoded with softplus to obtain positive radiance. A shared, 16-knot linear lookup g modulates the base reflection strength according to the absolute view/normal cosine:

<div class="equation">s = σ(logit(clamp(S/A)) + g(|d · n|)).</div>

For covered pixels with valid normals, the foreground color is

<div class="equation">F = T((1 − s) T⁻¹(C/A) + s E(r)), &nbsp; Î = A F + (1 − A) B.</div>

Here T⁻¹ converts display RGB to linear light; T clips to the supported display range and applies the sRGB transfer; B is white. Uncovered or invalid-normal pixels use the base compositing path. The angular knots initialize to zero. Lighting, base colors and reflection parameters are learned from training images; the original forest texture and reference normals do not enter optimization.

This design followed read-only inspection of the Chrome material's Facing → ColorRamp → glossy-color graph. Its artistic angular response and HDR illumination motivated a learned angular modulation and linear-light positive lighting, rather than imposing a generic Fresnel curve. The inspection informed the model family; no reference ramp values, lighting pixels or normals were supplied to fitting. The model has no learned GGX roughness or secondary visibility rays.

Deferred shading gives overlapping surfels a shared per-pixel normal objective. Holding compositing weights fixed, ignoring the kernel's numerical epsilon, and writing n = N/‖N‖, its normal-only gradient is

<div class="equation">∂L/∂nᵢ = wᵢ (I − nnᵀ) / ‖N‖ · ∂L/∂n.</div>

Thus contributors receive the same shading-normal correction, scaled by their weights and projected onto the normal's tangent space. Weight, position and angular-response paths also contribute in the implementation. This explains the gradient coupling; it does not guarantee correct normals or establish an advantage over a forward-shading control that was not run.

### 3.4 Initialization, losses and training stages

Initialization uses training alpha masks to carve a visual hull, smooth its signed-distance field, sample its zero crossings and derive surface orientations. It does not use Blender mesh vertices or normals. The same 100,000-point initializer is reused for the matched comparison.

Stage one fits appearance for 10,000 iterations while positions, scales, rotations and opacity remain frozen, separating initial appearance fitting from surface motion. Stage two reloads the learned state, restarts Adam, and jointly refines it for 5,000 iterations. Both stages use MSE. Refinement additionally uses

<div class="equation">L = L<sub>MSE</sub> + 0.1 L<sub>mask</sub> + λ<sub>n</sub>(t) L<sub>normal-depth</sub>.</div>

The mask term is binary cross-entropy between coverage and training alpha. Normal–depth consistency averages 1 − n̂ · n̂<sub>depth</sub> over valid neighborhoods, where n̂<sub>depth</sub> comes from back-projected rendered depth. Low coverage, depth jumps and degenerate stencils are excluded. Gradients flow through both normal and depth paths. The final sphere baseline ramps normal weight to 0.01 over the first 1,000 refinement steps. Stage-one geometry losses are disabled. Density changes, opacity resets, normal propagation and early stopping are disabled throughout this comparison.

The SH3 control uses the same surfels and geometry policy, with directional color instead of environment shading. Its SH bands activate at 1k/2k/3k during the first stage. Matching the optimization policy does not force the two methods to learn identical final geometry.

MSE was retained from the frozen-appearance diagnostic as a shared pixelwise objective, rather than selected through a loss-function ablation. Its relation to PSNR makes pixel fitting interpretable, while SSIM, LPIPS and crops assess complementary quality. A separate validation-only continuation trial examines replacing the refinement objective with L1 + D-SSIM; it does not change the main comparison's training protocol.

### 3.5 Implementation verification

Recorded development checks include 22 HDR finite-difference cases per device on CPU and CUDA with a nonconstant angular response, plus an 800px integration run covering growth, sidecar save/reload and evaluation. Tests also check that display conversion precedes alpha composition: mixing in display space would otherwise change partially covered edge colors. For the reference-normal intervention, checkpoint hashes and non-normal buffer fields remain unchanged, and both learned-normal baselines reproduce the prior lighting evaluation exactly. These checks support gradient, serialization and intervention correctness; they are recorded development evidence, not newly repeated full training runs. The [test code](https://github.com/wenn5055/3DGSresearch/blob/59c553f/scripts/test_surfel_hdr.py) and [verification record](https://github.com/wenn5055/3DGSresearch/blob/59c553f/docs/experiments/2026-10-30-angular-response.md) preserve details.

## 4. Experimental setup

### 4.1 Data and scope

The primary `justChromeBall` scene contains one smooth chrome sphere without the logo or base used in earlier Chrome experiments. It retains forest illumination and camera templates from Materials. Blender 5.0.1 captured 400 RGBA images at 800 × 800: 100 training, 100 validation and 200 test views. Capture used 128 samples, seed 7, Standard view transform, no look adjustment, exposure zero and gamma one. The recorded capture and pose audit checks disjoint splits.

![Dataset reference views](../reports/assets/2026-11-02-sphere/dataset-views.png)

*Figure 1. `justChromeBall`, validation views 0, 13, 73 and 88, left to right. Actual Blender reference captures. These illustrate the dataset, not training reconstructions.*

Training uses RGB and alpha from training images only. Validation supports monitoring and variant selection. Blender geometry and the packed forest environment are reserved for evaluation-only diagnostics. Original Materials results use their own scene and protocol and are reported separately.

### 4.2 Comparisons and selection

| Setting | SH3 | HDR-1.0 | HDR-0.75 |
| --- | --- | --- | --- |
| Initial points / seed | 100,000 / 7 | Same | Same |
| Resolution / steps | 800 × 800 / 10k + 5k | Same | Same |
| Initial tangent-scale multiplier | 1.0 | 1.0 | 0.75 |
| Appearance | SH3 on surfels | HDR/angular, SH0 base | HDR/angular, SH0 base |
| Refinement losses | MSE + mask + normal–depth | Same | Same |

SH3 versus HDR-1.0 is the primary comparison. HDR-0.75 changes only the initializer's two tangent scales; they remain learnable during refinement. It is an additional size-initialization trial, not a lower-point-count model or fixed final-size constraint. The recorded selection rule requires lower validation LPIPS, no decrease in foreground PSNR, and at least 99% foreground coverage. The selection/checkpoint record was frozen before the main 200-view test evaluation.

The comparison evaluates complete appearance treatments under the same geometry policy. It does not independently isolate HDR, angular modulation, deferred versus forward evaluation, or the surfel representation. The later normal-weight trial is a development diagnostic and does not replace the frozen main comparison.

### 4.3 Metrics, figures and computational cost

Scores are arithmetic means of per-view metrics on float renders in display RGB. PSNR uses unit-range MSE; SSIM uses the repository's reflected, uniform 11 × 11 RGB window; LPIPS uses AlexNet. This SSIM implementation should be distinguished from alternative Gaussian-window or grayscale implementations. Foreground PSNR excludes pixels below reference alpha 0.95; coverage measures the foreground fraction with predicted coverage at least 0.5. [Wang et al., 2004](https://www.cns.nyu.edu/~lcv/ssim/), [Zhang et al., 2018](https://richzhang.github.io/PerceptualSimilarity/)

The main figure cameras are 0, 13, 73 and 88. Validation diagnostics also retain view 51, a previously observed regression. Identical reference-defined crop bounds are used across variants. PNG figures are visual illustrations; metrics come from float renders rather than the saved 8-bit pictures.

Recorded runs use an NVIDIA RTX PRO 5000 Blackwell, Warp 1.18.0 and Python 3.12.10. Costs measure required checkpoint files, warmed forward rendering and fresh-process Warp allocation peaks. Timing includes host tiling and MSE but excludes image download, file I/O and perceptual metrics. Full measurement details are in Appendix A.4; overlapping work prevents a controlled training-speed claim.

## 5. Results

### 5.1 Main matched comparison

Table 1 gives the frozen main test results. The matched-size HDR model gains **4.3077 dB** over SH3 and reduces LPIPS by **35.1%**. Foreground PSNR improves by 4.5973 dB, so the gain is not explained solely by white-background pixels. All three models retain 100% foreground coverage under the stated threshold.

*Table 1. `justChromeBall`, all 200 test views; 100k surfels and 15k cumulative training steps. Higher PSNR/SSIM and lower LPIPS are better.*

| Model | PSNR (dB) | SSIM | LPIPS | Foreground PSNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| SH3 | 20.2655 | 0.821398 | 0.128760 | 13.6520 |
| HDR-1.0, matched primary | 24.5732 | 0.880892 | 0.083517 | 18.2494 |
| HDR-0.75, additional variant | 24.7453 | 0.884031 | 0.080988 | 18.4427 |

The gain is consistent across the saved cameras: HDR-1.0 improves PSNR, SSIM, LPIPS and foreground PSNR on **200/200 views**. Per-view PSNR gain has median 4.4439 dB and range 2.9349–5.5425 dB; every camera's gain is substantial rather than a few outliers driving the mean. Figure 2 visualises the effect distribution. It describes variation across cameras within one fitted model pair, not uncertainty across training seeds.

![Paired gains over all test views](../reports/assets/2026-11-07-report-analysis/paired-test-effects.png)

*Figure 2. `justChromeBall`, all 200 test cameras, matched HDR-1.0 versus SH3. Empirical cumulative distributions of paired whole-image PSNR gain, foreground PSNR gain and LPIPS reduction. Positive values favour HDR. These plots are computed from the frozen saved metric CSV, with no new rendering or model selection.*

![Main test reflection comparison](../reports/assets/2026-11-07-report-analysis/test-crops-views-0-73.png)

*Figure 3. `justChromeBall`, test views 0 and 73, two of the four preselected illustration cameras. Columns: reference, SH3, HDR-1.0 and HDR-0.75. Identical actual evaluation crops. HDR recovers more tree/sky structure, but thin branches/fencing remain smeared and locally warped. The last column changes initialization size. The full four-view sheet remains in the supplementary evidence.*

Validation gives the same ordering: SH3 obtains 20.6530 dB / 0.127524 LPIPS, HDR-1.0 24.1249 / 0.093994, and HDR-0.75 24.2308 / 0.091814. The size variant passes the recorded selection rule, but its smaller gain is nonuniform: on the frozen test comparison it improves PSNR on 198/200 views and LPIPS on 189/200, with mean gains of only 0.1721 dB and 0.002529 LPIPS reduction. This makes the appearance treatment the larger measured change in this study.

### 5.2 Refinement and initial footprint size

For the original size-1.0 HDR model, refinement improves all-100-view validation PSNR from 22.4996 to 24.1250 dB, SSIM from 0.842281 to 0.874920 and LPIPS from 0.119765 to 0.093994. PSNR improves on 95 of 100 views; SSIM and LPIPS improve on all 100. This is a continuation result with 5k additional optimization steps, not an equal-budget isolation of geometry updates.

The five diagnostic views show retained coverage and smaller final footprint support for HDR-0.75. In validation view 0, median candidate support diameter falls from 12.61 to 11.14 pixels. These statistics include candidate primitives that may be occluded. The normal/light defects and remaining blobs mean this size trial does not identify footprint size as the sole cause of blur.

![Preserved refinement failure](../reports/assets/2026-11-02-sphere/regression-view-51.png)

*Figure 4. `justChromeBall`, validation view 51: reference, frozen HDR at 10k, refined HDR at 15k total. This preserved regression view shows that average improvement does not guarantee improvement of every reflected region.*

### 5.3 Storage, rendering and training observations

*Table 2. Final checkpoint costs for Table 1. MB and GB use decimal units. Times sum the two recorded training stages; they exclude initialization and evaluation.*

| Model | Required checkpoint (MB) | Observed forward FPS | Recorded 10k + 5k training (s) | Largest stage Warp allocation (GB) |
| --- | ---: | ---: | ---: | ---: |
| SH3 | 23.50 | 273.63 | 172.42 | 1.320 |
| HDR-1.0 | 8.54 | 287.61 | 172.92 | 1.253 |
| HDR-0.75 | 8.54 | 303.98 | 173.17 | 1.253 |

HDR-1.0 requires about 63.7% less checkpoint storage in this format: a shared environment and per-surfel blend replace SH3's per-surfel directional coefficients. Common geometry and compressed sidecars contribute to the exact file totals. Forward rates are similar in scale and depend on the learned footprints/visibility. The measured difference therefore does not establish an isolated shader speed advantage.

## 6. Analysis and discussion

### 6.1 Is learned lighting losing reflection detail?

At matched environment resolution and display exposure, the learned map lacks much of the original forest's branch, fence and ground structure. This remains visible at lower display exposure, so highlight clipping alone does not explain the difference.

![Original and learned lighting](../reports/assets/2026-11-03-environment-check/comparison.png)

*Figure 5. `justChromeBall` forest environment and the learned environment from the 100k HDR-0.75 model. Left: original forest resampled to the renderer's directional convention at 512 × 256; right: learned map. Rows use identical 0 EV and −2 EV display transforms. No fitted exposure or rotation. Actual environment diagnostics.*

Replacing only the learned environment with the aligned reference map, while freezing geometry and materials, lowers all-100-view validation LPIPS from 0.091814 to 0.079982. PSNR falls from 24.2308 to 20.6415 dB and SSIM from 0.877177 to 0.848285. Finer structure returns, but reflections remain warped. A sharper-looking reflection can therefore disagree with pixelwise scores when brightness and jointly fitted appearance no longer match. This substitution is a diagnostic, not a trained improvement.

### 6.2 How much do normal errors matter?

A four-combination intervention evaluates learned/reference lighting with learned/reference shading normals on validation views 0, 13, 73, 88 and 51. Reference normals are smooth interpolated normals from the actual Blender sphere mesh, not an assumed analytic sphere. Only the rasterized normal-sum direction changes; its magnitude, coverage, depth, colors and reflection sums remain fixed. The production shader uses the substituted normal for both reflection direction and angular response.

*Table 3. Five-view diagnostic means for the same 100k HDR-0.75 checkpoint. These must not be compared as if they were 100-view or test means.*

| Lighting | Normals | PSNR (dB) | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| Learned | Learned | 23.8554 | 0.871914 | 0.096202 |
| Reference | Learned | 20.4016 | 0.844558 | 0.085452 |
| Learned | Reference | 23.0536 | 0.867943 | 0.101984 |
| Reference | Reference | 22.7073 | 0.943615 | 0.056796 |

![Normal substitution under fixed reference lighting](../reports/assets/2026-11-03-reference-normal-swap/normal-swap-highlight.png)

*Figure 6. `justChromeBall`, validation view 0, identical central crop. Left: Blender target. Middle: reference forest lighting with learned normals. Right: the same lighting and checkpoint with reference shading normals. Actual evaluation renders; no retraining or material replacement.*

With reference lighting fixed, correcting normals improves all three image metrics on all five cameras and recovers recognizable branches and fencing. Mean learned-normal error is only 1.4678 degrees. In a coplanar mirror example, a normal rotation θ changes reflected direction by 2θ; a 1.5° normal perturbation can therefore rotate a reflected ray by 3°. For a pure longitude change at the environment equator, that spans about 4.3 texels on a 512-wide map. This is illustrative geometry, not measured mean lookup displacement; actual sensitivity depends on ray direction and the environment's local structure.

The conditional reversal is the key evidence: replacing normals under **learned light** changes mean PSNR by −0.8018 dB and worsens LPIPS by 0.005782; under **reference light**, it gains 2.3057 dB and reduces LPIPS by 0.028657. The LPIPS reduction differs by 0.034439 between lighting conditions. Correcting the same normal field therefore has different effects depending on the lighting paired with it, consistent with coupled fitting rather than independently correct components.

The joint reference case has the best structural/perceptual scores but lower PSNR than the learned pair. Brightness/rim errors and fixed learned materials remain, and the substitution affects angular response as well as reflection direction. The experiment supports normal/light interaction but does not uniquely identify compensation, material mismatch or sampling as the remaining mechanism.

### 6.3 Does stronger normal–depth consistency help?

The follow-up changes normal–depth weight from 0.01 to 0.03 for 5k refinement from the same frozen 10k parent. It uses the original 100k, size-1.0 setup, rather than the size-0.75 checkpoint used in the oracle diagnostic. The run completed on 3 November; paired validation and reference-normal analysis were revisited for this draft.

*Table 4. Normal-weight trial, paired evaluation on all 100 validation views; same parent and refinement budget. Reference-normal error is the mean of five diagnostic-view means, using the same valid-pixel rule in both variants.*

| Normal–depth weight | PSNR (dB) | SSIM | LPIPS | Normal error (degrees) |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 24.1249 | 0.874920 | 0.093994 | 1.5728 |
| 0.03 | 24.0857 | 0.874125 | 0.094244 | 1.5943 |

The stronger weight slightly worsens aggregate image scores. This does not contradict the reference-normal intervention: consistency with the model's own depth is not supervision toward the true surface normal. A mechanism that favors agreement between two learned quantities need not correct both. The trial tests one coefficient under one schedule and does not establish that normal regularization is generally ineffective.

Reference-normal error rises slightly on all five diagnostic views, from 1.5728 to 1.5943 degrees. Normal coherence, **‖N‖/A**, measures alignment among contributing normals at a pixel; its mean increases from 0.998552 to 0.998927. It measures neither spatial smoothness nor agreement with the reference. Thus stronger consistency improves contributor alignment without improving accuracy here. PSNR improves on 19/100 views, SSIM on 13/100 and LPIPS on 46/100; coverage remains complete. These small one-seed differences do not establish a generally harmful regularizer.

![Matched normal-weight reflection crops](../reports/assets/2026-11-07-normal-weight-closeout/reflection-crops.png)

*Figure 7. `justChromeBall`, validation views 0 and 51 from top to bottom. Columns: Blender target, normal–depth weight 0.01, weight 0.03. Identical reference-defined central crops from actual paired-evaluation renders. The higher weight does not visibly recover the target's sharp detail in these views.*

### 6.4 Does a different refinement image loss help?

A supplementary validation-only trial replaces MSE with **0.8 L1 + 0.2 D-SSIM** for 5k refinement from the same size-1.0 frozen HDR parent. It retains the 100k count, seed, geometry-loss coefficients, learning rates and Adam restart. The explicit photometric override is active from step 1; D-SSIM is `(1−SSIM)/2` with the existing 11×11 window. Preflight matches 204 input fingerprints and 28 training/rendering sources. The common parent was learned with MSE, so this tests a continuation objective rather than the best loss for training from scratch.

*Table 5. Paired loss comparison on all 100 validation views. Normal error is the mean of five diagnostic-view means. The main 200-view test models are unchanged.*

| Refinement image loss | PSNR (dB) | SSIM | LPIPS | Normal error (degrees) |
| --- | ---: | ---: | ---: | ---: |
| MSE | 24.1249 | 0.874920 | 0.093994 | 1.5728 |
| L1 + D-SSIM | 23.5677 | 0.876249 | 0.084458 | 1.5759 |

LPIPS improves on all 100 views, with a 10.15% lower mean; SSIM improves on 64/100. PSNR improves on 1/100 and foreground PSNR on 4/100, with mean losses of 0.5572 and 0.5517 dB respectively. Coverage remains complete. Reference-normal error is essentially unchanged, differing by only +0.0031°. Thus the image objective changes the perceptual/pixelwise trade-off without establishing more accurate geometry.

![Refinement loss comparison](../reports/assets/2026-11-07-refinement-loss/reflection-highlight-view-0.png)

*Figure 8. `justChromeBall`, validation view 0: target / MSE refinement / L1 + D-SSIM refinement. Identical actual central crops. Contrast changes, but thin branches remain smeared or warped. The full five-view comparison and normal maps are preserved in the supplementary loss experiment.*

This is a consistent LPIPS benefit, not solved sharp reflection reconstruction. Changing both L1/MSE and the structural term prevents attributing the effect to D-SSIM alone; the relative scale of geometry penalties also changes. The trial remains a supplementary development result, with no test evaluation or further loss sweep.

### 6.5 What the experiments establish

| Question | Evidence | Supported inference | Remaining alternative or limit |
| --- | --- | --- | --- |
| Does HDR improve the matched SH3 comparison? | Better image metrics on all 200 test views; foreground gains and actual crops | Broad improvement for the complete HDR treatment in this scene | Final geometry can differ; components are not independently ablated |
| Why does sharp detail remain missing? | Normal correction helps with reference light but reverses under learned light | Lighting and normal fit interact; accurate normals matter for detailed reflections | Learned materials, shader and sampling mismatch remain |
| Does stronger consistency correct the normals? | Higher contributor coherence, slightly worse reference-normal and image scores | Self-consistency is not sufficient evidence of correct geometry | One coefficient, schedule and seed |
| Does L1/D-SSIM refinement recover detail? | Lower LPIPS on 100/100 validation views, lower PSNR, similar normal error; fine detail still missing | Loss choice changes the quality trade-off without resolving the structural failure | MSE-trained parent; combined objective and relative regularizer scale change |

Earlier Materials experiments provide a useful cross-check: the protected deferred-plus-SH3 model improves PSNR by 1.1147 dB but worsens LPIPS, and a logo refinement improves global scores without recovering recessed geometry. Local-reflection rays also hit nearby objects. These observations motivated the plain sphere and the use of metrics, crops and independent normal diagnostics together. Appendix B preserves this supporting history separately from the main comparison.

### 6.6 Limits of the evidence

The main study has one scene and one training seed; different cameras are not independent repetitions of training. Its conclusion is restricted to this setup. The SH3 and HDR models share a geometry policy but can learn different surfaces, and the combined HDR/angular treatment changes several appearance components. An equal-final-geometry or forward-shading control would be needed for stronger causal attribution.

The reference-light/normal studies use evaluation-only information on five previously inspected development cameras. They diagnose the fitted model, rather than demonstrating an improvement learned from training RGB.

The main selection was recorded before its 200-view evaluation, but later development occurred after those scores were visible. Historical Materials test scores had also been inspected earlier. These test results should therefore be described as a frozen recorded comparison, not a newly untouched benchmark for subsequent tuning. No later experiment changes the main reported models.

The model family was informed by scene-material inspection. Its image-reconstruction results do not establish calibrated light/material recovery or general relighting. Cost measurements retain the scopes in Appendix A.4.

## 7. Conclusion and future work

Deferred HDR/angular shading improves the matched sphere comparison substantially: +4.3077 dB test PSNR with lower LPIPS and smaller serialized checkpoints than SH3. High image scores still coexist with blurred and warped reflections. Lighting/normal interventions show contributions from both fitted inputs. Stronger consistency does not improve reference normals; L1/D-SSIM refinement lowers LPIPS at a PSNR cost without recovering the missing fine structure.

With more time, the first change would be to inspect normal/light errors earlier and test one targeted optimization change with independent normal diagnostics. Repeated seeds and another controlled reflective scene would check generalisation. Fixed-final-geometry and forward-shading controls would separate appearance capacity from rendering/optimization effects; position-dependent appearance would address local reflections in complex scenes.

The main lesson is that explicit normals and deferred reflection provide a useful representation, but neither guarantees correct disentanglement. Matched comparisons, reference diagnostics and preserved failures are necessary to distinguish an improved image fit from recovered reflection geometry.

## AI-use declaration

**Proposed wording for author review:** OpenAI Codex assisted with code inspection, implementation and debugging, experiment tooling, literature lookup, evidence consolidation, and drafting this report. Checks recorded in the repository include gradient checks, targeted regression checks, source/input/checkpoint fingerprints and paired image evaluation. The report draft was checked against stored metrics and actual experiment figures. **[AUTHOR REVIEW: verify this account, add any other AI tools used, and describe the checks and research decisions you personally performed. Confirm every final claim and citation before submission.]**

## References

1. Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics, 42(4). [Paper and implementation](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
2. Dai, P., Xu, J., Xie, W., Liu, X., Wang, H., and Xu, W. (2024). *High-quality Surface Reconstruction using Gaussian Surfels*. SIGGRAPH Conference Papers. [Paper](https://arxiv.org/abs/2404.17774); [implementation](https://github.com/turandai/gaussian_surfels).
3. Ye, K., Hou, Q., and Zhou, K. (2024). *3D Gaussian Splatting with Deferred Reflection*. SIGGRAPH Conference Papers. [Paper](https://arxiv.org/abs/2404.18454); [implementation](https://github.com/gapszju/3DGS-DR).
4. Kouros, G., Wu, M., and Tuytelaars, T. (2025). *RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting*. arXiv:2504.18468. [Paper](https://arxiv.org/abs/2504.18468).
5. Huang, B., Yu, Z., Chen, A., Geiger, A., and Gao, S. (2024). *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*. SIGGRAPH Conference Papers. [Paper and implementation](https://surfsplatting.github.io/).
6. Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J. T., and Srinivasan, P. P. (2022). *Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields*. CVPR. [Paper](https://dorverbin.github.io/refnerf/).
7. Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE Transactions on Image Processing, 13(4), 600–612. [Paper](https://www.cns.nyu.edu/~lcv/ssim/).
8. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR. [Paper and implementation](https://richzhang.github.io/PerceptualSimilarity/).
9. CP4282 course notes and assignment framework. *Gaussian Splatting Notes*. [Course notes](https://weitsang.github.io/gaussian-splatting-notes/); [project requirements](https://weitsang.github.io/gaussian-splatting-notes/project/). Accessed 7 November 2026.
10. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV. [Project and dataset source](https://www.matthewtancik.com/nerf); [code](https://github.com/bmild/nerf).

## Appendix A. Reproducibility and data attribution

The repository is [github.com/wenn5055/3DGSresearch](https://github.com/wenn5055/3DGSresearch). The recorded study's code/config/evidence commit is `59c553f`; individual runs additionally retain source/input fingerprints and source snapshots. This report is a later documentation revision. Dataset and scene attribution are recorded in `data/blender/SOURCE.md` and the consultation/source notes: Materials derives from the NeRF scene distribution, with the original Materials asset credited to elbrujodelatribu (CC0). The sphere capture retains the packed forest illumination and Materials camera templates. [NeRF source](https://github.com/bmild/nerf), [asset attribution](https://www.blendswap.com/blend/10120).

### A.1 Re-evaluate the recorded main models

Install the research requirements in a Python environment with CUDA support. Run the following from the repository root, using a fresh output directory:

```powershell
python scripts/evaluate_reflection_suite.py `
  outputs/Nov03-1357-justChromeBall-staged-sh3-refine-5k `
  outputs/Nov02-2023-justChromeBall-staged-hdr-refine-5k `
  outputs/Nov03-1402-justChromeBall-staged-hdr-refine-5k `
  --labels SH3 HDR-1.0 HDR-0.75 `
  --output outputs/report-main-test-repeat `
  --manifest transforms_test.json --views 0 13 73 88 `
  --metrics psnr ssim lpips --cpu-threads 2
```

Keep each model's PLY and surfel sidecar together; SH3 also requires its SH sidecar. All 400 sphere images and split manifests are tracked in Git. The main checkpoint bundle has been prepared locally as `outputs/final-report-reproduction/checkpoints.zip`, with model bytes verified against the frozen selection hashes. Extract it into the root of a **fresh clone** to preserve the original workspace's run metadata. Packaged metadata uses relative dataset/output paths and retains originals separately. **[AUTHOR REVIEW: upload or attach this prepared bundle and add its persistent location here. Full checkpoints remain outside Git.]**

### A.2 Retrain the historical 100k setup

The reproduction helper rebases all six archived stage configurations into the current checkout, retaining every non-path setting and the 100k count. It generates a training-only initializer, copies it into each arm with the recorded tangent-scale factor, and connects actual frozen-stage outputs to refinement. Preview the configs without initialization or training:

```powershell
python docs/final-report/reproduce_study.py --output outputs/reproduction-preview
```

To execute initialization and all six training stages, supply a different fresh output directory and `--run`. This uses the research Python environment and performs validation only; it does not automatically inspect test views. Preparation was checked for this revision; full training was not repeated. Fresh training may vary with software/GPU versions. The current 200k defaults and later density experiments are separate from this historical setup.

### A.3 Diagnostic reproduction and evidence

The reference-lighting and normal-intervention commands are in `docs/experiments/2026-11-03-reference-lighting-swap.md` and `docs/experiments/2026-11-03-reference-normal-swap.md`. They require the recorded Blender geometry export and aligned environment arrays. Reference exports remain evaluation-only. The normal-weight closeout includes paired validation, actual crops and normal errors; see the evidence map beside this report for exact artifacts and hashes.

### A.4 Cost measurement details

Forward timing warms each of four illustrated views once and measures ten repeats of the sequence with device synchronization. Required model-file totals include PLY plus surfel and, where needed, SH sidecars. Table 2 sums optimizer-stage times only; the common training-mask initializer separately took 68.79 s, excluding capture/evaluation. The Warp allocation figures exclude driver/context and external allocations. The evaluation-process high-water mark is cumulative across models and is not used as per-model training VRAM. Training jobs may overlap other work, so these records do not establish training throughput differences.

## Appendix B. Supporting scenes and geometry

Original Materials mixes reflection modeling with more complex geometry and local visibility. Its protected SH3/deferred-plus-SH3 comparison gives PSNR 27.3575/28.4721 dB and LPIPS 0.036361/0.038978: better pixel fit is not uniformly better perceptual quality. Chrome normal error improves from 25.88° to 15.71°, but thin tree reflections remain blurred. A four-camera ray audit also found reflected rays hitting nearby geometry, which a direction-only environment cannot directly model. These are a distinct protocol, not additional samples of the sphere experiment.

The enlarged Chrome scene includes a recessed logo and base. A matched fivefold position-learning-rate trial improves global image metrics but changes logo-region normal error only from 28.23° to 27.84°, leaving the recessed structure missing. This local failure motivated simplifying the scene rather than continuing generic parameter tuning.

![Preserved logo failure](../reports/assets/2026-11-02-logo-position5x/logo-comparison.png)

*Figure B1. Enlarged Chrome, validation view 18, fixed logo rectangle. Columns: reference, original refinement, fivefold position-rate refinement; rows: RGB and normals. Actual diagnostic comparisons preserve the missing recessed structure. Scene/protocol differ from the main sphere study.*

![Actual normal comparison](../reports/assets/2026-11-07-report-analysis/normal-example-view-0.png)

*Figure B2. `justChromeBall`, validation view 0. Columns: reference mesh normal, learned shading normal from HDR-0.75, and angular error. RGB normals encode world-space directions; error clips at 30°. This is a layout crop of the actual five-view diagnostic sheet, whose full version remains in the supplementary evidence; it is not training supervision.*
