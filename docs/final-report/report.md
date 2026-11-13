# Reconstructing Reflections with Deferred Gaussian Surfels

**CP4282 individual project — Final, revised 13 November 2026**<br>
**Author:** Tang Xiao Di Wendy<br>
**Code and experiment records:** [github.com/wenn5055/3DGSresearch](https://github.com/wenn5055/3DGSresearch)

## Abstract

A chrome sphere reflects a detailed image of its surroundings. Reconstructing that image is difficult: a small error in the surface direction can move an entire reflected branch or highlight. This project extends the course's Warp Gaussian splatting trainer with flat surface patches and a reflection model. On a controlled sphere scene, it compares this model with degree-three spherical harmonics (SH3), using the same starting surface and training budget. Across 200 test views, the reflection model increases mean PSNR from 20.27 to 24.57 dB and SSIM from 0.821 to 0.881, while reducing LPIPS by 35.1%. Fine reflections nevertheless remain blurred or distorted. Further experiments examine lighting, surface normals and training losses. Correcting both lighting and normals restores much more structure; increasing normal–depth consistency alone does not. Replacing MSE with L1 plus D-SSIM improves validation LPIPS by 10.1% but lowers PSNR. The project shows both the benefit of explicit reflection modelling and why a better image score does not necessarily mean correct geometry or sharp reflections.

## 1. Introduction

The aim of novel-view synthesis is to reconstruct a scene from photographs and render it from cameras that were not used for training. Reflective objects are a demanding case. As the camera moves, their highlights and reflected surroundings move across the surface. A model must explain these changes while also learning the object's shape.

This project asks: **Can a surface-based reflection model reconstruct a chrome sphere better than SH3 appearance under the same initialisation and training budget?** It also investigates two practical questions: what prevents the remaining reflections from becoming sharp, and which training losses help?

The work builds on the course trainer and the appearance-model discussion in Unit 11. Its main addition is a renderer that first combines surface information at each pixel, then computes the reflection. The experiments compare image quality, inspect reflection crops and measure surface-normal errors. Additional experiments test whether stronger geometric consistency (Trial 1) or a different image loss (Trial 2) improves the remaining reflection errors.

### 1.1 Initial exploration on Materials

The original plan included Materials and Ship. Work began on Materials by extending the course trainer from fixed RGB to view-dependent SH3 colour, building a working baseline before investigating reflection models. With staged MSE/D-SSIM training, the early fixed-RGB model reached **24.9365 dB PSNR / 0.907943 SSIM**, while SH3 reached **26.0690 dB / 0.922090**. These are means over the same 200 test views at 800 × 800, with seed 7, 30,000 completed iterations, 1,000 random initial Gaussians and capacity for one million.

![Early Materials baseline comparison](../reports/assets/2026-11-11-report-revision/materials-initial-exploration.png)

*Figure E1. Original Materials, test view 13. Columns: Blender target, fixed RGB with staged MSE/D-SSIM, SH3 with staged MSE/D-SSIM. Top: full images; bottom: identical Chrome crops, image bounds (200, 505)–(380, 685). Actual saved renders show recognisable materials and reflected colour, but both models miss the target's sharp Chrome detail. View 13 reuses an existing illustration camera; this is an early baseline result, not the later surfel comparison.*

This was an **initial exploration phase**, before the later decisions about training-only surface initialisation, fixed surfel count and appearance-first refinement. The version of the course assignment trainer used in these early experiments also used validation images when computing its initial mean colour; the later study uses training-only initialisation. These results document baseline development rather than an exact reproduction of the course's published scores, and are not pooled with the main study. [Archived results and protocol](https://github.com/wenn5055/3DGSresearch/blob/main/docs/findings.md#matched-materials-test-comparison)

In the initial dataset chosen, the Nerf synthetic Materials dataset (also in the course notes), there are a few complications that muddles the research direction. Chrome reflects nearby objects as well as the forest. A single environment map indexed only by direction cannot directly represent how those local reflections change with surface position and visibility. In the separate enlarged-Chrome scene, silhouette-based initialisation missed a recessed logo, adding a local shape-recovery problem.

I intentionally restrict the main study to an isolated, smooth chrome sphere under forest lighting. Removing nearby objects and recessed geometry makes it easier to investigate how the appearance approaches reconstruct environment reflections, without combining those results with failures in local reflection transport or hidden concavities. The task still requires learning surface normals and lighting from images. Materials and enlarged-Chrome results remain supporting evidence; the sphere study does not establish performance on more complex scenes.

## 2. Background: from directional colour to reflections

### 2.1 What this project takes from the course

Gaussian splatting represents a scene with many small, overlapping Gaussian blobs. A renderer projects them into an image and blends their contributions. Training changes their positions, shapes, opacity and appearance to reduce image error. Units 8 and 9 provide the Warp training framework used as this project's starting point. [Unit 8](https://weitsang.github.io/gaussian-splatting-notes/unit-8/), [Unit 9](https://weitsang.github.io/gaussian-splatting-notes/unit-9/)

Unit 11 explains how spherical harmonics let each primitive's colour change with viewing direction. SH is a weighted combination of fixed directional functions: lower degrees describe broader changes, while higher degrees provide more angular detail. The unit also motivates using surface normals and reflected directions for reflective appearance. That is the direction pursued here. [Unit 11: appearance beyond SH](https://weitsang.github.io/gaussian-splatting-notes/unit-11/#project-ideas-appearance-models-beyond-spherical-harmonics)

Our control uses **SH3**, meaning spherical harmonics up to degree three. It stores 16 coefficients per colour channel, or 48 scalar values including the constant term. It predicts colour from viewing direction. The reflection model instead uses a surface normal to determine which part of a shared image of the surroundings should be visible, combined with base colour and an angle-dependent reflection strength. This gives neighbouring surface elements a common source of reflected detail. The comparison tests these two ways of predicting appearance. Several appearance components change together, so it does not isolate the contribution of each component.

### 2.2 Papers that informed the implementation

| Work | Relevant idea | How it is used here |
| --- | --- | --- |
| [3DGS, Kerbl et al.](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | Gaussian scene representation and splatting | Background for the course framework |
| [Gaussian Surfels, Dai et al.](https://arxiv.org/abs/2404.17774) | Flat Gaussian surface patches and normal–depth consistency | Surface representation and geometry loss |
| [3DGS-DR, Ye et al.](https://arxiv.org/abs/2404.18454) | Compute reflections after blending surface attributes | Two-pass reflection rendering |
| [RGS-DR, Kouros et al.](https://arxiv.org/abs/2504.18468) | Deferred surfel shading and residual appearance | Closely related work; its full system is not reproduced |

[2DGS](https://surfsplatting.github.io/) is also relevant to surface-based rendering, but uses perspective-accurate ray–splat intersections. This project retains the course renderer's projection approximation: each Gaussian contributes over a soft, elliptical area in the image, called its **projected footprint**. A pixel filter broadens that footprint to reduce aliasing, such as flickering when a primitive becomes smaller than a pixel. Appendix A gives the technical distinction from 2DGS.

[Ref-NeRF](https://dorverbin.github.io/refnerf/) provides broader context for describing appearance through reflected directions. It uses a **neural radiance field (NeRF)**: a neural network predicts density and view-dependent colour at locations in space, and rendering combines samples along each camera ray. Our model instead stores explicit Gaussian surface elements.

These papers inform the design. The project contribution is implementing and investigating the chosen combination in Warp, with controlled comparisons and diagnostics. It does not claim a new rendering algorithm.

## 3. How the method works

### 3.1 Represent the object with surfels

A **surfel** is a small, flat surface element. Here it is a Gaussian patch with a position, orientation and two scales controlling its width within the surface plane. Its thickness is fixed to zero. Its **normal** is a unit vector perpendicular to the patch, describing which direction the surface faces.

The initial surfels come from the training images' alpha masks, which separate the object from the background. Combining these silhouettes across cameras estimates a surface around the object. This is a training-derived estimate, not the Blender mesh. The same 100,000 surfels initialise both main methods. As stated in Unit 12, initialisation matters because a splat receives image gradients only where its projected footprint contributes. Starting surfels near the estimated object surface helps foreground pixels guide training, rather than allocating many primitives to empty background. [Unit 12: initialisation and local optimisation](https://weitsang.github.io/gaussian-splatting-notes/unit-12/#why-the-renderer-makes-optimization-local)

Silhouettes cannot recover hidden recesses. That mattered in the earlier logo experiment and helped motivate using a plain sphere.

### 3.2 Render the surface, then compute its reflection

Rendering has two steps:

1. **Combine surface information at each pixel.** Blend the contributing surfels' base colours, normals, depth and reflection strengths, using the same visibility weights as image compositing.
2. **Shade the pixel.** Use the combined normal and camera ray to find the reflected direction. Sample the learned environment map in that direction, then blend the reflected light with the base colour.

An **environment map** is an image of the surroundings indexed by direction. The map here is learned from training images; the original forest texture is not supplied to the optimiser. It contains positive high-dynamic-range (HDR) lighting values and has resolution 512 × 256.

The shader also learns how the reflection contribution changes with viewing angle. It uses a shared table of 16 adjustable values. This choice followed inspection of the scene's artistic Chrome material, whose colour varies with angle. The material inspection informed the model's design; its actual curve values were not copied into training.

The reflection is evaluated **after** surface attributes have been combined, which is why the method is called deferred shading. Overlapping surfels then contribute to one pixel's shading calculation. During training, an image error can therefore update the normals of all those contributors. This is the gradient path motivated by 3DGS-DR and discussed in Unit 11. The exact equations are provided in Appendix A.

The implemented model is an HDR reflection blend. It does not estimate a full roughness-based material model or trace reflections of nearby objects.

![Schematic of the implemented rendering and training pipeline](../reports/assets/2026-11-07-report-readability/method-pipeline.svg)

*Figure M1. Schematic of the implemented method, not an experimental render. Image loss uses RGB; mask and normal–depth losses use the surface maps. Feedback updates appearance first, then geometry as well during refinement.*

### 3.3 Fit appearance first, then refine the surface

Both main methods follow the same schedule:

| Stage | Iterations | Parameters that can change |
| --- | ---: | --- |
| Appearance fitting | 10,000 | Appearance only; position, scale, orientation and opacity stay fixed |
| Refinement | 5,000 | Geometry, opacity and appearance |

At the start of training, appearance is still inaccurate, including the environment lighting in the reflection model. If geometry moves immediately, it can reduce image error by compensating for appearance errors rather than recovering the correct surface. The first stage therefore holds the estimated surface fixed while appearance is fitted. This is intended to reduce early geometry–appearance compensation; refinement then allows smaller geometry updates alongside appearance. Adam, the parameter-update algorithm, restarts its optimiser state at the transition; the learned model parameters are retained.

Both methods use the same geometry-loss settings and learning-rate schedules. Each retains **100,000 surfels** throughout training and completes **15,000 iterations: 10,000 appearance-fitting steps plus 5,000 refinement steps**. No surfels are added or removed, and early stopping is disabled. Here, a reported point count means the number of surfel primitives, not image pixels or sampled camera rays. The SH3 control gradually enables its three directional bands during the first stage. Following the same update policy does not force the two methods to learn identical final surfaces.

## 4. Loss functions: what training is asked to improve

Three types of loss are used here: image matching, foreground-mask matching and normal–depth consistency. They address different errors.

### 4.1 Image loss: match the reference colours and structure

The main SH3/HDR comparison uses **mean squared error (MSE)** in both training stages, retaining the shared objective from the fixed-surface appearance experiment. Unit 10 discusses its limitation: matching pixel values does not explicitly require neighbouring pixels to form the correct local structure. [Unit 10: MSE as a starting point](https://weitsang.github.io/gaussian-splatting-notes/unit-10/#why-mse-is-a-useful-but-limited-starting-point)

The additional loss experiment uses **L1 + D-SSIM**:

**L1 is mean absolute error:** it averages the absolute differences between rendered and target RGB values. Unlike MSE, it does not square those differences, so large errors receive less disproportionate weight. SSIM compares local brightness, contrast and structure; converting it to D-SSIM makes smaller values better. Together, L1 and D-SSIM ask for colour agreement and local structural similarity. Our implementation uses reflected, uniform 11 × 11 RGB windows.

<div class="equation">L<sub>image</sub> = 0.8 L<sub>1</sub> + 0.2 L<sub>D-SSIM</sub>, &nbsp; L<sub>D-SSIM</sub> = (1 − SSIM) / 2.</div>

Unit 10 combines **MSE** with D-SSIM after a warm-up. One additional experiment—the **image-loss trial, separate from the main comparison**—instead combines **L1** with D-SSIM from the first refinement step. It therefore tests a replacement objective, not an exact reproduction of the course curriculum. Because the surface pass already renders full images, its MSE baseline does not use the course curriculum's sparse warm-up. LPIPS is used for evaluation only, not as a training loss.

### 4.2 Mask loss: keep the object separate from the background

RGB matching alone can miss a geometry error if an incorrectly placed surfel has the same colour as the background. The mask loss separately compares rendered coverage—the accumulated opacity at each pixel—with the training image's alpha mask. It penalises missing object coverage and unwanted background coverage, using binary cross-entropy.

The mask weight is **0.1** during refinement. This supplies foreground information independently of the RGB colours. It uses only training masks and does not supply the true surface depth or orientation.

### 4.3 Normal–depth consistency: make two descriptions of the surface agree

Although Blender provides reference normal and depth maps, using them as training targets would supply additional geometric information unavailable in our RGB-and-mask setting. We therefore reserve those maps for evaluation and, for normals, diagnostic substitutions. Training instead uses **normal–depth consistency as a geometric regulariser**, encouraging two surface descriptions predicted by the model to agree without supplying the true geometry.

The model describes surface orientation in two ways. Surfel orientations produce a normal map: an image recording the surface direction at each pixel. A second normal is estimated from the model's rendered depth map.

**The silhouettes provide the initial 3D surface. The depth map used by the consistency loss is rendered from the current surfel geometry, so it evolves during refinement.**

For each pixel, the renderer computes where the camera ray meets each contributing surfel's plane. It blends those camera-z depths using the same front-to-back visibility and opacity weights used for image compositing, then divides by the accumulated coverage. The result is a visibility-weighted average depth, rather than simply the nearest intersection. No separate depth network or supplied depth image is used. Blender's reference depth in the evaluation comparisons is a separate map exported from the known mesh and does not enter this training loss. Appendix D.2 compares the rendered normals and depth with these independent mesh references.

To obtain that second normal, use the camera geometry and neighbouring depth values to locate their points in 3D. The vectors from left to right and from top to bottom describe the local surface slopes. Their cross product gives a direction perpendicular to that surface. After normalisation and alignment to the same camera coordinates, compare it with the rendered surfel normal:

<div class="equation">L<sub>normal-depth</sub> = mean(1 − n<sub>rendered</sub> · n<sub>from depth</sub>).</div>

Both vectors have unit length. Agreement gives zero loss; angular disagreement increases it. Gradients flow through both paths, so training can change surfel orientations and the geometry producing the depth map. Unreliable pixels near depth jumps, low coverage or invalid neighbourhoods are excluded.

This encourages agreement with the local surface slope; it does not simply make all normals point in one direction. Crucially, **both predictions come from the model itself**. Two incorrect predictions can agree, so consistency is not proof of accurate geometry. The separate Blender-normal evaluation checks that accuracy.

### 4.4 The objective actually used

The main refinement objective is

<div class="equation">L = L<sub>MSE</sub> + 0.1 L<sub>mask</sub> + λ<sub>n</sub>(t) L<sub>normal-depth</sub>.</div>

The normal regulariser's weight increases from zero to **0.01** over the first 1,000 refinement steps. The mask term, weighted **0.1**, supplies an additional silhouette constraint. Flat surfels and a fixed population size constrain the representation rather than adding loss terms. During the frozen stage, geometry losses are disabled. The image-loss trial replaces only L<sub>MSE</sub> with the L1/D-SSIM expression above. The consistency-weight trial changes the final normal weight from 0.01 to 0.03.

These coefficients are fixed experimental settings, not demonstrated optima. Keeping geometry coefficients unchanged also does not preserve their relative influence when the image-loss scale changes. Unit 10's discussion of geometric losses motivates checking geometry separately from image scores. [Unit 10: geometric terms](https://weitsang.github.io/gaussian-splatting-notes/unit-10/#3-add-a-geometric-term-when-extra-signals-are-available)

## 5. Experimental setup and evaluation

### 5.1 Scene, splits and fair comparison

The `justChromeBall` scene contains one smooth chrome sphere under forest lighting. It was captured in Blender at 800 × 800 pixels, with 100 training, 100 validation and 200 test views. Cameras come from the Materials templates, with disjoint split poses. Targets are composited on white.

![Sphere reference views](../reports/assets/2026-11-02-sphere/dataset-views.png)

*Figure 1. Chrome sphere, validation views 0, 13, 73 and 88, left to right. Actual Blender target images show the reflected scene the model should reconstruct.*

The main comparison holds the starting surface, 100,000 surfels, seed 7, resolution and 15k-step schedule fixed. SH3 uses directional colour; HDR-1.0 uses the reflection model. Both use surfels, so this comparison concerns appearance on a common representation rather than a switch from volumetric Gaussians to surfels.

The supplementary initialisation experiment, **HDR-0.75**, starts each surfel's two in-plane scales at 75% of their original values; all other initialiser fields are preserved. This reduces initial surfel widths, while retaining 100,000 surfels and the same image resolution. The scales can change during refinement. This variant is included as a supplementary result and supplies the lighting/normal diagnostics. The primary comparison is SH3 versus same-size HDR-1.0.

Only training RGB and alpha supply fitting gradients. Validation images are used to track reconstruction quality and compare experimental settings; they do not contribute training gradients. Blender normals and the original environment are used only for evaluation and diagnostic rendering. The loss and normal-weight trials are evaluated on validation views and do not replace the main test models.

### 5.2 What is measured

| Measure | What it tells us | Better direction |
| --- | --- | --- |
| PSNR | Pixelwise colour accuracy, expressed from MSE in decibels | Higher |
| SSIM | Similarity of local image structure | Higher |
| LPIPS, AlexNet | Similarity of image features computed by a fixed neural network | Lower |
| Foreground PSNR | Pixel accuracy restricted to the object | Higher |
| Normal error | Angular difference from reference mesh normals | Lower |

Unit 6 recommends reading image metrics together and investigating disagreements. We therefore pair them with actual reflection crops. Metrics use float renders with RGB values scaled to [0, 1], rather than saved 8-bit figures. Foreground PSNR uses reference alpha ≥ 0.95; coverage counts foreground pixels whose rendered coverage reaches 0.5. [Unit 6: evaluation](https://weitsang.github.io/gaussian-splatting-notes/unit-6/#reading-the-three-together)

## 6. Results and what they mean

### 6.1 The reflection model improves the main comparison

*Table 1. All 200 sphere test views; 100k surfels and 15k training steps per model. Higher PSNR/SSIM and lower LPIPS are better.*

| Model | PSNR (dB) | SSIM | LPIPS | Foreground PSNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| SH3 | 20.2655 | 0.821398 | 0.128760 | 13.6520 |
| HDR-1.0, matched primary | 24.5732 | 0.880892 | 0.083517 | 18.2494 |
| HDR-0.75, additional variant | 24.7453 | 0.884031 | 0.080988 | 18.4427 |

At the same starting size, HDR gains **4.31 dB PSNR** and reduces LPIPS by **35.1%**. The foreground gain is 4.60 dB, showing that the benefit is also present on the object. All models retain full foreground coverage under the stated threshold.

The benefit appears on **200/200 views** for PSNR, SSIM, LPIPS and foreground PSNR. The per-view PSNR gain has median 4.4439 dB and range 2.9349–5.5425 dB. This describes variation across cameras in one fitted model pair; it does not measure uncertainty across training seeds.

![Distribution of test-view improvements](../reports/assets/2026-11-07-report-analysis/paired-test-effects.png)

*Figure 2. Chrome sphere, all 200 test views, HDR-1.0 versus SH3. Each bar counts views within an improvement range: the horizontal axis gives the gain, and the vertical axis gives the number of views. PSNR gain is HDR minus SH3, in dB; LPIPS reduction is SH3 minus HDR, in absolute metric units. Zero means no change; values to its right favour HDR. All 200 views improve in every panel. These histograms use saved metrics, without additional rendering or model selection.*

![Full-view reference, SH3 and HDR-1.0 comparison](../reports/assets/2026-11-12-full-views/reference-sh3-hdr1-full-views.png)

*Figure 3a. Chrome sphere, test views 0, 13, 73 and 88, top to bottom. Columns: Blender reference, SH3, HDR-1.0. Both trained models use 100,000 surfels and 15k training steps. These actual saved full-frame images use matching camera framing and white backgrounds. HDR-1.0 captures more recognisable tree/sky regions than SH3, while fine reflections remain softer than the target. All four views reuse the original illustration cameras.*

The full views show the overall reconstruction; the enlarged reflection crops below make the remaining detail errors easier to inspect.

![Actual test reflection crops](../reports/assets/2026-11-07-report-analysis/test-crops-views-0-73.png)

*Figure 3b. Sphere test views 0 and 73. Columns: target, SH3, HDR-1.0, HDR-0.75. Actual crops show more tree/sky structure with HDR, while thin branches and fencing remain smeared or distorted. The full four-view crop comparison remains in the experiment evidence.*

HDR-0.75, whose two in-plane surfel scales start at 75% of their original values, adds a modest mean gain of 0.17 dB over HDR-1.0. It improves PSNR on 198/200 views and LPIPS on 189/200. It does not remove the visible reflection defects.

### 6.2 Refinement improves reflected structure, with remaining failures

For size-1.0 HDR, adding 5k refinement steps improves validation PSNR from 22.4996 to 24.1250 dB and LPIPS from 0.119765 to 0.093994. PSNR improves on 95/100 views; SSIM and LPIPS improve on all 100. This comparison includes extra training as well as geometry refinement, so it does not isolate the effect of geometry updates alone.

![Visible improvement after refinement](../reports/assets/2026-11-11-report-revision/refinement-improvements.png)

*Figure 4a. Chrome sphere, validation views 0 and 13, top to bottom; size-1.0 HDR. Columns: target, frozen appearance fit at 10k steps, refinement at 15k total. Identical reflection crops show better separation of tree/sky regions in view 0 and a clearer sky opening, tree shapes and foreground boundary in view 13. Fine branches and fencing remain unresolved. These actual saved renders reuse the original illustration views; the complete four-view comparison is preserved in the experiment evidence.*

The improvement is visible before considering the scores: broad reflected structures become more recognisable after refinement, although they remain distorted.

![Preserved refinement failure](../reports/assets/2026-11-02-sphere/regression-view-51.png)

*Figure 4b. Chrome sphere, validation view 51: target, frozen size-1.0 HDR at 10k steps, refined HDR at 15k total. This is the largest PSNR-regression view, selected after inspecting validation scores. It preserves the missing ground and fence detail despite the average improvement.*

The reflection model also uses less stored checkpoint data: **8.54 MB versus 23.50 MB** for SH3 in this implementation. A shared environment replaces many per-surfel directional coefficients. Observed forward rates are of similar scale, about 274–304 FPS. Exact timing/memory definitions and measurements are in Appendix C; these values do not establish a general shader-speed advantage.

### 6.3 Lighting and normal errors both limit detail

These lighting/normal diagnostics use the saved **HDR-0.75 additional variant**, rather than the primary HDR-1.0 model. All substitutions below use the same HDR-0.75 checkpoint; they diagnose its errors and do not replace the main comparison. Its learned environment is visibly less detailed than the original forest map, even when both are shown at the same resolution and display exposure.

![Original and learned environment maps at one matched exposure](../reports/assets/2026-11-11-report-revision/environment-comparison.png)

*Figure 5. Forest lighting for the chrome sphere: original map on the left; environment learned by HDR-0.75 on the right. Both are 512 × 256 in the renderer's direction convention and use the same display exposure (0 EV). Actual maps show missing branch, fence and ground structure in the learned lighting. This is a display comparison, not another training variant.*

Replacing the learned map with the original lighting lowers validation LPIPS from 0.091814 to 0.079982, but reflections remain distorted and PSNR falls. To check the normal contribution, a second diagnostic renders the same saved model with learned or reference lighting and learned or reference shading normals.

Here **reference** means Blender information used only for this diagnostic. Reference normals come from the actual sphere mesh, rather than an ideal analytic sphere. Normal replacement changes the blended normal's direction after rasterisation; coverage, depth, colours and learned material parameters stay fixed. It affects both reflected direction and the angle-dependent blend.

*Table 2. Means over five fixed validation views: 0, 13, 73, 88 and 51. Same HDR-0.75 model in all cases.*

| Lighting | Normals | PSNR (dB) | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| Learned | Learned | 23.8554 | 0.871914 | 0.096202 |
| Reference | Learned | 20.4016 | 0.844558 | 0.085452 |
| Learned | Reference | 23.0536 | 0.867943 | 0.101984 |
| Reference | Reference | 22.7073 | 0.943615 | 0.056796 |

![All four lighting and normal combinations](../reports/assets/2026-11-11-report-revision/lighting-normal-permutations.png)

*Figure 6. Chrome sphere, validation view 0, same HDR-0.75 checkpoint and identical reflection crop. Left: Blender target. In the 2 × 2 grid, rows use learned/reference lighting and columns use learned/reference normals. Top left: trained result, blurred and rippled; top right: reference normals with learned lighting, still blurred; bottom left: reference lighting with learned normals, detailed but rippled; bottom right: both reference inputs, recovering much more recognisable structure while still failing to reproduce the Blender target's contrast and darker colour tones. All four are actual saved diagnostic renders. Appendix D.4 shows two additional views.*

Correcting normals helps strongly **when reference lighting is used**: PSNR rises by 2.31 dB and LPIPS improves by 0.0287. Under learned lighting, the same normal replacement instead worsens mean LPIPS by 0.0058. This dependence on the lighting suggests that the learned map and normals have fitted together; correcting one input alone need not improve the image.

The learned normals' mean error is only about **1.47°**, but even a small error can move fine reflected content.

The joint reference case gives the best SSIM and LPIPS here, but its PSNR remains below the learned pair. Brightness, learned materials and shader/sampling differences remain. This diagnostic reinforces the need to assess structural similarity, pixel accuracy and visible reflection detail together.

The following two loss experiments test changes to the HDR-1.0 refinement objective: a stronger geometric regulariser and a replacement image loss.

### 6.4 Trial 1: Increasing normal–depth consistency

A stronger normal–depth penalty did not improve normal accuracy. This trial changes only the final consistency weight from 0.01 to 0.03, using the same HDR-1.0 frozen parent and 5k refinement.

*Table 3. All 100 validation views for image scores; five diagnostic views for normal error.*

| Normal–depth weight | PSNR (dB) | SSIM | LPIPS | Normal error (degrees) |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 24.1249 | 0.874920 | 0.093994 | 1.5728 |
| 0.03 | 24.0857 | 0.874125 | 0.094244 | 1.5943 |

The stronger weight slightly worsens image scores and reference-normal error. Contributing normals become more aligned with each other, but not more accurate against Blender. This is the distinction in Section 4.3: agreement between the model's normals and depth is not independent evidence that either is correct.

![Actual consistency-weight comparison](../reports/assets/2026-11-07-normal-weight-closeout/reflection-crops.png)

*Figure 7. Sphere validation views 0 and 51: target, consistency weight 0.01, weight 0.03. Identical actual crops preserve the missing branch and fence detail; increasing the weight does not recover it.*

### 6.5 Trial 2: L1 + D-SSIM improves LPIPS at a PSNR cost

The image-loss trial keeps the same HDR-1.0 frozen parent, seed, 5k refinement, learning rates and geometry-loss coefficients. Only the image objective changes from MSE to 0.8 L1 + 0.2 D-SSIM.

*Table 4. All 100 validation views for image scores; five diagnostic views for normal error. This is a follow-up, not a replacement for the main test comparison.*

| Refinement image loss | PSNR (dB) | SSIM | LPIPS | Normal error (degrees) |
| --- | ---: | ---: | ---: | ---: |
| MSE | 24.1249 | 0.874920 | 0.093994 | 1.5728 |
| L1 + D-SSIM | 23.5677 | 0.876249 | 0.084458 | 1.5759 |

LPIPS improves on **all 100 views**, with a **10.15% lower mean**. SSIM improves on 64/100. However, PSNR falls by 0.56 dB on average and improves on only 1/100 views. Foreground PSNR also falls by 0.55 dB, while coverage stays complete. Normal error is essentially unchanged.

![Actual image-loss comparison](../reports/assets/2026-11-07-refinement-loss/reflection-highlight-view-0.png)

*Figure 8. Sphere validation view 0: target, MSE refinement, L1 + D-SSIM refinement. Actual central crops show changed contrast, while thin branches remain smeared or distorted. Appendix D.5 shows the full five-view comparison.*

The loss choice changes the image-quality trade-off, but this trial does not recover the missing fine reflections. It also changes both the pixel penalty and the structural term, so it cannot tell us which caused the gain. A separate MSE + D-SSIM control would be needed to isolate those effects. This is a useful next comparison if loss design is pursued further; it is not a result supplied by the current study.

## 7. Discussion and limitations

Three findings answer the project's questions. First, the reflection model improves the tested SH3 comparison consistently across cameras. Second, accurate normals and lighting matter together: correcting both recovers detail that the trained model misses. Third, neither a stronger consistency penalty nor a better LPIPS score establishes more accurate geometry.

The main comparison has one scene and one training seed. Its 200 cameras provide many views of those fitted models, not 200 independent training repetitions. The method also changes several appearance components together, and its final geometry can differ from SH3's. Further controls would be needed to separate the effects of HDR, angular response and deferred evaluation.

The loss trial changes an existing MSE-trained model rather than starting both objectives from scratch. Its geometry coefficients also have a different relative scale after the image-loss change. The normal/lighting replacement tests use information unavailable during fitting and inspect five development cameras. They explain errors but do not demonstrate recovered lighting or materials suitable for general relighting.

The main settings were selected on validation and recorded before the main test evaluation. Later diagnostics occurred after those test scores were visible; historical Materials test results had also been inspected. The main models are therefore preserved as a recorded comparison, rather than repeatedly retuned using those test results. The follow-up loss and normal-weight trials use validation only.

## 8. Conclusion

Deferred reflection shading improves this controlled sphere reconstruction over SH3: +4.31 dB test PSNR and 35.1% lower LPIPS, with smaller stored checkpoints. It still misses fine reflected structure.

The most useful lesson is to check what an improvement means. Reference-lighting/normal renders reveal errors hidden by a jointly fitted image. Stronger consistency makes learned surface descriptions agree without improving their accuracy. L1 + D-SSIM improves perceptual similarity while sacrificing pixel accuracy. These results support evaluating losses with image metrics, reflection crops and independent geometry checks together.

With more time, I would make those diagnostics part of the process earlier, isolate the image-loss components, and repeat the main comparison across seeds and another reflective scene. The next step should answer one of those specific questions rather than continue broad parameter tuning.

## AI-use declaration

OpenAI Codex assisted with code inspection, implementation/debugging, experiment tooling, literature lookup and report drafting. Outputs were checked against recorded gradient tests, source/input/checkpoint hashes, saved evaluation results and actual rendered figures.

## References

1. CP4282, *Gaussian Splatting Notes*: [Unit 5—differentiable rendering](https://weitsang.github.io/gaussian-splatting-notes/unit-5/), [Unit 6—evaluation](https://weitsang.github.io/gaussian-splatting-notes/unit-6/), [Unit 8—synthetic Warp training](https://weitsang.github.io/gaussian-splatting-notes/unit-8/), [Unit 9—full Warp trainer](https://weitsang.github.io/gaussian-splatting-notes/unit-9/), [Unit 10—training loss](https://weitsang.github.io/gaussian-splatting-notes/unit-10/), [Unit 11—view-dependent appearance](https://weitsang.github.io/gaussian-splatting-notes/unit-11/), [Unit 12—initialisation](https://weitsang.github.io/gaussian-splatting-notes/unit-12/). Accessed 7 November 2026.
2. Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM TOG. [Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
3. Dai, P., Xu, J., Xie, W., Liu, X., Wang, H., and Xu, W. (2024). *High-quality Surface Reconstruction using Gaussian Surfels*. SIGGRAPH. [Paper](https://arxiv.org/abs/2404.17774).
4. Ye, K., Hou, Q., and Zhou, K. (2024). *3D Gaussian Splatting with Deferred Reflection*. SIGGRAPH. [Paper](https://arxiv.org/abs/2404.18454).
5. Kouros, G., Wu, M., and Tuytelaars, T. (2025). *RGS-DR: Deferred Reflections and Residual Shading in 2D Gaussian Splatting*. arXiv:2504.18468. [Paper](https://arxiv.org/abs/2504.18468).
6. Huang, B., Yu, Z., Chen, A., Geiger, A., and Gao, S. (2024). *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*. SIGGRAPH. [Paper](https://surfsplatting.github.io/).
7. Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J. T., and Srinivasan, P. P. (2022). *Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields*. CVPR. [Paper](https://dorverbin.github.io/refnerf/).
8. Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE TIP. [Paper](https://www.cns.nyu.edu/~lcv/ssim/).
9. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR. [Paper](https://richzhang.github.io/PerceptualSimilarity/).
10. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV. [Project and dataset source](https://www.matthewtancik.com/nerf).

## Appendix A. Technical details and attribution

### A.1 Surface representation and depth

Each surfel is a flat Gaussian with rotation Rᵢ and two positive in-plane scales, sₓᵢ and sᵧᵢ. These define its shape and surface direction:

- **Covariance:** Rᵢ diag(sₓᵢ², sᵧᵢ², 0) Rᵢᵀ. The zero third scale makes the surfel flat.
- **Normal:** nᵢ = Rᵢ(0,0,1)ᵀ, perpendicular to the surfel plane.

The renderer retains filtered screen-space Gaussian footprints and front-to-back blending. Depth comes from intersecting each pixel's camera ray with contributing surfel planes, with a guarded fallback to centre depth. It therefore does not implement the complete perspective-accurate 2DGS rasteriser.

The main study disables surfel growth, pruning, opacity resets, the published normal-propagation procedure and early stopping.

### A.2 From surface buffers to reflected colour

The first rendering pass combines surfel contributions at each pixel. Each visibility weight wᵢ includes the surfel's opacity and the light transmitted through preceding contributors.

| Stored quantity | Calculation | Meaning |
| --- | --- | --- |
| Base colour C | Σwᵢcᵢ | Weighted colour sum |
| Normal sum N | Σwᵢnᵢ | Weighted surface directions |
| Coverage A | Σwᵢ | Accumulated opacity |
| Reflection sum S | Σwᵢσ(mᵢ) | Weighted reflection strengths |
| Depth D | Σwᵢzᵢ / max(A, 10⁻⁸) | Average contributor depth along the camera's z-axis |

Here, cᵢ is the surfel's base colour, mᵢ its learned reflection parameter and zᵢ its ray–plane intersection depth. The sigmoid function σ maps mᵢ to a reflection strength between zero and one.

The shading pass then:

1. **Finds the reflection direction.** Normalise N to obtain n = N/‖N‖. The incoming camera-to-surface ray d reflects to r = d − 2(d·n)n.
2. **Samples the lighting.** Bilinearly sample the environment parameters in direction r, then apply softplus to obtain positive HDR radiance.
3. **Blends base colour and reflection.** The blend strength is s = σ(logit(clamp(S/A)) + g(|d·n|)). The function g adjusts the strength with viewing angle by interpolating 16 learned values, initially zero. Base RGB is converted to linear light, mixed with reflected radiance, clipped and converted back to display RGB before composition over white.

Training compares display RGB. Reference environment pixels, mesh normals and material-curve values are not supplied during fitting.

### A.3 How image errors update surfel normals

Because shading uses the combined pixel normal, an image error can update every contributing surfel's normal. Holding visibility weights fixed and omitting the small normalisation epsilon, the gradient reaching surfel i is:

<div class="equation">∂L/∂nᵢ = [wᵢ(I − nnᵀ)/‖N‖] (∂L/∂n).</div>

Here, I is the identity matrix and ∂L/∂n is the loss gradient at the pixel normal. The expression describes the path through normal blending; the implementation also differentiates the compositing weights and angle-dependent reflection blend.

### A.4 Attribution and implementation checks

| Component | Origin or influence |
| --- | --- |
| Projection, compositing and Adam infrastructure | Course/assignment Warp framework |
| Flat surfels and consistency formulation | Adapted Gaussian Surfels ideas |
| Deferred normal/environment shading | Adapted 3DGS-DR ideas |
| HDR/angular model, staged experiments and diagnostics | Project implementation |

Recorded development checks cover:

- **Gradients:** 22 HDR finite-difference cases per device on CPU and CUDA.
- **Rendering and persistence:** save/reload, 800px integration and display conversion before coverage composition.
- **Normal substitutions:** unchanged checkpoint hashes and non-normal buffers, with the previous lighting baselines reproduced.

These are recorded checks, not newly repeated training runs. Details are in the [test code](https://github.com/wenn5055/3DGSresearch/blob/59c553f/scripts/test_surfel_hdr.py) and dated experiment records. [Unit 9](https://weitsang.github.io/gaussian-splatting-notes/unit-9/#the-central-idea-differentiate-the-primitive-hand-write-the-blend) explains why derivatives are checked separately from forward rendering.

## Appendix B. Reproduce the results

For instructions on reproducing the results, see the [repository README](https://github.com/wenn5055/3DGSresearch/blob/main/README.md#reproduce-from-a-fresh-clone).

## Appendix C. Computational cost

*Table C1. Required model files, observed rendering rate, sum of two training-stage times, and largest stage Warp allocation. Decimal MB/GB.*

| Model | Checkpoint (MB) | Forward FPS | 10k + 5k training (s) | Warp allocation (GB) |
| --- | ---: | ---: | ---: | ---: |
| SH3 | 23.50 | 273.63 | 172.42 | 1.320 |
| HDR-1.0 | 8.54 | 287.61 | 172.92 | 1.253 |
| HDR-0.75 | 8.54 | 303.98 | 173.17 | 1.253 |

Runs used an NVIDIA RTX PRO 5000 Blackwell, Warp 1.18.0 and Python 3.12.10. Forward timing warms four views once, then measures ten repeated sequences with device synchronisation. It includes host tiling, rendering and MSE, excluding image download, file I/O and perceptual metrics. Rates depend on separately learned footprints and visibility.

Training times exclude the common initialiser (68.79 s), evaluation and startup. Jobs may overlap other work, so they are descriptive rather than a controlled speed comparison. Warp allocations exclude driver/context and external memory; they are not total VRAM. File sizes include required sidecars in this serialisation format.

The separate loss trial recorded 70.26 s for 5k L1/D-SSIM refinement, compared with 63.84 s for the MSE refinement. These are observations from different sessions, not a repeated timing benchmark.

## Appendix D. Supporting scenes and diagnostic comparisons

### D.1 Earlier Materials and Chrome scenes

In original Materials, protected deferred + SH3 improves validation PSNR from 27.3575 to 28.4721 dB but worsens LPIPS from 0.036361 to 0.038978. Chrome normal error improves from 25.88° to 15.71°, while thin reflected details remain missing. A ray diagnostic also found reflected rays hitting nearby objects, a limitation of a direction-only environment map.

The enlarged Chrome scene contains a recessed logo and base. Increasing position learning rates fivefold improves global image scores, but logo-region normal error changes only from 28.23° to 27.84°. The recess remains missing. These failures motivated simplifying the main scene, and are not pooled with the sphere results.

![Preserved logo failure](../reports/assets/2026-11-02-logo-position5x/logo-comparison.png)

*Figure D1. Enlarged Chrome, validation view 18. Columns: target, original refinement, fivefold position-rate refinement; rows: RGB and normals. Actual crops preserve the unrecovered logo geometry. This is a separate scene and protocol.*

### D.2 Surface normals and depth

The normal maps capture the broad spherical surface, but the angular-error map exposes local orientation errors that are less obvious in RGB-encoded normals. These comparisons use the primary HDR-1.0 model alongside SH3; the HDR-0.75 lighting diagnostic remains separate.

![Reference and learned sphere normals with angular errors](../presentation/assets/2026-11-12-qna/normal-comparison-errors.png)

*Figure D2. Chrome sphere, validation views 0 and 51, top to bottom. Columns: Blender mesh normal, SH3 shading normal, HDR-1.0 shading normal, HDR-1.0 angular error against the mesh. Actual saved float normals are encoded as RGB by mapping world-space XYZ from [−1, 1] to [0, 1]. Angular error uses a shared 0–5° scale, clipped. White pixels are outside the jointly valid reference foreground, where both models have coverage ≥ 0.5; white does not mean zero error. These are two selected validation views, not the five-view mean.*

The rendered depth similarly recovers the broad sphere shape while retaining local deviations from the mesh reference. Its shared display range allows comparison across both models and views. Agreement in broad depth alone does not establish accurate local surface normals, which depend on surface slopes.

![Reference and rendered sphere depth with absolute errors](../presentation/assets/2026-11-12-qna/depth-comparison-errors.png)

*Figure D3. Same scene and validation views 0 and 51. Columns: Blender mesh depth, SH3 rendered depth, HDR-1.0 rendered depth, HDR-1.0 absolute depth error against the mesh. The first three columns share one positive camera-z depth range across both views and all three surfaces, in scene units rather than metres. The error display spans 0–0.02 scene units, clipped; white pixels are excluded using the same joint foreground mask as Figure D2. Model depth is rendered from the current surfels; Blender depth is an independent evaluation reference and does not supervise training. These plots use saved float buffers, without new model rendering.*

### D.3 RGB differences

The difference maps locate errors on the reflected structures and around the rim. HDR-1.0 reduces many of the disagreements visible with SH3, while the retained ground-detail failure remains apparent in view 51. A pixelwise difference identifies where the image disagrees with the target, without separating lighting, geometry and material causes.

![RGB comparisons and absolute differences](../presentation/assets/2026-11-12-qna/rgb-differences.png)

*Figure D4. Chrome sphere, validation views 0 and 51. Columns: Blender target, SH3, HDR-1.0, SH3 absolute RGB difference, HDR-1.0 absolute RGB difference. Each error pixel is the mean absolute RGB difference after scaling saved 8-bit PNG channels to [0, 1]. Both error maps use the same 0–0.25 scale, clipped; black indicates small or zero error. These visual differences include the full frame and are distinct from the float-render inputs used for the reported PSNR/SSIM/LPIPS metrics.*

### D.4 Lighting and normal substitutions in additional views

The additional cameras show the same interaction investigated in Section 6.3: reference lighting retains distortion with learned normals, while reference normals retain blur with learned lighting. Replacing both recovers more recognisable structure, with residual contrast and brightness differences. All four combinations are retained, including the normal-only replacement that worsens mean LPIPS in the five-view diagnostic.

![Additional lighting and normal substitutions](../reports/assets/2026-11-13-appendix/lighting-normal-views-13-51.png)

*Figure D5. Chrome sphere, validation views 13 and 51. Columns: Blender target; learned lighting/learned normals; reference lighting/learned normals; learned lighting/reference normals; reference lighting/reference normals. Actual identical reflection crops from the same saved HDR-0.75 checkpoint used in Figure 6. Only lighting and shading-normal direction are substituted; geometry, coverage and learned material parameters are held fixed. These are additional views from the existing five-camera diagnostic, including the preserved view-51 failure, rather than new trained results.*

### D.5 Image-loss comparison across five views

The extended comparison supports the trade-off in Section 6.5. L1 + D-SSIM changes contrast and broad reflected structures across the selected cameras, but thin branches, fencing and ground texture remain blurred or distorted. The improved mean LPIPS therefore does not establish recovery of those details or more accurate geometry.

![Five-view refinement image-loss comparison](../reports/assets/2026-11-07-refinement-loss/reflection-crops.png)

*Figure D6. Chrome sphere, validation views 0, 13, 73, 88 and 51, top to bottom. Columns: Blender target, MSE refinement, L1 + D-SSIM refinement. Actual matched central reflection crops. Both refinements start from the same frozen HDR-1.0 parent and use 5k steps with the same geometry-loss coefficients. The view order reuses the fixed diagnostic cameras; the image-score means in Table 4 cover all 100 validation views, while these five views illustrate the remaining reflection errors.*
