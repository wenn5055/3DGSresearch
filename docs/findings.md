# Research findings

Record experiment findings in dated sections using Singapore dates (Asia/Singapore).
For each entry, state the question, settings, measured result, interpretation, and
evidence. Add later findings as new sections and retain earlier results for reference.

## 2026-10-28 — Materials: staged D-SSIM improves held-out structure

**Question:** Does adding D-SSIM improve the fixed-RGB trainer over MSE alone?

**Matched setup:** Materials; seed 7; 30,000 completed iterations; capacity 1,000,000;
1,000 randomly initialized splats; position learning-rate decay over 60,000 steps.
Both models were evaluated on the same 200 test views at 800 × 800, with a white
background and fixed-RGB rendering. Reported scores are arithmetic means of per-view
metrics; higher PSNR and SSIM are better.

| Training objective | Run | Test PSNR (dB) | Test SSIM |
| --- | --- | ---: | ---: |
| MSE throughout | `materials-Oct28-1833` | 24.7599 | 0.884506 |
| MSE warm-up, then 0.8 MSE + 0.2 D-SSIM | `materials-Oct28-1945-dssim` | 24.9365 | 0.907943 |
| Change from adding staged D-SSIM | | **+0.1766** | **+0.023437** |

The D-SSIM arm used reflected 11 × 11 windows. It switched to the blended, dense
objective at iteration **8,700**, when smoothed held-out validation MSE reached
**21.01 dB**, crossing the configured 21 dB gate.

- PSNR improved on **156/200** test views and decreased on 44.
- SSIM improved on **200/200** test views.
- Per-view PSNR changes ranged from −0.4084 to +0.7417 dB.

**Interpretation:** This matched pair supports a modest improvement in pixel accuracy
and a consistent improvement in measured structural similarity. The result tests the
staged loss as a whole; it does not isolate the effect of warm-up or establish the best
blend weight. This is one seed, and LPIPS was not measured.

**Protocol correction:** Earlier MSE summaries reported validation scores, whereas
the D-SSIM summary reported test scores. The MSE model was re-evaluated on the test
split for this comparison; it was not retrained. The baseline run's original summary
still contains its validation scores. The shared `test-renders` folder name does not
identify which split was evaluated; check the manifest in `run.json`.

**Evidence:**

- [MSE run settings](../outputs/materials-Oct28-1833/config.yaml)
  and [test metrics](../outputs/materials-Oct28-1833/eval-test/metrics.csv).
- [D-SSIM run settings](../outputs/materials-Oct28-1945-dssim/config.yaml),
  [training log](../outputs/materials-Oct28-1945-dssim/train.log),
  and [test metrics](../outputs/materials-Oct28-1945-dssim/test-metrics.csv).
- [Paired per-view comparison](../outputs/materials-Oct28-1833/eval-test/comparison-dssim.csv).

**Next check:** Repeat with additional matched seeds and inspect the views where
PSNR declined, keeping the training and evaluation settings fixed.

## 2026-10-28 — Consolidated baseline, SH, and evaluation findings

**Implemented today:** staged MSE/D-SSIM training; configurable SH degrees 0–3 with
validation-MSE band gates, appearance-to-position gradients, density-control state
inheritance, and checkpoint sidecars; Singapore-time folder names and name suffixes;
validation-only learning-rate sweep with matched-baseline reuse and source checks.
The baseline implementation passed 28 CPU/CUDA tests; the sweep adds six orchestration
tests. Short GPU smoke runs verified D-SSIM and all three SH bands, checkpoint loading,
and evaluation. Smoke metrics are integration checks, not quality benchmarks.

### Matched Materials test comparison

All rows use seed 7, 30,000 completed steps, capacity 1,000,000, 1,000 random initial
splats, 800 × 800 images, white background, and the same 200 test cameras. Position
learning-rate decay spans 60,000 steps; densification fraction is 0.2.

| Variant | Run | Mean test PSNR | Mean test SSIM |
| --- | --- | ---: | ---: |
| Fixed RGB, MSE | `materials-Oct28-1833` (re-evaluated on test) | 24.7599 | 0.884506 |
| Fixed RGB, staged D-SSIM | `materials-Oct28-1945-dssim` | 24.9365 | 0.907943 |
| Degree-2 SH, staged D-SSIM | `materials-Oct28-2157-sh2` | 25.6693 | 0.916397 |
| Degree-3 SH, staged D-SSIM | `materials-Oct28-2157-sh3` | 26.0690 | 0.922090 |

Degree 3 improves the mean over degree 2 by **0.3997 dB / 0.005693 SSIM**. Both
improve the scene means over fixed-RGB D-SSIM. This is a single-seed result and does
not establish superiority on every view or isolate an optimal appearance schedule.
SH gates use held-out MSE after D-SSIM activates; final logs confirm actual degrees
2 and 3, rather than only configured maxima. Keep each PLY with its `.sh.npz` file.

Evidence: [SH2 summary](../outputs/materials-Oct28-2157-sh2/summary.md),
[SH3 summary](../outputs/materials-Oct28-2157-sh3/summary.md), and their
[SH2](../outputs/materials-Oct28-2157-sh2/test-metrics.csv) /
[SH3](../outputs/materials-Oct28-2157-sh3/test-metrics.csv) per-view test metrics.

### What the earlier trials establish

The [consolidated experiment CSV](experiment-results-2026-10-28.csv) preserves all
completed, non-smoke Materials and Lego runs available at this update, including earlier
sessions. Split, resolution, configured/completed steps, capacity and learning-rate
schedule are recorded alongside scores; these mixed-condition rows are not a ranking.

- At 800 × 800 and 10k steps, increasing capacity from 100k to 200k/500k did not
  improve validation PSNR in the recorded runs (21.819 / 21.788 / 21.477 dB).
- A 1M-capacity, 60k-step run with position decay ending at 10k reached 21.922 dB
  on validation. The 30k-step run with decay spanning 60k reached 25.298 dB.
  These are different configurations/budgets, so this is evidence to inspect the
  schedule, not an isolated causal estimate of the learning-rate effect.
- The run using the inherited 2k decay schedule stopped at 45k and reached 19.497
  validation dB. Requested iterations alone do not specify the optimization schedule.
- Resolution changes also change metric difficulty. Do not compare the 256/512/800
  results as though they were the same evaluation protocol.
- Validation contains 100 views (`r_0`–`r_99`); test contains 200 (`r_0`–`r_199`).
  Historical `test-renders` directories sometimes hold validation renders. The manifest
  in `run.json`, not the directory label, identifies the split.

### Validation tuning and remaining measurements

The SH3 learning-rate sweep compares 0.0000625 / 0.000125 / 0.00025 on the full
validation split. It completed after midnight on 29 October; results are recorded
below and in the [sweep summary](../outputs/materials-Oct28-2246-sh3-lr-sweep/summary.md).
The sweep ranks validation PSNR, uses SSIM for exact ties, and requires the full
budget and actual SH3 activation. Test scores are excluded from selection.

LPIPS, repeated-seed variability, fixed reflection crops, rendering FPS, and peak
VRAM remain unmeasured for this comparison. Recorded training wall times came from
sessions with different concurrent workloads and are not controlled speed benchmarks.
The next implementation comparisons are fixed-RGB flattened surfels, then surfels
with deferred environment reflection; neither has a measured quality result yet.

### Lego sanity checks

The recorded 800×800, 100k-capacity, fixed-RGB Lego runs used seed 7, 1,000 initial
splats, MSE, and the same 200 test cameras. The 2k-step run scored **24.7731 dB /
0.845046 SSIM**; the 10k-step run scored **27.3365 dB / 0.902249 SSIM**. Both used
position decay over 2k steps. These are useful pipeline/budget checks, not evidence
that the same schedule is suitable for Materials.

Evidence: [2k summary](../outputs/lego-20261028T043116Z-8a700f36/summary.md) and
[10k summary](../outputs/lego-20261028T043530Z-5299e5c7/summary.md).

## 2026-10-29 — Separate project passes and initialization audit

**Completed SH3 validation sweep:** every candidate completed 30k steps and activated
degree 3. Selection used validation PSNR, with SSIM breaking exact ties:

| SH coefficient learning rate | Mean validation PSNR | Mean validation SSIM |
| ---: | ---: | ---: |
| 0.0000625 | 27.1189 | **0.936387** |
| 0.000125 | 26.7261 | 0.928987 |
| **0.00025** | **27.3376** | 0.933825 |

The selected rate is **0.00025**; the half rate had higher SSIM. This is a single-seed
validation choice, not a test-set improvement claim. The
[selected SH config](../outputs/materials-Oct28-2246-sh3-lr-sweep/best-config.yaml)
is saved. The fixed-RGB project presets do not use SH and are unchanged by this choice.

Implemented three independently runnable presets: a Gaussian RGB control,
rank-two surfels with view-independent trainable RGB, and surfels with deferred
environment reflection after a 10k-step RGB geometry warm-up. See
[project-passes.md](project-passes.md) for commands, equations, checkpoints and scope.
All default to full validation evaluation; no full-budget project quality claim is
available yet. The deferred model currently learns an environment texture and
reflection strength, without GGX roughness/Fresnel or a normal-depth regularizer.

**Initialization correction:** the historical trainer appended held-out images before
computing its initial mean colour, so validation influenced that one initialization
statistic. It did not supply training gradients, and test images were not used.
The historical matched comparisons share this policy, but they are not a strictly
training-only initialization protocol. All new project passes use training-only
statistics. The fresh Gaussian control is therefore required for the geometry
comparison; do not attribute its difference from archived runs solely to surfels.
The historical trainer remained unchanged throughout the SH sweep.

**Verification:** 46 CPU tests passed, including 12 new surfel tests; those 12 also
passed on CUDA. Finite differences cover geometry, overlapping alpha compositing,
normal-to-shader gradients, reflection strength, environment lookup and D-SSIM.
Checkpoint/evaluator round trips, tangent-plane splitting and training-only
initialization are covered. The runner's 11 tests also passed after the final
validation-default/provenance changes.

Three GPU integration runs each completed six training steps at 32×32, including
densification from 64 to 109 primitives, staged-loss transitions and evaluation on
all 100 validation views. They establish execution and serialization, not quality:

- [Gaussian control smoke](../outputs/materials-Oct28-2353-gaussian_control-smoke/summary.md).
- [Surfel RGB smoke](../outputs/materials-Oct28-2353-surfel_rgb-smoke/summary.md).
- [Deferred surfel smoke](../outputs/materials-Oct28-2353-surfel_deferred-smoke/summary.md).

Next measurements: run the three 30k-step presets separately, compare Gaussian
control versus surfel RGB on validation, then surfel RGB versus deferred reflection.
Inspect coverage, depth, normals and matched highlight crops before selecting
settings; reserve final test evaluation for frozen configurations.

### RGB surfel inspection at 24k steps

- Run: `materials-Oct29-1234-surfel-rgb`; still training when inspected. RGB pass,
  no deferred reflection. Preserve this first run as the unregularized reference.
- The colour snapshot reconstructs recognizable objects/highlights, but normals
  are mottled and depth/normal images expose many background floaters. A good RGB
  fit alone does not establish a coherent surface suitable for specular shading.
- Diagnostic caveat: normal/depth previews show coverage above 1e-6, so weak
  floaters appear more prominent than in the colour render. They are not all weak:
  in the top 100 rows, 38.4% of pixels have coverage >0.01 and the maximum is 0.982,
  while the saved RGB pixels are entirely white. Background-coloured primitives
  can therefore be concealed in the RGB fit. This is one snapshot, not a scene-wide
  geometry-error measurement.
- Validation MSE EMA decreases from 0.006252 at 10k to 0.002802 at 24k. This is the
  training-time held-out subset/EMA, not mean full-validation per-view PSNR.
- The completed Gaussian control (`materials-Oct29-1144-control`) scored 25.7263 dB /
  0.920230 SSIM on full validation. Wait for the surfel run's full validation before
  comparing image quality. Surface regularization remains a separate next experiment.

Evidence: [24k snapshot files](../outputs/materials-Oct29-1234-surfel-rgb/model_snapshots/),
[training log](../outputs/materials-Oct29-1234-surfel-rgb/train.log), and
[Gaussian control summary](../outputs/materials-Oct29-1144-control/summary.md).

## 2026-10-29 — Interim report: three project runs before geometry losses

**Finding:** Deferred surfel reflection improves validation image reconstruction
over both RGB variants in this first matched-seed comparison. Flattening alone
slightly reduces image scores. Geometry remains noisy, so improved rendering does
not yet establish accurate surface normals.

### Setup and what changed

- Materials: 100 training views; final evaluation on the same **100 validation
  views**, at 800×800 with white background. These three runs did not evaluate test.
- Each completed **30,000 steps**, seed 7, 1,000 random initial primitives,
  training-only colour initialization, 1M capacity and 1M final active primitives.
  Adam, density fraction 0.2, position decay over 60k steps; early stopping off.
- All use **SH0** and an MSE warm-up followed by **0.8 MSE + 0.2 D-SSIM** after
  the 21 dB held-out-MSE gate. The gate monitors the same eight validation cameras;
  reported final scores below use all 100, not the training-time EMA.
- **Control → RGB surfels:** replace volumetric Gaussian covariance with exact
  zero-thickness local-z covariance and tangent-plane splitting; retain trainable,
  view-independent RGB. Surfel diagnostics are enabled, unlike the control.
- **RGB surfels → deferred surfels:** independently train from scratch with a
  scheduled RGB warm-up, activating learned reflection strength and a 128×64
  environment map at step 10,000. Base RGB remains trainable. This adds appearance
  parameters; the total parameter counts are not matched.
- Saved root/shared/surfel-module Python hashes match across all three runs.
  **Neither foreground-mask loss nor normal–depth consistency loss was used.**

### Full-validation results

Scores are arithmetic means of per-view metrics; higher is better. Recomputed
from the CSVs and checked against the complete validation manifest.

| Variant / run evidence | Mean PSNR (dB) | Mean SSIM |
| --- | ---: | ---: |
| [Gaussian RGB control — 1144](../outputs/materials-Oct29-1144-control/summary.md) · [CSV](../outputs/materials-Oct29-1144-control/test-metrics.csv) | 25.7263 | 0.920230 |
| [RGB surfels — 1234](../outputs/materials-Oct29-1234-surfel-rgb/summary.md) · [CSV](../outputs/materials-Oct29-1234-surfel-rgb/test-metrics.csv) | 25.5759 | 0.918148 |
| [Deferred surfels — 1305](../outputs/materials-Oct29-1305-surfel-deferred/summary.md) · [CSV](../outputs/materials-Oct29-1305-surfel-deferred/test-metrics.csv) | **26.7132** | **0.927629** |

- **RGB surfels vs control:** −0.1505 dB / −0.002082 SSIM; improved PSNR on 34/100
  views and SSIM on 26/100. Flattening alone does not improve these image metrics.
- **Deferred vs RGB surfels:** **+1.1374 dB / +0.009481 SSIM**; improved PSNR on
  **99/100** views and SSIM on **86/100**.
- **Deferred vs control:** **+0.9869 dB / +0.007399 SSIM**; improved PSNR on 93/100
  views and SSIM on 77/100.
- Actual D-SSIM activation: control step 8,300; RGB surfels 8,900; deferred surfels
  8,800. All finished in D-SSIM; the deferred run finished with reflections active.
  The threshold rule is shared, but the resulting optimization schedules differ.

### Geometry observations

- Final RGB-surfel normals remain mottled, with many background floaters. The
  deferred normal preview shows fewer surrounding floaters, but object normals
  remain noisy. A smooth, accurate surface is not established by these previews.
- A reproducible **single-view background diagnostic** supports that visual change:
  use final `model.buffers.npz` for the preserved snapshot camera, validation frame
  `r_0`; background = ground-truth alpha exactly zero after 800×800 Lanczos resizing
  (431,674 pixels). RGB surfels have mean background coverage **0.04561**, with
  **47.61%** of background pixels above coverage 0.01. Deferred surfels reduce these
  to **0.01888** and **16.63%**. This measures projected background contamination
  in one view, not full-scene geometric accuracy. Alpha is used here for diagnosis
  only, not training supervision.
- Normal previews normalize the accumulated normal wherever coverage exceeds 1e-6,
  making weak floaters conspicuous. White RGB primitives can also conceal substantial
  opacity against white backgrounds, as recorded in the earlier 24k inspection.

Visual evidence: final [RGB normals](../outputs/materials-Oct29-1234-surfel-rgb/model.normals.png),
[deferred normals](../outputs/materials-Oct29-1305-surfel-deferred/model.normals.png),
[deferred colour](../outputs/materials-Oct29-1305-surfel-deferred/model.png), and
the [RGB](../outputs/materials-Oct29-1234-surfel-rgb/model.buffers.npz) /
[deferred](../outputs/materials-Oct29-1305-surfel-deferred/model.buffers.npz) float buffers.

### Interpretation and next iteration

- The deferred extension is promising for held-out appearance; attributing the gain
  specifically to correct normals requires further evidence. The model blends RGB
  with learned environment reflection; it is not a full GGX/roughness/Fresnel model.
- This is one scene and one seed. No ground-truth normal/depth error, repeated-seed
  uncertainty, LPIPS or controlled FPS/VRAM comparison was measured. These validation
  scores should not be ranked against the earlier SH **test** scores.
- Preserve these three runs as the **pre-regularization reference**. Next compare
  RGB surfels + foreground-mask loss, then + normal–depth consistency, followed by
  the corresponding deferred variant. Keep seed, capacity, budget and evaluation
  fixed; assess foreground coverage and geometry diagnostics alongside image scores.

## 2026-10-29 — Geometry-loss implementation ready for the next comparison

- Added separate **foreground-mask BCE** and **normal–depth consistency** terms.
  The original three presets keep both weights at zero; the preceding interim
  report remains the pre-regularization reference.
- New presets: [mask-only RGB](../config/materials_surfel_rgb_mask.yaml),
  [RGB with both losses](../config/materials_surfel_rgb_geometry.yaml), and
  [deferred with both losses](../config/materials_surfel_deferred_geometry.yaml).
  Each is a fresh 30k-step run with the previous seed/capacity/RGB objective.
- Mask weight starts at **0.1**. Normal-depth weight ramps from zero at 2k to
  **0.01 at 4k**. These are initial hypotheses; select weights on validation.
- Alpha supervision reads original **training** PNG alpha only. Consistency uses
  reliable foreground five-pixel neighbourhoods, excludes low coverage/depth jumps,
  and differentiates through both normalized normals and back-projected depth.
  Current composited centre-depth is an approximation, not ray/plane depth.
- Training logs separate image/mask/normal terms and valid normal-pixel counts.
  Held-out image-loss gates and final image metrics retain their previous meaning.
  Equations, thresholds, schedule and commands: [project passes](project-passes.md).
- Verification: **57 CPU tests** passed; **23 surfel tests also passed on CUDA**.
  New checks include finite differences through G-buffers and splat parameters,
  perspective-plane/camera-coordinate consistency, foreground/boundary/depth-jump
  guards, zero-coverage stability and training-only mask loading. Behavioural tests
  confirm removal pressure on RGB-invisible white floaters and orientation gradients
  without colour error.
- Three 8-step/64px GPU integration runs completed training and evaluation on all
  100 validation views: [mask only](../outputs/materials-Oct29-1347-rgb-mask-smoke-2/summary.md),
  [both RGB losses](../outputs/materials-Oct29-1348-rgb-geometry-smoke/summary.md), and
  [both deferred losses](../outputs/materials-Oct29-1348-deferred-geometry-smoke/summary.md).
  These are execution checks; the sparse random scene had no valid normal pixels
  at the last logged step. Separate gradient tests exercise active normal losses.
- A GPU probe of the trained RGB checkpoint on **training frame 0 at 128px** had
  **4,089 valid normal pixels**, raw normal loss 0.22745 and finite gradients.
  No optimizer update was performed; this verifies that the constraints can act on
  the existing geometry, not that regularized training improves it.
  [Probe evidence](../outputs/project-smoke-input-geometry-4001cc4d/real-checkpoint-probe.txt).
- Full-budget geometry-loss results are pending. Compare normal/depth diagnostics,
  background coverage and held-out RGB scores before claiming improvement.

**Environment-map note:** the first deferred run's learned map varies with direction
(RGB spatial standard deviations 0.0667/0.0726/0.0902), but median primitive reflection
strength is only 0.1009. The map starts from mean training colour and is not supervised
with a garden panorama. Base RGB, uncertain normals and reflection strengths can
explain the images together; a recognizable recovered garden is not guaranteed.
Lighting/material separation remains a later investigation.

## 2026-10-29 — Training audit fixes and geometry experiment design

**Question:** Why do noisy normals persist despite improved deferred RGB results?

The [pre-fix code audit](training-code-audit-2026-10-29.md) reproduced blocked RGB
endpoint recovery and global Adam resets during density events, including events
that add no children. It also identified continued final-step growth, footprint-based
pruning limitations, and the lack of active normal-depth consistency in the recent
propagation presets. These findings do not establish one isolated cause of normal
error. The [design update log](project-plan.md#design-update-log) records the
implementation choices and next controlled experiment.

**Implemented:** inward RGB recovery; inherited parent Adam moments with unchanged
clocks and preserved existing-splat histories (including SH/reflection); no reset for
zero-child events; configurable quiet refinement with independent pruning; prevention
of final-step structure updates; optional hybrid propagation/normal-depth loss.
The current 30k propagation presets enter quiet refinement at 24,001, and the 10k
preset at 8,001. D-SSIM remains `(1-SSIM)/2` and is now identified in run summaries.

**Verification:** all **84 tests passed on CPU**, and the same suite passed with
surfel and SH tests using **CUDA**. New checks cover inward recovery in dense/sparse/
surfel rendering, preserved parent and inherited child state for clones/splits,
zero-child events, SH band clocks, hybrid finite differences, and quiet refinement
with/without pruning. Logs: [CPU](../outputs/implementation-cpu-tests.log),
[CUDA configuration](../outputs/implementation-cuda-tests.log).

A [64-step GPU integration run](../outputs/materials_red-rose-silver-Oct29-1846-optimizer-consistency-smoke-2/summary.md)
completed training, checkpoint reload and evaluation on all 100 validation views.
It activated reflection at step 8 and quiet refinement at 49; active count stayed
3,396 through step 64. Its random scene had zero valid normal-depth pixels at the
logged steps, so active consistency gradients are established by the separate
gradient tests, not this smoke run. An initial integration attempt failed because
Warp used a cache outside the workspace; repeating with the documented project
cache completed. These checks do not measure a reconstruction improvement.

**Next measurement:** matched fresh 400px/200k-capacity, 30k-step deferred runs with
normal-depth coefficients **0 versus 0.01**, propagation off in both, and final 10k
steps fixed in topology. Only coefficient and output path differ between the new
[control](../config/materials_red-rose-silver_deferred_consistency_control.yaml) and
[consistency](../config/materials_red-rose-silver_deferred_consistency.yaml) presets.
Compare foreground RGB and reference normal/depth accuracy; do not infer surface
accuracy solely from PSNR selection. Full-budget results remain pending.

## 2026-10-29 — Frozen consistency ablation: completed test comparison

Following the user's decision to close tuning, both 30k-step checkpoints were
evaluated on all **200 test views at 400x400**, without retraining. Saved configs
match except coefficient/output path, and rendering dependencies match the training
source snapshots. The test split was not used to select another weight.

| Normal-depth weight | Test PSNR | Test SSIM | Foreground PSNR |
| --- | ---: | ---: | ---: |
| **0** | **34.5007 dB** | **0.987244** | **23.4455 dB** |
| 0.01 | 34.1435 dB | 0.984849 | 23.2170 dB |

The regularized model trails the mean by 0.3572 dB / 0.002394 SSIM, while improving
PSNR on 75/200 views and foreground PSNR on 85/200. Foreground alpha >=0.95 occupies
6.9234% of each image on average; full-image scores include substantial background.
Both have low mean alpha-zero background coverage (0.00004056 / 0.00004174).

Four illustration views (r_0, r_72, r_123, r_146) were selected by pose diversity
before model rendering. Eight identical reference-alpha component crops expose
reflection detail. Normal maps in the 0.01 arm are visibly smoother across these
views, with remaining structured patches; both models retain reflection blur and
mottling against reference RGB. Shared depth scales and saved raw buffers support
the comparison. These observations establish a visible coherence tradeoff, not
measured normal accuracy; Blender reference geometry evaluation remains deferred.

Evidence and report-ready prose:
[final experiment record](experiments/2026-10-29-normal-consistency-final.md),
[HTML figures](../outputs/materials-red-rose-silver-final-comparison-2026-10-29/index.html),
[200-view metrics](../outputs/materials-red-rose-silver-final-comparison-2026-10-29/test-metrics.csv),
[crop bounds/scores](../outputs/materials-red-rose-silver-final-comparison-2026-10-29/crop-metrics.csv),
and [provenance](../outputs/materials-red-rose-silver-final-comparison-2026-10-29/comparison.json).
All results here concern the edited three-ball scene and seed 7; do not merge them
with the earlier full Materials-scene benchmarks.

## 2026-10-29 — Ray-plane depth follow-up

Implemented opt-in per-pixel ray-plane depth and its full geometry/compositing
gradients. The matched 30k three-ball run preserved the density policy, 200k cap
and fixed final 10k steps. Validation PSNR fell from **34.0690 to 33.1900 dB**;
foreground PSNR fell from **23.1015 to 22.3733 dB**. Selected normal maps retain
patches and the environment remains noisy. Keep centre depth as the default;
the depth change alone does not resolve shape ambiguity. All 44 surfel tests pass
on CPU/CUDA. See the [experiment and comparison](experiments/2026-10-29-ray-plane-depth.md).


## 2026-10-29 ? First-principles reflection audit; stopped at user request

Training-alpha foreground pruning and continued late cleanup are implemented and
verified. On the edited scene, the corrected deferred SH3 model retains 133,700
surfels versus 200,000, scoring 37.0101 versus 37.1227 validation dB. The matched
SH-only surfel control scores 36.7667 dB: the deferred gain is modest and material/
LPIPS results are mixed. Consistency plus SH3 reduces independent normal error
from 46.36 to 22.78 degrees, with a PSNR tradeoff to 35.3783 dB.

The first new full-resolution Gaussian SH3 baseline regressed to 23.09 dB versus
27.3372 for the earlier tuned baseline. Remaining full runs were held to audit
shared optimizer/pruning changes; this is not evidence of a deferred victory.
The first controlled backbone probe was interrupted at the user's request, with
a 5k snapshot retained. All research jobs/controllers are stopped. No completed
backbone ablation or trained visual-hull comparison exists yet.

Evidence and exact resume state: [investigation log](experiments/2026-10-29-first-principles.md)
and [course-notes audit](course-notes-audit-2026-10-29.md). The completed edited
comparison is in `outputs/first-principles-2026-10-29/edited-pruning-val/index.html`.


## 2026-10-30 ? Fresh child optimizer state restores backbone convergence

Five matched Gaussian RGB probes used original Materials at 800px, 1M capacity,
1k initial points, seed 7, 10k fixed dense-MSE steps, no SH/reflection. Training
implementation, non-treatment settings and input hashes were checked equal.

| Policy | Foreground pruning | Validation PSNR | SSIM |
| --- | --- | ---: | ---: |
| Inherit parent history | On | 18.6229 | 0.774763 |
| Inherit parent history | Off | 17.2023 | 0.699172 |
| Fresh child history | On | **26.0921** | **0.908585** |
| Global reset on births | On | 25.7094 | 0.905184 |
| Global reset on births | Off | 25.7015 | 0.900073 |

Fresh children improve over inheritance in all 100 validation views, by 7.4692 dB
on average. They beat global resets by 0.38265 dB, improving 92/100 views. The
previous parent-state transfer was tested for copying correctness but severely
hurt fitting here. This demonstrates why that earlier theoretical fix needed
quality validation. It does not yet establish a final deferred-rendering win.

YAML training workflows now default to `densification.child_optimizer_state:
fresh_children`, preserving existing-slot histories/clocks and resetting new-slot
geometry, quaternion, SH and reflection moments. Historical inheritance and
legacy-reset modes remain explicit options. The global environment optimizer is
preserved. Verification: 117 CPU tests, 55 CUDA surfel tests and four new CUDA
policy tests passed; Gaussian and deferred-SH3 GPU runner smokes completed reload
and evaluation.

Full 30k Gaussian SH3 validation confirms recovery: **27.7246 dB**, **0.944346
SSIM**, **0.040411 LPIPS AlexNet**, versus 23.0872 / 0.899692 / 0.083528 with
inherited histories. Inputs and non-policy settings match; all 100 views improve
in all three metrics. The earlier tuned SH3 reference scores 27.3372 / 0.933825 /
0.048299. Actual chrome quality improves only slightly against that strong
reference (16.2963 versus 16.1635 dB on 96 visible views). Blurred reflections
remain. Completed matched random-initialized controls give **27.8901 dB / 0.042719
LPIPS** for surfel SH-only versus **27.7544 / 0.046788** for deferred + SH. Chrome
scores are 16.1483 versus 16.1334 dB; normal errors remain 53.87 versus 52.46
degrees. The current unconstrained deferred pipeline does not establish a win.

[Protocol, complete contrasts and scope](experiments/2026-10-30-backbone-resume.md).
Figures and machine-readable evidence: `outputs/first-principles-2026-10-30/`.

## 2026-10-30: Visibility pruning removes 40% with preserved validation quality

The corrected original Materials SH-only surfel control scores 27.8901 dB,
0.944841 SSIM, 0.042719 LPIPS, but has 53.87-degree Chrome normal error. Actual
all-training-view blending weights expose redundant capacity: the lowest 40% of
points carry just 0.20157% of the weight. Frozen removal leaves **600k points**,
**27.8975 dB**, **0.944875 SSIM**, **0.042633 LPIPS**, Chrome **16.1568 dB**;
required model files fall **238.9 to 142.3 MB**. These tiny quality changes mean
preserved quality, not a reflection breakthrough. The compact model inherits
the full parent's training cost; timing is pending an idle GPU.

The reusable export preserves raw point/SH/material state, verifies provenance,
and labels derived checkpoints separately. Three CPU regression tests and a
separate all-100-view production-export validation pass. Parent files are intact.
[Results, limits and reproduction](experiments/2026-10-30-frozen-pruning.md).

The training-only hull's initially better normals also drift during RGB-only
fitting: predefined validation view 0 changes from 15.40 to 38.25 degrees on
Chrome by 5k steps, before deferred activation. Depth MAE improves meanwhile.
Keep those controls, then test ray-plane depth/normal consistency and reflection
from step 1. An eight-step GPU smoke with a 512x256 environment and SH3 reload
passes; full protected controls remain pending. The enlarged chrome pair was
revised before quality training. [Protocol and snapshot evidence](experiments/2026-10-30-initializer-controls.md).

## 2026-10-30 wrap-up: stronger global fit, reflection goal still incomplete

The original protected deferred run completes 30k at 800px/1M/seed7 with
**28.4721 PSNR / 0.946323 SSIM / 0.038978 LPIPS**, Chrome **16.1883 dB**.
Chrome normals improve 42.05 -> 15.71 degrees relative to unregularized hull
deferred, but Chrome PSNR remains below the Gaussian's 16.2963. The protected
SH-only control was interrupted early during shutdown; no matched result exists.

On the separate enlarged Chrome diagnostic, a shared 100k training-mask surface
and 15k total steps give HDR with gentle post-lighting geometry refinement
**21.3361 / 0.821520 / 0.140111**, versus fixed HDR
20.3180 / 0.787574 / 0.169609 and fixed SH3
19.0057 / 0.767141 / 0.176790. This is a positive controlled continuation result,
with visible blur remaining. New-method 200-view tests were cancelled on request.

The already test-validated 600k export retains quality and now has paired idle-GPU
forward timing: **57.17 -> 103.96 FPS** for full versus compact, same four test
poses/ten repeats. An overlapped shutdown timing sample was discarded and retained
with an invalidation note. Training-cost reduction is not claimed.

[Illustrated report and committed per-view evidence](reports/2026-10-30-research-wrap-up.md).
All session jobs are stopped; full checkpoints/logs remain local.
