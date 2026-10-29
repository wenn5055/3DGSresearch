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
