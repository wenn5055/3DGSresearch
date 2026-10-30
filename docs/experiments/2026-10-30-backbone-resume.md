# Reflection research resumed, 30 October 2026

Started 12:47 Singapore at the user's instruction to proceed. The previous
session ran approximately 2h24m before the requested stop. Conservatively retain
the original ten-hour **total active-work** limit: stop by **20:23 Singapore**
(12:23 UTC) today. New controllers enforce that deadline; yesterday's controllers
and interrupted evidence remain unchanged.

Status: **active — backbone recovered; unconstrained deferred control is negative.**
All five cases are complete. Fresh children score 26.0921 dB versus 18.6229
with inherited histories and 25.7094 with global resets, with foreground rejection
on. The implementation passes 117 CPU tests, 55 CUDA surfel tests and four new
CUDA policy tests. Gaussian and deferred-SH3 GPU runner smokes completed reload
and all-view evaluation. Full 30k Gaussian SH3 validation now reaches **27.7246 dB,
SSIM 0.944346, LPIPS AlexNet 0.040411**. Matched surfel/deferred controls are
complete; initialization and geometry-protected controls are running/queued.
The original Materials benchmark remains
the user's confirmed target.

User metric preference: include **LPIPS AlexNet** alongside PSNR and SSIM in all
subsequent validation/test quality reports. The full Gaussian job was launched
with PSNR/SSIM before this request; its completed frozen-model validation
comparison includes LPIPS over all 100 views. Future run commands and probe
defaults include all three metrics. LPIPS uses float images normalized to
[-1, 1], and lower is better; the comparison tool limits CPU inference to two
threads. Training objectives/curriculum are separate from this reporting choice.
Working tree was clean at restart (`a4d8fdf`). CUDA is available. All generated
compiler caches stay under `outputs/`; source/input/runtime fingerprints accompany
each experiment. Completed results and context are in the
[previous investigation log](2026-10-29-first-principles.md).

## First decision: why did the full-resolution baseline regress?

Yesterday's new Gaussian SH3 model scored 23.09 dB, versus the earlier tuned
27.3372 dB. Optimizer history, foreground rejection and appearance/loss schedules
differ, so the gap does not identify one cause. The interrupted 8,300-step probe
has no completed metric result. Its PLY snapshot lacks optimizer state; rerun
from the original seed rather than presenting a warm start as an exact resume.

Run five sequential fixed-MSE Gaussian RGB probes at 800px, capacity 1M, 1k
initial points, seed 7, 10k steps, identical density/view sampling settings:

| Child-state policy | Foreground pruning | Purpose |
| --- | --- | --- |
| Inherit parent moments | On | Replicate the current backbone |
| Inherit parent moments | Off | Isolate foreground rejection |
| Fresh child moments, keep shared clocks | On | Course-recommended child-state alternative |
| Legacy global optimizer resets | On | Measure the earlier optimization policy |
| Legacy global optimizer resets | Off | Complete the policy/pruning comparison |

These are early convergence probes, not final specular-quality comparisons.
The diagnostic runner is already tested and committed; production behavior
remains unchanged while evidence is collected. Compare all 100 validation views,
training curves and actual population/removal counts. Log every case, including
negative results. Avoid selecting a policy solely from a small training loss.

## Subsequent decisions

Use the audit to repair or explain the shared training regression. Then test the
training-silhouette initializer against a matching random-initialized control,
with late SH and independent geometry/material/perceptual evaluation. The
initializer's before-training normal error is promising, but no trained quality
advantage has been measured. Prioritize the original Materials chrome object and
the strong historical SH baseline when assessing actual reflections; Ship remains
a secondary check. Do not claim a deferred advantage from a weakened baseline.

Keep changes and measured findings committed at reviewable milestones. Do not
start an additional renderer/normal-weight sweep while these questions remain
unanswered. Session state, controller, logs and result summaries live under
`outputs/first-principles-2026-10-30/`.

## Measurements

First completed case: `materials-Oct30-1249-backbone-inherit-fg`, all 100
validation views after 10k fixed-MSE steps: **18.6229 dB**, SSIM **0.774763**,
mean MSE **0.0141149**. Final population: 1M; cumulative children: 1,445,817;
ordinary density/opacity removals: 13,760; foreground removals: **433,057**.
The population accounting balances. This confirms poor early convergence and
working foreground rejection; the cause awaits the controlled comparisons.

### First matched pair, 13:05 Singapore

| Inherited child-state policy, 10k MSE | Foreground on | Foreground off |
| --- | ---: | ---: |
| Mean per-view validation PSNR, dB | **18.6229** | 17.2023 |
| SSIM | **0.774763** | 0.699172 |
| Mean validation MSE | **0.0141149** | 0.0199634 |
| Final active points | 1,000,000 | 1,000,000 |
| Cumulative foreground removals | 433,057 | 0 |

Turning foreground rejection on improves PSNR by **1.4206 dB**, on **100/100**
views; per-view differences range from +0.5155 to +3.2804 dB. Non-treatment
settings and training implementation hashes match. The treatment also changes
subsequent allocation/ADC behavior; it is the complete foreground policy's
effect, not just deletion of a fixed frozen set. This is early convergence at
one seed, but it rules out simply blaming the mask correction for this early
regression. Both cases remain poor; optimizer-state comparisons are pending.
Unmasked run: `materials-Oct30-1256-backbone-inherit-unmasked`.

Session `summarize_backbone.py` writes per-case metrics, matched-view contrasts,
curves and source/run fingerprints to `backbone-results.json` and CSV files.
It checks common non-treatment settings and training implementation hashes,
and remains marked partial until all five cases finish. View-level win counts
are descriptive; this single-seed audit is not a statistical generality claim.

The inherited recovery-window log calls a foreground-pruning recovery an
"opacity reset". Actual opacity resets are disabled in these probes. Explicit
foreground/density event lines and their counts identify what happened; defer
the wording fix until the common-source audit has finished.

### Fresh-child result, 13:13 Singapore

`materials-Oct30-1304-backbone-fresh-fg` completes 10k fixed-MSE steps at
**26.0921 dB**, SSIM **0.908585**, mean MSE **0.00263596**. Relative to inherited
child state with the same foreground policy, PSNR improves **7.4692 dB**, on
**100/100** validation views; per-view differences range from +5.6749 to
+9.6392 dB. Source hashes and non-treatment settings match.

This directly identifies child-history transfer as a major cause of poor early
convergence in the audited backbone. The previous change to inherit parent
moments was verified for state copying but had not been validated for quality;
its theoretical rationale was insufficient. This result is RGB before SH or
reflection, not a claim that deferred shading now beats a strong SH model.
The complete earlier 4.25 dB SH3 gap also includes curriculum/source differences.

The treatment clears moments only in newly allocated child slots and keeps all
shared clocks and existing-slot histories, including reused split-parent slots.
It does not reset unchanged live points, add per-child clocks, change initial
parameters or inject reference geometry. Cumulative births: 1,062,252; ordinary
prunes: 51,665; foreground prunes: 11,587; final population: 1M. Better optimization
therefore also changes which descendants survive and how capacity is allocated.

Fixed-pose figures: session `backbone-figures/rgb-comparison.png` and
`chrome-crops.png`. These use saved PNGs only; quantitative metrics use float
renders. The legacy-reset cases are still running. Keep production training
unchanged until every audit case has captured the same source. A full 30k SH3
validation is required before treating the regression as resolved.

### Completed audit and implementation, approximately 13:35 Singapore

| Policy | Foreground | Validation PSNR, dB | SSIM |
| --- | --- | ---: | ---: |
| Inherit parent history | On | 18.6229 | 0.774763 |
| Inherit parent history | Off | 17.2023 | 0.699172 |
| Fresh child history | On | **26.0921** | **0.908585** |
| Global reset on child creation | On | 25.7094 | 0.905184 |
| Global reset on child creation | Off | 25.7015 | 0.900073 |

All training implementation hashes, non-treatment settings and input-file hashes
match. Fresh versus global reset with foreground on improves PSNR by 0.38265 dB
in 92/100 views. Under global resets, foreground rejection has essentially equal
mean PSNR (+0.00798 dB, 55/100 views) and higher SSIM. Its benefit depends on the
optimizer context; the 1.42 dB inheritance-case gain is not universal. No fresh/
unmasked case was run. The selected fresh/on combination is the best measured
case, not a claim to have exhaustively optimized every combination.

Implemented `densification.child_optimizer_state` with `fresh_children` as the
YAML-workflow default, plus explicit `inherit` and `legacy_reset` ablations.
Fresh births copy parameter values but zero new-slot moments for geometry,
quaternion, SH and reflection, preserving live-slot histories, shared clocks and
the global environment optimizer. Reused split-parent slots retain history,
matching the tested treatment. Zero-child events reset nothing. Legacy restarts
occur only after actual child creation; this does not reintroduce the earlier
zero-child global-reset bug. Direct constructor defaults remain `inherit` for
API compatibility; both CLI workflows explicitly pass the configured policy.
Diagnostic probe presets explicitly choose `inherit` as their base policy before
the scoped intervention; a guard prevents misleading diagnostic labels.

The source candidate was staged separately, and the full audit finished before
production files changed. New regression tests cover Gaussian/deferred training,
SH, clones/splits, live/free rows, clocks, zero-child events and identical initial
renders across all policies. An initial integration check caught an argument
scope error in buffer initialization, corrected before verification. New tests
pass on CPU/CUDA; the full CPU suite is **117 passed**, the CUDA surfel suite
**55 passed**, plus **four new CUDA policy tests**. Sandbox-only temporary-directory
permission errors in the first broad check were resolved by running the unchanged
tests with reviewed access; both failed and successful logs are retained.

Eight-step GPU smokes: `materials-Oct30-1332-fresh-policy-gaussian-smoke` and
`materials-Oct30-1332-fresh-policy-surfel-smoke`. Both trained, saved, reloaded and
evaluated all validation views; the latter unlocked/saved SH3. These tiny models
verify integration, not quality. Next: rerun the original 800px/1M/30k Gaussian
SH3 preset with the fresh workflow default and otherwise unchanged settings.
Retain the historical 27.3372 dB model as the strong reference. Measure chrome
and overall quality before resuming deferred-vs-SH and initialization comparisons.

### Full Gaussian SH3 recovery, approximately 14:27 Singapore

Common float-render evaluation over all 100 original Materials validation views
at 800px (arithmetic means of per-view scores):

| Gaussian model | PSNR, dB | SSIM | LPIPS AlexNet, lower better | Foreground PSNR | Chrome PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inherited child histories | 23.0872 | 0.899692 | 0.083528 | 18.6712 | 14.5111 |
| Historical tuned SH3 reference | 27.3372 | 0.933825 | 0.048299 | 21.4286 | 16.1635 |
| Fresh child histories | **27.7246** | **0.944346** | **0.040411** | **21.8805** | **16.2963** |

Fresh and inherited models have matching input hashes and non-policy settings:
800px, capacity/final population 1M, 30k steps, seed 7. The fresh policy improves
PSNR, SSIM, foreground PSNR and LPIPS on all 100 views. The historical tuned model
has different curriculum/SH learning-rate settings, so its comparison is a strong
reference rather than a policy-only ablation. Fresh wins PSNR on 68/100 views,
SSIM on 94, LPIPS on 84; mean LPIPS is about 16.3% lower. All three models have
active SH degree 3, verified from their saved appearance sidecars.

Chrome scores cover the 96 views with visible reference Chrome pixels. Fresh
beats inheritance on 94/96, but beats the historical reference on only 49/96,
with a small mean +0.1328 dB. Material crops still show blurred reflections.
The shared backbone regression is resolved for this benchmark; high-quality
reflection reconstruction remains the research target. This is not a deferred win.

Fresh training switches MSE to DSSIM at step 7,000 versus 21,800 with inheritance.
It takes 1,883.52 trainer seconds; the required model checkpoint files total
238,859,495 bytes. Its fresh-process Warp allocation peak is 2,516,299,571 bytes,
which excludes CUDA driver/context and non-Warp allocations. GPU render timings
were deliberately skipped while a surfel control was training; measure them on
an otherwise idle GPU. Native GPU and common CPU MSE reductions differ slightly
(27.724982 versus 27.724554 dB); use the common comparison for model contrasts.

Evidence: [HTML and crops](../../outputs/first-principles-2026-10-30/materials-gaussian-validation/index.html),
[scores](../../outputs/first-principles-2026-10-30/materials-gaussian-validation/summary.csv),
[paired contrasts](../../outputs/first-principles-2026-10-30/gaussian-validation-contrasts.json).
The sequential surfel/deferred controls both explicitly request PSNR, SSIM and
LPIPS AlexNet. No new chrome scene is used.

### Output naming preference, 14:34 Singapore

New experiment and sweep folders begin `MonDD-HHMM-` in Singapore time, e.g.
`Oct30-1434-materials-fresh-deferred-sh3`. Use the same prefix for new report/study
folders. Existing folders and active runs retain their paths. Both sweep launchers
and the shared run launcher are updated; all 11 existing runner checks pass.
Commit: `eff28fe`. This naming change does not alter training mathematics.

The user then granted full access and authorized the new chrome dataset,
parameter changes and longer training while away. Keep original Materials as
the primary benchmark. Prepared training-only initializer controls and an
enlarged-mirror diagnostic, with provenance checks and separate reporting:
[decision record](2026-10-30-initializer-controls.md). The hard stop is unchanged.

### Matched SH-only surfel control, 14:47 Singapore

`materials-Oct30-1416-fresh-surfel-sh3` completes 30k steps with active SH3 and
1M surfels. Common all-100-view float-render scores: **27.8901 dB**, **0.944841
SSIM**, **0.042719 LPIPS**, foreground **22.0874 dB**, visible Chrome **16.1483 dB**.
The fresh Gaussian SH3 reference is 27.7246 / 0.944346 / 0.040411, Chrome 16.2963.
Thus surfels alone gain a little overall PSNR but do not improve chrome or LPIPS.

Independent surfel geometry exposes the key issue: **42.20 degrees** mean mesh
normal error overall, **53.87 degrees on Chrome**. Normal coverage is 0.99948
overall and 0.99972 on Chrome, so this is not merely a few uncovered pixels.
Camera-z depth MAE: 0.029247 overall, 0.037360 on Chrome. Low image loss does not
establish reflection-ready surface alignment. These normals exclude shader bump.

[Interim common report](../../outputs/Oct30-1446-materials-backbone-val/index.html),
with timing skipped during concurrent deferred training. The deferred control
started in `Oct30-1444-materials-fresh-deferred-sh3`, confirming the new naming
rule. Idle-GPU reports and the predeclared initializer pairs are queued next.

While the deferred model trained, an independent visibility audit and frozen
compression experiment reduced the SH-only model from 1M to **600k surfels** with
preserved validation/Chrome/LPIPS quality and **40.4% smaller required files**.
The reusable exporter preserves matching geometry/SH/material rows and records
parent provenance. This is inference compression, not an automatic new training
pruner or a reflection-normal fix. [Evidence and reproduction](2026-10-30-frozen-pruning.md).

### Completed random-initialized controls, 15:16 Singapore

All 100 original Materials validation views at 800px, common float-image scores:

| Model, 1M points / SH3 | PSNR | SSIM | LPIPS | Chrome PSNR | Chrome normal error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 27.7246 | 0.944346 | **0.040411** | **16.2963** | Not defined |
| Surfel, SH only | **27.8901** | **0.944841** | 0.042719 | 16.1483 | 53.87 degrees |
| Surfel, deferred + SH | 27.7544 | 0.942944 | 0.046788 | 16.1334 | 52.46 degrees |

The unconstrained deferred pass does not win image, perceptual or Chrome scores.
Its overall normal error is 44.32 degrees versus 42.20 for SH-only. Normal
coverage exceeds 0.999 in both cases, so the issue is surface alignment rather
than missing pixels. The current random-initialized pipeline does not establish
the proposal's desired reflection benefit. Do not report the old weak inherited
baseline as evidence of success.

Serialized same-process forward benchmarks, after training jobs finish: Gaussian
**57.94 FPS**, surfel **59.81**, deferred **60.12**. Scope includes host tiling and
forward rendering/MSE, excludes image download/metrics; four predefined cameras,
10 timed repeats. These close figures do not establish a meaningful deferred
speed advantage. Fresh training-process Warp allocation peaks: 2.516 GB Gaussian,
2.985 GB for both surfel modes; this is not total GPU VRAM. Observed trainer times
1,883.52 / 1,686.54 / 1,675.38 seconds are not a controlled speedup comparison:
some diagnostic jobs overlapped during training. Required files: 238.9 / 238.9 /
242.5 MB. No inference benchmark overlapped another research training job.

[Complete report, crops, independent geometry and performance](../../outputs/Oct30-1513-materials-random-controls-val/index.html).
Training-only hull initialization is running next. Snapshot diagnostics already
show its normal alignment deteriorating during the RGB-only warm-up, motivating
the protected schedule documented in the [initializer decision record](2026-10-30-initializer-controls.md).
