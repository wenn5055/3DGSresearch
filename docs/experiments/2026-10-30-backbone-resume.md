# Reflection research resumed, 30 October 2026

Started 12:47 Singapore at the user's instruction to proceed. The previous
session ran approximately 2h24m before the requested stop. Conservatively retain
the original ten-hour **total active-work** limit: stop by **20:23 Singapore**
(12:23 UTC) today. New controllers enforce that deadline; yesterday's controllers
and interrupted evidence remain unchanged.

Status: **active — fresh child optimizer state restores early convergence.**
The first three cases are complete. Fresh children score 26.0921 dB versus
18.6229 with inherited histories, both with foreground rejection. Finish the
legacy-reset controls, then integrate/test the selected policy and validate a
full SH3 model before further reflection comparisons. The original Materials
benchmark remains the user's confirmed target.
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
