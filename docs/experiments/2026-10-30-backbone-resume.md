# Reflection research resumed, 30 October 2026

Started 12:47 Singapore at the user's instruction to proceed. The previous
session ran approximately 2h24m before the requested stop. Conservatively retain
the original ten-hour **total active-work** limit: stop by **20:23 Singapore**
(12:23 UTC) today. New controllers enforce that deadline; yesterday's controllers
and interrupted evidence remain unchanged.

Status: **active — finish the backbone audit before choosing further training.**
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
