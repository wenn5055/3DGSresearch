# Course notes audit: what matters for this project

Reviewed the scopes/summaries of Units 0-18 and the relevant lifecycle,
initialization, appearance, compaction and foreground sections in detail.
The supplied singular `/gaussian-splatting-note` URL returns 404; the working
[course site](https://weitsang.github.io/gaussian-splatting-notes/) uses `notes`.
HTML/text snapshots and SHA-256 records are in
`outputs/first-principles-2026-10-29/course-notes/`.

**Main correction:** the existing opacity/ADC cleanup was never a foreground
detector. Freezing that cleanup after 20k retained weak surfels; RGB-white
background floaters can remain opaque. Training alpha must supply independent
empty-space evidence. Do not treat a 200k population as proof of useful geometry.

## Relevant findings and changes

| Notes | Finding in this repository | Action |
| --- | --- | --- |
| [Unit 7](https://weitsang.github.io/gaussian-splatting-notes/unit-7/) | Early opacity can hide other geometry; housekeeping and recovery matter. | Keep pruning/recovery explicit; do not prune on the last training step. |
| [Unit 9, stages 6-7](https://weitsang.github.io/gaussian-splatting-notes/unit-9/) | ADC only scores high-gradient candidates; opacity applies globally. Neither classifies foreground. | Add training-mask contradictions over the entire active population; log removal and growth separately. |
| [Unit 11](https://weitsang.github.io/gaussian-splatting-notes/unit-11/) | Directional colour and sharp reflected environment content solve different parts of appearance. | Late base SH remains useful; evaluate chrome crops and compare against SH-only controls. |
| [Unit 12](https://weitsang.github.io/gaussian-splatting-notes/unit-12/) | Random points and tiny footprints can receive no useful gradient. Local cloning cannot repair every bad allocation. | Inspect initialization and footprint/visibility distributions before increasing capacity or adding losses. |
| [Unit 14](https://weitsang.github.io/gaussian-splatting-notes/unit-14/) | True visibility importance sums transmittance times alpha across rays; opacity misses hidden or rarely useful surfels. | Keep this separate from mask pruning. The current ADC score is a bounding-box heuristic, not that importance statistic. |
| [Unit 17](https://weitsang.github.io/gaussian-splatting-notes/unit-17/) | The dynamic example receives masks and initial point labels; it does not discover foreground automatically from RGB. | Use our available training PNG alpha directly. Do not claim automatic semantic segmentation. |

### Concrete pruning defects

- The edited SH3 checkpoint has 11,689 active surfels below opacity 0.005 and
  58,543 below 0.02. Its 20k refinement freeze disabled later cleanup.
- The original-scene 200k scout has 26,582 surfels whose full projected supports
  lie in background in at least three selected training views. Their median
  opacity is 0.5181: opacity alone cannot identify these as empty-space geometry.
- Densification can refill deleted slots immediately. Active count alone hides
  turnover. Logs now separate `pruned`, `grown`, and before/after counts.
- Repeated foreground projections in other views do not disprove a background
  contradiction: those foreground pixels may come from occluding objects.

Raw diagnostic: `outputs/first-principles-2026-10-29/foreground-pruning-dry-run.json`.
The [training-mask visual audit](../outputs/first-principles-2026-10-29/foreground-evidence/training-mask-evidence.png)
shows RGB, the original alpha, and projected candidate centres for four diverse
training views. Red centres can overlap foreground in another view because of
occlusion. The support-box rule uses all 16 evidence views; this frozen audit
does not claim to reconstruct historical persistence or perform deletion.
Reproduce with the session-local `foreground_evidence.py` script.

### Implemented training-mask pruning

`src/cp4282_gs/surfels/foreground_pruning.py` checks full 3-sigma projected
support boxes plus a two-pixel margin, using 16 diverse **training** cameras.
A box must contain no alpha above 0.01 in at least three cameras, on two
successive pruning checks. Partial foreground overlap, soft alpha, clipped
footprints and behind-camera projections do not provide background evidence.
New/recycled children start with cleared evidence.

The filter supports both Gaussian and surfel controls. It is independently
scheduled every 500 steps from step 2k and continues during refinement. Ordinary
ADC/opacity refinement pruning defaults to every 500 steps; an explicit zero
still disables that route for archived ablations. Updated active SH presets
enable mask pruning and keep late cleanup on. Saved historical run configs and
source snapshots retain their original policy.

This is conservative silhouette-based empty-space rejection. It does **not**
detect hidden redundant layers inside the foreground, recover concavities from
silhouettes, or use contribution scores to delete surfels. In a white
background, even total blending weight alone can be misleading: an opaque white
floater can have high weight and almost no RGB removal error. Foreground-aware
evidence and visibility/redundancy are distinct checks.

Verification: 102 CPU tests passed; six new CUDA pruning tests passed; the
50-test CUDA surfel suite passed. Tests cover opaque white foreground/background,
multi-view persistence, partial/soft foreground protection, off-frame/behind-camera
supports, duplicate camera rejection, training-only masks, child reuse, and
pruning during growth-free refinement. A full GPU runner smoke removed 22 then
6 background surfels, and later removed 43 during refinement with zero growth.
These counts verify operation, not final reconstruction quality.

### Unit 14 contribution measurement: now measured directly

Added `scripts/measure_surfel_contributions.py`, a read-only raster pass that
accumulates actual front-to-back `T * alpha` per splat across **all 100 training
cameras**. It separately accumulates weights on foreground/background using the
original soft alpha and checks their conservation against rendered coverage.
This is independent of the bounding-box ADC proxy and does not prune anything.

| Frozen 200k checkpoint | Exactly zero contribution | Bottom 40% share of total weight | At least 99% background weight |
| --- | ---: | ---: | ---: |
| Edited scene, late SH3 and mask loss | 5,434 | 1.2336% | 0 |
| Original-scene unmasked deferred-SH3 scout | 8,312 | 0.3742% | 1,751 |

In the edited model, 302 never-contributing surfels have opacity at least 0.005;
two have opacity at least 0.1. Conversely, some surfels below the opacity floor
still contribute. This directly verifies the course's distinction between
opacity and visibility. It also shows why a full 200k allocation is not evidence
of 200k useful surface elements.

Three CPU and three CUDA tests verify exact front-to-back weights, soft-mask
partition, early-terminated occlusion, and the counterexample of an opaque white
background splat with high total weight and zero foreground contribution.
Visibility is not a safe deletion-error bound for the nonlinear deferred shader;
importance-based compaction still requires measured image quality and recovery.

Weights and provenance:
`outputs/first-principles-2026-10-29/contributions-edited-unpruned/` and
`outputs/first-principles-2026-10-29/contributions-materials-unmasked/`.

## Remaining course differences and scope

- **Optimizer history:** Unit 7 recommends fresh child state. The current research
  trainer inherits parent trajectories while preserving unchanged live state.
  That is a deliberate experimental policy, not a demonstrated universally better
  choice. The pruning correction does not silently change it; stale recycled-slot
  history must never survive a new child.
- **Initialization:** a capacity cap is not an allocation policy. If pruning alone
  leaves poor coverage, a training-only geometric initializer or controlled
  reallocation is a better next question than another normal-loss weight sweep.
- **Importance compaction:** do not remove an arbitrary fraction just to force a
  smaller count. Measure contribution, preserve useful detail, allow recovery,
  and check held-out quality before accepting a compressed model.
- **Physically based shading:** the current model is still an environment blend
  with optional base SH. Roughness/GGX/Fresnel and HDR decomposition are not
  implemented by the SH change or by pruning.

Units 0-5 support the existing camera/projection/compositing/gradient checks.
Unit 6 supports the fixed split, per-view scores and crop protocol; Unit 10
explains loss-stage bookkeeping after disturbances. Units 13 and 15 concern
city-scale/codec work, and Units 16 and 18 image/video/spacetime representations;
they do not justify expanding this static reflection project. Unit 16's 2D image
Gaussians are also distinct from oriented surfels in a 3D scene.
