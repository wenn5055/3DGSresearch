# Ray-plane depth: completed geometry experiment

Recorded 29 October 2026, Asia/Singapore. One scene, seed 7, 30,000 steps.

Ray-plane depth and its gradients pass the geometric and numerical checks, but
the trained model does not resolve the smooth-surface/cavity ambiguity. Selected
normal maps retain structured patches, and the environment map remains noisy.
Keep the feature opt-in; the default remains centre depth. This result does not
support replacing the previous model or further normal-weight tuning alone.

## Change and fixed settings

- New run: `materials_red-rose-silver-Oct29-2013-ray-plane-depth`.
- Historical comparison: `materials_red-rose-silver-Oct29-1906-deferred-tune-normal-1-0p01`.
- Only resolved training-config differences: `surfel.depth_mode` and output path.
  Both use normal-depth weight 0.01, ramped over steps 2k–4k, reflection from 3k,
  the same image/mask losses, data splits, seed, learning rates and density policy.
- Ray-plane mode replaces each contributing surfel's centre depth with its
  per-pixel plane-intersection depth before alpha blending. Backward propagation
  includes position, orientation and compositing-weight derivatives. Grazing
  intersections (absolute cosine <=0.001) and intersections at/before the near
  plane retain centre depth and its derivative. Footprints remain EWA projected
  surfels, sorting remains centre-based, and depth remains blended across layers.
- No new smoothing or distortion loss, propagation, or additional growth/pruning.
  Initial count 1,000; capacity 200,000; density interval 100 and fraction 0.2,
  matching the existing preset. Both logs stay within the 200k cap and show exactly
  200k active surfels throughout the final 10k steps, with topology fixed and
  opacity resets disabled. The count audit checks net active counts; it does not
  measure individual pre-refinement pruning events.
- New training completed in 818.61 seconds. Checkpoint sidecars save depth mode;
  evaluation restores it, with missing legacy fields interpreted as centre depth.

## Validation result

Both frozen checkpoints were rendered using the current evaluator on all **100
validation views**, at 400x400. The four diagnostic cameras are validation
0, 13, 21 and 25, chosen before completion from the pre-existing fixed validation
camera list. Foreground uses reference alpha >=0.95; foreground PSNR is the mean
of per-view PSNR values. No test views were used for this experiment.

| Depth mode | Validation PSNR | SSIM | Foreground PSNR |
| --- | ---: | ---: | ---: |
| Centre | 34.0690 dB | 0.985603 | 23.1015 dB |
| Ray-plane | 33.1900 dB | 0.983971 | 22.3733 dB |
| Change | -0.8790 dB | -0.001632 | -0.7282 dB |

Re-rendered baseline mean PSNR differs from its originally recorded validation
score by less than 0.000001 dB. Historical training and current evaluation source
snapshots are distinct; this is a frozen-checkpoint re-evaluation, not a claim that
their source fingerprints match. Ground-truth normal/depth accuracy was not
measured. The normal figures show no clear overall visual improvement on the
selected views. Neither learned environment map provides a clean recovered
panorama; both remain bounded display-RGB maps rather than HDR lighting estimates.

## Verification and artifacts

All 44 surfel tests passed on CPU and CUDA. Six new tests cover rendered tilted
planes in two camera frames, overlapping-depth finite differences for position,
rotation, scale and opacity, combined image/normal-loss gradients, degenerate
intersections, sidecar compatibility and unchanged density configuration. A
rendered tilted plane now has near-zero normal-depth loss, whereas centre depth
penalizes its correct tilted normal. All 100 validation renders/buffers were
checked for finite values.

- [Interactive comparison](../../outputs/ray-plane-depth-review-2026-10-29/index.html).
- [Normals](../../outputs/ray-plane-depth-review-2026-10-29/normals-comparison.png),
  [RGB](../../outputs/ray-plane-depth-review-2026-10-29/rgb-comparison.png),
  [depth](../../outputs/ray-plane-depth-review-2026-10-29/depth-comparison.png),
  [environment maps](../../outputs/ray-plane-depth-review-2026-10-29/environment-comparison.png).
- [Density audit](../../outputs/ray-plane-depth-review-2026-10-29/training-density.svg),
  [per-view metrics](../../outputs/ray-plane-depth-review-2026-10-29/validation-metrics.csv),
  [protocol/fingerprints](../../outputs/ray-plane-depth-review-2026-10-29/comparison.json).
- [Run summary](../../outputs/materials_red-rose-silver-Oct29-2013-ray-plane-depth/summary.md).
- Preset: `config/materials_red-rose-silver_deferred_ray_plane.yaml`.

The next controlled geometry experiment should examine concentration of
contributing depths along each ray. Plane-correct depth still permits multiple
competing layers and a self-consistent wrong shape; it supplies no direct evidence
that a region should be smooth. Repeated smoothing/weight sweeps alone are not
supported by this result. Do not combine a new density policy with that experiment.
