# Reflection research status

Updated **30 October, 17:23 Singapore**. Original Materials remains the primary benchmark.

| Completed model, 800px / 100 validation views | PSNR | SSIM | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Gaussian SH3 | 27.7246 | 0.944346 | 0.040411 |
| Random surfel SH3 | 27.8901 | 0.944841 | 0.042719 |
| Random deferred + SH3 | 27.7544 | 0.942944 | 0.046788 |
| Hull-initialized deferred + SH3 | 27.7503 | 0.942333 | 0.038428 |
| Hull-initialized surfel SH3 | 27.7921 | 0.942775 | 0.037801 |

**No original-scene deferred win yet.** Hull initialization improves perceptual quality,
but Chrome normals remain about 42° wrong. The matched SH-only control also wins.

**Pruning works:** the validated 600k export preserves quality and reduces required
files from 239 to 142 MB. All 200 test views also preserve quality: **27.3129 dB,
0.939948 SSIM, 0.044983 LPIPS** versus 27.3063 / 0.939810 / 0.045083 for the parent.

**Model limit:** in four predefined views, 12–23% of Chrome pixels reflect other
geometry. A global environment pass cannot directly trace that local parallax.

**Auxiliary Chrome:** deferred gains **0.47 dB on Chrome**, but global LPIPS is
worse (0.14569 versus 0.13021). Background/boundary error is 4.67x larger.

**Running:** protected original Materials pair and a small frozen-geometry
appearance diagnostic. An early-SH Gaussian reference and an HDR/angular method
with alpha-mask supervision are queued. Final choices use validation Chrome PSNR
before all-200-view test reports/videos. No full HDR quality result yet.
Hard stop: **20:23 Singapore**.

[Detailed research log](experiments/2026-10-30-backbone-resume.md) ·
[Initializer/protected protocol](experiments/2026-10-30-initializer-controls.md) ·
[Pruning evidence](experiments/2026-10-30-frozen-pruning.md) ·
[Reflection-model scope](experiments/2026-10-30-reflection-model-scope.md) ·
[HDR/mask protocol](experiments/2026-10-30-angular-response.md) ·
[Frozen-geometry diagnostic](experiments/2026-10-30-frozen-appearance.md) ·
[Normal-drift plot](../outputs/Oct30-1519-materials-geometry-trajectories/normal-error-trajectory.png)
