# Reflection research status

Updated **30 October, 16:04 Singapore**. Original Materials remains the primary benchmark.

| Completed model, 800px / 100 validation views | PSNR | SSIM | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Gaussian SH3 | 27.7246 | 0.944346 | 0.040411 |
| Random surfel SH3 | 27.8901 | 0.944841 | 0.042719 |
| Random deferred + SH3 | 27.7544 | 0.942944 | 0.046788 |
| Hull-initialized deferred + SH3 | 27.7503 | 0.942333 | 0.038428 |

**No deferred win established yet.** Hull initialization improves perceptual quality,
but Chrome normals remain about 42° wrong. Its matched SH-only control is running.

**Pruning works:** the validated 600k export preserves quality and reduces required
files from 239 to 142 MB. All 200 test views also preserve quality: **27.3129 dB,
0.939948 SSIM, 0.044983 LPIPS** versus 27.3063 / 0.939810 / 0.045083 for the parent.

**Model limit:** in four predefined views, 12–23% of Chrome pixels reflect other
geometry. A global environment pass cannot directly trace that local parallax.

**Next:** protect hull geometry with ray-plane depth/normal consistency and earlier
reflection training. The enlarged chrome pair runs first, followed by the protected
original Materials pair. Hard stop: **20:23 Singapore**.

[Detailed research log](experiments/2026-10-30-backbone-resume.md) ·
[Initializer/protected protocol](experiments/2026-10-30-initializer-controls.md) ·
[Pruning evidence](experiments/2026-10-30-frozen-pruning.md) ·
[Reflection-model scope](experiments/2026-10-30-reflection-model-scope.md) ·
[Normal-drift plot](../outputs/Oct30-1519-materials-geometry-trajectories/normal-error-trajectory.png)
