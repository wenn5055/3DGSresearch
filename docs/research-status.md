# Reflection research status

**Updated 3 November 2026: focus on specular reflection reconstruction.
Logo-specific geometry investigation closed; plain-sphere capture and staged HDR completed.**
[Today's illustrated closeout](reports/2026-11-02-research-wrap-up.md) includes
the new dataset, both training stages and reflection crops.
[Scope decision and proposed sphere diagnostic](research-scope-2026-11-02.md).
The completed position-rate trial improves image quality; the logo remains unresolved.
[Position-rate experiment](experiments/2026-11-02-logo-position-rate.md) and
[original Materials protected control](experiments/2026-11-02-protected-control.md).
The previous session stopped on request on 30 October at 17:55; its completed
evidence is in the [illustrated wrap-up report](reports/2026-10-30-research-wrap-up.md).

- **Pruning confirmed on 200 test views:** 1M -> 600k points, preserved quality,
  238.88 -> 142.30 MB; paired idle-GPU forward FPS 57.17 -> 103.96.
- **Original Materials protected deferred:** 28.4721 PSNR / 0.946323 SSIM /
  0.038978 LPIPS, Chrome 16.1883 dB on all 100 validation views.
  Matched protected SH-only: 27.3575 PSNR / 0.938769 SSIM / 0.036361 LPIPS,
  Chrome 15.8249 dB. Deferred gains 1.1147 dB globally and 0.3634 dB on Chrome,
  but has worse LPIPS. Chrome normal error improves 25.88 -> 15.71 degrees;
  sharp tree reflections remain unresolved.
- **Auxiliary Chrome staged diagnostic:** 100k points / 15k cumulative steps,
  HDR refinement 21.3361 / 0.821520 / 0.140111 versus fixed SH3
  19.0057 / 0.767141 / 0.176790. Images still lack sharp tree reflections.
- **Position LR x5 follow-up:** same parent and 5k refinement budget, 21.7076 /
  0.833744 / 0.132206; Chrome 16.0461 dB, normal error 6.79 degrees. Chrome
  depth error worsens 5.8%, while logo-region normal error improves only
  28.23 -> 27.84 degrees. Preserve separately; this does not recover the logo.
- **New `justChromeBall` dataset and HDR workflow:** 400 captured 800px views
  (100/100/200), training-mask initialization, 10k frozen HDR then 5k refinement.
  All-100-view validation PSNR improves **22.4996 -> 24.1250**, SSIM
  **0.842281 -> 0.874920**. PSNR improves on 95 views, SSIM on all 100;
  [3 November LPIPS follow-up](experiments/2026-11-03-sphere-lpips.md) improves
  **0.119765 -> 0.093994** on all 100 views without retraining. Fine reflections
  remain blurred. No SH/LDR sphere controls or test evaluation yet.
- **Full 30k joint HDR/angular + alpha-mask presets remain untrained.**
  Early-SH Gaussian, Ship and new-method test/video runs
  were cancelled before starting.

Next: inspect projected footprint/normal/lighting diagnostics for the remaining
sphere blobs before selecting a size experiment; specify equal-budget SH3, LDR deferred and HDR/angular sphere controls on
the same fixed estimated geometry, with matched refinement studied separately.
Today's sphere continuation is complete; those controlled comparisons remain
planned. Preserve the logo failure
as a limitation and disclose prior inspection of historical Materials test scores.
Full logs/checkpoints remain
in `outputs/first-principles-2026-10-30/` and the timestamped run folders.
