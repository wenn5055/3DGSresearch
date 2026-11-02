# Reflection research status

**Updated 2 November 2026, 13:28 Singapore: original Materials matched protected
SH-only control and paired validation completed.**
[Current protocol](experiments/2026-11-02-protected-control.md).
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
- **Implemented, smoke-tested, untrained:** combined HDR/angular + alpha-mask
  supervision presets. Early-SH Gaussian, Ship and new-method test/video runs
  were cancelled before starting.

Next: predeclare a bounded staged HDR/angular refinement experiment to improve
normal accuracy and reflection detail, then fix settings and evaluate untouched
test views. Full logs/checkpoints remain
in `outputs/first-principles-2026-10-30/` and the timestamped run folders.
