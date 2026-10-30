# Reflection research status

**Stopped on request, 30 October 17:55 Singapore.**
Start with the [illustrated wrap-up report](reports/2026-10-30-research-wrap-up.md).

- **Pruning confirmed on 200 test views:** 1M -> 600k points, preserved quality,
  238.88 -> 142.30 MB; paired idle-GPU forward FPS 57.17 -> 103.96.
- **Original Materials protected deferred:** 28.4721 PSNR / 0.946323 SSIM /
  0.038978 LPIPS, Chrome 16.1883 dB on all 100 validation views.
  Its matched protected SH-only control remains incomplete.
- **Auxiliary Chrome staged diagnostic:** 100k points / 15k cumulative steps,
  HDR refinement 21.3361 / 0.821520 / 0.140111 versus fixed SH3
  19.0057 / 0.767141 / 0.176790. Images still lack sharp tree reflections.
- **Implemented, smoke-tested, untrained:** combined HDR/angular + alpha-mask
  supervision presets. Early-SH Gaussian, Ship and new-method test/video runs
  were cancelled before starting.

Next: complete the original matched control, improve normal accuracy during
staged fitting, then evaluate untouched test views. Full logs/checkpoints remain
in `outputs/first-principles-2026-10-30/` and the timestamped run folders.
