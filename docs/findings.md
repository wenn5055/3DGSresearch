# Research findings

Record experiment findings in dated sections using Singapore dates (Asia/Singapore).
For each entry, state the question, settings, measured result, interpretation, and
evidence. Add later findings as new sections and retain earlier results for reference.

## 2026-10-28 — Materials: staged D-SSIM improves held-out structure

**Question:** Does adding D-SSIM improve the fixed-RGB trainer over MSE alone?

**Matched setup:** Materials; seed 7; 30,000 completed iterations; capacity 1,000,000;
1,000 randomly initialized splats; position learning-rate decay over 60,000 steps.
Both models were evaluated on the same 200 test views at 800 × 800, with a white
background and fixed-RGB rendering. Reported scores are arithmetic means of per-view
metrics; higher PSNR and SSIM are better.

| Training objective | Run | Test PSNR (dB) | Test SSIM |
| --- | --- | ---: | ---: |
| MSE throughout | `materials-Oct28-1833` | 24.7599 | 0.884506 |
| MSE warm-up, then 0.8 MSE + 0.2 D-SSIM | `materials-Oct28-1945-dssim` | 24.9365 | 0.907943 |
| Change from adding staged D-SSIM | | **+0.1766** | **+0.023437** |

The D-SSIM arm used reflected 11 × 11 windows. It switched to the blended, dense
objective at iteration **8,700**, when smoothed held-out validation MSE reached
**21.01 dB**, crossing the configured 21 dB gate.

- PSNR improved on **156/200** test views and decreased on 44.
- SSIM improved on **200/200** test views.
- Per-view PSNR changes ranged from −0.4084 to +0.7417 dB.

**Interpretation:** This matched pair supports a modest improvement in pixel accuracy
and a consistent improvement in measured structural similarity. The result tests the
staged loss as a whole; it does not isolate the effect of warm-up or establish the best
blend weight. This is one seed, and LPIPS was not measured.

**Protocol correction:** Earlier MSE summaries reported validation scores, whereas
the D-SSIM summary reported test scores. The MSE model was re-evaluated on the test
split for this comparison; it was not retrained. The baseline run's original summary
still contains its validation scores. The shared `test-renders` folder name does not
identify which split was evaluated; check the manifest in `run.json`.

**Evidence:**

- [MSE run settings](../outputs/materials-Oct28-1833/config.yaml)
  and [test metrics](../outputs/materials-Oct28-1833/eval-test/metrics.csv).
- [D-SSIM run settings](../outputs/materials-Oct28-1945-dssim/config.yaml),
  [training log](../outputs/materials-Oct28-1945-dssim/train.log),
  and [test metrics](../outputs/materials-Oct28-1945-dssim/test-metrics.csv).
- [Paired per-view comparison](../outputs/materials-Oct28-1833/eval-test/comparison-dssim.csv).

**Next check:** Repeat with additional matched seeds and inspect the views where
PSNR declined, keeping the training and evaluation settings fixed.
