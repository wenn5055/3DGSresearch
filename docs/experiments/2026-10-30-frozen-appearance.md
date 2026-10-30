# Frozen-geometry appearance diagnostic

Specified 30 October, 17:16 Singapore, before either diagnostic quality run.
Original Materials and its 30k controls remain the primary benchmark.

## Motivation and fixed design

In predefined Chrome validation view 0, the protected 30k LDR fit worsens normal
error from 5.99 to 10.46 degrees on all Chrome pixels, and from 2.76 to 4.50 in
a fixed 50-pixel-eroded interior. The training-mask sigma-3 initializer gives
5.19 / 1.78 degrees respectively. Coverage is effectively complete. This is one
diagnostic camera, not an all-view result or a new benchmark crop.
[Independent float-error figure](../../outputs/Oct30-1712-chrome-normal-errors/normal-errors.png).

Fit appearance on the same **fixed 100k sigma-3 training-mask basis**, at 800px,
10k iterations, seed 7 and pure MSE, with all structural schedules disabled:

| Variant | Appearance |
| --- | --- |
| Frozen surfel SH3 | RGB + SH bands at 1k/2k/3k. |
| Frozen HDR angular | RGB + scalar reflection blend, positive HDR environment 512x256 and 16 shared angular knots; no SH. |

Means, scales, quaternions and opacity stay bit-identical during optimization.
Their gradients are cleared and immutable parameter copies are restored after
the ordinary clamp. This also keeps rendered normal/depth/coverage buffers fixed.
Only appearance parameters learn. No reference geometry, material response or
environment texture enters training. Sigma 3 was chosen from the already recorded
initializer validation audit; it is still constructed from training masks only.

This isolates appearance capacity on an estimated surface. It does not show that
joint geometry reconstruction is solved, and it is not a matched comparison to
the 1M/30k runs. The no-SH HDR case also tests the proposal's smaller per-point
appearance idea, while retaining this shader's documented mirror-model limits.

## Checks, scheduling and evaluation

Nine HDR/freeze tests pass on CPU and CUDA, including exact fixed parameters and
normal/depth/coverage buffers, continued HDR/SH learning, structural-schedule
rejection, material adjoints and save/reload. A full workflow smoke precedes the
diagnostic pair. The existing renderer's default remains unfrozen.

Run the diagnostic pair sequentially **after the Chrome matched report finishes**.
They may overlap original Materials training to fit the wall-clock budget.
Record this interference explicitly: overlapping training times are observations,
not controlled cost comparisons. The diagnostic validation report must skip
timing, and all timed reports must wait until research GPU training ends.

Both presets are fixed before observing these fits. Evaluate both on all 100
validation views with PSNR/SSIM/LPIPS and geometry diagnostics. Then, if at least
10 minutes remain after the scheduled final reports and pruning timing, evaluate
both unchanged models on all 200 Chrome test views and export a comparison video.
No post-test retuning. Otherwise preserve the validation-only diagnostic and
mark the auxiliary test skipped at the deadline.

## 17:30 Singapore: complete 100-view validation

| Fixed 100k / 10k model | PSNR | SSIM | LPIPS | Chrome PSNR | Required files |
| --- | ---: | ---: | ---: | ---: | ---: |
| SH3 | 18.9031 | 0.766470 | 0.178251 | 13.4380 | 23,514,756 bytes |
| HDR angular, no SH | **20.3086** | **0.787162** | **0.170792** | **14.8053** | **8,546,509 bytes** |

HDR gains 1.4055 dB globally and 1.3673 dB on Chrome, with 63.65% smaller
required files. Rendered geometry diagnostics are identical across all views:
8.8174-degree overall normal error, 7.6335 on Chrome, coherence 0.995815 overall
/ 0.996544 Chrome. This is a positive appearance-capacity result on the auxiliary
scene, not successful joint reconstruction or a claim that SH is unnecessary
for rough/mixed materials. Direct image inspection still shows blurred trees.
The 10k budget is fixed for both appearance models; neither represents a fully
optimized 30k reference. Forward timing is intentionally skipped while primary
training runs.

[Complete report](../../outputs/Oct30-1728-chrome-frozen-appearance-val/index.html).
The [prespecified continuation study](2026-10-30-staged-refinement.md) now checks
whether a lighting warm-up permits gentler, useful geometry refinement.

## Wrap-up status

The short continuation study also completed validation; see its protocol and
the [illustrated report](../reports/2026-10-30-research-wrap-up.md). The user asked
to stop, so the planned frozen-model 200-view test/video was cancelled. No test
quality or speed claim is made for these new appearance models.
