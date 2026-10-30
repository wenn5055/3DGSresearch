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
