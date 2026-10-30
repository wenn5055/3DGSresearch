# Appearance warm-up followed by gentle geometry refinement

Specified 30 October, 17:27 Singapore, before continuation fits or test images.
This auxiliary study does not replace the original Materials benchmark.

The frozen 100k/10k HDR model reaches native validation PSNR **20.3089**, SSIM
**0.787162**, LPIPS **0.170792**. Its image is smoother but still visibly blurred.
The fixed-basis SH3 control is finishing. Normal errors around two degrees can
still blur fine reflected illumination; joint fitting from a flat environment
had previously worsened normals. Test whether learning lighting before allowing
small geometry updates improves this local optimization problem.

Continue three **fixed** arms for 5k additional steps from their 10k parents:

| Arm | Parent | Geometry |
| --- | --- | --- |
| HDR fixed | Frozen HDR 10k | Keep fixed. |
| HDR refine | Same frozen HDR 10k | Release means/scales/quaternions/opacity with smaller learning rates. |
| SH3 fixed | Frozen SH3 10k | Keep fixed; equal total appearance budget. |

Every arm restarts Adam, uses the same seed/camera sequence, 100k capacity, 800px,
pure MSE and no structural changes. Mask BCE weight 0.1 and normal/depth weight
0.01 (1k ramp) are configured in every arm; their geometry gradients are disabled
in the fixed arms. Shared geometry learning rates are position 1e-5 -> 1e-6 over
5k, scale 5e-4, rotation 1e-4, opacity 5e-3; they only affect the released arm.
Appearance learning rates remain unchanged. No reference normals/lighting enter
training. All arms have **15k cumulative steps**; native trainer time covers the
last stage, with parent history recorded separately.

The HDR fixed/refine contrast changes only geometry freezing and therefore tests
the value of releasing geometry after a common warm-up. Its physical regularizers
are part of that refinement treatment. SH3 fixed supplies an equal-budget
appearance comparison. This does not isolate warm-up from all possible joint
training schedules or establish performance for the original scene.

Run after the frozen pair's validation, possibly overlapping the still-running
original Materials controls. Skip timing in the diagnostic validation report.
Preserve every arm and report all 100 validation views, the three quality metrics,
Chrome normal/depth/coherence and images. If 10 minutes remain after all scheduled
final/frozen test reports, evaluate all three unchanged arms on 200 Chrome test
views and export a video. No test-based retuning or extra continuation after test.

## Completed validation; test cancelled for wrap-up

All three 15k-total arms completed. Fixed SH3: **19.0057 / 0.767141 / 0.176790**,
Chrome 13.5653 dB. Fixed HDR: **20.3180 / 0.787574 / 0.169609**, Chrome 14.8089.
Released HDR: **21.3361 / 0.821520 / 0.140111**, Chrome **15.7505 dB**.
Chrome normal error changes 7.6335 -> 7.2173 degrees and depth MAE
0.018205 -> 0.017007; normal coherence changes 0.996544 -> 0.994098.
This improves all quality metrics but leaves cloudy reflections. It does not
prove the warm-up schedule beats a fully matched joint-from-start schedule.
The user requested wrap-up, so the planned 200-view auxiliary test/video is unrun.
Native trainer time is stage-only; stage_history.json retains 15k cumulative cost.
Full validation: `outputs/Oct30-1739-chrome-staged-refinement-val`.
