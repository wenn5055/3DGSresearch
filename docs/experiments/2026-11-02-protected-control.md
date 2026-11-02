# Original Materials: complete the protected SH-only control

Started 2 November 2026 (Singapore), before inspecting any new results.

## Question and fixed protocol

Does deferred environment reflection improve the original Materials benchmark
relative to SH3 under the same protected geometry policy?

Run `config/materials_reflection_hull_regularized_surfel_sh3.yaml` fresh for
30,000 steps, 800x800, capacity 1M, 100k training-mask hull initial points and
seed 7. Match the completed protected deferred run at
`outputs/Oct30-1717-materials-hull-protected-deferred-sh3`: ray-plane depth,
normal-depth weight 0.01 from step 1 with a 2k ramp, foreground pruning, and SH3
starting at step 20,001. The planned differences are `surfel.mode` and output path.
Do not add HDR/angular shading or mask BCE to either arm of this comparison.

The interrupted 30 October SH-only run remains preserved and excluded. Start
fresh rather than treating its saved model as an exact optimizer resume.

Before launching, compare the resolved configurations, training-source hashes,
input hashes and installed package versions against the completed deferred run.
The initializer PLY must retain SHA256
`4f9acea4c85b93fcca19101ca4dffc35fd4de815c16263018b702cd51a736a64`.

## Evaluation and decision

After training and native validation, evaluate both frozen models sequentially
using the same current evaluator on all 100 validation views. Include float-render
PSNR, SSIM, LPIPS AlexNet, foreground and visible-Chrome metrics, normal/depth
diagnostics, model size, and paired forward timing. Use the existing evaluation-only
geometry references and material inspection. Illustration views remain 0, 13,
73 and 88, matching the previous protected-deferred report.

Inspect the full images, Chrome crops and normals alongside aggregate scores.
Report global quality and Chrome quality separately; a global improvement alone
does not establish sharper reflections. No test views are evaluated or used for
selection in this experiment. No parameter sweeps or extra datasets are queued.

## Execution

Preflight passed: resolved settings differ only in mode/output; all 203 input
hashes and installed package versions match the deferred run. Training sources
are unchanged. The two changed Python files are the comparison evaluator and
its unrelated HDR test file; both arms will use the same current evaluator.
The completed deferred checkpoint hashes also match their recorded provenance.

Session: `outputs/Nov02-1257-protected-control-session/`. Its `run_control.py`
launches the recorded fresh run, then paired validation, and retains exact
commands, process IDs, logs, preflight hashes and status in `state.json`.
It queued only this control and comparison. Both completed successfully.

The control started at 12:57 Singapore in
`outputs/Nov02-1257-materials-hull-protected-surfel-sh3/`. Its recorded runner
performs training followed by validation PSNR/SSIM/LPIPS. The controller then
writes the paired report to the session's `paired-validation/index.html` and
per-view/summary CSV files. Native output filenames containing `test` are runner
conventions; the explicitly selected manifest is `transforms_val.json`.

## Completed results

Finished 2 November at 13:28 Singapore. The fresh control completed all 30,000
steps in 1,708.67 seconds. Both frozen models were then evaluated on the same
100 validation views using the current evaluator. Renderer-source fingerprints
match their training snapshots. This is one seed, with no test-set evaluation.

| Metric | Protected SH3 | Protected deferred + SH3 |
| --- | ---: | ---: |
| PSNR (higher is better) | 27.3575 | 28.4721 |
| SSIM (higher is better) | 0.938769 | 0.946323 |
| LPIPS AlexNet (lower is better) | **0.036361** | 0.038978 |
| Foreground PSNR | 21.5681 | 22.9105 |
| Visible Chrome PSNR | 15.8249 | 16.1883 |
| Chrome normal error, degrees | 25.8763 | 15.7080 |
| Chrome depth MAE | 0.035091 | 0.030522 |

Adding the reflection branch improves overall PSNR by **1.1147 dB**, foreground
PSNR by **1.3424 dB**, and Chrome PSNR by **0.3634 dB**. Chrome normal error is
10.17 degrees lower. However, LPIPS is worse by 0.002617, and the predefined
Chrome crops still show blurred and distorted tree reflections. This supports
the reflection branch under this training policy, not a uniform perceptual win
or solved sharp reflections. Geometry is optimized in both arms, so its resulting
accuracy can differ; this is not a frozen-geometry appearance comparison.

Observed forward FPS was 68.19 for SH3 and 75.28 for deferred under the four-pose,
ten-repeat procedure. These are measurements of the separately learned models,
not isolated shader costs. Checkpoints total 238.63 versus 242.95 MB. Both training
processes recorded 2.985 GB Warp allocation peaks, excluding driver/context memory.
The older deferred training overlapped other work, so training times do not
establish a controlled speed comparison.

[Paired visual report](../../outputs/Nov02-1257-protected-control-session/paired-validation/index.html)
and [portable evidence](../reports/assets/2026-11-02/provenance.json) retain scores,
per-view metrics, geometry summaries, costs, crops, preflight and commands.

Next: use these as the completed protected control pair. Predeclare one bounded
staged HDR/angular refinement experiment and inspect Chrome detail on validation
before fixing the final test protocol. Do not reopen broad normal-weight tuning.
