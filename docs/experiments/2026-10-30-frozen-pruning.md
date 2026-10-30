# Frozen visibility pruning, 30 October 2026

The original Materials SH-only surfel model fits images reasonably but contains
substantial visibility redundancy. Measure actual front-to-back `T * alpha` over
all pixels of all 100 **training** cameras, then remove the lowest-weight rows
without changing the retained parameters or retraining. Validation/material
geometry never chooses individual points. This is separate from training-alpha
background rejection, which runs during training and can be followed by growth.

Parent: `outputs/materials-Oct30-1416-fresh-surfel-sh3`, 800px, 30k steps, active
SH3, 1M points. Importance: `outputs/Oct30-1454-materials-surfel-contributions/`.
There are **9,966 zero-contribution points**. The lowest 400k points carry only
**0.20157%** of the measured blending weight. This is a candidate for compression,
not a guarantee of preserved reflection quality; test the frozen exports.

## All 100 validation views

Common float-render metrics, arithmetic means; LPIPS AlexNet is lower better.
Chrome covers the 96 views with visible reference Chrome pixels.

| Export | Points | PSNR | SSIM | LPIPS | Chrome PSNR | Required files, decimal MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent | 1,000,000 | 27.8901 | 0.944841 | 0.042719 | 16.1483 | 238.9 |
| Remove 20% | 800,000 | 27.8909 | 0.944887 | 0.042710 | 16.1491 | 190.3 |
| Remove 40% | **600,000** | **27.8975** | 0.944875 | **0.042633** | **16.1568** | **142.3** |

The tiny mean gains should be read as preserved quality, not a meaningful image
quality breakthrough. Worst per-view 40%-pruning changes: PSNR -0.000969 dB,
LPIPS +0.0000933, SSIM -0.001629, Chrome PSNR -0.03193 dB. Chrome mean normal
error remains about 53.86 degrees: compression does not solve surface alignment.
Choose 40% for this frozen SH-only model from validation, then check it on the
held-out test views. Do not assume the same fraction is safe for deferred shading
or another scene; removing weights can also change interpolated normals.

[Successful report and crops](../../outputs/Oct30-1458-materials-frozen-pruning/Oct30-1502-validation/index.html),
[scores](../../outputs/Oct30-1458-materials-frozen-pruning/Oct30-1502-validation/summary.csv).
The first report attempt stopped while reading missing parent training logs,
after rendering the 20% model. Its output and failed protocol remain in
`Oct30-1459-validation/`; parent logs were supplied and the retry used a new
directory. No model tensor changed during that repair.

## Reusable export

`scripts/prune_frozen_splats.py` verifies the parent/checkpoint/input fingerprints,
training-manifest scope, finite weights and blending-weight conservation. New
measurements now record a weights-file SHA256; legacy measurements remain usable
with their available checkpoint/input checks and a new output weights digest.
Exports record whether the original weights digest was available.

Raw retained PLY rows stay unchanged, including ordering; matching SH and
reflection rows are compacted together. The global environment is preserved.
Derived artifacts have `status: derived`, a copied parent training log, original
training-source fingerprints and a separate pruning-source snapshot. They cannot
masquerade as a new completed training run. Comparison reports accept and identify
these derivatives, and label inherited training costs explicitly.

The production export `outputs/Oct30-1508-materials-pruned-0.4/` has **byte-identical
PLY, SH and material checkpoint files** to the fully validated 600k pilot. Three
CPU regression checks pass: global environment preservation when its height equals
the point count, rejection of stale checkpoints before writing, and rejection of
held-out/changed weights. Source/input data and the full parent remain intact.
The derived-artifact interface also passed a separate all-100-view validation,
reproducing every reported mean exactly:
[production export report](../../outputs/Oct30-1509-production-pruned-validation/index.html).

## Frozen-fraction test confirmation

The 40% fraction was chosen and committed from validation before evaluating all
200 original Materials test views. No fraction was retuned on test. Both models
reuse the original full training, with no post-pruning optimization.

| Test model | PSNR | SSIM | LPIPS | Chrome PSNR |
| --- | ---: | ---: | ---: | ---: |
| Parent, 1M | 27.3063 | 0.939810 | 0.045083 | 15.7626 |
| Compact, 600k | 27.3129 | 0.939948 | 0.044983 | 15.7706 |

Chrome is visible in all 200 test views. Worst per-view compact changes: PSNR
-0.000897 dB, LPIPS +0.0000626, SSIM -0.000958, Chrome PSNR -0.00706 dB. These
confirm preserved image/reflection quality for this frozen model, rather than a
substantial quality gain. [Full test report and figures](../../outputs/Oct30-1549-materials-frozen-pruning-test/index.html).

Timing was skipped while the deferred control trained. Required checkpoint size
falls about **40.4%**. No FPS speedup or reduced training peak/time is claimed yet;
the pruned artifact inherits the original 1M-point training cost. Measure rendering
on an otherwise idle GPU. Research controller/log state remains in
`outputs/first-principles-2026-10-30/`.
