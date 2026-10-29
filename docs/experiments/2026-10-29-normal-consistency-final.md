# Frozen normal-consistency ablation: final held-out comparison

Recorded 29 October 2026, Asia/Singapore. Status: completed evaluation; tuning closed.

The unregularized model achieves higher mean held-out image scores. Adding
normal-depth consistency produces visibly smoother normal/depth maps, with residual
patches and a modest image-score cost. Retain both runs as an appearance/coherence
tradeoff; reference normal/depth accuracy was not evaluated.

## Models and fixed protocol

- Control: `materials_red-rose-silver-Oct29-1852-deferred-tune-normal-0-0`.
- Regularized: `materials_red-rose-silver-Oct29-1906-deferred-tune-normal-1-0p01`.
- Dataset: edited Materials red/rose/silver scene, three balls; 100 training views,
  100 validation views, **200 disjoint test views**. These results are separate from
  the earlier full Materials scene and its archived baselines.
- Both runs: fresh random initialization of 1,000 surfels, seed 7, 30,000 completed
  steps, 400x400 images, white background, 200,000 final active surfels, SH0,
  learned 128x64 environment map, deferred reflection active from step 3,000.
- Image/mask objective: `0.8*L1 + 0.2*D-SSIM + 0.1*mask_BCE`, where
  `D-SSIM = mean((1-SSIM)/2)`. Normal-depth coefficient is 0 or 0.01; the positive
  coefficient ramps from step 2,000 to 4,000. The self-supervised target normal is
  derived from rendered depth, not Blender geometry.
- Propagation disabled in both; identical growth/pruning and learning-rate settings;
  position decay ends at 30k. Quiet refinement starts at 20,001, keeping topology
  fixed and disabling opacity resets for the final 10k steps.
- Saved resolved configs were checked: only normal-depth coefficient and output
  destination differ. Current rendering/evaluation dependency hashes match each
  training source snapshot. Checkpoints, manifest, all 200 test inputs and evaluation
  dependencies are fingerprinted in the evaluation record; evaluator source is archived.

The validation sweep selected weight 0 before this test evaluation: validation
PSNR/SSIM were 34.5822/0.987773 versus 34.0690/0.985603 for weight 0.01.
No training, further weight tuning or model selection was performed on test views.
At the user's request, this comparison closes the normal-weight ablation.

## Final test results

Each metric below is the arithmetic mean of its per-view value over all 200
test cameras. RGB metrics use floating-point predictions before display
quantization; SSIM uses reflected uniform 11x11 windows. Foreground PSNR uses
reference PNG alpha >=0.95 and averages per-view PSNR, rather than pooling pixels
across cameras. Those foreground pixels occupy **6.9234%** of the image on average.

| Normal-depth weight | Test PSNR (dB) | Test SSIM | Foreground PSNR (dB) |
| --- | ---: | ---: | ---: |
| **0** | **34.5007** | **0.987244** | **23.4455** |
| 0.01 | 34.1435 | 0.984849 | 23.2170 |
| Change: 0.01 minus 0 | -0.3572 | -0.002394 | -0.2285 |

The regularized model improves per-view PSNR on **75/200** test views, SSIM on
**13/200**, and foreground PSNR on **85/200**. The appearance tradeoff therefore
varies by view, even though each scene mean favors the unregularized model.
Mean predicted coverage in alpha-zero background is small for both: 0.00004056
(weight 0) versus 0.00004174 (weight 0.01). Background cleanup is not the main
visual difference between these two runs.

Evidence: [all per-view metrics](../../outputs/materials-red-rose-silver-final-comparison-2026-10-29/test-metrics.csv),
[crop scores/bounds](../../outputs/materials-red-rose-silver-final-comparison-2026-10-29/crop-metrics.csv),
and [protocol, inputs and source/checkpoint fingerprints](../../outputs/materials-red-rose-silver-final-comparison-2026-10-29/comparison.json).

## Matched figures and observations

The four illustration cameras, test **r_0, r_72, r_123 and r_146**, were chosen by
deterministic pose-diversity sampling before rendering either model. Eight crops
were then defined from the largest connected components of reference alpha, with
a six-pixel margin and identical half-open pixel bounds for both predictions and
reference. Occlusion can connect multiple balls into one component. These are
object/reflection regions, not annotated specular masks. Scores use original crop
pixels; upscaling is for display only.

The regularized normal maps show substantially less small-scale mottling across
the four selected cameras. The conspicuous silver-surface streaks in control views
r_0/r_123 are reduced, although structured normal patches remain. Both variants
retain reflection blur and surface-like colour mottling compared with the reference
RGB, particularly on smooth reflective regions. The regularizer improves the
visible coherence of the diagnostics without recovering every reflection detail.

Normal previews use world-space normals, restricted to reference alpha >=0.95 and
predicted coverage >=0.5. Depth previews use the **same 1st/99th percentile depth
range for both models within each camera**. Raw buffers are retained for these
four views. These display choices suppress low-coverage background clutter without
removing it from the full-image RGB or background-coverage metrics.

- [Interactive comparison report](../../outputs/materials-red-rose-silver-final-comparison-2026-10-29/index.html).
- [Matched full RGB views](../../outputs/materials-red-rose-silver-final-comparison-2026-10-29/rgb-comparison.png).
- [Matched object/reflection crops](../../outputs/materials-red-rose-silver-final-comparison-2026-10-29/reflection-crops.png).
- [Normal comparison](../../outputs/materials-red-rose-silver-final-comparison-2026-10-29/normal-comparison.png).
- [Depth comparison](../../outputs/materials-red-rose-silver-final-comparison-2026-10-29/depth-comparison.png).

## Interpretation for the report

> In the edited three-ball Materials scene, adding self-supervised normal-depth
> consistency with coefficient 0.01 reduced mean test PSNR from 34.5007 to
> 34.1435 dB and SSIM from 0.987244 to 0.984849, while visibly reducing mottling
> in the rendered normal maps. The results expose a tradeoff between RGB fitting
> and surface coherence in the current deferred surfel representation. The
> validation-selected zero-weight model remains the appearance reference, and
> the regularized run provides a complementary geometry ablation. These
> observations concern visual coherence; they do not measure normal accuracy.

This is one scene and one seed. The renderer uses projected rank-two surfels and
blended centre-depth, so normal-depth agreement does not by itself establish a
correct surface. The learned environment/scalar reflection blend also does not
model full roughness, coloured specular response or local interreflections. These
are limits of the implemented experiment; no further tuning is required for this
record. LPIPS, controlled FPS and peak VRAM are not measured here.

## Reproduction and artifacts

```bash
python scripts/compare_surfel_runs.py \
  outputs/materials_red-rose-silver-Oct29-1852-deferred-tune-normal-0-0 \
  outputs/materials_red-rose-silver-Oct29-1906-deferred-tune-normal-1-0p01 \
  --output outputs/NEW_FINAL_COMPARISON --device cuda:0
```

The output contains all 200 reference renders, 200 renders per model, selected
diagnostic maps/raw buffers, eight matched crop triplets, CSV metrics, an HTML
gallery, evaluation-source archive and provenance JSON. Rendering does not modify
either checkpoint. No reference normals/depth were generated or used.
