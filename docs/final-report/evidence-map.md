# Final report evidence map

Prepared 7 November 2026. Main study: 100k points, seed 7, 800px, 10k frozen + 5k refinement. The later 200k defaults are not the main comparison. All paths below are relative to this folder. Figure numbers refer to [report.md](report.md).

| Report claim | Primary evidence | Conditions and limits |
| --- | --- | --- |
| Matched HDR beats SH3 on the sphere | [Test means](../reports/assets/2026-11-03-sphere-study/test/summary.csv), [per-view results](../reports/assets/2026-11-03-sphere-study/test/metrics.csv), [protocol](../reports/assets/2026-11-03-sphere-study/test/protocol.json) | 200 test views; one scene/seed; both use surfels; complete appearance treatments, not a pure shader ablation |
| Main improvement spans all saved cameras; size gain is smaller/nonuniform | [Derived paired analysis](../reports/assets/2026-11-07-report-analysis/analysis.json), [effect plot](../reports/assets/2026-11-07-report-analysis/paired-test-effects.png), [analysis script](analyse_saved_results.py) | Saved CSV analysis only; per-view variation is descriptive camera dependence, not seed uncertainty or a new selection |
| Size-0.75 selection preceded test | [Selection](../reports/assets/2026-11-03-sphere-study/selection.json), [validation means](../reports/assets/2026-11-03-sphere-study/validation/summary.csv), [initializer audit](../reports/assets/2026-11-03-sphere-study/size-preflight.json) | Main primary contrast remains SH3 vs HDR-1.0; selected additional variant passes LPIPS/foreground/coverage rule |
| Refinement improves the original HDR baseline | [Stage evaluation](../experiments/2026-11-03-sphere-lpips.md), [frozen metrics](../reports/assets/2026-11-02-sphere/frozen.metrics.csv), [refined metrics](../reports/assets/2026-11-02-sphere/refine.metrics.csv) | 100 validation views; +5k steps; no equal-budget frozen control; LPIPS evaluated later without training |
| Smaller footprints do not resolve blur | [Footprint/normal measurements](../reports/assets/2026-11-03-sphere-study/diagnostics.csv), [actual test crops](../reports/assets/2026-11-03-sphere-study/test-reflection-crops.png) | Candidate support includes hidden/back-side primitives; initializer scale is learnable later |
| Storage/rendering measurements | [Performance record](../reports/assets/2026-11-03-sphere-study/test/performance.json), [archived stage runs](../reports/assets/2026-11-03-sphere-study/) | Serialized required files; forward timing scope documented; sums of both recorded training stages, not refinement alone; training times were not isolated |
| Learned environment lacks detail | [Aligned map comparison](../reports/assets/2026-11-03-environment-check/comparison.json), [actual maps](../reports/assets/2026-11-03-environment-check/comparison.png) | Matched grid and exposure, no fit to gain or rotation; diagnostic, not calibrated lighting recovery |
| Reference light changes detail/metrics | [Lighting-swap means](../reports/assets/2026-11-03-reference-lighting-swap/summary.csv), [protocol](../reports/assets/2026-11-03-reference-lighting-swap/protocol.json) | 100 validation views; only light replaced; all learned material/geometry parameters fixed |
| Normal errors contribute under reference light | [Normal-swap means](../reports/assets/2026-11-03-reference-normal-swap/summary.csv), [normal metrics](../reports/assets/2026-11-03-reference-normal-swap/normal-metrics.csv), [protocol](../reports/assets/2026-11-03-reference-normal-swap/protocol.json) | Five previously inspected validation views, HDR-0.75; normal direction only; includes reflection direction and angular-response effects |
| Effect of substituting normals reverses with lighting | [Conditional effects](../reports/assets/2026-11-07-report-analysis/analysis.json) | Same saved four-case experiment; LPIPS change +0.005782 under learned light vs −0.028657 under reference light; descriptive conditional reversal, no unique disentanglement claim |
| Higher consistency weight did not help | [Paired means](../reports/assets/2026-11-07-normal-weight-closeout/validation-summary.csv), [normal diagnostics](../reports/assets/2026-11-07-normal-weight-closeout/normal-diagnostics.csv), [audit](../reports/assets/2026-11-07-normal-weight-closeout/provenance.json) | Training completed 3 November; evaluation-only closeout 7 November; 100 validation + five normal views; size-1.0 model; separate from frozen main test |
| L1/D-SSIM improves LPIPS but not fine-detail recovery | [Paired means](../reports/assets/2026-11-07-refinement-loss/validation-summary.csv), [normal diagnostics](../reports/assets/2026-11-07-refinement-loss/normal-diagnostics.csv), [audit](../reports/assets/2026-11-07-refinement-loss/provenance.json), [experiment note](../experiments/2026-11-07-refinement-loss.md) | One user-requested 5k continuation from same MSE-trained parent; 100 validation + five normal views; PSNR cost and essentially unchanged normal accuracy; no test evaluation |
| Materials evidence is mixed | [Protected control](../experiments/2026-11-02-protected-control.md), [portable provenance](../reports/assets/2026-11-02/provenance.json) | 100 validation views; LDR deferred plus SH3, distinct protocol; PSNR improves but LPIPS worsens |
| Logo recovery remains unsuccessful | [Position-rate trial](../experiments/2026-11-02-logo-position-rate.md), [portable evidence](../reports/assets/2026-11-02-logo-position5x/provenance.json) | Different scene; matched LR trial; user-selected development view 18; local normal change small |
| Environment-only model misses local visibility | [Reflection-scope audit](../experiments/2026-10-30-reflection-model-scope.md) | Four original-Materials diagnostic views; first-bounce ray categories, not energy-weighted transport or an image-error bound |

## Figures

| Figure | Existing experimental asset | Scene / views / variants |
| --- | --- | --- |
| M1 | [Implemented-method schematic](../reports/assets/2026-11-07-report-readability/method-pipeline.svg), [provenance](../reports/assets/2026-11-07-report-readability/diagram-provenance.json) | Conceptual diagram of actual forward/training code, explicitly not an experiment render or planned result |
| 1 | [Dataset captures](../reports/assets/2026-11-02-sphere/dataset-views.png) | Sphere, validation 0/13/73/88, reference only |
| 2 | [Paired test effects](../reports/assets/2026-11-07-report-analysis/paired-test-effects.png) | Sphere, all 200 saved test rows, HDR-1.0 vs SH3; whole/foreground PSNR and LPIPS |
| 3 | [Compact main test crops](../reports/assets/2026-11-07-report-analysis/test-crops-views-0-73.png), [full original](../reports/assets/2026-11-03-sphere-study/test-reflection-crops.png) | Sphere, test 0/73 selected from original illustration cameras; target/SH3/HDR-1.0/HDR-0.75 |
| 4 | [Preserved regression](../reports/assets/2026-11-02-sphere/regression-view-51.png) | Sphere, validation 51, target/frozen/refined HDR |
| 5 | [Environment comparison](../reports/assets/2026-11-03-environment-check/comparison.png) | Original forest / HDR-0.75 learned map, matched 0/-2 EV |
| 6 | [Normal substitution](../reports/assets/2026-11-03-reference-normal-swap/normal-swap-highlight.png) | Sphere, validation 0, target/reference-light learned normals/reference-light reference normals |
| 7 | [Weight trial crops](../reports/assets/2026-11-07-normal-weight-closeout/reflection-crops.png) | Sphere, validation 0/51, target/weight 0.01/weight 0.03; actual saved renders |
| 8 | [Loss trial highlight](../reports/assets/2026-11-07-refinement-loss/reflection-highlight-view-0.png), [five-view sheet](../reports/assets/2026-11-07-refinement-loss/reflection-crops.png) | Sphere, validation 0 in report; all five fixed views in sheet; target/MSE/L1+D-SSIM; actual saved renders |
| D1 | [Logo failure](../reports/assets/2026-11-02-logo-position5x/logo-comparison.png) | Enlarged Chrome, validation 18, target/position LR 1x/5x, RGB and normals |
| D2 | [Compact normal example](../reports/assets/2026-11-07-report-analysis/normal-example-view-0.png), [full original](../reports/assets/2026-11-03-reference-normal-swap/normal-comparison.png) | Sphere, validation 0, reference/learned/error; layout crop only |

No experimental image is generated by AI. The weight-trial sheet composes actual evaluation PNG crops; its input hashes and crop coordinates are archived. Compact main-test/normal figures crop the tracked original sheets using recorded rectangles. The new plot uses saved metric rows. Model renders and quantitative metrics remain separate.

## Review and reproducibility additions

Three [independent review memos](reviews/revision-summary.md) informed the revision; central numerical/model conclusions are unchanged. [The reproduction launcher](reproduce_study.py) was checked in preparation-only mode, preserving every non-path setting of all six archived stage configs. [The local bundle manifest](../reports/assets/2026-11-07-report-analysis/checkpoint-bundle-manifest.json) records seven model-file hashes and the 35.23 MB checkpoint ZIP. All 400 sphere images/manifests are tracked. The author still needs a persistent bundle location; no publishing or retraining was performed.

After that review, the user explicitly requested the single L1/D-SSIM refinement trial. Its [controller](run_loss_trial.py) checked source/input hashes and preserved main checkpoints; it does not change the main bundle/selection. New training was limited to that 5k continuation, with validation only.

## Method checks used while drafting

- [HDR shader](../../src/cp4282_gs/surfels/hdr_shader.py): interpolate raw environment then softplus; interpolate 16 angular logit offsets; linear/display conversion before coverage composition.
- [Surfel trainer](../../src/cp4282_gs/surfels/trainer.py): frozen state includes positions, tangent scales, orientation and opacity; Adam restarts between stages.
- [Metric implementation](../../image_metrics.py): SSIM uses reflected uniform 11×11 RGB windows, not the original paper's complete reference implementation.
- [Archived study configs](../reports/assets/2026-11-03-sphere-study/): 100k capacity; frozen-stage geometry losses zero; refinement mask 0.1, normal weight 0.01 with 1k ramp.
- [Performance evaluator](../../scripts/evaluate_reflection_suite.py): four selected views, one warm-up sequence and ten timed sequences; MSE and host tiling included; perceptual metrics/file I/O excluded.

## Source and publication checks

Primary paper/project pages for 3DGS, Gaussian Surfels, 3DGS-DR, RGS-DR, 2DGS, Ref-NeRF, SSIM and LPIPS were consulted for this draft. Report references link directly to them. RGS-DR is related work, not claimed as reproduced. The course project page was read directly on 7 November: repo/reproduction link and AI declaration are required; it does not state a report format or page limit. Exact copied-code provenance and personal AI-use confirmation remain author review items.

The later audience/readability rewrite directly read Units 5, 6, 8, 9, 10, 11 and 12. [Course connections](course-connections.md) explains their role; [the source manifest](course-references.json) records primary-page hashes and checked section anchors. Section 4 now explains loss choices separately from implementation and results. The course's MSE/D-SSIM curriculum is explicitly distinguished from the project's L1/D-SSIM continuation. Numeric result tables and actual experiment images are preserved; the long gradient/shader details and cost records move to appendices. No experiment was rerun for that rewrite.
