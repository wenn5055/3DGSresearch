# Output folders — quick guide

Updated 29 Oct 2026 (Singapore).

- Each bullet says **what changed**; multiple changes are not isolated causal tests.
- `prepared` = no training; `smoke` = tiny execution check, not a quality experiment.
- Split/status at the end; full evidence in each folder. `decay` means position LR decay length.
- Auto-refreshed by runners; future `--note "goal / change"` supplies the purpose.
- Manual refresh: `python scripts/update_output_index.py`.

## Training runs

- [materials-20261027T081123Z-e0ca4ce2](../outputs/materials-20261027T081123Z-e0ca4ce2/) — Prepare Materials 256px/100k/2k setup; no training. [prepared; test]
- [materials-20261027T082508Z-51f0952e](../outputs/materials-20261027T082508Z-51f0952e/) — Materials baseline: 256px, 100k capacity, 2k steps. [completed; val]
- [materials-20261027T095514Z-332633f0](../outputs/materials-20261027T095514Z-332633f0/) — Prepare 5k-step/5k-decay variant; no training. [prepared; test]
- [materials-20261027T101525Z-47cd07b6](../outputs/materials-20261027T101525Z-47cd07b6/) — vs Oct27 082508: steps AND position decay 2k → 5k. [completed; val]
- [materials-20261027T102215Z-0b52e390](../outputs/materials-20261027T102215Z-0b52e390/) — vs Oct27 101525: steps AND position decay 5k → 10k. [completed; val]
- [materials-20261027T112424Z-c556467e](../outputs/materials-20261027T112424Z-c556467e/) — vs Oct27 102215: resolution 256 → 512px. [completed; val]
- [materials-20261027T113646Z-58a47e63](../outputs/materials-20261027T113646Z-58a47e63/) — vs Oct27 112424: resolution 512 → 800px; 100k capacity. [completed; val]
- [lego-20261028T043116Z-8a700f36](../outputs/lego-20261028T043116Z-8a700f36/) — Lego baseline: 800px, 100k capacity, 2k steps. [completed; test]
- [lego-20261028T043530Z-5299e5c7](../outputs/lego-20261028T043530Z-5299e5c7/) — vs Lego 043116: steps 2k → 10k; position decay stays 2k. [completed; test]
- [materials-20261028T045111Z-98b4397b](../outputs/materials-20261028T045111Z-98b4397b/) — vs Oct27 113646: capacity 100k → 200k; 10k steps. [completed; val]
- [materials-20261028T045722Z-9fe24bef](../outputs/materials-20261028T045722Z-9fe24bef/) — vs Oct28 045111: capacity 200k → 500k; 10k steps. [completed; val]
- [materials-20261028T050531Z-a8f4529f](../outputs/materials-20261028T050531Z-a8f4529f/) — vs Oct28 045722: capacity 500k → 1M AND steps 10k → 60k; decay stays 10k. [completed; val]
- [materials-Oct28-1801](../outputs/materials-Oct28-1801/) — vs Oct28 050531: position decay 10k → 2k; requested 60k, stopped at 45k. [completed; val]
- [materials-Oct28-1833](../outputs/materials-Oct28-1833/) — vs 1801: position decay 2k → 60k AND budget 60k → 30k; MSE reference; test re-eval in eval-test/. [completed; val]
- [materials-Oct28-1945-dssim](../outputs/materials-Oct28-1945-dssim/) — vs 1833: add staged 20% D-SSIM; matched 30k-step comparison. [completed; test]
- [materials-Oct28-2157-sh2](../outputs/materials-Oct28-2157-sh2/) — vs 1945-dssim: add SH through degree 2; same 30k budget. [completed; test]
- [materials-Oct28-2157-sh3](../outputs/materials-Oct28-2157-sh3/) — vs 2157-sh2: maximum SH degree 2 → 3. [completed; test]
- [materials-Oct28-2252-sh3-tune-lr-6p25e-05](../outputs/materials-Oct28-2252-sh3-tune-lr-6p25e-05/) — vs 2157-sh3: SH coefficient LR 0.000125 → 0.0000625. [completed; val]
- [materials-Oct28-2327-sh3-tune-lr-0p00025](../outputs/materials-Oct28-2327-sh3-tune-lr-0p00025/) — vs 2157-sh3: SH coefficient LR 0.000125 → 0.00025. [completed; val]
- [materials-Oct29-1144-control](../outputs/materials-Oct29-1144-control/) — Full project control: Gaussian RGB, training-only initialization, staged D-SSIM; 30k steps, early stopping off. [completed; val]
- [materials-Oct29-1234-surfel-rgb](../outputs/materials-Oct29-1234-surfel-rgb/) — vs 1144-control: flatten local z into surfels; same RGB/loss/budget. Inspect normals/depth/coverage as well as colour. [completed; val]
- [materials-Oct29-1305-surfel-deferred](../outputs/materials-Oct29-1305-surfel-deferred/) — vs 1234-surfel-rgb: add environment reflection after 10k RGB warm-up; independent 30k run, no geometry losses. [completed; val]
- [materials-Oct29-1347-rgb-mask-smoke](../outputs/materials-Oct29-1347-rgb-mask-smoke/) — Mask-loss smoke stopped before training: default Warp cache permission failure; retried with workspace cache. [failed; val]
- [materials-Oct29-1347-rgb-mask-smoke-2](../outputs/materials-Oct29-1347-rgb-mask-smoke-2/) — 8-step/64px check: load training alpha, train mask BCE, densify, save/evaluate. Not quality evidence. [completed; val]
- [materials-Oct29-1348-rgb-geometry-smoke](../outputs/materials-Oct29-1348-rgb-geometry-smoke/) — vs mask smoke: add normal-depth schedule and loss; tiny random scene, no valid normal pixels at final step. [completed; val]
- [materials-Oct29-1348-deferred-geometry-smoke](../outputs/materials-Oct29-1348-deferred-geometry-smoke/) — vs RGB geometry smoke: activate deferred reflection too; 8-step/64px integration check. [completed; val]

## Tuning

- [materials-Oct28-2246-sh3-lr-sweep](../outputs/materials-Oct28-2246-sh3-lr-sweep/) — SH coefficient LR: 0.000125 / 6.25e-05 / 0.00025; validation only. Selected 0.00025 by validation PSNR. [completed]

## Supporting folders

- [dssim-smoke-input-64a59d5c](../outputs/dssim-smoke-input-64a59d5c/) — tiny D-SSIM check config.
- [eval](../outputs/eval/) — legacy evaluation images/CSV; source checkpoint unrecorded here.
- [project-smoke-input-9a94e30b](../outputs/project-smoke-input-9a94e30b/) — tiny project-pass configs and launcher logs.
- [project-smoke-input-geometry-4001cc4d](../outputs/project-smoke-input-geometry-4001cc4d/) — tiny project-pass configs and launcher logs.
- [runner-validation-inputs-f2cfb4b9](../outputs/runner-validation-inputs-f2cfb4b9/) — small manifests/config for runner checks.
- [sh-smoke-input-2f60e9eb](../outputs/sh-smoke-input-2f60e9eb/) — tiny SH check config/launcher log.
- [warp-training_snapshots](../outputs/warp-training_snapshots/) — legacy direct-trainer snapshots; source run unrecorded here.
- `dssim-test-warp-cache`, `geometry-loss-warp-cache`, `runner-validation-warp-cache`, `sh-test-warp-cache`, `surfel-test-warp-cache` — compiled Warp kernels reused by checks; not experiments.

Scores: [findings](findings.md). Commands: [project passes](project-passes.md).
