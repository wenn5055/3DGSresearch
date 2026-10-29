# Ship benchmark source

Ship is the secondary scene in the original proposal and Unit 11. Download the
100 train / 100 validation / 200 test RGBA images and their three manifests to
`data/ship/`; keep the raw dataset out of commits. Images are 800x800.

- Original dataset: [NeRF](https://github.com/bmild/nerf), Mildenhall et al.
- Download mirror: [pablovela5620/nerf-synthetic-mirror](https://huggingface.co/datasets/pablovela5620/nerf-synthetic-mirror).
- Pinned mirror revision: `23e7eaa65ff836405329c7b6b4a34afaeb3a70e2`.
- Mirror cross-check: its `materials/transforms_train.json` and
  `materials/train/r_0.png` match the existing local Materials files byte for byte.
  This validates those two files, not every mirrored scene/file independently.
- The mirror's upstream README attributes Ship to gregzaal's
  [BlendSwap ship asset](https://www.blendswap.com/blend/8167), CC-BY-SA.
  Keep the source attribution with any distributed dataset/derived asset.
- Download manifest, exact per-file SHA-256 hashes, pinned URLs, image dimensions
  and completion status: `data/ship/SOURCE.json`.
- Session download script/log: `outputs/first-principles-2026-10-29/download_ship.py`
  and `download-ship.log`.

Ship should test whether the reflection extension preserves broader directional
effects, including water, rather than serving as a diffuse-only scene. Compare
models under one shared resolution, capacity, schedule, split and background.
Do not change appearance settings based on Ship test images.
