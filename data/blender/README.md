# Materials Blender scene

The edited scene is saved as `materials_red-rose-silver.blend` in this folder
(three balls). Keep `materials.blend` as the original source.

Each ball has two mesh parts: `SolidModel` and `CoreSolidMaterial`, with matching
numeric suffixes. Delete both parts of each unwanted ball. Keep the camera and
world lighting. The environment texture is packed into the file.

Downloaded 29 October 2026 from the
[Robust e-NeRF authors' scene collection](https://huggingface.co/datasets/wengflow/robust-e-nerf/tree/main/scenes).
See [upstream attribution](SOURCE.md). This copy opens in Blender 5.0.1; exact
lighting equivalence to the existing Materials images has not been checked.

Original SHA-256:
`cefb7887d1d54eaa62ff8ab3e11fe959e6ec4ef29fe9d2e708199502379da878`.

Current capture and training commands: [edited-scene workflow](../../docs/materials-red-rose-silver.md).
