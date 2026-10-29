# 3DGS consultation notes — 29 October 2026

**Latest revision:** the completed RGB result had visibly uneven normals. The user
requested the deferred-reflection paper's normal method while keeping the mask.
The active deferred preset now uses L1 + D-SSIM + mask BCE and reflection-driven
normal propagation; the old depth-normal sweep was interrupted. See the current
[method and commands](materials-red-rose-silver.md). Earlier notes below describe
the preceding depth-normal implementation.

## Requested next experiment

- Train normals with an explicit normal loss.
- Start with **two balls only**, restricted to the **second row**.
- Obtain the Materials Blender scene and capture multiple views with the same
  image/camera format as the original Materials dataset.
- User will delete the unwanted balls in Blender and save the two-ball scene.
  No subset training has been launched.

**Follow-up:** user saved `data/blender/materials_red-rose-silver.blend`.
Inspection found three balls (six mesh parts); use the edited file as supplied.
Capture script and dataset-specific RGB/deferred geometry presets are prepared:
[commands](materials-red-rose-silver.md).

## Normal loss

The current implementation already computes

\[
\mathcal L_{normal}=\frac{1}{|V|}\sum_{p\in V}
\left(1-\hat n_{render}(p)^\top\hat n_{depth}(p)\right),
\qquad
\hat n_{depth}=\operatorname{normalize}
\left(\partial_x P(D)\times\partial_y P(D)\right).
\]

Here, both normals are unit vectors in camera coordinates, facing the camera;
\(P(D)\) back-projects rendered camera-z depth. Valid pixels \(V\) exclude
background, low coverage, depth discontinuities and degenerate neighbourhoods.
Gradients flow through both normals and depth. Empty valid sets give zero loss.

See [implementation](../src/cp4282_gs/surfels/geometry_losses.py) and
[RGB geometry preset](../config/materials_surfel_rgb_geometry.yaml). The preset
ramps the normal weight from 0 at step 2,000 to 0.01 at step 4,000, alongside
photometric loss and alpha-mask loss (weight 0.1).

This is a normalized, masked variant of normal–depth consistency; the
[official 2DGS training code](https://github.com/hbb1/2d-gaussian-splatting/blob/main/train.py)
and [renderer](https://github.com/hbb1/2d-gaussian-splatting/blob/main/gaussian_renderer/__init__.py)
use different compositing/coverage normalization. Current local depth blends
surfel-centre depths. Consistency alone does not establish correct normals;
Blender ground-truth normals would allow an independent geometry check.

Verification today: all **11 existing geometry-loss tests pass on CPU and CUDA**,
including gradient and training-integration checks. No trainer code was changed.

## Blender scene and capture code

- Original NeRF scene archive: look for `blend_files.zip` in the
  [official data folder](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
  The archive name is identified in [the NeRF repository discussion](https://github.com/bmild/nerf/issues/195).
  The full archive was not downloaded. A Materials scene copy was obtained from
  the [Robust e-NeRF authors' dataset](https://huggingface.co/datasets/wengflow/robust-e-nerf/tree/main/scenes),
  whose attribution identifies the original NeRF archive as its source.
  Local file: `data/blender/materials.blend` (2,656,524 bytes).
  SHA-256: `cefb7887d1d54eaa62ff8ab3e11fe959e6ec4ef29fe9d2e708199502379da878`.
  It opens in Blender 5.0.1, contains 24 mesh parts and a camera, and includes
  packed `forest.exr` lighting plus `360Render` / `360_view_test.py` text blocks.
  Exact lighting equivalence to the existing RGB dataset has not been established.
- Upstream Materials asset: [Metal Materials Pack for Cycles](https://blendswap.com/blend/10120),
  by elbrujodelatribu, CC0. Prefer the NeRF-modified scene for matching lighting,
  scale and appearance; the original asset alone does not reproduce that setup.
- Capture implementation: [BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF).
  Source ZIP and extracted code are saved under
  `outputs/consultation-2026-10-29/`. Camera sampling is in `cos_operator.py` and
  `helper.py`; camera/image export is in `blender_nerf_operator.py`.
  Select NeRF export format. Its camera-on-sphere mode supports upper-hemisphere
  sampling; validation and rendered test splits need explicit preparation.

## Capture and training checklist

1. User saved `data/blender/materials_red-rose-silver.blend` with three balls.
   Each ball has `SolidModel` and `CoreSolidMaterial` mesh parts with matching
   suffixes; remove both parts of each unwanted ball. Keep the camera and lighting.
2. Prefer replaying the existing `data/materials/transforms_{train,val,test}.json`
   poses and horizontal FOV. Confirm framing after selecting the subset.
   Export 800 × 800 RGBA PNGs, `camera_angle_x`, relative image paths, and 4 × 4
   camera-to-world matrices. Local data has 100 train / 100 validation / 200 test views.
3. Save a separate dataset at `data/materials_red-rose-silver/`. Preserve disjoint
   splits and source scene/object/camera settings. Export float normals and depth
   for diagnostics with their coordinate conventions and depth definition recorded.
4. Point copied RGB-surfel and deferred-surfel geometry presets at that dataset.
   Compare matched runs with and without normal loss; inspect coverage, normals,
   depth and validation image metrics before expanding the experiment.

Blender is installed at `C:/Program Files/Blender Foundation/Blender 5.0/blender.exe`
(not on PATH). The edited scene is ready; use the new capture script and
[prepared commands](materials-red-rose-silver.md). Low-resolution capture/resume
and short RGB/deferred GPU training passed; full runs remain to be launched.
