<!-- ARCHIVE CANDIDATE (2026-10-29): Assignment 2 skeleton walkthrough; backward kernels and later research features are now implemented. Successors: 3dgs_trainer.py, docs/project-passes.md and docs/materials-red-rose-silver.md. -->

# Annotated Walkthrough: `3dgs_trainer.py`

This walkthrough explains the Assignment 2 trainer skeleton. Everything in the file is described
except the reference bodies of the two backward kernels, which are your task.

Assignment 1 built a renderer: given splats, produce an image. Assignment 2 runs that renderer
backwards: given images, produce splats. The reference renderer source is included as
`3dgs_renderer_v1.py`.

## What Changes From Assignment 1

In Assignment 1 the splats were fixed. You loaded a PLY, projected it, and shaded pixels. Here the
splats are unknown. You start from a rough point cloud and a set of photographs with known camera
poses, and you search for the splat parameters whose renders match those photographs.

That search is ordinary gradient descent:

1. Render the current splats from one training camera.
2. Compare against the real photograph. The disagreement is a single number, the loss.
3. Work out, for every parameter of every splat, which direction would reduce that number.
4. Take a small step in that direction.
5. Repeat, tens of thousands of times.

Step 3 is the whole of your assignment. Steps 1, 2, 4 and 5 are provided.

| Assignment 1 | Assignment 2 |
|---|---|
| Splats fixed, loaded from PLY | Splats are the unknowns being solved for |
| One render | ~20,000 renders |
| No loss | Mean squared error against the photograph |
| No gradients | Explicit hand-written backward pass |
| No optimizer | Adam, plus splitting, cloning and pruning splats |

## Why `importlib` Appears Again

Same reason as Assignment 1. A module name cannot begin with a digit, so files like
`3dgs_trainer.py` cannot be imported with a normal `import` statement. Where this file needs a
neighbour it uses:

```python
workspace_module = importlib.import_module("gaussian_first_tile_workspace_gpu")
```

The trainer also puts its sibling `shared/` directory on the import path so the helpers resolve:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent / "shared"))
```

`shared/` holds the data classes, tile builder and configuration schema. The Assignment 2 repository
contains all modules needed by the trainer; it does not depend on an Assignment 1 solution folder.

## The Parameters, and Why They Are Stored Oddly

A single splat is an anisotropic 3D Gaussian with a colour and an opacity. Five parameter groups:

| Stored as | Meaning | Why stored this way |
|---|---|---|
| `means` (`vec3`) | Centre in world space | Unconstrained already |
| `log_scales` (`vec3`) | Log of the three axis lengths | A scale must stay positive; a log can be any real number |
| `quaternions` (`vec4`) | Orientation | Four numbers instead of a 3×3 matrix with six constraints |
| `opacity_logits` (`float`) | Opacity before the sigmoid | An opacity must stay in [0, 1]; a logit can be any real number |
| `colors` (`vec3`) | RGB | Clamped to [0, 1] after each step |

This is the first principle worth internalising: **gradient descent wants unconstrained
parameters.** If you stored a scale directly, a large enough downhill step would make it negative,
which is meaningless. Storing `log(scale)` means every real value the optimizer can produce maps
to a legal scale. The exponential and the sigmoid are re-applied inside the kernels, so the
constraint is enforced by the parameterisation rather than by clipping after the fact.

You will see this in `alpha_at_pixel`:

```python
opacity = 1.0 / (1.0 + wp.exp(-opacity_logit))
```

and in `constrain_parameters`, which clamps log-scales into a band and colours into [0, 1] after
every optimizer step.

## Configuration

`TrainingConfig` (imported from `shared/training_config.py`) loads a YAML file, fills in defaults,
and rejects unknown keys. Sections:

| Section | Controls |
|---|---|
| `paths` | Dataset directory, output PNG, optional initial PLY |
| `runtime` | `cpu` or `gpu`, random seed |
| `model` | Image size, splat capacity, initial splat count, background |
| `training` | Iteration cap |
| `learning_rates` | Per-parameter Adam rates and the position decay schedule |
| `densification` | How often to add splats, and the opacity reset interval |
| `compact_box` | Per-splat support radius |
| `multiview_adc` | Multi-view scoring for densification and pruning |
| `sparse` | Sampled-pixel training |
| `adaptive` | Growth-plateau detection |
| `convergence` | Early stopping |
| `reporting` | Log, eval and snapshot cadence, and which views `fixed_eval` is measured on |

Rejecting unknown keys matters more than it sounds. A typo in a YAML key would otherwise be
silently ignored and you would spend an afternoon wondering why a setting had no effect.

## Loading the Data

`load_views` reads a NeRF-synthetic manifest — `transforms_train.json` by default. Each frame
gives a `transform_matrix` (camera-to-world, Blender convention) and an image path. It also takes
`frame_ids`, which loads only the frames you name; the held-out evaluation set uses that so it
decodes eight images instead of the whole validation split.

```python
def blender_pose_to_world_to_camera(transform: list[list[float]]) -> np.ndarray:
```

Two conversions happen here. Blender's camera looks down `-Z` with `+Y` up; the renderer wants
`+Z` forward. And the manifest stores camera-to-world, while projection needs world-to-camera, so
the matrix is inverted.

Images are RGBA. The alpha channel is composited over the configured background:

```python
rgb = rgb * alpha + background * (1 - alpha)
```

If you skipped this, the transparent surround would train as black, and the model would learn a
dark halo around the object.

The focal length comes from `camera_angle_x` in the manifest, scaled to the working resolution.

### The held-out evaluation set

The trainer loads a *second* manifest, `reporting.eval_manifest` (default `transforms_val.json`).
Those views are what `fixed_eval` is measured on, and they are never used to compute a gradient.

The reason is that a loss measured on pictures you are actively fitting only tells you the
optimiser is working, not that the model is any good. The difference is not subtle. In a long
reference run on this scene — 768x768, 20 000 iterations, far past what `config/` ships — the
training views read about 33.9 dB while the held-out views read about 28.3 dB, and the gap grows
from under 1 dB early on to over 5 dB by the end. The first number is memorisation; the second is
what the model would do on a camera it has never been shown. Your own run is much shorter and at
a lower resolution, so expect smaller numbers and a smaller gap — the point is the direction, not
the values.

Keeping both sets usable at once needs one trick. The trainer indexes cameras and ground truth by
a single view id, so the held-out views are **appended after** the training views in the same
arrays, and `run_training` is handed only the training-view array. Its random camera draw is
`rng.integers(len(targets))` over that shorter array, so it cannot reach a held-out view — the
separation is a property of the indexing, not a check that could be forgotten.

Which frames are chosen is decided by `select_diverse_frame_ids`: farthest-point sampling over the
unit camera directions, which spreads the eight views over the sphere. Simply taking every 25th
frame is worse than it looks — the validation cameras walk a spiral rather than an ordered sweep,
so an even stride can land two picks 2.4 degrees apart and measure nearly the same view twice.
Set `reporting.eval_view_ids` to pin exact frames when two runs must be scored on identical
cameras, or `reporting.eval_manifest: null` to go back to evaluating on training views.

The trainer prints the frames it picked on its first line, so the evaluation set is recoverable
from any log.

## Geometry: From 3D Gaussian To Screen-Space Ellipse

Three `@wp.func` helpers do the projection. A `@wp.func` is a device function: it runs inside a
kernel, and Warp can differentiate it.

### `projected_covariance`

A 3D Gaussian has covariance `Σ = R S Sᵀ Rᵀ`, where `R` comes from the quaternion and `S` is the
diagonal of scales. Projecting it to the image plane is not just dropping a coordinate, because
perspective projection is non-linear. The standard treatment linearises it: take the Jacobian `J`
of the projection at the splat's centre, and the screen-space covariance is

```
Σ' = J W Σ Wᵀ Jᵀ
```

with `W` the world-to-camera rotation. The function returns the three distinct entries of the
2×2 result, since it is symmetric.

`FILTER_VARIANCE = 0.3` is added to the diagonal. This is a low-pass filter: without it, a splat
smaller than a pixel would alias badly as the camera moves.

### `conic_at_centre`

Inverts that 2×2 covariance. The inverse is what you actually evaluate per pixel, because the
Gaussian's exponent is a quadratic form in the inverse covariance — the "conic".

### `alpha_at_pixel`

The workhorse. Given one splat and one pixel, return that splat's alpha there.

```python
point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)
z = wp.max(point[2], NEAR_PLANE)
centre_x = focal * point[0] / z + 0.5 * width
centre_y = focal * point[1] / z + 0.5 * height
conic = conic_at_centre(log_scale, quaternion, camera, point[0], point[1], z, focal)
dx, dy = px - centre_x, py - centre_y
q = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy
opacity = 1.0 / (1.0 + wp.exp(-opacity_logit))
```

`q` is the squared Mahalanobis distance from the pixel to the splat centre. The alpha is

```python
candidate = wp.min(opacity * wp.exp(-0.5 * q), 0.99)
```

Two cutoffs then apply:

- **Support radius.** Rather than a fixed 3-sigma disc, each splat gets its own cutoff derived
  from its opacity, when `compact_box.enabled` is true. A faint splat reaches the visibility floor
  sooner, so it can be truncated sooner. The same rule decides which tiles a splat is written
  into, so tiling and shading agree — if they disagreed, a splat could be listed in a tile and
  then contribute nothing, or worse, contribute in a tile it was never listed in.
- **`ALPHA_CUTOFF`** (1/255). Below one 8-bit level, the contribution cannot change the saved
  image, so it is dropped.

Note `alpha_at_pixel` is a pure function of the splat parameters. That is deliberate, and it is
what makes your backward pass tractable — see below.

## Tiling: Why Not Loop Over All Splats

A 800×800 image with 500,000 splats is 3.2×10¹¹ pixel-splat pairs if evaluated naively. Almost all
of them are zero, because a splat covers a small part of the screen.

So the screen is cut into 16×16 tiles (`TILE = 16`), and each splat is written into the tile
records it actually overlaps. `build_tiles` delegates to `GaussianFirstTileWorkspace`, which
produces two arrays:

| Array | Meaning |
|---|---|
| `pairs` | Sorted 64-bit keys. Upper bits: tile id and depth. Lower 32 bits: splat index. |
| `offsets` | Where each tile's run begins in `pairs` |

Packing depth into the sort key is a trick worth understanding: sorting the keys once sorts every
tile's splat list front-to-back simultaneously, because tile id is more significant than depth.
The renderer then walks `pairs[offsets[record] : offsets[record + 1]]` and gets correctly ordered
splats with no per-tile sorting.

A pixel finds its tile record with:

```python
record = batch_view * tiles_x * tiles_y + (pixel // width // tile) * tiles_x + (pixel % width // tile)
```

## The Forward Pass

`render_forward` is one thread per pixel. It is Assignment 1's compositing loop, plus a loss.

```python
rgb = wp.vec3(0.0, 0.0, 0.0)
transmittance = float(1.0)
for entry in range(offsets[record], offsets[record + 1]):
    splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
    alpha = alpha_at_pixel(...)
    if alpha > 0.0:
        colour = colour_at_view(color[splat])
        rgb = rgb + transmittance * alpha * colour
        transmittance = transmittance * (1.0 - alpha)
        if transmittance < TRANSMITTANCE_CUTOFF:
            break
rgb = rgb + transmittance * background
image[thread] = rgb
```

Read this as a recurrence. Write `T₀ = 1`, and for the i-th splat in front-to-back order:

```
contribution_i = T_i * α_i * c_i
T_{i+1}        = T_i * (1 - α_i)
```

`T_i` is the transmittance: the fraction of light from behind splat i that still reaches the
camera. The final pixel is the sum of contributions plus the background weighted by whatever
transmittance survives.

The early exit at `TRANSMITTANCE_CUTOFF` is not just an optimisation — once transmittance is
1e-4, nothing behind can change the pixel, so the loop stops. Your backward pass must respect the
same stopping rule, or it will accumulate gradient for splats the forward pass never used.

The loss is mean squared error, accumulated atomically across all pixels:

```python
difference = rgb - targets[view * pixels + pixel]
wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * pixels * view_count))
```

`wp.atomic_add` is required because thousands of threads add into the same scalar concurrently.

## The Pixel Gradient

`mse_pixel_gradient` computes `d(loss)/d(pixel)` for every pixel, into a buffer:

```python
difference = image[thread] - target[target_pixel]
pixel_gradient[thread] = difference * (2.0 / float(3 * pixels * view_count))
```

This is just the derivative of `mean((rendered - target)²)`. It is computed into a buffer rather
than inline in the backward kernel for a reason worth noting: later versions of this trainer blend
a different objective (spherical harmonics, D-SSIM) by overwriting this buffer before the backward
runs. The backward kernel does not need to know which loss produced the numbers.

## Your Task: The Backward Pass

Two kernels, `render_backward` and `render_sparse_backward`, both marked TODO.

### What Is Being Asked

The forward pass turned parameters into a pixel. You now have `d(loss)/d(pixel)` and want
`d(loss)/d(parameter)` for every parameter of every splat that touched that pixel.

This is the chain rule, applied through the compositing recurrence. Nothing more exotic. The work
is in doing it carefully.

### Why It Is Written By Hand

Warp can differentiate kernels automatically. But automatic differentiation of a loop whose trip
count varies per thread requires storing the whole forward trace, which at this scale does not
fit. So the kernel is declared

```python
@wp.kernel(enable_backward=False)
```

and you write the adjoint yourself as an ordinary forward loop. This is the standard approach in
production 3D Gaussian Splatting implementations, for the same reason.

You do get help for the hard part. `alpha_at_pixel` is a `@wp.func`, so

```python
wp.grad(alpha_at_pixel)(...)
```

gives you the derivative of alpha with respect to each of its arguments — mean, log-scale,
quaternion, opacity logit — without you differentiating the projection, the covariance, or the
quaternion-to-matrix conversion by hand. Two consequences to plan for: it returns one value per
argument of `alpha_at_pixel`, in order, and a Warp kernel cannot star-unpack a tuple, so every
returned adjoint needs a name even where you do not use it.

### How To Think About It

Two quantities need adjoints.

**Colour.** The i-th splat contributes `T_i * α_i * c_i` to the pixel. Differentiating with
respect to `c_i` is immediate. One wrinkle: `colour_at_view` clamps each channel to [0, 1], and a
channel sitting exactly at a clamp has zero derivative — pushing it further does nothing to the
image, so it should receive no gradient.

**Alpha.** Harder, because `α_i` appears twice in the recurrence. Raising it adds more of splat
i's own colour, *and* it lowers `T_{i+1}`, which dims everything behind splat i. So the derivative
has two terms of opposite sign. The second term needs to know the total contribution of all splats
behind i, which you have not computed at the time you visit i in a front-to-back walk.

There are two ways to get it: a second pass, or algebra. The kernel gives you `image[thread]`, the
finished pixel, and you can track the running prefix sum of contributions so far. Those two,
together with the current splat's own contribution and the transmittance, are enough to recover
what remains behind — which is why the TODO leaves you a commented `remaining_rgb` line as a hint.
Deriving that expression is part of the exercise.

Once you have the alpha adjoint, scale each parameter adjoint from `wp.grad` by it, and accumulate.

### Accumulation

Many pixels touch the same splat at the same time, so writes must be atomic:

```python
wp.atomic_add(mean_grad_flat, splat * 3 + 0, ...)
```

The buffers are flat and component-major: splat `i`'s mean occupies indices `3i, 3i+1, 3i+2`, and
its quaternion `4i … 4i+3`. They are flat because `wp.atomic_add` operates on scalars.

`pack_vec3_gradient` later repacks them into `vec3` arrays for Adam.

### Two Kernels, One Derivation

`render_sparse_backward` is the same algebra. What differs:

- One thread owns one *sampled* pixel, not one dense pixel, so addressing goes through
  `sample_xy[thread]`.
- The walk stops at `last_contributor[thread]`, the entry where the sparse forward pass stopped,
  rather than at the end of the tile's list.

They must agree, and you can check that yourself. Set `sparse.samples_per_tile` to `256`
(`TILE * TILE`), which makes the sampler fall back to the exact raster grid, so every pixel is
sampled exactly once and the sparse path is doing the dense path's work. The loss and every
gradient buffer should then match a dense run to floating-point noise. If they do not, one of the
two kernels is wrong.

## The Sparse Path

Rendering every pixel every iteration is wasteful when a stochastic estimate will do.
`sparse.enabled` turns on sampling: `samples_per_tile` pixels are drawn per 16×16 tile.

```python
def _draw_sparse_samples(self, samples_per_tile=None, rng=None) -> np.ndarray:
```

Sampling is done on the host with NumPy. That sounds slow, but it is cheap next to a kernel launch
and it buys an exact property: sample `s` belongs to tile `s // samples_per_tile` by construction,
so no sort is needed to group samples by tile.

One special case matters. When `samples_per_tile == TILE * TILE`, the offsets are the exact raster
grid instead of a random draw, so every pixel is sampled exactly once and the sparse path reduces
to the dense one. The gradient checker depends on this, so do not make the full-coverage case
random.

`gather_sample_targets` pulls the target colours at the sampled pixels; `sparse_mse_pixel_gradient`
is the MSE derivative over samples rather than pixels.

## Optimizers

`SplatOptimizers` owns Adam state and the gradient buffers your kernels write into.

Adam is used rather than plain gradient descent because the parameter groups have wildly different
scales — a position moves in world units, a logit in unbounded log-odds. Adam normalises each
parameter by a running estimate of its own gradient magnitude, so one learning rate per group is
workable.

The quaternion gets a hand-written step:

```python
@wp.kernel
def adam_quaternion_step(...)
```

because `warp.optim.Adam` does not accept `vec4` parameters. It keeps its own moment buffers and
step counter and renormalises to unit length afterwards, since only unit quaternions are rotations.

`position_learning_rate` decays log-linearly from `position_initial` to `position_final`:

```python
def exponential_learning_rate(step, initial, final, max_steps, delay_steps, delay_multiplier):
```

Positions need a high rate early, while the point cloud is still finding the object's shape, and a
low one later, when moving a splat mostly disturbs a converged neighbourhood.

After every step, `constrain_parameters` clamps log-scales into
`[SCALE_CLAMP_MIN_FRACTION, SCALE_CLAMP_MAX_FRACTION]` of the scene radius and colours into
[0, 1].

## One Training Step

`WarpImageTrainer.step`:

```python
loss = self.loss_and_gradients(view_ids, download_loss)
self.last_position_learning_rate = self.position_learning_rate(iteration)
self.optimizers.apply(self.last_position_learning_rate, self.capacity)
```

and `loss_and_gradients` dispatches to the dense or sparse path. Both do the same five things:
build tiles, zero the gradient buffers, run the forward kernel, compute the pixel gradient, run
your backward kernel, then pack gradients for Adam.

`download_loss` exists because reading the loss back to the host forces a device synchronisation.
On iterations that report nothing, it is skipped.

## Densification and Pruning

Gradient descent alone cannot add detail where there are no splats, or remove splats that have
become useless. `densify_and_prune` does both, periodically.

**Prune** removes splats whose opacity has fallen below a floor, and — when multi-view scoring is
enabled — splats that score badly across several views.

**Densify** picks parents by gradient magnitude:

```python
scores = np.linalg.norm(gradients, axis=1)
parents = [i for i in np.argsort(scores)[::-1] if self.active[i]]
```

A large position gradient means the renderer wants that splat in several places at once, which is
the signal that one splat is being asked to represent more detail than it can. Such splats are
either **split** (if larger than `percent_dense * scene_radius`) or **cloned** (if smaller) — the
two distinct reference operations.

Note the comment about parent selection: a pruned splat must not remain a clone source, because
its slot is now free and could be overwritten as a clone destination in the same pass.

Capacity is fixed. `self.active` is a boolean mask over `capacity` slots, and freed slots are
reused. This avoids reallocating device buffers mid-run.

## Knowing When To Stop

Three small classes, each solving a specific failure.

`ConvergenceTracker` stops training when `fixed_eval` — the held-out loss described above —
has plateaued within `min_delta` for `patience` consecutive checks. Any worsening resets
the patience counter, giving the model time to recover from a temporary regression or an objective
change. Stopping on a held-out reading rather than a training one is the point: a training loss can
still be falling while the model is only memorising, and a run stopped on that signal trains long
past any real gain. Note `rebaseline`, which restarts the comparison from a given reading rather
than treating it as a failure to improve.

`OpacityResetWindow` exists because opacity reset deliberately caps every splat's opacity, so the
next validation reading measures a model that was just knocked down. Without shielding, the
tracker reads that as a plateau and stops a run that is still improving. The window suspends
convergence bookkeeping from a reset until the loss recovers to its pre-reset value, then
rebaselines.

`GrowthPlateauTracker` watches the active splat count. When growth flattens, the population has
settled and late pruning is armed.

## `run_training` and `main`

`run_training` is the loop: sample a camera, step, periodically densify, evaluate, snapshot,
check convergence. `main` parses the config, loads both view sets, builds the trainer over the
combined arrays, and calls `run_training` with the training views alone — which is what keeps the
held-out cameras out of the optimiser.

```bash
python 3dgs_trainer.py config/3dgs_training_gpu.yaml
```

## Checking Your Work

Start with the gradient checkers. They are far more informative than watching a loss curve:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
```

It perturbs one parameter at a time and compares your analytic gradient against a finite
difference. Agreement to three or four significant figures is what you want; finite differences
are themselves approximate, so exact agreement is not expected.

Before you write anything it fails like this, which is what an unimplemented kernel looks like:

```
 ACTUAL: array(0.)
 DESIRED: array(0.061467)
```

Your analytic gradient is zero because nothing accumulates into the buffer yet. The finite
difference is not, because perturbing the parameter really does change the image.

For the sparse kernel, use the full-coverage equivalence described above.

**Before you write anything**, run the trainer as shipped. Every gradient buffer is zero, so Adam
applies nothing and no splat parameter moves. Training still runs to completion, and the active
splat count still grows, because densification is driven separately — so `fixed_eval` will drift a
little. What it will not do is improve. That is your baseline.

A useful debugging order:

1. Colour gradients only. Colour is the simplest adjoint and moves the image visibly.
2. Opacity.
3. Mean, log-scale and quaternion, all of which come from `wp.grad(alpha_at_pixel)`.

If the finite-difference check passes for colour but fails for the geometry parameters, the error
is almost certainly in the alpha adjoint rather than in the projection — `wp.grad` handles the
projection for you.
