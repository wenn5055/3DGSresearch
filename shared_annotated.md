<!-- ARCHIVE CANDIDATE (2026-10-29): Teaching-era shared-module map; predates D-SSIM, SH and surfel extensions. Successors: shared/ source docstrings and docs/project-passes.md. Archive this guide only; shared/ remains active. -->

# Annotated Walkthrough: the `shared/` package

`shared/` holds the code used by more than one program: the Assignment 1 reference renderer and the
Assignment 2 trainer. Nothing in it is a renderer or a trainer itself — it is the vocabulary they
both speak.

This document is a **map plus reference**. Most modules are short enough to read directly, so the
table points you at the source; `trainable_gaussian.py` is the one that needs explaining, and §1
covers it in full.

| Module | Used by | Where to read it |
|---|---|---|
| `camera.py` | renderer | [`shared/camera.py`](shared/camera.py) |
| `gaussian_set.py` | renderer, trainer, evaluator | [`shared/gaussian_set.py`](shared/gaussian_set.py) |
| `profiling.py` | trainer | [`shared/profiling.py`](shared/profiling.py) |
| `projected_gaussians.py` | renderer | [`shared/projected_gaussians.py`](shared/projected_gaussians.py) |
| `splat_math.py` | renderer, trainer, config schema | [`shared/splat_math.py`](shared/splat_math.py) |
| `tile_builder.py` | trainer | [`shared/tile_builder.py`](shared/tile_builder.py) |
| `trainable_gaussian.py` | trainers | §1 below |
| `training_config.py` | trainer | [`shared/training_config.py`](shared/training_config.py) |

`trainable_gaussian.py` is the one to read for Assignment 2: it is where every parameter your
backward pass writes a gradient for actually lives. `training_config.py` is worth skimming too — it
is the authoritative list of every YAML key the trainer accepts, with the defaults.

---

## 0. Why a shared package exists at all

The obvious reason is avoiding duplicate code. The important reason is avoiding **divergent**
duplicate code.

A renderer and a trainer must agree exactly on what a splat means: how a quaternion becomes a
rotation, where a Gaussian is truncated, how opacity is decoded from storage. If they disagree even
slightly, the trainer optimises splats under one set of rules and the renderer draws them under
another, and the picture you get back is not the picture the optimiser was working toward. The
symptom looks like poor model quality, and the cause is a definition that drifted — which is
extremely hard to find by staring at images.

That has happened in this codebase, twice. An earlier version had `v1` truncating splats at a fixed
3σ radius while `v2` and `v3` used the opacity-dependent compact support, so three renderers that
were supposed to be interchangeable quietly produced different images. Separately, a trainer and the
rasteriser used to measure its quality disagreed on the same cutoff, which cost 2.3 dB of reported
PSNR — the models were fine, the measurement was not.

In both cases each rule was individually defensible; having two of them was the bug, and the
symptom pointed somewhere other than the cause. `splat_math.py` exists so there is exactly one place
to state each of these conventions.

---

## 1. `trainable_gaussian.py` — splats a trainer can move

The renderers treat splats as fixed input. A trainer treats them as **variables**, adjusted to make
rendered images match photographs. That changes what the storage needs to do, in two ways this
module handles.

### 1.1 Unconstrained storage

```python
class TrainableGaussianSet:
    """Raw Warp parameters for N anisotropic 3D Gaussians.

    The arrays store unconstrained values where useful: log-scales and opacity logits.
    """
```

Compare with `GaussianSet` from [`shared/gaussian_set.py`](shared/gaussian_set.py), which stores physical
scales and opacities in $[0,1]$. Here the fields are `log_scales` and `opacity_logits` instead.

The reason is the same one that made `.ply` files store these encodings. Gradient descent adds a
step to a parameter and has no notion of "this value must stay positive" or "must stay below 1". If
it optimised a scale directly, one large step could make it negative and the covariance meaningless.
Optimising the **logarithm** means any real value maps back to a positive scale; optimising a
**logit** means any real value maps back into $(0,1)$. The constraint becomes impossible to violate
rather than something to check for.

The renderer applies `exp` and `sigmoid` when it needs physical values — this is why
`GaussianSet.from_ply` decodes exactly those two fields.

### 1.2 Validation up front

```python
        if means.shape != (count, 3) or log_scales.shape != (count, 3):
            raise ValueError("means and log_scales must have shape (N, 3).")
        if quaternions.shape != (count, 4) or colors.shape != (count, 3):
            raise ValueError("quaternions must have shape (N, 4), and colors must have shape (N, 3).")
        if opacity_logits.shape != (count,):
            raise ValueError("opacity_logits must have shape (N,).")
```

Blunt shape checks in the constructor. They look like clutter until you consider the alternative: a
`(N, 3)` array passed where `(N, 4)` was expected does not crash inside a Warp kernel, it silently
reads adjacent memory and produces plausible-looking garbage that only shows up as a model that
trains badly. Failing at construction converts a debugging session into an error message.

```python
        self.means = wp.array(means.astype(np.float32), dtype=wp.vec3,
                              device=device, requires_grad=requires_grad)
```

`requires_grad=True` is the new piece relative to v2's buffers. It tells Warp to allocate a matching
**gradient** array alongside this one, and to record the operations applied to it so derivatives can
be computed later. A renderer never needs this; a trainer needs it on every parameter it intends to
adjust.

### 1.3 `random_init`

```python
        means = rng.uniform(-0.45, 0.45, size=(count, 3)).astype(np.float32)
        means[:, 2] *= 0.3
        log_scales = np.tile(np.log(np.array([0.16, 0.18, 0.14], ...)), (count, 1))
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0], ...), (count, 1))
        opacity_logits = np.full(count, -0.5, dtype=np.float32)
```

A starting scene of random splats, before any training. Worth reading for what the specific choices
say:

- `means[:, 2] *= 0.3` flattens the initial cloud in depth — the scenes these units train on are
  shallow, and starting closer to the answer converges faster.
- `np.tile` repeats one row N times, so every splat starts the same size and unrotated.
- The quaternion `(1, 0, 0, 0)` is the **identity** rotation, the quaternion equivalent of doing
  nothing.
- `opacity_logits = -0.5` gives $\sigma(-0.5) \approx 0.38$: partly transparent, so early gradients
  can see through to splats behind and adjust them, rather than the front layer immediately hiding
  everything.

### 1.4 `sgd_step` and the update kernel

```python
    def sgd_step(self, tape: wp.Tape, position_lr: float = 2.0, ...):
        wp.launch(_sgd_update, dim=self.count,
                  inputs=[self.means, tape.gradients[self.means], ...])
```

`wp.Tape` is Warp's record of which operations touched which arrays, so `tape.gradients[x]` is the
gradient with respect to `x`. One thread per **splat** here, not per pixel — the pattern from v2 §2
applied to a different quantity.

```python
    splat = wp.tid()
    means[splat] = means[splat] - position_lr * mean_grad[splat]
    log_scales[splat] = log_scales[splat] - scale_lr * scale_grad[splat]
    ...
```

Plain gradient descent: step against the gradient. Note the **per-group learning rates** — position
2.0, rotation 0.3, and so on. The parameters have wildly different natural scales, and a single
shared rate large enough to move positions usefully would make rotations oscillate wildly.

```python
    opacity_logits[splat] = wp.clamp(
        opacity_logits[splat] - opacity_lr * opacity_grad[splat], -8.0, 4.0
    )
    ...
    q = quaternions[splat]
    quaternions[splat] = q / wp.sqrt(wp.dot(q, q) + 1.0e-8)
    log_scales[splat] = wp.vec3(wp.clamp(log_scales[splat][0], -4.0, 0.0), ...)
```

The step is followed by housekeeping that the unconstrained encoding cannot handle by itself:

- **Clamping the opacity logit** to $[-8, 4]$. The encoding keeps opacity in $(0,1)$ for any logit,
  but $\sigma(-20)$ is so close to zero that its gradient vanishes and the splat can never recover.
  The clamp keeps parameters in a range where learning still works.
- **Renormalising the quaternion.** This is the repair used by the renderer's quaternion conversion:
  a gradient step moves all four components independently and the result is no longer unit length,
  so the rotation is no longer a rotation. Dividing by the norm restores it. The `+ 1e-8` guards a
  quaternion that has collapsed to zero.
- **Clamping log-scales** to $[-4, 0]$, i.e. physical scales in $[e^{-4}, 1]$. The lower bound stops
  a splat shrinking below a pixel, where it contributes nothing and receives no useful gradient —
  a dead splat that can never come back.

Every one of these lines exists because a raw gradient step can push a parameter somewhere the
model cannot recover from. This is the practical difference between the maths of gradient descent
and a trainer that actually converges.

---

## 2. Reading order, once more

For Assignment 2:

1. §1 of this document — how a trainer stores splats it is allowed to move.
2. [`3dgs_trainer_annotated.md`](3dgs_trainer_annotated.md) — the training pipeline, Warp, and the
   two backward kernels that are your task.
3. [`shared/training_config.py`](shared/training_config.py) — when you want to change a setting and
   need to know what the key is called and what it defaults to.

[`3dgs_renderer_v1.py`](3dgs_renderer_v1.py) is Assignment 1's renderer, kept here only because the
Unit 8 synthetic trainers reuse it. Assignment 2 does not depend on it.
