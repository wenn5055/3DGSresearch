"""Finite-difference checks away from visibility/cutoff discontinuities."""

import importlib
from pathlib import Path

import numpy as np
import warp as wp

from gaussian_set import GaussianSet
from .trainer import SurfelTrainer

reference = importlib.import_module('3dgs_gradient_check_gpu')


def make_trainer(device='cpu', mode='rgb', blended=False, overlapping=False, capacity=None, geometry=False, shading_model='ldr'):
    initial = reference.one_splat_scene()
    if overlapping:
        initial = GaussianSet(
            np.array([[0.05, 0.02, 1.2], [-0.1, 0.03, 1.5]], np.float32),
            np.array([[0.4, 0.5, 0.2], [0.5, 0.35, 0.4]], np.float32),
            np.array([[0.96, 0.1, 0.2, 0.15], [0.95, 0.1, -0.15, 0.2]], np.float32),
            np.array([0.4, 0.55], np.float32), np.array([[0.4, 0.5, 0.6], [0.7, 0.3, 0.2]], np.float32))
    count = len(initial.means)
    targets = np.broadcast_to(np.array([0.2, 0.3, 0.4], np.float32), (1, 4, 4, 3)).copy()
    geometry_kwargs = {}
    if geometry:
        geometry_kwargs = {'training_view_count': 1, 'training_alpha': np.full((1, 4, 4), 0.9, np.float32),
                           'geometry_options': {'mask_weight': 0.1, 'normal_depth_weight': 0.03,
                                                'normal_start_iteration': 0, 'normal_ramp_iterations': 0,
                                                'foreground_threshold': 0.8, 'max_relative_depth_jump': 0.5}}
    trainer = SurfelTrainer(targets, np.eye(4, dtype=np.float32)[None], 4., 1., capacity or count, count,
        device, 7, init_scene=initial, training_options={'compact_box': {'enabled': False},
            'loss': {'mode': 'mse_dssim' if blended else 'mse', 'dssim_window': 3}},
        surfel_options={'mode': mode, 'environment_width': 8, 'environment_height': 4,
                        'shading_model': shading_model,
                        'specular_start_iteration': 0, 'initial_reflection': 0.3, 'save_debug': False}, **geometry_kwargs)
    if mode == 'deferred':
        u, v = np.meshgrid(np.arange(8) * 2 * np.pi / 8, np.linspace(-1, 1, 4))
        environment = np.stack([1.2 * np.sin(u) + 0.4 * v, -0.6 * np.cos(u) + 0.3 * v,
                                0.8 * np.sin(u + 0.5) - 0.4 * v], axis=-1)
        trainer.environment.assign(environment.astype(np.float32).reshape(-1, 3))
        if shading_model == 'hdr_angular':
            trainer.angular_gain.assign(np.linspace(-.8, .4, 16, dtype=np.float32))
    if blended:
        trainer.advance_dssim_stage()
    return trainer


def check_gradients(device='cpu', mode='rgb', blended=False, overlapping=False, verbose=True, geometry=False, shading_model='ldr'):
    trainer = make_trainer(device, mode, blended, overlapping, geometry=geometry, shading_model=shading_model)
    trainer.loss_and_gradients([0])
    if geometry and trainer.last_geometry_terms['normal_pixels'] == 0:
        raise AssertionError('Geometry gradient checks must exercise valid normal-depth pixels.')
    qaxis = int(np.argmax(np.abs(trainer.optimizers.quaternion_grad_flat.numpy()[:4])))
    cases = [
        ('mean x', trainer.means, wp.vec3, (0, 0), float(trainer.optimizers.mean_grad.numpy()[0, 0]), 0.002),
        ('mean z', trainer.means, wp.vec3, (0, 2), float(trainer.optimizers.mean_grad.numpy()[0, 2]), 0.002),
        ('tangent scale x', trainer.log_scales, wp.vec3, (0, 0), float(trainer.optimizers.scale_grad.numpy()[0, 0]), 0.002),
        ('tangent scale y', trainer.log_scales, wp.vec3, (0, 1), float(trainer.optimizers.scale_grad.numpy()[0, 1]), 0.002),
        ('unused normal scale', trainer.log_scales, wp.vec3, (0, 2), float(trainer.optimizers.scale_grad.numpy()[0, 2]), 0.002),
        ('quaternion', trainer.quaternions, wp.vec4, (0, qaxis), float(trainer.optimizers.quaternion_grad_flat.numpy()[qaxis]), 0.002),
        ('opacity', trainer.opacity, wp.float32, 0, float(trainer.optimizers.opacity_grad.numpy()[0]), 0.002),
        ('base green', trainer.color, wp.vec3, (0, 1), float(trainer.optimizers.color_grad.numpy()[0, 1]), 0.002),
    ]
    if mode == 'deferred':
        env = trainer.environment.grad.numpy()
        index = np.unravel_index(np.argmax(np.abs(env)), env.shape)
        cases += [('reflection logit', trainer.reflection_logits, wp.float32, 0, float(trainer.reflection_gradient.numpy()[0]), 0.01),
                  ('environment logit', trainer.environment, wp.vec3, index, float(env[index]), 0.01)]
        if shading_model == 'hdr_angular':
            gain = trainer.angular_gain.grad.numpy()
            index = int(np.argmax(np.abs(gain)))
            if np.max(np.abs(gain)) < 1e-7:
                raise AssertionError('Angular response gradient must be exercised.')
            cases += [('angular gain', trainer.angular_gain, wp.float32, index, float(gain[index]), 0.01)]
    results = []
    for label, field, dtype, index, analytic, epsilon in cases:
        numeric = reference.finite_difference(trainer, field, dtype, index, epsilon)
        tolerance = max(3e-5 if label == 'environment logit' else 3e-4, abs(numeric) * 0.03)
        if verbose:
            print(f'{mode}/{label:22s} analytic={analytic: .7f} numeric={numeric: .7f}', flush=True)
        if not np.isfinite(analytic) or abs(analytic - numeric) > tolerance:
            raise AssertionError(f'{mode}/{label}: analytic={analytic}, finite difference={numeric}, tolerance={tolerance}')
        if label == 'unused normal scale' and (analytic != 0 or numeric != 0):
            raise AssertionError('The removed normal-axis scale must have exactly zero derivative.')
        results.append((label, analytic, numeric))
    return results


def run(device='cpu'):
    wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[3] / 'outputs/surfel-test-warp-cache')
    wp.config.use_precompiled_headers = False
    wp.init()
    check_gradients(device, 'rgb')
    check_gradients(device, 'deferred', overlapping=True)
    check_gradients(device, 'deferred', blended=True)
    check_gradients(device, 'rgb', overlapping=True, geometry=True)
    check_gradients(device, 'deferred', blended=True, overlapping=True, geometry=True)
    print(f'All surfel geometry/material/environment gradient checks passed on {device}.', flush=True)
