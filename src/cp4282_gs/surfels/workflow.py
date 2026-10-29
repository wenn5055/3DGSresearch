"""Shared orchestration for two independently runnable surfel research passes."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp
import yaml

from gaussian_set import GaussianSet

from .config import SurfelTrainingConfig
from .geometry import base
from .trainer import GaussianRGBControl, SurfelTrainer


def train(config_path):
    config = SurfelTrainingConfig.load(Path(config_path))
    print(yaml.safe_dump(config.printable(), sort_keys=False), flush=True)
    wp.init()
    model, paths = config['model'], config['paths']
    device = 'cpu' if config['runtime']['arch'] == 'cpu' else 'cuda:0'
    if device != 'cpu' and not wp.is_cuda_available():
        raise RuntimeError('The surfel GPU pass requires CUDA.')
    background = np.ones(3, np.float32) if model['background'] == 'white' else np.zeros(3, np.float32)
    targets, cameras, focal, distance, _ = base.load_views(paths['data'], model['width'], model['height'], background)
    all_targets, all_cameras, eval_ids, frame_ids = base.resolve_evaluation_views(
        targets, cameras, paths['data'], background, config['reporting'])
    initial = GaussianSet.from_ply(paths['init_ply']) if paths['init_ply'] else None
    if initial is not None and len(initial.means) > model['capacity']:
        raise ValueError('Initial PLY exceeds model.capacity.')
    trainer_class = GaussianRGBControl if model['geometry'] == 'gaussian' else SurfelTrainer
    trainer = trainer_class(all_targets, all_cameras, focal, model['scene_radius'] or distance * 0.35,
        model['capacity'], model['initial_splats'], device, config['runtime']['seed'],
        init_scene=initial, background=background, learning_rates=config['learning_rates'],
        tile_pair_capacity=model['tile_pair_capacity'], surfel_options=config['surfel'],
        training_view_count=len(targets),
        training_options={key: config[key] for key in ('loss', 'sparse', 'compact_box', 'multiview_adc')})
    if paths['init_ply']:
        sidecar = paths['init_ply'].with_suffix('.surfel.npz')
        if sidecar.exists() and isinstance(trainer, SurfelTrainer):
            trainer.load_materials(sidecar, len(initial.means))
            print('Warm start: geometry/materials loaded; optimizer state starts fresh.', flush=True)
        elif sidecar.exists():
            raise ValueError('Initialize the Gaussian control from a Gaussian PLY, not a flattened-surfel checkpoint.')
    run_training(trainer, config, len(targets), eval_ids, frame_ids)


def run_training(trainer, config, training_views, eval_ids, frame_ids=None):
    reporting, convergence, adaptive = config['reporting'], config['convergence'], config['adaptive']
    density, adc, settings = config['densification'], config['multiview_adc'], config['surfel']
    output = Path(config['paths']['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = output.parent / (output.stem + '_snapshots')
    snapshots.mkdir(exist_ok=True)
    print(f"Geometry: {config['model']['geometry']}; training-only initialization statistics; dense rendering.", flush=True)
    print(base.describe_evaluation_views(eval_ids, frame_ids, reporting['eval_manifest']), flush=True)
    want_deferred = settings['mode'] == 'deferred'
    print(f"Surfel pass: {settings['mode']}; deferred start iteration: {settings['specular_start_iteration']}", flush=True)
    loss_curriculum = config['loss']['mode'] == 'mse_dssim'
    loss_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
    eval_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
    mse_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
    objective, mse, _ = trainer.evaluate_components(eval_ids)
    eval_ema.update(objective)
    mse_ema.update(mse)
    tracker = None if loss_curriculum or (want_deferred and not trainer.specular_active) else base.ConvergenceTracker(convergence, objective)
    growth = base.GrowthPlateauTracker(adaptive['late_prune_growth_window'], adaptive['late_prune_growth_percent'])
    reset_window = base.OpacityResetWindow()
    late_prune_armed = False
    stage_start = 0
    rng = np.random.default_rng(config['runtime']['seed'])

    def snapshot(path):
        base.save_rendered_image(trainer, path.with_suffix('.png'))
        if reporting['save_ply']:
            trainer.gaussian_set().to_ply(path.with_suffix('.ply'))
        trainer.save_sidecars(path)

    snapshot(snapshots / 'out_000000.png')
    started = perf_counter()
    completed = 0
    for iteration in range(1, config['training']['iterations'] + 1):
        completed = iteration
        if want_deferred and not trainer.specular_active and iteration >= settings['specular_start_iteration']:
            trainer.activate_specular()
            stage_start = iteration
            tracker = None
            loss_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
            objective, _, _ = trainer.evaluate_components(eval_ids)
            eval_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
            eval_ema.update(objective)
            reset_window = base.OpacityResetWindow()
            print(f'{iteration:6d}: advancing surfel stage rgb -> deferred (scheduled geometry warm-up complete).', flush=True)
        densify = bool(density['interval'] and iteration % density['interval'] == 0)
        late = adc['enabled'] and late_prune_armed and iteration % adc['late_prune_interval'] == 0
        evaluate = iteration % reporting['eval_every'] == 0
        save = iteration % reporting['snapshot_every'] == 0
        log = iteration % reporting['log_every'] == 0
        loss, gradients = trainer.step([rng.integers(training_views)], download_gradients=densify or late,
                                      download_loss=log or evaluate or save or loss_ema.value is None, iteration=iteration)
        if loss is not None:
            loss_ema.update(loss)
        if densify or late:
            views = rng.choice(training_views, min(adc['views'], training_views), replace=False) if adc['enabled'] else None
            trainer.densify_and_prune(gradients, density['fraction'] if densify else 0.0,
                                      view_ids=views, late_prune=late, percent_dense=density['percent_dense'])
            if densify and not late_prune_armed and growth.update(int(trainer.active.sum())):
                late_prune_armed = True
        if density['opacity_reset_interval'] and iteration % density['opacity_reset_interval'] == 0:
            reset_window.warn_if_unrecovered(iteration)
            reset_window.open(eval_ema.value, iteration)
            trainer.reset_opacity()
        if evaluate or save:
            objective, mse, _ = trainer.evaluate_components(eval_ids)
            eval_ema.update(objective)
            mse_ema.update(mse)
            if evaluate and reset_window.accepts(eval_ema.value, iteration):
                if loss_curriculum and trainer.dssim_stage == 0:
                    if mse_ema.value <= 10 ** (-adaptive['dssim_psnr_threshold'] / 10):
                        trainer.advance_dssim_stage()
                        stage_start = iteration
                        objective, _, _ = trainer.evaluate_components(eval_ids)
                        eval_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
                        eval_ema.update(objective)
                        reset_window = base.OpacityResetWindow()
                        tracker = (base.ConvergenceTracker(convergence, float('inf'))
                                   if not want_deferred or trainer.specular_active else None)
                        loss_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
                        loss_ema.update(loss if loss is not None else objective)
                        print(f'{iteration:6d}: advancing loss stage mse -> dssim on held-out MSE EMA.', flush=True)
                elif not want_deferred or trainer.specular_active:
                    if tracker is None:
                        tracker = base.ConvergenceTracker(convergence, float('inf'))
                    if reset_window.resumed_this_tick:
                        tracker.rebaseline(eval_ema.value)
                    else:
                        tracker.update(eval_ema.value, iteration - stage_start)
        if log:
            print(f'{iteration:6d}: ema={loss_ema.value:.6f}, loss_phase={trainer.last_loss_phase}, '
                  f'stage={trainer.loss_stage}, surfel_stage={trainer.surfel_stage}, '
                  f'fixed_eval_ema={eval_ema.value:.6f}, fixed_eval_mse_ema={mse_ema.value:.6f}, '
                  f'active={trainer.active.sum()}, pairs={trainer.last_pair_count}, '
                  f'lr_xyz={trainer.last_position_learning_rate:.8f}', flush=True)
        if save:
            snapshot(snapshots / f'out_{iteration:06d}.png')
        if tracker is not None and tracker.should_stop(iteration - stage_start):
            print(f'Stopping at iteration {iteration}: fixed validation objective plateaued.', flush=True)
            break
    trainer.evaluate(eval_ids)
    snapshot(output)
    print(f'Final loss stage: {trainer.loss_stage}', flush=True)
    print(f'Final surfel stage: {trainer.surfel_stage}', flush=True)
    if loss_curriculum and trainer.dssim_stage == 0:
        print('D-SSIM was not activated: the validation PSNR gate was not reached.', flush=True)
    if want_deferred and not trainer.specular_active:
        print('Deferred shading was not activated: the run ended during geometry warm-up.', flush=True)
    elapsed = perf_counter() - started
    print(f'Training finished after {completed:,} iterations in {elapsed:.2f}s ({elapsed / max(1, completed):.3f}s/iter).', flush=True)
