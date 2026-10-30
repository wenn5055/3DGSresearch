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
from .geometry_losses import load_training_alpha


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
    geometry = config['geometry_loss']
    extra = {}
    if model['geometry'] == 'surfel':
        extra['geometry_options'] = geometry
    if geometry['mask_weight'] or geometry['normal_depth_weight'] or config['surfel']['foreground_pruning']['enabled']:
        extra['training_alpha'] = load_training_alpha(paths['data'], model['width'], model['height'])
    trainer = trainer_class(all_targets, all_cameras, focal, model['scene_radius'] or distance * 0.35,
        model['capacity'], model['initial_splats'], device, config['runtime']['seed'],
        init_scene=initial, background=background, learning_rates=config['learning_rates'],
        tile_pair_capacity=model['tile_pair_capacity'], surfel_options=config['surfel'],
        training_view_count=len(targets),
        sh_degree=model['sh_degree'],
        child_optimizer_state=config['densification']['child_optimizer_state'],
        training_options={key: config[key] for key in ('loss', 'sparse', 'compact_box', 'multiview_adc')}, **extra)
    if paths['init_ply']:
        sidecar = paths['init_ply'].with_suffix('.surfel.npz')
        if sidecar.exists() and isinstance(trainer, SurfelTrainer):
            trainer.load_materials(sidecar, len(initial.means))
            print('Warm start: geometry/materials loaded; optimizer state starts fresh.', flush=True)
        elif sidecar.exists():
            raise ValueError('Initialize the Gaussian control from a Gaussian PLY, not a flattened-surfel checkpoint.')
        elif paths['init_ply'].with_suffix('.sh.npz').exists():
            trainer.load_appearance(paths['init_ply'].with_suffix('.sh.npz'), len(initial.means))
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
    print(f"Density child optimizer state: {trainer.optimizers.child_state_policy}", flush=True)
    want_deferred = settings['mode'] == 'deferred'
    target_sh_degree = config['model']['sh_degree']
    if target_sh_degree:
        print(f"Late base SH: maximum degree {target_sh_degree}, first band at {settings['sh_start_iteration']}, "
              f"band interval {settings['sh_band_interval']}.", flush=True)
    print(f"Surfel pass: {settings['mode']}; deferred start iteration: {settings['specular_start_iteration']}", flush=True)
    if config['model']['geometry'] == 'surfel':
        print(f"Surfel depth mode: {settings['depth_mode']}", flush=True)
        print(f"Deferred shading model: {settings['shading_model']}", flush=True)
    geometry = config['geometry_loss']
    if geometry['mask_weight'] or geometry['normal_depth_weight']:
        print(f"Geometry losses (training alpha only): mask={geometry['mask_weight']}, "
              f"normal_depth={geometry['normal_depth_weight']}, start={geometry['normal_start_iteration']}, "
              f"ramp={geometry['normal_ramp_iterations']}; validation gates remain photometric.", flush=True)
    loss_curriculum = config['loss']['mode'] == 'mse_dssim' and not getattr(trainer, 'uses_paper_loss', False)
    propagation = settings['normal_propagation']
    foreground = settings['foreground_pruning']
    if foreground['enabled']:
        print(f"Foreground/background pruning: training alpha only; start={foreground['start_iteration']}, "
              f"interval={foreground['interval']}, min_background_views={foreground['min_background_views']}, "
              f"persistence={foreground['persistence']}, training_view_ids={trainer.foreground_pruner.views}. "
              'Continues during quiet refinement; out-of-frame/occluded supports are not background.', flush=True)
    refinement = settings['refinement']
    if propagation['enabled']:
        print('Paper normal optimization: image/reflection gradients, opacity/scale propagation and colour noise; '
              f"normal-depth weight={geometry['normal_depth_weight']}.", flush=True)
    if refinement['start_iteration'] is not None:
        print(f"Quiet refinement from iteration {refinement['start_iteration']}: growth, opacity resets and "
              f"propagation off; ADC/opacity prune interval={refinement['prune_interval']} "
              '(0 disables scheduled ADC/opacity pruning).', flush=True)
        if foreground['enabled']:
            print('Foreground contradiction pruning remains active on its own interval during refinement.', flush=True)
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
    # Structural schedules must not change the sequence of training cameras.
    structure_rng = np.random.default_rng(np.random.SeedSequence([config['runtime']['seed'], 0xADC]))
    print('Sampling: independent training-camera and ADC-view random streams.', flush=True)

    def snapshot(path):
        base.save_rendered_image(trainer, path.with_suffix('.png'))
        if reporting['save_ply']:
            trainer.gaussian_set().to_ply(path.with_suffix('.ply'))
        trainer.save_sidecars(path)

    snapshot(snapshots / 'out_000000.png')
    started = perf_counter()
    completed = 0
    geometry_ready_before = not bool(geometry['normal_depth_weight'])
    for iteration in range(1, config['training']['iterations'] + 1):
        completed = iteration
        if (trainer.active_sh_degree < target_sh_degree and iteration >=
                settings['sh_start_iteration'] + trainer.active_sh_degree * settings['sh_band_interval']):
            trainer.advance_sh_stage()
            tracker = None
            stage_start = iteration
            loss_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
            objective, _, _ = trainer.evaluate_components(eval_ids)
            eval_ema = base.ExponentialMovingAverage(reporting['loss_ema_decay'])
            eval_ema.update(objective)
            reset_window = base.OpacityResetWindow()
            print(f'{iteration:6d}: unlocking late base SH degree {trainer.active_sh_degree}.', flush=True)
        refining = refinement['start_iteration'] is not None and iteration >= refinement['start_iteration']
        if refining and iteration == refinement['start_iteration']:
            trainer._opacity_frozen_until = 0
            tracker = None
            stage_start = iteration
            reset_window = base.OpacityResetWindow()
            print(f'{iteration:6d}: entering quiet refinement.', flush=True)
        geometry_ready = (not geometry['normal_depth_weight'] or iteration >=
                          geometry['normal_start_iteration'] + geometry['normal_ramp_iterations'])
        if geometry_ready and not geometry_ready_before:
            tracker = None
            stage_start = iteration
            print(f'{iteration:6d}: normal-depth regularization ramp complete.', flush=True)
        if not geometry_ready:
            tracker = None
        geometry_ready_before = geometry_ready
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
        propagating = (not refining and propagation['enabled'] and trainer.specular_active
                       and iteration <= propagation['until_iteration'])
        interval = propagation['densification_interval'] if propagating and density['interval'] else density['interval']
        can_change_structure = iteration < config['training']['iterations']
        densify = bool(not refining and can_change_structure and interval and iteration % interval == 0)
        late = bool(can_change_structure and (
            (refining and refinement['prune_interval'] and iteration % refinement['prune_interval'] == 0)
            or (not refining and adc['enabled'] and late_prune_armed and iteration % adc['late_prune_interval'] == 0)))
        evaluate = iteration % reporting['eval_every'] == 0
        save = iteration % reporting['snapshot_every'] == 0
        log = iteration % reporting['log_every'] == 0
        loss, gradients = trainer.step([rng.integers(training_views)], download_gradients=densify or late,
                                      download_loss=log or evaluate or save or loss_ema.value is None, iteration=iteration)
        if loss is not None:
            loss_ema.update(loss)
        if densify or late:
            views = structure_rng.choice(training_views, min(adc['views'], training_views), replace=False) if adc['enabled'] else None
            active_before = int(trainer.active.sum())
            children = trainer.densify_and_prune(gradients, density['fraction'] if densify else 0.0,
                                      view_ids=views, late_prune=late, percent_dense=density['percent_dense'])
            active_after = int(trainer.active.sum())
            print(f'{iteration:6d}: density: active_before={active_before}, '
                  f'pruned={active_before + len(children) - active_after}, grown={len(children)}, '
                  f'active_after={active_after}, late_prune={late}.', flush=True)
        if (foreground['enabled'] and can_change_structure and iteration >= foreground['start_iteration']
                and iteration % foreground['interval'] == 0):
            event = trainer.prune_foreground()
            if event['foreground_pruned']:
                reset_window.warn_if_unrecovered(iteration)
                reset_window.open(eval_ema.value, iteration)
                tracker = None
                stage_start = iteration
            print(f"{iteration:6d}: foreground pruning: candidates={event['background_candidates']}, "
                  f"pruned={event['foreground_pruned']}, unobserved={event['unobserved']}, "
                  f"max_background_votes={event['max_background_votes']}, "
                  f"active_before={event['active_before']}, active_after={event['active_after']}.", flush=True)
        # Include every deletion at this event in the net-growth history.
        if densify and not late_prune_armed and growth.update(int(trainer.active.sum())):
            late_prune_armed = True
        # Leave the final trained model intact; a last-step clamp would save holes
        # before opacity has any opportunity to recover.
        opacity_reset = bool(not refining and density['opacity_reset_interval']
                             and iteration < config['training']['iterations']
                             and iteration % density['opacity_reset_interval'] == 0)
        if opacity_reset:
            reset_window.warn_if_unrecovered(iteration)
            reset_window.open(eval_ema.value, iteration)
            trainer.reset_opacity()
        if propagation['enabled'] and not refining:
            event = trainer.propagate_normals(iteration, opacity_reset=opacity_reset)
            if event:
                reset_window.warn_if_unrecovered(iteration)
                reset_window.open(eval_ema.value, iteration)
                print(f"{iteration:6d}: normal propagation: expanded={event['expanded']}, "
                      f"colours_reset={event['colours_reset']}; opacity floor={propagation['opacity_floor']}.", flush=True)
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
            geometry_log = ''
            if getattr(trainer, 'geometry_loss', None) is not None:
                terms = trainer.last_geometry_terms
                geometry_log = (f", train_photo={trainer.last_photometric_loss:.6f}, mask={terms['mask']:.6f}, "
                                f"normal_depth={terms['normal_depth']:.6f}, normal_weight={terms['normal_weight']:.5f}, "
                                f"normal_pixels={terms['normal_pixels']}")
            print(f'{iteration:6d}: ema={loss_ema.value:.6f}, loss_phase={trainer.last_loss_phase}, '
                  f'stage={trainer.loss_stage}, surfel_stage={trainer.surfel_stage}, '
                  f'sh_stage=sh{trainer.active_sh_degree}, '
                  f'fixed_eval_ema={eval_ema.value:.6f}, fixed_eval_mse_ema={mse_ema.value:.6f}, '
                  f'active={trainer.active.sum()}, pairs={trainer.last_pair_count}, '
                  f'lr_xyz={trainer.last_position_learning_rate:.8f}{geometry_log}', flush=True)
        if save:
            snapshot(snapshots / f'out_{iteration:06d}.png')
        if (geometry_ready and trainer.active_sh_degree == target_sh_degree and tracker is not None
                and tracker.should_stop(iteration - stage_start)):
            print(f'Stopping at iteration {iteration}: fixed validation objective plateaued.', flush=True)
            break
    trainer.evaluate(eval_ids)
    snapshot(output)
    print(f'Final loss stage: {trainer.loss_stage}', flush=True)
    print(f'Final surfel stage: {trainer.surfel_stage}', flush=True)
    print(f'Final SH degree: {trainer.active_sh_degree} (configured maximum: {target_sh_degree})', flush=True)
    if loss_curriculum and trainer.dssim_stage == 0:
        print('D-SSIM was not activated: the validation PSNR gate was not reached.', flush=True)
    if want_deferred and not trainer.specular_active:
        print('Deferred shading was not activated: the run ended during geometry warm-up.', flush=True)
    elapsed = perf_counter() - started
    print(f'Training finished after {completed:,} iterations in {elapsed:.2f}s ({elapsed / max(1, completed):.3f}s/iter).', flush=True)
