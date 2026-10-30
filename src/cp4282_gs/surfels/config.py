"""Configuration for the independent fixed-RGB and deferred-reflection surfel passes."""

from copy import deepcopy
import math

from training_config import TrainingConfig

GEOMETRY_DEFAULTS = {
    'mask_weight': 0.0,
    'normal_depth_weight': 0.0,
    'normal_start_iteration': 2000,
    'normal_ramp_iterations': 2000,
    'min_coverage': 0.1,
    'foreground_threshold': 0.95,
    'max_relative_depth_jump': 0.05,
}


class SurfelTrainingConfig(TrainingConfig):
    DEFAULTS = deepcopy(TrainingConfig.DEFAULTS)
    DEFAULTS["model"]["geometry"] = "surfel"
    DEFAULTS["sparse"]["enabled"] = False
    DEFAULTS['geometry_loss'] = deepcopy(GEOMETRY_DEFAULTS)
    DEFAULTS["surfel"] = {
        "mode": "rgb",
        "shading_model": "ldr",
        "angular_lr": 0.001,
        "depth_mode": "centre",
        "specular_start_iteration": 10000,
        "initial_reflection": 0.1,
        "reflection_lr": 0.01,
        "environment_lr": 0.01,
        "environment_width": 128,
        "environment_height": 64,
        "save_debug": True,
        "sh_start_iteration": None,
        "sh_band_interval": 2000,
        "foreground_pruning": {
            "enabled": False,
            "start_iteration": 2000,
            "interval": 500,
            "views": 16,
            "min_background_views": 3,
            "persistence": 2,
            "background_alpha_threshold": 0.01,
            "margin_pixels": 2,
        },
        "photometric_loss": "inherit",
        "refinement": {
            "start_iteration": None,
            "prune_interval": 500,
        },
        "normal_propagation": {
            "enabled": False,
            "interval": 1000,
            "until_iteration": 24000,
            "scale_factor": 1.5,
            "scale_reflection_threshold": 0.02,
            "color_reflection_threshold": 0.1,
            "color_noise": 0.1,
            "opacity_floor": 0.9,
            "reflection_floor": 0.001,
            "opacity_freeze_iterations": 200,
            "densification_interval": 500,
        },
    }

    def validate(self):
        super().validate()
        settings = self.values["surfel"]
        geometry = self.values['model']['geometry']
        if geometry not in ('gaussian', 'surfel'):
            raise ValueError('model.geometry must be gaussian or surfel for project passes.')
        if geometry == 'gaussian' and settings['mode'] != 'rgb':
            raise ValueError('The Gaussian control pass uses surfel.mode: rgb.')
        degree = self.values['model']['sh_degree']
        sh_start = settings['sh_start_iteration']
        sh_interval = settings['sh_band_interval']
        if isinstance(sh_interval, bool) or not isinstance(sh_interval, int) or sh_interval < 1:
            raise ValueError('surfel.sh_band_interval must be a positive integer.')
        if sh_start is not None and (isinstance(sh_start, bool) or not isinstance(sh_start, int) or sh_start < 1):
            raise ValueError('surfel.sh_start_iteration must be null or a positive integer.')
        if degree:
            if sh_start is None or sh_start + (degree - 1) * sh_interval > self.values['training']['iterations']:
                raise ValueError('Configure an SH schedule that unlocks every band within the training budget.')
            if settings['mode'] == 'deferred' and sh_start <= settings['specular_start_iteration']:
                raise ValueError('Unlock residual SH after deferred reflection starts.')
            if settings['normal_propagation']['enabled'] and sh_start <= settings['normal_propagation']['until_iteration']:
                raise ValueError('Unlock residual SH after normal propagation ends.')
        if self.values["sparse"]["enabled"]:
            raise ValueError("Surfel G-buffers require dense rendering; set sparse.enabled: false.")
        if settings["mode"] not in ("rgb", "deferred"):
            raise ValueError("surfel.mode must be rgb or deferred.")
        if settings['shading_model'] not in ('ldr', 'hdr_angular'):
            raise ValueError('surfel.shading_model must be ldr or hdr_angular.')
        if settings['shading_model'] == 'hdr_angular' and (geometry != 'surfel' or settings['mode'] != 'deferred'):
            raise ValueError('hdr_angular shading requires deferred surfels.')
        if settings['depth_mode'] not in ('centre', 'ray_plane'):
            raise ValueError('surfel.depth_mode must be centre or ray_plane.')
        if geometry == 'gaussian' and settings['depth_mode'] != 'centre':
            raise ValueError('Ray-plane depth requires model.geometry: surfel.')
        for key in ("environment_width", "environment_height"):
            value = settings[key]
            if not isinstance(value, int) or isinstance(value, bool) or not 2 <= value <= 2048:
                raise ValueError(f"surfel.{key} must be an integer in [2, 2048].")
        value = settings["specular_start_iteration"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("surfel.specular_start_iteration must be a non-negative integer.")
        for key in ("reflection_lr", "environment_lr", "angular_lr"):
            value = settings[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"surfel.{key} must be finite and positive.")
        value = settings["initial_reflection"]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("surfel.initial_reflection must be in (0, 1).")
        if not isinstance(settings["save_debug"], bool):
            raise ValueError("surfel.save_debug must be true or false.")
        geometry_loss = self.values['geometry_loss']
        if settings['photometric_loss'] not in ('inherit', 'l1_dssim'):
            raise ValueError('surfel.photometric_loss must be inherit or l1_dssim.')
        foreground = settings['foreground_pruning']
        if not isinstance(foreground['enabled'], bool):
            raise ValueError('foreground_pruning.enabled must be boolean.')
        for key in ('start_iteration', 'interval', 'views', 'min_background_views', 'persistence', 'margin_pixels'):
            value = foreground[key]
            minimum = 0 if key in ('start_iteration', 'margin_pixels') else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'Invalid foreground_pruning.{key}.')
        if foreground['min_background_views'] < 2 or foreground['min_background_views'] > foreground['views']:
            raise ValueError('Foreground pruning needs at least two distinct background views, within the view budget.')
        value = foreground['background_alpha_threshold']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < .5:
            raise ValueError('foreground_pruning.background_alpha_threshold must be in [0, 0.5).')
        refinement = settings['refinement']
        start = refinement['start_iteration']
        if start is not None:
            if isinstance(start, bool) or not isinstance(start, int) or not 1 <= start <= self.values['training']['iterations']:
                raise ValueError('refinement.start_iteration must be null or an iteration within training.')
            if settings['mode'] == 'deferred' and start <= settings['specular_start_iteration']:
                raise ValueError('Refinement must start after reflection activation.')
        interval = refinement['prune_interval']
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
            raise ValueError('refinement.prune_interval must be a non-negative integer.')
        propagation = settings['normal_propagation']
        if not isinstance(propagation['enabled'], bool):
            raise ValueError('normal_propagation.enabled must be boolean.')
        for key in ('interval', 'until_iteration', 'densification_interval', 'opacity_freeze_iterations'):
            value = propagation[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'opacity_freeze_iterations' else 1):
                raise ValueError(f'Invalid normal_propagation.{key}.')
        for key in ('scale_factor', 'scale_reflection_threshold', 'color_reflection_threshold',
                    'color_noise', 'opacity_floor', 'reflection_floor'):
            value = propagation[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid normal_propagation.{key}.')
        if (not 1 <= propagation['scale_factor'] <= 2
                or not 0 <= propagation['color_noise'] <= 0.5
                or not 0 < propagation['opacity_floor'] < 1
                or not 0 < propagation['reflection_floor'] < 1
                or not 0 <= propagation['scale_reflection_threshold'] < 1
                or not 0 <= propagation['color_reflection_threshold'] < 1):
            raise ValueError('Normal propagation scale, noise, floor or threshold is out of range.')
        if propagation['enabled']:
            if geometry != 'surfel' or settings['mode'] != 'deferred':
                raise ValueError('Paper normal propagation requires deferred surfels.')
            if propagation['until_iteration'] <= settings['specular_start_iteration']:
                raise ValueError('Normal propagation must extend beyond reflection activation.')
            if propagation['until_iteration'] >= self.values['training']['iterations']:
                raise ValueError('Leave refinement iterations after normal propagation ends.')
        if geometry == 'gaussian' and settings['photometric_loss'] != 'inherit':
            raise ValueError('The Gaussian control uses its inherited image loss.')
        for key in ('mask_weight', 'normal_depth_weight'):
            value = geometry_loss[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'geometry_loss.{key} must be finite and non-negative.')
        if geometry == 'gaussian' and (geometry_loss['mask_weight'] or geometry_loss['normal_depth_weight']):
            raise ValueError('Geometry losses are implemented for model.geometry: surfel.')
        for key in ('normal_start_iteration', 'normal_ramp_iterations'):
            value = geometry_loss[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'geometry_loss.{key} must be a non-negative integer.')
        for key in ('min_coverage', 'foreground_threshold', 'max_relative_depth_jump'):
            value = geometry_loss[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'geometry_loss.{key} must be in (0, 1].')
