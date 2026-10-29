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
        "specular_start_iteration": 10000,
        "initial_reflection": 0.1,
        "reflection_lr": 0.01,
        "environment_lr": 0.01,
        "environment_width": 128,
        "environment_height": 64,
        "save_debug": True,
    }

    def validate(self):
        super().validate()
        settings = self.values["surfel"]
        geometry = self.values['model']['geometry']
        if geometry not in ('gaussian', 'surfel'):
            raise ValueError('model.geometry must be gaussian or surfel for project passes.')
        if geometry == 'gaussian' and settings['mode'] != 'rgb':
            raise ValueError('The Gaussian control pass uses surfel.mode: rgb.')
        if self.values["model"]["sh_degree"] != 0:
            raise ValueError("These surfel passes use fixed RGB or deferred reflection; set model.sh_degree: 0.")
        if self.values["sparse"]["enabled"]:
            raise ValueError("Surfel G-buffers require dense rendering; set sparse.enabled: false.")
        if settings["mode"] not in ("rgb", "deferred"):
            raise ValueError("surfel.mode must be rgb or deferred.")
        for key in ("environment_width", "environment_height"):
            value = settings[key]
            if not isinstance(value, int) or isinstance(value, bool) or not 2 <= value <= 2048:
                raise ValueError(f"surfel.{key} must be an integer in [2, 2048].")
        value = settings["specular_start_iteration"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("surfel.specular_start_iteration must be a non-negative integer.")
        for key in ("reflection_lr", "environment_lr"):
            value = settings[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"surfel.{key} must be finite and positive.")
        value = settings["initial_reflection"]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("surfel.initial_reflection must be in (0, 1).")
        if not isinstance(settings["save_debug"], bool):
            raise ValueError("surfel.save_debug must be true or false.")
        geometry_loss = self.values['geometry_loss']
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
