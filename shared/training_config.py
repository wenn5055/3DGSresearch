"""Validated training configuration for the Warp 3DGS trainers.

One YAML file merged over `DEFAULTS`, type-checked, and path-resolved. Lifted out of the version 1
trainer because nothing in it touches Warp, a kernel, or a trainer object: it is a schema and its
validation, and the later trainers subclass it only to add their own keys.

`TILE` comes from `splat_math` rather than from a trainer module, so that the bound on
`sparse.samples_per_tile` and the tile size the kernels rasterize with cannot drift apart.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import yaml

from splat_math import ALPHA_CUTOFF, TILE

DEFAULT_RESOLUTION = 256
DEFAULT_SPLATS = 100_000
DEFAULT_INITIAL_SPLATS = 50_000


class TrainingConfig:
    """Validated, merged training configuration loaded from one YAML file.

    Wraps a plain nested dict merged over `DEFAULTS`. Supports the same `config["section"]`
    indexing the rest of the module uses; loading, merging, validating, and formatting for
    display are all methods here rather than free functions passing a dict back and forth.
    """

    DEFAULTS = {
        "paths": {
            "data": "data/lego",
            "output": "warp-training.png",
            "init_ply": None,
        },
        "runtime": {
            "arch": "cpu",
            "seed": 7,
        },
        "model": {
            "resolution": DEFAULT_RESOLUTION,
            "width": None,
            "height": None,
            "capacity": DEFAULT_SPLATS,
            "initial_splats": DEFAULT_INITIAL_SPLATS,
            "tile_pair_capacity": None,
            "scene_radius": None,
            "background": "white",
        },
        "training": {
            "iterations": 2000,
        },
        "loss": {
            "mode": "mse",
            "dssim_weight": 0.2,
            "dssim_window": 11,
        },
        "learning_rates": {
            "position_initial": 0.00016,
            "position_final": 0.0000016,
            "position_max_steps": 2_000,
            "position_delay_steps": 0,
            "position_delay_multiplier": 0.01,
            "feature_dc": 0.0025,
            "opacity": 0.05,
            "scale": 0.005,
            "rotation": 0.001,
        },
        "densification": {
            "interval": 100,
            "fraction": 0.5,
            # A splat is "large" when its biggest scale exceeds percent_dense * scene_radius.
            # Large splats are split; small ones are cloned. This is the reference percent_dense.
            "percent_dense": 0.01,
            # Periodically cap opacity so faded splats must re-earn it or be pruned. 0 disables.
            "opacity_reset_interval": 0,
        },
        "compact_box": {
            "enabled": True,
            "beta": 0.5,
            "alpha_min": ALPHA_CUTOFF,
        },
        "multiview_adc": {
            "enabled": True,
            "views": 10,
            "loss_threshold": 0.1,
            "densify_score_threshold": 5.0,
            "prune_score_threshold": 0.9,
            # How often late-prune fires once the growth plateau has armed it. When it starts
            # is not configurable: it is detected, not scheduled.
            "late_prune_interval": 400,
            # Late pruning removes every active splat below this opacity. Measured on lego at
            # 800x800: 0.01 keeps too much of the faint tail (29.98 dB), 0.05 deletes the faint
            # population the model is largely built from (27.68 dB), 0.02 is the best of the
            # three (30.72 dB) at 15% fewer splats than 0.01.
            "opacity_cutoff": 0.02,
            "candidate_limit": 20000,
        },
        "sparse": {
            "enabled": True,
            "samples_per_tile": 256,
        },
        "adaptive": {
            # Held-out MSE PSNR that latches the MSE -> dense MSE + D-SSIM transition.
            "dssim_psnr_threshold": 25.0,
            "late_prune_growth_window": 3,
            "late_prune_growth_percent": 2.0,
        },
        "convergence": {
            "enabled": True,
            "min_iterations": 1200,
            "patience": 5,
            "min_delta": 1.0e-5,
        },
        "reporting": {
            "log_every": 20,
            "eval_every": 100,
            # `fixed_eval` is measured on views the optimiser never trains on, so a falling
            # fixed_eval means the model generalises rather than that it has memorised the
            # training cameras. `eval_manifest` names the validation manifest inside `paths.data`;
            # set it to null to fall back to evaluating on training views, which is what every
            # run before this key existed did.
            "eval_manifest": "transforms_val.json",
            # How many held-out views to evaluate on. Eight is enough for the mean to be steady
            # from one eval tick to the next -- four left visible jitter in the raw series -- and
            # cheap enough to render at every `eval_every` tick.
            "eval_views": 8,
            # Explicit frame indices into `eval_manifest`, or null to choose `eval_views` frames
            # automatically by farthest-point sampling over camera directions. Pin these when a
            # comparison must score two runs on exactly the same cameras.
            "eval_view_ids": None,
            "loss_ema_decay": 0.95,
            "snapshot_every": 200,
            "save_ply": True,
        },
    }

    def __init__(self, values: dict):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def __repr__(self) -> str:
        return f"TrainingConfig({self.values!r})"

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        config_path = path.expanduser().resolve()
        supplied = yaml.safe_load(config_path.read_text()) or {}
        config = cls(cls._merge(cls.DEFAULTS, supplied or {}))
        for key in ("data", "output", "init_ply"):
            value = config.values["paths"][key]
            if value is None:
                continue
            if not isinstance(value, (str, Path)):
                raise ValueError(f"paths.{key} must be a path string or null.")
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = config_path.parent / resolved
            config.values["paths"][key] = resolved.resolve()
        config.values["paths"]["output"] = config.values["paths"]["output"].with_suffix(".png")
        config.validate()
        return config

    @classmethod
    def _merge(cls, defaults: dict, supplied: dict, prefix: str = "") -> dict:
        if not isinstance(supplied, dict):
            location = prefix.removesuffix(".") or "configuration"
            raise ValueError(f"{location} must be a YAML mapping.")
        unknown = sorted(set(supplied) - set(defaults))
        if unknown:
            location = prefix.removesuffix(".") or "configuration"
            raise ValueError(f"Unknown key(s) in {location}: {', '.join(unknown)}")
        merged = deepcopy(defaults)
        for key, value in supplied.items():
            if isinstance(defaults[key], dict):
                merged[key] = cls._merge(defaults[key], value, f"{prefix}{key}.")
            else:
                merged[key] = value
        return merged

    def validate(self) -> None:
        paths = self.values["paths"]
        runtime = self.values["runtime"]
        model = self.values["model"]
        training = self.values["training"]
        loss = self.values["loss"]
        rates = self.values["learning_rates"]
        densification = self.values["densification"]
        compact = self.values["compact_box"]
        adc = self.values["multiview_adc"]
        sparse = self.values["sparse"]
        adaptive = self.values["adaptive"]
        convergence = self.values["convergence"]
        reporting = self.values["reporting"]
        if loss["mode"] not in ("mse", "mse_dssim"):
            raise ValueError("loss.mode must be 'mse' or 'mse_dssim'.")
        weight = loss["dssim_weight"]
        if (not isinstance(weight, (int, float)) or isinstance(weight, bool)
                or not math.isfinite(weight) or not 0.0 <= weight <= 1.0):
            raise ValueError("loss.dssim_weight must be a finite number in [0, 1].")
        window = loss["dssim_window"]
        if not isinstance(window, int) or isinstance(window, bool) or not 1 <= window <= 31 or window % 2 == 0:
            raise ValueError("loss.dssim_window must be an odd integer in [1, 31].")
        threshold = adaptive["dssim_psnr_threshold"]
        if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                or not math.isfinite(threshold) or not 0.0 <= threshold <= 100.0):
            raise ValueError("adaptive.dssim_psnr_threshold must be a finite number in [0, 100] dB.")
        if runtime["arch"] not in ("cpu", "gpu"):
            raise ValueError("runtime.arch must be 'cpu' or 'gpu'.")
        if not isinstance(runtime["seed"], int) or isinstance(runtime["seed"], bool):
            raise ValueError("runtime.seed must be an integer.")
        for name in ("data", "output"):
            if paths[name] is None:
                raise ValueError(f"paths.{name} cannot be null.")
        if model["background"] not in ("white", "black"):
            raise ValueError("model.background must be 'white' or 'black'.")
        for dimension in ("width", "height"):
            if model[dimension] is None:
                model[dimension] = model["resolution"]
        for name, value in (
            ("model.resolution", model["resolution"]),
            ("model.width", model["width"]),
            ("model.height", model["height"]),
            ("model.capacity", model["capacity"]),
            ("model.initial_splats", model["initial_splats"]),
            ("training.iterations", training["iterations"]),
            ("learning_rates.position_max_steps", rates["position_max_steps"]),
            ("reporting.log_every", reporting["log_every"]),
            ("reporting.eval_every", reporting["eval_every"]),
            ("reporting.eval_views", reporting["eval_views"]),
            ("reporting.snapshot_every", reporting["snapshot_every"]),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1.")
        if model["initial_splats"] > model["capacity"]:
            raise ValueError("model.initial_splats cannot exceed model.capacity.")
        if (
            model["tile_pair_capacity"] is not None
            and (
                not isinstance(model["tile_pair_capacity"], int)
                or isinstance(model["tile_pair_capacity"], bool)
                or model["tile_pair_capacity"] < 1
            )
        ):
            raise ValueError("model.tile_pair_capacity must be a positive integer or null.")
        if model["scene_radius"] is not None and model["scene_radius"] <= 0.0:
            raise ValueError("model.scene_radius must be positive or null.")
        if not isinstance(densification["interval"], int) or densification["interval"] < 0:
            raise ValueError("densification.interval must be a non-negative integer.")
        if not 0.0 < densification["fraction"] <= 1.0:
            raise ValueError("densification.fraction must be in (0, 1].")
        if not 0.0 < densification["percent_dense"] <= 1.0:
            raise ValueError("densification.percent_dense must be in (0, 1].")
        if (
            not isinstance(densification["opacity_reset_interval"], int)
            or isinstance(densification["opacity_reset_interval"], bool)
            or densification["opacity_reset_interval"] < 0
        ):
            raise ValueError(
                "densification.opacity_reset_interval must be a non-negative integer."
            )
        if reporting["eval_manifest"] is not None and (
            not isinstance(reporting["eval_manifest"], str)
            or not reporting["eval_manifest"].strip()
        ):
            raise ValueError("reporting.eval_manifest must be a non-empty string or null.")
        held_out_ids = reporting["eval_view_ids"]
        if held_out_ids is not None:
            if not isinstance(held_out_ids, list) or not held_out_ids:
                raise ValueError(
                    "reporting.eval_view_ids must be a non-empty list of frame indices or null."
                )
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in held_out_ids
            ):
                raise ValueError(
                    "reporting.eval_view_ids entries must be non-negative integers."
                )
            if len(set(held_out_ids)) != len(held_out_ids):
                raise ValueError("reporting.eval_view_ids must not repeat a frame index.")
            if reporting["eval_manifest"] is None:
                raise ValueError(
                    "reporting.eval_view_ids indexes reporting.eval_manifest, which is null."
                )
        if not 0.0 <= reporting["loss_ema_decay"] < 1.0:
            raise ValueError("reporting.loss_ema_decay must be in [0, 1).")
        if not isinstance(reporting["save_ply"], bool):
            raise ValueError("reporting.save_ply must be true or false.")
        if not isinstance(compact["enabled"], bool):
            raise ValueError("compact_box.enabled must be true or false.")
        if compact["beta"] <= 0.0:
            raise ValueError("compact_box.beta must be positive.")
        if compact["alpha_min"] <= 0.0:
            raise ValueError("compact_box.alpha_min must be positive.")
        if not isinstance(adc["enabled"], bool):
            raise ValueError("multiview_adc.enabled must be true or false.")
        for name in ("views", "late_prune_interval", "candidate_limit"):
            if not isinstance(adc[name], int) or isinstance(adc[name], bool) or adc[name] < 1:
                raise ValueError(f"multiview_adc.{name} must be an integer of at least 1.")
        for name in ("loss_threshold", "densify_score_threshold", "prune_score_threshold", "opacity_cutoff"):
            if adc[name] < 0.0:
                raise ValueError(f"multiview_adc.{name} must be non-negative.")
        if not isinstance(sparse["enabled"], bool):
            raise ValueError("sparse.enabled must be true or false.")
        sample_count = sparse["samples_per_tile"]
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or not 1 <= sample_count <= TILE * TILE
        ):
            raise ValueError(f"sparse.samples_per_tile must be an integer in [1, {TILE * TILE}].")
        if (
            not isinstance(adaptive["late_prune_growth_window"], int)
            or isinstance(adaptive["late_prune_growth_window"], bool)
            or adaptive["late_prune_growth_window"] < 1
        ):
            raise ValueError("adaptive.late_prune_growth_window must be an integer of at least 1.")
        if adaptive["late_prune_growth_percent"] < 0.0:
            raise ValueError("adaptive.late_prune_growth_percent must be non-negative.")
        if not isinstance(convergence["enabled"], bool):
            raise ValueError("convergence.enabled must be true or false.")
        for name in ("min_iterations", "patience"):
            if (
                not isinstance(convergence[name], int)
                or isinstance(convergence[name], bool)
                or convergence[name] < 1
            ):
                raise ValueError(f"convergence.{name} must be an integer of at least 1.")
        if convergence["min_delta"] < 0.0:
            raise ValueError("convergence.min_delta must be non-negative.")
        for name, value in rates.items():
            if name == "position_delay_steps":
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError("learning_rates.position_delay_steps must be a non-negative integer.")
            elif name == "position_delay_multiplier":
                if not 0.0 < value <= 1.0:
                    raise ValueError("learning_rates.position_delay_multiplier must be in (0, 1].")
            elif name == "feature_rest":
                if value < 0.0:
                    raise ValueError("learning_rates.feature_rest must be non-negative.")
            elif value <= 0.0:
                raise ValueError(f"learning_rates.{name} must be positive.")

    def printable(self) -> dict:
        result = deepcopy(self.values)
        for key, value in result["paths"].items():
            result["paths"][key] = None if value is None else str(value)
        return result
