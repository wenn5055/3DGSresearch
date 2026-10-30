# ARCHIVE CANDIDATE (2026-10-29): Camera helper used only by the teaching renderer/trainer chain.
# Current camera loading: 3dgs_trainer.py. Archive only with 3dgs_renderer_v1.py
# and both synthetic trainers; the rest of shared/ remains active.

"""Camera data model and +z-forward look-at construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray

    @classmethod
    def identity(cls, width: int, height: int, focal_length: float) -> "Camera":
        return cls(
            width, height, focal_length, focal_length,
            width / 2.0, height / 2.0, np.eye(4, dtype=np.float32),
        )

    @classmethod
    def from_look_at(
        cls, width: int, height: int, focal_length: float,
        position: np.ndarray, target: np.ndarray, up: np.ndarray,
    ) -> "Camera":
        position = np.asarray(position, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        forward = target - position
        length = np.linalg.norm(forward)
        if length < 1.0e-8:
            raise ValueError("Camera position and look-at target must be different.")
        forward /= length
        right = np.cross(up, forward)
        right_length = np.linalg.norm(right)
        if right_length < 1.0e-8:
            raise ValueError("Camera up vector must not be parallel to the viewing direction.")
        right /= right_length
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ position
        return cls(width, height, focal_length, focal_length,
                   width / 2.0, height / 2.0, world_to_camera)
