# ARCHIVE CANDIDATE (2026-10-29): Projected-record container used only by 3dgs_renderer_v1.py.
# Current workspace: gaussian_first_tile_workspace_gpu.py with shared/tile_builder.py.
# Archive with the teaching renderer and both synthetic trainers.

"""Screen-space records produced by projecting 3D Gaussians."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectedGaussians:
    centres: np.ndarray
    conics: np.ndarray
    depths: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
