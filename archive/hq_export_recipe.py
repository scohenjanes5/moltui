"""Archived high-quality PNG export recipe for future reuse.

This module mirrors the settings that produced the best artifact-free results.
It is intentionally standalone and not imported by the runtime app.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HQRendererProfile:
    flatten_perspective: float = 2.5
    fov_multiplier: float = 2.5
    ambient: float = 0.31
    diffuse_strength: float = 0.72
    specular_strength: float = 0.42
    shininess: float = 96.0
    light_dir: tuple[float, float, float] = (0.58, 0.56, -0.60)


@dataclass(frozen=True)
class HQIsosurfaceProfile:
    default_upsample: int = 3
    small_grid_upsample: int = 4
    small_grid_threshold: int = 1_000_000
    smooth_sigma: float = 0.5
    mesh_smooth_iters: int = 2
    small_grid_mesh_smooth_iters: int = 3
    mesh_smooth_alpha: float = 0.35


def pick_hq_isosurface_params(
    n_points: tuple[int, int, int],
    profile: HQIsosurfaceProfile = HQIsosurfaceProfile(),
) -> dict[str, float | int]:
    """Return adaptive extraction params based on cube grid size."""
    grid_points = int(np.prod(n_points))
    upsample = profile.default_upsample
    mesh_smooth_iters = profile.mesh_smooth_iters
    if grid_points <= profile.small_grid_threshold:
        upsample = profile.small_grid_upsample
        mesh_smooth_iters = profile.small_grid_mesh_smooth_iters
    return {
        "upsample": upsample,
        "smooth_sigma": profile.smooth_sigma,
        "mesh_smooth_iters": mesh_smooth_iters,
        "mesh_smooth_alpha": profile.mesh_smooth_alpha,
    }


def build_hq_renderer_kwargs(
    profile: HQRendererProfile = HQRendererProfile(),
) -> dict[str, float | tuple[float, float, float]]:
    """Return ImageRenderer kwargs used for final HQ PNG export."""
    return {
        "fov": 1.5 * profile.fov_multiplier,
        "ambient": profile.ambient,
        "diffuse_strength": profile.diffuse_strength,
        "specular_strength": profile.specular_strength,
        "shininess": profile.shininess,
        "light_dir": profile.light_dir,
    }
