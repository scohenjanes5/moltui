"""Archived algorithm snippets used for artifact-free isosurface exports.

These functions are copied from the production implementation for reference.
"""

from __future__ import annotations

import numpy as np


def recompute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return (normals / norms).astype(np.float64)


def laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    alpha: float,
) -> np.ndarray:
    if iterations <= 0:
        return vertices
    alpha = float(max(0.0, min(1.0, alpha)))
    v = vertices.astype(np.float64, copy=True)
    n_vertices = len(v)

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]
    edge_a = np.concatenate([i, j, k, j, k, i])
    edge_b = np.concatenate([j, k, i, i, j, k])

    for _ in range(iterations):
        accum = np.zeros_like(v)
        deg = np.zeros((n_vertices, 1), dtype=np.float64)
        np.add.at(accum, edge_a, v[edge_b])
        np.add.at(deg, edge_a, 1.0)
        avg = np.divide(accum, np.maximum(deg, 1.0))
        v = (1.0 - alpha) * v + alpha * avg
    return v
