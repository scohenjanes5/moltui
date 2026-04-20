#!/usr/bin/env python3
"""Tests for isosurface.py: marching cubes extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moltui.isosurface import IsosurfaceMesh, extract_isosurfaces
from moltui.parsers import parse_cube_data


def _write_cube(tmp_path: Path, name: str, center_x: float = 0.0) -> Path:
    n = 10
    step = 0.5
    origin = np.array([-2.0, -2.0, -2.0])
    x = origin[0] + step * np.arange(n)
    y = origin[1] + step * np.arange(n)
    z = origin[2] + step * np.arange(n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    data = np.exp(-0.5 * ((xx - center_x) ** 2 + yy**2 + zz**2)) - 0.2

    path = tmp_path / name
    values = data.ravel()
    value_lines = [
        " ".join(f"{v:.8e}" for v in values[i : i + 6]) for i in range(0, len(values), 6)
    ]
    lines = [
        "Cube file",
        "Generated for isosurface tests",
        f"1 {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}",
        f"{n} {step:.6f} 0.000000 0.000000",
        f"{n} 0.000000 {step:.6f} 0.000000",
        f"{n} 0.000000 0.000000 {step:.6f}",
        "1 0.0 0.0 0.0 0.0",
        *value_lines,
        "",
    ]
    path.write_text("\n".join(lines))
    return path


class TestExtractIsosurfaces:
    def test_returns_list_of_meshes(self, tmp_path: Path):
        cube_data = parse_cube_data(_write_cube(tmp_path, "sample.cube"))
        meshes = extract_isosurfaces(cube_data)
        assert isinstance(meshes, list)
        for m in meshes:
            assert isinstance(m, IsosurfaceMesh)

    def test_mesh_has_vertices_and_faces(self, tmp_path: Path):
        cube_data = parse_cube_data(_write_cube(tmp_path, "sample.cube"))
        meshes = extract_isosurfaces(cube_data)
        assert len(meshes) > 0
        for m in meshes:
            assert m.vertices.ndim == 2 and m.vertices.shape[1] == 3
            assert m.faces.ndim == 2 and m.faces.shape[1] == 3
            assert m.normals.shape == m.vertices.shape

    def test_normals_are_unit_length(self, tmp_path: Path):
        cube_data = parse_cube_data(_write_cube(tmp_path, "sample.cube"))
        meshes = extract_isosurfaces(cube_data)
        for m in meshes:
            norms = np.linalg.norm(m.normals, axis=1)
            np.testing.assert_array_almost_equal(norms, 1.0, decimal=3)

    def test_high_isovalue_produces_fewer_or_no_meshes(self, tmp_path: Path):
        cube_data = parse_cube_data(_write_cube(tmp_path, "sample.cube"))
        meshes_low = extract_isosurfaces(cube_data, isovalue=0.01)
        meshes_high = extract_isosurfaces(cube_data, isovalue=0.5)
        total_verts_low = sum(len(m.vertices) for m in meshes_low)
        total_verts_high = sum(len(m.vertices) for m in meshes_high)
        assert total_verts_high <= total_verts_low

    def test_colors_are_positive_negative(self, tmp_path: Path):
        cube_data = parse_cube_data(_write_cube(tmp_path, "sample.cube"))
        meshes = extract_isosurfaces(cube_data)
        if len(meshes) == 2:
            colors = {m.color for m in meshes}
            from moltui.isosurface import COLOR_NEGATIVE, COLOR_POSITIVE

            assert COLOR_POSITIVE in colors
            assert COLOR_NEGATIVE in colors


def test_extract_smoke(tmp_path: Path):
    cube_files = [
        _write_cube(tmp_path, "sample_a.cube", center_x=0.0),
        _write_cube(tmp_path, "sample_b.cube", center_x=0.4),
    ]
    for cube_file in cube_files:
        cd = parse_cube_data(cube_file)
        meshes = extract_isosurfaces(cd)
        assert isinstance(meshes, list)
