#!/usr/bin/env python3
"""Tests for Z-matrix parsing and coordinate conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.parsers import parse_zmat

WATER_ZMAT = """\
O
H 1 0.96
H 1 0.96 2 104.5
"""

WATER_VARS_ZMAT = """\
O
H 1 r1
H 1 r1 2 a1

r1 = 0.96
a1 = 104.5
"""

METHANE_ZMAT = """\
C
H 1 1.09
H 1 1.09 2 109.5
H 1 1.09 2 109.5 3 120.0
H 1 1.09 2 109.5 3 -120.0
"""

ETHANOL_ZMAT = """\
C
C 1 1.54
O 2 1.43 1 109.5
H 1 1.09 2 109.5 3 60.0
H 1 1.09 2 109.5 3 -60.0
H 1 1.09 2 109.5 3 180.0
H 2 1.09 1 109.5 3 60.0
H 2 1.09 1 109.5 3 -60.0
H 3 0.96 2 104.5 1 180.0
"""


def _write_zmat(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(content)
    return path


class TestParseZmatWater:
    def test_atom_count(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water.zmat", WATER_ZMAT))
        assert len(mol.atoms) == 3

    def test_elements(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water.zmat", WATER_ZMAT))
        symbols = [a.element.symbol for a in mol.atoms]
        assert symbols == ["O", "H", "H"]

    def test_oh_bond_length(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water.zmat", WATER_ZMAT))
        for i, j, dist in mol.get_bond_lengths():
            assert dist == pytest.approx(0.96, abs=0.01)

    def test_hoh_angle(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water.zmat", WATER_ZMAT))
        angles = mol.get_angles()
        assert len(angles) == 1
        assert angles[0][3] == pytest.approx(104.5, abs=0.5)

    def test_bonds_detected(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water.zmat", WATER_ZMAT))
        assert len(mol.bonds) == 2


class TestParseZmatWaterVars:
    """Same as water but using named variables."""

    def test_atom_count(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water_vars.zmat", WATER_VARS_ZMAT))
        assert len(mol.atoms) == 3

    def test_oh_bond_length(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water_vars.zmat", WATER_VARS_ZMAT))
        for i, j, dist in mol.get_bond_lengths():
            assert dist == pytest.approx(0.96, abs=0.01)

    def test_hoh_angle(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water_vars.zmat", WATER_VARS_ZMAT))
        angles = mol.get_angles()
        assert len(angles) == 1
        assert angles[0][3] == pytest.approx(104.5, abs=0.5)


class TestParseZmatMethane:
    def test_atom_count(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "methane.zmat", METHANE_ZMAT))
        assert len(mol.atoms) == 5

    def test_elements(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "methane.zmat", METHANE_ZMAT))
        symbols = sorted(a.element.symbol for a in mol.atoms)
        assert symbols == ["C", "H", "H", "H", "H"]

    def test_ch_bond_lengths(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "methane.zmat", METHANE_ZMAT))
        for _, _, dist in mol.get_bond_lengths():
            assert dist == pytest.approx(1.09, abs=0.01)

    def test_hch_angles_reasonable(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "methane.zmat", METHANE_ZMAT))
        angles = mol.get_angles()
        assert len(angles) > 0
        for _, _, _, angle in angles:
            assert 60.0 < angle < 180.0

    def test_four_bonds(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "methane.zmat", METHANE_ZMAT))
        assert len(mol.bonds) == 4


class TestParseZmatEthanol:
    def test_atom_count(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "ethanol.zmat", ETHANOL_ZMAT))
        assert len(mol.atoms) == 9

    def test_elements(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "ethanol.zmat", ETHANOL_ZMAT))
        symbols = sorted(a.element.symbol for a in mol.atoms)
        assert symbols == ["C", "C", "H", "H", "H", "H", "H", "H", "O"]

    def test_has_bonds(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "ethanol.zmat", ETHANOL_ZMAT))
        assert len(mol.bonds) >= 8  # at least 8 bonds in ethanol


class TestZmatCoordinateConversion:
    """Test the coordinate conversion produces valid 3D geometry."""

    def test_first_atom_at_origin(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water.zmat", WATER_ZMAT))
        np.testing.assert_array_almost_equal(mol.atoms[0].position, [0, 0, 0])

    def test_second_atom_along_x(self, tmp_path: Path):
        mol = parse_zmat(_write_zmat(tmp_path, "water.zmat", WATER_ZMAT))
        pos = mol.atoms[1].position
        assert pos[0] == pytest.approx(0.96, abs=0.01)
        assert pos[1] == pytest.approx(0.0, abs=1e-10)
        assert pos[2] == pytest.approx(0.0, abs=1e-10)

    def test_all_positions_finite(self, tmp_path: Path):
        zmat_files = [
            _write_zmat(tmp_path, "water.zmat", WATER_ZMAT),
            _write_zmat(tmp_path, "water_vars.zmat", WATER_VARS_ZMAT),
            _write_zmat(tmp_path, "methane.zmat", METHANE_ZMAT),
            _write_zmat(tmp_path, "ethanol.zmat", ETHANOL_ZMAT),
        ]
        for zmat_file in zmat_files:
            mol = parse_zmat(zmat_file)
            for atom in mol.atoms:
                assert np.all(np.isfinite(atom.position)), (
                    f"Non-finite position in {zmat_file.name}"
                )


def test_parse_zmat_smoke(tmp_path: Path):
    zmat_files = [
        _write_zmat(tmp_path, "water.zmat", WATER_ZMAT),
        _write_zmat(tmp_path, "water_vars.zmat", WATER_VARS_ZMAT),
        _write_zmat(tmp_path, "methane.zmat", METHANE_ZMAT),
        _write_zmat(tmp_path, "ethanol.zmat", ETHANOL_ZMAT),
    ]
    for zmat_file in zmat_files:
        mol = parse_zmat(zmat_file)
        assert len(mol.atoms) > 0
        assert all(a.position.shape == (3,) for a in mol.atoms)
