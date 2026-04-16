#!/usr/bin/env python3
"""Tests for Z-matrix parsing and coordinate conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.parsers import parse_zmat

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class TestParseZmatWater:
    def test_atom_count(self):
        mol = parse_zmat(EXAMPLES_DIR / "water.zmat")
        assert len(mol.atoms) == 3

    def test_elements(self):
        mol = parse_zmat(EXAMPLES_DIR / "water.zmat")
        symbols = [a.element.symbol for a in mol.atoms]
        assert symbols == ["O", "H", "H"]

    def test_oh_bond_length(self):
        mol = parse_zmat(EXAMPLES_DIR / "water.zmat")
        for i, j, dist in mol.get_bond_lengths():
            assert dist == pytest.approx(0.96, abs=0.01)

    def test_hoh_angle(self):
        mol = parse_zmat(EXAMPLES_DIR / "water.zmat")
        angles = mol.get_angles()
        assert len(angles) == 1
        assert angles[0][3] == pytest.approx(104.5, abs=0.5)

    def test_bonds_detected(self):
        mol = parse_zmat(EXAMPLES_DIR / "water.zmat")
        assert len(mol.bonds) == 2


class TestParseZmatWaterVars:
    """Same as water but using named variables."""

    def test_atom_count(self):
        mol = parse_zmat(EXAMPLES_DIR / "water_vars.zmat")
        assert len(mol.atoms) == 3

    def test_oh_bond_length(self):
        mol = parse_zmat(EXAMPLES_DIR / "water_vars.zmat")
        for i, j, dist in mol.get_bond_lengths():
            assert dist == pytest.approx(0.96, abs=0.01)

    def test_hoh_angle(self):
        mol = parse_zmat(EXAMPLES_DIR / "water_vars.zmat")
        angles = mol.get_angles()
        assert len(angles) == 1
        assert angles[0][3] == pytest.approx(104.5, abs=0.5)


class TestParseZmatMethane:
    def test_atom_count(self):
        mol = parse_zmat(EXAMPLES_DIR / "methane.zmat")
        assert len(mol.atoms) == 5

    def test_elements(self):
        mol = parse_zmat(EXAMPLES_DIR / "methane.zmat")
        symbols = sorted(a.element.symbol for a in mol.atoms)
        assert symbols == ["C", "H", "H", "H", "H"]

    def test_ch_bond_lengths(self):
        mol = parse_zmat(EXAMPLES_DIR / "methane.zmat")
        for _, _, dist in mol.get_bond_lengths():
            assert dist == pytest.approx(1.09, abs=0.01)

    def test_hch_angles_reasonable(self):
        mol = parse_zmat(EXAMPLES_DIR / "methane.zmat")
        angles = mol.get_angles()
        assert len(angles) > 0
        for _, _, _, angle in angles:
            assert 60.0 < angle < 180.0

    def test_four_bonds(self):
        mol = parse_zmat(EXAMPLES_DIR / "methane.zmat")
        assert len(mol.bonds) == 4


class TestParseZmatEthanol:
    def test_atom_count(self):
        mol = parse_zmat(EXAMPLES_DIR / "ethanol.zmat")
        assert len(mol.atoms) == 9

    def test_elements(self):
        mol = parse_zmat(EXAMPLES_DIR / "ethanol.zmat")
        symbols = sorted(a.element.symbol for a in mol.atoms)
        assert symbols == ["C", "C", "H", "H", "H", "H", "H", "H", "O"]

    def test_has_bonds(self):
        mol = parse_zmat(EXAMPLES_DIR / "ethanol.zmat")
        assert len(mol.bonds) >= 8  # at least 8 bonds in ethanol


class TestZmatCoordinateConversion:
    """Test the coordinate conversion produces valid 3D geometry."""

    def test_first_atom_at_origin(self):
        mol = parse_zmat(EXAMPLES_DIR / "water.zmat")
        np.testing.assert_array_almost_equal(mol.atoms[0].position, [0, 0, 0])

    def test_second_atom_along_x(self):
        mol = parse_zmat(EXAMPLES_DIR / "water.zmat")
        pos = mol.atoms[1].position
        assert pos[0] == pytest.approx(0.96, abs=0.01)
        assert pos[1] == pytest.approx(0.0, abs=1e-10)
        assert pos[2] == pytest.approx(0.0, abs=1e-10)

    def test_all_positions_finite(self):
        for zmat_file in EXAMPLES_DIR.glob("*.zmat"):
            mol = parse_zmat(zmat_file)
            for atom in mol.atoms:
                assert np.all(np.isfinite(atom.position)), (
                    f"Non-finite position in {zmat_file.name}"
                )


ZMAT_FILES = sorted(EXAMPLES_DIR.glob("*.zmat"))


@pytest.mark.parametrize("zmat_file", ZMAT_FILES, ids=lambda p: p.name)
def test_parse_zmat_smoke(zmat_file: Path):
    mol = parse_zmat(zmat_file)
    assert len(mol.atoms) > 0
    assert all(a.position.shape == (3,) for a in mol.atoms)
