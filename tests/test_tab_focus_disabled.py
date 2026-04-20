#!/usr/bin/env python3
"""Tests for disabling Tab-based focus cycling."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _install_skimage_stub() -> None:
    """Provide a lightweight skimage stub so app imports in test envs."""
    if "skimage.measure" in sys.modules:
        return
    skimage = types.ModuleType("skimage")
    measure = types.ModuleType("skimage.measure")

    def _marching_cubes(*_args, **_kwargs):  # pragma: no cover - never used in this test
        raise RuntimeError("marching_cubes stub called unexpectedly")

    measure.marching_cubes = _marching_cubes
    skimage.measure = measure
    sys.modules["skimage"] = skimage
    sys.modules["skimage.measure"] = measure


@pytest.mark.asyncio
async def test_tab_and_shift_tab_do_not_change_focus() -> None:
    _install_skimage_stub()

    from moltui.app import MoleculeView, MoltuiApp
    from moltui.elements import Atom, Molecule, get_element

    atoms = [
        Atom(get_element("O"), np.array([0.0, 0.0, 0.0])),
        Atom(get_element("H"), np.array([0.9, 0.0, 0.0])),
        Atom(get_element("H"), np.array([-0.3, 0.8, 0.0])),
    ]
    molecule = Molecule(atoms=atoms, bonds=[])
    molecule.detect_bonds()

    app = MoltuiApp(molecule=molecule, filepath="sample.xyz")

    async with app.run_test() as pilot:
        await pilot.pause()
        view = app.query_one(MoleculeView)
        table = app._active_panel_table()
        assert table is not None
        assert table.has_focus
        assert not view.has_focus

        await pilot.press("tab")
        await pilot.pause()
        table = app._active_panel_table()
        assert table is not None
        assert table.has_focus
        assert not view.has_focus

        await pilot.press("shift+tab")
        await pilot.pause()
        table = app._active_panel_table()
        assert table is not None
        assert table.has_focus
        assert not view.has_focus
