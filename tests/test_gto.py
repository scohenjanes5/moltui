"""Tests for the pure-NumPy GTO evaluator against PySCF reference.

Requires pyscf: install with `uv sync --extra test-pyscf`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moltui.gto import eval_gto, parse_molden

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

pyscf = pytest.importorskip("pyscf", reason="pyscf not installed — skipping GTO reference tests")


MOLDEN_FILES = [p.name for p in EXAMPLES_DIR.glob("*.molden")]


def _build_grid(coords_bohr: np.ndarray, grid_n: int = 60, padding: float = 5.0):
    """Build a cubic grid around atom positions (in Bohr)."""
    from moltui.gto import BOHR_TO_ANGSTROM

    padding_bohr = padding / BOHR_TO_ANGSTROM
    min_c = coords_bohr.min(axis=0) - padding_bohr
    max_c = coords_bohr.max(axis=0) + padding_bohr

    axes = [np.linspace(min_c[i], max_c[i], grid_n) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


@pytest.mark.parametrize("molden_file", MOLDEN_FILES)
def test_ao_shape_matches_pyscf(molden_file: str) -> None:
    """Check that the AO matrix has the same shape as PySCF's."""
    filepath = EXAMPLES_DIR / molden_file

    basis = parse_molden(filepath)
    grid_points = _build_grid(basis.atom_coords_bohr)
    ao_numpy = eval_gto(basis.shells, grid_points, basis.spherical)

    from pyscf.tools import molden as molden_tools

    pyscf_mol = molden_tools.load(str(filepath))[0]
    ao_pyscf = pyscf_mol.eval_gto("GTOval_sph", grid_points)

    assert ao_numpy.shape == ao_pyscf.shape, (
        f"Shape mismatch: numpy {ao_numpy.shape} vs pyscf {ao_pyscf.shape}"
    )


@pytest.mark.parametrize("molden_file", MOLDEN_FILES)
def test_mo_values_match_pyscf(molden_file: str) -> None:
    """Check that MO values on a grid match PySCF for multiple orbitals.

    Individual AO column ordering may differ between our parser and PySCF,
    but the MO products (ao @ mo_coeff) must match since both use consistent
    AO/coefficient pairs from the same molden file.
    """
    filepath = EXAMPLES_DIR / molden_file

    # NumPy
    basis = parse_molden(filepath)
    grid_points = _build_grid(basis.atom_coords_bohr)
    ao_numpy = eval_gto(basis.shells, grid_points, basis.spherical)

    # PySCF
    from pyscf.tools import molden as molden_tools

    result = molden_tools.load(str(filepath))
    pyscf_mol, pyscf_mo_coeff, pyscf_mo_occ = result[0], result[2], result[3]
    ao_pyscf = pyscf_mol.eval_gto("GTOval_sph", grid_points)

    # Test HOMO, LUMO, and a few others
    homo_idx = int(np.where(np.asarray(pyscf_mo_occ) > 0.5)[0][-1])
    n_mos = basis.mo_coefficients.shape[1]
    test_indices = sorted(
        {0, max(0, homo_idx - 1), homo_idx, min(n_mos - 1, homo_idx + 1), n_mos - 1}
    )

    for mo_idx in test_indices:
        mo_numpy = ao_numpy @ basis.mo_coefficients[:, mo_idx]
        mo_pyscf = ao_pyscf @ pyscf_mo_coeff[:, mo_idx]

        max_val = np.max(np.abs(mo_pyscf))
        if max_val < 1e-15:
            continue
        rel_err = np.max(np.abs(mo_numpy - mo_pyscf)) / max_val
        corr = np.corrcoef(mo_numpy, mo_pyscf)[0, 1]

        assert rel_err < 1e-10, f"MO {mo_idx}: relative error {rel_err:.2e}"
        assert corr > 1.0 - 1e-12, f"MO {mo_idx}: correlation {corr}"
