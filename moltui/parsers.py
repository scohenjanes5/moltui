from pathlib import Path

import numpy as np

from .elements import Atom, Molecule, get_element, get_element_by_number

BOHR_TO_ANGSTROM = 0.529177249


def parse_xyz(filepath: str | Path) -> Molecule:
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    # line 1 is comment, skip
    atoms = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(Atom(element=get_element(symbol), position=np.array([x, y, z])))

    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


def parse_cube(filepath: str | Path) -> Molecule:
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    # Lines 0-1: comments
    # Line 2: n_atoms, origin_x, origin_y, origin_z
    parts = lines[2].split()
    n_atoms = abs(int(parts[0]))

    # Lines 3-5: voxel axes (skip for atom-only parsing)
    # Lines 6 to 6+n_atoms-1: atom data
    atoms = []
    for i in range(n_atoms):
        parts = lines[6 + i].split()
        atomic_number = int(parts[0])
        # parts[1] is charge, skip
        x = float(parts[2]) * BOHR_TO_ANGSTROM
        y = float(parts[3]) * BOHR_TO_ANGSTROM
        z = float(parts[4]) * BOHR_TO_ANGSTROM
        atoms.append(
            Atom(
                element=get_element_by_number(atomic_number),
                position=np.array([x, y, z]),
            )
        )

    mol = Molecule(atoms=atoms, bonds=[])
    mol.detect_bonds()
    return mol


def load_molecule(filepath: str | Path) -> Molecule:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == ".xyz":
        return parse_xyz(filepath)
    elif suffix == ".cube":
        return parse_cube(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .xyz or .cube")
