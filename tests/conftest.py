"""Shared pytest fixtures and helpers for the mlip-quantum-thermal test suite.

Makes the standalone milestone scripts importable (they live at the repo root,
not in a package yet — see CLAUDE.md "refactor into a package only at
milestone 3") and provides small builders for synthetic rmc6f inputs and
in-memory configuration dicts.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def m1():
    """The milestone1_bands module, imported once per session.

    Importing is cheap: only numpy is imported at module load; ase/spglib/
    phonopy/mace imports are all lazy (inside the functions that use them).
    """
    import milestone1_bands

    return milestone1_bands


def write_rmc6f(path: Path, text: str) -> Path:
    """Write dedented rmc6f *text* to *path* and return it."""
    path.write_text(textwrap.dedent(text).lstrip("\n"))
    return path


def make_config(frac, elements, dims, cell, site_ids=None):
    """Build a config dict in the shape ``parse_rmc6f`` returns.

    Parameters
    ----------
    frac : array_like (N, 3)
        Fractional coordinates in the *supercell* frame.
    elements : list[str]
        Per-atom element symbols, length N.
    dims : array_like (3,)
        Supercell repetitions (Nx, Ny, Nz).
    cell : array_like (3, 3)
        Supercell lattice vectors (rows), Angstrom.
    site_ids : array_like (N,) or None
        Per-atom unit-cell site ids, or None when absent.
    """
    return {
        "cell": np.asarray(cell, dtype=float),
        "dims": np.asarray(dims, dtype=int),
        "elements": list(elements),
        "frac": np.asarray(frac, dtype=float) % 1.0,
        "site_ids": None if site_ids is None else np.asarray(site_ids, dtype=int),
    }
