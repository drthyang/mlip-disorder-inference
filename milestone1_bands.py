#!/usr/bin/env python3
"""milestone1_bands.py — correct phonon bands from an RMC ensemble via a foundation MLIP.

Pipeline (milestone 1 of the mlip-quantum-thermal project):

    .rmc6f ensemble  ->  fold + circular-average to one unit cell
                     ->  (optional) spglib symmetrization
                     ->  MLIP relaxation (MACE-MP-0 by default, float64)
                     ->  phonopy finite displacements
                     ->  band.yaml (+ relaxed.cif + summary.json)

The band.yaml is phonopy-standard and loads directly in viewer/ (and in any
phonopy-aware tool). The relaxed.cif is the equilibrium reference for
displacement analysis.

Usage examples:
    python milestone1_bands.py run_dir/                 # all *.rmc6f in dir
    python milestone1_bands.py cfg_001.rmc6f cfg_002.rmc6f --device cuda
    python milestone1_bands.py run_dir/ --dim 3 3 2 --calc chgnet
    python milestone1_bands.py run_dir/ --ref parent.cif --no-symmetrize

Notes:
  * Assumes an RMCProfile diagonal supercell (Nx Ny Nz of one unit cell).
  * Phonons require an ordered cell; mixed-occupancy sites are reduced to
    the majority element with a loud warning.
  * Replace parse_rmc6f() with your own parser from the archived engines
    if your files deviate from the tolerant format assumed here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TOL_LADDER = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)  # symprec ladder for reporting


# ----------------------------------------------------------------------------
# rmc6f parsing
# ----------------------------------------------------------------------------

def _clean_element(token: str) -> str:
    """'O2-' -> 'O', 'MN' -> 'Mn'. Keep at most two alphabetic characters."""
    letters = "".join(c for c in token if c.isalpha())[:2]
    if not letters:
        raise ValueError(f"cannot read an element symbol from {token!r}")
    return letters[0].upper() + letters[1:].lower()


def _cell_from_parameters(a, b, c, al, be, ga):
    """Standard crystallographic cell construction (angles in degrees)."""
    al, be, ga = (math.radians(x) for x in (al, be, ga))
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([va, vb, [cx, cy, cz]])


def parse_rmc6f(path: Path):
    """Tolerant .rmc6f reader.

    Returns dict with:
        cell      (3,3) supercell lattice vectors, Angstrom (rows)
        dims      (3,)  supercell repetitions (Nx, Ny, Nz)
        elements  list[str], length N
        frac      (N,3) fractional coordinates in the SUPERCELL frame
        site_ids  (N,)  original unit-cell site numbers, or None if absent
    """
    dims = None
    cell = None
    cell_params = None
    elements, frac, site_ids = [], [], []
    have_site_ids = True

    lines = path.read_text(errors="replace").splitlines()
    i = 0
    in_atoms = False
    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()
        if not in_atoms:
            if low.startswith("supercell"):
                ints = [int(t) for t in line.replace(":", " ").split()
                        if t.lstrip("+-").isdigit()]
                if len(ints) >= 3:
                    dims = np.array(ints[:3], dtype=int)
            elif low.startswith("lattice vectors"):
                rows = []
                for j in range(1, 4):
                    rows.append([float(t) for t in lines[i + j].split()[:3]])
                cell = np.array(rows)
                i += 3
            elif low.startswith("cell"):
                nums = [float(t) for t in line.replace(":", " ").split()
                        if _is_float(t)]
                if len(nums) >= 6:
                    cell_params = nums[:6]
            elif low.startswith("atoms"):
                in_atoms = True
        else:
            toks = line.split()
            if len(toks) >= 5 and toks[0].lstrip("+-").isdigit():
                # element = first alphabetic token after the index
                k = 1
                while k < len(toks) and not any(c.isalpha() for c in toks[k]):
                    k += 1
                elem = _clean_element(toks[k])
                k += 1
                # skip an optional bracketed label like [1]
                while k < len(toks) and toks[k].startswith("["):
                    k += 1
                # next three float-parsable tokens are the coordinates
                coords = []
                while k < len(toks) and len(coords) < 3:
                    if _is_float(toks[k]):
                        coords.append(float(toks[k]))
                    k += 1
                if len(coords) != 3:
                    i += 1
                    continue
                trailing = [int(t) for t in toks[k:] if t.lstrip("+-").isdigit()]
                elements.append(elem)
                frac.append(coords)
                if trailing:
                    site_ids.append(trailing[0])
                else:
                    have_site_ids = False
        i += 1

    if cell is None:
        if cell_params is None:
            raise ValueError(f"{path}: no lattice information found")
        cell = _cell_from_parameters(*cell_params)
    if dims is None:
        raise ValueError(f"{path}: no 'Supercell dimensions' line found")
    if not elements:
        raise ValueError(f"{path}: no atom lines parsed")

    return {
        "cell": cell,
        "dims": dims,
        "elements": elements,
        "frac": np.array(frac) % 1.0,
        "site_ids": np.array(site_ids) if have_site_ids else None,
    }


def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------------
# fold + average to one unit cell
# ----------------------------------------------------------------------------

def fold_and_average(configs, ref_cif: Path | None):
    """Circular-average folded site coordinates across cells and configs.

    Sites are grouped by the rmc6f site-id column when present, otherwise by
    nearest site of a reference CIF (required in that case).
    Returns (unit_cell (3,3), site_elements, site_frac (M,3), report dict).
    """
    dims = configs[0]["dims"]
    cell = configs[0]["cell"]
    for c in configs[1:]:
        if not np.array_equal(c["dims"], dims):
            raise ValueError("configurations disagree on supercell dimensions")
    unit_cell = cell / dims[:, None]  # rows a_s = N_i * a_u  (diagonal supercell)

    ref_sites = None
    if any(c["site_ids"] is None for c in configs):
        if ref_cif is None:
            raise SystemExit(
                "rmc6f files carry no site-id column; pass --ref parent.cif "
                "so atoms can be assigned to sites."
            )
        from ase.io import read as ase_read
        ref = ase_read(str(ref_cif))
        ref_sites = ref.get_scaled_positions() % 1.0

    zsum = defaultdict(lambda: np.zeros(3, dtype=complex))
    counts = defaultdict(int)
    elem_votes = defaultdict(Counter)

    for cfg in configs:
        unit_frac = (cfg["frac"] * dims) % 1.0  # fold into the unit cell
        if cfg["site_ids"] is not None:
            ids = cfg["site_ids"]
        else:
            d = unit_frac[:, None, :] - ref_sites[None, :, :]
            d -= np.round(d)  # minimum image in fractional space
            ids = np.argmin(np.einsum("ijk,ijk->ij", d, d), axis=1)
        phases = np.exp(2j * np.pi * unit_frac)
        for sid, ph, el in zip(ids, phases, cfg["elements"]):
            zsum[sid] += ph
            counts[sid] += 1
            elem_votes[sid][el] += 1

    site_elements, site_frac, mixed = [], [], []
    for sid in sorted(zsum):
        mean = (np.angle(zsum[sid] / counts[sid]) / (2 * np.pi)) % 1.0
        votes = elem_votes[sid]
        el, n = votes.most_common(1)[0]
        if len(votes) > 1:
            mixed.append((sid, dict(votes)))
        site_elements.append(el)
        site_frac.append(mean)

    report = {
        "n_configs": len(configs),
        "supercell_dims": dims.tolist(),
        "n_sites": len(site_frac),
        "mixed_occupancy_sites": mixed,
    }
    return unit_cell, site_elements, np.array(site_frac), report


# ----------------------------------------------------------------------------
# symmetry, relaxation, phonons
# ----------------------------------------------------------------------------

def spacegroup_ladder(atoms):
    import spglib
    cell = (atoms.cell.array, atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    return {f"{p:g}": spglib.get_spacegroup(cell, symprec=p) for p in TOL_LADDER}


def symmetrize(atoms, symprec):
    import spglib
    from ase import Atoms
    cell = (atoms.cell.array, atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    std = spglib.standardize_cell(cell, to_primitive=False,
                                  no_idealize=False, symprec=symprec)
    if std is None:
        print(f"  ! spglib could not symmetrize at symprec={symprec}; "
              "continuing unsymmetrized")
        return atoms
    lat, pos, nums = std
    return Atoms(numbers=nums, scaled_positions=pos, cell=lat, pbc=True)


def symmetrize_lattice(atoms, symprec):
    """Snap the cell metric onto its exact space-group symmetry.

    A relaxed cell carries ~fmax-level numerical noise in its lattice vectors
    (order 1e-5 Å). That is harmless for the force constants, but it breaks
    the band path: ``Phonopy.auto_band_structure()`` takes no ``symprec`` and
    seekpath therefore runs at its hard-coded 1e-5, so a cubic cell can be
    classified as triclinic (P1) and the "standard path" becomes the aP one —
    wrong labels, and the path never visits the real high-symmetry points.

    The metric tensor g = A·Aᵀ (Å²) obeys Rᵀ g R = g for every fractional
    rotation R of the true space group, so the noise is removed by averaging
    g over the group. The cell is then rebuilt as A' = S·A with
    S = g_sym^{1/2}·g^{-1/2}, which reproduces g_sym exactly (S g Sᵀ = g_sym)
    and reduces to the identity as g → g_sym, so the Cartesian frame is
    preserved. Fractional coordinates, atom order and atom count are
    untouched — nothing downstream that indexes atoms can be disturbed.

    Parameters
    ----------
    atoms : ase.Atoms
        Cell to idealize (modified copy returned).
    symprec : float
        Tolerance handed to spglib when finding the rotations, Å.

    Returns
    -------
    atoms_sym : ase.Atoms
        Copy with the symmetrized cell; scaled positions identical.
    report : dict
        ``max_lattice_shift_A`` (largest change in any lattice-vector
        component) and ``n_operations`` used. Both are zero/1 when spglib
        finds no symmetry, in which case the cell is returned unchanged.
    """
    import spglib

    A = np.asarray(atoms.cell.array, dtype=float)
    frac = atoms.get_scaled_positions()
    cell = (A, frac, atoms.get_atomic_numbers())
    sym = spglib.get_symmetry(cell, symprec=symprec)
    if sym is None or len(sym["rotations"]) == 0:
        return atoms.copy(), {"max_lattice_shift_A": 0.0, "n_operations": 0}

    rots = np.asarray(sym["rotations"], dtype=float)
    g = A @ A.T
    g_sym = np.einsum("nji,jk,nkl->il", rots, g, rots) / len(rots)

    def _spd_pow(m, p):
        w, v = np.linalg.eigh(m)
        return (v * np.power(np.maximum(w, 1e-300), p)) @ v.T

    S = _spd_pow(g_sym, 0.5) @ _spd_pow(g, -0.5)
    A_sym = S @ A

    out = atoms.copy()
    out.set_cell(A_sym)
    out.set_scaled_positions(frac)
    return out, {"max_lattice_shift_A": float(np.abs(A_sym - A).max()),
                 "n_operations": int(len(rots))}


def get_calculator(name, device, model_size):
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model=model_size, device=device,
                       default_dtype="float64")
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator()
    if name == "emt":  # dependency-free smoke test only
        from ase.calculators.emt import EMT
        return EMT()
    raise SystemExit(f"unknown calculator {name!r}")


def relax(atoms, calc, fmax):
    from ase.optimize import FIRE
    try:
        from ase.filters import FrechetCellFilter as CellFilter
    except ImportError:  # older ASE
        from ase.constraints import ExpCellFilter as CellFilter
    atoms.calc = calc
    print(f"  relaxing cell + positions to fmax < {fmax} eV/A ...")
    FIRE(CellFilter(atoms), logfile="-").run(fmax=fmax, steps=800)
    FIRE(atoms, logfile="-").run(fmax=fmax, steps=400)
    fres = float(np.abs(atoms.get_forces()).max())
    print(f"  residual max |F| = {fres:.2e} eV/A")
    return atoms, fres


def auto_dim(atoms, target=12.0, max_atoms=800):
    lengths = atoms.cell.lengths()
    dim = np.maximum(1, np.ceil(target / lengths)).astype(int)
    if int(np.prod(dim)) * len(atoms) > max_atoms:
        print(f"  ! auto supercell {dim.tolist()} gives "
              f"{int(np.prod(dim)) * len(atoms)} atoms; consider --dim")
    return dim


def phonopy_bands(atoms, calc, dim, displacement, npoints, eigenvectors, outdir,
                  symprec):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from ase import Atoms

    # Remove the relaxation's numerical noise from the cell metric first: the
    # band PATH comes from seekpath at its hard-coded symprec=1e-5, which no
    # phonopy argument reaches (see symmetrize_lattice).
    atoms, lat_report = symmetrize_lattice(atoms, symprec)
    print(f"  lattice symmetrized ({lat_report['n_operations']} ops, "
          f"max shift {lat_report['max_lattice_shift_A']:.2e} A)")

    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                        cell=atoms.cell.array,
                        scaled_positions=atoms.get_scaled_positions())
    # Use the same physical symmetry tolerance as the pre-relax symmetrization.
    # phonopy's default (1e-5) is tighter than the ~fmax-level numerical noise a
    # relaxed cell carries, so at 1e-5 an fcc cell can read as P1 and
    # primitive_matrix="auto" fails to reduce it -> folded bands (e.g. 12 modes
    # for fcc instead of 3). symprec here recovers the true primitive.
    phonon = Phonopy(unit, supercell_matrix=np.diag(dim),
                     primitive_matrix="auto", symprec=symprec)
    phonon.generate_displacements(distance=displacement)
    scells = phonon.supercells_with_displacements
    print(f"  {len(scells)} displaced supercells "
          f"({len(scells[0])} atoms each), d = {displacement} A")

    forces = []
    for n, sc in enumerate(scells, 1):
        a = Atoms(symbols=sc.symbols, scaled_positions=sc.scaled_positions,
                  cell=sc.cell, pbc=True)
        a.calc = calc
        t0 = time.time()
        forces.append(a.get_forces())
        print(f"    forces {n}/{len(scells)}  ({time.time() - t0:.1f}s)")
    phonon.forces = np.array(forces)
    phonon.produce_force_constants()

    band_yaml = outdir / "band.yaml"
    phonon.auto_band_structure(npoints=npoints,
                               with_eigenvectors=eigenvectors,
                               write_yaml=True,
                               filename=str(band_yaml))
    # phonopy >=4 replaces get_band_structure_dict() with the band_structure
    # property; .frequencies is a per-path-segment list of (npoints, nbands)
    # arrays in THz.
    bs = phonon.band_structure
    fmin = float(min(np.min(f) for f in bs.frequencies))
    return phonon, band_yaml, fmin, lat_report


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def collect_inputs(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.glob("*.rmc6f")))
        elif p.suffix.lower() == ".rmc6f":
            files.append(p)
    if not files:
        raise SystemExit("no .rmc6f files found in the given paths")
    return files


def read_moves_generated(path: Path):
    """Number of RMC moves generated, from the rmc6f header (dimensionless count).

    Reads only the header (stops at the ``Atoms:`` line), so it is cheap even
    for multi-MB configurations. Returns ``None`` when the file carries no
    ``Number of moves generated`` line (e.g. synthetic fixtures).
    """
    with Path(path).open(errors="replace") as fh:
        for i, line in enumerate(fh):
            low = line.strip().lower()
            if low.startswith("atoms"):
                break
            if low.startswith("number of moves generated"):
                for tok in line.replace(":", " ").split():
                    if tok.isdigit():
                        return int(tok)
            if i > 500:  # header should never be this long
                break
    return None


def drop_nonconverged(files):
    """Split inputs into (converged, dropped) by the rmc6f header move count.

    A configuration is dropped only when its header explicitly reports
    ``Number of moves generated: 0`` — an RMC run that never started. Files
    without the header line are kept (treated as converged), so synthetic or
    externally generated ensembles pass through untouched. Order is preserved.
    """
    kept, dropped = [], []
    for f in files:
        (dropped if read_moves_generated(f) == 0 else kept).append(f)
    return kept, dropped


def select_configs(files, stride=1, max_configs=None, seed=0):
    """Deterministically subsample a sorted list of input configurations.

    Ergonomics for large (500+) ensembles. First takes every ``stride``-th
    file (a cheap, ordered decimation), then, if more than ``max_configs``
    remain, draws a random sample of ``max_configs`` of them seeded by
    ``seed``. The returned list preserves the original order and is fully
    reproducible for a given (stride, max_configs, seed).

    Parameters
    ----------
    files : list[pathlib.Path]
        Candidate inputs (assumed already sorted, e.g. from collect_inputs).
    stride : int
        Keep every ``stride``-th file (>= 1).
    max_configs : int or None
        Cap on the number of configurations after striding; ``None`` means no
        cap. Must be >= 1 when given.
    seed : int
        Seed for the random subsample used only when the cap actually bites.

    Returns
    -------
    list[pathlib.Path]
    """
    if stride < 1:
        raise SystemExit("--stride must be >= 1")
    selected = list(files[::stride])
    if max_configs is not None:
        if max_configs < 1:
            raise SystemExit("--max-configs must be >= 1")
        if len(selected) > max_configs:
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(selected), size=max_configs, replace=False)
            selected = [selected[i] for i in sorted(keep)]
    return selected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+",
                    help=".rmc6f files and/or directories containing them")
    ap.add_argument("--ref", type=Path, default=None,
                    help="parent CIF for site assignment when rmc6f lacks site ids")
    ap.add_argument("--skip-nonconverged", action="store_true",
                    help="drop configs whose rmc6f header reports 'Number of "
                    "moves generated: 0' (RMC runs that never started)")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every k-th input configuration (ordered decimation)")
    ap.add_argument("--max-configs", type=int, default=None,
                    help="cap the number of configs used; seeded random subsample "
                    "of what remains after --stride (for 500+ config ensembles)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --max-configs random subsampling")
    ap.add_argument("--calc", default="mace", choices=["mace", "chgnet", "emt"])
    ap.add_argument("--model", default="medium",
                    choices=["small", "medium", "large"],
                    help="MACE-MP-0 model size (mace only)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--symprec", type=float, default=1e-3,
                    help="tolerance for spglib symmetrization before relaxing")
    ap.add_argument("--no-symmetrize", action="store_true")
    ap.add_argument("--dim", type=int, nargs=3, default=None,
                    metavar=("NX", "NY", "NZ"),
                    help="phonopy supercell (default: auto, >= ~12 A per axis)")
    ap.add_argument("--fmax", type=float, default=1e-3)
    ap.add_argument("--displacement", type=float, default=0.03,
                    help="finite-displacement distance, A. Default 0.03 (not "
                    "phonopy's DFT-tuned 0.01): at 0.01 A the MLIP forces sit "
                    "near the model noise floor and spurious imaginary modes "
                    "appear just off Gamma even for a stable metal.")
    ap.add_argument("--npoints", type=int, default=101)
    ap.add_argument("--no-eigenvectors", action="store_true",
                    help="omit eigenvectors (much smaller band.yaml)")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("m1_out"))
    args = ap.parse_args(argv)

    from ase import Atoms
    from ase.io import write as ase_write

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    all_files = collect_inputs(args.inputs)
    dropped = []
    if args.skip_nonconverged:
        all_files, dropped = drop_nonconverged(all_files)
        if dropped:
            print(f"  dropped {len(dropped)} non-converged config(s) "
                  "(0 RMC moves):")
            for f in dropped:
                print(f"      {f.name}")
        if not all_files:
            raise SystemExit("all input configurations are non-converged")
    files = select_configs(all_files, args.stride, args.max_configs, args.seed)
    if len(files) != len(all_files):
        print(f"  subsampled {len(files)}/{len(all_files)} configs "
              f"(stride={args.stride}, max_configs={args.max_configs}, "
              f"seed={args.seed})")
    print(f"[1/5] parsing {len(files)} configuration(s)")
    configs = [parse_rmc6f(f) for f in files]

    print("[2/5] folding + circular averaging")
    unit_cell, elems, frac, fold_report = fold_and_average(configs, args.ref)
    if fold_report["mixed_occupancy_sites"]:
        print("  ! mixed-occupancy sites reduced to majority element "
              "(phonons need an ordered cell):")
        for sid, votes in fold_report["mixed_occupancy_sites"]:
            print(f"      site {sid}: {votes}")
    atoms = Atoms(symbols=elems, scaled_positions=frac,
                  cell=unit_cell, pbc=True)
    sg_raw = spacegroup_ladder(atoms)
    print("  space group vs tolerance:", sg_raw)

    if not args.no_symmetrize:
        atoms = symmetrize(atoms, args.symprec)
        print("  after symmetrization:", spacegroup_ladder(atoms))

    print(f"[3/5] MLIP relaxation ({args.calc})")
    calc = get_calculator(args.calc, args.device, args.model)
    atoms, fres = relax(atoms, calc, args.fmax)
    sg_relaxed = spacegroup_ladder(atoms)
    print("  after relaxation:", sg_relaxed)
    relaxed_cif = outdir / "relaxed.cif"
    ase_write(str(relaxed_cif), atoms)

    print("[4/5] phonopy finite displacements")
    dim = np.array(args.dim) if args.dim else auto_dim(atoms)
    print(f"  supercell {dim.tolist()}")
    phonon, band_yaml, fmin, lat_report = phonopy_bands(
        atoms, calc, dim, args.displacement, args.npoints,
        not args.no_eigenvectors, outdir, args.symprec)

    print("[5/5] summary")
    stable = fmin > -0.05  # THz; small negative acoustic dips near Gamma are noise
    if not stable:
        print(f"  ! imaginary modes present (min frequency {fmin:.3f} THz): "
              "the averaged/symmetrized cell is not a local minimum of the "
              "MLIP surface -- a candidate frozen distortion. Re-run on the "
              "distorted subgroup cell, or treat with effective-FC methods "
              "at temperature (milestones 2-3).")
    summary = {
        "inputs": [str(f) for f in files],
        "sampling": {"n_available": len(all_files), "n_used": len(files),
                     "n_dropped_nonconverged": len(dropped),
                     "stride": args.stride, "max_configs": args.max_configs,
                     "seed": args.seed},
        "fold": fold_report,
        "spacegroup_ladder_raw": sg_raw,
        "spacegroup_ladder_relaxed": sg_relaxed,
        "calculator": {"name": args.calc, "model": args.model,
                       "device": args.device, "dtype": "float64"},
        "relax": {"fmax_target": args.fmax, "residual_fmax": fres},
        "phonopy": {"supercell": dim.tolist(),
                    "displacement_A": args.displacement,
                    "npoints": args.npoints,
                    "eigenvectors": not args.no_eigenvectors,
                    "lattice_symmetrization": lat_report},
        "min_band_frequency_THz": fmin,
        "dynamically_stable_at_0K": bool(stable),
        "outputs": {"band_yaml": str(band_yaml),
                    "relaxed_cif": str(relaxed_cif)},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {band_yaml}, {relaxed_cif}, {outdir/'summary.json'}")
    print("  -> open band.yaml in viewer/ (cd viewer && npm run dev); "
          "use relaxed.cif as the displacement reference.")


if __name__ == "__main__":
    sys.exit(main())
