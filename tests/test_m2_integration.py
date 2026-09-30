"""M2 integration tests (slow, EMT): quantum sampling statistics and the full
sample-mode pipeline on the Cu fcc fixture.

The quantum-MSD test is the physics guard for design decision D1: snapshot
displacement statistics must follow the quantum (zero-point-including)
harmonic distribution, not classical equipartition.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import md_run as m2
from fixtures.make_synthetic_ensemble import make_fcc_cu_ensemble

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cu_phonon():
    """EMT-relaxed Cu cell + phonopy harmonic model (3x3x3)."""
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter

    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    atoms.calc = EMT()
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-4, steps=500)
    FIRE(atoms, logfile=None).run(fmax=1e-4, steps=200)
    phonon = m2.harmonic_model(atoms, EMT(), np.array([3, 3, 3]), 0.03, 1e-3)
    return atoms, phonon


def _snapshot_msd(phonon, T, n=48, seed=0):
    snaps, ideal = m2.quantum_snapshots(phonon, n, T, seed)
    d = np.array([s.get_positions() - ideal.get_positions() for s in snaps])
    return float((d**2).sum(axis=2).mean())


def _analytic_msd(phonon, T):
    """Phonopy thermal displacements: sum over xyz of <u^2>, A^2."""
    phonon.run_mesh([8, 8, 8], with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_thermal_displacements(temperatures=[T])
    td = phonon.thermal_displacements.thermal_displacements  # (1, natom*3)
    return float(td[0].reshape(-1, 3).sum(axis=1).mean())


def test_quantum_snapshots_match_analytic_msd(cu_phonon):
    """Sampled MSD tracks the quantum harmonic <u^2> at low AND high T."""
    _, phonon = cu_phonon
    for T in (5.0, 300.0):
        sampled = _snapshot_msd(phonon, T)
        analytic = _analytic_msd(phonon, T)
        assert sampled == pytest.approx(analytic, rel=0.25), (T, sampled,
                                                              analytic)


def test_zero_point_dominates_at_low_T(cu_phonon):
    """At 5 K the sampled MSD is zero-point-dominated: far above the
    classical equipartition prediction, and a sizable fraction of 300 K."""
    _, phonon = cu_phonon
    msd5 = _snapshot_msd(phonon, 5.0)
    msd300 = _snapshot_msd(phonon, 300.0)
    assert msd5 > 0.10 * msd300          # classical would give ~5/300 = 0.017
    assert msd5 > 3 * (5.0 / 300.0) * msd300


@pytest.fixture(scope="module")
def m2_run(tmp_path_factory):
    """Full sample-mode run on the Cu fixture (EMT, small box)."""
    root = tmp_path_factory.mktemp("m2")
    ens = root / "ens"
    out = root / "out"
    make_fcc_cu_ensemble(ens, n_configs=4, dims=(2, 2, 2), sigma=0.06, seed=0)
    argv = ["sample", str(ens), "--calc", "emt", "-T", "50",
            "--rmax", "5.0", "--dr", "0.02", "--nsnapshots", "12",
            "--cutoff", "5.0",  # must stay below half the 3x3x3 box (5.4 A)
            "--npoints", "11", "-o", str(out)]
    try:
        import hiphive  # noqa: F401
    except ImportError:
        argv.append("--no-band-t")
    m2.main(argv)
    return out


def test_m2_outputs_exist(m2_run):
    for name in ("closure.json", "gr_sim.dat", "sq_sim.dat",
                 "relaxed_expt.cif"):
        assert (m2_run / name).is_file(), name


def test_m2_gr_first_peak_at_fcc_shell(m2_run):
    from ase.io import read
    a = read(str(m2_run / "relaxed_expt.cif")).cell.lengths()[0]
    arr = np.loadtxt(m2_run / "gr_sim.dat")
    r, G = arr[:, 0], arr[:, 1]
    peak = r[np.argmax(G)]
    assert peak == pytest.approx(a / np.sqrt(2), abs=0.1)


def test_m2_band_t_close_to_harmonic(m2_run):
    pytest.importorskip("hiphive")
    closure = json.loads((m2_run / "closure.json").read_text())
    assert closure["band_T"]["written"] is True
    assert (m2_run / "band_T.yaml").is_file()
    # Effective bands track the harmonic ones to ~10%: zero-point sampling
    # (u_rms ~ 0.05 A) genuinely renormalises the 2nd-order fit against EMT
    # anharmonicity, so exact agreement is neither expected nor desired —
    # the guard is against a broken fit (order-of-magnitude errors).
    assert closure["band_T"]["max_abs_dw_THz_vs_harmonic"] < 1.0


def test_m2_self_closure_rw_is_small(m2_run, tmp_path):
    """Using the simulation's own F(Q) as 'data' must give scale~1, Rw~0."""
    arr = np.loadtxt(m2_run / "sq_sim.dat")
    Q, F = arr[:, 0], arr[:, 1]
    p = tmp_path / "self.fq"
    body = "\n".join(f"  {q:.4f}  {f:.8f}" for q, f in zip(Q, F))
    p.write_text(f"  {len(Q)}\nself-closure\n{body}\n")
    Qd, Fd = m2.parse_fq(p)
    arr2 = np.loadtxt(m2_run / "gr_sim.dat")
    r, G = arr2[:, 0], arr2[:, 1]
    from ase.io import read
    atoms = read(str(m2_run / "relaxed_expt.cif"))
    # recompute rho0 for the sampling supercell = same as unit cell density
    rho0 = len(atoms) / atoms.get_volume()
    Fs = m2.gr_to_fq(r, G, Qd, rho0)
    s, o, rw = m2.fit_scale_offset(Fd, Fs)
    assert s == pytest.approx(1.0, abs=0.01)
    assert abs(o) < 0.01
    assert rw < 0.01


def test_m2_xray_closure_cli(m2_run, tmp_path):
    """`--radiation xray` end to end. Monatomic Cu has X-ray and neutron
    weights both ≡ 1, so the neutron run's own F(Q) is exact X-ray data for
    the same snapshots (same seed): the box-matched default must close to
    scale 1 / Rw 0, and gr_partials.npz must reproduce closure.json."""
    arr = np.loadtxt(m2_run / "sq_sim.dat")
    data = tmp_path / "self.fq"
    body = "\n".join(f"  {q:.4f}  {f:.10f}" for q, f in arr[:, :2])
    data.write_text(f"  {len(arr)}\nself-closure\n{body}\n")
    out = tmp_path / "xray"
    m2.main(["sample", str(m2_run.parent / "ens"), "--calc", "emt",
             "-T", "50", "--rmax", "5.0", "--dr", "0.02", "--nsnapshots",
             "12", "--no-band-t", "--data", str(data), "--radiation", "xray",
             "-o", str(out)])
    closure = json.loads((out / "closure.json").read_text())
    fit = closure["closure_fit"]
    assert (fit["radiation"], fit["compare"]) == ("xray", "box")
    assert fit["box_length_A"] == pytest.approx(10.0)
    assert fit["scale"] == pytest.approx(1.0, abs=1e-4)
    assert fit["Rw_Q"] < 1e-4 and fit["Rw_r"] < 1e-4
    assert closure["structure"]["symprec"] == pytest.approx(1e-3)

    z = np.load(out / "gr_partials.npz")
    symbols = [s for s, n in zip(z["species"], z["counts"]) for _ in range(n)]
    g = {tuple(k[2:].split("_")): z[k] for k in z.files if k.startswith("g_")}
    Qd, Fd = m2.parse_fq(data)
    again, _ = m2.closure_fit(z["r"], g, symbols, float(z["rho0"]), Qd, Fd,
                              "xray", 0.0, "box")
    assert again["Rw_Q"] == pytest.approx(fit["Rw_Q"], rel=1e-9, abs=1e-12)
    assert again["scale"] == pytest.approx(fit["scale"], rel=1e-12)
