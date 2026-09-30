"""harmonic_pdf.py — analytic quantum-harmonic g(r), validated on EMT fcc Cu.

Known answers: (1) the FFT displacement correlations equal phonopy's own
supercell correlation matrix (RandomDisplacements.run_correlation_matrix)
for every pair, cross terms included — this pins the eigenvector phase
convention; (2) coordination numbers, shell positions and CORRELATED shell
widths equal those of phonopy-sampled quantum snapshots; (3) the normal-
coordinate variance has the zero-point and equipartition limits. EMT only —
fast, no torch.
"""

from __future__ import annotations

import numpy as np
import pytest

import harmonic_pdf as hp
import md_run

A_CU = 3.61


@pytest.fixture(scope="module")
def cu_phonon():
    from ase.build import bulk
    from ase.calculators.emt import EMT

    at = bulk("Cu", "fcc", a=A_CU, cubic=True)
    return md_run.harmonic_model(at, EMT(), np.array([4, 4, 4]), 0.01, 1e-3,
                                 primitive_matrix="P")


def test_mode_qsq_limits():
    conv = hp._qsq_conversion()
    f = np.array([2.0])
    # T = 0: zero point, ħ/(2ω)
    assert hp.mode_qsq(f, 0.0)[0] == pytest.approx(conv / 2.0 / f[0])
    # high T: equipartition, ⟨Q²⟩ ω² = k_B T  ->  ⟨Q²⟩ ∝ T / f²
    hi = hp.mode_qsq(f, 5000.0)[0] / hp.mode_qsq(f, 2500.0)[0]
    assert hi == pytest.approx(2.0, rel=1e-3)
    # translations and imaginary modes excluded
    assert np.all(hp.mode_qsq(np.array([0.0, -0.3, 0.005]), 5.0) == 0.0)


def test_correlations_equal_phonopy_supercell_matrix(cu_phonon):
    from phonopy.phonon.random_displacements import RandomDisplacements

    T, M = 50.0, 4
    q, V, _ = hp.scaled_modes(cu_phonon, M, T)
    frac = np.asarray(cu_phonon.unitcell.scaled_positions)
    masses = np.asarray(cu_phonon.unitcell.masses)
    rd = RandomDisplacements(cu_phonon.supercell, cu_phonon.primitive,
                             cu_phonon.force_constants)
    rd.run_correlation_matrix(T)
    uu = rd.uu
    spos = np.asarray(cu_phonon.supercell.scaled_positions) * M
    site = np.array([int(np.argmin((((p - frac + 0.5) % 1 - 0.5)**2).sum(1)))
                     for p in spos])
    cell = np.rint(spos - frac[site]).astype(int) % M
    U = hp.site_covariances(V, masses)
    for j in range(len(frac)):
        row = hp.row_correlations(q, V, frac, masses, j, M)
        i0 = np.where((site == j) & np.all(cell == 0, axis=1))[0][0]
        mine = row[cell[:, 0], cell[:, 1], cell[:, 2], site]
        assert np.abs(mine - uu[i0]).max() < 1e-12
        assert np.allclose(U[j], uu[i0, i0], atol=1e-14)


def test_shells_match_quantum_snapshots(cu_phonon):
    T, dr, r_max = 300.0, 0.005, 7.0
    r, g, info = hp.harmonic_partials(cu_phonon, T, M=4, r_max=r_max, dr=dr,
                                      log=None)
    snaps, ideal = md_run.quantum_snapshots(cu_phonon, 300, T, seed=1)
    rs, gs = md_run.pair_histograms([s.get_positions() for s in snaps],
                                    ideal.get_chemical_symbols(),
                                    ideal.cell.array, r_max, dr)
    rho = 4 / A_CU**3
    n_box = len(ideal)
    for lo, hi, n_expect in [(2.0, 2.9, 12), (3.2, 4.1, 6), (4.1, 4.8, 24)]:
        stats = []
        for rr, gg in ((r, g[("Cu", "Cu")]), (rs, gs[("Cu", "Cu")])):
            m = (rr > lo) & (rr < hi)
            w = 4 * np.pi * rho * rr[m]**2 * gg[m] * dr
            mu = (w * rr[m]).sum() / w.sum()
            stats.append((w.sum(), mu,
                          np.sqrt((w * (rr[m] - mu)**2).sum() / w.sum())))
        (nh, mh, sh), (ns, ms, ss) = stats
        # exact crystal (5e-3: the windows catch neighbouring shells' tails)
        assert nh == pytest.approx(n_expect, rel=5e-3)
        # the finite box normalizes by N(N-1), the crystal by N^2
        assert ns * (n_box - 1) / n_box == pytest.approx(nh, rel=5e-3)
        assert mh == pytest.approx(ms, abs=2e-3)
        assert sh == pytest.approx(ss, rel=0.02)
    # correlated motion: the first shell is much narrower than 2 × u_rms²
    m = (r > 2.0) & (r < 2.9)
    w = r[m]**2 * g[("Cu", "Cu")][m]
    sd1 = np.sqrt((w * (r[m] - 2.553)**2).sum() / w.sum())
    assert sd1 < 0.9 * np.sqrt(2) * info["u_rms"][0]


def test_g_tends_to_one_and_is_normalized(cu_phonon):
    r, g, _ = hp.harmonic_partials(cu_phonon, 5.0, M=8, r_max=25.0,
                                   dr=0.01, log=None)
    m = (r > 15) & (r < 25)
    # at 5 K the peaks are sharp; the running integral of ρ g 4πr² tracks
    # the ideal-gas count
    rho = 4 / A_CU**3
    n_cum = np.cumsum(4 * np.pi * rho * r**2 * g[("Cu", "Cu")] * 0.01)
    n_gas = 4 / 3 * np.pi * rho * r**3
    assert np.abs(n_cum[m] / n_gas[m] - 1).max() < 0.05


def test_needs_unit_cell_as_primitive():
    from ase.build import bulk
    from ase.calculators.emt import EMT

    at = bulk("Cu", "fcc", a=A_CU, cubic=True)
    ph = md_run.harmonic_model(at, EMT(), np.array([2, 2, 2]), 0.01, 1e-3)
    with pytest.raises(ValueError, match="primitive_matrix"):
        hp.scaled_modes(ph, 2, 5.0)
