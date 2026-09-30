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


# ------------------------------------------------------ static offsets

def _cu_static(delta=0.05):
    """A 1×1×2 static pattern on the 4 Cu sites: site-dependent vectors,
    sign alternating between even and odd cells along z. Å."""
    v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    s = np.zeros((1, 1, 2, 4, 3))
    s[0, 0, 0] = delta * v
    s[0, 0, 1] = -delta * v
    return s


def test_uniform_static_offset_changes_nothing(cu_phonon):
    r, g0, _ = hp.harmonic_partials(cu_phonon, 5.0, M=4, r_max=8.0, log=None)
    shift = np.broadcast_to([0.03, -0.02, 0.05], (1, 1, 2, 4, 3))
    _, g1, info = hp.harmonic_partials(cu_phonon, 5.0, M=4, r_max=8.0,
                                       static=shift, log=None)
    assert info["static_period"] == [1, 1, 2]
    assert np.abs(g1[("Cu", "Cu")] - g0[("Cu", "Cu")]).max() < 1e-9


def test_static_pattern_matches_snapshots_plus_offsets(cu_phonon):
    """Static + quantum: the analytic g(r) with a 1x1x2 static pattern equals
    pair histograms of phonopy quantum snapshots with the same offsets
    added atom by atom (the independent construction)."""
    T, dr, r_max = 300.0, 0.005, 7.0
    static = _cu_static()
    r, g, _ = hp.harmonic_partials(cu_phonon, T, M=4, r_max=r_max, dr=dr,
                                   static=static, log=None)
    snaps, ideal = md_run.quantum_snapshots(cu_phonon, 300, T, seed=2)
    frac = np.asarray(cu_phonon.unitcell.scaled_positions)
    spos = ideal.get_scaled_positions() * 4
    site = np.array([int(np.argmin((((p - frac + 0.5) % 1 - 0.5)**2).sum(1)))
                     for p in spos])
    cz = np.rint(spos[:, 2] - frac[site, 2]).astype(int) % 2
    offs = static[0, 0, cz, site]
    rs, gs = md_run.pair_histograms([s.get_positions() + offs for s in snaps],
                                    ideal.get_chemical_symbols(),
                                    ideal.cell.array, r_max, dr)
    _, g_plain, _ = hp.harmonic_partials(cu_phonon, T, M=4, r_max=r_max,
                                         dr=dr, log=None)
    rho, n_box = 4 / A_CU**3, len(ideal)
    for lo, hi in [(2.0, 2.95), (3.2, 4.1), (4.1, 4.8)]:
        stats = []
        for rr, gg in ((r, g[("Cu", "Cu")]), (rs, gs[("Cu", "Cu")]),
                       (r, g_plain[("Cu", "Cu")])):
            m = (rr > lo) & (rr < hi)
            w = 4 * np.pi * rho * rr[m]**2 * gg[m] * dr
            mu = (w * rr[m]).sum() / w.sum()
            stats.append((w.sum(), mu,
                          np.sqrt((w * (rr[m] - mu)**2).sum() / w.sum())))
        (nh, mh, sh), (ns, ms, ss), (_, _, s0) = stats
        assert ns * (n_box - 1) / n_box == pytest.approx(nh, rel=5e-3)
        assert mh == pytest.approx(ms, abs=2e-3)
        assert sh == pytest.approx(ss, rel=0.02)
        assert sh > s0 * 1.05          # the static pattern broadens shells


def test_extra_u2_adds_uncorrelated_pair_variance(cu_phonon):
    """An extra isotropic variance u² per site widens every shell by
    exactly 2u² in variance (uncorrelated by construction)."""
    dr, u2 = 0.002, 0.03**2
    r, g0, _ = hp.harmonic_partials(cu_phonon, 300.0, M=4, r_max=6.0, dr=dr,
                                    log=None)
    _, g1, _ = hp.harmonic_partials(cu_phonon, 300.0, M=4, r_max=6.0, dr=dr,
                                    extra_u2=u2, log=None)
    m = (r > 2.0) & (r < 3.1)
    var = []
    for g in (g0, g1):
        w = r[m]**2 * g[("Cu", "Cu")][m]
        mu = (w * r[m]).sum() / w.sum()
        var.append((w * (r[m] - mu)**2).sum() / w.sum())
    assert var[1] - var[0] == pytest.approx(2 * u2, rel=0.03)


# ------------------------------------------------------------- domains

def test_domain_limits(cu_phonon):
    """ξ = ∞ is long-range order; ξ → 0 with an isotropic variant
    covariance s²·I is the incoherent limit, identical to adding s² of
    uncorrelated width; a finite ξ lies between and conserves counts."""
    model = hp.prepare_model(cu_phonon, 300.0, M=4)
    static, s2 = _cu_static(0.05), 0.04**2
    cov = np.broadcast_to(s2 * np.eye(3), (4, 3, 3))
    kw = dict(r_max=8.0, dr=0.005, model=model, log=None)
    r, g_lro, _ = hp.harmonic_partials(None, 300.0, static=static, **kw)
    _, g_inf, _ = hp.harmonic_partials(None, 300.0, static=static,
                                       domain_xi=np.inf, incoherent_cov=cov,
                                       **kw)
    assert np.array_equal(g_inf[("Cu", "Cu")], g_lro[("Cu", "Cu")])
    _, g_0, _ = hp.harmonic_partials(None, 300.0, static=static,
                                     domain_xi=1e-9, incoherent_cov=cov, **kw)
    _, g_w, _ = hp.harmonic_partials(None, 300.0, extra_u2=s2, **kw)
    assert np.abs(g_0[("Cu", "Cu")] - g_w[("Cu", "Cu")]).max() < 1e-9
    _, g_x, _ = hp.harmonic_partials(None, 300.0, static=static,
                                     domain_xi=5.0, incoherent_cov=cov, **kw)
    m = (r > 2.0) & (r < 2.95)                       # first shell
    rho = 4 / A_CU**3
    for g in (g_lro, g_0, g_x):
        n1 = (4 * np.pi * rho * r[m]**2 * g[("Cu", "Cu")][m] * 0.005).sum()
        assert n1 == pytest.approx(12.0, rel=5e-3)
    # the mixture weight at the first shell is exp(-2.55/5) = 0.60
    P = np.exp(-2.553 / 5.0)
    mix = P * g_lro[("Cu", "Cu")][m] + (1 - P) * g_0[("Cu", "Cu")][m]
    assert np.abs(g_x[("Cu", "Cu")][m] - mix).max() < 0.02 * mix.max()


def test_domains_need_incoherent_cov(cu_phonon):
    with pytest.raises(ValueError, match="incoherent_cov"):
        hp.harmonic_partials(cu_phonon, 5.0, M=4, r_max=5.0,
                             static=_cu_static(), domain_xi=10.0, log=None)
