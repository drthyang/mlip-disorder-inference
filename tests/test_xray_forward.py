"""X-ray forward model (md_run.xray_*): RMCProfile's convention, pinned.

Public tests use analytic known answers (form-factor normalization, weight
closure, transform consistency). The private-data tests reproduce
RMCProfile's OWN outputs for GaTa4Se8 config 1 — the X-ray total F(Q) from
its partials, the partials from its g(r), and its box-convolved data column —
and skip when data/ is absent (the data stays private).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import md_run as m2

REPO = Path(__file__).resolve().parent.parent
ENS = REPO / "data/ensemble_20A_5K"
N_ATOMS = {"Ga": 2048, "Ta": 8192, "Se": 16384}
BOX = 82.8504                       # Å, the 8×8×8 RMC box edge
QDAMP = 0.03894923941186408         # RESOLUTION_CORRECTION in the GTS .dat


def _symbols():
    return [el for el, n in N_ATOMS.items() for _ in range(n)]


# ------------------------------------------------------------ public tests

@pytest.mark.parametrize("el,Z", [("Ga", 31), ("Ta", 73), ("Se", 34),
                                  ("Cu", 29)])
def test_form_factor_is_Z_at_Q0(el, Z):
    assert m2.xray_f0(el, np.array([0.0]))[0] == pytest.approx(Z, abs=0.05)


def test_form_factor_decreases_with_Q():
    f = m2.xray_f0("Ta", np.linspace(0, 27, 50))
    assert np.all(np.diff(f) < 0)


def test_unknown_element_raises():
    with pytest.raises(SystemExit, match="form factor"):
        m2.xray_f0("Xx", np.array([1.0]))


def test_xray_weights_sum_to_one_at_every_Q():
    Q = np.linspace(0.5, 30, 60)
    w, species = m2.xray_weights(["Ga"] * 4 + ["Ta"] * 16 + ["Se"] * 32, Q)
    assert species == ["Ga", "Se", "Ta"]
    assert np.allclose(sum(w.values()), 1.0)
    # heavy Ta carries far more X-ray than neutron contrast (b are ~equal)
    wn, _ = m2.neutron_weights(["Ga"] * 4 + ["Ta"] * 16 + ["Se"] * 32)
    x_ratio = w[("Ta", "Ta")][0] / w[("Se", "Se")][0]
    n_ratio = wn[("Ta", "Ta")] / wn[("Se", "Se")]
    assert x_ratio > 5 * n_ratio


def test_partial_fq_matches_gr_to_fq_without_damping():
    r = np.arange(0.01, 20.0, 0.01)
    g = 1.0 + 0.5 * np.exp(-(r - 2.5)**2 / 0.01) - (r < 2.0)
    Q = np.linspace(0.8, 20, 40)
    rho0 = 0.05
    got = m2.partial_fq(r, {("A", "A"): g}, Q, rho0)[("A", "A")]
    ref = m2.gr_to_fq(r, g - 1.0, Q, rho0)
    assert np.allclose(got, ref, atol=1e-10)


def test_qdamp_envelope_is_gaussian():
    """A delta-like shell at r0 transformed with qdamp is scaled by
    exp(-(qdamp r0)^2 / 2) — the Gaussian (PDFgui) form."""
    r = np.arange(0.01, 30.0, 0.01)
    r0, q = 20.0, 0.05
    g = np.ones_like(r)
    g[np.argmin(abs(r - r0))] += 50.0
    Q = np.linspace(1, 10, 20)
    und = m2.partial_fq(r, {("A", "A"): g}, Q, 0.05)[("A", "A")]
    dmp = m2.partial_fq(r, {("A", "A"): g}, Q, 0.05, qdamp=q)[("A", "A")]
    # (g-1) is zero except at the shell, so the ratio is the envelope there
    assert np.allclose(dmp / und, np.exp(-0.5 * (q * r0)**2), rtol=1e-3)


def test_xray_fq_monatomic_is_the_partial():
    r = np.arange(0.01, 10.0, 0.01)
    g = {("Cu", "Cu"): 1.0 + np.exp(-(r - 2.55)**2 / 0.005) - (r < 2.2)}
    Q = np.linspace(1, 15, 30)
    tot = m2.xray_fq(r, g, ["Cu"] * 4, Q, 0.085)
    part = m2.partial_fq(r, g, Q, 0.085)[("Cu", "Cu")]
    assert np.allclose(tot, part)


# ----------------------------------------------- private: RMCProfile itself

needs_data = pytest.mark.skipif(
    not (ENS / "GTS_5K_1_FQ1partials.csv").is_file(),
    reason="needs the private GTS RMC outputs in data/")


def _rmcprofile_partials():
    P = np.loadtxt(ENS / "GTS_5K_1_FQ1partials.csv", delimiter=",",
                   usecols=range(7))
    X = np.loadtxt(ENS / "GTS_5K_1_XFQ1.csv", delimiter=",", skiprows=1)
    return P, X


@needs_data
def test_reproduces_rmcprofile_total_from_its_partials():
    P, X = _rmcprofile_partials()
    Q = P[:, 0]
    order = [("Ga", "Ga"), ("Ga", "Ta"), ("Ga", "Se"),
             ("Ta", "Ta"), ("Ta", "Se"), ("Se", "Se")]
    w, _ = m2.xray_weights(_symbols(), Q)
    F = sum(w[tuple(sorted(p))] * P[:, k + 1] for k, p in enumerate(order))
    assert np.abs(F - X[:, 1]).max() < 1e-6


@needs_data
def test_reproduces_rmcprofile_partials_from_its_gr():
    P, _ = _rmcprofile_partials()
    G = np.genfromtxt(ENS / "GTS_5K_1_PDFpartials.csv", delimiter=",",
                      skip_header=1)[:, :7]
    r = G[:, 0]
    order = [("Ga", "Ga"), ("Ga", "Ta"), ("Ga", "Se"),
             ("Ta", "Ta"), ("Ta", "Se"), ("Se", "Se")]
    rho0 = sum(N_ATOMS.values()) / BOX**3
    parts = m2.partial_fq(r, {p: G[:, k + 1] for k, p in enumerate(order)},
                          P[:, 0], rho0, qdamp=QDAMP)
    for k, p in enumerate(order):
        assert np.abs(parts[p] - P[:, k + 1]).max() < 1e-6, p


@needs_data
def test_box_convolution_reproduces_rmcprofile_data_column():
    _, X = _rmcprofile_partials()
    Q, F = m2.parse_fq(ENS / "scale_ft_rmc.fq")
    rho0 = sum(N_ATOMS.values()) / BOX**3
    conv = m2.rmc_box_convolve(Q, F, BOX, rho0)
    _, _, rw = m2.fit_scale_offset(X[:, 2], conv)
    assert rw < 0.005                    # raw data: rw ≈ 0.21
