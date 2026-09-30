"""md_run.closure_fit: radiation weighting and box-matched comparison.

Known answers on a synthetic CsCl-type TaSe crystal whose partial g_ab(r)
are analytic lattice sums of Gaussian shells (distinct widths per partial, so
the weighting matters). "Data" are the X-ray F(Q) of that crystal out to
60 Å — the stand-in for an instrument-sharp measurement — and the
"simulation" is the same crystal histogrammed only to 20 Å, as md_run's
finite sampling box does (the GTS closure: Q to 27 Å⁻¹, rmax 20 Å). Heavy Ta against light Se is the GaTa4Se8
contrast problem in miniature: X-ray and neutron weights differ ~2×.
"""

from __future__ import annotations

import numpy as np
import pytest

import md_run as m2

A = 3.2                                  # Å, CsCl cell edge
RHO0 = 2.0 / A**3                        # Å⁻³, two atoms per cell
SIGMA = {("Se", "Se"): 0.15, ("Se", "Ta"): 0.12, ("Ta", "Ta"): 0.09}  # Å
SYMBOLS = ["Ta", "Se"] * 8
Q = np.arange(0.8, 27.0, 0.02)
QDAMP = 0.04


def _cscl_partials(r_max, dr=0.01):
    """Partial g_ab(r) of CsCl TaSe, Gaussian-broadened shells.

    Normalized as md_run.pair_histograms: ρ_b g_ab(r) 4πr² dr counts the b
    atoms around an a atom (ρ_b = 1/A³ for either species).
    """
    r = np.arange(0.5 * dr, r_max, dr)
    n = int(np.ceil((r_max + 1.0) / A)) + 1
    k = np.arange(-n, n + 1)
    lat = np.stack(np.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3) * A

    def shells(vecs, sigma):
        d = np.linalg.norm(vecs, axis=1)
        d, mult = np.unique(np.round(d[(d > 1e-9) & (d < r_max + 1.0)], 8),
                            return_counts=True)
        dens = (mult * np.exp(-0.5 * ((r[:, None] - d) / sigma)**2)
                ).sum(axis=1) / (np.sqrt(2.0 * np.pi) * sigma)
        return dens / (4.0 * np.pi * r**2 / A**3)

    g = {("Se", "Se"): shells(lat, SIGMA[("Se", "Se")]),
         ("Ta", "Ta"): shells(lat, SIGMA[("Ta", "Ta")]),
         ("Se", "Ta"): shells(lat + 0.5 * A, SIGMA[("Se", "Ta")])}
    return r, g


@pytest.fixture(scope="module")
def crystal():
    r_d, g_d = _cscl_partials(60.0)
    F_data = m2.xray_fq(r_d, g_d, SYMBOLS, Q, RHO0, QDAMP)
    r, g = _cscl_partials(20.0, dr=0.02)
    return r, g, F_data


def test_weighted_fq_neutron_is_the_legacy_total_G_route(crystal):
    """Per-partial neutron F(Q) equals the old gr_to_fq(total_G) path."""
    r, g, _ = crystal
    parts = m2.partial_fq(r, g, Q, RHO0)
    new = m2.weighted_fq(parts, SYMBOLS, Q, "neutron")
    old = m2.gr_to_fq(r, m2.total_G(r, g, SYMBOLS), Q, RHO0)
    assert np.allclose(new, old, rtol=0, atol=1e-10)


def test_weighted_fq_xray_is_xray_fq(crystal):
    r, g, _ = crystal
    parts = m2.partial_fq(r, g, Q, RHO0, QDAMP)
    assert np.allclose(m2.weighted_fq(parts, SYMBOLS, Q, "xray"),
                       m2.xray_fq(r, g, SYMBOLS, Q, RHO0, QDAMP), atol=1e-12)


def test_weighted_fq_rejects_unknown_radiation(crystal):
    r, g, _ = crystal
    with pytest.raises(ValueError, match="radiation"):
        m2.weighted_fq(m2.partial_fq(r, g, Q, RHO0), SYMBOLS, Q, "electron")


def test_default_closure_reproduces_the_original_m2_numbers(crystal):
    """neutron + raw + qdamp 0 is the original closure, number for number."""
    r, g, F_data = crystal
    fit, cur = m2.closure_fit(r, g, SYMBOLS, RHO0, Q, F_data)
    G = m2.total_G(r, g, SYMBOLS)
    s, o, rw_q = m2.fit_scale_offset(F_data, m2.gr_to_fq(r, G, Q, RHO0))
    Gd = m2.fq_to_gr(Q, F_data, r, RHO0)
    _, _, rw_r = m2.fit_scale_offset(Gd[r > 1.5], G[r > 1.5])
    assert fit["scale"] == pytest.approx(s, rel=1e-9)
    assert fit["offset"] == pytest.approx(o, abs=1e-9)
    assert fit["Rw_Q"] == pytest.approx(rw_q, rel=1e-9)
    assert fit["Rw_r"] == pytest.approx(rw_r, rel=1e-9)
    assert np.allclose(cur["G_sim"], G)
    assert fit["box_length_A"] is None
    assert fit["r_window_A"] == pytest.approx([1.5, 20.0])


def test_xray_box_closure_recovers_the_model(crystal):
    """Right weights + box-matched comparison: scale 1, Rw ≈ 0, although
    the model is truncated at 20 Å and the data are not. The residual
    left is Q-window leakage of the r > 20 Å shells, as in RMCProfile's
    own CONVOLVE comparison."""
    r, g, F_data = crystal
    fit, cur = m2.closure_fit(r, g, SYMBOLS, RHO0, Q, F_data, "xray", QDAMP,
                              "box")
    assert fit["box_length_A"] == pytest.approx(40.0)
    assert fit["scale"] == pytest.approx(1.0, abs=0.01)
    assert fit["Rw_Q"] < 0.04
    assert fit["Rw_r"] < 0.02
    dev = np.abs(cur["F_sim"] - cur["F_data"]).max()
    assert dev < 0.1 * np.abs(cur["F_data"]).max()


def test_truncation_and_weighting_are_each_separable(crystal):
    """The variants attribute a bad closure: comparing raw adds the
    box-truncation ripple, neutron weights add the contrast error."""
    r, g, F_data = crystal
    fit, _ = m2.closure_fit(r, g, SYMBOLS, RHO0, Q, F_data, "xray", QDAMP,
                            "box")
    v = {k: x["Rw_Q"] for k, x in fit["variants"].items()}
    assert set(v) == {"neutron_raw", "neutron_box", "xray_raw", "xray_box"}
    assert v["xray_box"] == pytest.approx(fit["Rw_Q"])
    assert v["xray_raw"] > 5 * v["xray_box"]        # truncation ripple
    assert v["neutron_box"] > 5 * v["xray_box"]     # wrong contrast
    assert v["neutron_raw"] > v["xray_raw"]


def test_xray_raw_uses_the_fourier_route_in_r(crystal):
    """X-ray weights are Q-dependent, so G_sim(r) is the FT of F_sim over
    the measured window — the same operator the data G(r) went through."""
    r, g, F_data = crystal
    fit, cur = m2.closure_fit(r, g, SYMBOLS, RHO0, Q, F_data, "xray", QDAMP,
                              "raw")
    Fs = m2.xray_fq(r, g, SYMBOLS, Q, RHO0, QDAMP)
    assert np.allclose(cur["G_sim"], m2.fq_to_gr(Q, Fs, r, RHO0))
    assert fit["G_sim"].startswith("FT")
    assert fit["box_length_A"] is None


def test_variants_skip_radiations_without_tables():
    """A neutron closure never needs X-ray form factors: H has b_coh but no
    f0 entry, and the old neutron route must keep working for it."""
    r = np.arange(0.01, 8.0, 0.02)
    g = {("H", "H"): 1.0 + np.exp(-(r - 2.0)**2 / 0.01) - (r < 1.5)}
    Qg = np.linspace(1.0, 15.0, 200)
    F = m2.partial_fq(r, g, Qg, 0.05)[("H", "H")]
    fit, _ = m2.closure_fit(r, g, ["H"] * 4, 0.05, Qg, F)
    assert set(fit["variants"]) == {"neutron_raw", "neutron_box"}
    assert fit["Rw_Q"] < 1e-8
    with pytest.raises(SystemExit, match="form factor"):
        m2.closure_fit(r, g, ["H"] * 4, 0.05, Qg, F, radiation="xray")


def test_closure_rejects_unknown_comparison(crystal):
    r, g, F_data = crystal
    with pytest.raises(ValueError, match="compare"):
        m2.closure_fit(r, g, SYMBOLS, RHO0, Q, F_data, compare="window")
