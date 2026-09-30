"""Frame handling of the M3 ensemble driver (mode_project.ensemble_amplitudes).

The projector lives in the refined structure's setting, which differs from
the RMC/CIF frame by a parent rotation and a half-cell origin shift. Known
answers: a single irrep variant injected at 0.10 Å in the pattern frame,
written out as an RMC-frame rmc6f box, must be read back at 0.10 Å in its own
channel through the file-level driver — and projecting the RMC-frame
coordinates directly (no transform) must NOT, which is why the transform
exists.

Requires reference/gts_mode_patterns.json and data/GTS_5K.cif.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mode_project as mp

REPO = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (mp.REF_JSON.is_file() and mp.DEFAULT_CIF.is_file()),
    reason="needs gts_mode_patterns.json and data/GTS_5K.cif")

A_CUB = 10.3563


@pytest.fixture(scope="module")
def setup():
    return mp.projection_setup()


def test_aligned_frame_is_rotated_shifted_ideal(setup):
    pred = (setup["ideal"] @ setup["R"].T + setup["shift"]) % 1.0
    d = (pred - setup["aligned"] + 0.5) % 1.0 - 0.5
    assert np.abs(d).max() < 1e-9
    # GTS: the frames genuinely differ — a non-identity rotation
    assert not np.array_equal(setup["R"], np.eye(3, dtype=int))


def _variant_box_rmc_frame(setup, key, target=0.10):
    """An 8x8x8 box carrying one star variant of `key` at `target` Å,
    built in the pattern frame and mapped back to the RMC frame.
    Returns (X (N,3) cell units, sid, ijk)."""
    proj = setup["projector"]
    i0, _ = proj["blocks"][key]
    ax, G = proj["variants"][i0]
    lam = target * np.sqrt(2048.0 / mp.compact_inner((ax, G), (ax, G)))
    aligned, R, shift, n_site = (setup["aligned"], setup["R"],
                                 setup["shift"], setup["n_site"])
    X, sid, ijk = [], [], []
    for i in range(8):
        for j in range(8):
            for k in range(8):
                cell = np.array([i, j, k])
                for s in range(52):
                    u = lam * G[s, cell[ax] & 1] / A_CUB
                    X.append(cell + aligned[s] + u)
                    sid.append(s + 1)
                    ijk.append(cell)
    X, sid, ijk = np.array(X), np.array(sid), np.array(ijk)
    X_rmc = ((X - shift) @ R) % 8.0               # R^-1 = R^T
    ijk_rmc = ((ijk - n_site[sid - 1]) @ R) % 8
    return X_rmc, sid, ijk_rmc.astype(int)


def _write_rmc6f(path, X, sid, ijk, moves=12345):
    symbols = ["Ga", "Ta", "Se"]
    lines = ["(Version 6f format configuration file)",
             f"Number of moves generated:           {moves}",
             "Supercell dimensions:                8 8 8",
             f"Cell (Ang/deg):    {8 * A_CUB:.6f} {8 * A_CUB:.6f} "
             f"{8 * A_CUB:.6f} 90.0 90.0 90.0",
             "Atoms:"]
    for n, (x, s, c) in enumerate(zip(X / 8.0, sid, ijk), 1):
        lines.append(f"{n} {symbols[s % 3]} [1] {x[0]:.12f} {x[1]:.12f} "
                     f"{x[2]:.12f} {s} {c[0]} {c[1]} {c[2]}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("key", ["X5", "X3", "W4", "D"])
def test_injected_variant_recovered_through_transform(setup, key):
    X, sid, ijk = _variant_box_rmc_frame(setup, key)
    out = mp.config_amplitudes(X, sid, ijk, A_CUB, setup, windows=(8,))
    amp = dict(zip(setup["keys"], out["w8"][0]))
    # 7 %: the printed patterns' ~1 % rounding leaks a few % between the
    # X-point channels (X3 reads 0.095); the frame error is a factor ~5
    assert amp[key] == pytest.approx(0.10, rel=0.07), amp
    for other in ("X5", "X3", "W4", "D"):
        if other != key:
            assert amp[other] < 0.02, (other, amp)
    # the random-sign null kills the coherent pattern
    assert dict(zip(setup["keys"], out["null_w8"][0]))[key] < 0.04


def test_projecting_rmc_frame_directly_scrambles(setup):
    """Regression guard for the pitfall: without the frame map, a pure W4
    variant reads a small fraction of its amplitude in its own channel."""
    X, sid, ijk = _variant_box_rmc_frame(setup, "W4")
    S = mp.window_parity_sums(X % 1.0, sid, ijk, setup["ideal"], A_CUB, 8)
    amp = mp.project_all_windows(S, setup["projector"], 8)
    assert amp["W4"][0] < 0.05


def test_file_driver_roundtrip(setup, tmp_path):
    X, sid, ijk = _variant_box_rmc_frame(setup, "X5", target=0.08)
    f = _write_rmc6f(tmp_path / "box_1.rmc6f", X, sid, ijk, moves=777)
    Xr, sidr, ijkr, a_cub, moves = mp.read_rmc6f_box(f)
    assert moves == 777 and a_cub == pytest.approx(A_CUB)
    assert np.allclose(Xr, X, atol=1e-9) and np.array_equal(ijkr, ijk)
    out = mp.ensemble_amplitudes([f], setup, log=None)
    k = list(setup["keys"]).index("X5")
    for w in (2, 4, 8):
        assert out[f"amp_w{w}"].shape == (1, (8 // w)**3, len(setup["keys"]))
        assert out[f"rms_w{w}"][0, k] == pytest.approx(0.08, rel=0.05)
    assert out["moves"][0] == 777


# ------------------------------------------- positive-control static field

def test_published_field_in_rmc_frame_projects_to_published(setup):
    """The paper's Table IV distortion, mapped to RMC-frame static offsets
    and tiled into an 8x8x8 box, reads the published amplitudes through the
    file-level driver at every window scale (long-range order)."""
    F = mp.published_field(setup)
    static = mp.slab_field_to_rmc_static(F, setup)
    assert sorted(static.shape[:3]) == [1, 1, 2]
    X, sid, ijk = mp.static_box(setup, static, A_CUB)
    out = mp.config_amplitudes(X, sid, ijk, A_CUB, setup)
    pub = setup["ref"]["published_amplitudes_A"]
    keys = list(setup["keys"])
    for key, tol in [("X5", 0.05), ("X3", 0.10), ("W4", 0.15), ("D", 0.10)]:
        for w in (2, 4, 8):
            got = out[f"w{w}"][:, keys.index(key)]
            assert np.allclose(got, got[0], rtol=1e-6)          # every window
            assert got[0] == pytest.approx(pub[key], rel=tol), (key, w)


def test_rmc_static_route_equals_pattern_frame_route(setup):
    F = mp.published_field(setup)
    X, sid, ijk = mp.static_box(setup, mp.slab_field_to_rmc_static(F, setup),
                                A_CUB)
    via_rmc = mp.config_amplitudes(X, sid, ijk, A_CUB, setup, windows=(8,))
    ax, G = mp.field_to_compact(F)
    S = np.zeros((3, 2, 52, 3))
    for px in range(2):
        for py in range(2):
            for pz in range(2):
                par = (px, py, pz)
                for a in range(3):
                    S[a, par[a]] += 64.0 * G[:, par[ax]]
    direct = mp.project_all(S, setup["projector"])
    for j, k in enumerate(setup["keys"]):
        assert via_rmc["w8"][0, j] == pytest.approx(direct[k], abs=1e-6)


def test_remove_uniform_part(setup):
    F = mp.published_field(setup)
    Fs = mp.remove_uniform_part(F, setup)
    assert np.allclose(mp.remove_uniform_part(Fs, setup), Fs, atol=1e-12)
    # a pure k=0 field (the removed part) is annihilated
    assert np.abs(mp.remove_uniform_part(F - Fs, setup)).max() < 1e-12
    # the staggered channels are untouched
    amp = {}
    for tag, field in (("full", F), ("sb", Fs)):
        X, sid, ijk = mp.static_box(
            setup, mp.slab_field_to_rmc_static(field, setup), A_CUB)
        amp[tag] = mp.config_amplitudes(X, sid, ijk, A_CUB, setup,
                                        windows=(8,))["w8"][0]
    keys = list(setup["keys"])
    for key in ("X5", "X3", "W4", "D"):
        j = keys.index(key)
        assert amp["sb"][j] == pytest.approx(amp["full"][j], abs=1e-3)
    assert amp["sb"][keys.index("G1")] < amp["full"][keys.index("G1")]
