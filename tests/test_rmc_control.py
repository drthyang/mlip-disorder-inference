"""rmc_control.py — synthetic-data I/O, noise estimate, staging, comparison.

Pure-helper known answers plus an EMT fcc-Cu smoke test of the whole
`synth` → `stage` chain (fast, no torch). The GTS run itself is private.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import md_run
import rmc_control as rc
from fixtures.make_synthetic_ensemble import make_fcc_cu_ensemble

SUBMIT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=job_000
#SBATCH --output=out_000.out
#SBATCH -t 24:00:00
#SBATCH -N 1

export OMP_NUM_THREADS=8
RMCProfile_PATH=/opt/RMCProfile_package
cd /some/original/run/dir/
$RMCProfile_PATH/exe/rmcprofile GTS_5K_000
"""

DAT = """TITLE :: test
RESOLUTION_CORRECTION_NOT_THIS :: 9
XRAY_RECIPROCAL_SPACE_DATA ::
  > FILENAME :: scale_ft_rmc.fq
  > RESOLUTION_CORRECTION :: 0.0389
END ::
"""


def test_write_fq_roundtrip_and_layout(tmp_path):
    Q = np.arange(0.8, 1.2, 0.01)
    F = np.sin(Q) - 0.5
    rc.write_fq(tmp_path / "x.fq", Q, F)
    lines = (tmp_path / "x.fq").read_text().splitlines()
    assert lines[0].strip() == str(len(Q)) and lines[1].startswith("rmc S(Q)")
    Q2, F2 = md_run.parse_fq(tmp_path / "x.fq")
    assert np.allclose(Q2, Q, atol=1e-15) and np.allclose(F2, F, atol=1e-15)


def test_noise_sigma_recovers_white_noise_level():
    rng = np.random.default_rng(3)
    Q = np.arange(0.8, 27.0, 0.01)
    true = np.where(Q < 10, 0.02, 0.002)
    F = 0.3 * np.sin(2.0 * Q) * np.exp(-0.05 * Q) + true * rng.standard_normal(
        len(Q))
    sig = rc.noise_sigma(Q, F)
    for lo, hi, s in [(2, 9, 0.02), (12, 26, 0.002)]:
        m = (Q > lo) & (Q < hi)
        assert np.median(sig[m]) == pytest.approx(s, rel=0.12)


def test_noise_sigma_is_robust_to_resolution_limited_peaks():
    """Dense Bragg-like peaks 4 grid points wide (GTS's resolution) plus
    white noise 1e-3: the 6th-difference estimate finds the noise, a 2nd
    difference reads the peak curvature instead."""
    rng = np.random.default_rng(5)
    Q = np.arange(0.8, 8.0, 0.01)
    centres = np.arange(1.0, 8.0, 0.25)
    F = sum(2.0 * np.exp(-0.5 * ((Q - c) / 0.04)**2) for c in centres)
    F = F + 1e-3 * rng.standard_normal(len(Q))
    m = (Q > 2) & (Q < 7)
    assert np.median(rc.noise_sigma(Q, F)[m]) == pytest.approx(1e-3, rel=0.3)
    assert np.median(rc.noise_sigma(Q, F, order=2)[m]) > 3e-3


def test_dat_value_reads_subordinate_keywords(tmp_path):
    (tmp_path / "a.dat").write_text(DAT)
    assert rc.dat_value(tmp_path / "a.dat", "FILENAME") == "scale_ft_rmc.fq"
    assert float(rc.dat_value(tmp_path / "a.dat",
                              "RESOLUTION_CORRECTION")) == 0.0389
    assert rc.dat_value(tmp_path / "a.dat", "NOPE") is None


def test_render_submit_changes_only_the_anchors():
    out = rc.render_submit(SUBMIT_TEMPLATE, "GTS_5K_null", 7)
    assert "#SBATCH --job-name=gts_5k_null_7" in out
    assert "#SBATCH --output=GTS_5K_null_7.out" in out
    assert 'cd "$SLURM_SUBMIT_DIR"' in out
    assert out.rstrip().endswith("exe/rmcprofile GTS_5K_null_7")
    for keep in ("#SBATCH -t 24:00:00", "#SBATCH -N 1",
                 "export OMP_NUM_THREADS=8",
                 "RMCProfile_PATH=/opt/RMCProfile_package"):
        assert keep in out
    with pytest.raises(ValueError, match="expected one match"):
        rc.render_submit(SUBMIT_TEMPLATE.replace("cd /some", "pushd /x"),
                         "S", 1)


def _projection_npz(rng, n, scale):
    keys = np.array(["D", "X5"])
    out = {"keys": keys, "moves": np.full(n, 2_200_000)}
    for w in (2, 4, 8):
        base = rng.normal(0.1, 0.01, size=(n, 2))
        out[f"rms_w{w}"] = base * np.array([1.0, scale])
        out[f"rms_null_w{w}"] = rng.normal(0.08, 0.01, size=(n, 2))
    return out


def test_compare_projections_classifies_excess_and_explained():
    rng = np.random.default_rng(0)
    meas = _projection_npz(rng, 200, np.sqrt(2.0))      # X5 power ×2
    ctrl = _projection_npz(rng, 40, 1.0)
    rep = rc.compare_projections(meas, ctrl, n_boot=500)
    x5, d = rep["w4"]["X5"], rep["w4"]["D"]
    assert x5["r_ctrl"] == pytest.approx(2.0, rel=0.1)
    assert x5["class"] == "excess" and x5["ci95"][0] > 1
    assert d["class"] == "explained" and d["ci95"][0] < 1 < d["ci95"][1]
    assert rep["ensembles"]["n_control"] == 40
    with pytest.raises(ValueError, match="keys"):
        rc.compare_projections(meas, {**ctrl, "keys": np.array(["D", "X3"])})


def _cu_inputs(tmp_path):
    """Ideal Cu start box, a fake measured F(Q), and a .dat — the three
    inputs `synth` reads."""
    start = make_fcc_cu_ensemble(tmp_path / "start", n_configs=1,
                                 dims=(4, 4, 4), sigma=0.0)[0]
    Q = np.arange(0.8, 20.0, 0.01)
    rng = np.random.default_rng(1)
    rc.write_fq(tmp_path / "meas.fq", Q,
                0.5 * np.sin(3 * Q) * np.exp(-0.1 * Q)
                + 0.005 * rng.standard_normal(len(Q)))
    (tmp_path / "run.dat").write_text(DAT)
    return start, tmp_path / "meas.fq", tmp_path / "run.dat"


def test_synth_then_stage_on_emt_copper(tmp_path):
    start, meas, dat = _cu_inputs(tmp_path)
    synth = tmp_path / "synth"
    assert rc.main(["synth", "--start", str(start), "--data", str(meas),
                    "--dat", str(dat), "--calc", "emt", "--fc-dim", "3",
                    "--grid", "6", "--rmax", "30", "-T", "300",
                    "-o", str(synth)]) == 0
    man = json.loads((synth / "synth_manifest.json").read_text())
    assert man["forward_model"]["qdamp_A-1"] == pytest.approx(0.0389)
    assert man["model"]["relaxed_vs_start_max_shift_A"] < 1e-3   # fcc: exact
    Q, F = md_run.parse_fq(synth / "synth_noiseless.fq")
    # fcc (111) Bragg peak of a = 3.61 Å at Q = 2π√3/a ≈ 3.01 Å⁻¹
    m = (Q > 2.0) & (Q < 4.0)
    assert Q[m][np.argmax(F[m])] == pytest.approx(2 * np.pi * np.sqrt(3) / 3.61,
                                                  abs=0.05)
    Qn, Fn = md_run.parse_fq(synth / "scale_ft_rmc.fq")
    assert np.allclose(Qn, Q) and not np.allclose(Fn, F)       # noise added

    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "GTS_5K.dat").write_text(DAT)
    (tpl / "GTS_5K.rmc6f").write_text(start.read_text())
    (tpl / "submit.sh").write_text(SUBMIT_TEMPLATE)
    (tpl / "optimization.dat").write_text("2000\n0.2\n")
    run = tmp_path / "run"
    assert rc.main(["stage", "--synth", str(synth), "--template", str(tpl),
                    "--n-chains", "3", "-o", str(run)]) == 0
    for i in (1, 2, 3):
        assert (run / f"GTS_5K_null_{i}.dat").read_text() == DAT
        assert (run / f"GTS_5K_null_{i}.rmc6f").read_text() == start.read_text()
        assert f"rmcprofile GTS_5K_null_{i}" in (run / f"submit_{i}.sh"
                                                 ).read_text()
    assert (run / "scale_ft_rmc.fq").read_bytes() == \
        (synth / "scale_ft_rmc.fq").read_bytes()
    assert (run / "optimization.dat").is_file() and (run / "README.md").is_file()
    assert json.loads((run / "stage_manifest.json").read_text())["n_chains"] == 3


def test_stage_refuses_a_dat_reading_another_file(tmp_path):
    synth = tmp_path / "synth"
    synth.mkdir()
    rc.write_fq(synth / "scale_ft_rmc.fq", np.arange(1, 2, 0.1),
                np.zeros(10))
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "GTS_5K.dat").write_text(DAT.replace("scale_ft_rmc.fq", "x.fq"))
    (tpl / "GTS_5K.rmc6f").write_text("x")
    (tpl / "submit.sh").write_text(SUBMIT_TEMPLATE)
    with pytest.raises(SystemExit, match="reads 'x.fq'"):
        rc.main(["stage", "--synth", str(synth), "--template", str(tpl),
                 "-o", str(tmp_path / "run")])


# ------------------------------------------------------ positive control

def _arm(rng, n, extra_x5_power=0.0):
    """Projection-npz dict: a noisy pedestal plus extra X5 power (Å²)."""
    keys = np.array(["D", "X5"])
    out = {"keys": keys, "moves": np.full(n, 2_000_000)}
    for w in (2, 4, 8):
        a2 = rng.normal(0.01, 0.001, size=(n, 2))
        a2[:, 1] += extra_x5_power
        out[f"rms_w{w}"] = np.sqrt(a2)
        out[f"rms_null_w{w}"] = np.sqrt(np.full((n, 2), 0.008))
    return out


def test_recovery_and_calibrated_static_amplitude():
    rng = np.random.default_rng(1)
    inj = {f"w{w}": {"D": 0.02, "X5": 0.12} for w in (2, 4, 8)}
    null = _arm(rng, 64)
    pos = _arm(rng, 64, extra_x5_power=0.5 * 0.12**2)       # ρ = 0.5
    meas = _arm(rng, 400, extra_x5_power=0.5 * 0.08**2)     # static 0.08 Å
    res = rc.recovery(meas, null, pos, inj, n_boot=400)
    x5 = res["w4"]["X5"]
    assert x5["recovery"] == pytest.approx(0.5, abs=0.06)
    assert x5["ci95"][0] < 0.5 < x5["ci95"][1]
    assert x5["static_estimate_A"] == pytest.approx(0.08, rel=0.1)
    # no injected D power recovered -> estimate undefined or huge CI
    assert abs(res["w4"]["D"]["recovery"]) < 3.0


def test_compare_cli_with_positive_arm(tmp_path):
    rng = np.random.default_rng(2)
    inj = {f"w{w}": {"D": 0.02, "X5": 0.12} for w in (2, 4, 8)}
    for name, d in (("meas", _arm(rng, 200, 0.004)), ("null", _arm(rng, 40)),
                    ("pos", _arm(rng, 40, 0.0144))):
        np.savez(tmp_path / f"{name}.npz", **d)
    synth = tmp_path / "synth_positive_x1"
    synth.mkdir()
    (synth / "synth_manifest.json").write_text(json.dumps(
        {"injection": {"mode": "published", "injected_amplitudes_A": inj}}))
    out = tmp_path / "rep.json"
    assert rc.main(["compare", "--measured", str(tmp_path / "meas.npz"),
                    "--control", str(tmp_path / "null.npz"),
                    "--positive", str(tmp_path / "pos.npz"), str(synth),
                    "--n-boot", "200", "-o", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["positive"]["synth_positive_x1"]["w4"]["X5"]["recovery"] == \
        pytest.approx(1.0, abs=0.15)
    assert rep["ensembles"]["n_synth_positive_x1"] == 40


def test_stage_refuses_arm_data_mismatch(tmp_path):
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "GTS_5K.dat").write_text(DAT)
    (tpl / "GTS_5K.rmc6f").write_text("x")
    (tpl / "submit.sh").write_text(SUBMIT_TEMPLATE)
    for arm, mode in (("positive_x1", "none"), ("null", "published")):
        synth = tmp_path / f"synth_{arm}"
        synth.mkdir()
        rc.write_fq(synth / "scale_ft_rmc.fq", np.arange(1, 2, 0.1),
                    np.zeros(10))
        (synth / "synth_manifest.json").write_text(
            json.dumps({"injection": {"mode": mode}}))
        with pytest.raises(SystemExit, match="null data for the null arm"):
            rc.main(["stage", "--synth", str(synth), "--template", str(tpl),
                     "--arm", arm, "-o", str(tmp_path / f"run_{arm}")])


GTS_START = Path(__file__).resolve().parent.parent / \
    "data/ensemble_20A_5K/GTS_5K.rmc6f"
GTS_CIF = Path(__file__).resolve().parent.parent / "data/GTS_5K.cif"


@pytest.mark.skipif(not (GTS_START.is_file() and GTS_CIF.is_file()),
                    reason="needs the private GTS start box and CIF")
def test_gts_injection_known_answer_and_frame_guard():
    atoms0, _ = rc.start_cell(GTS_START)
    static, info = rc.injection_static("published", 1.0, False, GTS_CIF,
                                       atoms0)
    assert sorted(static.shape[:3]) == [1, 1, 2]
    a4 = info["injected_amplitudes_A"]["w4"]
    assert a4["X5"] == pytest.approx(0.1196, rel=0.05)
    assert a4["X3"] == pytest.approx(0.0719, rel=0.1)
    s2, info2 = rc.injection_static("published", 2.0, False, GTS_CIF, atoms0)
    assert np.allclose(s2, 2 * static)
    assert info2["injected_amplitudes_A"]["w8"]["X5"] == pytest.approx(
        2 * info["injected_amplitudes_A"]["w8"]["X5"], rel=1e-6)
    shuffled = atoms0.copy()
    shuffled.set_scaled_positions(np.roll(atoms0.get_scaled_positions(), 1, 0))
    with pytest.raises(SystemExit, match="disagree"):
        rc.injection_static("published", 1.0, False, GTS_CIF, shuffled)
    assert rc.injection_static("none", 1.0, False, GTS_CIF, atoms0)[0] is None
