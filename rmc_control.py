#!/usr/bin/env python3
"""rmc_control.py — the RMC control experiment for the milestone-3 verdicts.

Question: how much per-irrep amplitude does RMC MANUFACTURE from data that
contain none? The M3 verdicts compare windowed irrep amplitudes of the RMC
ensemble against a quantum baseline plus a single scalar noise fraction; they
cannot tell "X5 is static" from "X5 is where RMC's single-atom move
statistics pile up". The control answers that directly: give RMCProfile
synthetic data from a model with ZERO static disorder — the MLIP quantum-
harmonic null model — run it exactly as the real data were run, and push the
resulting boxes through the same projector. Per irrep and window scale,

    r_ctrl = ⟨A²⟩_measured / ⟨A²⟩_control

is then the amplitude excess over everything RMC does on its own; r_ctrl ≈ 1
means the measured amplitude is explained by RMC + quantum motion alone.
Design: docs/control-experiment-plan.md.

Subcommands
    synth    null model -> synthetic X-ray F(Q) on the measured Q grid, as a
             diffractometer would record it from the infinite crystal:
             MACE quantum-harmonic g(r) (harmonic_pdf.py, correlated widths),
             RMCProfile's resolution envelope and X-ray weights (md_run), and
             noise matched to the measured data. Written UNconvolved —
             RMCProfile applies its box convolution itself.
    stage    a NERSC run directory that mirrors the original ensemble run
             (same .dat, same ideal starting box, same auxiliary inputs,
             same job shape); only the data file differs.
    compare  measured vs control projections (npz files written by
             `python mode_project.py ...`) -> r_ctrl per irrep with
             bootstrap intervals -> control_report.json.

Units: Å, Å⁻¹, THz, K; F(Q) = S(Q) − 1 dimensionless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import md_run
import milestone1_bands as m1

FQ_TITLE = "rmc S(Q)#L"


# ----------------------------------------------------------------------------
# pure helpers (unit-tested)
# ----------------------------------------------------------------------------

def write_fq(path: Path, Q, F, title: str = FQ_TITLE):
    """Write an RMCProfile/STOG .fq file (count line, title line, Q F rows).

    Mirrors the measured file's layout so the unchanged .dat can read it;
    `md_run.parse_fq` reads it back.
    """
    lines = [f"{len(Q):12d}", f"{title:<80s}"]
    lines += [f"  {q:.17f}      {f: .17f}     " for q, f in zip(Q, F)]
    Path(path).write_text("\n".join(lines) + "\n")


def noise_sigma(Q, F, half_width=0.5, order=6):
    """Local white-noise level of a measured F(Q), robust to its peaks.

    Normalized `order`-th differences e_k = Σ_i (−1)^i C(order, i) F_{k+i−order/2}
    / √Σ C² have the variance of white noise while suppressing a signal that
    is smooth on the grid scale by ~(ΔQ/w)^order for peak width w; σ(Q) is
    1.4826 × the median absolute deviation of e over ±half_width Å⁻¹.

    Order 6, not 2: resolution-limited Bragg peaks (w ≈ 4 grid points for
    GTS) leak through a 2nd difference at ~50× the apparent noise at low Q;
    at order 6 the leakage measured on a noiseless synthetic pattern is
    ≤10 % of the measured estimate at every Q. STOG-processed data are
    filtered, so this is a GRID-SCALE (white) level and a lower bound on
    correlated noise — see docs/control-experiment-plan.md.
    """
    from math import comb

    F = np.asarray(F, dtype=float)
    c = np.array([(-1)**i * comb(order, i) for i in range(order + 1)], float)
    c /= np.sqrt((c * c).sum())
    e = np.convolve(F, c, mode="same")
    p = order // 2
    e[:p], e[len(e) - p:] = e[p], e[len(e) - p - 1]
    h = max(1, int(round(half_width / (Q[1] - Q[0]))))
    sig = np.empty_like(F)
    for k in range(len(F)):
        w = e[max(0, k - h):k + h + 1]
        sig[k] = 1.4826 * np.median(np.abs(w - np.median(w)))
    return sig


def dat_value(dat_path: Path, keyword: str):
    """First value after `keyword ::` in an RMCProfile .dat (string) or None.

    The keyword must match exactly (case-insensitive), so a longer keyword
    sharing its prefix is never read by mistake.
    """
    for line in Path(dat_path).read_text().splitlines():
        parts = line.strip().lstrip(">").split("::", 1)
        if (len(parts) == 2 and parts[0].strip().upper() == keyword.upper()
                and parts[1].split()):
            return parts[1].split()[0]
    return None


def compare_projections(meas, ctrl, windows=(2, 4, 8), n_boot=2000, seed=0,
                        positives=None):
    """Per-irrep control ratio r_ctrl = ⟨A²⟩_meas / ⟨A²⟩_ctrl, all scales.

    meas / ctrl: mappings with 'keys', 'moves' and per scale 'rms_w{w}',
    'rms_null_w{w}' (n_cfg, n_keys) as written by mode_project's driver.
    Bootstrap resamples configurations of BOTH ensembles; the interval is
    the 2.5–97.5 percentile. Classification of the interval:
    'excess' (lower bound > 1), 'deficit' (upper < 1), else 'explained'.
    positives: optional list of (name, projections, injected) positive-
    control arms, injected = {f"w{w}": {key: A}} from the arm's synth
    manifest; adds a 'positive' section (see `recovery`).
    """
    keys = [str(k) for k in meas["keys"]]
    if keys != [str(k) for k in ctrl["keys"]]:
        raise ValueError("measured and control projections disagree on keys")
    rng = np.random.default_rng(seed)
    out = {}
    for w in windows:
        a2m = np.asarray(meas[f"rms_w{w}"])**2
        a2c = np.asarray(ctrl[f"rms_w{w}"])**2
        r = a2m.mean(0) / a2c.mean(0)
        boot = np.empty((n_boot, len(keys)))
        for b in range(n_boot):
            im = rng.integers(0, len(a2m), len(a2m))
            ic = rng.integers(0, len(a2c), len(a2c))
            boot[b] = a2m[im].mean(0) / a2c[ic].mean(0)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        n2m = np.asarray(meas[f"rms_null_w{w}"])**2
        n2c = np.asarray(ctrl[f"rms_null_w{w}"])**2
        rows = {}
        for j, k in enumerate(keys):
            cls = ("excess" if lo[j] > 1 else "deficit" if hi[j] < 1
                   else "explained")
            rows[k] = {
                "r_ctrl": round(float(r[j]), 3),
                "ci95": [round(float(lo[j]), 3), round(float(hi[j]), 3)],
                "class": cls,
                "amplitude_measured_A": round(float(np.sqrt(a2m[:, j].mean())), 4),
                "amplitude_control_A": round(float(np.sqrt(a2c[:, j].mean())), 4),
                "measured_over_own_null": round(float(a2m[:, j].mean()
                                                      / n2m[:, j].mean()), 3),
                "control_over_own_null": round(float(a2c[:, j].mean()
                                                     / n2c[:, j].mean()), 3),
            }
        out[f"w{w}"] = rows
    out["ensembles"] = {
        "n_measured": int(len(meas["rms_w4"])),
        "n_control": int(len(ctrl["rms_w4"])),
        "moves_measured_median": float(np.median(meas["moves"])),
        "moves_control_median": float(np.median(ctrl["moves"])),
    }
    if positives:
        out["positive"] = {name: recovery(meas, ctrl, pos, injected, windows,
                                          n_boot, seed)
                           for name, pos, injected in positives}
        for name, pos, _ in positives:
            out["ensembles"][f"n_{name}"] = int(len(pos["rms_w4"]))
            out["ensembles"][f"moves_{name}_median"] = float(
                np.median(pos["moves"]))
    return out


def recovery(meas, null, pos, injected, windows=(2, 4, 8), n_boot=2000,
             seed=0):
    """Positive-control calibration per irrep and window scale.

    ρ = (⟨A²⟩_pos − ⟨A²⟩_null) / A²_injected is the fraction of a KNOWN
    static power that RMC puts back into its boxes (1 = full recovery; the
    known answer is the injected field's own projection, so pattern leakage
    cancels). The measured ensemble's calibrated static amplitude is then
    A_static = √(max(⟨A²⟩_meas − ⟨A²⟩_null, 0) / ρ) — valid if recovery is
    linear in power, which two arms at different scales test. Intervals:
    joint bootstrap over all three ensembles; A_static is undefined (None)
    where ρ ≤ 0.
    """
    keys = [str(k) for k in meas["keys"]]
    for d in (null, pos):
        if [str(k) for k in d["keys"]] != keys:
            raise ValueError("projection files disagree on keys")
    rng = np.random.default_rng(seed + 1)
    res = {}
    for w in windows:
        a2m = np.asarray(meas[f"rms_w{w}"])**2
        a2n = np.asarray(null[f"rms_w{w}"])**2
        a2p = np.asarray(pos[f"rms_w{w}"])**2
        inj2 = np.array([injected[f"w{w}"][k]**2 for k in keys])

        def stats(im, i_n, ip):
            dn = a2n[i_n].mean(0)
            rho = (a2p[ip].mean(0) - dn) / inj2
            ex = np.maximum(a2m[im].mean(0) - dn, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(rho > 0, np.sqrt(ex / rho), np.nan)
            return rho, est

        rho, est = stats(slice(None), slice(None), slice(None))
        boot_r = np.empty((n_boot, len(keys)))
        boot_e = np.empty((n_boot, len(keys)))
        for b in range(n_boot):
            boot_r[b], boot_e[b] = stats(rng.integers(0, len(a2m), len(a2m)),
                                         rng.integers(0, len(a2n), len(a2n)),
                                         rng.integers(0, len(a2p), len(a2p)))
        r_lo, r_hi = np.percentile(boot_r, [2.5, 97.5], axis=0)
        with np.errstate(invalid="ignore"):
            e_lo, e_hi = np.nanpercentile(boot_e, [2.5, 97.5], axis=0)
        res[f"w{w}"] = {k: {
            "injected_A": round(float(np.sqrt(inj2[j])), 4),
            "recovery": round(float(rho[j]), 3),
            "ci95": [round(float(r_lo[j]), 3), round(float(r_hi[j]), 3)],
            "static_estimate_A": (None if not np.isfinite(est[j])
                                  else round(float(est[j]), 4)),
            "static_ci95": [None if not np.isfinite(x) else round(float(x), 4)
                            for x in (e_lo[j], e_hi[j])],
        } for j, k in enumerate(keys)}
    return res


# ----------------------------------------------------------------------------
# synth
# ----------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def start_cell(start_rmc6f: Path):
    """The starting box folded to its unit cell, in RMC site-id order."""
    from ase import Atoms

    cfg = m1.parse_rmc6f(Path(start_rmc6f))
    unit_cell, elems, frac, _ = m1.fold_and_average([cfg], None)
    return Atoms(symbols=elems, scaled_positions=frac, cell=unit_cell,
                 pbc=True), cfg


def mean_structure_phonon(phonon, frac_mean, symprec):
    """Same force constants, different mean positions (same atom order).

    The FCs are computed at the MLIP minimum; the synthetic data may carry
    the experimental mean positions instead (design D-mean). Lattice
    idealized first, per CLAUDE.md.
    """
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    uc = phonon.unitcell
    at = Atoms(symbols=list(uc.symbols), scaled_positions=frac_mean,
               cell=uc.cell, pbc=True)
    at, _ = m1.symmetrize_lattice(at, symprec)
    unit = PhonopyAtoms(symbols=at.get_chemical_symbols(), cell=at.cell.array,
                        scaled_positions=at.get_scaled_positions())
    ph = Phonopy(unit, supercell_matrix=phonon.supercell_matrix,
                 primitive_matrix="P", symprec=symprec)
    ph.force_constants = phonon.force_constants
    return ph


def injection_static(inject, scale, keep_gamma, cif, atoms0):
    """Static offsets for the positive-control arm, and their known answer.

    inject='published': the paper's refined P-4̄2₁m distortion (SM Table IV)
    — minus its k = 0 part unless keep_gamma — times `scale`, as RMC-frame
    offsets on the 52-site cell (mode_project). The known answer is that
    field's own windowed projection through the committed projector (so
    pattern-rounding leakage cancels in the recovery ratio).

    Returns (static (p1,p2,p3,52,3) Å or None, info dict for the manifest).
    """
    if inject == "none":
        return None, {"mode": "none"}
    import mode_project as mp

    setup = mp.projection_setup(cif)
    a_cub = float(atoms0.cell.lengths()[0])
    frac0 = atoms0.get_scaled_positions()
    dev = np.abs(((frac0 - setup["ideal"] + 0.5) % 1.0 - 0.5) * a_cub).max()
    if atoms0.get_chemical_symbols() != list(setup["elem"]) or dev > 0.05:
        raise SystemExit(f"start cell and {cif.name} disagree (site order / "
                         f"frame; max deviation {dev:.3f} Å)")
    F = mp.published_field(setup)
    if not keep_gamma:
        F = mp.remove_uniform_part(F, setup)
    static = mp.slab_field_to_rmc_static(scale * F, setup)
    X, sid, ijk = mp.static_box(setup, static, a_cub)
    amp = mp.config_amplitudes(X, sid, ijk, a_cub, setup)
    keys = [str(k) for k in setup["keys"]]
    injected = {f"w{w}": {k: round(float(np.sqrt((amp[f"w{w}"][:, j]**2
                                                    ).mean())), 5)
                          for j, k in enumerate(keys)}
                for w in mp.WINDOWS}
    info = {"mode": "published (SM Table IV total)", "scale": scale,
            "gamma_part": "kept" if keep_gamma else "removed (k = 0)",
            "cif": str(cif), "period_cells": list(static.shape[:3]),
            "max_static_A": round(float(np.abs(static).max()), 4),
            "published_amplitudes_A": setup["ref"]["published_amplitudes_A"],
            "injected_amplitudes_A": injected,
            "note": "static offsets on the parent's quantum widths and "
                    "correlations; single domain, long-range ordered"}
    return static, info


def build_model(args, out):
    """Start cell, box, RMCProfile resolution and the MLIP harmonic model.

    Shared by `synth` and `scan`: relaxes internal coordinates at the
    experimental lattice, computes FCs on fc_dim³, and returns the phonopy
    object carrying the chosen mean positions (see `mean_structure_phonon`).
    Writes relaxed_null.cif into `out`.
    """
    from ase.io import write as ase_write
    from ase.optimize import FIRE

    qdamp = float(dat_value(args.dat, "RESOLUTION_CORRECTION") or 0.0)
    print(f"[1/5] start cell from {args.start.name}; qdamp = {qdamp} Å⁻¹ "
          f"(from {args.dat.name})")
    atoms0, cfg = start_cell(args.start)
    box = np.asarray(cfg["cell"], dtype=float)
    box_len = float(np.linalg.norm(box[0]))
    n_box = len(cfg["elements"])
    rho0 = n_box / abs(np.linalg.det(box))
    print(f"  {len(atoms0)}-site cell, a = {atoms0.cell.lengths().round(4)}; "
          f"box {box_len:.4f} Å, {n_box} atoms, ρ₀ = {rho0:.6f} Å⁻³")

    print(f"[2/5] {args.calc} relax at the experimental lattice + FCs on "
          f"{args.fc_dim}³")
    calc = m1.get_calculator(args.calc, args.device, args.model)
    relaxed = atoms0.copy()
    relaxed.calc = calc
    FIRE(relaxed, logfile=None).run(fmax=args.fmax, steps=1000)
    dmean = relaxed.get_positions() - atoms0.get_positions()
    shift = float(np.linalg.norm(dmean, axis=1).max())
    print(f"  max |F| = {np.abs(relaxed.get_forces()).max():.1e} eV/Å; "
          f"relaxed vs start positions: max shift {shift:.4f} Å")
    ase_write(str(out / "relaxed_null.cif"), relaxed)
    phonon = md_run.harmonic_model(relaxed, calc, np.array([args.fc_dim] * 3),
                                   args.displacement, args.symprec,
                                   primitive_matrix="P", compact_fc=True)
    frac_mean = (atoms0 if args.mean == "start" else relaxed
                 ).get_scaled_positions()
    ph_mean = mean_structure_phonon(phonon, frac_mean, args.symprec)
    return {"atoms0": atoms0, "box_len": box_len, "rho0": rho0,
            "qdamp": qdamp, "shift": shift, "ph_mean": ph_mean}


def closure_metrics(Q, F_meas, F_syn, box_len, rho0, dr=0.01):
    """How far a synthetic F(Q) is from the measured one, as RMC sees it.

    Both sides box-convolved (`md_run.rmc_box_convolve`), then scale+offset
    fitted: Rw(Q). G(r) of both (FT to L/2), scale+offset fitted over
    1.5 Å < r < L/2: Rw(r), also reported over the local (1.5–5 Å) and
    medium-range (5 Å – L/2) windows with the same fit. Dimensionless.
    """
    conv_meas = md_run.rmc_box_convolve(Q, F_meas, box_len, rho0)
    conv_syn = md_run.rmc_box_convolve(Q, F_syn, box_len, rho0)
    s, o, rw_q = md_run.fit_scale_offset(conv_meas, conv_syn)
    r_g = np.arange(dr, box_len / 2, dr)
    G_meas = md_run.fq_to_gr(Q, F_meas, r_g, rho0)
    G_syn = md_run.fq_to_gr(Q, F_syn, r_g, rho0)
    m = r_g > 1.5
    s_r, o_r, rw_r = md_run.fit_scale_offset(G_meas[m], G_syn[m])
    fit_g = s_r * G_syn + o_r

    def rw(sel):
        return float(np.sqrt(((G_meas[sel] - fit_g[sel])**2).sum()
                             / (G_meas[sel]**2).sum()))
    return {"scale": s, "offset": o, "Rw_Q": rw_q, "Rw_r": rw_r,
            "Rw_r_local": rw(m & (r_g < 5.0)), "Rw_r_mid": rw(r_g >= 5.0),
            "scale_r": s_r, "offset_r": o_r, "conv_meas": conv_meas,
            "conv_syn": conv_syn, "r_g": r_g, "G_meas": G_meas,
            "G_syn": G_syn}


def cmd_synth(args):
    import harmonic_pdf as hp

    if args.outdir is None:
        args.outdir = Path("results/rmc_control/" + (
            "synth" if args.inject == "none"
            else f"synth_positive_x{args.inject_scale:g}"))
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mdl = build_model(args, out)
    atoms0, box_len, rho0, qdamp, shift, ph_mean = (
        mdl["atoms0"], mdl["box_len"], mdl["rho0"], mdl["qdamp"],
        mdl["shift"], mdl["ph_mean"])
    static, injection = injection_static(args.inject, args.inject_scale,
                                         args.keep_gamma, args.cif, atoms0)
    if static is not None:
        inj4 = injection["injected_amplitudes_A"]["w4"]
        print(f"  injecting {injection['mode']} ×{args.inject_scale:g} "
              f"(Γ {injection['gamma_part']}): max static "
              f"{injection['max_static_A']} Å; " + ", ".join(
                  f"{k} {v:.4f}" for k, v in inj4.items()) + " Å")

    print(f"[3/5] harmonic g(r): T = {args.temperature} K, grid {args.grid}³, "
          f"r ≤ {args.rmax} Å")
    r, g, info = hp.harmonic_partials(ph_mean, args.temperature, M=args.grid,
                                      r_max=args.rmax, dr=args.dr,
                                      static=static)
    fmin = float(np.min(ph_mean.qpoints.frequencies))   # the grid just run

    print("[4/5] X-ray F(Q) on the measured grid + noise")
    Q, F_meas = md_run.parse_fq(args.data)
    symbols = list(ph_mean.unitcell.symbols)
    F_clean = md_run.xray_fq(r, g, symbols, Q, rho0, qdamp)
    # the comparison RMCProfile makes: both sides box-convolved, scale+offset
    cm = closure_metrics(Q, F_meas, F_clean, box_len, rho0, args.dr)
    s, o, rw_q, rw_r = cm["scale"], cm["offset"], cm["Rw_Q"], cm["Rw_r"]
    conv_meas, conv_syn, r_g = cm["conv_meas"], cm["conv_syn"], cm["r_g"]
    G_meas, G_syn, s_r, o_r = (cm["G_meas"], cm["G_syn"], cm["scale_r"],
                               cm["offset_r"])
    sig_meas = noise_sigma(Q, F_meas)
    sig = args.noise_scale * sig_meas / s          # into synthetic units
    rng = np.random.default_rng(args.seed)
    F_noisy = F_clean + sig * rng.standard_normal(len(Q))
    print(f"  synthetic vs measured (box-convolved, scale+offset): "
          f"Rw(Q) = {rw_q:.3f}, scale = {s:.3f}; Rw(r) = {rw_r:.3f}")
    print(f"  noise: σ(Q) {sig.min():.4f}–{sig.max():.4f} "
          f"(×{args.noise_scale}, seed {args.seed})")

    print("[5/5] outputs")
    write_fq(out / "scale_ft_rmc.fq", Q, F_noisy)
    write_fq(out / "synth_noiseless.fq", Q, F_clean)
    np.savez(out / "synth_partials.npz", r=r, Q=Q, F_clean=F_clean,
             F_noisy=F_noisy, sigma=sig, F_measured=F_meas,
             conv_measured=conv_meas, conv_synth=conv_syn, r_g=r_g,
             G_measured=G_meas, G_synth=G_syn, U=info["U"],
             static=np.zeros(0) if static is None else static,
             **{f"g_{a}_{b}": v for (a, b), v in g.items()})
    manifest = {
        "purpose": ("zero-static-disorder synthetic X-ray F(Q) for the RMC "
                    "control experiment" if static is None else
                    "synthetic X-ray F(Q) with a KNOWN static distortion "
                    "(positive control)") +
                   " (docs/control-experiment-plan.md)",
        "injection": injection,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "inputs": {"start": str(args.start), "start_sha256": _sha256(args.start),
                   "data": str(args.data), "data_sha256": _sha256(args.data),
                   "dat": str(args.dat)},
        "model": {"calculator": args.calc, "size": args.model,
                  "dtype": "float64", "lattice": "experimental (fixed)",
                  "fc_supercell": [args.fc_dim] * 3,
                  "displacement_A": args.displacement,
                  "relaxed_vs_start_max_shift_A": round(shift, 5),
                  "mean_positions": args.mean,
                  "min_frequency_on_grid_THz": round(fmin, 4)},
        "sampling": {"temperature_K": args.temperature,
                     "statistics": "quantum (zero point + Bose)",
                     "q_grid": args.grid,
                     "correlations_within_A": round(
                         args.grid * float(atoms0.cell.lengths().min()) / 2, 2),
                     "r_max_A": args.rmax, "dr_A": args.dr,
                     "u_rms_per_site_A": [round(float(x), 5)
                                          for x in info["u_rms"]]},
        "forward_model": {"radiation": "xray",
                          "form_factors": "Waasmaier-Kirfel (RMCProfile)",
                          "normalization": "<f>^2 Faber-Ziman",
                          "qdamp_A-1": qdamp, "envelope": "exp(-(qdamp r)^2/2)",
                          "rho0_A-3": rho0, "written": "unconvolved"},
        "noise": {"estimator": "rolling MAD of 6th differences, ±0.5 Å⁻¹",
                  "scale": args.noise_scale, "seed": args.seed,
                  "sigma_range": [float(sig.min()), float(sig.max())]},
        "vs_measured": {"Rw_Q_box_convolved": round(rw_q, 4),
                        "scale": round(s, 4), "offset": round(o, 5),
                        "Rw_r_1.5_to_half_box": round(rw_r, 4)},
        "outputs": {"scale_ft_rmc.fq": _sha256(out / "scale_ft_rmc.fq")},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "synth_manifest.json").write_text(json.dumps(manifest, indent=2))
    _plot_synth(out, Q, conv_meas, s * conv_syn + o, sig, r_g, G_meas,
                s_r * G_syn + o_r)
    print(f"  wrote {out}/scale_ft_rmc.fq (+ noiseless, npz, manifest, png) "
          f"in {time.time() - t0:.0f} s")
    return 0


def _plot_synth(out, Q, conv_meas, conv_syn, sig, r, G_meas, G_syn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(9, 10))
    ax[0].plot(Q, conv_meas, lw=0.8, label="measured (box-convolved)")
    ax[0].plot(Q, conv_syn, lw=0.8, label="synthetic null (box-convolved, "
               "scale+offset)")
    ax[0].set(xlabel="Q (Å⁻¹)", ylabel="F(Q)")
    ax[0].legend(fontsize=8)
    ax[1].plot(r, G_meas, lw=0.8, label="measured")
    ax[1].plot(r, G_syn, lw=0.8, label="synthetic null")
    ax[1].set(xlim=(1.5, 20), xlabel="r (Å)", ylabel="G(r) (FT, L/2)")
    ax[1].legend(fontsize=8)
    ax[2].semilogy(Q, sig, lw=0.8)
    ax[2].set(xlabel="Q (Å⁻¹)", ylabel="added noise σ(Q)")
    fig.tight_layout()
    fig.savefig(out / "synth_vs_measured.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# scan
# ----------------------------------------------------------------------------

SCAN_METRICS = ("Rw_Q", "Rw_r", "Rw_r_local", "Rw_r_mid")


def parabolic_min(x, y):
    """Minimum of y(x) on a grid, refined by the parabola through the lowest
    point and its neighbours. Returns (x_min, y_min, on_edge)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i]), float(y[i]), True
    c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    if c[0] <= 0:
        return float(x[i]), float(y[i]), False
    xm = -c[1] / (2 * c[0])
    return float(xm), float(np.polyval(c, xm)), False


def scan_summary(scales, u_extra, grids):
    """Best scale per metric: at u_extra = 0, per u_extra, and profiled over
    u_extra (min over the nuisance at each scale). grids: {metric:
    (n_u, n_s)}. Returns a JSON-ready dict."""
    out = {}
    for k in SCAN_METRICS:
        G = np.asarray(grids[k])
        per_u = [dict(zip(("scale", "Rw", "on_edge"),
                          parabolic_min(scales, G[i])))
                 for i in range(len(u_extra))]
        prof = G.min(axis=0)
        iu, js = np.unravel_index(np.argmin(G), G.shape)
        out[k] = {
            "no_extra_width": per_u[0] if u_extra[0] == 0 else None,
            "per_u_extra": {f"{u:g}": v for u, v in zip(u_extra, per_u)},
            "profiled": dict(zip(("scale", "Rw", "on_edge"),
                                 parabolic_min(scales, prof))),
            "grid_best": {"scale": float(scales[js]),
                          "u_extra_A": float(u_extra[iu]),
                          "Rw": float(G[iu, js])},
        }
    return out


def cmd_scan(args):
    """Forward closure of the published distortion's amplitude.

    For every (scale s, extra width u) the analytic g(r) of the null model
    with s × (published field, k = 0 removed) as static offsets and u² extra
    uncorrelated variance, X-ray F(Q), and its distance from the measured
    F(Q) as RMC sees it (`closure_metrics`). The MLIP model is built once.
    """
    import harmonic_pdf as hp

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mdl = build_model(args, out)
    static1, inj = injection_static("published", 1.0, args.keep_gamma,
                                    args.cif, mdl["atoms0"])
    scales = np.array([float(x) for x in args.scales.split(",")])
    u_extra = np.array([float(x) for x in args.u_extra.split(",")])
    Q, F_meas = md_run.parse_fq(args.data)
    symbols = list(mdl["ph_mean"].unitcell.symbols)
    print(f"[3/5] scan: {len(scales)} scales × {len(u_extra)} extra widths "
          f"(grid {args.grid}³, r ≤ {args.rmax} Å)")
    grids = {k: np.zeros((len(u_extra), len(scales)))
             for k in SCAN_METRICS + ("scale",)}
    F_all = np.zeros((len(u_extra), len(scales), len(Q)))
    for iu, u in enumerate(u_extra):
        for js, s in enumerate(scales):
            t1 = time.time()
            r, g, _ = hp.harmonic_partials(
                mdl["ph_mean"], args.temperature, M=args.grid,
                r_max=args.rmax, dr=args.dr,
                static=None if s == 0 else s * static1, extra_u2=u * u,
                log=None)
            F = md_run.xray_fq(r, g, symbols, Q, mdl["rho0"], mdl["qdamp"])
            cm = closure_metrics(Q, F_meas, F, mdl["box_len"], mdl["rho0"],
                                 args.dr)
            for k in grids:
                grids[k][iu, js] = cm[k]
            F_all[iu, js] = F
            print(f"  u_extra {u:.3f} Å  ×{s:<4g}  Rw(Q) {cm['Rw_Q']:.4f}  "
                  f"Rw(r) {cm['Rw_r']:.4f} [<5 Å {cm['Rw_r_local']:.4f}, "
                  f">5 Å {cm['Rw_r_mid']:.4f}]  ({time.time() - t1:.0f} s)")

    print("[4/5] summary")
    summary = scan_summary(scales, u_extra, grids)
    for k in SCAN_METRICS:
        s0, pr, gb = (summary[k]["no_extra_width"], summary[k]["profiled"],
                      summary[k]["grid_best"])
        print(f"  {k:10s} best scale: {s0['scale']:.2f} at u_extra = 0 "
              f"(Rw {s0['Rw']:.4f}{', edge' if s0['on_edge'] else ''}); "
              f"{pr['scale']:.2f} with the width free (Rw {pr['Rw']:.4f}; "
              f"grid best ×{gb['scale']:g}, u_extra {gb['u_extra_A']:g} Å)")
    result = {
        "purpose": "forward closure of the published P-4̄2₁m distortion's "
                   "amplitude against the measured F(Q), with an extra "
                   "uncorrelated width as nuisance",
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "settings": {"grid": args.grid, "r_max_A": args.rmax,
                     "temperature_K": args.temperature,
                     "calculator": f"{args.calc}-{args.model}",
                     "mean_positions": args.mean,
                     "injected_field": inj["mode"],
                     "gamma_part": inj["gamma_part"]},
        "injected_amplitudes_at_scale_1_A": inj["injected_amplitudes_A"]["w8"],
        "scales": scales.tolist(), "u_extra_A": u_extra.tolist(),
        "grids": {k: np.round(v, 5).tolist() for k, v in grids.items()},
        "summary": summary,
        "runtime_s": round(time.time() - t0, 1),
    }
    print("[5/5] outputs")
    (out / "scan.json").write_text(json.dumps(result, indent=2))
    np.savez(out / "scan.npz", scales=scales, u_extra=u_extra, Q=Q,
             F_measured=F_meas, F=F_all,
             **{k: v for k, v in grids.items()})
    _plot_scan(out, scales, u_extra, grids, summary)
    print(f"  wrote {out}/scan.json, scan.npz, scan.png in "
          f"{time.time() - t0:.0f} s")
    return 0


def _plot_scan(out, scales, u_extra, grids, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"Rw_Q": "Rw(Q), box-convolved F(Q)",
              "Rw_r": "Rw(r), 1.5 Å – L/2",
              "Rw_r_local": "Rw(r), 1.5–5 Å", "Rw_r_mid": "Rw(r), 5 Å – L/2"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, k in zip(axes.ravel(), SCAN_METRICS):
        for iu, u in enumerate(u_extra):
            ax.plot(scales, grids[k][iu], "o-", ms=3, lw=1,
                    label=f"u_extra = {u:g} Å")
        pr = summary[k]["profiled"]
        ax.axvline(pr["scale"], color="k", lw=0.6, ls="--")
        ax.set(title=titles[k], xlabel="× published distortion",
               ylabel="Rw")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "scan.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# stage
# ----------------------------------------------------------------------------

def render_submit(template: str, stem: str, i: int) -> str:
    """Per-chain SLURM script from the original submit.sh.

    Replaces the job name, output file, working directory (→ the submit
    directory) and the rmcprofile stem; everything else (QOS, walltime,
    node shape, OpenMP settings, RMCProfile path) is kept verbatim. Raises if
    any anchor is missing, so a changed template fails loudly.
    """
    rules = [
        (r"(#SBATCH --job-name=)\S+", rf"\g<1>{stem.lower()}_{i}"),
        (r"(#SBATCH --output=)\S+", rf"\g<1>{stem}_{i}.out"),
        (r"^cd .*$", 'cd "$SLURM_SUBMIT_DIR"'),
        (r"(exe/rmcprofile )\S+", rf"\g<1>{stem}_{i}"),
    ]
    text = template
    for pat, rep in rules:
        text, n = re.subn(pat, rep, text, flags=re.M)
        if n != 1:
            raise ValueError(f"submit template: expected one match for {pat!r}"
                             f", found {n}")
    return text


def cmd_stage(args):
    if args.outdir is None:
        args.outdir = Path(f"results/rmc_control/{args.arm}_run")
    tpl, synth, out = args.template, args.synth, args.outdir
    fq = synth / "scale_ft_rmc.fq"
    dat = tpl / args.dat_name
    start = tpl / args.start_name
    submit_tpl = (tpl / "submit.sh").read_text()
    for p in (fq, dat, start):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    data_file = dat_value(dat, "FILENAME")
    if data_file != fq.name:
        raise SystemExit(f"{dat.name} reads {data_file!r}, not {fq.name!r}")
    synth_man = (json.loads((synth / "synth_manifest.json").read_text())
                 if (synth / "synth_manifest.json").is_file() else None)
    injected = (synth_man or {}).get("injection", {}).get("mode", "none")
    if (args.arm == "null") != (injected == "none"):
        raise SystemExit(f"arm {args.arm!r} but {synth} carries injection "
                         f"{injected!r} — null data for the null arm only")
    if args.stem is None:
        args.stem = f"GTS_5K_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fq, out / fq.name)
    for aux in args.aux:
        if (tpl / aux).is_file():
            shutil.copy2(tpl / aux, out / aux)
    for i in range(1, args.n_chains + 1):
        stem = f"{args.stem}_{i}"
        shutil.copy2(dat, out / f"{stem}.dat")     # byte copy: the GTS .dat
        shutil.copy2(start, out / f"{stem}.rmc6f")  # is CRLF — keep it so
        (out / f"submit_{i}.sh").write_text(
            render_submit(submit_tpl, args.stem, i))
    (out / "submit_all.sh").write_text(
        "#!/bin/bash\n# Submit every control chain (mirrors submit_seq.sh).\n"
        f"for i in $(seq 1 {args.n_chains}); do\n"
        "    sbatch submit_${i}.sh\n    sleep 1\ndone\n")
    for f in list(out.glob("submit*.sh")):
        f.chmod(0o755)
    manifest = {
        "arm": args.arm, "n_chains": args.n_chains, "stem": args.stem,
        "template_dir": str(tpl),
        "identical_to_original": [args.dat_name, args.start_name,
                                  "submit.sh job shape", *args.aux],
        "differs": ["scale_ft_rmc.fq (synthetic)"],
        "sha256": {"scale_ft_rmc.fq": _sha256(out / fq.name),
                   "dat": _sha256(dat), "start": _sha256(start)},
        "synth_manifest": synth_man,
        "staged": time.strftime("%Y-%m-%d %H:%M"),
    }
    (out / "stage_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(_stage_readme(args, synth_man))
    print(f"  staged {args.n_chains} chains in {out} "
          f"({sum(1 for _ in out.iterdir())} files)")
    return 0


def _stage_readme(args, synth_man=None):
    inj = (synth_man or {}).get("injection", {"mode": "none"})
    if inj["mode"] == "none":
        what = ("the MLIP quantum-harmonic null model (zero static\n"
                "disorder by construction)")
        readout = "control.npz"
    else:
        a4 = inj["injected_amplitudes_A"]["w4"]
        what = (f"the published P-4̄2₁m distortion ×{inj['scale']:g} (Γ part "
                f"{inj['gamma_part']};\nX5 {a4['X5']:.4f}, X3 {a4['X3']:.4f}, "
                f"W4 {a4['W4']:.4f}, Δ {a4['D']:.4f} Å through the projector) "
                "as\nSTATIC offsets on the same MLIP quantum motion — a "
                "known answer")
        readout = f"{args.arm}.npz"
    return f"""# RMC control run — {args.arm} arm ({args.n_chains} chains)

Synthetic X-ray F(Q) of {what}, refined by RMCProfile exactly as the
measured 5 K data were: same `.dat`, same ideal starting box, same auxiliary
inputs, same job shape. Only `scale_ft_rmc.fq` differs. Generated by
`rmc_control.py synth/stage` (see `stage_manifest.json`, `docs/control-
experiment-plan.md` in the repo). Private data — do not publish.

## Run (NERSC Perlmutter, as the original)

    rsync -a {args.outdir.name}/ perlmutter:<scratch>/{args.outdir.name}/
    cd <scratch>/{args.outdir.name} && ./submit_all.sh

Each `submit_<i>.sh` is the original `submit.sh` with only the job name,
output file, working directory (`$SLURM_SUBMIT_DIR`) and stem changed:
one CPU node, 8 OpenMP threads, 24 h — the original chains reached
~2.2 M generated moves in that budget. Check that the RMCProfile path in
the scripts still exists.

Cost: {args.n_chains} chains × 1 node × 24 h (the original allocation shape).
The chains use 8 threads each, so several could share a node; that changes
per-chain speed and therefore the move count reached, so compare at matched
moves if you pack them.

## Bring back

The final `{args.stem}_<i>.rmc6f` of every chain (plus `.chi2`/`.log` for
convergence). Then, in the repo:

    python mode_project.py <returned_dir> --exclude AVERAGE -o {readout}

and pass it to `rmc_control.py compare` (see docs/control-experiment-plan.md
— the positive arms enter as `--positive <npz> <synth_dir>`).
"""


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------

def cmd_compare(args):
    meas = dict(np.load(args.measured))
    ctrl = dict(np.load(args.control))
    positives = []
    for npz, synth_dir in args.positive or []:
        man = json.loads((Path(synth_dir) / "synth_manifest.json").read_text())
        inj = man.get("injection", {})
        if inj.get("mode", "none") == "none":
            raise SystemExit(f"{synth_dir} is not a positive-control synth")
        positives.append((Path(synth_dir).name, dict(np.load(npz)),
                          inj["injected_amplitudes_A"]))
    rep = compare_projections(meas, ctrl, n_boot=args.n_boot, seed=args.seed,
                              positives=positives)
    rep["inputs"] = {"measured": str(args.measured),
                     "control": str(args.control),
                     "positive": [list(map(str, p)) for p in
                                  (args.positive or [])]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"  r_ctrl at w={args.window} (measured/control ⟨A²⟩, 95 % CI):")
    for k, row in rep[f"w{args.window}"].items():
        print(f"    {k:3s} {row['r_ctrl']:6.3f}  [{row['ci95'][0]:.3f}, "
              f"{row['ci95'][1]:.3f}]  {row['class']}")
    for name, arm in rep.get("positive", {}).items():
        print(f"  positive arm {name} at w={args.window}: recovery ρ, and the "
              "measured static amplitude it implies (95 % CI):")
        for k, row in arm[f"w{args.window}"].items():
            est = row["static_estimate_A"]
            print(f"    {k:3s} inj {row['injected_A']:.4f}  ρ {row['recovery']:6.3f}"
                  f" [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]  A_static "
                  + ("—" if est is None else f"{est:.4f} {row['static_ci95']}"))
    e = rep["ensembles"]
    print(f"  {e['n_measured']} measured vs {e['n_control']} control configs;"
          f" median moves {e['moves_measured_median']:.0f} vs "
          f"{e['moves_control_median']:.0f}")
    print(f"  wrote {args.out}")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(s, grid, rmax):
        s.add_argument("--start", type=Path, required=True,
                       help="the ideal starting box every RMC chain began "
                       "from")
        s.add_argument("--data", type=Path, required=True,
                       help="measured .fq (Q grid, noise level, closure)")
        s.add_argument("--dat", type=Path, required=True,
                       help="the RMCProfile .dat (RESOLUTION_CORRECTION)")
        s.add_argument("-T", "--temperature", type=float, default=5.0)
        s.add_argument("--calc", default="mace",
                       choices=["mace", "chgnet", "emt"])
        s.add_argument("--model", default="small",
                       choices=["small", "medium", "large"])
        s.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        s.add_argument("--fmax", type=float, default=1e-3)
        s.add_argument("--symprec", type=float, default=1e-3)
        s.add_argument("--displacement", type=float, default=0.03)
        s.add_argument("--fc-dim", type=int, default=4,
                       help="FC supercell edge in unit cells")
        s.add_argument("--grid", type=int, default=grid,
                       help="q-grid / correlation box edge in unit cells")
        s.add_argument("--rmax", type=float, default=rmax,
                       help="g(r) range, Å (Qdamp envelope must have died)")
        s.add_argument("--dr", type=float, default=0.01)
        s.add_argument("--mean", default="start",
                       choices=["start", "relaxed"],
                       help="mean positions of the synthetic crystal: the "
                       "RMC starting (experimental) positions, or the MLIP "
                       "minimum")
        s.add_argument("--keep-gamma", action="store_true",
                       help="keep the injected field's k = 0 "
                       "(parent-reference) part")
        s.add_argument("--cif", type=Path, default=Path("data/GTS_5K.cif"),
                       help="parent CIF in RMC site-id order (injection)")

    s = sub.add_parser("synth", help="synthetic null-model F(Q)")
    model_args(s, grid=16, rmax=120.0)
    s.add_argument("--noise-scale", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--inject", default="none", choices=["none", "published"],
                   help="positive control: add the published P-4̄2₁m "
                   "distortion as static offsets")
    s.add_argument("--inject-scale", type=float, default=1.0,
                   help="multiply the injected field (1 = published)")
    s.add_argument("-o", "--outdir", type=Path, default=None,
                   help="default results/rmc_control/synth, or "
                   "synth_positive_x<scale> with --inject")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("scan", help="forward closure: published-distortion "
                       "scale (× extra width) vs the measured F(Q)")
    model_args(s, grid=8, rmax=60.0)
    s.add_argument("--scales", default="0,0.5,1,1.5,2,2.5,3,3.5,4",
                   help="comma-separated multiples of the published field")
    s.add_argument("--u-extra", default="0,0.02,0.04,0.06,0.08",
                   help="comma-separated extra isotropic uncorrelated "
                   "displacement, Å rms per component (nuisance width)")
    s.add_argument("-o", "--outdir", type=Path,
                   default=Path("results/rmc_control/scale_scan"))
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("stage", help="NERSC run directory")
    s.add_argument("--synth", type=Path, required=True)
    s.add_argument("--template", type=Path, required=True,
                   help="the original ensemble run directory")
    s.add_argument("--dat-name", default="GTS_5K.dat")
    s.add_argument("--start-name", default="GTS_5K.rmc6f")
    s.add_argument("--aux", nargs="*", default=["optimization.dat"])
    s.add_argument("--arm", default="null",
                   help="'null' (needs uninjected synth data) or a positive "
                   "arm name, e.g. positive_x1")
    s.add_argument("--stem", default=None, help="default GTS_5K_<arm>")
    s.add_argument("--n-chains", type=int, default=64)
    s.add_argument("-o", "--outdir", type=Path,
                   default=None, help="default results/rmc_control/<arm>_run")
    s.set_defaults(func=cmd_stage)

    s = sub.add_parser("compare", help="measured vs control projections")
    s.add_argument("--measured", type=Path, required=True)
    s.add_argument("--control", type=Path, required=True)
    s.add_argument("--positive", nargs=2, action="append",
                   metavar=("NPZ", "SYNTH_DIR"),
                   help="a positive-control arm: its projections and the "
                   "synth directory holding its injected amplitudes "
                   "(repeatable)")
    s.add_argument("-w", "--window", type=int, default=4, choices=[2, 4, 8])
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--out", type=Path,
                   default=Path("results/rmc_control/control_report.json"))
    s.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
