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


def compare_projections(meas, ctrl, windows=(2, 4, 8), n_boot=2000, seed=0):
    """Per-irrep control ratio r_ctrl = ⟨A²⟩_meas / ⟨A²⟩_ctrl, all scales.

    meas / ctrl: mappings with 'keys', 'moves' and per scale 'rms_w{w}',
    'rms_null_w{w}' (n_cfg, n_keys) as written by mode_project's driver.
    Bootstrap resamples configurations of BOTH ensembles; the interval is
    the 2.5–97.5 percentile. Classification of the interval:
    'excess' (lower bound > 1), 'deficit' (upper < 1), else 'explained'.
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
    return out


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


def cmd_synth(args):
    import harmonic_pdf as hp
    from ase.io import write as ase_write
    from ase.optimize import FIRE

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
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

    print(f"[3/5] harmonic g(r): T = {args.temperature} K, grid {args.grid}³, "
          f"r ≤ {args.rmax} Å")
    r, g, info = hp.harmonic_partials(ph_mean, args.temperature, M=args.grid,
                                      r_max=args.rmax, dr=args.dr)
    fmin = float(np.min(ph_mean.qpoints.frequencies))   # the grid just run

    print("[4/5] X-ray F(Q) on the measured grid + noise")
    Q, F_meas = md_run.parse_fq(args.data)
    symbols = list(ph_mean.unitcell.symbols)
    F_clean = md_run.xray_fq(r, g, symbols, Q, rho0, qdamp)
    # the comparison RMCProfile makes: both sides box-convolved, scale+offset
    conv_meas = md_run.rmc_box_convolve(Q, F_meas, box_len, rho0)
    conv_syn = md_run.rmc_box_convolve(Q, F_clean, box_len, rho0)
    s, o, rw_q = md_run.fit_scale_offset(conv_meas, conv_syn)
    r_g = np.arange(args.dr, box_len / 2, args.dr)
    G_meas = md_run.fq_to_gr(Q, F_meas, r_g, rho0)
    G_syn = md_run.fq_to_gr(Q, F_clean, r_g, rho0)
    m = r_g > 1.5
    s_r, o_r, rw_r = md_run.fit_scale_offset(G_meas[m], G_syn[m])
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
             **{f"g_{a}_{b}": v for (a, b), v in g.items()})
    manifest = {
        "purpose": "zero-static-disorder synthetic X-ray F(Q) for the RMC "
                   "control experiment (docs/control-experiment-plan.md)",
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
        "synth_manifest": json.loads((synth / "synth_manifest.json").read_text())
        if (synth / "synth_manifest.json").is_file() else None,
        "staged": time.strftime("%Y-%m-%d %H:%M"),
    }
    (out / "stage_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(_stage_readme(args))
    print(f"  staged {args.n_chains} chains in {out} "
          f"({sum(1 for _ in out.iterdir())} files)")
    return 0


def _stage_readme(args):
    return f"""# RMC control run — {args.arm} arm ({args.n_chains} chains)

Synthetic X-ray F(Q) of the MLIP quantum-harmonic null model (zero static
disorder by construction), refined by RMCProfile exactly as the measured
5 K data were: same `.dat`, same ideal starting box, same auxiliary inputs,
same job shape. Only `scale_ft_rmc.fq` differs. Generated by
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

    python mode_project.py <returned_dir> --exclude AVERAGE -o control.npz
    python rmc_control.py compare --measured results/m3_projections_aligned.npz \\
        --control control.npz -o results/rmc_control/control_report.json
"""


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------

def cmd_compare(args):
    meas = dict(np.load(args.measured))
    ctrl = dict(np.load(args.control))
    rep = compare_projections(meas, ctrl, n_boot=args.n_boot, seed=args.seed)
    rep["inputs"] = {"measured": str(args.measured),
                     "control": str(args.control)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"  r_ctrl at w={args.window} (measured/control ⟨A²⟩, 95 % CI):")
    for k, row in rep[f"w{args.window}"].items():
        print(f"    {k:3s} {row['r_ctrl']:6.3f}  [{row['ci95'][0]:.3f}, "
              f"{row['ci95'][1]:.3f}]  {row['class']}")
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

    s = sub.add_parser("synth", help="synthetic null-model F(Q)")
    s.add_argument("--start", type=Path, required=True,
                   help="the ideal starting box every RMC chain began from")
    s.add_argument("--data", type=Path, required=True,
                   help="measured .fq (sets the Q grid and the noise level)")
    s.add_argument("--dat", type=Path, required=True,
                   help="the RMCProfile .dat (RESOLUTION_CORRECTION)")
    s.add_argument("-T", "--temperature", type=float, default=5.0)
    s.add_argument("--calc", default="mace", choices=["mace", "chgnet", "emt"])
    s.add_argument("--model", default="small",
                   choices=["small", "medium", "large"])
    s.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    s.add_argument("--fmax", type=float, default=1e-3)
    s.add_argument("--symprec", type=float, default=1e-3)
    s.add_argument("--displacement", type=float, default=0.03)
    s.add_argument("--fc-dim", type=int, default=4,
                   help="FC supercell edge in unit cells")
    s.add_argument("--grid", type=int, default=16,
                   help="q-grid / correlation box edge in unit cells")
    s.add_argument("--rmax", type=float, default=120.0,
                   help="g(r) range, Å (Qdamp envelope must have died)")
    s.add_argument("--dr", type=float, default=0.01)
    s.add_argument("--mean", default="start", choices=["start", "relaxed"],
                   help="mean positions of the synthetic crystal: the RMC "
                   "starting (experimental) positions, or the MLIP minimum")
    s.add_argument("--noise-scale", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--outdir", type=Path,
                   default=Path("results/rmc_control/synth"))
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("stage", help="NERSC run directory")
    s.add_argument("--synth", type=Path, required=True)
    s.add_argument("--template", type=Path, required=True,
                   help="the original ensemble run directory")
    s.add_argument("--dat-name", default="GTS_5K.dat")
    s.add_argument("--start-name", default="GTS_5K.rmc6f")
    s.add_argument("--aux", nargs="*", default=["optimization.dat"])
    s.add_argument("--stem", default="GTS_5K_null")
    s.add_argument("--arm", default="null")
    s.add_argument("--n-chains", type=int, default=64)
    s.add_argument("-o", "--outdir", type=Path,
                   default=Path("results/rmc_control/null_run"))
    s.set_defaults(func=cmd_stage)

    s = sub.add_parser("compare", help="measured vs control projections")
    s.add_argument("--measured", type=Path, required=True)
    s.add_argument("--control", type=Path, required=True)
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
