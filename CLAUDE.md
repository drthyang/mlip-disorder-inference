# CLAUDE.md — mlip-quantum-thermal

## What this project is

Native Python pipeline that **infers how much observed disorder is static and
how much is dynamic**, using an MLIP-generated quantum-thermal ensemble as the
physical null model, plus `viewer/` — a self-contained browser front end for
the band/mode files it emits. Rationale in `README.md`, plan in `ROADMAP.md`.

The machinery is statistical: joint least-squares mode projection with a
pseudo-inverse Gram (`mode_project.py`), two explicit null models
(`random_signs`, `shuffle_cells`), variance decomposition into a measured
noise fraction, bootstrap confidence, and threshold classification
(`verdicts.py`). Fitting (hiPhive force constants, scale+offset closure) is a
subroutine, not the thesis.

**Scope pivot (2026-07-23, in progress.)** RMC is being demoted from inference
engine to screening tool: evidence should come from forward closure against
F(Q) + Bragg + S(Q,E), not from statistics on RMC configurations. See
`docs/idea-dynamic-refinement.md`. Gating question still open: do the SEQUOIA/
ARCS reductions retain the full S(Q,E) map with the elastic line, or only the
multiphonon-corrected GDOS? The answer decides how much of milestone 3
survives.

Core idea: the viewer's covariance route inverts noisy **amplitudes** into
frequencies; this pipeline derives frequencies from MLIP **forces**, using the
RMC ensemble for experiment-constrained structures, sampling geometry, and
amplitudes.

## Hard boundaries — do not violate

- **The name is settled: `mlip-quantum-thermal`.** It names the physics the
  repo generates — an MLIP-driven quantum-thermal ensemble, zero-point
  included — not the statistical method applied on top. The method has already
  changed twice (refinement → inference → forward closure) and will change
  again; that is **not** a reason to rename. Earned, not aspirational:
  `md_run.quantum_snapshots` samples ⟨u²⟩ ∝ (ħ/2ω)coth(ħω/2k_BT) as the
  default path, classical Langevin is the demoted cross-check, and
  `test_zero_point_dominates_at_low_T` guards it. Prior names
  (`rmc-mlip-phonons`, `mlip-dynamic-refinement`, `mlip-disorder-inference`)
  are dead; GitHub redirects them.
- **The Python pipeline never depends on `viewer/`.** Every milestone must run
  start to finish without node installed. `viewer/` reads the pipeline's
  output files; nothing flows back.
- **`band.yaml` stays phonopy-standard** (`auto_band_structure` output). This
  is no longer an external contract — it is a deliberate choice for interop
  with phonopy, Euphonic, OVITO and phononwebsite. Extensions go in **new
  sidecar files** (`verdicts.json`), never by mutating `band.yaml`.
- Always emit `relaxed.cif` — it is the displacement reference.
- `default_dtype="float64"` for every MLIP force/phonon evaluation.
- `data/`, `results/`, `releases/`, `m1_out*/` are git-ignored; **never commit
  ensembles, trajectories, results, or model weights — the data stays
  private.** `releases/<dataset>/` is the local viewer bundle (band yamls,
  verdicts.json, CIFs, mode files + a README): assembled on disk for the
  viewer to load, never pushed. No exceptions — a bundle was briefly
  committed on 2026-07-20 and purged from history on 2026-07-21.
- Units are Å, eV, THz (phonopy defaults); state units in every docstring.

## Current state

Milestones 1–3 are implemented and have run end to end on the GTS 5 K data:

| Script | Emits |
| --- | --- |
| `milestone1_bands.py` | `band.yaml`, `relaxed.cif`, `summary.json` |
| `md_run.py` | `closure.json`, `gr_sim.dat`, `sq_sim.dat`, `band_T.yaml` |
| `hiphive_fit.py` | `band_rmc.yaml`, `fit_report.json` |
| `mode_project.py` + `verdicts.py` | `verdicts.json` |
| `export_modes.py` | `modes_irrep.yaml`, per-mode `.xyz` |
| `viewer/` | browser front end for any of the band yamls |

**Band paths: always idealize the cell metric first.** `auto_band_structure`
takes no `symprec` and seekpath runs at its hardcoded 1e-5, so ~1e-5 Å of
residual relaxation noise makes a cubic cell read as P1 and the "standard
path" becomes the triclinic one. Every `Phonopy` construction site therefore
calls `milestone1_bands.symmetrize_lattice(atoms, symprec)` first — it
averages the metric tensor over the space group and rebuilds the cell,
**lattice only**, leaving fractional coordinates and atom order untouched
(hiPhive's FCP evaluation and the `ideal`-supercell indexing depend on that).
Fixed 2026-07-23; do not add a `Phonopy(...)` call without it.

## Environment / commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python milestone1_bands.py <run_dir>/ -o m1_out     # real run (MACE-MP-0)
python milestone1_bands.py <run_dir>/ --calc emt    # plumbing smoke test
pytest -q
cd viewer && npm install && npm run dev             # browser front end
```

The EMT smoke test needs only numpy/ase/spglib/phonopy/seekpath — no torch.

Note: ASE's EMT is a metals-only toy potential (Cu, Al, Ag, Au, Ni, Pd, Pt).
Smoke-test fixtures must therefore be a **Cu fcc** synthetic ensemble, not an
ionic compound.

## Conventions

- Python ≥ 3.11, type hints on public functions, numpy-style docstrings.
- Pure functions + a thin argparse CLI. Milestone scripts stay runnable
  standalone; refactor into a `mlip_quantum_thermal/` package only at
  milestone 3.
- Every physics routine gets a synthetic-data unit test (follow the
  wrap-around circular-mean pattern).
- Keep `ROADMAP.md` checkboxes and `CHANGELOG.md` current — update both at the
  end of every completed task, in the same commit.

## Immediate tasks — Milestone-1 hardening (do in order, stop after each)

1. **Test suite.** `tests/` with pytest: rmc6f parser fixtures (with/without
   site-id columns, bracketed labels, `Cell (Ang/deg)` vs `Lattice vectors`
   headers), circular-mean wrap-around, `--ref` CIF site assignment,
   `auto_dim`, mixed-occupancy majority reduction + warning.
2. **Fixture generator.** `tests/fixtures/make_synthetic_ensemble.py`:
   writes N noisy rmc6f configs of a Cu fcc supercell (known answer:
   3 acoustic branches, ω → 0 at Γ).
3. **EMT end-to-end.** Run the full script on the Cu fixture with
   `--calc emt`; fix any ASE/phonopy API friction until it produces
   `band.yaml`/`relaxed.cif`/`summary.json`. Add this as an integration test
   (marked `slow`).
4. **Real-deps run.** Install requirements; run MACE-MP-0 `--model small` on
   the same fixture. Assert: dynamically stable (min band frequency
   > −0.05 THz), acoustic branches vanish at Γ.
5. **Ensemble ergonomics.** `--max-configs` and `--stride` subsampling for
   500+ config ensembles; deterministic with `--seed`.
6. Update `ROADMAP.md` + `CHANGELOG.md`.

## Milestone 2 (only after M1 is green)

`md_run.py`: MLIP-MD at the experimental temperature (ASE Langevin, then
NPT), trajectory → G(r) and S(Q) for closure against measured data, and
temperature-effective force constants → `band_T.yaml`. Design notes first in
`docs/milestone2-plan.md`; get the plan reviewed before implementing.

## Milestone 3 (design doc before code)

`docs/verdicts-schema.md` — the sidecar contract consumed by the viewer:
per-mode `{q, branch, omega_THz, amplitude_ratio, well: single|double,
barrier_meV, verdict: dynamic|static|mixed, confidence}`. Then
`hiphive_fit.py` (RMC displacements + MLIP forces → effective FCs → bands),
mode-projection amplitudes vs (ħ/2ω)coth(ħω/2k_BT), and E(Q) mode mapping.
