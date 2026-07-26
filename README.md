# mlip-quantum-thermal

Deciding how much of the disorder seen in total-scattering and spectroscopy
data is **frozen** and how much is **motion** — by generating the thermal
ensemble from a machine-learned interatomic potential (MLIP) with quantum
statistics as the physical null model, and testing the measurement against it.

The methods are statistical, not just least-squares: mode projections scored
against explicit null models, variance decomposition of the observed
displacements, bootstrap confidence on every verdict, and model comparison
against measured G(r)/F(Q). Structural refinement is one tool inside that —
not the frame around it.

Self-contained: the Python pipeline computes, and `viewer/` is a static
browser front end for the files it emits. The pipeline never depends on the
viewer — every milestone runs without node installed.

| File             | What it is                                             |
| ---------------- | ------------------------------------------------------ |
| `band.yaml`      | 0 K harmonic bands (phonopy-standard)                  |
| `band_T.yaml`    | temperature-effective bands, quantum-sampled           |
| `band_rmc.yaml`  | effective bands from RMC snapshots + MLIP forces       |
| `relaxed.cif`    | the equilibrium displacement reference                 |
| `closure.json`   | G(r)/F(Q) closure against measured total scattering    |
| `summary.json`   | provenance and stability report                        |
| `verdicts.json`  | per-mode static/dynamic badges                         |
| `modes_irrep.yaml` | published distortion patterns as Bloch eigenvectors  |

## Scope pivot (2026-07-23, in progress)

RMC is being demoted from **inference engine** to **screening tool**. Total
scattering is the energy integral of S(Q,E), so G(r)/S(Q) alone cannot
distinguish a frozen distortion from a soft mode from order–disorder hopping —
which is exactly why the milestone-3 verdicts needed an RMC ensemble, a
quantum null, and a fitted noise fraction to attack the question indirectly.
Inelastic data resolves it *as data*: at a given Q, a frozen distortion is
elastic, a soft mode is a peak at ±ħω, and hopping is a quasielastic
Lorentzian.

So the intended evidence chain is forward closure against F(Q) + Bragg +
S(Q,E), with RMC used to *suggest* what belongs in the refinement move space
rather than to carry a verdict. Design note: `docs/idea-dynamic-refinement.md`.

**Open gating question:** whether the available SEQUOIA/ARCS reductions retain
the full S(Q,E) map with the elastic line, or only multiphonon-corrected GDOS.
The answer decides how much of the pivot is reachable now.

## Why

Covariance-derived phonons invert amplitudes into frequencies, and RMC
amplitudes are noise-limited and under-constrained (transverse correlations,
near-Γ acoustics in a fixed box, static-disorder inflation). This pipeline
derives frequencies from **MLIP forces** instead; the RMC ensemble supplies
the experiment-constrained structures, sampling geometry, and amplitudes.

## Milestones

1. **Harmonic bands** *(this repo, now)* — fold + circular-average the
   ensemble to one unit cell, MLIP-relax, phonopy finite displacements,
   export `band.yaml` + `relaxed.cif`. Script: `milestone1_bands.py`.
2. **Finite-temperature bands** — MLIP-MD at the experimental temperature;
   renormalized dispersions and linewidths (TDEP / DynaPhoPy-style), plus
   G(r)/S(Q) closure against the measured data.
3. **Experiment-constrained force constants** — hiPhive fit with RMC
   configurations as displacement snapshots and MLIP forces evaluated on
   them; per-mode amplitude vs (ħ/2ω)coth(ħω/2k_BT), E(Q) mode mapping,
   `verdicts.json`.
4. **Application + methods paper** — real datasets, active-learning
   fine-tuning where the foundation model falls short.

## Quickstart (milestone 1)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# all *.rmc6f in a run folder, MACE-MP-0 medium on CPU:
python milestone1_bands.py path/to/run_dir/ -o m1_out

# GPU, explicit phonopy supercell, smaller band.yaml:
python milestone1_bands.py run_dir/ --device cuda --dim 3 3 2 --no-eigenvectors

# large ensemble: keep every 2nd config, then a deterministic 100-config sample:
python milestone1_bands.py run_dir/ --stride 2 --max-configs 100 --seed 0
```

Then view the result:

```bash
cd viewer && npm install && npm run dev
```

Drop `m1_out/band.yaml` onto the page (or deep-link it with
`?load=<url>`) to plot the bands, click any point to animate that mode in 3D,
and open the *Simulated INS* tab for a powder S(|Q|,E) map. Everything runs
client-side; nothing is uploaded.

`--calc emt` runs a dependency-free smoke test of the plumbing (metals-only
toy potential — not for science).

## Caveats

- Universal MLIPs (MACE-MP-0, CHGNet) inherit PBE-level systematics and a
  few-percent global softening: judge band **shape** first; fine-tune on DFT
  for absolute frequencies (milestone 4).
- Finite displacements default to **0.03 Å**, not phonopy's DFT-tuned 0.01 Å:
  at 0.01 Å a universal MLIP's forces sit near its noise floor and spurious
  imaginary modes appear just off Γ even for a stable crystal. Override with
  `--displacement`.
- Phonons need an **ordered** cell: mixed-occupancy sites are reduced to the
  majority element with a warning.
- A diagonal RMCProfile supercell (Nx Ny Nz) is assumed.
- Imaginary modes on the averaged/symmetrized cell are a *result*, not an
  error: they flag a candidate frozen distortion. Re-run on the distorted
  subgroup cell, or handle at temperature in milestones 2–3.

## License

MIT © 2026 Tsung-Han Yang
