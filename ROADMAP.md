# mlip-quantum-thermal — Development Roadmap

*Last updated: 2026-09-29. Self-contained: the Python pipeline computes and
`viewer/` displays. Detached from `rmc-phonon-dynamics` on 2026-07-23; see
README "Scope pivot".*

---

## Done

- [x] **Repo scaffold** — README, requirements, `CLAUDE.md`, and
      `milestone1_bands.py` (rmc6f → averaged unit cell → MLIP relax →
      phonopy `band.yaml` + `relaxed.cif` + `summary.json`). Parser and
      circular averaging unit-verified on synthetic wrap-around data.

## Done — Milestone 1 hardening

*Green end-to-end on synthetic fcc Cu (EMT + MACE-MP-0) **and on the real
target dataset**: GaTa₄Se₈ 5 K RMC ensemble, single config and all 493
converged configs (F-4̄3m, dynamically stable, 39 branches). `pytest -q` is
50 tests (unit + `slow`/`mace` integration).*

- [x] pytest suite: parser fixtures, circular mean, `--ref` CIF assignment,
      `auto_dim`, mixed-occupancy handling (`tests/`, 21 tests)
- [x] synthetic Cu fcc ensemble generator (`tests/fixtures/`)
- [x] EMT end-to-end integration test; fix ASE/phonopy API friction
      (threaded `--symprec` into phonopy so `primitive_matrix="auto"` finds the
      fcc primitive; migrated off deprecated `get_band_structure_dict()`)
- [x] first real MACE-MP-0 run on the fixture (stability + Γ acoustics
      asserts) — surfaced + fixed the 0.01 Å MLIP-noise imaginary-mode artifact
      (default displacement now 0.03 Å)
- [x] target-dataset run: GaTa₄Se₈ 5 K (`data/`, 8×8×8 = 26,624-atom configs).
      Config 1 and the full 493-converged-config ensemble both → F-4̄3m,
      dynamically stable, same MACE minimum (max band Δω 0.0009 THz); ensemble
      average recovers F-4̄3m at 10× tighter symprec (0.01 vs 0.1). Caveats to
      carry: +1.6 % lattice overexpansion (MACE-MP-0 small), no NAC/LO–TO yet.
      Added `--skip-nonconverged` (drops 0-move configs; found exactly the 8
      bad ones: 0, 141, 161, 170, 413, 419, 442, 492).
- [x] `--max-configs` / `--stride` / `--seed` for 500+ config ensembles
      (`select_configs`, recorded in `summary.json["sampling"]`)

## Milestone 2 — finite temperature

- [x] `docs/milestone2-plan.md` — reviewed over three rounds (quantum sampling
      at 5 K, closure at the experimental lattice, hiPhive engine, powder
      q-star degeneracy, MACE-as-null-model confirmed by probes)
- [x] `md_run.py`: `sample` mode = harmonic **quantum** sampling (zero-point
      correct at 5 K; MSD verified against phonopy's analytic ⟨u²⟩);
      `md` mode = classical Langevin cross-check with low-T warning.
      NPT deferred until a higher-T dataset exists.
- [x] closure: native partial-g_ij(r) → Faber–Ziman G(r) → F(Q)=S(Q)−1 on the
      measured STOG grid, scale+offset fitted (mirrors RMC), Rw in Q and r
- [x] temperature-effective force constants → `band_T.yaml` (hiPhive 2nd
      order; validated band_T ≈ band on Cu/EMT)
- [x] GTS acceptance run (5 K dataset, cubic null model): u_rms(quantum) =
      0.040 Å/comp vs RMC ensemble 0.095 Å/comp (the M3 target gap);
      baseline closure Rw(Q) = 0.74 / Rw(r) = 0.58 (Q-space dominated by
      box-truncation vs instrument peak-shape mismatch — needs resolution
      convolution before the residual is fully interpretable); measured G(r)
      peaks systematically broader than the quantum null model = static/
      anharmonic excess. band_T ≈ band (max |Δω| = 0.18 THz at 5 K) —
      machinery validated on the real system.
- [ ] closure refinement: instrument-resolution / box-size-matched F(Q)
      comparison so the null-model residual isolates the distortion.
      **Found 2026-09-29: the GTS F(Q) is X-ray data** (the RMC `.dat` fits
      `XRAY_RECIPROCAL_SPACE_DATA`), but `md_run.py`'s closure weights with
      neutron b_coh — Ta carries ~6× more relative contrast in X-rays, so
      the M2 acceptance Rw(Q) = 0.74 / scale 0.63 are partly a weighting
      mismatch. The X-ray forward model now exists (`md_run.xray_fq`); the
      closure CLI still needs a `--radiation` switch and a rerun.

## Milestone 3 — experiment-constrained FCs + verdicts

- [x] `docs/verdicts-schema.md` — **drafted, awaiting review**: model-free
      X5/X3/W4/Δ1 projections from the paper's supplemental tables, star
      pooling, stiff-mode RMC-noise ruler, ω-source hierarchy
      (GDOS → QE → MACE), SOC-tiered well fields, synthetic validation suite
- [x] mode-pattern reference ingested: `reference/gts_mode_patterns.json`
      from the SM Tables II–X (pypdf extraction + verification suite).
      Found and corrected a probable sign typo in printed SM Table VIII
      (X₃/Se3 δz/c: −0.0053 → +0.0053; flagged for author verification) —
      after correction, all six implied mode amplitudes reproduce the
      published Table II values to ≤3 % (X₅ 0.1201/0.1196 …), pinning the
      normalization convention (parent-primitive-cell norm, ÷√8).
- [x] `mode_project.py`: model-free projection engine — AMPLIMODES↔SHELX
      site mapping solved numerically (Hungarian assignment + whole-orbit
      anchor criterion), patterns expanded via SG-113 ops (covariance of the
      refined field verified exact), parent idealized by Reynolds
      projection, frame alignment with free origin, star pooling via the
      full parent-group variant orbit, and JOINT (competitive) projection
      to kill rounding-borne cross-irrep leakage. Validated: published
      amplitudes reproduced from the printed tables; injected modes
      recovered across arms. (Windowed OP + noise ruler: next.)
- [x] real-ensemble projection run (490 configs, 4 s): **the local
      structure is X₅-distorted at roughly half the coherent static
      amplitude (mean 0.056 Å vs published 0.120, distribution reaching
      0.131), while X₃ is locally absent (0.018 vs published 0.072 — same
      k-star/subspace as X₅, so noise-immune contrast)**. W₄ at ~published
      with fat tail; config 190 is a genuinely ordered box (X₅ = 0.110).
      No ensemble-wide χ²–amplitude correlation. Note: global projection is
      domain-diluted → 0.056 is a lower bound on the local X₅ amplitude.
- [x] windowed/local projections + nulls + quantum yardstick →
      **first `verdicts.json` (schema v0.1) emitted.** Corrected method:
      the cross-channel raw-amplitude comparison was invalid (channel
      pedestals differ 3×); verdicts use the three-component expectation
      r = A²_meas / [A²_qh + f_noise·A²_null] with f_noise = 0.82 measured
      (σ_tot 0.095 vs σ_qh 0.040), the quantum baseline sampled on the
      full 26,624-atom box through the identical projector (u_rms 0.0693 Å
      = the M2 value, independent cross-check). **Verdicts (w=4, ~41 Å):
      X₅ 2.19 mixed, W₄ 2.50 mixed, Γ₃ 2.38 mixed*, Γ₁ 1.69 mixed*,
      X₃ 1.37 dynamic, Δ 0.79 dynamic (*Γ reference caveat). All channels
      drop to ~0.6–0.75 globally: local static order at 20–40 Å, domain-
      cancelled at box scale — the order–disorder picture, mode-resolved.**
- [x] `hiphive_fit.py`: RMC displacement snapshots + MLIP forces →
      effective FCs → `band_rmc.yaml` sidecar. GTS run (8 instantaneous
      boxes, `--exclude AVERAGE`): **mean softening −0.10 THz vs the
      harmonic null** (−0.22 on a disjoint snapshot set — stable), max
      shift ~1 THz; shallow instabilities (≥ −0.29 THz) only at
      incommensurate ⟨112⟩-like q, at the fit-noise scale — **no X-point
      well emerges**: RMC sampling reweights the MACE surface but cannot
      inject the well it lacks (consistent with probe D / SOC tiering).
      Caveat: config 326 (partial run) was 1 of 8 snapshots.
- [x] mode-projection amplitudes vs (ħ/2ω)coth(ħω/2k_BT) — done via the
      quantum baseline through the identical projector (see verdicts)
- [x] E(Q) mode mapping: irrep patterns → null-model phonon branches over
      the full k-stars. Γ₁ cleanly carried (0.88 on the 8.3 THz breathing
      branch); **X₅/W₄ spread over many branches (max overlap ~0.10)** —
      the frozen pattern is not a cubic eigenmode; the 13-meV fingerprint
      requires the distorted-phase (M4 fine-tuned) model. ω fields written
      into verdicts.json (omega_source = mace-null).
- [x] emit `verdicts.json` (per-mode static/dynamic badges) — **M3
      COMPLETE**
- [x] `viewer/` — 3D mode animator, band plot and simulated-INS panel
      vendored in-repo (2026-07-23) from `rmc-phonon-dynamics` @ MIT; the
      covariance route (`math/symmetrize.js`, `symmetry.js`, `cells.js`, the
      WebGPU S(k) pipeline) was deliberately left behind.
- [ ] viewer: overlay several band yamls on one axis + verdict badge panel
- [ ] **control experiment for the M3b verdicts** — run RMCProfile on
      *synthetic* F(Q) generated from the MLIP quantum ensemble (zero static
      disorder by construction), push the output through `mode_project.py`
      unchanged, and read off the per-mode amplitude RMC manufactures from
      data containing none. The existing nulls (`shuffle_cells`,
      `random_signs`) take the configs as given and cannot test this;
      `f_noise` is currently one scalar applied flat across mode space.
      Until this runs, "X5 is static" is not separable from "X5 is where the
      RMC move statistics pile up". Started 2026-09-29:
  - [x] RMCProfile's forward model pinned exactly (`md_run.xray_*`): WK
        X-ray form factors with ⟨f⟩² normalization, partials with a
        **Gaussian** Qdamp envelope, and CONVOLVE = G(r) truncation of the
        *data* at L/2 — each reproduced from RMCProfile's own output files
        to ≤5e-8 (the data column to Rw 0.001).
  - [x] frame-correct ensemble projection driver (`python mode_project.py
        <ensemble> -o proj.npz`): maps every box from the RMC frame into the
        pattern frame (parent rotation + half-cell shift) before projecting;
        known-answer tests through an rmc6f round trip. The 490-config
        measured ensemble now projects reproducibly in 23 s. **`verdicts.json`
        (v0.1) cannot be regenerated from committed code** — its scratch
        driver was never committed and its measured amplitudes sit 2–7 %
        from every reconstruction tried (X5 w4: published 0.3012, committed
        0.3225). Regenerate it through this driver (needs the quantum
        baseline recomputed in the same frame).
  - [x] `harmonic_pdf.py`: analytic quantum-harmonic partial g(r) of the
        infinite crystal — correlated pair widths from phonon eigenvectors
        on an M³ q-grid via FFT, no box, no sampling noise. Correlations
        equal phonopy's own supercell ⟨uuᵀ⟩ to 1e-18; shells match sampled
        quantum snapshots (coordination exact, widths to 0.2 %).
  - [x] synthetic null F(Q) for GTS (`rmc_control.py synth`, 46 s):
        MACE-MP-0 small at the experimental lattice, 5 K quantum, 16³
        grid, r ≤ 120 Å, grid-scale noise from 6th differences. Against
        the measured data (both box-convolved): Rw(Q) 0.248, Rw(r) 0.40.
  - [x] staged 64-chain run (`results/rmc_control/null_run/`, `.dat` /
        start box / `optimization.dat` byte-identical to the original;
        64 chains → 5–6 % SE on the control ⟨A²⟩ at w = 4)
  - [ ] RMCProfile runs (user, NERSC) → `rmc_control.py compare`
  - [ ] positive-control arm (published P-4̄2₁m distortion + quantum
        motion: RMC's recovery of a known amplitude)
  - [ ] regenerate `verdicts.json` through the committed driver with the
        control-calibrated per-mode noise

## Vision — model-space RMC / "dynamic EPSR" (concept, post-M3)

- [ ] `docs/idea-dynamic-refinement.md` captures the long-range inversion:
      refine a model (structure ⊕ potential) whose MLIP-generated quantum
      ensemble reproduces G(r) + S(Q) + Bragg + INS S(Q,ω) jointly — static
      vs dynamic separated by construction. Literature-mapped 2026-07-19;
      the assembled package appears novel. M3 is its linearized first pass;
      M2 is its forward evaluator.

## Milestone 4 — application & methods paper

- [ ] real datasets end-to-end; error budget of covariance vs MLIP bands
- [ ] **DFT fine-tuning — now on the critical path** (probe D): MACE-MP-0
      penalizes the refined GaTa₄Se₈ P-4̄2₁m distortion by +293 meV/f.u.
      (pure single well at cubic) where the paper's plain-PBE QE retains it.
      **Tiered honestly (mechanism is SOC — user):** Tier 1, harmonic
      dynamics *at* the ordered structure, is PBE-representable (the paper's
      own PDOS evidence) → plain-PBE/Δ-learning fine-tune buys tetragonal
      phonons and the 13-meV forward model. Tier 2, landscape energetics
      (well depth/barriers/degeneracy splittings), requires PBE+SOC(+U)
      training data — conditional claims only (PBE+SOC w/o U is metallic in
      the doubled cell). Tier 3, the nonadiabatic Mexican-hat fluctuation
      mechanism and T* selection, is beyond ANY adiabatic MLIP — it is
      measured by M3 and encoded phenomenologically by the dynamic-EPSR
      refinement (docs/idea-dynamic-refinement.md), not derived ab initio.
- [ ] manuscript draft
