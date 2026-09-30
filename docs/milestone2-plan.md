# Milestone 2 design — finite-temperature closure and effective force constants

*Status: **draft for review** (CLAUDE.md gates M2 implementation on this
review). Author: pipeline; target dataset: GaTa₄Se₈ 5 K RMC ensemble.*

## Goal

Close the loop between the MLIP model and the measured data, and produce
temperature-effective force constants:

1. **Closure**: simulate G(r) and S(Q) from MLIP-sampled configurations and
   compare quantitatively against the measured `scale_ft_rmc.fq`
   (S(Q), Q = 0.8–27 Å⁻¹, ΔQ = 0.01 Å⁻¹, 2619 points). **Correction
   (2026-09-29): these are X-ray data**, not neutron as first written
   here — see D4.
2. **`band_T.yaml`**: effective (temperature-renormalized) phonon bands at the
   experimental temperature, as a **new sidecar** — `band.yaml` stays frozen.

Non-goals for M2: verdicts/static-dynamic classification (M3), NAC/Born
charges (optional M3+), viewer changes (other repo), DFT fine-tuning (M4).

## Physical design decisions (the ones review should challenge)

### D1 — 5 K is a quantum problem: sampling method

At 5 K, k_BT ≈ 0.43 meV ≈ 0.1 THz — essentially every phonon in this material
(lowest optical 1.3 THz, acoustic zone-boundary ~1–2 THz) is in its ground
state. Mode amplitudes follow ⟨u²⟩ ∝ (ħ/2ω)·coth(ħω/2k_BT) → ħ/2ω (zero-point
motion), while **classical MD gives equipartition k_BT/ω²** — wrong by orders
of magnitude for the high-ω modes at this temperature. A classical Langevin
trajectory at 5 K would produce far-too-sharp G(r) peaks and a bad closure.

**Decision:** primary sampler = **harmonic quantum sampling**: random
supercell displacements drawn from the M1 phonopy model with quantum
statistics at T = 5 K (phonopy's random-displacement machinery includes
zero-point occupation). MLIP-MD (ASE Langevin) is kept in `md_run.py` as an
explicitly *classical* cross-check and for future higher-T datasets, with its
5 K caveat printed loudly.

### D2 — lattice for closure: experimental, not MLIP-relaxed

MACE-MP-0 (small) overexpands a by +1.6 % (10.526 vs 10.356 Å). G(r) peak
*positions* scale with a, so closure must be computed **at the experimental
lattice** (fixed cell, a = 10.3563 Å; internal coordinates re-relaxed under
that constraint). The pure MLIP-relaxed structure remains the M1 theory-side
reference. Both provenances recorded in the output JSON.

### D3 — effective-FC engine: hiPhive, introduced in M2, reused in M3

`band_T.yaml` needs a displacement/force → force-constant fit. Proposal: use
**hiPhive** as the single fitting engine for both M2 (fit on our own
quantum-sampled snapshots + MLIP forces — a self-consistency loop with known
provenance) and M3 (same machinery, but RMC snapshots as the displacement
source — the experiment-constrained fit this project exists for). At 5 K the
M2 renormalization should be tiny (band_T ≈ band); *that agreement is itself a
validation* of the sampling + fitting machinery before M3 trusts it.

### D4 — S(Q)/G(r) computation: small native implementation

Partial pair histograms g_ij(r) → coherent-neutron-weighted total G(r) →
windowed Fourier transform to S(Q) on exactly the measured Q-grid. Native
(numpy) rather than an external PDF package: keeps deps light and every step
unit-testable against analytic cases, matching repo conventions. Neutron
b_coh from standard tables (Ga 7.288, Ta 6.91, Se 7.97 fm — confirm source).
Closure metric: Rw over the measured Q-window (same definition RMCProfile
uses, so numbers are comparable to the RMC fit's own residual).

*Correction (2026-09-29):* the GTS data are X-ray (the RMC `.dat` fits
`XRAY_RECIPROCAL_SPACE_DATA`, Qdamp 0.0389 Å⁻¹). `md_run.py` now takes
`--radiation {neutron,xray}` and `--qdamp`; the X-ray route transforms
each partial and applies RMCProfile's Q-dependent form-factor weights,
and by default compares both sides box-convolved at L = 2·rmax so the
histogram's truncation is not scored as misfit (`md_run.closure_fit`).

## Deliverables

```
md_run.py            # CLI: sample (quantum) | md (classical Langevin/NPT)
                     #      → snapshots → forces → G(r), S(Q), band_T.yaml
m2_out/
  gr_sim.dat         # r (Å), G(r) — simulated, neutron-weighted
  sq_sim.dat         # Q (Å⁻¹), S(Q) — on the measured grid
  closure.json       # Rw, grids, sampling provenance, lattice used
  band_T.yaml        # hiPhive effective FCs → phonopy band structure at T
```

Units: Å, eV, THz throughout (phonopy defaults); stated in every docstring.
All MLIP evaluations `default_dtype="float64"`.

## Validation / tests (same discipline as M1)

- Unit: g(r) histogram on an ideal fcc lattice (analytic peak positions);
  neutron weighting closure (weights sum to ⟨b⟩²-normalized 1); FT round-trip
  on a synthetic Gaussian; quantum ⟨u²⟩ of generated snapshots vs the analytic
  coth expression per mode; Rw metric on known inputs.
- Integration (slow): Cu/EMT end-to-end sample → G(r)/S(Q) → band_T; assert
  band_T ≈ band at low T and G(r) first peak at a/√2.
- Real-data acceptance: GaTa₄Se₈ closure Rw reported and compared against the
  RMC fit's own residual; not a pass/fail gate (MLIP systematics expected),
  but recorded in `closure.json`.

## Resolved in review (2026-07-19)

1. **Measured data**: `scale_ft_rmc.fq` is STOG-generated. Verified high-Q
   level ≈ 0 (mean +0.0004 over the last 500 points, oscillations decaying)
   and low-Q → −0.87, i.e. the convention is **F(Q) = S(Q) − 1**
   (Faber–Ziman). The RMC fit ran with **scale and offset free**, so the
   closure comparison fits (and reports) the same two parameters —
   shape-plus-scale Rw, mirroring the RMC treatment.
2. **G(r) and S(Q) are the same measurement** (r-space = FT of this F(Q)).
   Primary Rw is quoted in Q-space; G(r) shown for interpretation only.
3. **The 5 K phase is NOT cubic** (user-confirmed lower symmetry). This is
   *the physics target of the whole project*, and it reframes M2–M3:
   - The cubic F-4̄3m MLIP model (M1) is the **null model**, not the answer.
   - The cubic RMC box suppresses global strain, so the distortion lives in
     **internal displacement patterns** — consistent with the residual P1
     signal at symprec 10⁻³ surviving the 493-config average.
   - **Closure becomes a discriminator, not a validation**: the residual
     between quantum-sampled-cubic F(Q) and the measured F(Q) *contains the
     distortion signature*. A perfect fit is not expected and not the goal.
   - M3 verdicts then decompose it: which modes carry static (frozen)
     displacement excess vs quantum-dynamic amplitudes.

## Resolved in review, round 2 (2026-07-19)

7. **Low-T structure: P-4̄2₁m with a 1×1×2 doubled cell.** The order
   parameter therefore lives at **q = (0,0,½)** of the cubic conventional
   cell — a displacement pattern alternating in sign between adjacent unit
   cells along one cubic axis. Consequences:
   - The 8×8×8 RMC box is commensurate with the doubling (8 even): the
     ensemble *can* express the distortion, per axis, per domain.
   - A single-conventional-cell fold (52 sites) **cannot** express the
     primary order parameter (the staggered component cancels): M3 analysis
     must fold to 1×1×2 (104 sites) and/or use staggered projections.
   - The M1 F-4̄3m phonon model contains q = (0,0,½) on the Γ–X path (it lies
     inside the 2×2×2 phonopy supercell) and shows **no soft mode there**.
8. **This RMC fit used F(Q) + G(r), no Bragg profile** (Bragg intended for a
   future refit). The ensemble mean is therefore not Bragg-anchored; treat
   long-range averages accordingly, and prefer a Bragg-included refit before
   quantitative M3 claims about the *average* structure.
9. **Distortion driver: correlation/SOC (user's assessment).** A PBE-level
   foundation MLIP is *not expected* to host the distorted well. Design
   stance confirmed: MACE = symmetric null model; the distortion is detected
   from the RMC displacement statistics vs the null model, and **M4 DFT+U/SOC
   fine-tuning is required** (elevated from optional) for distorted-phase
   phonons.

## Null-model probe results (2026-07-19, pre-implementation)

Two quick probes run after review (analysis in scratch, not committed):

- **MACE hosts no distorted well** (probe B): the unsymmetrized 493-config
  ensemble-average (P1) relaxes straight back to F-4̄3m at both the
  experimental and free lattice, ΔE(P1−sym) = 0.000 meV. Confirms the
  correlation/SOC assessment — MACE-MP-0 is strictly the symmetric null
  model; the distortion must be detected from the data side.
- **The data prefer the distortion** (probe A): projecting each config onto
  the (0,0,½) staggered order parameter, the ensemble bulk shows no frozen
  pattern (‖Q‖ at noise level, no χ²-correlation, ensemble mean incoherent —
  domains/rarity). But the **3 independent chains with the largest staggered
  amplitude (0.06–0.09 Å; configs 105≡206, 23, 190, all x-axis) are exactly
  the ensemble's best sustained fits: χ² ≈ 314–374 vs median 705.** RMC only
  rarely reaches the coherent arrangement (entropic barrier), and when it
  does, the measured F(Q)+G(r) reward it by ~2× in χ².
- Ensemble bookkeeping: duplicate chains exist (105/206, 439/404, 459/208 —
  identical χ² trajectories, likely seed collisions); config 376's final
  χ² = 123 is a last-report artifact (plateau was ~700); configs
  280/311/313/316/318/326 are partial runs (0.08–0.5M moves) with truncated
  χ² files — `--skip-nonconverged` catches only 0-move configs, so consider
  a stricter move-count / χ²-sanity criterion.

Implications: (i) M3's mode-projected amplitude test has a concrete target —
quantify the staggered amplitude *distribution* against the quantum-dynamic
expectation; (ii) recommend an RMC refit **with the Bragg profile** (the
superlattice peaks constrain the order parameter directly) and/or chains
seeded from a P-4̄2₁m starting config; (iii) M2 proceeds per plan — closure
residuals of the cubic null model are now expected to carry the distortion.

### Revisions forced by Yang et al., PRR 4, 033123 (2022) — read 2026-07-19

The user's own paper on GaTa₄Se₈ (T* = 50 K order–disorder spin-orbital
transition; P-4̄2₁m with a = 10.3437, c = 20.6878 Å; QE/phonopy DFT PDOS;
SEQUOIA Ei = 60 meV) **corrects three probe conclusions above**:

1. **Probe A projected the weak channel.** The primary distortion modes are
   **X5 (amp 0.120 Å) and X3 (0.072 Å) at k = (0,1,0)** — the X point,
   *commensurate with the conventional cell*; the cell-doubling modes
   Δ1 (0.021 Å) and W4 (0.026 Å) are ~5× weaker (paper Table II). Our
   (0,0,½) staggered projection tested the ~0.02 Å Δ1-type channel — below
   the measured per-config noise floor, so the bulk null result was
   guaranteed. The X-point primary CAN live inside the single-cell fold:
   the residual P1 signal at symprec 10⁻³ is the candidate carrier. M3 must
   project X5/X3 (and W4/Δ1) using the paper's displacement patterns.
2. **Probe B's "MACE hosts no distorted well" was premature — now settled
   definitively (probe D, 2026-07-19).** The paper's plain-PBE (no SOC/U)
   Quantum-ESPRESSO relaxation *retains* the distorted P-4̄2₁m structure
   (Table I: ℓ = 3.129 vs s = 2.927 Å). MACE was therefore retested starting
   **from the refined experimental structure** (data/GTS_P-421m_LT.cif,
   104 atoms, bimodal Ta–Ta 2.94–3.07 Å, spread 0.135 Å): it collapses
   completely — all 48 bonds → one length (spread 0.000) at both fixed and
   free cell, ΔE(tet−cub) = 0.00 meV/f.u. The E(λ) scan along
   cubic→distortion is a clean single parabola: the experimental distortion
   (max |u| = 0.118 Å ≈ the X5 amplitude) costs **+293 meV/f.u.** on the
   MACE-MP-0-small surface where QE-PBE has a minimum. Conclusion: a
   *foundation-model fidelity* failure (training-coverage), not an
   XC-functional-level one — **M4 fine-tuning against the paper's QE-PBE
   setup is on the critical path** for distorted-phase dynamics and the
   13-meV analysis; the target is turning a +293 meV/f.u. hill into a
   shallow well. The cubic MACE null model remains valid for M3 amplitude
   verdicts (GDOS shape confirmed), caveat the 12–15 meV window.
3. **The 13-meV diagnostic needs reconciling.** Their DFT cubic PDOS
   notably *lacks* ~13 meV weight (its appearance is the tetragonal
   fingerprint: 2A₁+2A₂+2B₁+3B₂+4E modes); our MACE **cubic** GDOS shows a
   13.2 meV (Ga-flavoured) peak. Either MACE softening pulls the cubic
   15-meV band down, or MACE's cubic spectrum genuinely differs from QE —
   must be resolved before MACE serves the 13-meV analysis. Their SEQUOIA
   resolution polynomial (Appendix E) should be applied to all comparisons.

Independent confirmations the blind pipeline got right: measured G(r)
broader than the cubic quantum model with the strongest residual at
2.5–4.5 Å = the paper's "cubic fails for r < 5 Å" (two Ta–Ta distances
2.95/3.06 Å); overall GDOS shape matches their measured/DFT PDOS.

### Powder degeneracy note (review round 3, 2026-07-19)

The RMC fit is against **powder** data, so the three doubling arms 2×1×1 /
1×2×1 / 1×1×2 are exactly degenerate. Consequences adopted:

- **The ensemble-average F-4̄3m recovery says nothing about the distortion.**
  With arm and phase degeneracy, chains order along random arms with random
  phases, so the grand average restores cubic symmetry *even if every chain
  were fully ordered*. The average is a pipeline consistency check only; the
  residual P1 at symprec 10⁻³ likewise cannot contain the primary OP (it
  cancels) — only Γ/zone-center secondary components or noise.
- **Only per-config, arm-pooled statistics are meaningful** (probe A's
  max-over-axes was already the right frame; the proper null is the order
  statistic of 3 arms — the ~8σ tail survives this correction).
- **M3 verdicts aggregate the q-star**: report star-summed staggered
  amplitude vs quantum expectation, never per-arm; powder cannot assign arms.
- **Within-box domains along different arms are allowed**: add a local /
  windowed staggered field (or diffuse S(q) near the star) in M3 to measure
  coherence length — the global single-arm projection undercounts mixed-arm
  order.
- **Verified: the shared RMC starting config is perfectly ideal**
  (GTS_5K_0.rmc6f: σ_site = 0.0000 Å) — the all-x coincidence of the three
  ordered chains has no structural seed; P = 1/9 by chance, and the arm
  label carries no physics.

## Still open for review

4. Neutron b_coh values/source to standardize on
   (proposal: Sears 1992 tables — Ga 7.288, Ta 6.91, Se 7.97 fm).
5. Is the classical-MD leg worth running at 5 K at all beyond the caveat demo,
   or reserve MD for a future higher-T dataset?
6. hiPhive cutoffs/order for the effective fit (start: 2nd order only,
   cutoff ~6 Å, 2×2×2 supercell of the conventional cell)?
