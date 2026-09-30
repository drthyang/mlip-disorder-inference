# RMC control experiment — design and status

*Started 2026-09-29. Code: `rmc_control.py` (synth / stage / compare),
`harmonic_pdf.py`, `md_run.xray_*`, `mode_project.py` (ensemble driver).*

## Question

The M3 verdicts (`verdicts.json` v0.1) compare windowed irrep amplitudes of
the 490-config RMC ensemble with an expectation built from a quantum
baseline plus **one scalar noise fraction** (f_noise = 0.82) applied flat
across mode space. The nulls available so far (`random_signs`,
`shuffle_cells`) operate on the configurations as given, so they cannot
separate *"X₅ is static"* from *"X₅ is where RMCProfile's single-atom move
statistics pile up"*.

The control answers it directly: refine **synthetic data from a model with
zero static disorder** exactly as the measured data were refined, and push
the result through the same projector. Per irrep m and window scale w:

    r_ctrl(m, w) = ⟨A²⟩_measured / ⟨A²⟩_control

is the amplitude excess over everything RMC + quantum motion produce on their
own. r_ctrl ≈ 1 means the measured amplitude is explained without static
order; a bootstrap interval above 1 is an excess the null cannot make.

## What is held fixed (byte-identical to the original run)

- RMCProfile binary (NERSC `RMCProfile_package`, 2024-09 build) and job shape:
  1 Perlmutter CPU node, 8 OpenMP threads, 24 h (≈ 2.2 M generated moves).
- `GTS_5K.dat` — fits X-ray F(Q) and its FT G(r); `CONVOLVE`, fitted scale
  and offset, `RESOLUTION_CORRECTION :: 0.0389`, curvature restraints,
  weight optimization. Copied byte for byte (it is CRLF).
- The starting box `GTS_5K.rmc6f` — the ideal 8×8×8 F-4̄3m box every one of
  the 500 original chains started from — and `optimization.dat`.
- The projector and frame handling (`python mode_project.py`): measured and
  control boxes go through identical code.

Only `scale_ft_rmc.fq` differs.

## The synthetic data (`rmc_control.py synth`)

**Forward model = RMCProfile's own, pinned from its output files**, not from
the manual (`tests/test_xray_forward.py`):

| convention | how it was fixed | agreement |
| --- | --- | --- |
| X-ray f₀: Waasmaier–Kirfel, ⟨f⟩² Faber–Ziman weights | total F(Q) from RMCProfile's partials | 4e-8 |
| partials: rectangle-rule sine FT on r_k = k·dr at the box ρ₀, **Gaussian** envelope exp(−(0.0389 r)²/2) | partial F(Q) from RMCProfile's g(r) | 5e-8 |
| `CONVOLVE ::` acts on the *data*: G(r) truncated at L/2 | RMCProfile's "F(Q)_Expt" column from the raw file | Rw 0.001 |

Consequence: the synthetic file must be the **infinite-crystal** F(Q) that a
diffractometer would record, written unconvolved. RMCProfile convolves it
itself, just as it convolved the measured data.

**Null model.** MACE-MP-0 small (float64), relaxed at the experimental
lattice (a = 10.3563 Å), finite-displacement FCs on a 4³ conventional
supercell (3,328 atoms, 7 displacements). Quantum occupations (ħ/ω)(n_B + ½)
at 5 K. Mean positions = the RMC starting positions (the experimental
average; the MACE minimum lies within 0.016 Å, so the choice is immaterial).
u_rms per component 0.034–0.044 Å (M2: 0.040).

**g(r) without a box** (`harmonic_pdf.py`). Lattice sum of per-pair radial
Gaussians to r = 120 Å (17.6 M pairs; the Qdamp envelope is 2e-5 there). The
widths carry the displacement correlations, σ² = d̂ᵀ(U_j + U_j' − Σ − Σᵀ)d̂,
with Σ_jj'(R) from phonon eigenvectors on a 16³ q-grid (correlations kept to
83 Å). Validated on EMT Cu: Σ equals phonopy's supercell ⟨uuᵀ⟩ to 1e-18, and
shell widths match sampled snapshots to 0.2 %. Converged: an 8³ grid with
r ≤ 60 Å changes the box-convolved F(Q) — what RMC fits — by 7e-4 relative.

**Noise.** White noise at the measured data's grid-scale level, from a
rolling MAD of **6th** differences (σ = 2e-4 to 6e-3). A 2nd-difference
estimate reads the resolution-limited Bragg-peak curvature as noise; at low
Q it overstated the noise ~50× (the measured file is smoother at the grid
scale than even the noiseless synthetic pattern — STOG filtering). This is a
lower bound on correlated noise; see Limitations.

**Distance from the real data** (both box-convolved, scale + offset fitted):
Rw(Q) = 0.248, Rw(r) = 0.40, scale 0.72. The measured peaks are broader at
high Q and the 2.95/3.06 Å Ta–Ta split is absent from the null, as expected
if the static distortion is real. (M2 recorded Rw(Q) = 0.74 for the same null
model — neutron-weighted and box-truncated; see ROADMAP.)

## The run (`rmc_control.py stage`)

`results/rmc_control/null_run/` (git-ignored, 96 MB): 64 chains
`GTS_5K_null_<i>.{dat,rmc6f}`, `submit_<i>.sh` (the original `submit.sh`
with only job name, output, working directory → `$SLURM_SUBMIT_DIR` and stem
changed), `submit_all.sh`, `README.md`, and `stage_manifest.json` with
hashes plus the full synth provenance.

**Why 64 chains.** At the verdict scale (w = 4), the per-config ⟨A²⟩ of the
measured ensemble scatters with CV 0.42–0.45 for X₅/W₄ (0.18–0.35 for the
others). If the control scatters alike, 64 chains give a 5–6 % standard
error on ⟨A²⟩_ctrl; 32 would give 8 %. The measured side is 490 configs.

**Cost.** 64 node-days in the original shape (1 node per 8-thread chain).
Chains could share a node, but that changes per-chain speed and hence the
move count reached. Compare at matched moves if packed.

## Readout (`rmc_control.py compare`)

    python mode_project.py <returned_dir> --exclude AVERAGE -o control.npz
    python rmc_control.py compare --measured results/m3_projections_aligned.npz \
        --control control.npz

Per irrep and scale: r_ctrl with a two-sided bootstrap 95 % interval
(configs of both ensembles resampled). The interval is classed *excess*
(> 1), *explained* (contains 1) or *deficit* (< 1), with each ensemble's
ratio to its own random-sign null alongside. It also yields a per-mode noise
fraction, f_noise(m) = (A²_ctrl − A²_qh)/A²_null, to replace the flat scalar
in `verdicts.py`.

Decision rule for the M3 claims: the X₅ / W₄ "mixed" verdicts survive only
if their w = 4 intervals are *excess*. If they are *explained*, the local
order the verdicts report is an RMC artifact at that scale.

## Limitations and open items

1. **Representability.** The control data come from a model a harmonic box
   can represent almost exactly; the measured data do not (the real chains
   plateau at Rw ≈ 4 %). RMC may therefore fit the control more tightly and
   manufacture less — so compare final Rw alongside r_ctrl. A
   representability-matched arm (null data from a deliberately different
   model, e.g. broadened widths) is a candidate second step.
2. **Noise.** The added noise is grid-scale white noise at the measured
   level; the true (filtered, correlated) noise is unknown. The misfit
   floor in the real run was systematic, so this should matter little.
   `--noise-scale` exists for a sensitivity arm.
3. **Positive control (recommended next arm).** Synthetic data from the
   published P-4̄2₁m distortion plus the same quantum motion, to measure
   RMC's *recovery* of a known X₅/X₃/W₄ amplitude. It needs static offsets
   in `harmonic_pdf.harmonic_partials` (a 1×1×2 mean cell); not yet built.
4. **One MLIP.** The null's dynamics are MACE-MP-0 small at the cubic
   structure. MACE lacks the distorted-phase well (probe D), which is
   correct for a null model but fixes its soft-mode amplitudes.
5. **`NUMBER_DENSITY :: 0.057329`** in the original `.dat` does not match the
   box (0.046815 Å⁻³). It is kept unchanged for fidelity, but check what
   RMCProfile uses it for.
6. **`verdicts.json` v0.1 is not reproducible** from committed code (see
   CHANGELOG). Regenerate it through `mode_project.py`'s driver, recomputing
   the quantum baseline in the same frame, before comparing with the control.

## Status

- [x] forward model pinned; projection driver; harmonic g(r); synth; stage
- [ ] run 64 chains on Perlmutter (user)
- [ ] `compare`; per-mode f_noise; regenerate `verdicts.json` with the control
- [ ] positive-control arm
