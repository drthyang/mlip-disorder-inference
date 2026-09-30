# Changelog

## [Unreleased]
### Added — RMC control experiment (started 2026-09-29)
- **RMCProfile's X-ray forward model, pinned exactly** (`md_run.py`:
  `XRAY_WK`, `xray_f0`, `xray_weights`, `partial_fq`, `xray_fq`,
  `rmc_box_convolve`). Every convention was fixed by reproducing
  RMCProfile's own output files for GTS config 1, not taken from the manual:
  Waasmaier–Kirfel f0 with ⟨f⟩² Faber–Ziman normalization (total F(Q) from
  its partials to 4e-8); partials as a rectangle-rule sine transform of
  g_ab(r) on r_k = k·dr at the box density with `RESOLUTION_CORRECTION`
  applied as a **Gaussian** envelope exp(−(q·r)²/2) (to 5e-8 — the manual's
  exp(−r·q) is not what the binary does); and `CONVOLVE ::` acting on the
  *data* as a G(r) truncation at half the box edge (its "F(Q)_Expt" column
  to Rw 0.001). Consequence for the control experiment: synthetic data must
  be the infinite-crystal F(Q), written unconvolved.
  `tests/test_xray_forward.py` (13 tests; 3 need the private data).

- **Frame-correct ensemble projection driver** (`mode_project.py`:
  `projection_setup`, `read_rmc6f_box`, `to_aligned_frame`,
  `config_amplitudes`, `ensemble_amplitudes`, and a CLI writing an npz with
  per-window measured and random-sign-null amplitudes at w = 2/4/8). The
  published patterns are anchored in the refined structure's setting, which
  differs from the RMC/CIF frame by a parent rotation (an axis permutation
  with signs) and a half-cell origin shift; every box is mapped into that
  frame (X' = R·X + t, ijk' = R·ijk + n_site) before projecting. Projecting
  RMC-frame coordinates directly scrambles the channels — a pure injected
  W₄ at 0.10 Å reads 0.02, a pure X₅ reads 0.42 with 0.25 in W₄.
  `tests/test_projection_frame.py` (7 known-answer tests incl. an rmc6f
  round trip and a regression guard for the pitfall).
- **`harmonic_pdf.py`** — partial g_ab(r) of the infinite quantum-harmonic
  crystal as a lattice sum of per-pair radial Gaussians whose widths carry
  the displacement–displacement correlations, σ² = d̂ᵀ(U_j + U_j' − Σ − Σᵀ)d̂,
  with Σ_jj'(R) summed by FFT over the q-grid commensurate with an M³ box of
  the unit cell (phonopy eigenvectors, (ħ/ω)(n_B + ½) quantum weights,
  pairs outside the minimum-image cube uncorrelated). Validated on EMT fcc
  Cu (`tests/test_harmonic_pdf.py`, 5 tests): every Σ block equals
  phonopy's `RandomDisplacements.run_correlation_matrix` to 1e-12 (actual
  7e-19), and coordination numbers, shell positions and correlated shell
  widths match sampled quantum snapshots (first-shell σ 0.0868 vs 0.0867 Å,
  vs 0.106 Å uncorrelated). `md_run.harmonic_model` gains
  `primitive_matrix` / `compact_fc` (defaults unchanged); note that in
  phonopy ≥ 4 `primitive_matrix=None` means "auto" — the identity is `"P"`.
- **`rmc_control.py`** — the control experiment's CLI. `synth`: MACE
  quantum-harmonic null model → `harmonic_pdf` g(r) → X-ray F(Q) on the
  measured grid in RMCProfile's convention, written unconvolved, plus
  white noise at the measured grid-scale level (rolling MAD of 6th
  differences — a 2nd difference reads resolution-limited Bragg curvature
  as noise and overstated it ~50× at low Q), a manifest and a diagnostic
  figure. `stage`: a NERSC run directory mirroring the original ensemble
  run byte for byte except the data file (per-chain `submit_<i>.sh`
  rendered from the original `submit.sh`, anchors checked). `compare`:
  r_ctrl = ⟨A²⟩_meas/⟨A²⟩_ctrl per irrep and scale with bootstrap
  intervals. `tests/test_rmc_control.py` (8 tests incl. an EMT-Cu
  synth→stage run). Design and decision rule:
  `docs/control-experiment-plan.md`.
- **GTS control, ready to run.** Synthetic null data generated (46 s;
  converged — an 8³ grid / 60 Å run differs by 7e-4 in what RMC fits) and
  64 chains staged in `results/rmc_control/null_run/` (git-ignored).
  With X-ray weighting and no box truncation, the null model sits at
  Rw(Q) = 0.248 / Rw(r) = 0.40 from the measured data (M2 recorded 0.74 /
  0.58 neutron-weighted).
- **Static offsets in `harmonic_pdf.harmonic_partials`** (`static=`, a
  (p1, p2, p3, n, 3) Å field periodic over p unit cells): the mean
  separations carry a static superstructure while widths and correlations
  stay the parent's quantum ones — the construction the positive-control
  arm needs (MACE cannot host the distorted phase, probe D). Tests: a
  uniform offset changes nothing (1e-9); a 1×1×2 Cu pattern matches
  phonopy snapshots with the same offsets added (shell counts, positions,
  widths to 2 %).
- **The published distortion as a static field** (`mode_project.py`:
  `published_field` — SM Table IV expanded over the slab orbits;
  `remove_uniform_part` — drops the k = 0 part, i.e. the Γ offset between
  the paper's AMPLIMODES parent and ours; `slab_field_to_rmc_static` —
  pattern frame → RMC-frame (p1, p2, p3, 52, 3) offsets for
  `harmonic_pdf`; `static_box`). `projection_setup` now also returns the
  reference, label mapping and orbits. Tiled into an RMC-frame 8³ box the
  field reads X5 0.1183 / X3 0.0684 / W4 0.0278 / Δ 0.0205 Å (published
  0.1196 / 0.0719 / 0.026 / 0.0212) at every window scale, identical to
  the pattern-frame route. 3 new tests in `tests/test_projection_frame.py`.
- **Positive-control arms, ready to run.** `rmc_control.py synth --inject
  published [--inject-scale s] [--keep-gamma]` adds the published P-4̄2₁m
  distortion (k = 0 part removed) as static offsets on the null model's
  quantum motion and records the known answer (the field's own windowed
  projection) in the manifest; a frame guard refuses a start cell that
  disagrees with the parent CIF. `stage --arm` refuses null data under a
  positive arm and vice versa, defaults the stem to `GTS_5K_<arm>`, and
  writes an arm-specific README. `compare --positive NPZ SYNTH_DIR`
  (repeatable) adds the recovery ρ = (⟨A²⟩_pos − ⟨A²⟩_null)/A²_inj and the
  calibrated static amplitude √((⟨A²⟩_meas − ⟨A²⟩_null)/ρ) with joint
  bootstrap intervals. GTS ×1 and ×2 arms synthesized (77 s each; 35 M
  pairs) and staged, 32 chains each. 4 new tests. Side result: against the
  measured F(Q), Rw(Q) falls from 0.248 (null) to 0.213 (×1) and 0.149
  (×2) — the data prefer the distortion, and more of it (partly degenerate
  with too-narrow MACE widths). Design, power table and caveats in
  `docs/control-experiment-plan.md`.

### Found
- **`verdicts.json` v0.1 is not reproducible from committed code.** The
  scratch driver that produced its windowed amplitudes was never
  committed. The committed driver gives, for the same 490 configs at w = 4,
  X₅ 0.3225 / W₄ 0.1807 / Γ₃ 0.2035 / Γ₁ 0.1429 / X₃ 0.0760 / Δ 0.0532 Å vs
  the published 0.3012 / 0.1684 / 0.1893 / 0.1357 / 0.0757 / 0.0543; no
  reconstruction tried (RMC vs aligned frame, RMS vs mean over windows,
  ideal vs ensemble-mean reference) matches exactly. The random-sign nulls
  agree to <1 %. The verdicts should be regenerated through the committed
  driver; the quantum baseline must then be recomputed in the same frame.
- **The GTS total-scattering data are X-ray, but the M2 closure is
  neutron-weighted.** The RMC `.dat` fits `XRAY_RECIPROCAL_SPACE_DATA`;
  `md_run.py` weights its simulated G(r)/F(Q) with neutron b_coh, under
  which Ta has ~6× less relative contrast. The recorded M2 closure
  (Rw(Q) = 0.74, scale 0.63) is therefore partly a weighting mismatch.
  Not yet rerun — see ROADMAP.

### Changed
- **Renamed `rmc-mlip-phonons` → `mlip-quantum-thermal` (2026-07-26).** The
  name describes the physics the repo generates — an MLIP quantum-thermal
  ensemble used as the null model — rather than the statistical method layered
  on top, which is what the two intermediate names did and why they churned.
  GitHub redirects the old URLs. Milestone-3 package target is now
  `mlip_quantum_thermal/`.
- **Scope pivot: RMC demoted from inference engine to screening tool.**
  Total scattering is the energy integral of S(Q,E), so G(r)/S(Q) cannot by
  itself distinguish a frozen distortion (elastic) from a soft mode (peak at
  ±ħω) from order–disorder hopping (quasielastic) — the ambiguity milestone 3
  attacks indirectly with an RMC ensemble, a quantum null and a fitted noise
  fraction. Intended evidence chain is now forward closure against measured
  data, with RMC used to suggest refinement coordinates rather than to carry a
  verdict. Rationale in README; gating question (do the SEQUOIA/ARCS
  reductions retain the elastic line, or only GDOS?) still open.
- **Detached from `rmc-phonon-dynamics`.** The "no web UI in this repo" and
  "frozen interchange contract" boundaries in CLAUDE.md are retired — there is
  no external counterparty. `band.yaml` stays phonopy-standard by choice
  (phonopy / Euphonic / OVITO / phononwebsite interop), not by contract.

### Added
- **`viewer/`** — self-contained browser front end, vendored from
  `rmc-phonon-dynamics` `web/src` (MIT, same author). Carried over: the
  Three.js `CrystalViewer` and `BrillouinZoneViewer`, the band-structure plot,
  mode inspector, simulated-INS panel + S(Q,E) worker, the phonopy band-yaml
  reader, and 7 general crystallography modules (~4.3 kloc). Deliberately left
  behind: the entire covariance route (`math/symmetrize.js`, `symmetry.js`,
  `cells.js`, `compute/engine.js`, `pipeline.js`, `Sk_kernel.wgsl`) and the
  RMC dataset shell (`pages/`, `io/readers.js`, `io/sqgr.js`). `src/App.jsx`
  is new; `?load=<url>` deep-links a band yaml. `npm run validate` (6 suites)
  and `npm run build` both pass.
- **EMT end-to-end re-verified against installed dependencies** (ase 3.29,
  phonopy 4.4, spglib 2.7): 12-config synthetic Cu fcc ensemble →
  `milestone1_bands.py --calc emt` recovers Fm-3m (225), natom = 1 primitive,
  min band frequency 3.1e-6 THz (acoustic branches vanish at Γ). The resulting
  `band.yaml` loads and animates in `viewer/`. Full suite: 73 passed,
  17 skipped (optional deps / private data).

### Fixed
- **Band paths were computed at seekpath's hardcoded `symprec=1e-5`** and were
  therefore wrong for every relaxed cell. `Phonopy.auto_band_structure()`
  takes no `symprec`, so ~1e-5 Å of residual relaxation noise made seekpath
  read fcc Cu as **aP2 / P1**, giving the triclinic path Γ-X | Y-Γ | Γ-Z …
  instead of cF2's Γ-X-U | K-Γ-L-W-X. Consequences: meaningless
  high-symmetry labels, a path that never visits the real X/L/W/K points, and
  — because the path depended on numerical noise — no guarantee that band
  yamls from *different* scripts shared a q-path, which silently invalidates
  overlaying them or differencing their frequencies (`md_run.py` and
  `hiphive_fit.py` both report an element-wise `max|Δω|` against the harmonic
  bands).

  Fixed with `milestone1_bands.symmetrize_lattice()`: the metric tensor
  g = A·Aᵀ obeys Rᵀ g R = g for every fractional rotation of the true space
  group, so the noise is averaged out over the group and the cell rebuilt as
  A' = S·A with S = g_sym^{1/2}·g^{-1/2} (exact: S g Sᵀ = g_sym; S → I as
  g → g_sym, so the Cartesian frame is preserved). **Lattice only** —
  fractional coordinates, atom order and atom count are untouched, which the
  hiPhive FCP evaluation and the `ideal`-supercell indexing depend on.
  Applied at all four `Phonopy` construction sites (`milestone1_bands.py`,
  `md_run.py` ×2, `hiphive_fit.py`); the shift is reported in `summary.json`
  under `phonopy.lattice_symmetrization`.

  Verified on the Cu fixture: 192 operations found, max lattice shift
  3.1e-5 Å, path restored to Γ-X-U | K-Γ-L-W-X, reciprocal lattice now
  exactly symmetric, min frequency −1.8e-6 THz (still stable). New unit test
  `tests/test_symmetrize_lattice.py` (6 cases) covers the fixed point, noise
  removal, coordinate/order/orientation preservation, the no-symmetry
  passthrough, and the seekpath regression itself.

### Verified
- **Cross-repo units audit vs phonopy defaults (2026-07-21)**: this repo is
  clean — frequencies are THz everywhere via phonopy's default factor
  √(eV/amu)/Å/2π = 15.63330 (never overridden), THz↔meV uses 4.135667696
  matching phonopy's constant, quantum sampling and thermal displacements go
  through phonopy's own machinery (Å²), hiPhive force constants are eV/Ų,
  and the released band yamls carry THz as documented (26 items verified by
  computation, 0 defects). The companion viewer's unit handling was fixed in
  `rmc-phonon-dynamics` commit 138837d (band.yaml THz→meV on load,
  mass-weighted eigenvector convention, q·(n+τ) Bloch phase, `labels:`
  parsing, THz exports with a self-describing `frequency_unit:` key).

### Added
- **`releases/gts_5k/`** — the **local** viewer bundle (git-ignored; assembled
  on disk, never pushed — see the Removed note below): all three band files
  with standard-path eigenvectors (`band.yaml` harmonic, `band_T.yaml`
  quantum-effective, `band_rmc.yaml` experiment-constrained; max
  band_T-vs-band_rmc difference 1.03 THz), both CIF references, the mode
  files (`modes_irrep.yaml` compact + `modes_irrep_112.yaml` Γ-folded),
  seven animated `.xyz` trajectories, `verdicts.json`, and the provenance
  JSONs, with a README describing each file and its caveats. `md_run.py`
  gains `--exclude` (band_T regenerated instantaneous-only, with
  eigenvectors).

### Removed
- **`releases/` is no longer committed and has been purged from git history**
  (2026-07-21). The bundle was briefly tracked on 2026-07-20; the data is
  private, so the commit was rewritten out and force-pushed, and `releases/`
  joined `data/`/`results/` in `.gitignore`. The bundle still lives locally
  for the viewer — regenerate any file with the scripts at the repo root.
- **Milestone 3 complete.** `hiphive_fit.py` (RMC snapshots + MLIP forces →
  `band_rmc.yaml` + `fit_report.json`; general geometry mapping, cached
  forces, `--exclude` filter) and `verdicts.py` (tested assembly of
  `verdicts.json` + the E(Q) pattern→branch mapping). GTS results: the
  experiment-constrained effective bands soften by −0.10 THz on average vs
  the harmonic null with no X-point instability (shallow incommensurate
  dips at the fit-noise scale only); the E(Q) mapping shows the X₅/W₄
  frozen patterns spread over many cubic branches (max overlap ~0.10)
  while Γ₁ is cleanly carried (0.88) — the 13-meV fingerprint requires the
  distorted-phase model (M4). ω fields written into verdicts.json.
- **First `verdicts.json` (schema v0.1)** — the milestone-3 sidecar for the
  viewer, from the full measurement chain: star-pooled joint projections of
  490 configs at window scales 2/4/8 cells, random-sign noise null,
  quantum-harmonic baseline sampled on the full 26,624-atom box through
  the identical projector (u_rms 0.0693 Å, matching M2 independently), and
  the three-component expectation r = A²_meas/[A²_qh + f_noise·A²_null]
  (f_noise = 0.82 measured). Verdicts at w=4: X₅/W₄/Γ₃/Γ₁ **mixed**
  (local static+dynamic order at 20–40 Å, domain-cancelled globally),
  X₃/Δ **dynamic**. Windowed machinery (`window_parity_sums`,
  `project_all_windows`) + nulls with tested selectivity (cell-shuffle
  preserves X-point channels exactly — diagnostic, not a noise floor).
- `reference/gts_mode_patterns.json` + `reference/ingest_sm_tables.py`:
  the PRR 4, 033123 supplemental Tables II–X (refined structures, total
  10 K displacement, and the six irrep patterns Γ₁/Γ₃/Δ/X₃/X₅/W₄) extracted
  from the SM PDF (git-ignored) into a committed reference. Verification
  suite (4 tests): row counts/multiplicities, |u| consistency, Table-IV
  decomposition onto the six patterns (all sites at rounding level), and
  implied amplitudes reproducing the published values to ≤3 % under the
  parent-primitive normalization (÷√8). **Corrected a probable sign typo in
  printed SM Table VIII (X₃, Se3, δz/c: −0.0053 → +0.0053)** — documented
  in the JSON and flagged for author verification. Suite: 70 passed.
- `docs/verdicts-schema.md`: milestone-3 design draft (awaiting review) —
  the `verdicts.json` sidecar contract (core fields per CLAUDE.md + optional
  extensions) and the mode-projection method: model-free projections onto
  the paper's supplemental Tables IV–X patterns, star pooling, per-config
  distributions, windowed/local projections, the stiff-mode RMC-noise
  ruler, ω-source hierarchy, SOC-tiered well/barrier fields, and a
  synthetic validation suite built on the M2 quantum sampler.
- `docs/idea-dynamic-refinement.md`: concept note for the long-range
  inversion ("model-space RMC" / "dynamic EPSR") — Metropolis/gradient
  refinement of a model whose MLIP-generated quantum ensemble reproduces
  total scattering + Bragg + INS jointly; two flavors (structure vs
  potential refinement), literature map (EPSR, DiffTRe, TDS inversion,
  RMC-phonons, hybrid RMC-MLIP, uMLIP-INS), cost/identifiability analysis,
  and the GTS order-parameter demonstration path.
- `md_run.py` (milestone 2): `sample` mode draws random supercell snapshots
  from the phonopy model with **quantum statistics** (zero-point included;
  phonopy's finite-temperature random displacements — the force-constant
  cache is saved/restored around the dataset swap, phonopy ≥4 invalidates
  it). `md` mode is a classical Langevin NVT cross-check that warns loudly
  below 100 K. Native closure machinery: species-partial g_ij(r) (chunked
  minimum-image histograms), Faber–Ziman neutron weighting (Sears 1992
  b_coh), sine transforms G(r) ↔ F(Q)=S(Q)−1 on the measured STOG grid, and
  a scale+offset Rw fit mirroring the RMC convention. Optional hiPhive
  2nd-order effective-FC fit → `band_T.yaml` sidecar (band.yaml untouched).
  Outputs: `closure.json`, `gr_sim.dat`, `sq_sim.dat`, `relaxed_expt.cif`,
  `band_T.yaml`.
- Tests: 10 analytic unit tests (fcc first-shell + coordination, g(r)
  normalisation identity, FT round-trip, Bragg-position check, weight
  closure, scale/offset fit, .fq parser) and 6 slow integration tests on
  Cu/EMT — including the D1 physics guard: sampled MSD matches phonopy's
  quantum ⟨u²⟩ at 5 K and 300 K, and is zero-point-dominated at 5 K.
  Suite: 66 passed.
- `requirements.txt`: + `hiphive>=1.5`.
- First M2 acceptance run on the GaTa₄Se₈ 5 K dataset (results git-ignored):
  quantum u_rms = 0.040 Å/component vs the RMC ensemble's 0.095 — the
  displacement-variance gap M3 will decompose (with the caveat that RMC
  single-atom moves decorrelate neighbours, inflating site variance while
  still fitting pair widths). Baseline closure Rw(Q) = 0.74, Rw(r) = 0.58;
  measured G(r) peaks systematically broader than the quantum-harmonic null
  model. band_T tracks band to 0.18 THz at 5 K.
- `docs/milestone2-plan.md`: milestone-2 design draft (awaiting review).
  Decisions put forward: harmonic **quantum** sampling instead of classical
  Langevin MD at 5 K (zero-point-dominated regime), closure computed at the
  experimental lattice (MACE +1.6 % would shift G(r) peaks), hiPhive as the
  shared M2/M3 effective-FC engine, native unit-tested G(r)/S(Q) with Rw
  against the measured `scale_ft_rmc.fq` grid.
- `--skip-nonconverged`: drop input configs whose rmc6f header reports
  `Number of moves generated: 0` (RMC runs that never started). Cheap
  header-only scan (`read_moves_generated` / `drop_nonconverged`); files
  without the header line are kept, so synthetic ensembles pass through.
  Drop count recorded in `summary.json["sampling"]`. 6 unit tests.
- First real-dataset runs (GaTa₄Se₈ 5 K RMC ensemble, 26,624-atom configs):
  single converged config and all 493 converged configs both produce a
  dynamically stable F-4̄3m band structure at the same MACE-MP-0 minimum
  (max band Δω 0.0009 THz); the ensemble average recovers F-4̄3m at 10×
  tighter symprec than a single config. Not committed (data/ and results/
  are git-ignored); see ROADMAP for caveats (lattice +1.6 %, no NAC yet).

## [0.2.0] — 2026-07-19

Milestone-1 hardening. The pipeline now runs end-to-end against the installed
ase / spglib / phonopy / MACE stack, with a full test suite and two physics
fixes that the first real runs surfaced.

### Added
- `.gitignore` covering ensembles, results, `m1_out*/`, venv, and model weights.
- `tests/` pytest suite for `milestone1_bands`: rmc6f parser fixtures
  (with/without site-id columns, bracketed labels, `Cell (Ang/deg)` vs
  `Lattice vectors` headers, ionic-label element cleaning, missing-section
  errors), circular-mean wrap-around + supercell folding, `--ref` CIF site
  assignment, `auto_dim` heuristic, and mixed-occupancy majority reduction.
  `tests/conftest.py` makes the standalone script importable; `pytest.ini`.
- `tests/fixtures/make_synthetic_ensemble.py`: deterministic generator of N
  noisy fcc-Cu `.rmc6f` configs (known answer: 3 acoustic branches, ω → 0 at
  Γ). Runnable standalone or imported. Tests verify parseability, that folding
  recovers the ideal 4-site fcc basis, seed-determinism, and the noise-free
  exact limit.
- `tests/test_emt_end_to_end.py` (marked `slow`): EMT full-pipeline integration
  test — asserts all three outputs are written, `relaxed.cif` is cubic Cu,
  exactly 3 branches vanish at Γ, and no imaginary modes. First run of the
  ase/spglib/phonopy leg against installed deps (phonopy 4.4, ase 3.29,
  spglib 2.7).
- `tests/test_mace_end_to_end.py` (marked `slow` + `mace`): real-dependency
  integration test running MACE-MP-0 (small, float64) on the fcc-Cu fixture —
  no imaginary modes (min > −0.05 THz), 3 branches vanishing at Γ, float64.
  Auto-skips when mace-torch is absent (first run downloads/caches the ~31 MB
  MACE-MP-0 weights).
- Ensemble subsampling for large (500+ config) runs: `--stride` (ordered
  decimation), `--max-configs` (seeded random cap), and `--seed`, via the pure
  `select_configs` helper. Deterministic; recorded in `summary.json["sampling"]`.
- `pytest.ini`: `slow` / `mace` markers and filters for benign spglib / phonopy
  / torch deprecation warnings.

### Changed
- Default `--displacement` raised from 0.01 to **0.03 Å**. phonopy's 0.01 Å is
  DFT-tuned; at that distance a universal MLIP's forces sit near the model
  noise floor. In the first real MACE-MP-0 (small) run this produced a spurious
  ~−0.33 THz mode just off Γ for dynamically-stable fcc Cu; 0.03 Å removes it
  (verified: ASR / `symmetrize_force_constants` does not help, displacement does).

### Fixed
- **Phonopy used the wrong symmetry tolerance.** `phonopy_bands` now threads
  `--symprec` into `Phonopy(...)`; phonopy's default (1e-5) is tighter than the
  ~fmax residual asymmetry of a relaxed cell, so `primitive_matrix="auto"` read
  fcc as P1 and produced folded bands (12 modes for Cu instead of 3). With the
  fix the fcc primitive is found: 3 acoustic branches, ω → 0 at Γ.
- Migrated off phonopy's deprecated `get_band_structure_dict()` to the
  `band_structure` property (phonopy ≥4).

## [0.1.0] — 2026-07-19
### Added
- Repo scaffold: README, ROADMAP, CLAUDE.md, requirements.
- `milestone1_bands.py`: rmc6f ensemble → folded/circular-averaged unit cell →
  spglib symmetrization → MLIP relaxation (MACE-MP-0, float64) → phonopy
  finite displacements → `band.yaml` + `relaxed.cif` + `summary.json`.
- Parser and circular averaging verified on synthetic wrap-around data.
