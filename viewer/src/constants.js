// web/src/constants.js
//
// Physical constants and element tables for the RMC phonon pipeline.
// Mirrors src_gpu/constants.py and viz/sqeworker.js so the browser results
// match the legacy Python/viewer code.

// ── Atomic masses [amu] — phonopy's default mass table ──────────────────────
// Copied from src_gpu/constants.py ATOMIC_MASS. These are 2011-era IUPAC
// values (e.g. Se 78.96, Mo 95.96), NOT the 2021 revision — kept deliberately
// identical to phonopy's structure/atomic_data.py so masses agree with the
// phonopy-produced band.yaml files this pipeline emits. 2011↔2021 differences
// shift frequencies by ≤7e-5 relative — negligible.
export const ATOMIC_MASS = {
  "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007,
  "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
  "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Sc": 44.956,
  "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
  "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.96, "Br": 79.904,
  "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.96,
  "Tc": 98.0, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
  "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33,
  "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Pm": 145.0, "Sm": 150.36, "Eu": 151.96,
  "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
  "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22,
  "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0,
  "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04, "Pa": 231.04,
  "U": 238.03, "Np": 237.0, "Pu": 244.0, "Am": 243.0, "Cm": 247.0
};

// ── Derived energy conversion ───────────────────────────────────────────────
// E [meV] = ENERGY_CONV * sqrt(T [K] / lambda [amu*A^2]) — classical equipartition.
// MUST match src_gpu/constants.py exactly:
//   ENERGY_CONV = HBAR_meVs * sqrt(KB_J / (AMU_TO_KG * ANG2_TO_M2)) ~= 0.600181852836787
// (The previous value here, ~210438, was wrong by a factor of ~3.5e5.)
const HBAR_meVs = 6.582119569e-13;    // reduced Planck constant [meV*s]
const KB_J = 1.380649e-23;            // Boltzmann constant [J/K]
const AMU_TO_KG = 1.66053906660e-27;  // [kg/amu]
const ANG2_TO_M2 = 1e-20;             // A^2 -> m^2
export const ENERGY_CONV = HBAR_meVs * Math.sqrt(KB_J / (AMU_TO_KG * ANG2_TO_M2));

// ── k-vector phase convention ───────────────────────────────────────────────
// src_gpu's Bloch phase is exp(i * cell_idx . kvec) with INTEGER cell indices,
// so kvec must be radians per cell = 2*pi * (conventional-cell fraction).
// Validated by S(k=G) == S(Gamma) (see src_gpu/validate_kpath_2pi.py).
export const TWO_PI_PHASE = 2.0 * Math.PI;

// THz <-> meV (band.yaml convention is THz; we compute internally in meV).
export const THZ_TO_MEV = 4.135667696;

// ── 3D viewer element appearance ────────────────────────────────────────────
// Full Jmol/CPK color table so ANY structure is colored (not just a handful of
// elements). A few entries (C, O, Se, Ga, Te) keep this app's tuned values; the
// rest are standard Jmol CPK. User-overridable in the viewer.
export const DEFAULT_COLORS = {
  H: '#ffffff', D: '#e0e0ff', He: '#d9ffff', Li: '#cc80ff', Be: '#c2ff00', B: '#ffb5b5',
  C: '#444444', N: '#3050f8', O: '#ff3030', F: '#90e050', Ne: '#b3e3f5', Na: '#ab5cf2',
  Mg: '#8aff00', Al: '#bfa6a6', Si: '#f0c8a0', P: '#ff8000', S: '#ffff30', Cl: '#1ff01f',
  Ar: '#80d1e3', K: '#8f40d4', Ca: '#3dff00', Sc: '#e6e6e6', Ti: '#bfc2c7', V: '#a6a6ab',
  Cr: '#8a99c7', Mn: '#9c7ac7', Fe: '#e06633', Co: '#f090a0', Ni: '#50d050', Cu: '#c88033',
  Zn: '#7d80b0', Ga: '#a67e5b', Ge: '#668f8f', As: '#bd80e3', Se: '#ff9900', Br: '#a62929',
  Kr: '#5cb8d1', Rb: '#702eb0', Sr: '#00ff00', Y: '#94ffff', Zr: '#94e0e0', Nb: '#73c2c9',
  Mo: '#54b5b5', Tc: '#3b9e9e', Ru: '#248f8f', Rh: '#0a7d8c', Pd: '#006985', Ag: '#c0c0c0',
  Cd: '#ffd98f', In: '#a67573', Sn: '#668080', Sb: '#9e63b5', Te: '#d4aa00', I: '#940094',
  Xe: '#429eb0', Cs: '#57178f', Ba: '#00c900', La: '#70d4ff', Ce: '#ffffc7', Pr: '#d9ffc7',
  Nd: '#c7ffc7', Pm: '#a3ffc7', Sm: '#8fffc7', Eu: '#61ffc7', Gd: '#45ffc7', Tb: '#30ffc7',
  Dy: '#1fffc7', Ho: '#00ff9c', Er: '#00e675', Tm: '#00d452', Yb: '#00bf38', Lu: '#00ab24',
  Hf: '#4dc2ff', Ta: '#4da6ff', W: '#2194d6', Re: '#267dab', Os: '#266696', Ir: '#175487',
  Pt: '#d0d0e0', Au: '#ffd123', Hg: '#b8b8d0', Tl: '#a6544d', Pb: '#575961', Bi: '#9e4fb5',
  Po: '#ab5c00', At: '#754f45', Rn: '#428296', Fr: '#420066', Ra: '#007d00', Ac: '#70abfa',
  Th: '#00baff', Pa: '#00a1ff', U: '#008fff', Np: '#0080ff', Pu: '#006bff', Am: '#545cf2',
  Cm: '#785ce3', Bk: '#8a4fe3', Cf: '#a136d4', Es: '#b31fd4',
};
// Cordero single-bond covalent radii [Å] (fallbacks; user-overridable).
export const COVALENT_R = {
  H: 0.31, D: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57,
  Ne: 0.58, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06,
  K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26,
  Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20, Kr: 1.16,
  Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54, Tc: 1.47, Ru: 1.46, Rh: 1.42,
  Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39, Xe: 1.40,
  Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04, Pr: 2.03, Nd: 2.01, Pm: 1.99, Sm: 1.98, Eu: 1.98,
  Gd: 1.96, Tb: 1.94, Dy: 1.92, Ho: 1.92, Er: 1.89, Tm: 1.90, Yb: 1.87, Lu: 1.87, Hf: 1.75,
  Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36, Hg: 1.32, Tl: 1.45,
  Pb: 1.46, Bi: 1.48, Po: 1.40, At: 1.50, Rn: 1.50, Fr: 2.60, Ra: 2.21, Ac: 2.15, Th: 2.06,
  Pa: 2.00, U: 1.96, Np: 1.90, Pu: 1.87, Am: 1.80, Cm: 1.69,
};

// ── Neutron total scattering cross-sections [barn] (NIST, natural abundance) ──
// From src_gpu/constants.py NEUTRON_SCATT_SIGMA.
export const NEUTRON_SCATT_SIGMA = {
  H: 82.02, D: 7.64, He: 1.34, Li: 1.37, Be: 7.63, B: 5.24, C: 5.551, N: 11.51, O: 4.232,
  F: 4.018, Ne: 2.62, Na: 3.28, Mg: 3.71, Al: 1.503, Si: 2.167, P: 3.31, S: 1.026, Cl: 16.8,
  K: 1.96, Ar: 0.68, Ca: 2.83, Sc: 23.4, Ti: 4.35, V: 5.10, Cr: 3.49, Mn: 2.15, Fe: 11.62,
  Co: 6.07, Ni: 18.5, Cu: 8.03, Zn: 4.054, Ga: 6.83, Ge: 8.42, As: 5.48, Se: 8.30, Br: 5.9,
  Kr: 7.66, Rb: 6.32, Sr: 6.25, Y: 7.76, Zr: 6.46, Nb: 6.253, Mo: 5.71, Tc: 6.0, Ru: 5.1,
  Rh: 4.81, Pd: 4.5, Ag: 4.99, Cd: 6.5, In: 2.61, Sn: 4.89, Sb: 3.9, Te: 4.25, I: 3.55,
  Xe: 4.3, Cs: 4.23, Ba: 3.38, La: 9.81, Ce: 2.94, Pr: 3.58, Nd: 16.5, Pm: 16.0, Sm: 39.0,
  Eu: 9.2, Gd: 175.0, Tb: 23.0, Dy: 34.4, Ho: 8.8, Er: 8.0, Tm: 8.5, Yb: 23.4, Lu: 5.9,
  Hf: 5.88, Ta: 6.01, W: 4.87, Re: 13.5, Os: 12.6, Ir: 14.0, Pt: 11.71, Au: 7.63, Hg: 26.5,
  Tl: 9.89, Pb: 11.11, Bi: 9.16, Th: 12.63, U: 14.16
};

// ── Neutron coherent scattering lengths b [fm] ──────────────────────────────
// From viz/sqeworker.js B_COH (used by the S(Q,E) calculation).
export const B_COH = {
  H: -3.739, He: 3.26, Li: -1.90, Be: 7.79, B: 5.30, C: 6.646, N: 9.36, O: 5.803,
  F: 5.654, Na: 3.63, Mg: 5.375, Al: 3.449, Si: 4.1491, P: 5.13, S: 2.847, Cl: 9.577,
  K: 3.67, Ca: 4.70, Sc: 12.29, Ti: -3.370, V: -0.3824, Cr: 3.635, Mn: -3.73,
  Fe: 9.45, Co: 2.49, Ni: 10.3, Cu: 7.718, Zn: 5.680, Ga: 7.288, Ge: 8.185,
  As: 6.58, Se: 7.970, Br: 6.795, Rb: 7.09, Sr: 7.02, Y: 7.75, Zr: 7.16, Nb: 7.054,
  Mo: 6.715, Tc: 6.80, Ru: 7.03, Rh: 5.88, Pd: 5.91, Ag: 5.922, Cd: 4.87, In: 4.065,
  Sn: 6.225, Sb: 5.57, Te: 5.80, I: 5.28, Cs: 5.42, Ba: 5.07, La: 8.24, Ce: 4.84,
  Pr: 4.58, Nd: 7.69, Sm: 0.80, Eu: 7.22, Gd: 6.5, Tb: 7.38, Dy: 16.9, Ho: 8.01,
  Er: 7.79, Tm: 7.07, Yb: 12.43, Lu: 7.21, Hf: 7.77, Ta: 6.91, W: 4.86, Re: 9.2,
  Os: 10.7, Ir: 10.6, Pt: 9.60, Au: 7.63, Hg: 12.692, Tl: 8.776, Pb: 9.405, Bi: 8.532
};
