// viewer/src/math/highsym.js
//
// Standard high-symmetry points (Setyawan & Curtarolo, Comp. Mat. Sci. 49
// (2010) 299 — the convention seekpath also uses) in the PRIMITIVE reciprocal
// basis, plus suggested band paths. Combined with the primitive cell
// (bravais.js) and the Wigner-Seitz BZ (brillouin.js) this reproduces the
// seekpath k-path for the common Bravais lattices.
//
// CAVEAT (documented): the variant-dependent formulas assume the conventional
// cell is in the S-C STANDARD SETTING (axis ordering / choice). RMC cells from
// CIFs usually are, but this code does not standardize the cell (that needs
// spglib, which can't run in a static browser). buildBZModel additionally
// checks every point lies on/in the BZ, which catches gross setting/transcription
// errors. For a new low-symmetry material, spot-check labels against seekpath.

import { brillouinZone } from './brillouin.js';
import { primToConv, fracToCart } from './bravais.js';
import { highSymmetryPoints, reciprocalLattice, detectSystem } from './reciprocal.js';

const T3 = 1 / 3;

// ── Static (parameter-free) high-symmetry tables ────────────────────────────
const STATIC = {
  CUB: { pts: { GAMMA: [0, 0, 0], X: [0, 0.5, 0], M: [0.5, 0.5, 0], R: [0.5, 0.5, 0.5] },
    path: [['GAMMA', 'X'], ['X', 'M'], ['M', 'GAMMA'], ['GAMMA', 'R'], ['R', 'X'], ['M', 'R']] },
  FCC: { pts: { GAMMA: [0, 0, 0], X: [0.5, 0, 0.5], L: [0.5, 0.5, 0.5], W: [0.5, 0.25, 0.75], K: [0.375, 0.375, 0.75], U: [0.625, 0.25, 0.625] },
    path: [['GAMMA', 'X'], ['X', 'W'], ['W', 'K'], ['K', 'GAMMA'], ['GAMMA', 'L'], ['L', 'U'], ['U', 'W'], ['W', 'L'], ['L', 'K'], ['U', 'X']] },
  BCC: { pts: { GAMMA: [0, 0, 0], H: [0.5, -0.5, 0.5], P: [0.25, 0.25, 0.25], N: [0, 0, 0.5] },
    path: [['GAMMA', 'H'], ['H', 'N'], ['N', 'GAMMA'], ['GAMMA', 'P'], ['P', 'H'], ['P', 'N']] },
  TET: { pts: { GAMMA: [0, 0, 0], X: [0, 0.5, 0], M: [0.5, 0.5, 0], Z: [0, 0, 0.5], R: [0, 0.5, 0.5], A: [0.5, 0.5, 0.5] },
    path: [['GAMMA', 'X'], ['X', 'M'], ['M', 'GAMMA'], ['GAMMA', 'Z'], ['Z', 'R'], ['R', 'A'], ['A', 'Z'], ['X', 'R'], ['M', 'A']] },
  ORC: { pts: { GAMMA: [0, 0, 0], X: [0.5, 0, 0], S: [0.5, 0.5, 0], Y: [0, 0.5, 0], Z: [0, 0, 0.5], U: [0.5, 0, 0.5], R: [0.5, 0.5, 0.5], T: [0, 0.5, 0.5] },
    path: [['GAMMA', 'X'], ['X', 'S'], ['S', 'Y'], ['Y', 'GAMMA'], ['GAMMA', 'Z'], ['Z', 'U'], ['U', 'R'], ['R', 'T'], ['T', 'Z'], ['Y', 'T'], ['U', 'X'], ['S', 'R']] },
  HEX: { pts: { GAMMA: [0, 0, 0], A: [0, 0, 0.5], K: [T3, T3, 0], H: [T3, T3, 0.5], M: [0.5, 0, 0], L: [0.5, 0, 0.5] },
    path: [['GAMMA', 'M'], ['M', 'K'], ['K', 'GAMMA'], ['GAMMA', 'A'], ['A', 'L'], ['L', 'H'], ['H', 'A'], ['L', 'M'], ['K', 'H']] },
};

const cosd = (d) => Math.cos(d * Math.PI / 180);

// ── Parameter-dependent (variant) generators ────────────────────────────────
function bct({ a, c }) {
  if (c < a) {                                   // BCT1
    const e = (1 + (c * c) / (a * a)) / 4;
    return { variant: 'BCT1', pts: {
      GAMMA: [0, 0, 0], M: [-0.5, 0.5, 0.5], N: [0, 0.5, 0], P: [0.25, 0.25, 0.25], X: [0, 0, 0.5], Z: [e, e, -e], Z1: [-e, 1 - e, e] },
      path: [['GAMMA', 'X'], ['X', 'M'], ['M', 'GAMMA'], ['GAMMA', 'Z'], ['Z', 'P'], ['P', 'N'], ['N', 'Z1'], ['Z1', 'M'], ['X', 'P']] };
  }
  const e = (1 + (a * a) / (c * c)) / 4, z = (a * a) / (2 * c * c);   // BCT2
  return { variant: 'BCT2', pts: {
    GAMMA: [0, 0, 0], N: [0, 0.5, 0], P: [0.25, 0.25, 0.25], SIGMA: [-e, e, e], SIGMA1: [e, 1 - e, -e],
    X: [0, 0, 0.5], Y: [-z, z, 0.5], Y1: [0.5, 0.5, -z], Z: [0.5, 0.5, -0.5] },
    path: [['GAMMA', 'X'], ['X', 'Y'], ['Y', 'SIGMA'], ['SIGMA', 'GAMMA'], ['GAMMA', 'Z'], ['Z', 'SIGMA1'], ['SIGMA1', 'N'], ['N', 'P'], ['P', 'Y1'], ['Y1', 'Z'], ['X', 'P']] };
}

function rhl({ alpha }) {
  const ca = cosd(alpha);
  if (alpha < 90) {                              // RHL1
    const e = (1 + 4 * ca) / (2 + 4 * ca), nu = 0.75 - e / 2;
    return { variant: 'RHL1', pts: {
      GAMMA: [0, 0, 0], B: [e, 0.5, 1 - e], B1: [0.5, 1 - e, e - 1], F: [0.5, 0.5, 0], L: [0.5, 0, 0], L1: [0, 0, -0.5],
      P: [e, nu, nu], P1: [1 - nu, 1 - nu, 1 - e], P2: [nu, nu, e - 1], Q: [1 - nu, nu, 0], X: [nu, 0, -nu], Z: [0.5, 0.5, 0.5] },
      path: [['GAMMA', 'L'], ['L', 'B1'], ['B', 'Z'], ['Z', 'GAMMA'], ['GAMMA', 'X'], ['Q', 'F'], ['F', 'P1'], ['P1', 'Z'], ['L', 'P']] };
  }
  const th = Math.tan((alpha * Math.PI / 180) / 2);   // RHL2
  const e = 1 / (2 * th * th), nu = 0.75 - e / 2;
  return { variant: 'RHL2', pts: {
    GAMMA: [0, 0, 0], F: [0.5, -0.5, 0], L: [0.5, 0, 0], P: [1 - nu, -nu, 1 - nu], P1: [nu, nu - 1, nu - 1], Q: [e, e, e], Q1: [1 - e, -e, -e], Z: [0.5, -0.5, 0.5] },
    path: [['GAMMA', 'P'], ['P', 'Z'], ['Z', 'Q'], ['Q', 'GAMMA'], ['GAMMA', 'F'], ['F', 'P1'], ['P1', 'Q1'], ['Q1', 'L'], ['L', 'Z']] };
}

function orcc({ a, b }) {
  const z = (1 + (a * a) / (b * b)) / 4;
  return { variant: 'ORCC', pts: {
    GAMMA: [0, 0, 0], A: [z, z, 0.5], A1: [-z, 1 - z, 0.5], R: [0, 0.5, 0.5], S: [0, 0.5, 0], T: [-0.5, 0.5, 0.5],
    X: [z, z, 0], X1: [-z, 1 - z, 0], Y: [-0.5, 0.5, 0], Z: [0, 0, 0.5] },
    path: [['GAMMA', 'X'], ['X', 'S'], ['S', 'R'], ['R', 'A'], ['A', 'Z'], ['Z', 'GAMMA'], ['GAMMA', 'Y'], ['Y', 'X1'], ['X1', 'A1'], ['A1', 'T'], ['T', 'Y'], ['Z', 'T']] };
}

function orci({ a, b, c }) {
  const z = (1 + (a * a) / (c * c)) / 4, e = (1 + (b * b) / (c * c)) / 4;
  const d = (b * b - a * a) / (4 * c * c), mu = (a * a + b * b) / (4 * c * c);
  return { variant: 'ORCI', pts: {
    GAMMA: [0, 0, 0], L: [-mu, mu, 0.5 - d], L1: [mu, -mu, 0.5 + d], L2: [0.5 - d, 0.5 + d, -mu], R: [0, 0.5, 0], S: [0.5, 0, 0],
    T: [0, 0, 0.5], W: [0.25, 0.25, 0.25], X: [-z, z, z], X1: [z, 1 - z, -z], Y: [e, -e, e], Y1: [1 - e, e, -e], Z: [0.5, 0.5, -0.5] },
    path: [['GAMMA', 'X'], ['X', 'L'], ['L', 'T'], ['T', 'W'], ['W', 'R'], ['R', 'X1'], ['X1', 'Z'], ['Z', 'GAMMA'], ['GAMMA', 'Y'], ['Y', 'S'], ['S', 'W'], ['L1', 'Y'], ['Y1', 'Z']] };
}

function orcf({ a, b, c }) {
  const inva = 1 / (a * a), invb = 1 / (b * b), invc = 1 / (c * c);
  if (inva > invb + invc + 1e-9) {               // ORCF1
    const z = (1 + (a * a) / (b * b) - (a * a) / (c * c)) / 4, e = (1 + (a * a) / (b * b) + (a * a) / (c * c)) / 4;
    return { variant: 'ORCF1', pts: {
      GAMMA: [0, 0, 0], A: [0.5, 0.5 + z, z], A1: [0.5, 0.5 - z, 1 - z], L: [0.5, 0.5, 0.5], T: [1, 0.5, 0.5], X: [0, e, e], X1: [1, 1 - e, 1 - e], Y: [0.5, 0, 0.5], Z: [0.5, 0.5, 0] },
      path: [['GAMMA', 'Y'], ['Y', 'T'], ['T', 'Z'], ['Z', 'GAMMA'], ['GAMMA', 'X'], ['X', 'A1'], ['A1', 'Y'], ['T', 'X1'], ['X', 'A'], ['A', 'Z'], ['L', 'GAMMA']] };
  }
  if (inva < invb + invc - 1e-9) {               // ORCF2
    const e = (1 + (a * a) / (b * b) - (a * a) / (c * c)) / 4;
    const phi = (1 + (c * c) / (b * b) - (c * c) / (a * a)) / 4, d = (1 + (b * b) / (a * a) - (b * b) / (c * c)) / 4;
    return { variant: 'ORCF2', pts: {
      GAMMA: [0, 0, 0], C: [0.5, 0.5 - e, 1 - e], C1: [0.5, 0.5 + e, e], D: [0.5 - d, 0.5, 1 - d], D1: [0.5 + d, 0.5, d], L: [0.5, 0.5, 0.5],
      H: [1 - phi, 0.5 - phi, 0.5], H1: [phi, 0.5 + phi, 0.5], X: [0, 0.5, 0.5], Y: [0.5, 0, 0.5], Z: [0.5, 0.5, 0] },
      path: [['GAMMA', 'Y'], ['Y', 'C'], ['C', 'D'], ['D', 'X'], ['X', 'GAMMA'], ['GAMMA', 'Z'], ['Z', 'D1'], ['D1', 'H'], ['H', 'C'], ['C1', 'Z'], ['X', 'H1'], ['H', 'Y'], ['L', 'GAMMA']] };
  }
  // ORCF3 (1/a² == 1/b² + 1/c²)
  const z = (1 + (a * a) / (b * b) - (a * a) / (c * c)) / 4, e = (1 + (a * a) / (b * b) + (a * a) / (c * c)) / 4;
  return { variant: 'ORCF3', pts: {
    GAMMA: [0, 0, 0], A: [0.5, 0.5 + z, z], A1: [0.5, 0.5 - z, 1 - z], L: [0.5, 0.5, 0.5], T: [1, 0.5, 0.5], X: [0, e, e], Y: [0.5, 0, 0.5], Z: [0.5, 0.5, 0] },
    path: [['GAMMA', 'Y'], ['Y', 'T'], ['T', 'Z'], ['Z', 'GAMMA'], ['GAMMA', 'X'], ['X', 'A1'], ['A1', 'Y'], ['X', 'A'], ['A', 'Z'], ['L', 'GAMMA']] };
}

// Generic primitive-cell point set (BZ face/edge/corner) — safe fallback for
// lattices not in S-C standard setting; the buildBZModel guard prunes any point
// that still lands outside the BZ for very oblique cells.
function genericP(variant) {
  return { variant, pts: {
    GAMMA: [0, 0, 0], X: [0.5, 0, 0], Y: [0, 0.5, 0], Z: [0, 0, 0.5],
    L: [0.5, 0.5, 0], M: [0, 0.5, 0.5], N: [0.5, 0, 0.5], R: [0.5, 0.5, 0.5] },
    path: [['X', 'GAMMA'], ['GAMMA', 'Y'], ['L', 'GAMMA'], ['GAMMA', 'Z'], ['N', 'GAMMA'], ['GAMMA', 'M'], ['R', 'GAMMA']] };
}

function tri(bravais) {
  // Variant from reciprocal-lattice angles: 'a' set if obtuse, 'b' if acute.
  const B = bravais.B_prim;
  const ang = (u, v) => Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) /
    (Math.hypot(...u) * Math.hypot(...v))))) * 180 / Math.PI;
  const ga = ang(B[0], B[1]);
  if (ga >= 90) {
    return { variant: 'TRI1a', pts: {
      GAMMA: [0, 0, 0], L: [0.5, 0.5, 0], M: [0, 0.5, 0.5], N: [0.5, 0, 0.5], R: [0.5, 0.5, 0.5], X: [0.5, 0, 0], Y: [0, 0.5, 0], Z: [0, 0, 0.5] },
      path: [['X', 'GAMMA'], ['GAMMA', 'Y'], ['L', 'GAMMA'], ['GAMMA', 'Z'], ['N', 'GAMMA'], ['GAMMA', 'M'], ['R', 'GAMMA']] };
  }
  return { variant: 'TRI1b', pts: {
    GAMMA: [0, 0, 0], L: [0.5, -0.5, 0], M: [0, 0, 0.5], N: [-0.5, -0.5, 0.5], R: [0, -0.5, 0.5], X: [0, -0.5, 0], Y: [0.5, 0, 0], Z: [-0.5, 0, 0.5] },
    path: [['X', 'GAMMA'], ['GAMMA', 'Y'], ['L', 'GAMMA'], ['GAMMA', 'Z'], ['N', 'GAMMA'], ['GAMMA', 'M'], ['R', 'GAMMA']] };
}

function tableFor(bravais) {
  const m = bravais.metric;
  switch (bravais.code) {
    case 'BCT': return bct(m);
    case 'RHL': return rhl(m);
    case 'ORCC': return orcc(m);
    case 'ORCI': return orci(m);
    case 'ORCF': return orcf(m);
    // MCL/MCLC need cell standardization (S-C unique-axis setting) we can't do
    // without spglib, so use a generic in-BZ primitive set rather than risk
    // wrong labels. CUB/FCC/BCC/TET/ORC/HEX/BCT/RHL/ORCC/ORCI/ORCF are exact S-C.
    case 'MCL': return genericP('MCL (generic; S-C setting needs standardization)');
    case 'MCLC': return genericP('MCLC (generic; S-C setting needs standardization)');
    case 'TRI': return tri(bravais);
    default: return STATIC[bravais.code] || STATIC.CUB;
  }
}

const DISPLAY = { GAMMA: 'Γ', SIGMA: 'Σ', SIGMA1: 'Σ₁' };
export const displayLabel = (l) => DISPLAY[l] || l;
export function hasTable(code) { return !!(STATIC[code]) || ['BCT', 'RHL', 'ORCC', 'ORCI', 'ORCF', 'MCL', 'TRI'].includes(code); }

// True if cartesian k lies on/in the primitive Wigner-Seitz BZ.
function inBZ(kCart, B, tol = 1e-6) {
  for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) for (let k = -2; k <= 2; k++) {
    if (!i && !j && !k) continue;
    const G = [i * B[0][0] + j * B[1][0] + k * B[2][0], i * B[0][1] + j * B[1][1] + k * B[2][1], i * B[0][2] + j * B[1][2] + k * B[2][2]];
    const g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    if (kCart[0] * G[0] + kCart[1] * G[1] + kCart[2] * G[2] > g2 / 2 + tol) return false;
  }
  return true;
}

/** Assemble the BZ model: points (primitive frac + conventional frac + cartesian),
 *  suggested path, primitive Wigner-Seitz cell, chosen variant. A runtime guard
 *  ensures every emitted point lies in the BZ: if the S-C set doesn't (cell not
 *  in standard setting), fall back to a generic set and prune stragglers. */
export function buildBZModel(bravais) {
  const B = bravais.B_prim;
  let tbl = tableFor(bravais);
  const cartOf = (frac) => fracToCart(frac, B);
  const allIn = (t) => Object.values(t.pts).every(f => inBZ(cartOf(f), B));
  if (!allIn(tbl)) tbl = genericP(`${tbl.variant} → generic (cell not in S-C standard setting)`);

  const points = {};
  for (const [label, frac] of Object.entries(tbl.pts)) {
    const cart = cartOf(frac);
    if (!inBZ(cart, B)) continue; // prune any straggler outside the BZ
    points[label] = { frac, fracConv: primToConv(frac, bravais.T), cart, display: displayLabel(label) };
  }
  const path = tbl.path.filter(([a, b]) => points[a] && points[b]);
  return { code: bravais.code, variant: tbl.variant, points, path, bz: brillouinZone(B) };
}

/**
 * Conventional-cell BZ model (cell-framework default, P = I). Same shape as
 * buildBZModel, but in the CONVENTIONAL reciprocal cell: high-symmetry points are
 * the simple-setting conventional-fractional points (reciprocal.js HIGH_SYM, e.g.
 * cubic X at ½), so `fracConv` is the point itself — no primitive→conventional
 * fold. This is what the S(k) calculation needs when computing over the
 * conventional cell, and it removes the spurious Γ→X mirror symmetry that the
 * primitive seekpath path produces (primitive X folds onto a conventional
 * reciprocal-lattice vector ≡ Γ). The displayed zone is the Wigner-Seitz cell of
 * the conventional reciprocal lattice.
 */
export function buildConventionalBZModel(bravais) {
  const Bc = bravais.B_conv;                    // 2π conventional reciprocal rows
  const hs = highSymmetryPoints(bravais.system);
  const points = {};
  for (const [label, frac] of Object.entries(hs.points)) {
    points[label] = { frac, fracConv: frac, cart: fracToCart(frac, Bc), display: displayLabel(label) };
  }
  const seq = hs.defaultPath || [];
  const path = [];
  for (let i = 0; i < seq.length - 1; i++) if (points[seq[i]] && points[seq[i + 1]]) path.push([seq[i], seq[i + 1]]);
  return { code: 'conventional', variant: 'conventional', points, path, bz: brillouinZone(Bc) };
}

/**
 * BZ model for a CUSTOM SUPERCELL L = diag(n)·A_conv. The supercell has its own
 * (smaller) reciprocal cell and its own crystal system — e.g. a 1×1×2 cubic
 * supercell is tetragonal — so the BZ shape and high-symmetry path follow it.
 * Points are in the SUPERCELL's own reciprocal-fractional coords (fed straight to
 * the phase; the pipeline relabels with the matching P = diag(n)).
 */
export function buildSupercellBZModel(bravais, n) {
  const Ac = bravais.A_conv;
  const L = [0, 1, 2].map(i => [Ac[i][0] * n[i], Ac[i][1] * n[i], Ac[i][2] * n[i]]); // diag(n)·A_conv
  const B = reciprocalLattice(L);
  const { system } = detectSystem(L[0], L[1], L[2], [1, 1, 1]);
  const hs = highSymmetryPoints(system);
  const points = {};
  for (const [label, frac] of Object.entries(hs.points)) {
    points[label] = { frac, fracConv: frac, cart: fracToCart(frac, B), display: displayLabel(label) };
  }
  const seq = hs.defaultPath || [];
  const path = [];
  for (let i = 0; i < seq.length - 1; i++) if (points[seq[i]] && points[seq[i + 1]]) path.push([seq[i], seq[i + 1]]);
  return { code: `${n.join('×')} · ${system}`, variant: 'supercell', points, path, bz: brillouinZone(B) };
}
