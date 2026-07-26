// viewer/src/math/bravais.js
//
// Determine the Bravais lattice (crystal system + centering) of an RMC dataset
// the way seekpath/spglib does for the k-path tool: detect centering from the
// basis sites, build the standardized PRIMITIVE cell, and provide the transform
// between primitive and conventional reciprocal coordinates.
//
//   - high-symmetry points/BZ are expressed in the PRIMITIVE reciprocal basis
//     (so the displayed BZ is the true primitive Wigner-Seitz cell);
//   - k_conv = k_prim · T  converts to CONVENTIONAL reciprocal coords, which is
//     what the calculation needs (the RMC supercell tiles the conventional cell).
//
// This is a lightweight JS replacement for the seekpath backend (no spglib): it
// covers the standard centerings (P/F/I/C/A/B) detected directly from the basis,
// which matches seekpath for the common Bravais lattices.

import { mat3Inverse, mat3Transpose, detectSystem } from './reciprocal.js';

const TWO_PI = 2 * Math.PI;

function matmul3(M, A) {
  // rows of result = M rows combine rows of A
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    out[i][j] = M[i][0] * A[0][j] + M[i][1] * A[1][j] + M[i][2] * A[2][j];
  return out;
}
function scale3(A, s) { return A.map(r => r.map(v => v * s)); }

/** Conventional→primitive transform M for a centering letter (A_prim = M·A_conv). */
export function centeringMatrix(letter) {
  return CENTERING_M[letter] || CENTERING_M.P;
}

// Centering -> conventional→primitive transform M (A_prim = M · A_conv).
const CENTERING_M = {
  P: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  F: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
  I: [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
  C: [[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1]],   // S-C base-centered primitive
  A: [[1, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]],
  B: [[0.5, 0, 0.5], [0, 1, 0], [-0.5, 0, 0.5]],
};

// Candidate centering translations (fractional, conventional cell).
const CENTERING_VECS = {
  I: [[0.5, 0.5, 0.5]],
  F: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
  C: [[0.5, 0.5, 0]],
  A: [[0, 0.5, 0.5]],
  B: [[0.5, 0, 0.5]],
};

function isCentering(basis, vecs, tol = 0.06) {
  for (const t of vecs) {
    for (const s of basis) {
      const target = [(s.frac[0] + t[0]) % 1, (s.frac[1] + t[1]) % 1, (s.frac[2] + t[2]) % 1];
      let found = false;
      for (const s2 of basis) {
        if (s2.el !== s.el) continue;
        let d = 0;
        for (let k = 0; k < 3; k++) { let dk = Math.abs(target[k] - s2.frac[k]); dk = Math.min(dk, 1 - dk); d += dk * dk; }
        if (Math.sqrt(d) < tol) { found = true; break; }
      }
      if (!found) return false;
    }
  }
  return true;
}

function detectCentering(basis) {
  if (!basis || basis.length === 0) return 'P';
  // F before I/C since F implies the single C/A/B too.
  if (isCentering(basis, CENTERING_VECS.F)) return 'F';
  if (isCentering(basis, CENTERING_VECS.I)) return 'I';
  if (isCentering(basis, CENTERING_VECS.C)) return 'C';
  if (isCentering(basis, CENTERING_VECS.A)) return 'A';
  if (isCentering(basis, CENTERING_VECS.B)) return 'B';
  return 'P';
}

// (system, centering) -> Bravais code used by the high-symmetry tables.
function bravaisCode(system, centering) {
  const map = {
    cubic: { P: 'CUB', F: 'FCC', I: 'BCC' },
    tetragonal: { P: 'TET', I: 'BCT' },
    orthorhombic: { P: 'ORC', F: 'ORCF', I: 'ORCI', C: 'ORCC', A: 'ORCC', B: 'ORCC' },
    hexagonal: { P: 'HEX' },
    rhombohedral: { P: 'RHL' },
    monoclinic: { P: 'MCL', C: 'MCLC' },
    triclinic: { P: 'TRI' },
  };
  return (map[system] && map[system][centering]) || (map[system] && map[system].P) || 'CUB';
}

/**
 * Analyze the conventional lattice + basis into a Bravais description.
 *   A_conv : conventional cell rows (Å)  [= v_super / dim]
 *   basis  : [{ el, frac:[x,y,z] }] within-cell basis sites
 * Returns { system, centering, code, A_conv, A_prim, B_conv, B_prim, T }.
 */
export function analyzeBravais(A_conv, basis) {
  const metric = detectSystemFromMatrix(A_conv);
  const { system } = metric;
  let centering = detectCentering(basis);
  // Centering must be compatible with the system (e.g. no F in hexagonal).
  const allowed = { cubic: 'PFI', tetragonal: 'PI', orthorhombic: 'PFICAB', hexagonal: 'P', rhombohedral: 'P', monoclinic: 'PC', triclinic: 'P' };
  if (!(allowed[system] || 'P').includes(centering)) centering = 'P';

  const M = CENTERING_M[centering] || CENTERING_M.P;
  const A_prim = matmul3(M, A_conv);

  const B_conv = scale3(mat3Transpose(mat3Inverse(A_conv)), TWO_PI); // 2π reciprocal
  const B_prim = scale3(mat3Transpose(mat3Inverse(A_prim)), TWO_PI);
  const T = matmul3(B_prim, mat3Inverse(B_conv));                    // k_conv = k_prim · T

  // M = conventional→primitive transform (A_prim = M·A_conv). Used as the
  // computation-cell P for the primitive cell (cells.buildCellLabeling) so the
  // dispersion is computed unfolded. M = I for primitive (P) lattices.
  return { system, centering, code: bravaisCode(system, centering), metric, A_conv, A_prim, B_conv, B_prim, T, M };
}

// detectSystem in reciprocal.js takes (v1,v2,v3,dim); wrap for an A matrix.
function detectSystemFromMatrix(A) {
  return detectSystem(A[0], A[1], A[2], [1, 1, 1]);
}

/** k_prim (row vec) · T -> k_conv. */
export function primToConv(kPrim, T) {
  return [
    kPrim[0] * T[0][0] + kPrim[1] * T[1][0] + kPrim[2] * T[2][0],
    kPrim[0] * T[0][1] + kPrim[1] * T[1][1] + kPrim[2] * T[2][1],
    kPrim[0] * T[0][2] + kPrim[1] * T[1][2] + kPrim[2] * T[2][2],
  ];
}

/** k_prim (fractional) -> cartesian using primitive reciprocal rows B_prim. */
export function fracToCart(kPrim, B) {
  return [
    kPrim[0] * B[0][0] + kPrim[1] * B[1][0] + kPrim[2] * B[2][0],
    kPrim[0] * B[0][1] + kPrim[1] * B[1][1] + kPrim[2] * B[2][1],
    kPrim[0] * B[0][2] + kPrim[1] * B[1][2] + kPrim[2] * B[2][2],
  ];
}
