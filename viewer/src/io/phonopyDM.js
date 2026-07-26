// viewer/src/io/phonopyDM.js
//
// Reader for the phononwebsite / phonopy "dynamical-matrix" JSON export
// (format "phonopy-dynamical-matrix-v1") — e.g. the per-material files from the
// Materials Project phonon database. Unlike a band.yaml/.json, this file does
// NOT store frequencies/eigenvectors; it carries the real-space compact force
// constants plus the Gonze–Lee non-analytic (LO–TO) correction, and the bands
// are obtained by building the dynamical matrix D(q) at each q-point and
// diagonalizing it.
//
// The construction below is a faithful port of the phononwebsite reference
// implementation (see archive/viz/phonon_assets/main.min.js:
// buildDynamicalMatrixBlocks / getGonzeReciprocalCorrection / …). Diagonalization
// reuses the app's own Hermitian solver (math/diagonalize.eigh) instead of the
// WASM Eigen backend, and the result is packaged as the app's viewer model.

import { eigh } from '../math/diagonalize.js';
import { fromBandText } from './viewermodel.js';

const THZ2CM1 = 33.35641;
const CM1_TO_MEV = 0.12398419843320026;   // 1 cm⁻¹ in meV

export function isPhonopyDM(obj) {
  return !!obj && (obj.format === 'phonopy-dynamical-matrix-v1' ||
    (!!obj.dynamical_matrix && obj.dynamical_matrix.format === 'phonopy-dynamical-matrix-v1'));
}

/* ── linear algebra helpers (ported) ─────────────────────────────────────── */
function determinant3x3(m) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
function invert3x3(m) {
  const det = determinant3x3(m);
  if (Math.abs(det) < 1e-16) return null;
  const id = 1 / det;
  return [
    [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id],
    [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id],
    [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id],
  ];
}
const matVec = (m, v) => [
  m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
  m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
  m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/* ── Gonze–Lee non-analytic (LO–TO) correction (ported) ──────────────────── */
function complexBlockTensor(n) {
  return Array.from({ length: n }, () => Array.from({ length: n },
    () => [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]));
}
function dielectricPart(q, eps) {
  let t = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) t += q[i] * eps[i][j] * q[j];
  return t;
}
function kkTensor(g, qCart, qDirCart, eps, lambda, tol) {
  const qK = [g[0] + qCart[0], g[1] + qCart[1], g[2] + qCart[2]];
  const norm = Math.sqrt(dot3(qK, qK));
  const kk = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  if (norm < tol) {
    if (!qDirCart) return kk;
    const dp = dielectricPart(qDirCart, eps);
    if (!(Math.abs(dp) > 1e-16)) return kk;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) kk[i][j] = qDirCart[i] * qDirCart[j] / dp;
    return kk;
  }
  const dp = dielectricPart(qK, eps);
  if (!(Math.abs(dp) > 1e-16)) return kk;
  const pref = Math.exp(-dp / (4 * lambda * lambda)) / dp;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) kk[i][j] = qK[i] * qK[j] * pref;
  return kk;
}
function ddTensorPart(gList, qCart, qDirCart, eps, posCar, lambda, tol) {
  const n = posCar.length;
  const dd = complexBlockTensor(n);
  for (let g = 0; g < gList.length; g++) {
    const gv = gList[g];
    const kk = kkTensor(gv, qCart, qDirCart, eps, lambda, tol);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let ph = 0;
      for (let a = 0; a < 3; a++) ph += (posCar[i][a] - posCar[j][a]) * gv[a];
      ph *= 2 * Math.PI;
      const c = Math.cos(ph), s = Math.sin(ph);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const v = kk[a][b];
        dd[i][j][a][b][0] += v * c;
        dd[i][j][a][b][1] += v * s;
      }
    }
  }
  return dd;
}
function multiplyBorns(ddIn, born) {
  const n = born.length;
  const dd = complexBlockTensor(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      const t = dd[i][j][a][b];
      for (let ap = 0; ap < 3; ap++) for (let bp = 0; bp < 3; bp++) {
        const zz = born[i][ap][a] * born[j][bp][b];
        t[0] += ddIn[i][j][ap][bp][0] * zz;
        t[1] += ddIn[i][j][ap][bp][1] * zz;
      }
    }
  return dd;
}
function gonzeReciprocalCorrection(payload, qpoint, qDirection) {
  const nac = payload.nac;
  if (!nac || nac.method !== 'gonze') return null;
  const inv = invert3x3(payload.primitive_lattice);
  if (!inv) return null;
  const qCart = matVec(inv, qpoint);
  const qDirCart = qDirection ? matVec(inv, qDirection) : null;
  const ddPart = ddTensorPart(nac.g_list, qCart, qDirCart, nac.dielectric, nac.positions_car, nac.lambda, nac.q_direction_tolerance || 1e-5);
  const dd = multiplyBorns(ddPart, nac.born);
  const n = payload.masses.length;
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    dd[i][i][a][b][0] -= nac.dd_q0.real[i][a][b];
    dd[i][i][a][b][1] -= nac.dd_q0.imag[i][a][b];
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const scale = nac.nac_factor / Math.sqrt(payload.masses[i] * payload.masses[j]);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      dd[i][j][a][b][0] *= scale;
      dd[i][j][a][b][1] *= scale;
    }
  }
  return dd;
}

/* ── acoustic-sum-rule (translational) on the compact force constants ─────── */
function asrForceConstants(payload) {
  const fc = payload.force_constants_compact;
  const asr = (payload.acoustic_sum_rule === false || payload.acoustic_sum_rule === 'off')
    ? 'off' : (typeof payload.acoustic_sum_rule === 'string' ? payload.acoustic_sum_rule : 'translational');
  if (asr !== 'translational') return fc;
  // Remove the drift along the supercell-atom index (the one summed below).
  const c = fc.map(p => p.map(s => [s[0].slice(), s[1].slice(), s[2].slice()]));
  for (let i = 0; i < c.length; i++)
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      let drift = 0;
      for (let s = 0; s < c[i].length; s++) drift += c[i][s][a][b];
      drift /= c[i].length;
      for (let s = 0; s < c[i].length; s++) c[i][s][a][b] -= drift;
    }
  return c;
}

/* ── dynamical matrix D(q) → flat real/imag (N=3·natoms) ─────────────────── */
function dynamicalMatrix(payload, qpoint, qDirection) {
  const masses = payload.masses;
  const fc = payload._asr || (payload._asr = asrForceConstants(payload));
  const svecs = payload.shortest_vectors;
  const multi = payload.multiplicity;
  const s2pp = payload.s2pp_map;
  const natoms = masses.length;
  const N = natoms * 3;
  const nac = payload.nac ? gonzeReciprocalCorrection(payload, qpoint, qDirection) : null;

  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  for (let i = 0; i < natoms; i++) {
    for (let j = 0; j < natoms; j++) {
      const sqrtMass = Math.sqrt(masses[i] * masses[j]);
      const bRe = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bIm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      if (nac) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { bRe[a][b] += nac[i][j][a][b][0]; bIm[a][b] += nac[i][j][a][b][1]; }

      for (let s = 0; s < s2pp.length; s++) {
        if (s2pp[s] !== j) continue;
        const count = multi[s][i][0], addr = multi[s][i][1];
        let pRe = 0, pIm = 0;
        for (let v = 0; v < count; v++) {
          const sv = svecs[addr + v];
          const ph = 2 * Math.PI * (sv[0] * qpoint[0] + sv[1] * qpoint[1] + sv[2] * qpoint[2]);
          pRe += Math.cos(ph); pIm += Math.sin(ph);
        }
        const factor = 1 / (sqrtMass * count);
        const blk = fc[i][s];
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
          const val = blk[a][b] * factor;
          bRe[a][b] += val * pRe; bIm[a][b] += val * pIm;
        }
      }
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const r = i * 3 + a, c = j * 3 + b;
        re[r * N + c] = bRe[a][b]; im[r * N + c] = bIm[a][b];
      }
    }
  }
  // Hermitize.
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const sr = 0.5 * (re[i * N + j] + re[j * N + i]);
      const si = 0.5 * (im[i * N + j] - im[j * N + i]);
      re[i * N + j] = sr; re[j * N + i] = sr;
      im[i * N + j] = si; im[j * N + i] = -si;
    }
    im[i * N + i] = 0;
  }
  return { re, im, N };
}

const eigToCm1 = (val, conv) => {
  const mag = Math.sqrt(Math.abs(val));
  return (val < 0 ? -mag : mag) * conv * THZ2CM1;
};

// q-direction (for the LO–TO term at Γ): the path's travel direction there.
function qDirectionAt(qpoints, i) {
  const q = qpoints[i];
  if (Math.hypot(q[0], q[1], q[2]) > 1e-6) return null;   // only matters at Γ
  const nb = (i + 1 < qpoints.length) ? qpoints[i + 1] : qpoints[i - 1];
  if (!nb) return null;
  const d = (i + 1 < qpoints.length) ? [nb[0] - q[0], nb[1] - q[1], nb[2] - q[2]]
    : [q[0] - nb[0], q[1] - nb[1], q[2] - nb[2]];
  const n = Math.hypot(d[0], d[1], d[2]);
  return n > 1e-9 ? [d[0] / n, d[1] / n, d[2] / n] : null;
}

/**
 * Build the app's viewer model from a phonopy-dynamical-matrix JSON object.
 * Diagonalizes D(q) at every q-point (with Gonze NAC when present) → bands (meV)
 * + eigendisplacements, and packages structure + k-path metadata.
 */
export function fromPhonopyDM(obj) {
  const payload = obj.dynamical_matrix && obj.dynamical_matrix.format === 'phonopy-dynamical-matrix-v1'
    ? obj.dynamical_matrix : obj;
  // Pull the structure either from the top-level export or the matrix payload.
  const lattice = obj.lattice || payload.primitive_lattice;
  const atomTypes = obj.atom_types;
  const posRed = obj.atom_pos_red;
  const masses = payload.masses;
  const conv = Number(payload.frequency_conversion_factor) || 1;
  if (!payload.force_constants_compact || !lattice || !atomTypes || !posRed) {
    throw new Error('Not a recognized phonopy-dynamical-matrix file (missing force constants / structure).');
  }
  const natoms = masses.length;
  const qpoints = obj.qpoints;
  const avgMass = Number(obj.average_mass) || (masses.reduce((a, b) => a + b, 0) / masses.length);

  const bands = [], eigvecs = [];
  for (let qi = 0; qi < qpoints.length; qi++) {
    const { re, im, N } = dynamicalMatrix(payload, qpoints[qi], qDirectionAt(qpoints, qi));
    const { eigenvalues, eigenvectors } = eigh(re, im, N);   // ascending
    const row = new Array(N), evRow = new Array(N);
    for (let m = 0; m < N; m++) {
      bandsSet(row, m, eigToCm1(eigenvalues[m], conv) * CM1_TO_MEV);
      // Keep the raw unit-norm eigenvectors of the MASS-WEIGHTED dynamical
      // matrix — the phonopy convention shared by every model source.
      // Consumers apply their own mass weighting (the 3D viewer divides by
      // sqrt(m) via baseStructure.masses; the S(Q,E) structure factor uses
      // b^2/m directly on these vectors).
      evRow[m] = { real: eigenvectors[m].real, imag: eigenvectors[m].imag };
    }
    bands.push(row); eigvecs.push(evRow);
  }

  // Structure → baseStructure (one primitive cell; row order = atom order).
  const atomDic = {};
  atomTypes.forEach((el, idx) => { (atomDic[el] || (atomDic[el] = [])).push(idx); });
  const uniqueRN = atomTypes.map((_, idx) => idx);
  const hsym_xyz = Float64Array.from(posRed.flat());
  const baseStructure = {
    atomDic, dim: [1, 1, 1],
    v1: lattice[0], v2: lattice[1], v3: lattice[2],
    uniqueRN, atomType: uniqueRN.slice(),
    hsym_xyz, masses: Float64Array.from(masses),
    cellIdx: new Float64Array(natoms * 3),
  };

  // k-path metadata: continuous path (one segment) + high-symmetry labels.
  const hsymIndex = {};
  for (const [idx, label] of (obj.highsym_qpts || [])) hsymIndex[idx] = label === 'GAMMA' ? 'Γ' : label;

  return {
    bands, eigvecs, qPoints: qpoints,
    kpathMeta: { segSizes: [qpoints.length], hsymIndex },
    temperature: 300, source: 'file',
    baseStructure,
  };
}

// Tiny helper so the band-energy assignment reads clearly above.
function bandsSet(row, m, v) { row[m] = v; }

/**
 * Load a viewer model from raw file text, auto-detecting the format:
 * a phonopy dynamical-matrix JSON → compute bands here; anything else
 * (band.yaml / band.json) → the standard `fromBandText` reader.
 */
export function modelFromText(text) {
  let obj = null;
  try { obj = JSON.parse(text); } catch { /* not JSON → treat as a band.yaml */ }
  if (obj && isPhonopyDM(obj)) return fromPhonopyDM(obj);
  return fromBandText(text);
}
