// viewer/src/io/viewermodel.js
//
// A single "viewer model" feeds the Viewer page, whether the data came from an
// in-memory runner result or a loaded band.yaml/.json. It is always a UNIT-CELL
// representation (one cell, one atom per basis site); the 3D viewer tiles it by
// the user's supercell nx,ny,nz. Shape:
//
//   { bands, eigvecs, qPoints, kpathMeta, temperature, source,
//     baseStructure: { atomDic, dim:[1,1,1], v1,v2,v3, uniqueRN, atomType,
//                      hsym_xyz, cellIdx } }
//
// eigvecs[q][m] = { real, imag } (length 3*nSites) with rows ordered by
// uniqueRN (sorted reference numbers) — consistent with the diagonalizer.

import yaml from 'js-yaml';
import { conventionalLattice, reciprocalLattice } from '../math/reciprocal.js';
import { THZ_TO_MEV, ATOMIC_MASS } from '../constants.js';

// Strip phonopy/latex decorations from a path label: '$\\Gamma$' → 'Γ',
// '$\\mathrm{X}$' → 'X'.
function cleanLabel(s) {
  return String(s)
    .replace(/\\Gamma/g, 'Γ')
    .replace(/\\mathrm\{([^}]*)\}/g, '$1')
    .replace(/[${}\\]/g, '');
}

/** Convert in-memory pipeline results into the unit-cell viewer model. */
export function fromResults(results, kpathMeta) {
  const bs = results.baseStructure;
  // Unit cell shown in the 3D viewer = the COMPUTATION cell (L = P·A_conv). For
  // the default conventional cell this equals A_conv; for a custom/primitive cell
  // it is the chosen sub-/super-cell (siteBasis fractions are in L units).
  const Aconv = conventionalLattice(bs.v1, bs.v2, bs.v3, bs.dim);
  const A = (bs.compCell && bs.compCell.L) || Aconv;
  // The band-path q-points are in the COMPUTATION cell's own reciprocal coords, so
  // the plot's x-axis distances use that cell's reciprocal (= reciprocal of A).
  const bandRecip = reciprocalLattice(A);

  // The S(k) rows / eigenvector components are ordered by basis site τ. Prefer
  // the τ-ordered `siteBasis` (cell-framework) so site r ↔ eigvec row r exactly;
  // fall back to the per-reference-number layout for older results.
  let natom, hsym, atomType, atomDic, uniqueRN, masses;
  if (bs.siteBasis && bs.siteBasis.length) {
    natom = bs.siteBasis.length;
    hsym = new Float64Array(natom * 3);
    masses = new Float64Array(natom);
    atomType = [];
    atomDic = {};
    for (let r = 0; r < natom; r++) {
      const f = bs.siteBasis[r].frac;
      for (let c = 0; c < 3; c++) hsym[r * 3 + c] = ((f[c] % 1) + 1) % 1;
      atomType.push(r + 1);
      const el = bs.siteBasis[r].element || 'X';
      (atomDic[el] || (atomDic[el] = [])).push(r + 1);
      masses[r] = ATOMIC_MASS[el] || 0;
    }
    uniqueRN = atomType.slice();
  } else {
    uniqueRN = bs.uniqueRN;
    natom = uniqueRN.length;
    const rnFirst = new Map();
    for (let i = 0; i < bs.atomType.length; i++) {
      if (!rnFirst.has(bs.atomType[i])) rnFirst.set(bs.atomType[i], i);
    }
    hsym = new Float64Array(natom * 3);
    for (let r = 0; r < natom; r++) {
      const ai = rnFirst.get(uniqueRN[r]);
      for (let c = 0; c < 3; c++) hsym[r * 3 + c] = ((bs.hsym_xyz[ai * 3 + c] % 1) + 1) % 1;
    }
    atomDic = bs.atomDic;
    atomType = Array.from(uniqueRN);
    masses = new Float64Array(natom);
    const rnToEl = {};
    for (const [el, rns] of Object.entries(atomDic || {})) for (const rn of rns) rnToEl[rn] = el;
    for (let r = 0; r < natom; r++) masses[r] = ATOMIC_MASS[rnToEl[uniqueRN[r]]] || 0;
  }

  return {
    bands: results.bands,
    eigvecs: results.eigvecs,
    qPoints: results.qPoints,
    kpathMeta,
    temperature: results.temperature,
    source: 'runner',
    baseStructure: {
      atomDic, dim: [1, 1, 1],
      v1: A[0], v2: A[1], v3: A[2], bandRecip,
      uniqueRN, atomType,
      hsym_xyz: hsym, masses, cellIdx: new Float64Array(natom * 3),
    },
  };
}

/** Parse a band.yaml or band.json string into the viewer model.
 *
 * UNITS: phonopy's band.yaml stores `frequency` in THz (its default factor
 * sqrt(eV/amu)/Å/2π = 15.6333); the viewer model is meV throughout, so by
 * default frequencies are converted ×4.1356677 on load. A `frequency_unit:
 * meV|THz` key in the file (written by this app's own exporter) overrides
 * the default; the `unit` option overrides both.
 *
 * EIGENVECTORS are kept exactly as stored — phonopy's convention: unit-norm
 * eigenvectors of the MASS-WEIGHTED dynamical matrix (displacement of atom
 * a ∝ e_a/√m_a). Consumers apply their own mass weighting; per-site masses
 * are carried in baseStructure.masses (points[].mass, else the mass table).
 */
export function fromBandText(text, { unit = null } = {}) {
  let doc;
  try { doc = JSON.parse(text); } catch { doc = yaml.load(text); }
  if (!doc || !doc.phonon) throw new Error('Not a band.yaml/.json (no "phonon" block).');

  const points = doc.points || [];
  const natom = points.length;
  const lattice = doc.lattice || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  // Basis sites: RN = index+1, grouped into atomDic by symbol.
  const atomDic = {};
  const atomType = [];
  const hsym = new Float64Array(natom * 3);
  const masses = new Float64Array(natom);
  for (let i = 0; i < natom; i++) {
    const sym = points[i].symbol || 'X';
    (atomDic[sym] || (atomDic[sym] = [])).push(i + 1);
    atomType.push(i + 1);
    const c = points[i].coordinates || [0, 0, 0];
    hsym[i * 3] = c[0]; hsym[i * 3 + 1] = c[1]; hsym[i * 3 + 2] = c[2];
    masses[i] = Number(points[i].mass) || ATOMIC_MASS[sym] || 0;
  }
  const uniqueRN = atomType.slice();

  const fileUnit = unit || doc.frequency_unit || 'THz';
  const conv = /mev/i.test(String(fileUnit)) ? 1 : THZ_TO_MEV;
  const qPoints = [];
  const bands = [];
  const eigvecs = [];
  const hsymIndex = {};
  let hasEig = true;
  for (let qi = 0; qi < doc.phonon.length; qi++) {
    const p = doc.phonon[qi];
    qPoints.push(p['q-position'] || [0, 0, 0]);
    if (p.label) hsymIndex[qi] = String(p.label).replace(/\$|\\/g, '');
    const bandRow = [];
    const evRow = [];
    for (const mode of (p.band || [])) {
      bandRow.push((mode.frequency || 0) * conv);
      if (mode.eigenvector) {
        const real = new Float64Array(3 * natom);
        const imag = new Float64Array(3 * natom);
        for (let a = 0; a < natom; a++) {
          const ea = mode.eigenvector[a];
          for (let c = 0; c < 3; c++) { real[a * 3 + c] = ea[c][0]; imag[a * 3 + c] = ea[c][1]; }
        }
        evRow.push({ real, imag });
      } else { hasEig = false; evRow.push(null); }
    }
    bands.push(bandRow);
    eigvecs.push(evRow);
  }

  const segSizes = doc.segment_nqpoint || [qPoints.length];

  // Path labels: phonopy writes a top-level `labels:` list of [start, end]
  // per segment (no per-point label keys) — map them onto segment boundaries.
  if (Array.isArray(doc.labels)) {
    let off = 0;
    for (let s = 0; s < segSizes.length; s++) {
      const pair = doc.labels[s];
      if (pair && pair.length === 2) {
        if (hsymIndex[off] === undefined) hsymIndex[off] = cleanLabel(pair[0]);
        hsymIndex[off + segSizes[s] - 1] = cleanLabel(pair[1]);
      }
      off += segSizes[s];
    }
  }

  return {
    bands, eigvecs: hasEig ? eigvecs : null, qPoints,
    kpathMeta: { segSizes, hsymIndex },
    temperature: 5, source: 'file', hasEig,
    baseStructure: {
      atomDic, dim: [1, 1, 1],
      v1: lattice[0], v2: lattice[1], v3: lattice[2],
      uniqueRN, atomType, hsym_xyz: hsym, masses,
      cellIdx: new Float64Array(natom * 3),
    },
  };
}
