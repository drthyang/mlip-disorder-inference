// viewer/src/compute/ins.js
//
// Build a compact, transferable representation of the phonon results for the
// S(Q,E)/DOS worker.
//
// Key point: the worker only needs the per-mode neutron structure factor
//   F2(q,mode) = Σ_site (b_coh[el]² / mass[el]) · |e_site|²
// — a SINGLE scalar per mode, not the full eigenvector. So we precompute F2 here
// (tiny: nq·nModes) instead of shipping nq·nModes·nSites·6 floats. This both
// avoids the large-buffer allocation that crashed the tab and cuts transfer cost.

import { ATOMIC_MASS, B_COH } from '../constants';
import { conventionalLattice, reciprocalLattice } from '../math/reciprocal';

export function buildInsData(results) {
  const bs = results.baseStructure;
  const { atomDic, dim, v1, v2, v3, uniqueRN } = bs;
  const recip = reciprocalLattice(conventionalLattice(v1, v2, v3, dim));

  // Basis sites with eigenvector row + neutron weight b²/m. Eigenvector rows are
  // τ-ordered (one per computation-cell basis site), so build the site list from
  // the τ-ordered element list when available. The legacy atomDic RN loop only
  // covers the conventional P = I grouping — with a primitive computation cell
  // its RN→row lookups miss and would produce NaN weights.
  const sites = [];
  if (bs.segSymbols && bs.segSymbols.length) {
    bs.segSymbols.forEach((el, r) => {
      const b = B_COH[el] || 0;
      const m = ATOMIC_MASS[el] || 0;
      sites.push({ row: r, b2: m > 0 ? (b * b) / m : 0 });
    });
  } else {
    const rnToRow = new Map((uniqueRN || []).map((rn, r) => [rn, r]));
    for (const el of Object.keys(atomDic)) for (const rn of atomDic[el]) {
      const row = rnToRow.get(rn);
      if (row == null) continue;
      const b = B_COH[el] || 0;
      const m = ATOMIC_MASS[el] || 0;
      sites.push({ row, b2: m > 0 ? (b * b) / m : 0 });
    }
  }
  const nSites = sites.length;

  const nq = results.qPoints.length;
  const nModes = results.bands[0].length;

  const freqs = new Float64Array(nq * nModes);
  const qpos = new Float64Array(nq * 3);
  const F2 = new Float64Array(nq * nModes);

  for (let qi = 0; qi < nq; qi++) {
    qpos[qi * 3] = results.qPoints[qi][0];
    qpos[qi * 3 + 1] = results.qPoints[qi][1];
    qpos[qi * 3 + 2] = results.qPoints[qi][2];
    for (let m = 0; m < nModes; m++) {
      freqs[qi * nModes + m] = results.bands[qi][m];
      const ev = results.eigvecs[qi][m];
      const re = ev.real, im = ev.imag;
      let f2 = 0;
      for (let s = 0; s < nSites; s++) {
        const b2 = sites[s].b2;
        if (b2 === 0) continue;
        const r = sites[s].row * 3;
        f2 += b2 * (re[r] * re[r] + im[r] * im[r]
          + re[r + 1] * re[r + 1] + im[r + 1] * im[r + 1]
          + re[r + 2] * re[r + 2] + im[r + 2] * im[r + 2]);
      }
      F2[qi * nModes + m] = f2;
    }
  }

  return {
    data: { nq, nModes, freqs, qpos, F2, recip },
    transfer: [freqs.buffer, qpos.buffer, F2.buffer],
  };
}
