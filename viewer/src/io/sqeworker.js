// viewer/src/io/sqeworker.js
//
// Off-main-thread simulated INS: powder S(|Q|,E) and phonon DOS.
// Ported from viz/sqeworker.js. Operates on compact transferred typed arrays
// (see compute/ins.js): the per-mode structure factor F2 is precomputed on the
// main thread, so only nq*nModes scalars are shipped (not the full eigenvectors)
// — this avoids the large-buffer allocation that crashed the tab. Frequencies
// are meV throughout the viewer model: the RMC pipeline computes meV directly,
// and phonopy band.yaml (THz) is converted ×4.1356677 at load (viewermodel.js).
//
// Protocol: postMessage({ data, params }) -> { powResult, dosResult, timings }.
//   data = { nq, nModes, freqs, qpos, F2, recip }

const KB = 0.08617333;   // meV / K
const HBAR2_2MN = 2.0723; // meV * A^2

const LUT_N = 2048, LUT_RANGE = 5, LUT_INV_STEP = LUT_N / LUT_RANGE;
const GAUSS_LUT = new Float64Array(LUT_N + 2);
for (let i = 0; i <= LUT_N + 1; i++) { const x = i / LUT_INV_STEP; GAUSS_LUT[i] = Math.exp(-0.5 * x * x); }

function bose(omega, T) {
  if (T <= 0 || omega <= 0) return 0;
  const x = omega / (KB * T);
  return x > 100 ? 0 : 1.0 / (Math.exp(x) - 1.0);
}

// Deposit one q-point's modes (using precomputed structure factor F2) into one
// S(E) column (`grid`). freqs/F2 are indexed by base = qi*nModes.
function accumulateModes(F2arr, freqs, base, nModes, T, sigma, Emin, dE, nE, grid) {
  const cutoff = 4 * sigma, norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const cutoff_dE = cutoff / dE, inv_step = LUT_INV_STEP, lut = GAUSS_LUT, inv_sigma = 1 / sigma;

  for (let mi = 0; mi < nModes; mi++) {
    const omega = freqs[base + mi];
    if (Math.abs(omega) < 1e-6) continue;
    const F2 = F2arr[base + mi];
    if (F2 === 0) continue;

    const absOmega = Math.abs(omega);
    const occ = omega > 0 ? bose(absOmega, T) + 1 : bose(absOmega, T);
    const amp = (F2 * occ / absOmega) * norm;
    const iCenter = (omega - Emin) / dE;
    const iLo = Math.max(0, Math.floor(iCenter - cutoff_dE));
    const iHi = Math.min(nE - 1, Math.ceil(iCenter + cutoff_dE));
    for (let Ei = iLo; Ei <= iHi; Ei++) {
      let d = (Emin + Ei * dE - omega) * inv_sigma;
      if (d < 0) d = -d;
      if (d >= LUT_RANGE) continue;
      const fi = d * inv_step, i0 = fi | 0;
      grid[Ei] += amp * (lut[i0] + (fi - i0) * (lut[i0 + 1] - lut[i0]));
    }
  }
}

function computePowderSqE(D, T, sigma, Emin, Emax, nE, nQbins, Ei) {
  const { nq, nModes, freqs, qpos, F2, recip: rl } = D;

  const bv = rl.map(v => [2 * Math.PI * v[0], 2 * Math.PI * v[1], 2 * Math.PI * v[2]]);
  const bmag = bv.map(v => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));

  const ki_plot = Ei > 0 ? Math.sqrt(Ei / HBAR2_2MN) : 0;
  const Q_max_plot = Ei > 0 ? Math.max(1.5, 2 * ki_plot * 1.05)
    : Math.max(1.5, Math.sqrt(Math.max(Emax, 1) / HBAR2_2MN) * 1.2);
  const dQ = Q_max_plot / nQbins;

  const b_min = Math.min(...bmag) || 1;
  const sigQ = Math.max(1.5 * dQ, 0.4 * b_min), inv_sigQ = 1 / sigQ;

  const n_max = Math.min(20, Math.ceil(Q_max_plot / b_min) + 1);
  const G_reach = Q_max_plot + Math.max(...bmag);
  const G_buf = [];
  for (let n1 = -n_max; n1 <= n_max; n1++) for (let n2 = -n_max; n2 <= n_max; n2++) for (let n3 = -n_max; n3 <= n_max; n3++) {
    const Gx = n1 * bv[0][0] + n2 * bv[1][0] + n3 * bv[2][0];
    const Gy = n1 * bv[0][1] + n2 * bv[1][1] + n3 * bv[2][1];
    const Gz = n1 * bv[0][2] + n2 * bv[1][2] + n3 * bv[2][2];
    if (Math.sqrt(Gx * Gx + Gy * Gy + Gz * Gz) < G_reach) { G_buf.push(Gx, Gy, Gz); }
  }
  const nG = G_buf.length / 3;
  const Gflat = Float64Array.from(G_buf);

  const dE = (Emax - Emin) / (nE - 1);
  const Eaxis = new Float64Array(nE);
  for (let i = 0; i < nE; i++) Eaxis[i] = Emin + i * dE;
  const S = new Float64Array(nQbins * nE);
  const norm = new Float64Array(nQbins);
  const tmpS = new Float64Array(nE);
  const Qspan = Math.ceil(3.5 * sigQ / dQ);
  const wq_acc = new Float64Array(nQbins), wn_acc = new Float64Array(nQbins);
  const lut = GAUSS_LUT, inv_step = LUT_INV_STEP;

  for (let qi = 0; qi < nq; qi++) {
    tmpS.fill(0);
    accumulateModes(F2, freqs, qi * nModes, nModes, T, sigma, Emin, dE, nE, tmpS);

    const qx = 2 * Math.PI * (qpos[qi * 3] * rl[0][0] + qpos[qi * 3 + 1] * rl[1][0] + qpos[qi * 3 + 2] * rl[2][0]);
    const qy = 2 * Math.PI * (qpos[qi * 3] * rl[0][1] + qpos[qi * 3 + 1] * rl[1][1] + qpos[qi * 3 + 2] * rl[2][1]);
    const qz = 2 * Math.PI * (qpos[qi * 3] * rl[0][2] + qpos[qi * 3 + 1] * rl[1][2] + qpos[qi * 3 + 2] * rl[2][2]);

    wq_acc.fill(0); wn_acc.fill(0);
    for (let gi = 0; gi < nG; gi++) {
      const Qvx = qx + Gflat[3 * gi], Qvy = qy + Gflat[3 * gi + 1], Qvz = qz + Gflat[3 * gi + 2];
      const Q = Math.sqrt(Qvx * Qvx + Qvy * Qvy + Qvz * Qvz);
      if (Q <= 0 || Q > Q_max_plot) continue;
      const Q2 = Q * Q, Qi_f = Q / dQ;
      const Qi_lo = Math.max(0, Math.floor(Qi_f - Qspan));
      const Qi_hi = Math.min(nQbins - 1, Math.ceil(Qi_f + Qspan));
      for (let Qi = Qi_lo; Qi <= Qi_hi; Qi++) {
        let d = ((Qi + 0.5) * dQ - Q) * inv_sigQ;
        if (d < 0) d = -d;
        if (d >= LUT_RANGE) continue;
        const fi = d * inv_step, i0 = fi | 0;
        const wG = lut[i0] + (fi - i0) * (lut[i0 + 1] - lut[i0]);
        wq_acc[Qi] += Q2 * wG; wn_acc[Qi] += wG;
      }
    }
    for (let Qi = 0; Qi < nQbins; Qi++) {
      if (wq_acc[Qi] === 0) continue;
      norm[Qi] += wn_acc[Qi];
      const base = Qi * nE, w = wq_acc[Qi];
      for (let e = 0; e < nE; e++) S[base + e] += w * tmpS[e];
    }
  }

  for (let Qi = 0; Qi < nQbins; Qi++) {
    if (norm[Qi] > 0) { const base = Qi * nE, invN = 1 / norm[Qi]; for (let e = 0; e < nE; e++) S[base + e] *= invN; }
  }
  let Smax = 0;
  for (let i = 0; i < S.length; i++) if (S[i] > Smax) Smax = S[i];
  return { S, nX: nQbins, nE, Eaxis, Smax, xMin: 0, xMax: Q_max_plot, Ei: Ei > 0 ? Ei : 0 };
}

function computePhononDOS(D, sigma, Emin, Emax, nE) {
  const { nq, nModes, freqs } = D;
  const dE = (Emax - Emin) / (nE - 1);
  const halfWidth = Math.ceil(4 * sigma / dE), kLen = 2 * halfWidth + 1, pad = halfWidth;
  const histLen = nE + 2 * pad, hist = new Float64Array(histLen);
  let count = 0;
  for (let i = 0; i < nq * nModes; i++) {
    const omega = freqs[i];
    if (Math.abs(omega) < 1e-6) continue;
    count++;
    const fpos = (omega - Emin) / dE + pad, i0 = Math.floor(fpos), fr = fpos - i0;
    if (i0 >= 0 && i0 < histLen) hist[i0] += (1 - fr);
    if (i0 + 1 >= 0 && i0 + 1 < histLen) hist[i0 + 1] += fr;
  }
  const kernel = new Float64Array(kLen), knorm = 1 / (sigma * Math.sqrt(2 * Math.PI)), inv2s2 = 0.5 / (sigma * sigma);
  for (let i = 0; i < kLen; i++) { const x = (i - halfWidth) * dE; kernel[i] = knorm * Math.exp(-x * x * inv2s2); }
  const dos = new Float64Array(nE);
  for (let i = 0; i < histLen; i++) {
    const hi = hist[i]; if (hi === 0) continue;
    const jBase = i - pad - halfWidth, kLo = Math.max(0, -jBase), kHi = Math.min(kLen - 1, nE - 1 - jBase);
    for (let k = kLo; k <= kHi; k++) dos[jBase + k] += hi * kernel[k];
  }
  if (count > 0) { const inv = 1 / count; for (let i = 0; i < nE; i++) dos[i] *= inv; }
  let dosMax = 0;
  for (let i = 0; i < nE; i++) if (dos[i] > dosMax) dosMax = dos[i];
  return { dos, nE, Emin, Emax, dosMax };
}

// Exported for testing/debugging (test/ins_debug.mjs). The worker hook below is
// only registered in a real Worker context (guarded so Node imports are safe).
export { computePowderSqE, computePhononDOS };

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = (ev) => {
    const { data, params } = ev.data;
    try {
      const t0 = performance.now();
      const powResult = computePowderSqE(data, params.T, params.sigma, params.Emin, params.Emax, params.nE, params.nQbins, params.Ei);
      const t1 = performance.now();
      const dosResult = computePhononDOS(data, params.sigma, params.Emin, params.Emax, params.nE);
      const t2 = performance.now();
      const transfer = [];
      if (powResult) transfer.push(powResult.S.buffer, powResult.Eaxis.buffer);
      if (dosResult) transfer.push(dosResult.dos.buffer);
      self.postMessage({ success: true, powResult, dosResult, timings: { pow: t1 - t0, dos: t2 - t0 } }, transfer);
    } catch (err) {
      self.postMessage({ success: false, error: (err && err.message) || String(err) });
    }
  };
}
