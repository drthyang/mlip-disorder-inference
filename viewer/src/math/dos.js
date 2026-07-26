// viewer/src/math/dos.js
//
// Phonon density of states from a flat list of mode energies (meV) sampled over
// a uniform q-grid: Gaussian-broadened histogram, normalized to unit area.
// Pure + testable; the heavy part (eigenvalues over the grid) is the pipeline.

export function phononDOS(energies, { sigma = 1.0, Emin = 0, Emax = 50, nE = 300 } = {}) {
  const E = new Float64Array(nE);
  const dos = new Float64Array(nE);
  const dE = (Emax - Emin) / (nE - 1);
  for (let i = 0; i < nE; i++) E[i] = Emin + i * dE;

  const norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const inv2s2 = 0.5 / (sigma * sigma);
  const cutoff = 4 * sigma;
  let count = 0;
  for (let k = 0; k < energies.length; k++) {
    const w = energies[k];
    if (!Number.isFinite(w) || Math.abs(w) < 1e-6) continue;  // skip ~zero/acoustic-at-Γ artifacts
    count++;
    const lo = Math.max(0, Math.floor((w - cutoff - Emin) / dE));
    const hi = Math.min(nE - 1, Math.ceil((w + cutoff - Emin) / dE));
    for (let i = lo; i <= hi; i++) {
      const d = E[i] - w;
      dos[i] += norm * Math.exp(-d * d * inv2s2);
    }
  }
  if (count > 0) { const inv = 1 / count; for (let i = 0; i < nE; i++) dos[i] *= inv; }
  let dosMax = 0;
  for (let i = 0; i < nE; i++) if (dos[i] > dosMax) dosMax = dos[i];
  return { E, dos, dosMax, count };
}
