// viewer/test/diagonalize_test.mjs
//
// Eigenvector correctness of the complex-Hermitian eigh under DEGENERACY.
//
// The 2N real-symmetric embedding [A, −B; B, A] doubles every complex
// eigenvector v as (x, y) and (−y, x) = i·v. For a g-fold degenerate complex
// eigenvalue the real solver returns an arbitrary real basis of the 2g-dim
// subspace, and naively taking every other sorted column can yield v and i·v —
// the SAME complex eigenvector twice (this actually happened: |<v1,v2>| = 1).
// eigh must return complex-orthonormal eigenvectors that each satisfy the
// eigen-equation. Degeneracies are the norm once S(k) symmetrization is on.

import { eigh } from '../src/math/diagonalize.js';

let fails = 0;
const ok = (c, m) => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${m}`); if (!c) fails++; };

// H = Σ_k λ_k v_k v_k† from an explicit complex-orthonormal basis.
function buildH(N, lam, cols) {
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  for (let k = 0; k < N; k++) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    re[i * N + j] += lam[k] * (cols[k].re[i] * cols[k].re[j] + cols[k].im[i] * cols[k].im[j]);
    im[i * N + j] += lam[k] * (cols[k].im[i] * cols[k].re[j] - cols[k].re[i] * cols[k].im[j]);
  }
  return { re, im };
}
const dotc = (u, v, N) => {
  let re = 0, im = 0;
  for (let i = 0; i < N; i++) { re += u.real[i] * v.real[i] + u.imag[i] * v.imag[i]; im += u.real[i] * v.imag[i] - u.imag[i] * v.real[i]; }
  return Math.hypot(re, im);
};
// max |H v − λ v| over components
function eigResidual(H, N, lam, v) {
  let worst = 0;
  for (let i = 0; i < N; i++) {
    let re = 0, im = 0;
    for (let j = 0; j < N; j++) {
      re += H.re[i * N + j] * v.real[j] - H.im[i * N + j] * v.imag[j];
      im += H.re[i * N + j] * v.imag[j] + H.im[i * N + j] * v.real[j];
    }
    worst = Math.max(worst, Math.abs(re - lam * v.real[i]), Math.abs(im - lam * v.imag[i]));
  }
  return worst;
}

function checkSpectrum(name, N, lam, cols) {
  const H = buildH(N, lam, cols);
  const { eigenvalues, eigenvectors } = eigh(H.re, H.im, N);
  const want = [...lam].sort((a, b) => a - b);
  let evErr = 0;
  for (let i = 0; i < N; i++) evErr = Math.max(evErr, Math.abs(eigenvalues[i] - want[i]));
  ok(evErr < 1e-9, `${name}: eigenvalues correct (|Δ| = ${evErr.toExponential(1)})`);
  let maxOff = 0, minDiag = 1;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const d = dotc(eigenvectors[i], eigenvectors[j], N);
    if (i === j) minDiag = Math.min(minDiag, d); else maxOff = Math.max(maxOff, d);
  }
  ok(maxOff < 1e-8 && minDiag > 1 - 1e-8, `${name}: eigenvectors complex-orthonormal (max off-diag |<vi,vj>| = ${maxOff.toExponential(1)})`);
  let res = 0;
  for (let i = 0; i < N; i++) res = Math.max(res, eigResidual(H, N, eigenvalues[i], eigenvectors[i]));
  ok(res < 1e-8, `${name}: every vector satisfies H v = λ v (max residual ${res.toExponential(1)})`);
}

console.log('\nDouble degeneracy (3×3, λ = 1,1,3, complex basis):');
{
  const s2 = 1 / Math.sqrt(2);
  const cols = [
    { re: [s2, 0.5, 0], im: [0, 0.5, 0] },
    { re: [-0.5, s2, 0], im: [0.5, 0, 0] },
    { re: [0, 0, 1], im: [0, 0, 0] },
  ];
  checkSpectrum('3×3 double', 3, [1, 1, 3], cols);
}

console.log('\nTriple degeneracy (4×4, λ = 2,2,2,5):');
{
  // complex-orthonormal 4×4 basis via two complex Givens-style mixes
  const s2 = 1 / Math.sqrt(2);
  const cols = [
    { re: [s2, 0, 0.5, 0], im: [0, 0.5, 0, 0] },
    { re: [0, s2, 0, 0.5], im: [-0.5, 0, 0, 0] },
    { re: [-0.5, 0, s2, 0], im: [0, 0, 0, 0.5] },
    { re: [0, -0.5, 0, s2], im: [0, 0, -0.5, 0] },
  ];
  // Gram-Schmidt the hand-built columns so H is exactly what we claim.
  const N = 4;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < i; j++) {
      let pr = 0, pi = 0;
      for (let n = 0; n < N; n++) { pr += cols[j].re[n] * cols[i].re[n] + cols[j].im[n] * cols[i].im[n]; pi += cols[j].re[n] * cols[i].im[n] - cols[j].im[n] * cols[i].re[n]; }
      for (let n = 0; n < N; n++) { cols[i].re[n] -= pr * cols[j].re[n] - pi * cols[j].im[n]; cols[i].im[n] -= pr * cols[j].im[n] + pi * cols[j].re[n]; }
    }
    let n2 = 0;
    for (let n = 0; n < N; n++) n2 += cols[i].re[n] ** 2 + cols[i].im[n] ** 2;
    const inv = 1 / Math.sqrt(n2);
    for (let n = 0; n < N; n++) { cols[i].re[n] *= inv; cols[i].im[n] *= inv; }
  }
  checkSpectrum('4×4 triple', 4, [2, 2, 2, 5], cols);
}

console.log('\nNon-degenerate regression (4×4 random Hermitian):');
{
  let s = 42; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
  const N = 4;
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = i; j < N; j++) {
    const a = rnd(), b = i === j ? 0 : rnd();
    re[i * N + j] = a; re[j * N + i] = a; im[i * N + j] = b; im[j * N + i] = -b;
  }
  const { eigenvalues, eigenvectors } = eigh(re, im, N);
  let maxOff = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) maxOff = Math.max(maxOff, dotc(eigenvectors[i], eigenvectors[j], N));
  ok(maxOff < 1e-8, `orthonormal (max off-diag ${maxOff.toExponential(1)})`);
  let res = 0;
  for (let i = 0; i < N; i++) res = Math.max(res, eigResidual({ re, im }, N, eigenvalues[i], eigenvectors[i]));
  ok(res < 1e-8, `eigen-equation residual ${res.toExponential(1)}`);
}

console.log(`\n${fails === 0 ? '✅ diagonalize OK' : `❌ ${fails} failed`}`);
process.exit(fails ? 1 : 0);
