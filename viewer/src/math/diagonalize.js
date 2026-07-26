// viewer/src/math/diagonalize.js
import { Matrix, EigenvalueDecomposition } from 'ml-matrix';

// ENERGY_CONV is imported by callers from ../constants.js and passed into
// eigenvaluesToMev(); it is intentionally NOT redefined here.

/**
 * Diagonalize a complex Hermitian matrix (A + iB) by constructing a 2N x 2N real symmetric matrix.
 * Returns eigenvalues and complex eigenvectors.
 * 
 * @param {Float64Array} Sk_real - A, flat array of size NxN
 * @param {Float64Array} Sk_imag - B, flat array of size NxN
 * @param {number} N - Dimension of the original matrix
 */
export function eigh(Sk_real, Sk_imag, N) {
  // Construct 2N x 2N matrix M = [A, -B; B, A]
  const M = new Matrix(2 * N, 2 * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const a = Sk_real[i * N + j];
      const b = Sk_imag[i * N + j];
      
      M.set(i, j, a);                 // Top-left: A
      M.set(i, j + N, -b);            // Top-right: -B
      M.set(i + N, j, b);             // Bottom-left: B
      M.set(i + N, j + N, a);         // Bottom-right: A
    }
  }
  
  // Symmetrize to fix tiny float32 accumulation asymmetries
  for (let i = 0; i < 2 * N; i++) {
    for (let j = i + 1; j < 2 * N; j++) {
      const val = (M.get(i, j) + M.get(j, i)) / 2;
      M.set(i, j, val);
      M.set(j, i, val);
    }
  }

  const evd = new EigenvalueDecomposition(M);
  const realEigenvalues = evd.realEigenvalues;
  const eigenvectorsMatrix = evd.eigenvectorMatrix;

  // The eigenvalues appear in identical pairs (λ, λ): every complex eigenvector
  // v shows up twice in the real embedding, as (x, y) and (−y, x) = i·v. Sort by
  // eigenvalue; the N unique eigenvalues are every other sorted value.
  const paired = [];
  for (let i = 0; i < 2 * N; i++) {
    paired.push({ val: realEigenvalues[i], col: i });
  }
  paired.sort((a, b) => a.val - b.val);

  const finalEigenvalues = new Float64Array(N);
  for (let k = 0; k < N; k++) finalEigenvalues[k] = paired[2 * k].val;

  // Eigenvector extraction. Naively mapping columns 2k to complex vectors x + iy
  // breaks for DEGENERATE eigenvalues: the real solver returns an arbitrary real
  // orthonormal basis of the 2g-dimensional real subspace, and two chosen columns
  // can map to the SAME complex vector (v and i·v — real-orthogonal but complex-
  // linearly dependent). So: cluster near-equal eigenvalues, then complex
  // Gram-Schmidt over each cluster's columns, keeping the first g independent
  // complex vectors. For non-degenerate spectra this reduces to the naive pick.
  const scale = Math.max(Math.abs(paired[0].val), Math.abs(paired[2 * N - 1].val), 1e-300);
  const clusterTol = 1e-7 * scale;
  const bounds = [];
  let clusterStart = 0;
  for (let i = 1; i <= 2 * N; i++) {
    if (i === 2 * N || paired[i].val - paired[i - 1].val > clusterTol) { bounds.push([clusterStart, i]); clusterStart = i; }
  }
  // Every complex eigenvalue contributes exactly 2 real copies, so clusters must
  // have even size; an odd one means the tolerance split a true pair — re-join.
  for (let i = 0; i < bounds.length - 1; i++) {
    if ((bounds[i][1] - bounds[i][0]) % 2 === 1) { bounds[i][1] = bounds[i + 1][1]; bounds.splice(i + 1, 1); i--; }
  }

  const finalEigenvectors = [];
  const colToComplex = (col) => {
    // For M v = λ v with v = [x; y], the complex eigenvector is x + iy.
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < N; i++) { re[i] = eigenvectorsMatrix.get(i, col); im[i] = eigenvectorsMatrix.get(i + N, col); }
    return { re, im };
  };
  for (const [a, b] of bounds) {
    const g = (b - a) / 2;
    const accepted = [];
    for (let c = a; c < b && accepted.length < g; c++) {
      const { re: vr, im: vi } = colToComplex(paired[c].col);
      for (const u of accepted) {
        // v -= <u, v> u  (complex projection; <u, v> = Σ conj(u)·v)
        let pr = 0, pi = 0;
        for (let i = 0; i < N; i++) { pr += u.real[i] * vr[i] + u.imag[i] * vi[i]; pi += u.real[i] * vi[i] - u.imag[i] * vr[i]; }
        for (let i = 0; i < N; i++) { vr[i] -= pr * u.real[i] - pi * u.imag[i]; vi[i] -= pr * u.imag[i] + pi * u.real[i]; }
      }
      let normSq = 0;
      for (let i = 0; i < N; i++) normSq += vr[i] * vr[i] + vi[i] * vi[i];
      if (normSq > 1e-8) {
        const inv = 1 / Math.sqrt(normSq);
        for (let i = 0; i < N; i++) { vr[i] *= inv; vi[i] *= inv; }
        accepted.push({ real: vr, imag: vi });
      }
    }
    // Numerical fallback (a full complex span guarantees g survivors; keep the
    // legacy behaviour rather than dropping a mode if that ever fails).
    for (let c = a; accepted.length < g && c < b; c += 2) {
      const { re: vr, im: vi } = colToComplex(paired[c].col);
      let normSq = 0;
      for (let i = 0; i < N; i++) normSq += vr[i] * vr[i] + vi[i] * vi[i];
      const inv = 1 / Math.sqrt(Math.max(normSq, 1e-12));
      for (let i = 0; i < N; i++) { vr[i] *= inv; vi[i] *= inv; }
      accepted.push({ real: vr, imag: vi });
    }
    for (const v of accepted) finalEigenvectors.push(v);
  }

  return { eigenvalues: finalEigenvalues, eigenvectors: finalEigenvectors };
}

/**
 * Convert S(k) eigenvalues [amu A^2] to phonon energies [meV].
 * Mirrors python: E = ENERGY_CONV * sqrt(T / lambda)
 */
export function eigenvaluesToMev(eigenvalues, tempK, energyConv) {
  const result = new Float64Array(eigenvalues.length);
  const threshold = 1e-4;
  
  for (let i = 0; i < eigenvalues.length; i++) {
    const ev = eigenvalues[i];
    const absEv = Math.abs(ev);
    
    if (absEv < threshold) {
      result[i] = 0.0;
    } else {
      let energy = energyConv * Math.sqrt(tempK / absEv);
      if (isNaN(energy)) energy = 0.0;
      
      // Soft modes (ev < 0) returned as negative energies
      result[i] = ev >= 0 ? energy : -energy;
    }
  }
  return result;
}
