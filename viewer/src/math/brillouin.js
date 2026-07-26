// viewer/src/math/brillouin.js
//
// First Brillouin zone = Wigner-Seitz cell of the reciprocal lattice: the set of
// points closer to the origin than to any other reciprocal lattice point. We
// build it as the intersection of half-spaces G·x ≤ |G|²/2 (perpendicular
// bisector of each reciprocal lattice vector G), then extract vertices, faces
// and edges for rendering. General — works for any (primitive) reciprocal cell.

import { mat3Inverse } from './reciprocal.js';

/**
 * @param {number[][]} B reciprocal lattice rows (b1,b2,b3), cartesian.
 * @returns {{vertices:number[][], faces:number[][][], edges:number[][][]}}
 */
export function brillouinZone(B) {
  // Candidate reciprocal lattice vectors (skip 0). ±2 covers second shell.
  const G = [];
  for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) for (let k = -2; k <= 2; k++) {
    if (i === 0 && j === 0 && k === 0) continue;
    G.push([i * B[0][0] + j * B[1][0] + k * B[2][0], i * B[0][1] + j * B[1][1] + k * B[2][1], i * B[0][2] + j * B[1][2] + k * B[2][2]]);
  }
  const gmin = Math.min(...G.map(g => Math.hypot(...g)));
  // Only near planes can bound the WS cell.
  const planes = G.filter(g => Math.hypot(...g) <= 2.2 * gmin)
    .map(g => ({ n: g, d: (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) / 2 }));
  const np = planes.length;

  const inside = (x, tol = 1e-6) => planes.every(p => p.n[0] * x[0] + p.n[1] * x[1] + p.n[2] * x[2] <= p.d + tol);

  // Vertices = intersections of plane triples that satisfy all constraints.
  const verts = [];
  for (let a = 0; a < np; a++) for (let b = a + 1; b < np; b++) for (let c = b + 1; c < np; c++) {
    const m = [planes[a].n, planes[b].n, planes[c].n];
    let inv;
    try { inv = mat3Inverse(m); } catch { continue; }
    const rhs = [planes[a].d, planes[b].d, planes[c].d];
    const x = [
      inv[0][0] * rhs[0] + inv[0][1] * rhs[1] + inv[0][2] * rhs[2],
      inv[1][0] * rhs[0] + inv[1][1] * rhs[1] + inv[1][2] * rhs[2],
      inv[2][0] * rhs[0] + inv[2][1] * rhs[1] + inv[2][2] * rhs[2],
    ];
    if (!isFinite(x[0]) || !isFinite(x[1]) || !isFinite(x[2])) continue;
    if (inside(x)) verts.push(x);
  }

  // Dedupe vertices.
  const uniq = [];
  for (const v of verts) {
    if (!uniq.some(u => Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]) < 1e-4)) uniq.push(v);
  }

  // Faces: vertices lying on each bounding plane, ordered around its normal.
  const faces = [];
  const edgeSet = new Set();
  const edges = [];
  for (const p of planes) {
    const on = uniq.filter(v => Math.abs(p.n[0] * v[0] + p.n[1] * v[1] + p.n[2] * v[2] - p.d) < 1e-4);
    if (on.length < 3) continue;
    // Order by angle in the plane.
    const c = [0, 0, 0];
    for (const v of on) { c[0] += v[0]; c[1] += v[1]; c[2] += v[2]; }
    c[0] /= on.length; c[1] /= on.length; c[2] /= on.length;
    const nrm = Math.hypot(...p.n);
    const nhat = [p.n[0] / nrm, p.n[1] / nrm, p.n[2] / nrm];
    let ref = [on[0][0] - c[0], on[0][1] - c[1], on[0][2] - c[2]];
    const rl = Math.hypot(...ref); ref = ref.map(x => x / rl);
    const u2 = [nhat[1] * ref[2] - nhat[2] * ref[1], nhat[2] * ref[0] - nhat[0] * ref[2], nhat[0] * ref[1] - nhat[1] * ref[0]];
    const ang = (v) => {
      const dx = [v[0] - c[0], v[1] - c[1], v[2] - c[2]];
      return Math.atan2(dx[0] * u2[0] + dx[1] * u2[1] + dx[2] * u2[2], dx[0] * ref[0] + dx[1] * ref[1] + dx[2] * ref[2]);
    };
    const ordered = on.slice().sort((p1, p2) => ang(p1) - ang(p2));
    faces.push(ordered);
    for (let i = 0; i < ordered.length; i++) {
      const a = ordered[i], b = ordered[(i + 1) % ordered.length];
      const key = [a, b].map(p => p.map(x => Math.round(x * 1e4)).join(',')).sort().join('|');
      if (!edgeSet.has(key)) { edgeSet.add(key); edges.push([a, b]); }
    }
  }

  return { vertices: uniq, faces, edges };
}
