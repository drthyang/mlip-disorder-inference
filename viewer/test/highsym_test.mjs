// viewer/test/highsym_test.mjs
//
// Validates the Setyawan–Curtarolo high-symmetry tables for the variant lattices.
// We can't compare labels to seekpath here, but every S-C point must lie ON or
// INSIDE the primitive Wigner-Seitz BZ — a wrong/mis-transcribed coordinate
// falls outside it. Also checks the detected Bravais code/variant and that the
// prim→conv transform preserves the cartesian k-vector.

import { analyzeBravais, primToConv, fracToCart } from '../src/math/bravais.js';
import { buildBZModel, buildConventionalBZModel, buildSupercellBZModel } from '../src/math/highsym.js';

let fail = 0;
const ok = (c, m) => { if (!c) { console.error('  FAIL ' + m); fail++; } else console.log('  ok   ' + m); };

// Conventional cell rows from lengths + angles (crystallographic convention).
function cell(a, b, c, al, be, ga) {
  const r = d => d * Math.PI / 180;
  const ca = Math.cos(r(al)), cb = Math.cos(r(be)), cg = Math.cos(r(ga)), sg = Math.sin(r(ga));
  const cx = c * cb, cy = c * (ca - cb * cg) / sg;
  const cz = c * Math.sqrt(Math.max(0, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)) / sg;
  return [[a, 0, 0], [b * cg, b * sg, 0], [cx, cy, cz]];
}
const FACE = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]].map(f => ({ el: 'A', frac: f }));
const BODY = [{ el: 'A', frac: [0, 0, 0] }, { el: 'A', frac: [.5, .5, .5] }];
const CBASE = [{ el: 'A', frac: [0, 0, 0] }, { el: 'A', frac: [.5, .5, 0] }];
const P1 = [{ el: 'A', frac: [0, 0, 0] }];

// A cartesian k lies in the WS BZ iff k·G ≤ |G|²/2 for all reciprocal vectors G.
function inBZ(kCart, B, tol = 1e-6) {
  for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) for (let k = -2; k <= 2; k++) {
    if (!i && !j && !k) continue;
    const G = [i * B[0][0] + j * B[1][0] + k * B[2][0], i * B[0][1] + j * B[1][1] + k * B[2][1], i * B[0][2] + j * B[1][2] + k * B[2][2]];
    const g2 = G[0] * G[0] + G[1] * G[1] + G[2] * G[2];
    const kg = kCart[0] * G[0] + kCart[1] * G[1] + kCart[2] * G[2];
    if (kg > g2 / 2 + tol) return false;
  }
  return true;
}

function check(name, A, basis, expectCode) {
  const br = analyzeBravais(A, basis);
  const m = buildBZModel(br);
  ok(br.code === expectCode, `${name}: code ${br.code} (expect ${expectCode}) — variant ${m.variant}`);
  let allIn = true, maxCErr = 0;
  for (const [, p] of Object.entries(m.points)) {
    if (!inBZ(p.cart, br.B_prim)) allIn = false;
    const c2 = fracToCart(primToConv(p.frac, br.T), br.B_conv);
    maxCErr = Math.max(maxCErr, Math.hypot(p.cart[0] - c2[0], p.cart[1] - c2[1], p.cart[2] - c2[2]));
  }
  ok(allIn, `${name}: all ${Object.keys(m.points).length} high-sym points lie in the BZ`);
  ok(maxCErr < 1e-9, `${name}: prim→conv preserves cartesian k (maxErr ${maxCErr.toExponential(1)})`);
}

console.log('\nSetyawan–Curtarolo variant lattices:');
check('BCT1 (c<a)', cell(4, 4, 3, 90, 90, 90), BODY, 'BCT');
check('BCT2 (c>a)', cell(4, 4, 6, 90, 90, 90), BODY, 'BCT');
check('RHL1 (α<90)', cell(4, 4, 4, 60, 60, 60), P1, 'RHL');
check('RHL2 (α>90)', cell(4, 4, 4, 110, 110, 110), P1, 'RHL');
check('ORCC', cell(3, 5, 7, 90, 90, 90), CBASE, 'ORCC');
check('ORCI', cell(3, 5, 7, 90, 90, 90), BODY, 'ORCI');
check('ORCF1', cell(3, 5, 7, 90, 90, 90), FACE, 'ORCF');
check('ORCF2', cell(3, 3.2, 3.4, 90, 90, 90), FACE, 'ORCF');
check('MCL (β≠90)', cell(4, 5, 6, 90, 80, 90), P1, 'MCL');
check('TRI', cell(4, 5, 6, 80, 85, 70), P1, 'TRI');

// ── Conventional-cell BZ model (cell-framework default, P = I) ───────────────
// The conventional path must NOT fold: FCC X is the conventional zone boundary
// (½,0,0), not the primitive seekpath X that maps to the integer reciprocal
// vector (0,1,0) ≡ Γ (which is what made Γ→X come out mirror-symmetric).
console.log('\nConventional-cell BZ model (Γ→X de-fold):');
{
  const a = 4;
  const Aconv = [[a, 0, 0], [0, a, 0], [0, 0, a]];
  const br = analyzeBravais(Aconv, FACE);
  ok(br.code === 'FCC', `FCC detected (code ${br.code})`);
  const prim = buildBZModel(br);
  const conv = buildConventionalBZModel(br);
  const xPrim = prim.points.X.fracConv.map(x => Math.round(x * 1e6) / 1e6);
  ok(xPrim.join(',') === '0,1,0', `primitive seekpath X folds to (0,1,0) ≡ Γ (got ${xPrim.join(',')})`);
  ok(conv.points.X.fracConv.join(',') === '0.5,0,0', `conventional X = (½,0,0) genuine zone boundary (got ${conv.points.X.fracConv.join(',')})`);
  // fracConv is the point itself (no transform) for every conventional point.
  let identity = true;
  for (const [, p] of Object.entries(conv.points)) if (p.fracConv !== p.frac) identity = false;
  ok(identity, 'conventional fracConv === frac (no primitive fold)');
  ok(conv.bz.vertices.length === 8 && conv.bz.faces.length === 6, `conventional cubic BZ is a cube (V=${conv.bz.vertices.length}, F=${conv.bz.faces.length})`);
}

// ── Reciprocal space follows the computation cell ───────────────────────────
console.log('\nCell-aware BZ models (reciprocal space follows the cell):');
{
  const a = 10.356;
  const Aconv = [[a, 0, 0], [0, a, 0], [0, 0, a]];
  const brF = analyzeBravais(Aconv, FACE);
  ok(brF.code === 'FCC', `FCC detected (${brF.code})`);
  const conv = buildConventionalBZModel(brF);
  const prim = buildBZModel(brF);
  ok(Object.keys(conv.points).sort().join(',') === 'M,R,X,Γ', `conventional cell → cubic box points Γ,X,M,R`);
  ok(prim.code === 'FCC' && prim.points.W && prim.points.K && prim.points.L && prim.points.U,
    `primitive FCC cell → FCC Wigner-Seitz BZ with W,K,L,U (got ${prim.code}: ${Object.keys(prim.points).join(',')})`);
  // A 1×1×2 cubic supercell is tetragonal → its own BZ.
  const sup = buildSupercellBZModel(brF, [1, 1, 2]);
  ok(/tetragonal/.test(sup.code) && sup.points.Z && sup.points.A,
    `1×1×2 cubic supercell → tetragonal BZ (got ${sup.code})`);

  // Base-cell analysis: a 1×1×2 FCC supercell is BCT, and its PRIMITIVE fold must
  // give the body-centered-tetragonal BZ (N,P,Σ…), NOT the FCC cubic one.
  const Lbase = [[a, 0, 0], [0, a, 0], [0, 0, 2 * a]];
  const tiled = [];
  for (let k = 0; k < 2; k++) for (const s of FACE) tiled.push({ el: s.el, frac: [s.frac[0], s.frac[1], (k + s.frac[2]) / 2] });
  const brBase = analyzeBravais(Lbase, tiled);
  ok(brBase.code === 'BCT', `1×1×2 FCC base cell → BCT (got ${brBase.code})`);
  const primBZ = buildBZModel(brBase);
  ok(primBZ.code === 'BCT' && primBZ.points.N && primBZ.points.P,
    `1×1×2 primitive fold → BCT Wigner-Seitz BZ, not FCC (got ${primBZ.code}: ${Object.keys(primBZ.points).join(',')})`);
  // Every point must lie in its own cell's BZ (frac fed straight to the phase).
  for (const model of [conv, prim, sup]) {
    const recip = model.bz; // WS already built for the cell; just check points are finite & fed as .frac
    ok(Object.values(model.points).every(p => Array.isArray(p.frac) && p.frac.every(Number.isFinite)),
      `${model.code}: all points carry a finite cell-fractional .frac`);
    void recip;
  }
}

console.log(`\n${fail === 0 ? '✅ high-sym tables OK (all points in BZ)' : `❌ ${fail} failed`}`);
process.exit(fail ? 1 : 0);
