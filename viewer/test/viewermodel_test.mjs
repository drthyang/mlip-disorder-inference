// viewer/test/viewermodel_test.mjs
//
// Proves the vibration mode is assigned to the CORRECT atom through the whole
// chain: pipeline results -> viewer model -> (the 3D viewer's row lookup) and
// results -> band.yaml object -> band.json -> reload. Uses non-contiguous
// reference numbers (RN 2 and 5) so a naive index mapping would fail.

import { fromResults, fromBandText } from '../src/io/viewermodel.js';
import { generateBandJson } from '../src/io/writers.js';

let fail = 0;
const ok = (c, m) => { if (!c) { console.error('  FAIL ' + m); fail++; } else console.log('  ok   ' + m); };
const approx = (a, b, t = 1e-9) => Math.abs(a - b) <= t;

// Synthetic supercell: 2 basis sites — RN 2 (element 'A'), RN 5 (element 'B') —
// tiled over 2 cells along x (4 atoms). One mode where ONLY site B moves (+x).
const results = {
  baseStructure: {
    v1: [8, 0, 0], v2: [0, 4, 0], v3: [0, 0, 4], dim: [2, 1, 1],
    atomDic: { A: [2], B: [5] },
    uniqueRN: [2, 5],
    atomType: [2, 5, 2, 5],                        // per supercell atom
    hsym_xyz: new Float64Array([0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6]),
    cellIdx: new Float64Array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
  },
  // 1 q-point, 6 modes. Pick mode 3 (index 3) as "B moves +x".
  bands: [[1, 2, 3, 4, 5, 6]],
  eigvecs: [[
    mode([0, 0, 0, 0, 0, 0]),     // 0
    mode([1, 0, 0, 0, 0, 0]),     // 1: A moves +x  (row 0)
    mode([0, 1, 0, 0, 0, 0]),     // 2
    mode([0, 0, 0, 1, 0, 0]),     // 3: B moves +x  (row 1)
    mode([0, 0, 0, 0, 1, 0]),     // 4
    mode([0, 0, 0, 0, 0, 1]),     // 5
  ]],
  qPoints: [[0, 0, 0]],
  temperature: 5,
};
function mode(realArr) { return { real: Float64Array.from(realArr), imag: new Float64Array(6) }; }

console.log('\n[1] fromResults: unit-cell model keeps site identity');
const model = fromResults(results, { segSizes: [1], hsymIndex: {} });
ok(model.baseStructure.uniqueRN.join() === '2,5', 'uniqueRN = [2,5]');
ok(model.baseStructure.atomType.join() === '2,5', 'atomType (sites) = [2,5]');
// representative positions: site RN2 -> [0.1..], site RN5 -> [0.6..]
ok(approx(model.baseStructure.hsym_xyz[0], 0.1) && approx(model.baseStructure.hsym_xyz[3], 0.6), 'site positions match their reference numbers');

console.log('\n[2] the 3D viewer row lookup assigns mode 3 to the B atom only');
{
  // Replicate CrystalViewer: atom s (uniqueRN order) -> row = rnToRow(atomType[s]).
  const rnToRow = new Map(model.baseStructure.uniqueRN.map((rn, r) => [rn, r]));
  const rev = {}; for (const [el, idxs] of Object.entries(model.baseStructure.atomDic)) idxs.forEach(i => rev[i] = el);
  const ev = model.eigvecs[0][3];               // "B moves +x"
  const disp = model.baseStructure.atomType.map(rn => {
    const r = rnToRow.get(rn);
    return { el: rev[rn], rn, d: Math.hypot(ev.real[r * 3], ev.real[r * 3 + 1], ev.real[r * 3 + 2]) };
  });
  const movers = disp.filter(x => x.d > 1e-9);
  ok(movers.length === 1 && movers[0].el === 'B' && movers[0].rn === 5, 'only element B (RN 5) moves for mode 3');
  ok(disp.find(x => x.el === 'A').d === 0, 'element A (RN 2) is stationary for mode 3');
}

console.log('\n[3] round-trip results -> band.json -> reload preserves assignment');
{
  const json = generateBandJson(model.baseStructure, model.qPoints, model.bands, model.eigvecs, model.kpathMeta);
  const re = fromBandText(json);
  // points order is element-grouped (A then B); atom index 1 = B.
  const ev = re.eigvecs[0][3];
  const aMove = Math.hypot(ev.real[0], ev.real[1], ev.real[2]);   // atom 0 = A
  const bMove = Math.hypot(ev.real[3], ev.real[4], ev.real[5]);   // atom 1 = B
  ok(aMove === 0 && bMove > 0, 'after reload, mode 3 still moves only the B atom');
  ok(re.baseStructure.atomDic.B?.length === 1 && re.baseStructure.atomDic.A?.length === 1, 'elements preserved (A,B)');
}

console.log(`\n${fail === 0 ? '✅ mapping correct' : `❌ ${fail} failed`}`);
process.exit(fail ? 1 : 0);
