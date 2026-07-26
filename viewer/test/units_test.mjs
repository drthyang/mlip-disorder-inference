// viewer/test/units_test.mjs
//
// Units convention shared with phonopy:
//   * phonopy band.yaml `frequency` is THz (default factor sqrt(eV/amu)/Å/2π)
//     → the viewer model stores meV, so loading converts ×4.1356677 by default;
//   * a `frequency_unit:` key overrides the default (our exporter writes THz
//     with the key, legacy in-app exports without a key would be THz too);
//   * eigenvectors stay in the phonopy convention (mass-weighted, unit norm);
//     baseStructure.masses carries per-site masses for consumers;
//   * writer→loader round-trip preserves energies exactly;
//   * phonopy's top-level `labels:` list is mapped to segment boundaries.

import { fromBandText } from '../src/io/viewermodel.js';
import { generatePhonopyBandYaml, generateBandJson } from '../src/io/writers.js';
import { THZ_TO_MEV } from '../src/constants.js';

let fail = 0;
const ok = (c, m) => { if (!c) { console.error('  FAIL ' + m); fail++; } else console.log('  ok   ' + m); };
const approx = (a, b, t = 1e-9) => Math.abs(a - b) <= t;

const yamlText = (unitLine) => `nqpoint: 2
npath: 1
${unitLine}segment_nqpoint:
- 2
natom: 2
lattice:
- [ 4.0, 0.0, 0.0 ]
- [ 0.0, 4.0, 0.0 ]
- [ 0.0, 0.0, 4.0 ]
labels:
- [ '$\\\\Gamma$', '$\\\\mathrm{X}$' ]
points:
- symbol: Ga
  coordinates: [ 0.0, 0.0, 0.0 ]
  mass: 69.723
- symbol: Ta
  coordinates: [ 0.5, 0.5, 0.5 ]
  mass: 180.95
phonon:
- q-position: [ 0.0, 0.0, 0.0 ]
  distance: 0.0
  band:
  - # 1
    frequency: 1.0
    eigenvector:
    - # atom 1
      - [ 1.0, 0.0 ]
      - [ 0.0, 0.0 ]
      - [ 0.0, 0.0 ]
    - # atom 2
      - [ 0.0, 0.0 ]
      - [ 0.0, 0.0 ]
      - [ 0.0, 0.0 ]
- q-position: [ 0.5, 0.0, 0.0 ]
  distance: 0.125
  band:
  - # 1
    frequency: 2.0
    eigenvector:
    - # atom 1
      - [ 0.0, 0.0 ]
      - [ 1.0, 0.0 ]
      - [ 0.0, 0.0 ]
    - # atom 2
      - [ 0.0, 0.0 ]
      - [ 0.0, 0.0 ]
      - [ 0.0, 0.0 ]
`;

console.log('\n[1] phonopy default: frequency is THz → model meV (×4.1356677)');
{
  const m = fromBandText(yamlText(''));
  ok(approx(m.bands[0][0], THZ_TO_MEV), `1 THz → ${THZ_TO_MEV} meV (got ${m.bands[0][0]})`);
  ok(approx(m.bands[1][0], 2 * THZ_TO_MEV), '2 THz → 8.2713 meV');
}

console.log('\n[2] frequency_unit: meV is honored (self-describing files)');
{
  const m = fromBandText(yamlText('frequency_unit: meV\n'));
  ok(approx(m.bands[0][0], 1.0), 'frequency stored as-is when the file says meV');
}

console.log('\n[3] masses carried; eigenvectors untouched (phonopy convention)');
{
  const m = fromBandText(yamlText(''));
  ok(approx(m.baseStructure.masses[0], 69.723) && approx(m.baseStructure.masses[1], 180.95),
    'baseStructure.masses = points[].mass');
  ok(approx(m.eigvecs[0][0].real[0], 1.0), 'eigenvector components verbatim');
}

console.log('\n[4] phonopy top-level labels → segment boundaries (Γ, X)');
{
  const m = fromBandText(yamlText(''));
  ok(m.kpathMeta.hsymIndex[0] === 'Γ', `label[0] = Γ (got ${m.kpathMeta.hsymIndex[0]})`);
  ok(m.kpathMeta.hsymIndex[1] === 'X', `label[end] = X (got ${m.kpathMeta.hsymIndex[1]})`);
}

console.log('\n[5] writer → loader round-trip preserves meV energies exactly');
{
  const base = fromBandText(yamlText(''));
  for (const [name, gen] of [['yaml', generatePhonopyBandYaml], ['json', generateBandJson]]) {
    const text = gen(base.baseStructure, base.qPoints, base.bands, base.eigvecs, base.kpathMeta);
    ok(/frequency_unit/.test(text), `${name} export carries frequency_unit`);
    const re = fromBandText(text);
    ok(approx(re.bands[0][0], base.bands[0][0], 1e-6) && approx(re.bands[1][0], base.bands[1][0], 1e-6),
      `${name} round-trip energies match (${re.bands[0][0]} meV)`);
  }
}

console.log(`\n${fail === 0 ? '✅ units contract holds' : `❌ ${fail} failed`}`);
process.exit(fail ? 1 : 0);
