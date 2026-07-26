// viewer/src/math/reciprocal.js
//
// Lattice-aware reciprocal cell + high-symmetry k-path generation.
//
// The RMC supercell is a tiling of the CONVENTIONAL cell, and src_gpu's Bloch
// phase indexes that conventional cell. So (unlike the legacy seekpath path,
// which works in the primitive cell and transforms to conventional) we work
// directly in CONVENTIONAL-cell fractional reciprocal coordinates. The pipeline
// applies the 2*pi factor (constants.TWO_PI_PHASE) when feeding the kernel.
//
// Crystal SYSTEM is detected from the conventional-cell metric (a,b,c,angles).
// Centering (F/I/C) is NOT detected here — that needs spglib-level symmetry
// analysis. High-symmetry points are the standard simple-setting (P) points of
// each system (Setyawan & Curtarolo 2010 conventions). This is a large
// improvement over the previous "always cubic" behaviour; see
// FEATURE_PARITY_REPORT.md for the documented limitation.

// ── 3x3 linear algebra helpers ──────────────────────────────────────────────
export function mat3Inverse(m) {
  // m is row-major [[r0],[r1],[r2]]
  const [a, b, c] = m[0];
  const [d, e, f] = m[1];
  const [g, h, i] = m[2];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-300) throw new Error('Singular lattice matrix');
  const invDet = 1 / det;
  return [
    [A * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet],
    [B * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet],
    [C * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet],
  ];
}

export function mat3Transpose(m) {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
}

/**
 * Conventional unit-cell lattice (rows = a,b,c in Angstrom).
 * A_super = diag(dim)·A_conv, so each supercell VECTOR (row) is divided by its
 * own repeat count: row i = v_i / dim[i]. (Dividing components by dim[j] instead
 * is only equivalent for orthogonal cells or uniform dim.)
 */
export function conventionalLattice(v1, v2, v3, dim) {
  return [
    [v1[0] / dim[0], v1[1] / dim[0], v1[2] / dim[0]],
    [v2[0] / dim[1], v2[1] / dim[1], v2[2] / dim[1]],
    [v3[0] / dim[2], v3[1] / dim[2], v3[2] / dim[2]],
  ];
}

/**
 * Reciprocal lattice WITHOUT 2*pi: B = inv(A).T (rows = a*,b*,c*, units A^-1).
 * Matches the band.yaml `reciprocal_lattice` convention in src_gpu/Writers.py.
 */
export function reciprocalLattice(A) {
  return mat3Transpose(mat3Inverse(A));
}

function norm3(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function angleDeg(a, b) {
  const c = dot3(a, b) / (norm3(a) * norm3(b));
  return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
}

/**
 * Detect the crystal system from the conventional-cell metric.
 * Returns { system, a, b, c, alpha, beta, gamma }.
 */
export function detectSystem(v1, v2, v3, dim, tolLen = 1e-2, tolAng = 1.0) {
  const A = conventionalLattice(v1, v2, v3, dim);
  const a = norm3(A[0]), b = norm3(A[1]), c = norm3(A[2]);
  const alpha = angleDeg(A[1], A[2]);
  const beta = angleDeg(A[0], A[2]);
  const gamma = angleDeg(A[0], A[1]);

  const eqLen = (x, y) => Math.abs(x - y) / Math.max(x, y) < tolLen;
  const is90 = (x) => Math.abs(x - 90) < tolAng;
  const is120 = (x) => Math.abs(x - 120) < tolAng;

  let system = 'triclinic';
  if (is90(alpha) && is90(beta) && is90(gamma)) {
    if (eqLen(a, b) && eqLen(b, c)) system = 'cubic';
    else if (eqLen(a, b) || eqLen(b, c) || eqLen(a, c)) system = 'tetragonal';
    else system = 'orthorhombic';
  } else if (is90(alpha) && is90(beta) && is120(gamma) && eqLen(a, b)) {
    system = 'hexagonal';
  } else if (eqLen(a, b) && eqLen(b, c) &&
             Math.abs(alpha - beta) < tolAng && Math.abs(beta - gamma) < tolAng) {
    system = 'rhombohedral';
  } else if (is90(alpha) && is90(gamma)) {
    system = 'monoclinic';
  }
  return { system, a, b, c, alpha, beta, gamma };
}

// ── High-symmetry points (conventional-cell fractional reciprocal coords) ────
const T = 1 / 3, S = 2 / 3;
const HIGH_SYM = {
  cubic: {
    points: { 'Γ': [0, 0, 0], X: [0.5, 0, 0], M: [0.5, 0.5, 0], R: [0.5, 0.5, 0.5] },
    defaultPath: ['Γ', 'X', 'M', 'Γ', 'R', 'X'],
  },
  tetragonal: {
    points: {
      'Γ': [0, 0, 0], X: [0, 0.5, 0], M: [0.5, 0.5, 0],
      Z: [0, 0, 0.5], R: [0, 0.5, 0.5], A: [0.5, 0.5, 0.5],
    },
    defaultPath: ['Γ', 'X', 'M', 'Γ', 'Z', 'R', 'A', 'Z'],
  },
  orthorhombic: {
    points: {
      'Γ': [0, 0, 0], X: [0.5, 0, 0], Y: [0, 0.5, 0], Z: [0, 0, 0.5],
      S: [0.5, 0.5, 0], U: [0.5, 0, 0.5], T: [0, 0.5, 0.5], R: [0.5, 0.5, 0.5],
    },
    defaultPath: ['Γ', 'X', 'S', 'Y', 'Γ', 'Z', 'U', 'R', 'T', 'Z'],
  },
  hexagonal: {
    points: {
      'Γ': [0, 0, 0], M: [0.5, 0, 0], K: [T, T, 0],
      A: [0, 0, 0.5], L: [0.5, 0, 0.5], H: [T, T, 0.5],
    },
    defaultPath: ['Γ', 'M', 'K', 'Γ', 'A', 'L', 'H', 'A'],
  },
  rhombohedral: {
    // Approximate; rhombohedral special points depend on the angle. Provide a
    // safe generic set in conventional coords.
    points: { 'Γ': [0, 0, 0], L: [0.5, 0, 0], F: [0.5, 0.5, 0], Z: [0.5, 0.5, 0.5] },
    defaultPath: ['Γ', 'L', 'F', 'Γ', 'Z'],
  },
  monoclinic: {
    points: {
      'Γ': [0, 0, 0], X: [0.5, 0, 0], Y: [0, 0.5, 0], Z: [0, 0, 0.5],
      C: [0.5, 0.5, 0], A: [0.5, 0, 0.5], D: [0.5, 0.5, 0.5],
    },
    defaultPath: ['Γ', 'X', 'C', 'Y', 'Γ', 'Z', 'A', 'D'],
  },
  triclinic: {
    points: { 'Γ': [0, 0, 0], X: [0.5, 0, 0], Y: [0, 0.5, 0], Z: [0, 0, 0.5] },
    defaultPath: ['Γ', 'X', 'Γ', 'Y', 'Γ', 'Z'],
  },
};
void S; // S (2/3) reserved for future hexagonal variants

export function highSymmetryPoints(system) {
  return HIGH_SYM[system] || HIGH_SYM.cubic;
}

/**
 * Build per-segment k-path from a label path.
 *   sym_pnts : {label: [f1,f2,f3]} conventional fractional reciprocal coords
 *   pathLabels : ['Γ','X','M', ...]
 *   kstep : intervals per segment (kstep+1 points per segment, endpoints incl.)
 *
 * Returns { qFrac: [[..],..], segSizes: [..], hsymIndex: {flatQ: label} }.
 * Junction points are duplicated (phonopy band convention), matching
 * src_gpu/kpath.build_kpath. NOT scaled by 2*pi — the pipeline does that.
 */
export function buildKPath(sym_pnts, pathLabels, kstep = 20) {
  const qFrac = [];
  const segSizes = [];
  const hsymIndex = {};
  for (let i = 0; i < pathLabels.length - 1; i++) {
    const a = sym_pnts[pathLabels[i]];
    const b = sym_pnts[pathLabels[i + 1]];
    if (!a || !b) continue;
    const n = Math.max(2, kstep + 1);
    hsymIndex[qFrac.length] = pathLabels[i];
    for (let j = 0; j < n; j++) {
      const t = j / (n - 1);
      qFrac.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]);
    }
    segSizes.push(n);
  }
  if (qFrac.length > 0) hsymIndex[qFrac.length - 1] = pathLabels[pathLabels.length - 1];
  return { qFrac, segSizes, hsymIndex };
}

/**
 * Build a k-path from explicit per-segment {from,to,npoints} (labels into
 * sym_pnts). Allows different point counts per segment and discontinuous breaks.
 * Junction points are duplicated (phonopy convention). Not scaled by 2*pi.
 */
export function buildKPathFromSegments(sym_pnts, segments) {
  const qFrac = [], segSizes = [], hsymIndex = {};
  for (const seg of segments) {
    const a = sym_pnts[seg.from], b = sym_pnts[seg.to];
    if (!a || !b) continue;
    const n = Math.max(2, Math.round(seg.npoints) || 2);
    hsymIndex[qFrac.length] = seg.from;
    for (let j = 0; j < n; j++) {
      const t = j / (n - 1);
      qFrac.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]);
    }
    segSizes.push(n);
    hsymIndex[qFrac.length - 1] = seg.to;
  }
  return { qFrac, segSizes, hsymIndex };
}

/**
 * Cumulative physical path distance (Å^-1, no 2*pi) for each q-point, with the
 * increment zeroed at segment starts (matches src_gpu/Writers distances). Used
 * by the band-structure plot x-axis.
 *   qFrac : fractional q-points; recip : reciprocal lattice rows; segSizes.
 */
export function pathDistances(qFrac, recip, segSizes) {
  const n = qFrac.length;
  const segStarts = new Set();
  let off = 0;
  for (const sz of (segSizes || [n])) { segStarts.add(off); off += sz; }
  const d = [0];
  for (let i = 1; i < n; i++) {
    if (segStarts.has(i)) { d.push(d[i - 1]); continue; }
    const dq = [qFrac[i][0] - qFrac[i - 1][0], qFrac[i][1] - qFrac[i - 1][1], qFrac[i][2] - qFrac[i - 1][2]];
    const cx = dq[0] * recip[0][0] + dq[1] * recip[1][0] + dq[2] * recip[2][0];
    const cy = dq[0] * recip[0][1] + dq[1] * recip[1][1] + dq[2] * recip[2][1];
    const cz = dq[0] * recip[0][2] + dq[1] * recip[1][2] + dq[2] * recip[2][2];
    d.push(d[i - 1] + Math.sqrt(cx * cx + cy * cy + cz * cz));
  }
  return d;
}
