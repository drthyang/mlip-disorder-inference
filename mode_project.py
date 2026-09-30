#!/usr/bin/env python3
"""mode_project.py — milestone 3: model-free irrep-mode projections.

Projects RMC ensemble configurations onto the published GaTa4Se8 distortion
modes (Yang et al., PRR 4, 033123, SM Tables V–X via
reference/gts_mode_patterns.json), star-pooled over all arms, phases and
domain orientations of the cubic parent. No MLIP, no relaxation: the
displacement reference is the experimental average structure.

Chain (all pure functions):
    patterns (20 AMPLIMODES sites, tetragonal frac)
      -> site mapping to parent orbits (global assignment, never by name)
      -> full 104-atom (52 sites x 2 z-parities) mode fields, Å
      -> star variants via the parent cubic space group + slab translations
      -> per-config projections A_m(c) in the PUBLISHED amplitude
         convention (mode-field norm per parent primitive cell), so a fully
         ordered single-domain box reads A_X5 = 0.1196 Å directly.

Units: Å throughout; fractional coordinates noted per frame (cubic
conventional a ≈ 10.356 Å vs tetragonal c = 2a slab).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
REF_JSON = REPO / "reference/gts_mode_patterns.json"

# P-42_1m (113) operations, verbatim from the refined CIF symmetry loop.
SG113_OPS = [
    "x, y, z", "y, -x, -z", "-x, -y, z", "-y, x, -z",
    "x+1/2, -y+1/2, -z", "-x+1/2, y+1/2, -z",
    "-y+1/2, -x+1/2, z", "y+1/2, x+1/2, z",
]


def parse_op(op: str):
    """'x+1/2, -y+1/2, -z' -> (R (3,3), t (3,)) acting on fractional coords."""
    R = np.zeros((3, 3))
    t = np.zeros(3)
    for i, comp in enumerate(op.split(",")):
        comp = comp.replace(" ", "")
        for j, ax in enumerate("xyz"):
            m = re.search(rf"(-?)({ax})", comp)
            if m:
                R[i, j] = -1.0 if m.group(1) == "-" else 1.0
                comp = comp.replace(m.group(0), "", 1)
        m = re.search(r"([+-]?\d+)/(\d+)", comp)
        if m:
            t[i] = float(m.group(1)) / float(m.group(2))
    return R, t


def load_reference(path: Path = REF_JSON):
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------
# slab construction: 52 cubic sites x 2 z-parities = the 104-atom cell
# ----------------------------------------------------------------------------

def build_slab(cubic_frac52: np.ndarray):
    """Tetragonal-frame fractional positions of the 1x1x2 slab.

    Returns (104, 3) array ordered as [(site 0, p=0) ... (site 51, p=0),
    (site 0, p=1) ...]; tetragonal frac = (x, y, (z_cubic + p)/2).
    """
    out = []
    for p in (0, 1):
        t = np.array(cubic_frac52, dtype=float)
        t[:, 2] = (t[:, 2] + p) / 2.0
        out.append(t)
    return np.vstack(out)


def frac_to_cart(tet_frac, a, c):
    f = np.asarray(tet_frac, dtype=float)
    return f * np.array([a, a, c])


def expand_refined(ref, a, c):
    """Expand SM Table II's 20 sites to 104 atoms via the SG-113 ops.

    Returns tetragonal fractional positions (104, 3) and element labels.
    """
    ops = [parse_op(s) for s in SG113_OPS]
    pos, elem = [], []
    for lab, row in ref["table_II_refined"].items():
        x = np.array(row["xyz"], dtype=float)
        seen = []
        for R, t in ops:
            y = (R @ x + t) % 1.0
            if not any(np.linalg.norm((y - z + 0.5) % 1 - 0.5) < 1e-4
                       for z in seen):
                seen.append(y)
        el = re.match(r"[A-Za-z]+", lab).group(0)
        pos.extend(seen)
        elem.extend([el] * len(seen))
    return np.array(pos), elem


def displacement_field(slab_frac, slab_elem, refined_frac, refined_elem,
                       a, c, max_disp=0.5):
    """Per-slab-atom displacement (Å): nearest refined atom minus parent.

    Minimum image in tetragonal fractional space; element identity enforced.
    Returns (n, 3) Å; raises if any atom lacks a partner within `max_disp` Å.
    """
    slab_frac = np.asarray(slab_frac)
    refined_frac = np.asarray(refined_frac)
    scale = np.array([a, a, c])
    d = refined_frac[None, :, :] - slab_frac[:, None, :]
    d -= np.round(d)
    d = d * scale
    r2 = (d * d).sum(-1)
    same = (np.array(slab_elem)[:, None] ==
            np.array(refined_elem)[None, :])
    r2 = np.where(same, r2, np.inf)
    j = np.argmin(r2, axis=1)
    best = np.sqrt(r2[np.arange(len(slab_frac)), j])
    if best.max() > max_disp:
        raise RuntimeError(
            f"unmatched atom: max displacement {best.max():.3f} Å")
    return d[np.arange(len(slab_frac)), j]


def align_parent_frame(cubic_frac52, elem52, refined_frac, refined_elem,
                       a, c):
    """Align the cubic parent frame with the refined tetragonal frame.

    The refined setting (SM Table II) may differ from the cubic CIF frame by
    a parent point-group rotation AND an arbitrary origin shift (e.g. half a
    cubic cell along z — not a parent translation). For each candidate
    rotation, the shift is found by clustering the pairwise difference
    vectors slab->refined (the true shift recurs once per atom within the
    physical distortion scale ~0.07 Å; wrong rotations never cluster).
    Returns (aligned_cubic_frac52, (R, t_tet), max_disp_Å).
    """
    ops = cubic_symmetry_ops(cubic_frac52, elem52)
    rotations = {tuple(np.rint(R).astype(int).ravel()): np.rint(R).astype(int)
                 for R, _ in ops}
    slab_elem = np.array(list(elem52) * 2)
    ref_el = np.array(refined_elem)
    best = None
    for R in rotations.values():
        frac = (np.asarray(cubic_frac52) @ R.T) % 1.0
        slab = build_slab(frac)
        # candidate shifts: all same-element pairwise differences, clustered
        diff = (np.asarray(refined_frac)[None, :, :] - slab[:, None, :]) % 1.0
        same = slab_elem[:, None] == ref_el[None, :]
        cand = diff[same]                              # (n_pairs, 3)
        keys, counts = np.unique(np.round(cand / 0.02).astype(int) % 50,
                                 axis=0, return_counts=True)
        t_tet = (keys[np.argmax(counts)] * 0.02) % 1.0
        # refine the shift as the tight-cluster mean, then score
        d = (cand - t_tet + 0.5) % 1.0 - 0.5
        close = np.abs(d * np.array([a, a, c])).max(axis=1) < 0.25
        if close.sum() < len(slab) // 2:
            continue
        t_tet = (t_tet + d[close].mean(axis=0)) % 1.0
        try:
            D = displacement_field((slab + t_tet) % 1.0, slab_elem,
                                   refined_frac, refined_elem, a, c,
                                   max_disp=np.inf)
        except RuntimeError:
            continue
        worst = float(np.linalg.norm(D, axis=1).max())
        if best is None or worst < best[2]:
            aligned = (frac + t_tet * np.array([1.0, 1.0, 2.0])) % 1.0
            best = (aligned, (R, t_tet), worst)
    if best is None or best[2] > 0.2:
        raise RuntimeError(f"no aligning parent operation found "
                           f"(best max disp {best and best[2]})")
    return best


# ----------------------------------------------------------------------------
# label -> orbit mapping and pattern expansion
# ----------------------------------------------------------------------------

def orbits_of_slab(slab_frac, slab_elem):
    """Group the 104 slab atoms into SG-113 orbits.

    Returns list of dicts {members: [idx], ops: [(R, t)] mapping the first
    member (rep) to each member}.
    """
    ops = [parse_op(s) for s in SG113_OPS]
    unassigned = set(range(len(slab_frac)))
    orbits = []
    while unassigned:
        rep = min(unassigned)
        members, mops = [], []
        for R, t in ops:
            y = (R @ slab_frac[rep] + t) % 1.0
            for j in sorted(unassigned):
                if slab_elem[j] == slab_elem[rep] and np.linalg.norm(
                        (y - slab_frac[j] + 0.5) % 1 - 0.5) < 2e-2:
                    if j not in members:
                        members.append(j)
                        mops.append((R, t))
                    break
        for j in members:
            unassigned.discard(j)
        orbits.append({"members": members, "ops": mops})
    return orbits


def expand_vec_over_orbit(orbit, anchor_pos: int, vec: np.ndarray):
    """Transport a vector given at orbit member index `anchor_pos` (position
    within orbit['members']) to every member. Returns (n_members, 3)."""
    Rk, _ = orbit["ops"][anchor_pos]
    v0 = np.linalg.inv(Rk) @ vec       # vector at the orbit-internal rep
    return np.array([R @ v0 for R, _ in orbit["ops"]])


def _orbit_anchor_cost(orbit, D, u):
    """For each candidate anchor member, the summed misfit of the
    orbit-expanded u against the exact D field. Returns (costs, best_pos)."""
    costs = []
    for k in range(len(orbit["members"])):
        exp = expand_vec_over_orbit(orbit, k, u)
        costs.append(float(np.linalg.norm(
            exp - D[orbit["members"]], axis=1).sum()))
    costs = np.array(costs)
    return costs, int(np.argmin(costs))


def map_labels_to_orbits(ref, slab_frac, slab_elem, D, a, c):
    """Global assignment of AMPLIMODES labels to slab orbits.

    Cost of (label, orbit) = the misfit of the label's Table-IV vector
    expanded over the WHOLE orbit against the exact D field, minimised over
    the anchor member — this both assigns the orbit and fixes the anchor
    unambiguously (a wrong anchor fails on the other members even when it
    matches at one). Solved per (element, multiplicity) class with the
    Hungarian algorithm so near-duplicate vectors (Ta2/Ta5) are resolved
    globally. Returns ({label: (orbit_index, anchor_pos)}, orbits).
    """
    from scipy.optimize import linear_sum_assignment

    orbits = orbits_of_slab(slab_frac, slab_elem)
    scale = np.array([a, a, c])
    table = ref["table_IV_total"]
    mapping = {}
    for el in ("Ga", "Ta", "Se"):
        for wp, mult in (("4e", 4), ("8f", 8)):
            labs = [l for l, r in table.items()
                    if l.startswith(el) and r["wp"] == wp]
            orbs = [oi for oi, o in enumerate(orbits)
                    if slab_elem[o["members"][0]] == el
                    and len(o["members"]) == mult]
            if not labs:
                continue
            assert len(labs) == len(orbs), (el, wp, len(labs), len(orbs))
            cost = np.zeros((len(labs), len(orbs)))
            anchor = {}
            for i, lab in enumerate(labs):
                u = np.array(table[lab]["u_frac"]) * scale
                for j, oi in enumerate(orbs):
                    costs, kbest = _orbit_anchor_cost(orbits[oi], D, u)
                    cost[i, j] = costs[kbest]
                    anchor[(i, j)] = kbest
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                # per-member rounding (a few times 5e-3 Å) summed over the
                # orbit; the hard gate is the downstream amplitude test
                assert cost[i, j] < 0.03 * mult, (labs[i], cost[i, j])
                mapping[labs[i]] = (orbs[j], anchor[(i, j)])
    return mapping, orbits


def expand_patterns(ref, slab_frac, slab_elem, mapping, orbits, a, c):
    """Per-irrep 104-atom displacement fields (Å), {mode_key: (104, 3)}.

    The printed pattern applies at each label's AMPLIMODES representative;
    other orbit members carry the op-rotated vector: if member = (R|t)(rep'),
    with rep' the orbit's internal rep, vectors transform as R applied to
    the rep' vector. We therefore first express the pattern at the internal
    rep, then rotate to every member.
    """
    fields = {}
    for key, mode in ref["modes"].items():
        F = np.zeros((len(slab_frac), 3))
        for lab, (oi, anchor_pos) in mapping.items():
            d = np.array(mode["pattern_frac"][lab]["d_frac"]) * \
                np.array([a, a, c])
            orb = orbits[oi]
            F[orb["members"]] = expand_vec_over_orbit(orb, anchor_pos, d)
        fields[key] = F
    return fields


# ----------------------------------------------------------------------------
# star variants and projections
# ----------------------------------------------------------------------------

def idealize_parent(cubic_frac52, elem52, symprec=0.05):
    """Symmetrize the cubic average onto exact Wyckoff positions, in place.

    Reynolds-operator projection: each atom's position is replaced by the
    average of its images under every space-group operation (with the
    permutation each op induces), killing per-site averaging noise while
    PRESERVING the input atom order (the RMC site-id indexing). Returns the
    idealized (52, 3) fractional positions.
    """
    frac = np.asarray(cubic_frac52, dtype=float) % 1.0
    ops = cubic_symmetry_ops(frac, elem52, symprec=symprec)
    acc = np.zeros_like(frac)
    for R, t in ops:
        y = (frac @ R.T + t) % 1.0
        d = (frac[None, :, :] - y[:, None, :] + 0.5) % 1.0 - 0.5
        perm = np.argmin((d * d).sum(-1), axis=1)  # image j lands on perm[j]
        # accumulate op(x_j) at slot perm[j], unwrapped near the original
        for j, i in enumerate(perm):
            delta = (y[j] - frac[i] + 0.5) % 1.0 - 0.5
            acc[i] += frac[i] + delta
    return (acc / len(ops)) % 1.0


def cubic_symmetry_ops(cubic_frac52, elem52, symprec=0.02):
    """Space-group ops of the ideal cubic parent (F-4̄3m), cubic frac frame."""
    import spglib
    from ase.data import atomic_numbers

    cell = (np.eye(3), np.asarray(cubic_frac52) % 1.0,
            [atomic_numbers[e] for e in elem52])
    sym = spglib.get_symmetry(cell, symprec=symprec)
    return list(zip(sym["rotations"], sym["translations"]))


def field_to_compact(F104):
    """(104,3) slab field -> compact (axis, G[52,2,3]) with axis='z'."""
    G = np.zeros((52, 2, 3))
    G[:, 0] = F104[:52]
    G[:, 1] = F104[52:]
    return (2, G)  # axis index 2 = z


def make_variants(compact, cubic_frac52, elem52, ops):
    """Orbit of a compact slab field under the parent group + slab shifts.

    Each variant is (axis, G[52,2,3]): the doubling axis may rotate to x/y.
    Deduplicated by rounding. Site identification via position matching in
    cubic fractional coordinates.
    """
    ax0, G0 = compact
    pos = np.asarray(cubic_frac52) % 1.0
    variants = {}
    for R, t in ops:
        # doubling axis transforms with R (fractional cubic ops: R orthogonal)
        e = np.zeros(3)
        e[ax0] = 1.0
        e2 = R @ e
        ax1 = int(np.argmax(np.abs(e2)))
        for n in range(2):  # slab shift along the ORIGINAL axis (phase)
            G1 = np.zeros((52, 2, 3))
            ok = True
            for s in range(52):
                for p in range(2):
                    X = pos[s].copy()
                    X[ax0] += (p + n)          # cubic-cell offset along ax0
                    Y = R @ X + t
                    Ys = Y % 1.0
                    d = (pos - Ys + 0.5) % 1.0 - 0.5
                    j = int(np.argmin((d * d).sum(1)))
                    if (d[j] * d[j]).sum() > 1e-3:
                        ok = False
                        break
                    p1 = int(np.floor(Y[ax1] - Ys[ax1] + 0.5)) % 2
                    G1[j, p1] = R @ G0[s, p]
                if not ok:
                    break
            if ok:
                key = (ax1, tuple(np.round(G1, 6).ravel()))
                variants[key] = (ax1, G1)
    return list(variants.values())


def compact_inner(f1, f2, n_cells=512):
    """Box inner product of two compact fields (Å² · atoms)."""
    ax1, G1 = f1
    ax2, G2 = f2
    tot = 0.0
    for px in range(2):
        for py in range(2):
            for pz in range(2):
                par = (px, py, pz)
                tot += float((G1[:, par[ax1]] * G2[:, par[ax2]]).sum())
    return tot * n_cells / 8.0


def parity_sums(unit_frac, sid, ijk, mean_site, a, signs=None):
    """Per-config parity-resolved displacement sums S[axis, p, site, 3] (Å).

    unit_frac: (N,3) positions folded to the cubic unit cell frame;
    sid: (N,) site ids 1..52; ijk: (N,3) integer cell offsets.
    signs: optional (N,) ±1 array — the random-sign null (see
    `random_signs`) multiplies each atom's displacement before summation.
    """
    d = unit_frac - mean_site[sid - 1]
    d -= np.round(d)
    d *= a
    if signs is not None:
        d = d * signs[:, None]
    S = np.zeros((3, 2, 52, 3))
    idx = sid - 1
    for ax in range(3):
        par = ijk[:, ax] & 1
        flat = (par * 52 + idx).astype(int)
        for comp in range(3):
            acc = np.bincount(flat, weights=d[:, comp], minlength=104)
            S[ax, 0, :, comp] = acc[:52]
            S[ax, 1, :, comp] = acc[52:]
    return S


def build_projector(fields, cubic_frac52, elem52):
    """Joint projector over ALL irreps' star variants.

    The printed patterns carry ~1 % rounding, so independent per-irrep
    subspace projections leak power between irreps (a 24-dim span grabs a
    few % of a larger mode's content). The joint (competitive) least
    squares assigns each component to the basis that fits it best; the
    per-irrep amplitude is then the norm of that irrep's block of the
    fitted field. Returns the projector dict used by `project_all`.
    """
    ops = cubic_symmetry_ops(cubic_frac52, elem52)
    keys = sorted(fields)
    variants_all, blocks = [], {}
    for key in keys:
        vs = make_variants(field_to_compact(fields[key]), cubic_frac52,
                           elem52, ops)
        blocks[key] = (len(variants_all), len(variants_all) + len(vs))
        variants_all.extend(vs)
    G = np.array([[compact_inner(v, w) for w in variants_all]
                  for v in variants_all])
    return {"keys": keys, "variants": variants_all, "blocks": blocks,
            "gram": G, "gram_pinv": np.linalg.pinv(G, rcond=1e-6),
            "rank": int(np.linalg.matrix_rank(G, tol=1e-6 * G.max()))}


def project_all(S, proj):
    """Joint per-config amplitudes for every irrep, published convention (Å).

    Solves u ≈ Σ_v x_v f_v over all variants jointly, then reads
    A_m = ||Σ_{v∈m} x_v f_v|| / sqrt(N_prim), N_prim = 2048 parent
    primitive cells in the 8×8×8 box. Returns {mode_key: A}.
    """
    q = np.array([float((G * S[ax].transpose(1, 0, 2)).sum())
                  for ax, G in proj["variants"]])
    x = proj["gram_pinv"] @ q
    out = {}
    for key, (i0, i1) in proj["blocks"].items():
        Gmm = proj["gram"][i0:i1, i0:i1]
        a2 = float(x[i0:i1] @ Gmm @ x[i0:i1])
        out[key] = np.sqrt(max(a2, 0.0) / 2048.0)
    return out


def window_parity_sums(unit_frac, sid, ijk, mean_site, a, w, signs=None):
    """Windowed parity-resolved displacement sums.

    Splits the 8×8×8 box into (8/w)³ cubic windows of side `w` cells
    (w even, so every parity triple is equally represented per window).
    Returns S_w of shape (n_windows, 3, 2, 52, 3) in Å — the same object
    `parity_sums` builds, per window. `signs` as in `parity_sums`.
    """
    assert 8 % w == 0 and w % 2 == 0, "window side must be even and divide 8"
    nw = 8 // w
    d = unit_frac - mean_site[sid - 1]
    d -= np.round(d)
    d *= a
    if signs is not None:
        d = d * signs[:, None]
    win = ((ijk[:, 0] // w) * nw + ijk[:, 1] // w) * nw + ijk[:, 2] // w
    idx = sid - 1
    S = np.zeros((nw**3, 3, 2, 52, 3))
    for ax in range(3):
        par = ijk[:, ax] & 1
        flat = ((win * 3 + ax) * 2 + par) * 52 + idx
        for comp in range(3):
            acc = np.bincount(flat, weights=d[:, comp],
                              minlength=nw**3 * 3 * 2 * 52)
            S[:, ax, :, :, comp] = acc.reshape(nw**3, 3, 2, 52)[:, ax]
    return S


def project_all_windows(S_w, proj, w):
    """Joint per-window amplitudes for every irrep (published convention, Å).

    The window Gram is the global Gram scaled by the cell fraction
    (w³/512) — valid because even-sided windows sample every parity sector
    uniformly. Normalization uses the window's own primitive count 4w³, so
    a coherently ordered window reads the published amplitude regardless
    of window size. Returns {mode_key: (n_windows,) array}.
    """
    frac = w**3 / 512.0
    gram = proj["gram"] * frac
    gram_pinv = np.linalg.pinv(gram, rcond=1e-6)
    nprim = 4.0 * w**3
    out = {k: np.zeros(len(S_w)) for k in proj["keys"]}
    for iw, S in enumerate(S_w):
        q = np.array([float((G * S[ax].transpose(1, 0, 2)).sum())
                      for ax, G in proj["variants"]])
        x = gram_pinv @ q
        for key, (i0, i1) in proj["blocks"].items():
            a2 = float(x[i0:i1] @ gram[i0:i1, i0:i1] @ x[i0:i1])
            out[key][iw] = np.sqrt(max(a2, 0.0) / nprim)
    return out


def shuffle_cells(unit_frac, sid, ijk, rng):
    """Coherence DIAGNOSTIC (not a full noise floor): permute cell indices
    independently per site.

    Destroys inter-cell correlations only. An X-point pattern is constant
    across conventional cells (it lives inside the cell), so it SURVIVES
    this shuffle exactly, while Δ/W-type (cell-alternating) coherence is
    destroyed — the difference measured/shuffled therefore separates
    intra-cell from inter-cell coherence. Returns a shuffled ijk array.
    """
    ijk2 = ijk.copy()
    for s in range(1, 53):
        m = np.where(sid == s)[0]
        ijk2[m] = ijk[m][rng.permutation(len(m))]
    return ijk2


def random_signs(n, rng):
    """The universal noise null: ±1 per atom.

    Multiplying displacements by random signs destroys EVERY coherent
    pattern (intra-cell and inter-cell alike) while preserving each site's
    displacement-magnitude distribution — the incoherent pedestal every
    channel's amplitude must be measured against. Pass to `parity_sums` /
    `window_parity_sums` via `signs`.
    """
    return rng.choice([-1.0, 1.0], size=n)


def load_rmc_config(path: Path, n_header_stop: str = "atoms"):
    """(unit_frac (N,3) cubic-unit-cell frame, sid (N,), ijk (N,3)).

    NOTE: the result is in the RMC (CIF) frame, not the pattern frame the
    projector is built in — see `to_aligned_frame` before projecting.
    """
    with open(path) as fh:
        for n, line in enumerate(fh):
            if line.strip().lower().startswith(n_header_stop):
                break
    arr = np.loadtxt(path, skiprows=n + 1, usecols=(3, 4, 5, 6, 7, 8, 9))
    unit = (arr[:, 0:3] * 8.0) % 1.0
    return unit, arr[:, 3].astype(int), arr[:, 4:7].astype(int)


# ----------------------------------------------------------------------------
# ensemble driver: RMC frame -> pattern frame -> windowed projections
# ----------------------------------------------------------------------------

DEFAULT_CIF = REPO / "data/GTS_5K.cif"
WINDOWS = (2, 4, 8)


def projection_setup(cif: Path = DEFAULT_CIF, ref_path: Path = REF_JSON):
    """Build everything needed to project RMC configurations.

    The published patterns live in the refined structure's setting, which
    differs from the RMC/CIF frame by a parent rotation R and an origin
    shift t (for GTS: an axis permutation with signs and half a cubic cell —
    not a parent translation). `align_parent_frame` finds (R, t) with site
    labels kept: aligned[s] = (R·ideal[s] + t) mod 1. The projector is built
    in the aligned frame, so every config must be mapped there first
    (`to_aligned_frame`); projecting RMC-frame coordinates directly scrambles
    channels (a pure W₄ at 0.10 Å reads 0.02 — tests/test_projection_frame).

    Parameters
    ----------
    cif : P1 CIF of the cubic parent whose atom order is the RMC site-id
        order (1..52) — data/GTS_5K.cif for GTS.
    ref_path : the mode-pattern reference JSON.

    Returns
    -------
    dict with keys: elem (52 symbols), ideal (52,3) idealized parent in the
    RMC frame, aligned (52,3) in the pattern frame, R (3,3) int, shift (3,)
    cubic fractional, n_site (52,3) int = floor(R·ideal + shift), fields
    ({mode: (104,3) Å}), projector, keys, a_ref (tetragonal a, Å).
    """
    from ase.io import read

    ref = load_reference(ref_path)
    a, c = ref["cell"]["a_A"], ref["cell"]["c_A"]
    at = read(str(cif))
    elem = at.get_chemical_symbols()
    ideal = idealize_parent(at.get_scaled_positions() % 1.0, elem)
    refined, refined_elem = expand_refined(ref, a, c)
    aligned, (R, t_tet), _ = align_parent_frame(ideal, elem, refined,
                                                refined_elem, a, c)
    R = np.rint(R).astype(int)
    shift = np.asarray(t_tet) * np.array([1.0, 1.0, 2.0])
    n_site = np.floor(ideal @ R.T + shift + 1e-9).astype(int)
    slab = build_slab(aligned)
    slab_elem = list(elem) * 2
    D = displacement_field(slab, slab_elem, refined, refined_elem, a, c)
    mapping, orbits = map_labels_to_orbits(ref, slab, slab_elem, D, a, c)
    fields = expand_patterns(ref, slab, slab_elem, mapping, orbits, a, c)
    proj = build_projector(fields, aligned, elem)
    return {"elem": elem, "ideal": ideal, "aligned": aligned, "R": R,
            "shift": shift, "n_site": n_site, "fields": fields,
            "projector": proj, "keys": proj["keys"], "a_ref": a}


def read_rmc6f_box(path: Path):
    """Positions of an 8×8×8 rmc6f box in conventional-cell units.

    Returns (X (N,3) in [0,8), sid (N,) site ids 1..52, ijk (N,3) the
    reference-cell offsets from the file, a_cub Å, moves_generated int).
    """
    header = {}
    with open(path) as fh:
        for n, line in enumerate(fh):
            low = line.strip().lower()
            if low.startswith("atoms"):
                break
            if ":" in line:
                k, v = line.split(":", 1)
                header[k.strip().lower()] = v.split()
    dims = np.array(header["supercell dimensions"][:3], dtype=int)
    if not np.all(dims == 8):
        raise ValueError(f"{path}: projector assumes an 8x8x8 box, got {dims}")
    cell_len = float(header["cell (ang/deg)"][0])
    moves = int(header.get("number of moves generated", ["-1"])[0])
    arr = np.loadtxt(path, skiprows=n + 1, usecols=(3, 4, 5, 6, 7, 8, 9))
    return (arr[:, 0:3] * dims, arr[:, 3].astype(int),
            arr[:, 4:7].astype(int), cell_len / dims[0], moves)


def to_aligned_frame(X, sid, ijk, setup):
    """Map a box from the RMC frame into the pattern (aligned) frame.

    X' = R·X + shift (mod 8) and ijk' = R·ijk + n_site[s] (mod 8), so each
    atom keeps its site label and its displacement from its own reference
    cell is rotated, d' = R·d. Returns (unit_frac (N,3), ijk' (N,3)) for
    `parity_sums` / `window_parity_sums` with mean_site = setup['aligned'].
    """
    R, shift, n_site = setup["R"], setup["shift"], setup["n_site"]
    Xp = (np.asarray(X) @ R.T + shift) % 8.0
    ijkp = (np.asarray(ijk) @ R.T + n_site[np.asarray(sid) - 1]) % 8
    return Xp % 1.0, ijkp.astype(int)


def config_amplitudes(X, sid, ijk, a_cub, setup, windows=WINDOWS, rng=None):
    """Windowed irrep amplitudes of one box, measured and random-sign null.

    Returns {f"w{w}": (n_windows, n_keys), f"null_w{w}": (n_windows, n_keys)}
    in Å (published per-primitive-cell convention), key order setup['keys'].
    """
    rng = np.random.default_rng(0) if rng is None else rng
    unit, ijkp = to_aligned_frame(X, sid, ijk, setup)
    proj, keys, mean = setup["projector"], setup["keys"], setup["aligned"]
    signs = random_signs(len(sid), rng)
    out = {}
    for w in windows:
        for tag, sg in (("w", None), ("null_w", signs)):
            S_w = window_parity_sums(unit, sid, ijkp, mean, a_cub, w, signs=sg)
            amp = project_all_windows(S_w, proj, w)
            out[f"{tag}{w}"] = np.column_stack([amp[k] for k in keys])
    return out


def ensemble_amplitudes(files, setup, windows=WINDOWS, seed=0, log=print):
    """Frame-correct windowed projections of a list of rmc6f boxes.

    Returns a dict ready for np.savez: keys, files, moves, and per scale w
    the full window arrays amp_w{w} / null_w{w} (n_cfg, n_windows, n_keys)
    plus the per-config RMS over windows rms_w{w} / rms_null_w{w}
    (n_cfg, n_keys) — the objects `verdicts.assemble` consumes. Units Å.
    """
    rng = np.random.default_rng(seed)
    acc = {}
    moves = []
    for n, f in enumerate(files, 1):
        X, sid, ijk, a_cub, mv = read_rmc6f_box(Path(f))
        moves.append(mv)
        for k, v in config_amplitudes(X, sid, ijk, a_cub, setup, windows,
                                      rng).items():
            acc.setdefault(k, []).append(v)
        if log and (n % 50 == 0 or n == len(files)):
            log(f"  projected {n}/{len(files)}")
    out = {"keys": np.array(setup["keys"]),
           "files": np.array([Path(f).name for f in files]),
           "moves": np.array(moves)}
    for w in windows:
        amp = np.array(acc[f"w{w}"])
        nul = np.array(acc[f"null_w{w}"])
        out[f"amp_w{w}"], out[f"null_w{w}"] = amp, nul
        out[f"rms_w{w}"] = np.sqrt((amp**2).mean(axis=1))
        out[f"rms_null_w{w}"] = np.sqrt((nul**2).mean(axis=1))
    return out


def main(argv=None):
    """CLI: frame-correct windowed projections of an rmc6f ensemble -> npz."""
    import argparse

    import milestone1_bands as m1

    ap = argparse.ArgumentParser(description="Windowed irrep projections of "
                                 "an 8x8x8 RMC ensemble (pattern frame).")
    ap.add_argument("inputs", nargs="+", help=".rmc6f files or directories")
    ap.add_argument("--cif", type=Path, default=DEFAULT_CIF,
                    help="P1 parent CIF in RMC site-id order")
    ap.add_argument("--exclude", default="AVERAGE",
                    help="drop files whose NAME matches this regex")
    ap.add_argument("--skip-nonconverged", action="store_true")
    ap.add_argument("--drop", default="",
                    help="comma-separated config numbers to drop (e.g. "
                    "duplicate chains 206,404,208)")
    ap.add_argument("--seed", type=int, default=0, help="random-sign seed")
    ap.add_argument("-o", "--out", type=Path, default=Path("projections.npz"))
    args = ap.parse_args(argv)

    files = m1.collect_inputs(args.inputs)
    if args.exclude:
        files = [f for f in files if not re.search(args.exclude, f.name)]
    if args.skip_nonconverged:
        files, dropped = m1.drop_nonconverged(files)
        print(f"  dropped {len(dropped)} non-converged config(s)")
    drop = {int(x) for x in args.drop.split(",") if x.strip()}
    if drop:
        def num(f):
            m = re.search(r"_(\d+)\.rmc6f$", f.name)
            return int(m.group(1)) if m else None
        files = [f for f in files if num(f) not in drop]
    print(f"[1/2] projector ({args.cif.name}); {len(files)} configs")
    setup = projection_setup(args.cif)
    print("[2/2] windowed projections, windows", WINDOWS)
    out = ensemble_amplitudes(files, setup, seed=args.seed)
    np.savez(args.out, **out)
    for w in WINDOWS:
        a = np.sqrt((out[f"rms_w{w}"]**2).mean(axis=0))
        n = np.sqrt((out[f"rms_null_w{w}"]**2).mean(axis=0))
        print(f"  w={w}: " + "  ".join(
            f"{k} {x:.4f}/{y:.4f}" for k, x, y in zip(setup["keys"], a, n))
            + "   (measured/null, Å)")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
