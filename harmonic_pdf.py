#!/usr/bin/env python3
"""harmonic_pdf.py — quantum-harmonic pair distribution of an infinite crystal.

The partial g_ab(r) of a harmonic crystal whose atoms carry quantum
(zero-point + thermal) displacement statistics, INCLUDING the correlation
between the displacements of different atoms (which sharpens near-neighbour
peaks). Evaluated as a lattice sum of per-pair radial Gaussians — no finite
box, no sampling noise — so it stands in for what a diffractometer sees from
the null model. Used by rmc_control.py to synthesize zero-static-disorder
total-scattering data for the RMC control experiment.

Physics. For atom j in cell 0 and atom j' in cell R, the relative
displacement is Gaussian with covariance

    C = U_j + U_j' − Σ_jj'(R) − Σ_jj'(R)ᵀ,
    U_j = ⟨u_j u_jᵀ⟩,   Σ_jj'(R) = ⟨u_j(0) u_j'(R)ᵀ⟩,
    Σ_jj'(R) = (1/N) Σ_qν ⟨|Q_qν|²⟩ e_jν(q) e*_j'ν(q) / √(m_j m_j')
                    · exp(−2πi q·(R + x_j' − x_j)),
    ⟨|Q|²⟩ = (ħ/ω)(n_B(ω, T) + ½),

with phonopy's eigenvectors (phase convention with atomic positions) on the
q-grid commensurate with an M×M×M box of the unit cell, summed by FFT. The
radial distribution of each pair is the isotropic-Gaussian shell profile
with the width projected on the bond, σ² = d̂ᵀ C d̂. Pairs beyond the
minimum-image cube of the M-box are treated as uncorrelated (Σ = 0).

The phonopy object must use the unit cell itself as its primitive cell
(primitive_matrix="P"; in phonopy ≥4 None means "auto"), so q and R are in
the unit cell's own basis.

Units: Å, THz, amu, K; g_ab(r) dimensionless (→ 1 at large r).
"""

from __future__ import annotations

import numpy as np


def _qsq_conversion():
    """ħ/(amu·Å²·2π·THz): ⟨|Q|²⟩ in amu·Å² is this / f_THz · (n + ½)."""
    from phonopy.physical_units import get_physical_units

    u = get_physical_units()
    return u.Hbar * u.EV / u.AMU / u.THz / (2 * np.pi) / u.Angstrom**2


def mode_qsq(freqs_thz, temperature, cutoff=0.01):
    """Normal-coordinate variance ⟨|Q|²⟩ = (ħ/ω)(n_B + ½) per mode, amu·Å².

    Quantum statistics (zero point included). Modes at or below `cutoff` THz
    (the Γ translations, and any imaginary modes, which phonopy reports as
    negative) are excluded — the same rule as phonopy's random-displacement
    sampler.
    """
    from phonopy.physical_units import get_physical_units

    f = np.asarray(freqs_thz, dtype=float)
    ok = f > cutoff
    fs = np.where(ok, f, 1.0)
    if temperature > 0:
        u = get_physical_units()
        x = u.THzToEv * fs / (u.KB * temperature)
        n = 1.0 / np.expm1(np.minimum(x, 700.0))
    else:
        n = np.zeros_like(fs)
    return np.where(ok, _qsq_conversion() / fs * (n + 0.5), 0.0)


def commensurate_grid(M: int) -> np.ndarray:
    """(M³, 3) q-points k/M in the unit cell's reciprocal basis, ordered so
    reshape(M, M, M, ...) matches np.fft.fftn axes."""
    i = np.arange(M)
    g = np.stack(np.meshgrid(i, i, i, indexing="ij"), axis=-1)
    return g.reshape(-1, 3) / float(M)


def scaled_modes(phonon, M, temperature, cutoff=0.01):
    """Eigenvectors scaled by √⟨|Q|²⟩ on the M-grid.

    Returns (q (nq,3), V (nq, 3n, 3n) complex with V[:, :, ν] = e_ν √⟨|Q_ν|²⟩,
    freqs (nq, 3n) THz). Memory: nq·(3n)²·16 bytes (1.6 GB for n=52, M=16).
    """
    if len(phonon.primitive) != len(phonon.unitcell):
        raise ValueError("harmonic_pdf needs primitive_matrix=\"P\" (the unit "
                         "cell as phonopy's primitive cell)")
    q = commensurate_grid(M)
    phonon.run_qpoints(q, with_eigenvectors=True)
    freqs = np.array(phonon.qpoints.frequencies)
    V = np.array(phonon.qpoints.eigenvectors)
    V *= np.sqrt(mode_qsq(freqs, temperature, cutoff))[:, None, :]
    return q, V, freqs


def site_covariances(V, masses):
    """U_j = ⟨u_j u_jᵀ⟩ (n, 3, 3), Å², from the scaled modes of `scaled_modes`."""
    n = len(masses)
    U = np.zeros((n, 3, 3))
    for j in range(n):
        Vj = V[:, 3 * j:3 * j + 3, :]
        U[j] = np.einsum("qan,qbn->ab", Vj, np.conj(Vj)).real
        U[j] /= V.shape[0] * masses[j]
    return U


def row_correlations(q, V, frac, masses, j, M):
    """Σ_jj'(R) for fixed j, all j' and all R on the M-grid, Å².

    Returns a real array (M, M, M, n, 3, 3): [R, j', α, β] = ⟨u_jα(0) u_j'β(R)⟩.
    U_j is the [0, 0, 0, j] block.
    """
    n = len(frac)
    nq = len(q)
    Vj = V[:, 3 * j:3 * j + 3, :]                          # (nq, 3, 3n)
    B = Vj @ np.conj(np.transpose(V, (0, 2, 1)))            # (nq, 3, 3n)
    phase = np.exp(-2j * np.pi * q @ (np.asarray(frac) - frac[j]).T)
    B = B.reshape(nq, 3, n, 3) * phase[:, None, :, None]
    S = np.fft.fftn(B.reshape(M, M, M, 3, n, 3), axes=(0, 1, 2)).real
    S /= float(nq)
    S /= np.sqrt(masses[j] * np.asarray(masses))[None, None, None, None, :,
                                                  None]
    return np.ascontiguousarray(np.transpose(S, (0, 1, 2, 4, 3, 5)))


def _cells_within(lattice, r_max):
    """Integer cell offsets n with |n·A| possibly ≤ r_max + one cell."""
    lengths = np.linalg.norm(lattice, axis=1)
    K = int(np.ceil(r_max / lengths.min())) + 1
    i = np.arange(-K, K + 1)
    return np.stack(np.meshgrid(i, i, i, indexing="ij"), -1).reshape(-1, 3)


def harmonic_partials(phonon, temperature, M=16, r_max=120.0, dr=0.01,
                      n_sigma=5.0, cutoff=0.01, log=print):
    """Partial g_ab(r) of the infinite quantum-harmonic crystal.

    Parameters
    ----------
    phonon : Phonopy with force constants and primitive == unit cell. The
        mean structure is phonopy's unit cell (positions and lattice as
        given — pass the cell whose mean positions the data should carry).
    temperature : K.
    M : q-grid (and correlation box) size in unit cells; displacement
        correlations are kept for pairs inside the minimum-image cube of the
        M-box and dropped beyond (Σ → 0, uncorrelated widths).
    r_max, dr : Å. Grid r_k = k·dr, k = 1..r_max/dr (RMCProfile's grid).
    n_sigma : kernel half-width in σ.

    Returns
    -------
    r : (nbins,) Å;  g : dict[(a, b)] -> (nbins,), a <= b alphabetically;
    info : dict with U (n,3,3) Å², u_rms per site, and pair counts.
    """
    cell = phonon.unitcell
    lattice = np.asarray(cell.cell, dtype=float)
    frac = np.asarray(cell.scaled_positions, dtype=float)
    symbols = list(cell.symbols)
    masses = np.asarray(cell.masses, dtype=float)
    n = len(frac)
    species = sorted(set(symbols))
    pair_keys = [(a, b) for i, a in enumerate(species) for b in species[i:]]
    key_index = {}
    for k, (a, b) in enumerate(pair_keys):
        key_index[(a, b)] = key_index[(b, a)] = k

    q, V, _ = scaled_modes(phonon, M, temperature, cutoff)
    U = site_covariances(V, masses)

    nbins = int(round(r_max / dr))
    r = dr * np.arange(1, nbins + 1)
    counts = np.zeros((len(pair_keys), nbins))
    cells = _cells_within(lattice, r_max)
    half = M // 2
    n_pairs = 0
    for j in range(n):
        row = row_correlations(q, V, frac, masses, j, M)
        dfrac = cells[:, None, :] + frac[None, :, :] - frac[j]     # (C, n, 3)
        dcart = dfrac @ lattice
        dist = np.linalg.norm(dcart, axis=-1)
        keep = (dist > 1e-6) & (dist <= r_max)
        ci, jp = np.nonzero(keep)
        d = dist[ci, jp]
        dhat = dcart[ci, jp] / d[:, None]
        C = U[j][None] + U[jp]
        nc = cells[ci]
        corr = np.all(np.abs(nc) < half, axis=1)
        idx = np.mod(nc[corr], M)
        Sg = row[idx[:, 0], idx[:, 1], idx[:, 2], jp[corr]]
        C[corr] -= Sg + np.transpose(Sg, (0, 2, 1))
        sig = np.sqrt(np.einsum("pi,pij,pj->p", dhat, C, dhat))
        site_key = np.array([key_index[(symbols[j], b)] for b in symbols])
        _accumulate(counts, r, dr, d, sig, site_key[jp], n_sigma)
        n_pairs += len(d)
    del V

    vol = abs(np.linalg.det(lattice))
    shell = 4.0 * np.pi * r**2 * dr
    g = {}
    for k, (a, b) in enumerate(pair_keys):
        na, nb = symbols.count(a), symbols.count(b)
        n_ord = na * nb if a == b else 2 * na * nb
        g[(a, b)] = vol * counts[k] / (n_ord * shell)
    u_rms = np.sqrt(np.trace(U, axis1=1, axis2=2) / 3.0)
    if log:
        log(f"  harmonic g(r): {n} sites, M = {M}, T = {temperature} K, "
            f"{n_pairs} pairs to {r_max} Å; u_rms per component "
            f"{u_rms.min():.4f}–{u_rms.max():.4f} Å")
    return r, g, {"U": U, "u_rms": u_rms, "n_pairs": n_pairs,
                  "pair_keys": pair_keys}


def _accumulate(counts, r, dr, d, sig, kidx, n_sigma, chunk=200_000):
    """Add each pair's radial profile, integrated per bin, into counts.

    Profile: (r/d)·N(r; d, σ) — the exact shell distribution of an isotropic
    3D Gaussian displacement for d ≫ σ (the (r+d) image term is dropped),
    evaluated at bin centres × dr. Pairs with σ below dr/2 are binned whole.
    """
    nbins = len(r)
    for s in range(0, len(d), chunk):
        dd, ss, kk = d[s:s + chunk], sig[s:s + chunk], kidx[s:s + chunk]
        ss = np.maximum(ss, 0.5 * dr)
        W = int(np.ceil(n_sigma * ss.max() / dr)) + 1
        centre = np.rint(dd / dr).astype(int) - 1          # r_k = (k+1)·dr
        off = np.arange(-W, W + 1)
        b = centre[:, None] + off[None, :]
        rb = (b + 1) * dr
        z = (rb - dd[:, None]) / ss[:, None]
        w = (rb / dd[:, None]) * np.exp(-0.5 * z * z) / (
            np.sqrt(2 * np.pi) * ss[:, None]) * dr
        w[np.abs(z) > n_sigma] = 0.0
        ok = (b >= 0) & (b < nbins)
        flat = (kk[:, None] * nbins + b)[ok]
        counts += np.bincount(flat, weights=w[ok],
                              minlength=counts.size).reshape(counts.shape)
