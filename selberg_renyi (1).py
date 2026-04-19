#!/usr/bin/env python3
"""
selberg_renyi.py — Quenched Rényi-m free energy for the Quantum Family Tree
Kevin Donahue | March 2026

Numerically tests the Selberg continuation conjecture (§1.1 of open_issues):

    lim_{m→1}  F_que(m,k) / [k(m−1)]  =  m² · ln(5/2)  [nats]

where F_que(m,k) = E_tree[ −log Tr ρ_A^m ] is the quenched free energy
for a k-bond cut in the D→∞ tree, and m² = (1−η₂)(1−2η₂)/η₂ = 3/10.

Uses the SAME gate hash (BLAKE2b) and Choi recursion as gram_recursion_v6.py.
Results are directly comparable with the Paper 1 dataset.

Key physics
-----------
At m→1: F_que(m,k)/[k(m-1)] → E[S_VN]/k = (c_VN/3)/ln2 = m²·ln(5/2) [nats]
  This is confirmed at 10.34σ by Paper 1.  Selberg §1.1 asks for an analytic
  proof via the continuation of F_que(m,k) to non-integer m.

Annealed vs quenched:
  F_ann(m,k)/k = −log η_{2m} [exact via Weingarten]
  F_que(m,k)/k = E[−log Tr ρ_A^m]  [quenched, computed here]
  At m→1: both → E[S_VN]/k but from very different directions for m≠1.
  At m=2: F_ann = k·log(5/2), F_que < F_ann (Jensen gap = K-P correction).

Usage
-----
python3 selberg_renyi.py [OPTIONS]

  --D       int    20     Tree depth (converged by D≈10; 20 is generous)
  --N       int    200    Independent trees per m-value batch
  --npos    int    200    Random positions per tree per k
  --kmax    int    3      Max genealogical scale k = 1..kmax
  --warmup  int    20     Root-qubit warm-up steps
  --seed    int    42     RNG seed
  --m_dense flag         Dense m-grid near m=1 for sharper extrapolation
"""

from __future__ import annotations

import argparse
import hashlib
import time
from fractions import Fraction

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EPS    = 1e-14
M_SQ   = 3.0 / 10.0                    # m² = (1−η₂)(1−2η₂)/η₂
TARGET = M_SQ * np.log(5.0 / 2.0)      # lim F_que/[k(m-1)] in nats ≈ 0.27489

# Exact moments η_{2m} = E[c(V)^m] for U(4) — from §1.2
ETA_EXACT = {
    1: float(Fraction(2, 5)),
    2: float(Fraction(13, 70)),
    3: float(Fraction(55, 567)),
    4: float(Fraction(5213, 93555)),
}

# ─────────────────────────────────────────────────────────────────────────────
# Gate generation — IDENTICAL to gram_recursion_v6.py
# ─────────────────────────────────────────────────────────────────────────────

def haar_u4(rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))[np.newaxis, :]


def node_gate(tree_seed: int, start: int, end: int) -> np.ndarray:
    """Deterministic Haar-random U(4) — same BLAKE2b hash as v6."""
    h = hashlib.blake2b(
        f"{tree_seed}:{start}:{end}".encode(), digest_size=32
    ).digest()
    seed_int = int.from_bytes(h, "little") & 0xFFFFFFFFFFFFFFFF
    return haar_u4(np.random.default_rng(seed_int))


# ─────────────────────────────────────────────────────────────────────────────
# Choi-matrix helpers — IDENTICAL to gram_recursion_v6.py
# ─────────────────────────────────────────────────────────────────────────────

def _choi_build(n_in, n_out, T_fn):
    C = np.zeros((n_in * n_out, n_in * n_out), dtype=complex)
    for i in range(n_in):
        for j in range(n_in):
            sig = np.zeros((n_in, n_in), dtype=complex)
            sig[i, j] = 1.0
            C[i*n_out:(i+1)*n_out, j*n_out:(j+1)*n_out] = T_fn(sig)
    return C


def _apply_choi(C, rho, n_in=2):
    n_out = C.shape[0] // n_in
    out = np.zeros((n_out, n_out), dtype=complex)
    for i in range(n_in):
        for j in range(n_in):
            out += rho[i, j] * C[i*n_out:(i+1)*n_out, j*n_out:(j+1)*n_out]
    return out


def _channel_compose_tensor(C_L, C_R, V):
    n_L, n_R = C_L.shape[0] // 2, C_R.shape[0] // 2
    n_out = n_L * n_R

    def T_merged(sigma):
        rho4 = np.zeros((4, 4), dtype=complex)
        rho4[0,0]=sigma[0,0]; rho4[0,2]=sigma[0,1]
        rho4[2,0]=sigma[1,0]; rho4[2,2]=sigma[1,1]
        joint = V @ rho4 @ V.conj().T
        out = np.zeros((n_out, n_out), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        coeff = joint[i*2+j, k*2+l]
                        if abs(coeff) < EPS:
                            continue
                        sL = np.zeros((2,2), dtype=complex); sL[i,k] = 1.0
                        sR = np.zeros((2,2), dtype=complex); sR[j,l] = 1.0
                        out += coeff * np.kron(_apply_choi(C_L, sL),
                                               _apply_choi(C_R, sR))
        return out

    return _choi_build(2, n_out, T_merged)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy Choi recursion — IDENTICAL to gram_recursion_v6.py
# ─────────────────────────────────────────────────────────────────────────────

def gram_lazy(tree_seed, node_start, node_end, a_start, a_end,
              rho_in, _cache=None):
    if node_end <= a_start or node_start >= a_end:
        return None
    if _cache is None:
        _cache = {}
    cache_key = (node_start, node_end)
    if cache_key in _cache:
        return _cache[cache_key]
    if node_end - node_start == 1:
        result = _choi_build(2, 2, lambda r: r.copy())
        _cache[cache_key] = result
        return result

    mid = (node_start + node_end) // 2
    V   = node_gate(tree_seed, node_start, node_end)
    rho4 = np.zeros((4,4), dtype=complex)
    rho4[0,0]=rho_in[0,0]; rho4[0,2]=rho_in[0,1]
    rho4[2,0]=rho_in[1,0]; rho4[2,2]=rho_in[1,1]
    joint = V @ rho4 @ V.conj().T

    rho_L = np.array([[joint[0,0]+joint[1,1], joint[0,2]+joint[1,3]],
                      [joint[2,0]+joint[3,1], joint[2,2]+joint[3,3]]])
    rho_R = np.array([[joint[0,0]+joint[2,2], joint[0,1]+joint[2,3]],
                      [joint[1,0]+joint[3,2], joint[1,1]+joint[3,3]]])

    C_L = gram_lazy(tree_seed, node_start, mid, a_start, a_end, rho_L, _cache)
    C_R = gram_lazy(tree_seed, mid, node_end, a_start, a_end, rho_R, _cache)

    if C_L is None and C_R is None:
        return None

    if C_L is None:
        n_R = C_R.shape[0] // 2
        def T_r(sig):
            r4=np.zeros((4,4),dtype=complex)
            r4[0,0]=sig[0,0]; r4[0,2]=sig[0,1]
            r4[2,0]=sig[1,0]; r4[2,2]=sig[1,1]
            j_=V@r4@V.conj().T
            rR_=np.array([[j_[0,0]+j_[2,2],j_[0,1]+j_[2,3]],
                          [j_[1,0]+j_[3,2],j_[1,1]+j_[3,3]]])
            return _apply_choi(C_R, rR_)
        result = _choi_build(2, n_R, T_r)
        _cache[cache_key] = result
        return result

    if C_R is None:
        n_L = C_L.shape[0] // 2
        def T_l(sig):
            r4=np.zeros((4,4),dtype=complex)
            r4[0,0]=sig[0,0]; r4[0,2]=sig[0,1]
            r4[2,0]=sig[1,0]; r4[2,2]=sig[1,1]
            j_=V@r4@V.conj().T
            rL_=np.array([[j_[0,0]+j_[1,1],j_[0,2]+j_[1,3]],
                          [j_[2,0]+j_[3,1],j_[2,2]+j_[3,3]]])
            return _apply_choi(C_L, rL_)
        result = _choi_build(2, n_L, T_l)
        _cache[cache_key] = result
        return result

    result = _channel_compose_tensor(C_L, C_R, V)
    _cache[cache_key] = result
    return result


def make_root_qubit_stationary(rng, warmup=20):
    rho = np.zeros((2,2), dtype=complex); rho[0,0] = 1.0
    for _ in range(warmup):
        V = haar_u4(rng)
        rho4 = np.zeros((4,4), dtype=complex)
        rho4[0,0]=rho[0,0]; rho4[0,2]=rho[0,1]
        rho4[2,0]=rho[1,0]; rho4[2,2]=rho[1,1]
        out = V @ rho4 @ V.conj().T
        rho = np.array([[out[0,0]+out[1,1], out[0,2]+out[1,3]],
                        [out[2,0]+out[3,1], out[2,2]+out[3,3]]])
    return rho


def safe_herm(m):
    return 0.5 * (m + m.conj().T)


# ─────────────────────────────────────────────────────────────────────────────
# Rényi free energy  (NEW — the core addition over gram_recursion_v6)
# ─────────────────────────────────────────────────────────────────────────────

def renyi_free_energy_from_eigs(eigs: np.ndarray, m: float) -> float:
    """
    Compute  F_que(m) = −log Tr(ρ^m)  [nats]  from normalised eigenvalues.

    At m=1: Tr ρ^1 = 1 → F_que = 0.  The slope at m=1 is S_VN.
    For m≠1: F_que = −log(Σᵢ λᵢ^m).
    """
    eigs = eigs[eigs > EPS]
    if eigs.size == 0:
        return 0.0
    eigs = eigs / eigs.sum()
    tr_m = float(np.sum(eigs ** m))
    return -np.log(max(tr_m, 1e-300))


def svn_from_eigs(eigs: np.ndarray) -> float:
    """S_VN [nats] from normalised eigenvalues."""
    eigs = eigs[eigs > EPS]
    if eigs.size == 0:
        return 0.0
    eigs = eigs / eigs.sum()
    return float(-np.sum(eigs * np.log(eigs + 1e-300)))


# ─────────────────────────────────────────────────────────────────────────────
# Annealed reference
# ─────────────────────────────────────────────────────────────────────────────

def log_eta_m(m: float) -> float:
    """
    Estimate log η_{2m} = log E[c(V)^m] via cubic interpolation through
    the four exact integer moments η_2, η_4, η_6, η_8.
    """
    if abs(m - 1.0) < 1e-10:
        return np.log(ETA_EXACT[1])   # = log(2/5)
    m_int = int(round(m))
    if abs(m - m_int) < 1e-10 and m_int in ETA_EXACT:
        return np.log(ETA_EXACT[m_int])
    # Cubic in m through exact points m=1,2,3,4
    ms   = np.array([1.0, 2.0, 3.0, 4.0])
    lge  = np.array([np.log(ETA_EXACT[i]) for i in [1, 2, 3, 4]])
    poly = np.polyfit(ms, lge, 3)
    return float(np.polyval(poly, m))


# ─────────────────────────────────────────────────────────────────────────────
# Measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def measure_selberg(D, k_max, n_trees, n_pos, m_values, rng,
                   warmup=20, verbose=True):
    """
    Compute F_que(m,k) = E[-log Tr ρ_A^m] for all m in m_values, k=1..kmax.

    Returns dict:
      results[m]    = {'k_vals', 'F_que' [array], 'F_que_se' [array]}
      results['_svn'] = {'k_vals', 'means', 'stds'}
    """
    N_leaves = 1 << D
    k_vals   = [k for k in range(1, k_max + 1) if (1 << k) < N_leaves]

    # Per-tree accumulators
    tree_F  = {m: {k: [] for k in k_vals} for m in m_values}
    tree_svn = {k: [] for k in k_vals}

    t_start = time.time()

    for trial in range(n_trees):
        t0        = time.time()
        tree_seed = int(rng.integers(0, 2**62))
        rho_in    = make_root_qubit_stationary(rng, warmup)

        for k in k_vals:
            ell     = 1 << k
            max_pos = min(N_leaves - ell, (1 << 62) - 1)
            pos_arr = rng.integers(0, max_pos + 1, size=n_pos)

            # Per-position: one eigendecomposition, all m values
            f_per_pos  = {m: [] for m in m_values}
            svn_per_pos = []

            for s in pos_arr:
                C = gram_lazy(tree_seed, 0, N_leaves, int(s), int(s)+ell,
                              rho_in)
                if C is None:
                    for m in m_values:
                        f_per_pos[m].append(0.0)
                    svn_per_pos.append(0.0)
                    continue

                rho_A = safe_herm(_apply_choi(C, rho_in))
                eigs  = np.linalg.eigvalsh(rho_A).real
                eigs  = eigs[eigs > EPS]
                if eigs.size > 0:
                    eigs = eigs / eigs.sum()

                svn_per_pos.append(svn_from_eigs(eigs))
                for m in m_values:
                    f_per_pos[m].append(renyi_free_energy_from_eigs(eigs, m))

            for m in m_values:
                tree_F[m][k].append(float(np.mean(f_per_pos[m])))
            tree_svn[k].append(float(np.mean(svn_per_pos)))

        if verbose and ((trial + 1) % max(1, n_trees // 10) == 0
                        or trial == 0):
            elapsed = time.time() - t_start
            done    = trial + 1
            eta     = elapsed / done * (n_trees - done)
            # Spot-check: F_que(m,k=1)/(m-1) for m=0.9 and m=1.1
            def spot(m_val):
                data = tree_F[m_val][1]
                if not data: return '?'
                return f'{np.mean(data)/(m_val-1):.4f}'
            print(f"  tree {done:>4}/{n_trees}  "
                  f"this={time.time()-t0:.1f}s  ETA={eta:.0f}s  "
                  f"F/(Δm·k): m=0.9:{spot(0.9) if 0.9 in m_values else '-'}  "
                  f"m=1.1:{spot(1.1) if 1.1 in m_values else '-'}",
                  flush=True)

    # Aggregate
    results = {}
    for m in m_values:
        F   = np.array([np.mean(tree_F[m][k]) for k in k_vals])
        Fse = np.array([
            np.std(tree_F[m][k], ddof=1) / np.sqrt(max(len(tree_F[m][k]), 2))
            for k in k_vals
        ])
        results[m] = dict(k_vals=k_vals, F_que=F, F_que_se=Fse)

    svn_m = np.array([np.mean(tree_svn[k]) for k in k_vals])
    svn_s = np.array([
        np.std(tree_svn[k], ddof=1) / np.sqrt(max(len(tree_svn[k]), 2))
        for k in k_vals
    ])
    results['_svn'] = dict(k_vals=k_vals, means=svn_m, stds=svn_s)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# WLS slope helper
# ─────────────────────────────────────────────────────────────────────────────

def wls_slope(k_vals, values, errors):
    k = np.array(k_vals, dtype=float)
    w = 1.0 / (np.array(errors)**2 + 1e-20)
    W = w.sum(); Wk = (w*k).sum(); Wk2 = (w*k**2).sum()
    Wv = (w*values).sum(); Wkv = (w*k*values).sum()
    det = W*Wk2 - Wk**2
    if det < 1e-20:
        return float('nan'), float('nan')
    a  = (W*Wkv - Wk*Wv) / det
    sa = np.sqrt(W / det)
    return float(a), float(sa)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results, m_values, D, n_trees, n_pos):
    k_vals = results[m_values[0]]['k_vals']

    print()
    print("=" * 72)
    print(f"Selberg continuation  |  D={D}  N={n_trees}  npos={n_pos}")
    print("=" * 72)
    print(f"Conjecture:  lim_{{m→1}} F_que(m,k)/[k(m-1)] = "
          f"m²·ln(5/2) = {TARGET:.8f} nats")
    print(f"Annealed:    lim_{{m→1}} F_ann(m,k)/[k(m-1)] = "
          f"-E[log c] ≈ 1.00619 nats")
    print(f"Suppression factor m² = {M_SQ}  (β₁=1/10 confirmed Paper 1)")
    print()

    # ── S_VN check ────────────────────────────────────────────────
    svn = results['_svn']
    a_svn, sa_svn = wls_slope(k_vals, svn['means'], svn['stds'])
    print(f"S_VN slope a (nats/k):  {a_svn:.6f} ± {sa_svn:.6f}")
    print(f"  In bits:  {a_svn/np.log(2):.6f} ± {sa_svn/np.log(2):.6f}")
    print(f"  Theory:   {TARGET:.6f} nats  [{(a_svn-TARGET)/sa_svn:+.2f}σ]")
    print()

    # ── Main table: F_que(m,k)/[k(m-1)] vs m ─────────────────────
    print(f"{'m':>5}  {'a(m) [nats/k]':>14}  {'a(m)/(m-1)':>12}  "
          f"{'a_ann/(m-1)':>12}  {'diff vs tgt':>12}")
    print("-" * 60)

    ratio_quench  = []
    ratio_anneal  = []

    for m in m_values:
        r = results[m]
        a_m, sa_m = wls_slope(k_vals, r['F_que'], r['F_que_se'])
        a_ann = -log_eta_m(m)    # F_ann/k = -log(eta_{2m})

        dm = m - 1.0
        if abs(dm) < 0.005:
            q_str = f"    →{a_svn:.5f}"
            n_str = f"    →{1.00619:.5f}"
        else:
            rq = a_m / dm
            ra = a_ann / dm
            q_str = f"{rq:12.5f}"
            n_str = f"{ra:12.5f}"
            ratio_quench.append((m, rq))
            ratio_anneal.append((m, ra))
            diff_str = f"{rq - TARGET:+.5f}"
            print(f"{m:5.2f}  {a_m:14.6f}  {q_str}  {n_str}  {diff_str}")
            continue

        print(f"{m:5.2f}  {a_m:14.6f}  {q_str}  {n_str}  (m=1 limit)")

    # ── Extrapolation to m=1 ──────────────────────────────────────
    print()
    print("Extrapolation of a(m)/(m-1) to m=1:")

    left  = [(m, r) for m, r in ratio_quench if m < 1.0]
    right = [(m, r) for m, r in ratio_quench if m > 1.0]

    if len(left) >= 2:
        ms, rs = zip(*left[-2:])
        slope  = (rs[-1] - rs[-2]) / (ms[-1] - ms[-2])
        extrap = rs[-1] + slope * (1.0 - ms[-1])
        print(f"  From below (m<1): {extrap:.6f} nats  "
              f"[Δ = {extrap - TARGET:+.6f},  target = {TARGET:.6f}]")

    if len(right) >= 2:
        ms, rs = zip(*right[:2])
        slope  = (rs[1] - rs[0]) / (ms[1] - ms[0])
        extrap = rs[0] - slope * (ms[0] - 1.0)
        print(f"  From above (m>1): {extrap:.6f} nats  "
              f"[Δ = {extrap - TARGET:+.6f},  target = {TARGET:.6f}]")

    if left and right:
        avg = (left[-1][1] + right[0][1]) / 2
        print(f"  Bracket midpoint: {avg:.6f} nats  "
              f"[Δ = {avg - TARGET:+.6f}]")

    # ── Quenched vs annealed gap ───────────────────────────────────
    print()
    print("Quenched vs Annealed gap  [F_ann/k − F_que/k]  [nats]:")
    print(f"  (Jensen inequality: gap ≥ 0 for m > 1, ≤ 0 for m < 1)")
    print(f"  {'m':>5}  {'F_que/k':>10}  {'F_ann/k':>10}  {'gap':>10}")
    for m in m_values:
        r = results[m]
        a_m, _ = wls_slope(k_vals, r['F_que'], r['F_que_se'])
        a_ann   = -log_eta_m(m)
        gap     = a_ann - a_m
        print(f"  {m:5.2f}  {a_m:10.5f}  {a_ann:10.5f}  {gap:+10.5f}")

    print()
    print(f"At m=2 (exact checks):")
    r2 = results.get(2.0) or results.get(2)
    if r2:
        a2, sa2 = wls_slope(k_vals, r2['F_que'], r2['F_que_se'])
        a_ann2  = -np.log(ETA_EXACT[2])   # = log(70/13) ≈ 1.684
        print(f"  F_que(2,k)/k  = {a2:.6f} ± {sa2:.6f} nats  (quenched)")
        print(f"  F_ann(2,k)/k  = {a_ann2:.6f} nats  (annealed = log(1/η₄) exact)")
        print(f"  Jensen gap    = {a_ann2 - a2:+.6f} nats")
        print(f"  F_que/(m-1)   = {a2:.6f} nats/k  "
              f"[target {TARGET:.6f}, ratio {a2/TARGET:.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Selberg §1.1: quenched Rényi-m free energy")
    parser.add_argument("--D",       type=int,  default=20)
    parser.add_argument("--N",       type=int,  default=200)
    parser.add_argument("--npos",    type=int,  default=200)
    parser.add_argument("--kmax",    type=int,  default=3)
    parser.add_argument("--warmup",  type=int,  default=20)
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--m_dense", action="store_true",
                        help="Dense m-grid near 1 (slower, sharper extrap)")
    args = parser.parse_args()

    if args.m_dense:
        m_values = sorted({
            0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94,
            0.96, 0.98, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20,
            1.30, 1.50, 2.00
        })
    else:
        m_values = [0.50, 0.70, 0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 1.50, 2.00]

    rng = np.random.default_rng(args.seed)

    print(f"selberg_renyi.py  |  D={args.D}  N={args.N}  "
          f"npos={args.npos}  kmax={args.kmax}  seed={args.seed}")
    print(f"m values ({len(m_values)}): {m_values}")
    print(f"Target: m²·ln(5/2) = {TARGET:.8f} nats")
    print()

    results = measure_selberg(
        D=args.D, k_max=args.kmax, n_trees=args.N,
        n_pos=args.npos, m_values=m_values, rng=rng,
        warmup=args.warmup, verbose=True
    )

    print_report(results, m_values, args.D, args.N, args.npos)


if __name__ == "__main__":
    main()
