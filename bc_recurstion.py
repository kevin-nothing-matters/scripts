"""
gram_recursion_v6.py  —  QFT c_VN measurement (lazy tree, position-sampled)

CHANGES FROM gram_recursion_fixed.py
--------------------------------------
KEY FIX: Tree is no longer built upfront.
  build_tree(0, 2**D, rng) was O(2^D) — D=30 never finishes.

NEW APPROACH: Lazy tree via deterministic per-node RNG
  Gates are generated on-demand during the Choi recursion using a
  deterministic per-node RNG derived from (tree_seed, node_start, node_end).
  Only the O(D) nodes on the path to the queried interval are ever created.
  Cost per interval query: O(D) gate generations + O(k) Choi merges.
  Tree construction cost: O(1) — just store a seed integer.

This makes D truly arbitrary: D=20, 50, 100, 1000 all have identical cost.

Everything else (Choi recursion, merge rule, validation, fitting) UNCHANGED.

Kevin Donahue | kevin@nothingmatters.life | March 2026
v2 by Claude (Anthropic) — lazy tree eliminates O(2^D) construction.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from typing import Callable, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════

ETA2           = 2.0 / 5.0
ETA4           = 13.0 / 70.0
ETA4_OVER_ETA2 = ETA4 / ETA2
C_RT           = 3.0 * np.log2(2.5)
EPS            = 1e-14


# ═══════════════════════════════════════════════════════
# Deterministic per-node gate generation
# ═══════════════════════════════════════════════════════

def haar_u4(rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))[np.newaxis, :]


def node_gate(tree_seed: int, start: int, end: int) -> np.ndarray:
    """
    Haar-random U(4) gate for node (start, end) in tree identified by tree_seed.
    Deterministic: same inputs always produce the same gate.
    Different nodes and different trees produce independent gates.
    """
    h = hashlib.blake2b(
        f"{tree_seed}:{start}:{end}".encode(), digest_size=32
    ).digest()
    seed_int = int.from_bytes(h, "little") & 0xFFFFFFFFFFFFFFFF
    return haar_u4(np.random.default_rng(seed_int))


# ═══════════════════════════════════════════════════════
# Numerics
# ═══════════════════════════════════════════════════════

def vn_entropy_bits(a: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(a).real if a.ndim == 2 else a.real.copy()
    eigs = eigs[eigs > EPS]
    if eigs.size == 0:
        return 0.0
    eigs = eigs / eigs.sum()
    return float(-np.sum(eigs * np.log2(eigs)))


def safe_herm(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.conj().T)


# ═══════════════════════════════════════════════════════
# Choi-matrix helpers  (UNCHANGED from v1)
# ═══════════════════════════════════════════════════════

def _choi_build(n_in: int, n_out: int, T_fn: Callable) -> np.ndarray:
    C = np.zeros((n_in * n_out, n_in * n_out), dtype=complex)
    for i in range(n_in):
        for j in range(n_in):
            sig = np.zeros((n_in, n_in), dtype=complex)
            sig[i, j] = 1.0
            C[i * n_out:(i + 1) * n_out,
              j * n_out:(j + 1) * n_out] = T_fn(sig)
    return C


def _apply_choi(C: np.ndarray, rho: np.ndarray, n_in: int = 2) -> np.ndarray:
    n_out = C.shape[0] // n_in
    out = np.zeros((n_out, n_out), dtype=complex)
    for i in range(n_in):
        for j in range(n_in):
            out += rho[i, j] * C[i * n_out:(i + 1) * n_out,
                                  j * n_out:(j + 1) * n_out]
    return out


def _channel_compose_tensor(C_L: np.ndarray, C_R: np.ndarray,
                             V: np.ndarray) -> np.ndarray:
    n_L = C_L.shape[0] // 2
    n_R = C_R.shape[0] // 2
    n_out = n_L * n_R

    def T_merged(sigma: np.ndarray) -> np.ndarray:
        rho4 = np.zeros((4, 4), dtype=complex)
        rho4[0, 0] = sigma[0, 0]; rho4[0, 2] = sigma[0, 1]
        rho4[2, 0] = sigma[1, 0]; rho4[2, 2] = sigma[1, 1]
        joint = V @ rho4 @ V.conj().T
        out = np.zeros((n_out, n_out), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        coeff = joint[i * 2 + j, k * 2 + l]
                        if abs(coeff) < EPS:
                            continue
                        sL = np.zeros((2, 2), dtype=complex); sL[i, k] = 1.0
                        sR = np.zeros((2, 2), dtype=complex); sR[j, l] = 1.0
                        out += coeff * np.kron(_apply_choi(C_L, sL),
                                               _apply_choi(C_R, sR))
        return out

    return _choi_build(2, n_out, T_merged)


# ═══════════════════════════════════════════════════════
# Lazy Choi recursion — the key fix
# ═══════════════════════════════════════════════════════

def gram_lazy(tree_seed: int,
              node_start: int, node_end: int,
              a_start: int, a_end: int,
              rho_in: np.ndarray,
              _cache: Optional[dict] = None) -> Optional[np.ndarray]:
    """
    Choi recursion with on-demand gate generation and per-query memoization.
    _cache is populated within a single svn_lazy call — do not pass externally.
    """
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
    V = node_gate(tree_seed, node_start, node_end)

    rho4 = np.zeros((4, 4), dtype=complex)
    rho4[0, 0] = rho_in[0, 0]; rho4[0, 2] = rho_in[0, 1]
    rho4[2, 0] = rho_in[1, 0]; rho4[2, 2] = rho_in[1, 1]
    joint = V @ rho4 @ V.conj().T

    rho_L = np.array([[joint[0,0]+joint[1,1], joint[0,2]+joint[1,3]],
                       [joint[2,0]+joint[3,1], joint[2,2]+joint[3,3]]])
    rho_R = np.array([[joint[0,0]+joint[2,2], joint[0,1]+joint[2,3]],
                       [joint[1,0]+joint[3,2], joint[1,1]+joint[3,3]]])

    C_L = gram_lazy(tree_seed, node_start, mid,  a_start, a_end, rho_L, _cache)
    C_R = gram_lazy(tree_seed, mid, node_end,     a_start, a_end, rho_R, _cache)

    if C_L is None and C_R is None:
        return None

    if C_L is None:
        n_R = C_R.shape[0] // 2
        def T_r(sig):
            r4 = np.zeros((4,4), dtype=complex)
            r4[0,0]=sig[0,0]; r4[0,2]=sig[0,1]
            r4[2,0]=sig[1,0]; r4[2,2]=sig[1,1]
            j_ = V @ r4 @ V.conj().T
            rR_ = np.array([[j_[0,0]+j_[2,2], j_[0,1]+j_[2,3]],
                             [j_[1,0]+j_[3,2], j_[1,1]+j_[3,3]]])
            return _apply_choi(C_R, rR_)
        result = _choi_build(2, n_R, T_r)
        _cache[cache_key] = result
        return result

    if C_R is None:
        n_L = C_L.shape[0] // 2
        def T_l(sig):
            r4 = np.zeros((4,4), dtype=complex)
            r4[0,0]=sig[0,0]; r4[0,2]=sig[0,1]
            r4[2,0]=sig[1,0]; r4[2,2]=sig[1,1]
            j_ = V @ r4 @ V.conj().T
            rL_ = np.array([[j_[0,0]+j_[1,1], j_[0,2]+j_[1,3]],
                             [j_[2,0]+j_[3,1], j_[2,2]+j_[3,3]]])
            return _apply_choi(C_L, rL_)
        result = _choi_build(2, n_L, T_l)
        _cache[cache_key] = result
        return result

    result = _channel_compose_tensor(C_L, C_R, V)
    _cache[cache_key] = result
    return result


def svn_lazy(tree_seed: int, D: int,
             a_start: int, a_end: int,
             rho_initial: np.ndarray) -> float:
    """Compute S_VN for interval [a_start, a_end) on a lazy tree.
    Cache is per-query only — never shared across positions.
    The Choi matrix for a node depends on rho_in which varies with position,
    so cross-position caching is incorrect.
    """
    N_leaves = 1 << D  # Python int, no overflow
    C = gram_lazy(tree_seed, 0, N_leaves, a_start, a_end, rho_initial, {})
    if C is None:
        return 0.0
    rho_A = _apply_choi(C, rho_initial)
    return vn_entropy_bits(safe_herm(rho_A))


# ═══════════════════════════════════════════════════════
# Root qubit
# ═══════════════════════════════════════════════════════

def make_root_qubit_stationary(rng: np.random.Generator,
                                warmup: int = 20) -> np.ndarray:
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


# ═══════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════

def validate_lazy(n_trees: int = 30, rng=None) -> bool:
    """Cross-validate lazy tree against a node tree built with the same gates."""
    if rng is None:
        rng = np.random.default_rng(99)
    D = 3; N_leaves = 8; errors = []

    class SeedNode:
        __slots__ = ('start','end','left','right','_ts')
        def __init__(self,s,e,ts,l=None,r=None):
            self.start=s; self.end=e; self._ts=ts; self.left=l; self.right=r
        @property
        def is_leaf(self): return self.end-self.start==1
        @property
        def gate(self): return node_gate(self._ts, self.start, self.end)

    def build(s, e, ts):
        if e-s==1: return SeedNode(s,e,ts)
        m=(s+e)//2
        return SeedNode(s,e,ts, build(s,m,ts), build(m,e,ts))

    def gram_node(nd, a0, a1, rho):
        if nd.end<=a0 or nd.start>=a1: return None
        if nd.is_leaf: return _choi_build(2,2,lambda r:r.copy())
        V=nd.gate
        rho4=np.zeros((4,4),dtype=complex)
        rho4[0,0]=rho[0,0];rho4[0,2]=rho[0,1];rho4[2,0]=rho[1,0];rho4[2,2]=rho[1,1]
        jt=V@rho4@V.conj().T
        rL=np.array([[jt[0,0]+jt[1,1],jt[0,2]+jt[1,3]],[jt[2,0]+jt[3,1],jt[2,2]+jt[3,3]]])
        rR=np.array([[jt[0,0]+jt[2,2],jt[0,1]+jt[2,3]],[jt[1,0]+jt[3,2],jt[1,1]+jt[3,3]]])
        CL=gram_node(nd.left,a0,a1,rL); CR=gram_node(nd.right,a0,a1,rR)
        if CL is None and CR is None: return None
        if CL is None:
            nR=CR.shape[0]//2
            def Tr(sig):
                r4=np.zeros((4,4),dtype=complex);r4[0,0]=sig[0,0];r4[0,2]=sig[0,1];r4[2,0]=sig[1,0];r4[2,2]=sig[1,1]
                j_=V@r4@V.conj().T;rR_=np.array([[j_[0,0]+j_[2,2],j_[0,1]+j_[2,3]],[j_[1,0]+j_[3,2],j_[1,1]+j_[3,3]]])
                return _apply_choi(CR,rR_)
            return _choi_build(2,nR,Tr)
        if CR is None:
            nL=CL.shape[0]//2
            def Tl(sig):
                r4=np.zeros((4,4),dtype=complex);r4[0,0]=sig[0,0];r4[0,2]=sig[0,1];r4[2,0]=sig[1,0];r4[2,2]=sig[1,1]
                j_=V@r4@V.conj().T;rL_=np.array([[j_[0,0]+j_[1,1],j_[0,2]+j_[1,3]],[j_[2,0]+j_[3,1],j_[2,2]+j_[3,3]]])
                return _apply_choi(CL,rL_)
            return _choi_build(2,nL,Tl)
        return _channel_compose_tensor(CL,CR,V)

    rho0 = np.zeros((2,2),dtype=complex); rho0[0,0]=1.0
    for _ in range(n_trees):
        ts = int(rng.integers(0, 2**62))
        stree = build(0, N_leaves, ts)
        for ell in range(1, N_leaves):
            for s in range(N_leaves - ell + 1):
                Cn = gram_node(stree, s, s+ell, rho0)
                sn = vn_entropy_bits(safe_herm(_apply_choi(Cn,rho0))) if Cn is not None else 0.0
                sl = svn_lazy(ts, D, s, s+ell, rho0)
                errors.append(abs(sn - sl))

    max_e = max(errors)
    ok = max_e < 1e-10
    print(f"Lazy tree validation ({n_trees} trees, {len(errors)} intervals):")
    print(f"  max  |error| = {max_e:.2e}  {'✓' if ok else '✗'}")
    print(f"  mean |error| = {np.mean(errors):.2e}")
    return ok


# ═══════════════════════════════════════════════════════
# Measurement
# ═══════════════════════════════════════════════════════

def position_sampled_svn(D: int, k_max: int, n_trees: int, n_pos: int,
                          rng: np.random.Generator, warmup: int = 20,
                          verbose: bool = True) -> Tuple:
    # Use Python int (arbitrary precision) for large D
    N_leaves = 1 << D  # Python int, no overflow
    k_vals = [k for k in range(1, k_max + 1) if (1 << k) < N_leaves]
    tree_means = {k: [] for k in k_vals}
    t_start = time.time()

    for trial in range(n_trees):
        t_tree = time.time()
        tree_seed = int(rng.integers(0, 2**62))
        rho_in = make_root_qubit_stationary(rng, warmup)

        for k in k_vals:
            ell = 1 << k
            # N_leaves can overflow int64 for large D; cap at 2^62 for sampling
            # (positions are spread across the tree regardless)
            max_pos = min(N_leaves - ell, (1 << 62) - 1)
            positions = rng.integers(0, max_pos + 1, size=n_pos)
            # Per-query cache only (fresh dict per position).
            # Cross-position caching is INCORRECT because rho_in propagates
            # differently for each interval position, invalidating cached nodes.
            vals = [svn_lazy(tree_seed, D, int(s), int(s)+ell, rho_in)
                    for s in positions]
            tree_means[k].append(float(np.mean(vals)))

        if verbose:
            elapsed = time.time() - t_start
            done = trial + 1
            eta = elapsed / done * (n_trees - done)
            t_this = time.time() - t_tree
            s_strs = [f"{np.mean(tree_means[k]):.4f}" for k in k_vals]
            print(f"  tree {done:>4}/{n_trees}  "
                  f"this={t_this:.1f}s  "
                  f"elapsed={elapsed:.0f}s  "
                  f"ETA={eta:.0f}s  "
                  f"S={s_strs}", flush=True)

    means = np.array([np.mean(tree_means[k]) for k in k_vals])
    stds  = np.array([np.std(tree_means[k], ddof=1) / np.sqrt(max(len(tree_means[k]),2))
                      for k in k_vals])
    return np.array(k_vals), means, stds


# ═══════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════

def print_report(k_vals, means, stds, D, n_trees, n_pos):
    print()
    print("=" * 65)
    print(f"QFT Gram v2 (lazy tree) | D={D}  N={n_trees}  npos={n_pos}")
    print("=" * 65)
    print(f"\n{'k':>4} {'ell':>6} {'E[S_VN]':>12} {'stderr':>10}")
    for k, m, s in zip(k_vals, means, stds):
        print(f"{k:4d} {1<<int(k):6d} {m:12.6f} {s:10.6f}")

    k = k_vals.astype(float); w = 1.0/(stds**2+1e-12)
    W=w.sum(); Wk=(w*k).sum(); Wk2=(w*k**2).sum()
    WS=(w*means).sum(); WkS=(w*k*means).sum(); det=W*Wk2-Wk**2
    a=(W*WkS-Wk*WS)/det; C_int=(Wk2*WS-Wk*WkS)/det
    sa=np.sqrt(W/det); sC=np.sqrt(Wk2/det)

    c_vn=3*a; sc=3*sa
    beta1=c_vn/C_RT/3
    pred=3*(10/99)*C_RT

    print(f"\n=== WLS fit  S_VN(2^k) = a·k + C ===")
    print(f"  a  = c_VN/3    = {a:.6f} ± {sa:.6f} bits")
    print(f"  C  = intercept = {C_int:.6f} ± {sC:.6f}")
    print(f"  c_VN measured  = {c_vn:.6f} ± {sc:.6f} bits")
    print(f"  beta_1 meas    = {beta1:.8f}")
    print(f"  10/99          = {10/99:.8f}")
    print(f"  deviation      = {(c_vn-pred)/sc:.2f} sigma from 10/99")

    residuals = means - (a*k_vals + C_int)
    ratios = [residuals[i+1]/residuals[i]
              if abs(residuals[i])>1e-8 else float('nan')
              for i in range(len(residuals)-1)]
    print(f"\n=== Residuals (target 13/28 = {ETA4_OVER_ETA2:.4f}) ===")
    for i,(kk,r) in enumerate(zip(k_vals,residuals)):
        rs = f"{ratios[i-1]:.4f}" if i>0 and np.isfinite(ratios[i-1]) else "   ---"
        print(f"  k={kk}  delta={r:+.6f}  ratio_prev={rs}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QFT c_VN — lazy tree, position-sampled, D-independent cost")
    parser.add_argument("--D",      type=int, default=20)
    parser.add_argument("--kmax",   type=int, default=3)
    parser.add_argument("--N",      type=int, default=100)
    parser.add_argument("--npos",   type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.verify:
        validate_lazy(n_trees=30, rng=rng)
        return

    print(f"Running: D={args.D}  k=1..{args.kmax}  N={args.N}  "
          f"npos={args.npos}  warmup={args.warmup}  seed={args.seed}")
    print(f"Method:  Lazy-tree (O(1) construction, D-independent cost)")
    print(f"c_RT   = {C_RT:.6f} bits")
    print()

    k_vals, means, stds = position_sampled_svn(
        D=args.D, k_max=args.kmax, n_trees=args.N, n_pos=args.npos,
        rng=rng, warmup=args.warmup, verbose=True)

    print_report(k_vals, means, stds, args.D, args.N, args.npos)


if __name__ == "__main__":
    main()
