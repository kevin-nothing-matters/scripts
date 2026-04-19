#!/usr/bin/env python3
"""
first_law_k23.py  —  Issue #1 (Unified Open Issues)
====================================================
Extends Paper 2 §4 (first law δ⟨H_A⟩/δS = 1) from k=1 to k=2,3.

Strategy: use the rank-2 theorem.
  S_VN(ρ_A) = S_VN(ρ_carry)   for any aligned k-interval,
  where ρ_carry is the 2×2 carry qubit at the subtree root.

The modular Hamiltonian H_A = -log ρ_A acts on the full boundary,
but its spectrum is identical to -log ρ_carry.  Under perturbations of
gates on the PATH from the tree root to the subtree root:
  δλᵢ ≠ 0  →  δS ≠ 0  →  ratio is well-defined.
Under within-subtree perturbations:
  δλᵢ = 0  (rank-2 theorem)  →  δS = 0  →  indeterminate; excluded.

Expected: monotone convergence to 1 as D increases, identical to k=1.
"""

import numpy as np
import hashlib
import time
import argparse
import json
from datetime import datetime

EPS_ZERO = 1e-14


# ─── Gate generation (identical to gram_recursion_v6 spec) ──────────────────

def haar_u4(rng):
    """Haar-uniform U(4) via QR with phase correction."""
    Z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    ph = np.diag(R) / np.abs(np.diag(R))
    return Q * ph


def node_gate(tree_seed, start, end):
    """Deterministic Haar-random U(4) for node [start, end) via BLAKE2b-256."""
    key = f"{tree_seed}:{start}:{end}".encode()
    digest = hashlib.blake2b(key, digest_size=32).digest()
    seed64 = int.from_bytes(digest[:8], 'little') & 0x7FFFFFFFFFFFFFFF
    rng = np.random.default_rng(seed64)
    return haar_u4(rng)


# ─── Root state (stationary carry qubit) ────────────────────────────────────

def make_stationary_root(rng, warmup=20):
    """
    Warm-up iteration to find the fixed-point carry qubit state.
    Apply warmup random U(4) gates, each time tracing out ancilla,
    to converge to the stationary state of the Choi channel.
    """
    rho = np.array([[1, 0], [0, 0]], dtype=complex)  # start |0><0|
    anc0 = np.array([[1, 0], [0, 0]], dtype=complex)
    for _ in range(warmup):
        V = haar_u4(rng)
        rho4 = V @ np.kron(rho, anc0) @ V.conj().T
        # Trace out ancilla (qubit 1)
        rho = rho4.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)
        rho = (rho + rho.conj().T) / 2
        rho /= np.trace(rho).real
    return rho


# ─── Carry-qubit propagation ───────────────────────────────────────────────

def propagate_carry(tree_seed, N, k, interval_j, rho_root,
                    perturb_node=None, G=None, eps=0.0):
    """
    Propagate the carry qubit from root to the subtree root of
    aligned interval j at level k.

    If perturb_node=(ns,ne) and G are given, replaces the gate at that
    node with exp(i*eps*G) @ V before applying it.

    Returns the 2×2 carry density matrix at the subtree root.
    """
    subtree_start = interval_j * (1 << k)
    subtree_end   = (interval_j + 1) * (1 << k)
    ns, ne = 0, N
    rho = rho_root.copy()
    anc0 = np.array([[1, 0], [0, 0]], dtype=complex)

    while ns != subtree_start or ne != subtree_end:
        mid = (ns + ne) // 2
        V = node_gate(tree_seed, ns, ne)

        # Optional gate perturbation
        if perturb_node is not None and (ns, ne) == perturb_node:
            evals_g, evecs_g = np.linalg.eigh(G)
            expG = evecs_g @ np.diag(np.exp(1j * eps * evals_g)) @ evecs_g.conj().T
            V = expG @ V

        # Apply V to (carry ⊗ |0⟩)
        rho4 = V @ np.kron(rho, anc0) @ V.conj().T

        # Descend left or right
        if subtree_start < mid:
            # Left child: trace out qubit 1
            rho = rho4.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)
            ne = mid
        else:
            # Right child: trace out qubit 0
            rho = rho4.reshape(2, 2, 2, 2).trace(axis1=0, axis2=2)
            ns = mid

        rho = (rho + rho.conj().T) / 2
        rho /= np.trace(rho).real

    return rho


# ─── Entropy & modular Hamiltonian ──────────────────────────────────────────

def vn_entropy_bits(rho):
    """Von Neumann entropy in bits."""
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > EPS_ZERO]
    vals /= vals.sum()
    return -float(np.dot(vals, np.log2(vals)))


def modular_hamiltonian_bits(rho):
    """H = -log₂ ρ  (on support)."""
    evals, evecs = np.linalg.eigh(rho)
    log2_evals = np.where(evals > EPS_ZERO, -np.log2(evals), 0.0)
    return evecs @ np.diag(log2_evals) @ evecs.conj().T


# ─── Single-ratio computation ──────────────────────────────────────────────

def compute_ratio(tree_seed, N, k, interval_j, rho_root,
                  perturb_node, G, eps):
    """
    Compute δ⟨H_A⟩/δS for one (tree, interval, perturbation) triple.

    δ⟨H_A⟩ = Tr[δρ_carry · H_carry]  (bits)
    δS      = S_VN(ρ_carry') - S_VN(ρ_carry)  (bits)

    Returns None if |δS| < 1e-8 (near-degenerate perturbation).
    """
    rho0 = propagate_carry(tree_seed, N, k, interval_j, rho_root)
    rho1 = propagate_carry(tree_seed, N, k, interval_j, rho_root,
                           perturb_node=perturb_node, G=G, eps=eps)

    S0 = vn_entropy_bits(rho0)
    S1 = vn_entropy_bits(rho1)
    dS = S1 - S0
    if abs(dS) < 1e-8:
        return None

    H0 = modular_hamiltonian_bits(rho0)
    drho = rho1 - rho0
    dH = np.real(np.trace(drho @ H0))

    return dH / dS


# ─── Path nodes (gates on the root-to-subtree-root path) ────────────────────

def path_nodes(N, k, interval_j):
    """
    Return list of (ns, ne) for all gates on the path from root
    to the subtree root of aligned interval j at level k.
    These are the gates whose perturbation gives δS ≠ 0.
    """
    subtree_start = interval_j * (1 << k)
    subtree_end   = (interval_j + 1) * (1 << k)
    ns, ne = 0, N
    nodes = []
    while ns != subtree_start or ne != subtree_end:
        nodes.append((ns, ne))
        mid = (ns + ne) // 2
        if subtree_start < mid:
            ne = mid
        else:
            ns = mid
    return nodes


# ─── Batch test ─────────────────────────────────────────────────────────────

def run_test(D_list, k_list, n_trees=5, n_perturb=250,
             eps=0.001, warmup=20, seed=42):
    """Run the first law test across depths and interval sizes."""
    rng_master = np.random.default_rng(seed)
    all_results = {}

    print(f"{'D':>3}  {'k':>3}  {'mean':>12}  {'SE':>10}  "
          f"{'pull':>8}  {'n':>6}  {'filtered':>8}")
    print("-" * 65)

    for D in D_list:
        N = 1 << D
        for k in k_list:
            if k >= D:
                continue  # need at least 1 path step
            n_intervals = N // (1 << k)
            ratios = []
            n_filtered = 0

            for t in range(n_trees):
                tree_seed = int(rng_master.integers(0, 2**62))
                rho_root = make_stationary_root(
                    np.random.default_rng(rng_master.integers(0, 2**62)),
                    warmup=warmup)

                for _ in range(n_perturb):
                    # Random interval
                    j = int(rng_master.integers(0, n_intervals))
                    # Random path node to perturb
                    pnodes = path_nodes(N, k, j)
                    if not pnodes:
                        continue
                    pnode = pnodes[int(rng_master.integers(0, len(pnodes)))]
                    # Random 4×4 Hermitian traceless perturbation generator
                    G4 = rng_master.standard_normal((4, 4)) + \
                         1j * rng_master.standard_normal((4, 4))
                    G4 = (G4 + G4.conj().T) / 2
                    G4 -= np.eye(4) * np.trace(G4).real / 4

                    r = compute_ratio(tree_seed, N, k, j, rho_root,
                                      pnode, G4, eps)
                    if r is not None and abs(r) < 10:
                        ratios.append(r)
                    else:
                        n_filtered += 1

            if not ratios:
                print(f"{D:>3}  {k:>3}  {'---':>12}  {'---':>10}  "
                      f"{'---':>8}  {0:>6}  {n_filtered:>8}")
                continue

            arr = np.array(ratios)
            mean_r = arr.mean()
            se = arr.std(ddof=1) / np.sqrt(len(arr))
            pull = (mean_r - 1.0) / se if se > 0 else float('inf')
            all_results[(D, k)] = {
                'mean': float(mean_r), 'se': float(se),
                'pull': float(pull), 'n': len(arr),
                'filtered': n_filtered
            }
            print(f"{D:>3}  {k:>3}  {mean_r:>12.6f}  {se:>10.6f}  "
                  f"{pull:>+8.2f}σ  {len(arr):>6}  {n_filtered:>8}")

    return all_results


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="First law δ⟨H_A⟩/δS = 1 test at k=1,2,3")
    ap.add_argument("--mode", choices=["crosscheck", "extend", "full"],
                    default="full")
    ap.add_argument("--D_list", type=int, nargs="+", default=None)
    ap.add_argument("--k_list", type=int, nargs="+", default=None)
    ap.add_argument("--n_trees", type=int, default=5)
    ap.add_argument("--n_perturb", type=int, default=250)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()

    if args.D_list and args.k_list:
        results = run_test(
            D_list=args.D_list, k_list=args.k_list,
            n_trees=args.n_trees, n_perturb=args.n_perturb,
            eps=args.eps, warmup=args.warmup, seed=args.seed)
    elif args.mode == "crosscheck":
        print("=== Cross-check: Paper 2 §4 (k=1 only) ===")
        results = run_test(D_list=[2, 4, 6], k_list=[1],
                           n_trees=args.n_trees, n_perturb=args.n_perturb,
                           seed=args.seed)
    elif args.mode == "extend":
        print("=== Extension: k=2,3 ===")
        results = run_test(D_list=[4, 6, 8], k_list=[1, 2, 3],
                           n_trees=args.n_trees, n_perturb=args.n_perturb,
                           seed=args.seed)
    else:  # full
        print("=== Cross-check: Paper 2 §4 (k=1) ===")
        r1 = run_test(D_list=[2, 4, 6], k_list=[1],
                      n_trees=args.n_trees, n_perturb=args.n_perturb,
                      seed=args.seed)
        print()
        print("=== Extension: k=1,2,3 at D=4..10 ===")
        r2 = run_test(D_list=[4, 6, 8, 10], k_list=[1, 2, 3],
                      n_trees=args.n_trees, n_perturb=args.n_perturb,
                      seed=args.seed + 57)
        print()
        print("=== High-statistics at D=6,8 ===")
        r3 = run_test(D_list=[6, 8], k_list=[1, 2, 3],
                      n_trees=args.n_trees, n_perturb=500,
                      seed=args.seed + 777)
        results = {**r1, **r2, **r3}

    elapsed = time.time() - t0
    print(f"\nWall time: {elapsed:.1f}s")

    # Save JSON
    outfile = args.output or "first_law_k23_results.json"
    json_results = {
        f"D={k[0]}_k={k[1]}": v for k, v in results.items()
    }
    json_results["_meta"] = {
        "date": datetime.now().isoformat(),
        "args": vars(args),
        "wall_time_s": elapsed
    }
    with open(outfile, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {outfile}")
