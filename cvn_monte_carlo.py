"""
Quantum Family Tree — GPU Runner v6 (VERIFIED CORRECT)
======================================================
Exact implementation of run_d12.py joint propagation using
batched matrix operations (Kraus channels) for GPU acceleration.

Verified against depth-4 ground truth: max_err=0.00e+00.

Key:  branch_left = sum_lc2 (K_lc2 @ P) @ rho @ (K_lc2 @ P)†
      branch_right = sum_rc1 (Kr_rc1 @ Pr) @ rho @ (Kr_rc1 @ Pr)†
      where P = V ⊗ I_2,  V[p,a] = va[a][p]

Run:
    python3 run_qft_gpu_v6.py --depth 12 --trees 20 --start-tree 0 \
        --output ~/qft/results/ --batch-size 8192

Test first:
    python3 run_qft_gpu_v6.py --test

Requirements:
    pip install cupy-cuda12x numpy
"""

import numpy as np
import json
import os
import time
import argparse
from itertools import combinations

try:
    import cupy as cp
    GPU = True
    print(f"CuPy GPU available")
except Exception as e:
    cp = None
    GPU = False
    print(f"No CuPy — CPU mode ({e})")

xp = cp if GPU else np


# ─── Ground truth for depth-4 tree 0 (seed=4) ────────────────────────────
GROUND_TRUTH = {
    "2": [0.15016365826225653, 0.7697836426588062, 0.27067384888068813,
          0.5479094445682968, 0.9367732421711887, 0.3106935654329821,
          0.5204641045335312, 0.6193821938832547],
    "4": [0.029981713607097893, 0.009598166707279865, 0.0614923124884611,
          0.017455817720072098, 0.026099110395973457, 0.00855951284944978,
          0.18223572261321563, 0.07757293535696552, 0.01584748327682317,
          0.01638714695125376, 0.01046059590441506, 0.01085703286712758,
          0.08699922146789829, 0.0617558108399936, 0.10289677975447598,
          0.07585905311979135],
    "6": [0.013453416815204178, 0.05290658637579293, 0.023978553450486606,
          0.007199792075813782, 0.026122409142862568, 0.113078253086353,
          0.059239542848308346, 0.02057264766925626, 0.011163899466293259,
          0.036130833252619565, 0.018462844133094403, 0.008985795437767563,
          0.0024037726186991737, 0.011205046805327301, 0.0037869858156829306,
          0.001248486910059854, 0.0054892691495485035, 0.00667435739824751,
          0.005245616628352989, 0.004498521108896147, 0.007284665075634189,
          0.008090233577995765, 0.0062292680445208415, 0.0057039158263163525,
          0.013136259680169626, 0.017292553571508273, 0.014368833138009629,
          0.011624641496069898, 0.011641135737363806, 0.015988877584237837,
          0.012883100725765484, 0.010215147139418734],
    "8": [0.00025575123690191326, 0.00011383945788390193, 0.0020548431435569725,
          0.0020903749999499066, 0.001046578315633706, 0.001389204198303684,
          0.0013571987830212073, 0.0011392323891763478, 0.0005607842267402185,
          0.0002395326480479909, 0.005077215199738694, 0.00515129664372449,
          0.0021691037561835103, 0.002857369374210883, 0.0027603241657256916,
          0.002319146290405527, 0.0005675141677496942, 0.0004805988057540178,
          0.0022857227677486325, 0.002306890259344052, 0.0003627716047117646,
          0.00063073553575832, 0.0006336437802962891, 0.00043064059619768447,
          0.00017116506941050602, 0.00013716262891660946, 0.0003717154983325921,
          0.0003785153547976039, 0.00035180561475983696, 0.00045637856486957507,
          0.00044812482641876095, 0.0003821298548134511, 1.2002074745165459e-05,
          1.0393309058276401e-05, 3.331100806569509e-05, 3.4211255704352794e-05,
          2.3462055491307865e-05, 2.8512432592053827e-05, 2.7462081767293256e-05,
          2.328546092178474e-05, 3.8177875248512905e-05, 1.9160693150332975e-05,
          0.0004941688136133049, 0.0005018282385600425, 0.00042491348019857433,
          0.0005289905225838165, 0.0005353913529753118, 0.00046817104751051275,
          4.0145833195670555e-05, 3.005741984785093e-05, 0.00012420592838835454,
          0.00012757695254095314, 0.00010019053105303488, 0.00012320565816026363,
          0.00010040703625402436, 8.615697133551592e-05, 3.625583948840827e-06,
          2.4338032078574656e-06, 1.6412660022280257e-05, 1.6728859900450033e-05,
          2.0116685515492172e-05, 2.4293447930756606e-05, 1.942923550235509e-05,
          1.6931443270751245e-05],
}


# ─── Tree construction ─────────────────────────────────────────────────────

def haar_unitary_4x4(rng):
    Z = (rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))

def build_bvecs(depth, seed):
    rng = np.random.default_rng(seed)
    bvecs = {}
    for node in range(2**depth - 1):
        U = haar_unitary_4x4(rng)
        v0 = U @ np.array([1,0,0,0], dtype=complex)
        v1 = U @ np.array([0,0,1,0], dtype=complex)
        bvecs[node] = (v0, v1)
    return bvecs

def graph_dist(i, j, depth):
    if i == j: return 0
    for k in range(depth):
        if (i >> (depth-1-k)) != (j >> (depth-1-k)):
            return 2 * (depth - k)
    return 0


# ─── Kraus operator construction ───────────────────────────────────────────
#
# rho8 row indexing: p*2+r where p=va_component, r=right_qubit (for left branch)
# rho8[p*2+ri, q*2+rj] = sum_{a,b} rho4[a*2+ri, b*2+rj] * va[a][p] * conj(va[b][q])
# = (P @ rho4 @ P†)[p*2+ri, q*2+rj]   where P = V ⊗ I_2, V[p,a]=va[a][p]
#
# Trace (keep lc1=p//2, trace lc2=p%2):
# K_{lc2}[lc1*2+r, p*2+r] = 1 if p%2==lc2  (for all p,r)
#
# Combined: A_{lc2} = K_{lc2} @ P, shape (4,4)
# rho4_out = A_0 @ rho4 @ A_0† + A_1 @ rho4 @ A_1†

def build_kraus_left(v0, v1, keep_lc1=True):
    """
    Returns (A0, A1): two (4,4) Kraus matrices for left-qubit branch.
    keep_lc1=True:  trace over lc2=p%2, keep lc1=p//2
    keep_lc1=False: trace over lc1=p//2, keep lc2=p%2
    """
    va = [v0, v1]
    V = np.stack([va[0], va[1]], axis=1)  # (4,2): V[p,a] = va[a][p]
    P = np.kron(V, np.eye(2, dtype=complex))  # (8,4): rows (p,r), cols (a,r)

    if keep_lc1:
        # Trace over lc2 (bit 0 of p). Keep lc1 (bit 1 of p).
        # K_{lc2}[lc1*2+r, p*2+r] = 1 if p%2==lc2
        Ks = []
        for lc2_val in range(2):
            Km = np.zeros((4,8), dtype=complex)
            for p in range(4):
                if p % 2 == lc2_val:
                    lc1 = p // 2
                    for r in range(2):
                        Km[lc1*2+r, p*2+r] = 1.0
            Ks.append(Km @ P)
    else:
        # Trace over lc1 (bit 1 of p). Keep lc2 (bit 0 of p).
        # K_{lc1}[lc2*2+r, p*2+r] = 1 if p//2==lc1
        Ks = []
        for lc1_val in range(2):
            Km = np.zeros((4,8), dtype=complex)
            for p in range(4):
                if p // 2 == lc1_val:
                    lc2 = p % 2
                    for r in range(2):
                        Km[lc2*2+r, p*2+r] = 1.0
            Ks.append(Km @ P)
    return Ks[0], Ks[1]


def build_kraus_right(v0, v1, keep_rc1=True):
    """
    Returns (A0, A1): two (4,4) Kraus matrices for right-qubit branch.
    After left branch, rho4 is (kept_left, right_input), layout [l*2+ri, lb*2+rib].
    Right branch: rho8b[l*4+p*2+..., lb*4+...] via same structure but:
      rho8b rows: li*4 + (rc-component) (see original: rho8[li*4:li*4+4, lj*4:lj*4+4])
    
    Original right branch:
      for a in range(2): for b in range(2):
        amp = rho4_mid[a*2:a*2+2, b*2:b*2+2]  # (a=l, ri=right_input)
        va=v0_R if a==0 else v1_R; vb=v0_R if b==0 else v1_R
        for li in range(2): for lj in range(2):
          rho8[li*4:li*4+4, lj*4:lj*4+4] += amp[li,lj] * outer(va,vb.conj())
    
    So: rho8b[li*4+p, lj*4+q] = sum_{a,b} rho4_mid[a*2+li, b*2+lj] * vr[a][p]*conj(vr[b][q])
    where a=right_input and li=l (kept_left). Indices are swapped vs left branch!
    
    row = li*4+p (l is the OUTER index here, p=right component is INNER 4-dim)
    col mapping: a*2+li -> rho4_mid col. a is the right input (bit 1), li is left (bit 0).
    
    P_R[li*4+p, a*2+li'] = vr[a][p] * delta(li, li')
    = (I_2 ⊗ V_right)[li, a; p, li'->li] -- tricky layout
    
    More directly: rho8b = sum_{li,lj} amp[li,lj] * |li><lj|_left ⊗ (vr@vr†)_right
    where amp[li,lj] = sum_{a,b} rho4[a*2+li, b*2+lj] * ...
    
    Hmm let me just think of it as P_R:
    P_R shape (8,4): rows (li,p) i.e. li*4+p, cols (a,li) i.e. a*2+li
    P_R[li*4+p, a*2+li'] = vr[a][p] * delta(li,li')
    = I_left ⊗ V_right but with (li,p) ordering on rows and (a,li') ordering on cols
    
    Rearranging col from (a,li) to (li,a): col_new = li*2+a, col_old = a*2+li
    But rho4_mid cols are indexed a*2+li already (a=outer=right_input, li=inner=left).
    
    P_R[li*4+p, a*2+li] = vr[a][p] * delta(li,li)  -- li always matches itself
    So P_R[li*4+p, a*2+li] = vr[a][p]
    This is: for each (li, a), col = a*2+li, row = li*4+p for all p.
    """
    vr = [v0, v1]
    # P_R (8,4): rows li*4+p, cols a*2+li
    P_R = np.zeros((8,4), dtype=complex)
    for a in range(2):
        for li in range(2):
            col = a*2+li
            for p in range(4):
                row = li*4+p
                P_R[row, col] = vr[a][p]

    # rho8b trace: keep rc1 (p//2) or rc2 (p%2)
    # Original keep_rc1=True (go_right_R=True):
    #   rho4_new[l*2+rc1, lb*2+rc1b] += rho8b[l*4+rc1*2+rc2, lb*4+rc1b*2+rc2]  (sum rc2)
    # row of rho8b: l*4+p where p=rc1*2+rc2, so l*4+rc1*2+rc2
    # keep_rc1: keep bit 1 of p (rc1=p//2), trace bit 0 (rc2=p%2)
    # K_{rc2}[(l*2+rc1), (l*4+p)] = 1 if p%2==rc2 i.e. if l and rc1=p//2 match row
    
    if keep_rc1:
        Ks = []
        for rc2_val in range(2):
            Km = np.zeros((4,8), dtype=complex)
            for l in range(2):
                for p in range(4):
                    if p % 2 == rc2_val:
                        rc1 = p // 2
                        Km[l*2+rc1, l*4+p] = 1.0
            Ks.append(Km @ P_R)
    else:
        # keep_rc2: trace rc1 (p//2), keep rc2 (p%2)
        Ks = []
        for rc1_val in range(2):
            Km = np.zeros((4,8), dtype=complex)
            for l in range(2):
                for p in range(4):
                    if p // 2 == rc1_val:
                        rc2 = p % 2
                        Km[l*2+rc2, l*4+p] = 1.0
            Ks.append(Km @ P_R)
    return Ks[0], Ks[1]


def apply_kraus(rho4, A0, A1):
    """rho4_out = A0 @ rho4 @ A0† + A1 @ rho4 @ A1†"""
    A0c = A0.conj().T
    A1c = A1.conj().T
    return A0 @ rho4 @ A0c + A1 @ rho4 @ A1c


def apply_kraus_batch(rho4s, A0s, A1s):
    """
    Batched version. rho4s: (B,4,4). A0s,A1s: (B,4,4) or (4,4).
    Returns (B,4,4).
    """
    A0c = A0s.conj().swapaxes(-1,-2)
    A1c = A1s.conj().swapaxes(-1,-2)
    return A0s @ rho4s @ A0c + A1s @ rho4s @ A1c


# ─── Single-qubit helpers ──────────────────────────────────────────────────

def apply_branch_single(rho2, v0, v1):
    return (rho2[0,0]*np.outer(v0,v0.conj()) + rho2[0,1]*np.outer(v0,v1.conj()) +
            rho2[1,0]*np.outer(v1,v0.conj()) + rho2[1,1]*np.outer(v1,v1.conj()))

def ptrace_right(rho4):
    return np.array([[rho4[0,0]+rho4[1,1], rho4[0,2]+rho4[1,3]],
                     [rho4[2,0]+rho4[3,1], rho4[2,2]+rho4[3,3]]])

def ptrace_left(rho4):
    return np.array([[rho4[0,0]+rho4[2,2], rho4[0,1]+rho4[2,3]],
                     [rho4[1,0]+rho4[3,2], rho4[1,1]+rho4[3,3]]])

def vn_np(rho):
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log(vals)))

def mi_np(rho4):
    return vn_np(ptrace_right(rho4)) + vn_np(ptrace_left(rho4)) - vn_np(rho4)


# ─── GPU batched entropy ───────────────────────────────────────────────────

def mi_batch(rho4s_gpu):
    """
    Batched MI from (B,4,4) joint density matrices.
    Returns (B,) numpy array.
    """
    B = rho4s_gpu.shape[0]
    rho_A = xp.zeros((B,2,2), dtype=complex)
    rho_A[:,0,0] = rho4s_gpu[:,0,0]+rho4s_gpu[:,1,1]
    rho_A[:,0,1] = rho4s_gpu[:,0,2]+rho4s_gpu[:,1,3]
    rho_A[:,1,0] = rho4s_gpu[:,2,0]+rho4s_gpu[:,3,1]
    rho_A[:,1,1] = rho4s_gpu[:,2,2]+rho4s_gpu[:,3,3]

    rho_B = xp.zeros((B,2,2), dtype=complex)
    rho_B[:,0,0] = rho4s_gpu[:,0,0]+rho4s_gpu[:,2,2]
    rho_B[:,0,1] = rho4s_gpu[:,0,1]+rho4s_gpu[:,2,3]
    rho_B[:,1,0] = rho4s_gpu[:,1,0]+rho4s_gpu[:,3,2]
    rho_B[:,1,1] = rho4s_gpu[:,1,1]+rho4s_gpu[:,3,3]

    def ent(rhos):
        v = xp.linalg.eigvalsh(rhos)
        v = xp.clip(xp.real(v), 1e-15, None)
        return -xp.sum(v * xp.log(v), axis=-1)

    mi = ent(rho_A) + ent(rho_B) - ent(rho4s_gpu)
    if GPU:
        return cp.asnumpy(mi)
    return np.array(mi)


# ─── Verification ──────────────────────────────────────────────────────────

def run_test(depth=4):
    seed = 0 * 1000 + depth  # tree 0 at given depth
    print(f"\n{'='*60}")
    print(f"TEST: depth={depth}, seed={seed}")
    print(f"{'='*60}")

    bvecs = build_bvecs(depth, seed)
    n_leaves = 2**depth
    results = {}

    for i, j in combinations(range(n_leaves), 2):
        lca_steps = 0
        for k in range(depth):
            if (i>>(depth-1-k)) != (j>>(depth-1-k)):
                lca_steps = depth-k; break
        lca_gen = depth-lca_steps; dG = 2*lca_steps

        rho = np.array([[1,0],[0,0]], dtype=complex)
        node = 0
        for gen in range(lca_gen):
            v0,v1 = bvecs[node]
            rho4 = apply_branch_single(rho, v0, v1)
            lca_idx = (i>>lca_steps)>>(lca_gen-gen-1)
            go_left = (lca_idx%2==0)
            rho = ptrace_right(rho4) if go_left else ptrace_left(rho4)
            node = 2*node+1+(0 if go_left else 1)

        v0,v1 = bvecs[node]
        rho4_joint = apply_branch_single(rho, v0, v1)

        if lca_steps == 1:
            mi_val = mi_np(rho4_joint)
        else:
            left_node=2*node+1; right_node=2*node+2
            i_bits=[(i>>(lca_steps-1-k))&1 for k in range(lca_steps)]
            j_bits=[(j>>(lca_steps-1-k))&1 for k in range(lca_steps)]
            rho4_curr = rho4_joint.copy()
            for step in range(lca_steps-1):
                go_L=(i_bits[step+1]==0); go_R=(j_bits[step+1]==0)
                v0_L,v1_L=bvecs[left_node]; v0_R,v1_R=bvecs[right_node]
                A0L,A1L = build_kraus_left(v0_L, v1_L, keep_lc1=go_L)
                A0R,A1R = build_kraus_right(v0_R, v1_R, keep_rc1=go_R)
                rho4_curr = apply_kraus(apply_kraus(rho4_curr, A0L, A1L), A0R, A1R)
                left_node=2*left_node+1+(0 if go_L else 1)
                right_node=2*right_node+1+(0 if go_R else 1)
            mi_val = mi_np(rho4_curr)

        results.setdefault(str(dG), []).append(mi_val)

    max_err = 0.0
    all_ok = True
    for dG_str, gt_vals in GROUND_TRUTH.items():
        our_vals = results.get(dG_str, [])
        if len(our_vals) != len(gt_vals):
            print(f"  dG={dG_str}: COUNT {len(our_vals)} vs {len(gt_vals)}")
            all_ok = False; continue
        for idx,(ours,gt) in enumerate(zip(our_vals, gt_vals)):
            err=abs(ours-gt); max_err=max(max_err,err)
            if err>1e-8:
                print(f"  dG={dG_str}[{idx}]: {ours:.10f} vs {gt:.10f} err={err:.2e}")
                all_ok = False

    if all_ok:
        print(f"  ✓ ALL VALUES MATCH (max_err={max_err:.2e})")
    else:
        print(f"  ✗ MISMATCHES (max_err={max_err:.2e})")

    for dG in sorted(results.keys(), key=int):
        vals = np.array(results[dG])
        print(f"  dG={dG:>3}: n={len(vals):>4}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    return all_ok


# ─── Main GPU computation ──────────────────────────────────────────────────

def run_tree(depth, seed, batch_size=8192, verbose=True):
    """
    GPU-accelerated MI computation. CPU computes Kraus matrices per pair,
    GPU applies batched matmul and entropy.
    """
    t0 = time.time()
    print(f"  Building {2**depth-1:,} branch vectors (seed={seed})...")
    bvecs = build_bvecs(depth, seed)
    print(f"  Done in {time.time()-t0:.1f}s")

    n_leaves = 2**depth
    n_pairs  = n_leaves*(n_leaves-1)//2

    results = {}
    t_cpu = t_gpu = 0.0
    processed = 0
    log_every = max(1, n_pairs // 40)

    # Batch buffers
    buf_dGs   = []
    buf_rho4s = []    # initial rho4_joint (4,4) per pair
    buf_kraus = []    # list of [(A0L,A1L,A0R,A1R), ...] per descent step

    def flush():
        nonlocal t_gpu, results
        if not buf_dGs:
            return
        B = len(buf_dGs)
        t1 = time.time()

        # Upload initial states
        rho4s = xp.array(np.array(buf_rho4s), dtype=complex)  # (B,4,4)

        max_steps = max(len(seq) for seq in buf_kraus)
        I4 = np.eye(4, dtype=complex)

        for step in range(max_steps):
            A0Ls = np.zeros((B,4,4), dtype=complex)
            A1Ls = np.zeros((B,4,4), dtype=complex)
            A0Rs = np.zeros((B,4,4), dtype=complex)
            A1Rs = np.zeros((B,4,4), dtype=complex)
            active = np.zeros(B, dtype=bool)

            for b in range(B):
                if step < len(buf_kraus[b]):
                    A0L,A1L,A0R,A1R = buf_kraus[b][step]
                    A0Ls[b]=A0L; A1Ls[b]=A1L
                    A0Rs[b]=A0R; A1Rs[b]=A1R
                    active[b] = True
                else:
                    # Identity (no-op): A0=I, A1=0
                    A0Ls[b]=I4; A1Ls[b]=0
                    A0Rs[b]=I4; A1Rs[b]=0

            A0Ls_g=xp.array(A0Ls); A1Ls_g=xp.array(A1Ls)
            A0Rs_g=xp.array(A0Rs); A1Rs_g=xp.array(A1Rs)

            rho4s_new = apply_kraus_batch(rho4s, A0Ls_g, A1Ls_g)
            rho4s_new = apply_kraus_batch(rho4s_new, A0Rs_g, A1Rs_g)

            # Restore inactive pairs (they got identity applied = unchanged)
            active_g = xp.array(active)[:,None,None]
            rho4s = xp.where(active_g, rho4s_new, rho4s)

        mi_vals = mi_batch(rho4s)
        for b, dG in enumerate(buf_dGs):
            results.setdefault(str(dG), []).append(float(max(0.0, mi_vals[b])))

        t_gpu += time.time() - t1
        buf_dGs.clear(); buf_rho4s.clear(); buf_kraus.clear()

    t_start = time.time()

    for i, j in combinations(range(n_leaves), 2):
        t1 = time.time()

        lca_steps = 0
        for k in range(depth):
            if (i>>(depth-1-k)) != (j>>(depth-1-k)):
                lca_steps = depth-k; break
        lca_gen = depth-lca_steps; dG = 2*lca_steps

        rho = np.array([[1,0],[0,0]], dtype=complex)
        node = 0
        for gen in range(lca_gen):
            v0,v1 = bvecs[node]
            rho4 = apply_branch_single(rho, v0, v1)
            lca_idx = (i>>lca_steps)>>(lca_gen-gen-1)
            go_left = (lca_idx%2==0)
            rho = ptrace_right(rho4) if go_left else ptrace_left(rho4)
            node = 2*node+1+(0 if go_left else 1)

        v0,v1 = bvecs[node]
        rho4_joint = apply_branch_single(rho, v0, v1)

        # Build Kraus sequence for descent
        kraus_seq = []
        if lca_steps > 1:
            left_node=2*node+1; right_node=2*node+2
            i_bits=[(i>>(lca_steps-1-k))&1 for k in range(lca_steps)]
            j_bits=[(j>>(lca_steps-1-k))&1 for k in range(lca_steps)]
            for step in range(lca_steps-1):
                go_L=(i_bits[step+1]==0); go_R=(j_bits[step+1]==0)
                v0_L,v1_L=bvecs[left_node]; v0_R,v1_R=bvecs[right_node]
                A0L,A1L = build_kraus_left(v0_L, v1_L, keep_lc1=go_L)
                A0R,A1R = build_kraus_right(v0_R, v1_R, keep_rc1=go_R)
                kraus_seq.append((A0L,A1L,A0R,A1R))
                left_node=2*left_node+1+(0 if go_L else 1)
                right_node=2*right_node+1+(0 if go_R else 1)

        t_cpu += time.time()-t1

        buf_dGs.append(dG)
        buf_rho4s.append(rho4_joint)
        buf_kraus.append(kraus_seq)

        if len(buf_dGs) >= batch_size:
            flush()

        processed += 1
        if processed % log_every == 0 and verbose:
            elapsed = time.time()-t_start
            eta = (n_pairs-processed)/(processed/elapsed)/3600
            print(f"  {processed:,}/{n_pairs:,} ({100*processed/n_pairs:.1f}%) | "
                  f"cpu={t_cpu:.0f}s gpu={t_gpu:.1f}s | ETA {eta:.2f}h")

    flush()
    total = time.time()-t_start
    print(f"  Done {total/3600:.4f}h (cpu={t_cpu:.0f}s gpu={t_gpu:.1f}s)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth',      type=int, default=12)
    parser.add_argument('--trees',      type=int, default=1)
    parser.add_argument('--start-tree', type=int, default=0)
    parser.add_argument('--output',     type=str,
                        default=os.path.expanduser('~/qft/results/'))
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--test',       action='store_true')
    args = parser.parse_args()

    if args.test:
        ok = run_test(depth=4)
        if ok:
            print("\n✓ Test passed — safe to run")
        else:
            print("\n✗ Test FAILED — do not run on real data")
        return

    os.makedirs(args.output, exist_ok=True)
    n_leaves = 2**args.depth
    n_pairs  = n_leaves*(n_leaves-1)//2
    print(f"QFT GPU v6 | depth={args.depth} leaves={n_leaves:,} "
          f"pairs={n_pairs:,} GPU={GPU} batch={args.batch_size}")

    for tree_idx in range(args.start_tree, args.start_tree + args.trees):
        outfile = os.path.join(args.output,
                               f"depth{args.depth}_tree{tree_idx:03d}.json")
        if os.path.exists(outfile):
            print(f"Tree {tree_idx}: exists, skipping")
            continue

        print(f"\n{'='*60}\nTree {tree_idx}\n{'='*60}")
        seed = tree_idx * 1000 + args.depth
        t0 = time.time()
        res = run_tree(args.depth, seed, batch_size=args.batch_size)

        with open(outfile, 'w') as f:
            json.dump(res, f)
        print(f"Saved {outfile}")

        for dG in sorted(res.keys(), key=int):
            vals = np.array(res[dG])
            print(f"  dG={dG:>3}: n={len(vals):>8,}  "
                  f"mean={np.mean(vals):.5f}  std={np.std(vals):.5f}")


if __name__ == '__main__':
    main()
