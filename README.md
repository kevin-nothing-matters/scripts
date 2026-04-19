# Scripts

Reproducibility code for the Quantum Family Tree trilogy. Each script
is self-contained and produces the numerical result named in its
description. Python 3.10+, NumPy, SciPy.

## Setup

```bash
pip install numpy scipy
```

Monte Carlo scripts at depth ≥ 50 benefit from multiple cores.
Per-script flags are documented at the top of each file.

## Scripts

### `weingarten_eta2.py`
Computes the Haar-averaged second and fourth moments of U(4) exactly
via Weingarten integration. Verifies η₂ = 2/5 and η₄ = 13/70 as exact
rationals, and confirms the two-replica transfer matrix spectrum is
{1, 2/5}. Reproduces Proposition 3 of Paper 1. Runtime: seconds.

### `cvn_monte_carlo.py`
The main numerical result of Paper 1. Generates Haar-random binary
trees at depths D ∈ {20, 30, 40, 50, 100, 150}, computes the quenched
von Neumann entropy over 500+ trees per depth, and extracts the slope
c_VN = (9/10) log₂(5/2) ≈ 1.1897 bits/generation. Reproduces the
10.34σ measurement. Runtime: hours at D=150; tune with `--n-trees`
and `--depths`.

### `bc_recursion.py`
Exact (b,c) recursion tracking position-averaged annealed purity and
the Rényi-2 entropy rate. Preserves carry variance via the 4-replica
formulation. Reproduces h_{S_2} = 14/45 to four digits and validates
the MPDO transfer architecture. Runtime: minutes.

### `first_law_test.py`
Tests the first-law identity δ⟨H⟩/δS = 1 at k = 1, 2, 3 using the
modular Hamiltonian proxy H = −log ρ_A. Reproduces the first-law
theorem of Paper 2. Runtime: minutes.

## Reproducibility policy

All random seeds used in the papers are documented inline. Main
results reproduce to within Monte Carlo uncertainty at those seeds.
The 156-qubit IBM Heron hardware run reported in Paper 1 is not
reproduced here; it requires Qiskit and IBM Quantum access.

## License

MIT — see LICENSE at the repository root.
