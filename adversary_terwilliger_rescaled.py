"""
Rescaled symmetry-reduced adversary SDP for symmetric Boolean functions.

This file implements an analytic Step-2-style Terwilliger block reduction
for the primal adversary SDP after Step 1 orbit reduction.

Main choices:
  * The objective is ALWAYS the full symmetric sum sum_{x,y} Gamma[x,y].
    There is no half_objective option.
  * The solver works from Hamming-layer values, not full truth tables.
  * It caches the Terwilliger block precomputation.
  * It rescales optimization variables:
        alpha_k       = C(n,k) beta_k
        eta_{a,b,t}   = N_{a,b,t} gamma_{a,b,t}
    where N_{a,b,t} is the ordered-pair multiplicity used in the full
    objective. Thus the mass constraints and objective have O(1) coefficients.
  * It can optionally apply a positive diagonal congruence scaling D B D
    to each PSD block, preserving PSD while improving numerical balance.

The implementation assumes that the promise domain is a union of full
Hamming layers.
"""

from __future__ import annotations

from collections import defaultdict, deque
from functools import lru_cache
from math import comb, sqrt
from numbers import Number
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cvxpy as cp


# ============================================================
# Types
# ============================================================

LayerValue = Dict[int, int]
OrbitKeyH = Tuple[int, int, int, int, int]  # (x1, y1, |x|, |y|, |x cap y|)
GammaKey = Tuple[int, int, int]             # (a, b, t), with a <= b


# ============================================================
# Basic helpers
# ============================================================

def normalize_input_bits(x: Any) -> Tuple[int, ...]:
    """
    Convert an input to a 0/1 tuple.

    Accepted formats:
      - tuple/list in {0,1}
      - tuple/list in {-1,1}
      - string like '0101'
    """
    if isinstance(x, str):
        bits = tuple(int(c) for c in x)
        if not set(bits).issubset({0, 1}):
            raise ValueError(f"String input {x!r} is not binary.")
        return bits

    x_tuple = tuple(int(v) for v in x)
    vals = set(x_tuple)

    if vals.issubset({0, 1}):
        return x_tuple

    if vals.issubset({-1, 1}):
        return tuple(1 if v == 1 else 0 for v in x_tuple)

    raise ValueError(f"Input {x!r} is not in {{0,1}}^n or {{-1,1}}^n.")


def normalize_output(v: Any) -> int:
    """
    Normalize Boolean output to 0/1.
    Treats -1, 0, False as 0 and everything else as 1.
    """
    return 0 if v in (-1, 0, False) else 1


def hamming_weight(x: Sequence[int]) -> int:
    return sum(1 for bit in x if int(bit) == 1)


def intersection_weight(x: Sequence[int], y: Sequence[int]) -> int:
    return sum(1 for a, b in zip(x, y) if int(a) == 1 and int(b) == 1)


@lru_cache(maxsize=None)
def binom_safe(n: int, k: int) -> int:
    """
    Safe binomial coefficient. Returns 0 outside the valid range.
    Cached because these values occur repeatedly in the block formulas.
    """
    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        return 0
    return comb(n, k)


@lru_cache(maxsize=None)
def full_layer_size(n: int, k: int) -> int:
    return binom_safe(int(n), int(k))


def normalize_layer_value(layer_value: Mapping[int, Any]) -> LayerValue:
    """
    Normalize a layer -> output map into sorted integer keys and 0/1 values.
    """
    out: LayerValue = {}
    for k_raw, v_raw in layer_value.items():
        k = int(k_raw)
        out[k] = normalize_output(v_raw)
    return dict(sorted(out.items()))


# ============================================================
# Common symmetric Boolean functions by layer
# ============================================================

def OR_layers(n: int) -> LayerValue:
    return {k: int(k > 0) for k in range(n + 1)}


def AND_layers(n: int) -> LayerValue:
    return {k: int(k == n) for k in range(n + 1)}


def EXACT_layers(n: int, r: int) -> LayerValue:
    r = int(r)
    return {k: int(k == r) for k in range(n + 1)}


def THRESHOLD_layers(n: int, r: int) -> LayerValue:
    r = int(r)
    return {k: int(k >= r) for k in range(n + 1)}


def PARITY_layers(n: int) -> LayerValue:
    return {k: k % 2 for k in range(n + 1)}


def DJ_layers(N: int) -> LayerValue:
    """
    Deutsch-Jozsa promise problem on truth-table length N.
    Promise layers are 0, N/2, and N.
    Output 0 on constant strings and 1 on balanced strings.
    """
    N = int(N)
    if N % 2 != 0:
        raise ValueError("Deutsch-Jozsa truth-table length N must be even.")
    return {0: 0, N // 2: 1, N: 0}


# ============================================================
# Orbit multiplicities for full symmetric objective
# ============================================================

@lru_cache(maxsize=None)
def num_pairs_in_orbit(n: int, a: int, b: int, t: int) -> int:
    """
    Number of ordered pairs (x,y) in {0,1}^n x {0,1}^n with
        |x| = a, |y| = b, |x cap y| = t.
    """
    n = int(n)
    a = int(a)
    b = int(b)
    t = int(t)

    if not (0 <= a <= n and 0 <= b <= n):
        return 0
    if not (max(0, a + b - n) <= t <= min(a, b)):
        return 0

    return binom_safe(n, a) * binom_safe(a, t) * binom_safe(n - a, b - t)


@lru_cache(maxsize=None)
def gamma_objective_multiplicity(n: int, a: int, b: int, t: int) -> int:
    """
    Ordered-pair multiplicity for the stored symmetric gamma variable.

    Variables are stored with a <= b. The full objective is
        sum_{x,y} Gamma[x,y].
    If a < b, both orientations contribute.
    """
    n = int(n)
    a = int(a)
    b = int(b)
    t = int(t)
    if a == b:
        return num_pairs_in_orbit(n, a, b, t)
    return num_pairs_in_orbit(n, a, b, t) + num_pairs_in_orbit(n, b, a, t)


# ============================================================
# Schrijver/Terwilliger block coefficients
# ============================================================

@lru_cache(maxsize=None)
def beta_schrijver(m: int, i: int, j: int, k: int, t: int) -> int:
    """
    Schrijver's beta coefficient beta^t_{i,j,k} for the Terwilliger
    algebra of the m-dimensional Hamming cube.

    Formula:
        sum_u (-1)^(u-t) C(u,t) C(m-2k,u-k)
              C(m-k-u,i-u) C(m-k-u,j-u).
    """
    m = int(m)
    i = int(i)
    j = int(j)
    k = int(k)
    t = int(t)

    total = 0

    # C(m-2k, u-k) forces k <= u <= m-k.
    for u in range(k, m - k + 1):
        c1 = binom_safe(u, t)
        if c1 == 0:
            continue

        c2 = binom_safe(m - 2 * k, u - k)
        if c2 == 0:
            continue

        c3 = binom_safe(m - k - u, i - u)
        if c3 == 0:
            continue

        c4 = binom_safe(m - k - u, j - u)
        if c4 == 0:
            continue

        sign = -1 if ((u - t) % 2) else 1
        total += sign * c1 * c2 * c3 * c4

    return total


@lru_cache(maxsize=None)
def terwilliger_block_coeff(m: int, i: int, j: int, k: int, t: int) -> float:
    """
    Coefficient of M^t_{i,j} in the kth Schrijver/Terwilliger block:
        C(m-2k,i-k)^(-1/2) C(m-2k,j-k)^(-1/2) beta^t_{i,j,k}.
    """
    m = int(m)
    i = int(i)
    j = int(j)
    k = int(k)
    t = int(t)

    denom_i = binom_safe(m - 2 * k, i - k)
    denom_j = binom_safe(m - 2 * k, j - k)

    if denom_i == 0 or denom_j == 0:
        return 0.0

    beta = beta_schrijver(m, i, j, k, t)
    if beta == 0:
        return 0.0

    return beta / sqrt(denom_i * denom_j)


# ============================================================
# Terwilliger block precomputation
# ============================================================

_PRECOMP_CACHE: Dict[Tuple[int, Tuple[int, ...]], Dict[str, Any]] = {}


def precompute_adversary_terwilliger_blocks(
    n: int,
    allowed_layers: Iterable[int],
) -> Dict[str, Any]:
    """
    Precompute the analytic Terwilliger block diagonalization for the
    H = Stab(1) invariant algebra relevant to

        M_1 = diag(beta) - Gamma o Delta_1.

    The precomputation depends only on n and the set of promised layers.
    """
    n = int(n)
    allowed_layers_tuple = tuple(sorted(int(k) for k in allowed_layers))
    allowed_set = set(allowed_layers_tuple)
    m = n - 1

    if m < 0:
        raise ValueError("n must be nonnegative.")

    blocks = []

    for k in range(m // 2 + 1):
        rest_weights = list(range(k, m - k + 1))

        block_indices: List[Tuple[int, int]] = []
        for epsilon in (0, 1):
            for i in rest_weights:
                total_weight = epsilon + i
                if total_weight in allowed_set:
                    block_indices.append((epsilon, i))

        if not block_indices:
            continue

        terms = []

        # term = (row, col, H-orbit key, coefficient)
        for row, (x1, i) in enumerate(block_indices):
            for col, (y1, j) in enumerate(block_indices):
                tau_min = max(0, i + j - m)
                tau_max = min(i, j)

                for tau in range(tau_min, tau_max + 1):
                    coeff = terwilliger_block_coeff(m, i, j, k, tau)
                    if coeff == 0:
                        continue

                    a = x1 + i
                    b = y1 + j
                    t = x1 * y1 + tau
                    key: OrbitKeyH = (x1, y1, a, b, t)
                    terms.append((row, col, key, coeff))

        blocks.append({
            "k": k,
            "indices": block_indices,
            "size": len(block_indices),
            "terms": terms,
        })

    return {
        "n": n,
        "m": m,
        "allowed_layers": allowed_layers_tuple,
        "blocks": blocks,
        "terwilliger_block_sizes": [B["size"] for B in blocks],
        "num_terwilliger_blocks": len(blocks),
        "active_cache": {},
    }


def get_terwilliger_precomp(n: int, allowed_layers: Iterable[int]) -> Dict[str, Any]:
    """
    Cached precomputation. Use this if solving several symmetric functions
    on the same n and same union of Hamming layers.
    """
    key = (int(n), tuple(sorted(int(k) for k in allowed_layers)))

    if key not in _PRECOMP_CACHE:
        _PRECOMP_CACHE[key] = precompute_adversary_terwilliger_blocks(
            n=key[0],
            allowed_layers=key[1],
        )

    return _PRECOMP_CACHE[key]


# ============================================================
# Function-specific active block terms
# ============================================================

def layer_signature(layer_value: Mapping[int, int]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(k), int(v)) for k, v in layer_value.items()))


def active_key_for_M1(key: OrbitKeyH, layer_value: Mapping[int, int]) -> bool:
    """
    Whether an H-orbit basis coefficient can be nonzero in

        M_1 = diag(beta) - Gamma o Delta_1

    for the given layer output pattern.
    """
    x1, y1, a, b, t = key

    # Diagonal contribution from diag(beta).
    if x1 == y1 and a == b == t:
        return True

    # Off-diagonal adversary contribution from Gamma o Delta_1.
    if x1 != y1 and layer_value[a] != layer_value[b]:
        return True

    return False


def split_block_by_active_pattern(block: Mapping[str, Any], layer_value: Mapping[int, int]) -> List[List[int]]:
    """
    Further split a Terwilliger block using the zero pattern of the specific
    matrix M_1. This is an exact instance-specific decomposition.
    """
    size = int(block["size"])
    adj = [set([i]) for i in range(size)]

    for row, col, key, coeff in block["terms"]:
        if coeff != 0 and active_key_for_M1(key, layer_value):
            adj[row].add(col)
            adj[col].add(row)

    seen = [False] * size
    components = []

    for v in range(size):
        if seen[v]:
            continue

        q = deque([v])
        seen[v] = True
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)

            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)

        components.append(sorted(comp))

    return components


def gamma_key(a: int, b: int, t: int) -> GammaKey:
    aa, bb = min(int(a), int(b)), max(int(a), int(b))
    return (aa, bb, int(t))


def compile_active_blocks(
    precomp: Dict[str, Any],
    layer_value: Mapping[int, int],
    split_specific_blocks: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build and cache the function-specific active terms for the SDP blocks.

    Each active term is stored as:
        (row, col, signed_coeff, kind, data)

    where:
      - kind == "beta",  data = layer k,          contribution = signed_coeff * beta[k]
      - kind == "gamma", data = (a,b,t), a <= b, contribution = signed_coeff * gamma[a,b,t]

    The sign for gamma is already included, because M_1 has - Gamma o Delta_1.
    """
    normalized_layer_value = normalize_layer_value(layer_value)
    sig = layer_signature(normalized_layer_value)
    cache_key = (sig, bool(split_specific_blocks))

    active_cache = precomp.setdefault("active_cache", {})
    if cache_key in active_cache:
        return active_cache[cache_key]

    active_blocks = []

    for block in precomp["blocks"]:
        size = int(block["size"])

        if split_specific_blocks:
            components = split_block_by_active_pattern(block, normalized_layer_value)
        else:
            components = [list(range(size))]

        for comp in components:
            local = {old: new for new, old in enumerate(comp)}
            active_terms = []

            for row, col, key, coeff in block["terms"]:
                if row not in local or col not in local:
                    continue

                x1, y1, a, b, t = key

                # Diagonal beta term.
                if x1 == y1 and a == b == t:
                    active_terms.append((local[row], local[col], float(coeff), "beta", int(a)))
                    continue

                # Gamma term. It is present only if first bits differ and outputs differ.
                if x1 != y1 and normalized_layer_value[a] != normalized_layer_value[b]:
                    gkey = gamma_key(a, b, t)
                    active_terms.append((local[row], local[col], -float(coeff), "gamma", gkey))
                    continue

            # If a component contributes no terms, its PSD constraint is 0 >= 0.
            if active_terms:
                active_blocks.append({
                    "size": len(comp),
                    "terms": active_terms,
                })

    active_cache[cache_key] = active_blocks
    return active_blocks


# ============================================================
# CVXPY helpers
# ============================================================

def scalar_to_1x1(expr: Any):
    if isinstance(expr, Number):
        expr = cp.Constant(float(expr))
    return cp.reshape(expr, (1, 1), order="C")


def scalar_bmat(entries: List[List[Any]]):
    return cp.bmat([[scalar_to_1x1(e) for e in row] for row in entries])


def available_default_solver():
    """
    Choose a reasonable default installed solver.
    MOSEK is generally preferred for SDPs if licensed; SDPA is tried next;
    SCS is a fallback.
    """
    installed = set(cp.installed_solvers())
    if "MOSEK" in installed:
        return cp.MOSEK
    if "SDPA" in installed:
        return cp.SDPA
    return cp.SCS


def solve_cvxpy_problem(
    problem: cp.Problem,
    solver=None,
    verbose: bool = False,
    solver_options: Optional[Mapping[str, Any]] = None,
):
    opts = dict(solver_options or {})

    if solver is not None:
        return problem.solve(solver=solver, verbose=verbose, **opts)

    chosen = available_default_solver()
    if chosen == cp.SCS:
        # These are intentionally conservative defaults for an SDP fallback.
        opts.setdefault("eps", 1e-6)
        opts.setdefault("max_iters", 200000)

    return problem.solve(solver=chosen, verbose=verbose, **opts)


# ============================================================
# Rescaling helpers
# ============================================================

def coefficient_scale_for_term(n: int, kind: str, data: Any) -> float:
    """
    Return the denominator that converts scaled variables back to the
    original SDP variables.

    alpha_k = C(n,k) beta_k      => beta_k = alpha_k / C(n,k)
    eta_g   = N_g gamma_g        => gamma_g = eta_g / N_g
    """
    if kind == "beta":
        return float(full_layer_size(n, int(data)))

    if kind == "gamma":
        a, b, t = data
        mult = gamma_objective_multiplicity(n, a, b, t)
        if mult <= 0:
            raise ValueError(f"Zero gamma objective multiplicity for {(a, b, t)}.")
        return float(mult)

    raise ValueError(f"Unknown term kind {kind!r}.")


def block_row_scales(
    n: int,
    m_block: int,
    terms: List[Tuple[int, int, float, str, Any]],
    enabled: bool = True,
) -> List[float]:
    """
    Compute diagonal scaling factors d_i for a congruence D B D.

    The heuristic uses the absolute coefficient mass in each row after
    variable rescaling. This does not change the feasibility condition:
        B >= 0 iff D B D >= 0
    for positive diagonal D.
    """
    if not enabled:
        return [1.0] * m_block

    row_mass = [0.0 for _ in range(m_block)]

    for row, col, signed_coeff, kind, data in terms:
        scale = coefficient_scale_for_term(n, kind, data)
        c = abs(float(signed_coeff)) / scale
        row_mass[row] += c
        if col != row:
            row_mass[col] += c

    d = []
    for mass in row_mass:
        if mass <= 0:
            d.append(1.0)
        else:
            # Clamp to avoid creating extreme constants from near-zero rows.
            mass = max(mass, 1e-300)
            d.append(1.0 / sqrt(mass))
    return d


# ============================================================
# Main rescaled solver from layer values
# ============================================================

def adversary_primal_step2_rescaled_from_layers(
    n: int,
    layer_value: Mapping[int, Any],
    solver=None,
    verbose: bool = False,
    solver_options: Optional[Mapping[str, Any]] = None,
    precomp: Optional[Dict[str, Any]] = None,
    split_specific_blocks: bool = True,
    diagonal_blocks_as_inequalities: bool = True,
    balance_psd_blocks: bool = True,
) -> Dict[str, Any]:
    """
    Solve the symmetry-reduced primal adversary SDP using rescaled variables.

    Objective convention:
      This function ALWAYS maximizes the full symmetric sum
          sum_{x,y} Gamma[x,y].
      There is intentionally no half-objective option.

    Rescaled variables:
        alpha_k     = C(n,k) beta_k
        eta_{a,b,t} = N_{a,b,t} gamma_{a,b,t}

    Thus the normalization constraints become
        sum_{k in f^{-1}(0)} alpha_k = 1/2,
        sum_{k in f^{-1}(1)} alpha_k = 1/2,
    and the objective becomes
        maximize sum_g eta_g.
    """
    n = int(n)
    layer_value_norm = normalize_layer_value(layer_value)

    if not layer_value_norm:
        raise ValueError("layer_value is empty.")

    allowed_layers = tuple(sorted(layer_value_norm.keys()))

    for k in allowed_layers:
        if k < 0 or k > n:
            raise ValueError(f"Layer {k} is outside 0..n.")

    zero_layers = [k for k in allowed_layers if layer_value_norm[k] == 0]
    one_layers = [k for k in allowed_layers if layer_value_norm[k] == 1]

    if not zero_layers or not one_layers:
        raise ValueError("The function must take both output values on the promise domain.")

    if precomp is None:
        precomp = get_terwilliger_precomp(n, allowed_layers)
    else:
        if int(precomp["n"]) != n:
            raise ValueError("Precomputation n does not match.")
        if tuple(precomp["allowed_layers"]) != allowed_layers:
            raise ValueError("Precomputation allowed_layers do not match.")

    # ----------------------------------------------------------
    # Rescaled variables
    # ----------------------------------------------------------
    alpha = {
        k: cp.Variable(nonneg=True, name=f"alpha_{k}")
        for k in allowed_layers
    }

    gamma_key_set = set()
    for a in allowed_layers:
        for b in allowed_layers:
            if layer_value_norm[a] == layer_value_norm[b]:
                continue

            aa, bb = min(a, b), max(a, b)
            for t in range(max(0, aa + bb - n), min(aa, bb) + 1):
                gamma_key_set.add((aa, bb, t))

    gamma_keys = sorted(gamma_key_set)

    eta = {
        key: cp.Variable(name=f"eta_{key[0]}_{key[1]}_{key[2]}")
        for key in gamma_keys
    }

    # ----------------------------------------------------------
    # Constraints
    # ----------------------------------------------------------
    constraints = []

    constraints.append(cp.sum([alpha[k] for k in zero_layers]) == 0.5)
    constraints.append(cp.sum([alpha[k] for k in one_layers]) == 0.5)

    active_blocks = compile_active_blocks(
        precomp=precomp,
        layer_value=layer_value_norm,
        split_specific_blocks=split_specific_blocks,
    )

    final_block_sizes: List[int] = []
    scalar_inequality_count = 0
    psd_cone_sizes: List[int] = []

    for block in active_blocks:
        m_block = int(block["size"])
        terms = block["terms"]

        row_scales = block_row_scales(
            n=n,
            m_block=m_block,
            terms=terms,
            enabled=balance_psd_blocks,
        )

        # Optional cheap case: diagonal-only block -> scalar inequalities.
        if diagonal_blocks_as_inequalities and all(row == col for row, col, *_ in terms):
            diag_entries = [0 for _ in range(m_block)]

            for row, _col, signed_coeff, kind, data in terms:
                if kind == "beta":
                    var = alpha[int(data)]
                elif kind == "gamma":
                    var = eta[data]
                else:
                    raise ValueError(f"Unknown term kind {kind!r}.")

                denom = coefficient_scale_for_term(n, kind, data)
                c = float(signed_coeff) / denom
                c *= row_scales[row] * row_scales[row]
                diag_entries[row] = diag_entries[row] + c * var

            for expr in diag_entries:
                if isinstance(expr, Number) and abs(float(expr)) == 0:
                    continue
                constraints.append(expr >= 0)
                scalar_inequality_count += 1

            final_block_sizes.append(m_block)
            continue

        entries = [[0 for _ in range(m_block)] for __ in range(m_block)]

        for row, col, signed_coeff, kind, data in terms:
            if kind == "beta":
                var = alpha[int(data)]
            elif kind == "gamma":
                var = eta[data]
            else:
                raise ValueError(f"Unknown term kind {kind!r}.")

            denom = coefficient_scale_for_term(n, kind, data)
            c = float(signed_coeff) / denom
            c *= row_scales[row] * row_scales[col]
            entries[row][col] = entries[row][col] + c * var

        Bred = scalar_bmat(entries)
        Bred = 0.5 * (Bred + Bred.T)

        if m_block == 1:
            constraints.append(Bred[0, 0] >= 0)
            scalar_inequality_count += 1
        else:
            constraints.append(Bred >> 0)
            psd_cone_sizes.append(m_block)

        final_block_sizes.append(m_block)

    # ----------------------------------------------------------
    # Full symmetric objective in rescaled variables:
    #     sum_{x,y} Gamma[x,y] = sum_g eta_g.
    # ----------------------------------------------------------
    objective_expr = cp.sum([eta[key] for key in gamma_keys])

    problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    value = solve_cvxpy_problem(
        problem,
        solver=solver,
        verbose=verbose,
        solver_options=solver_options,
    )

    # Recover original variables for readability.
    beta_values = {}
    for k, var in alpha.items():
        beta_values[k] = None if var.value is None else float(var.value) / float(full_layer_size(n, k))

    gamma_values = {}
    for key, var in eta.items():
        mult = float(gamma_objective_multiplicity(n, *key))
        gamma_values[key] = None if var.value is None else float(var.value) / mult

    alpha_values = {k: alpha[k].value for k in alpha}
    eta_values = {key: eta[key].value for key in eta}

    return {
        "value": value,
        "status": problem.status,
        "objective_convention": "full_symmetric_sum",
        "alpha": alpha_values,
        "eta": eta_values,
        "beta": beta_values,
        "gamma": gamma_values,
        "allowed_layers": allowed_layers,
        "terwilliger_block_sizes": precomp["terwilliger_block_sizes"],
        "final_block_sizes": final_block_sizes,
        "psd_cone_sizes": psd_cone_sizes,
        "scalar_inequality_count": scalar_inequality_count,
        "num_active_blocks": len(active_blocks),
        "num_psd_cones": len(psd_cone_sizes),
        "num_eta_variables": len(gamma_keys),
        "num_alpha_variables": len(alpha),
        "precomp": precomp,
        "solver_stats": problem.solver_stats,
        "problem": problem,
    }


# ============================================================
# Wrapper from explicit truth/promise table
# ============================================================

def layer_values_from_f_values(n: int, f_values: Mapping[Any, Any]) -> LayerValue:
    """
    Convert an explicit table x -> f(x) into layer values.
    This verifies that the domain is a union of full Hamming layers.

    Convenient for small n. For large n, prefer passing layer_value directly.
    """
    n = int(n)
    layer_value: LayerValue = {}
    layer_count = defaultdict(int)

    for x_raw, fx_raw in f_values.items():
        x = normalize_input_bits(x_raw)

        if len(x) != n:
            raise ValueError(f"Input {x_raw!r} has length {len(x)} but expected {n}.")

        k = hamming_weight(x)
        fx = normalize_output(fx_raw)
        layer_count[k] += 1

        if k in layer_value and layer_value[k] != fx:
            raise ValueError(f"Function is not symmetric: layer {k} has both outputs.")

        layer_value[k] = fx

    for k, count in layer_count.items():
        expected = full_layer_size(n, k)
        if count != expected:
            raise ValueError(
                f"Layer {k} is incomplete: saw {count} points, "
                f"but the full layer has {expected}."
            )

    return dict(sorted(layer_value.items()))


def adversary_primal_step2_rescaled_from_f_values(
    n: int,
    f_values: Mapping[Any, Any],
    solver=None,
    verbose: bool = False,
    solver_options: Optional[Mapping[str, Any]] = None,
    precomp: Optional[Dict[str, Any]] = None,
    split_specific_blocks: bool = True,
    diagonal_blocks_as_inequalities: bool = True,
    balance_psd_blocks: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper accepting an explicit dictionary x -> f(x).
    For large n, avoid this wrapper and use layer values directly.
    """
    layer_value = layer_values_from_f_values(n, f_values)

    return adversary_primal_step2_rescaled_from_layers(
        n=n,
        layer_value=layer_value,
        solver=solver,
        verbose=verbose,
        solver_options=solver_options,
        precomp=precomp,
        split_specific_blocks=split_specific_blocks,
        diagonal_blocks_as_inequalities=diagonal_blocks_as_inequalities,
        balance_psd_blocks=balance_psd_blocks,
    )


# ============================================================
# Short public API
# ============================================================

def solve_symmetric_adversary_rescaled(
    n: int,
    layer_value: Mapping[int, Any],
    solver=None,
    verbose: bool = False,
    solver_options: Optional[Mapping[str, Any]] = None,
    split_specific_blocks: bool = True,
    diagonal_blocks_as_inequalities: bool = True,
    balance_psd_blocks: bool = True,
) -> Dict[str, Any]:
    """
    Recommended high-level API.

    Example:
        res = solve_symmetric_adversary(10, OR_layers(10))

    The objective convention is always the full symmetric sum.
    """
    layer_value_norm = normalize_layer_value(layer_value)
    precomp = get_terwilliger_precomp(n, layer_value_norm.keys())

    return adversary_primal_step2_rescaled_from_layers(
        n=n,
        layer_value=layer_value_norm,
        solver=solver,
        verbose=verbose,
        solver_options=solver_options,
        precomp=precomp,
        split_specific_blocks=split_specific_blocks,
        diagonal_blocks_as_inequalities=diagonal_blocks_as_inequalities,
        balance_psd_blocks=balance_psd_blocks,
    )


# Backward-compatible aliases with the new full-objective convention.
adversary_primal_step2_optimized_from_layers = adversary_primal_step2_rescaled_from_layers
adversary_primal_step2_optimized_from_f_values = adversary_primal_step2_rescaled_from_f_values


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    n = 3
    result = solve_symmetric_adversary_rescaled(n, OR_layers(n), verbose=False)

    print("status:", result["status"])
    print("value:", result["value"])
    print("objective convention:", result["objective_convention"])
    print("Terwilliger block sizes:", result["terwilliger_block_sizes"])
    print("Final active block sizes:", result["final_block_sizes"])
    print("PSD cone sizes:", result["psd_cone_sizes"])
    print("Scalar inequalities:", result["scalar_inequality_count"])
    print("beta:", result["beta"])
    print("gamma:", result["gamma"])