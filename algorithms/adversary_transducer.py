"""
adversary_transducer.py

Compute an input-independent transducer unitary U from a dual/general-adversary
SDP solution for a finite function f : D -> [p].

The implemented SDP is the vector/factorization form used for function-evaluating
transducers:

    minimize    max_x 1/2 * sum_i (||u_{x,i}||^2 + ||v_{x,i}||^2)
    subject to  1[f(x) != f(y)] = sum_{i:x_i != y_i} <u_{x,i}, v_{y,i}>

The bilinear vectors are represented convexly by PSD block Gram matrices

    M_i = [[U_i, Z_i],
           [Z_i^T, V_i]] >= 0,

where Z_i[x,y] = <u_{x,i}, v_{y,i}>.

From a feasible solution, the code builds the canonical transducer catalyst

    w^up_{x,i}   = (u_{x,i} + v_{x,i}) / 2
    w^down_{x,i} = (u_{x,i} - v_{x,i}) / 2

and computes an input-independent unitary U such that, for every x in D,

    U ( |0>  + after_query_catalyst_x )
      = |f(x)> + before_query_catalyst_x.

Here "+" means direct sum, matching the public/private-space convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Any

import numpy as np
import scipy.linalg as la

try:
    import cvxpy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _CVXPY_IMPORT_ERROR = exc
else:
    _CVXPY_IMPORT_ERROR = None


Array = np.ndarray


@dataclass(frozen=True)
class GeneralAdversarySolution:
    inputs: list[tuple[int, ...]]
    outputs: list[int]
    objective_value: float
    status: str
    u: list[Array]          # u[i][x, :] is u_{x,i}
    v: list[Array]          # v[i][x, :] is v_{x,i}
    block_grams: list[Array]
    cross_grams: list[Array]
    alphabet_values: list[list[int]]


@dataclass(frozen=True)
class TransducerUnitary:
    U: Array
    source_states: Array    # columns: |0> plus after-query catalyst
    target_states: Array    # columns: |f(x)> plus before-query catalyst
    public_dim: int
    private_dim: int
    basis_slices: dict[tuple[int, str, int], slice]
    gram_error: float
    map_error: float
    unitarity_error: float


def all_binary_inputs(n: int) -> list[tuple[int, ...]]:
    """Return all bit strings of length n as tuples in lexicographic order."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return [tuple((k >> (n - 1 - j)) & 1 for j in range(n)) for k in range(2**n)]


def normalize_instance(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any],
) -> tuple[list[tuple[int, ...]], list[int], list[dict[Any, int]], dict[Any, int]]:
    """
    Map arbitrary coordinate/output labels to dense integer labels.

    The state-generating transducer uses a distinguished query-register state |0>.
    Dense labels preserve equality/inequality relations, which are all the SDP needs.
    """
    if len(inputs) == 0:
        raise ValueError("inputs must be non-empty")
    if len(inputs) != len(outputs):
        raise ValueError("len(inputs) must equal len(outputs)")
    n = len(inputs[0])
    if any(len(x) != n for x in inputs):
        raise ValueError("all inputs must have the same length")

    coord_maps: list[dict[Any, int]] = []
    norm_inputs: list[tuple[int, ...]] = []
    for j in range(n):
        vals = sorted({x[j] for x in inputs})
        # Put 0 first if present so Boolean labels stay unchanged.
        vals = ([0] if 0 in vals else []) + [a for a in vals if a != 0]
        coord_maps.append({a: k for k, a in enumerate(vals)})
    for x in inputs:
        norm_inputs.append(tuple(coord_maps[j][x[j]] for j in range(n)))

    out_vals = sorted(set(outputs))
    out_vals = ([0] if 0 in out_vals else []) + [a for a in out_vals if a != 0]
    output_map = {a: k for k, a in enumerate(out_vals)}
    norm_outputs = [output_map[y] for y in outputs]
    return norm_inputs, norm_outputs, coord_maps, output_map


def _require_cvxpy() -> None:
    if cp is None:  # pragma: no cover
        raise ImportError(
            "cvxpy is required for solve_general_adversary_sdp(). "
            "Install it with `pip install cvxpy`."
        ) from _CVXPY_IMPORT_ERROR


def _psd_factor(M: Array, tol: float = 1e-8) -> Array:
    """
    Return R with M approximately equal to R @ R.T.

    Small negative eigenvalues caused by SDP solver tolerance are clipped to zero.
    Rows of R are the realized Gram vectors.
    """
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    vals, vecs = la.eigh(M)
    max_val = float(np.max(np.abs(vals))) if vals.size else 0.0
    cutoff = max(tol, tol * max(1.0, max_val))
    if np.min(vals) < -100 * cutoff:
        raise ValueError(
            f"matrix has a negative eigenvalue {np.min(vals):.3e}; "
            "the SDP solution may be inaccurate"
        )
    keep = vals > cutoff
    if not np.any(keep):
        return np.zeros((M.shape[0], 0), dtype=float)
    return vecs[:, keep] * np.sqrt(vals[keep])[None, :]


def solve_general_adversary_sdp(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[Any, ...]], Any] | None = None,
    *,
    solver: str | None = None,
    solver_kwargs: dict[str, Any] | None = None,
    psd_factor_tol: float = 1e-8,
) -> GeneralAdversarySolution:
    """
    Solve the general-adversary SDP and factor the resulting Gram matrices.

    Args:
        inputs: finite domain D, as tuples/lists of coordinate values.
        outputs: f(x) values for each x; either outputs or f must be supplied.
        f: function used to compute outputs when outputs is None.
        solver: cvxpy solver name. If omitted, CLARABEL is preferred, then SCS.
        solver_kwargs: extra keyword args passed to Problem.solve().
        psd_factor_tol: tolerance used when extracting vectors from PSD blocks.

    Returns:
        A GeneralAdversarySolution containing explicit vector rows u_{x,i}, v_{x,i}.
    """
    _require_cvxpy()
    raw_inputs = [tuple(x) for x in inputs]
    if outputs is None:
        if f is None:
            raise ValueError("provide outputs or f")
        raw_outputs = [f(x) for x in raw_inputs]
    else:
        raw_outputs = list(outputs)
        if len(raw_outputs) != len(raw_inputs):
            raise ValueError("len(outputs) must equal len(inputs)")

    norm_inputs, norm_outputs, _, _ = normalize_instance(raw_inputs, raw_outputs)
    m = len(norm_inputs)
    n = len(norm_inputs[0])
    alphabet_values = [sorted({x[j] for x in norm_inputs} | {0}) for j in range(n)]

    A_vars = []  # Gram of u_{.,i}
    B_vars = []  # Gram of v_{.,i}
    Z_vars = []  # Cross Gram <u_{x,i}, v_{y,i}>
    constraints = []
    t = cp.Variable(name="adv")
    constraints.append(t >= 0)

    for i in range(n):
        A = cp.Variable((m, m), symmetric=True, name=f"Ugram_{i}")
        B = cp.Variable((m, m), symmetric=True, name=f"Vgram_{i}")
        Z = cp.Variable((m, m), name=f"Zcross_{i}")
        block = cp.bmat([[A, Z], [Z.T, B]])
        constraints.append(block >> 0)
        A_vars.append(A)
        B_vars.append(B)
        Z_vars.append(Z)

    for x_idx in range(m):
        cost_x = 0
        for i in range(n):
            cost_x += A_vars[i][x_idx, x_idx] + B_vars[i][x_idx, x_idx]
        constraints.append(0.5 * cost_x <= t)

    for x_idx, x in enumerate(norm_inputs):
        for y_idx, y in enumerate(norm_inputs):
            lhs_terms = [Z_vars[i][x_idx, y_idx] for i in range(n) if x[i] != y[i]]
            lhs = cp.sum(lhs_terms) if lhs_terms else 0
            rhs = 1.0 if norm_outputs[x_idx] != norm_outputs[y_idx] else 0.0
            constraints.append(lhs == rhs)

    problem = cp.Problem(cp.Minimize(t), constraints)
    if solver_kwargs is None:
        solver_kwargs = {}
    if solver is None:
        installed = set(cp.installed_solvers())
        solver = "CLARABEL" if "CLARABEL" in installed else "SCS"
    problem.solve(solver=solver, **solver_kwargs, verbose = True)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SDP solve failed with status {problem.status}")

    u_vecs: list[Array] = []
    v_vecs: list[Array] = []
    block_grams: list[Array] = []
    cross_grams: list[Array] = []
    for i in range(n):
        M = np.block([
            [np.asarray(A_vars[i].value, dtype=float), np.asarray(Z_vars[i].value, dtype=float)],
            [np.asarray(Z_vars[i].value, dtype=float).T, np.asarray(B_vars[i].value, dtype=float)],
        ])
        M = 0.5 * (M + M.T)
        R = _psd_factor(M, tol=psd_factor_tol)
        u_vecs.append(R[:m, :])
        v_vecs.append(R[m:, :])
        block_grams.append(M)
        cross_grams.append(np.asarray(Z_vars[i].value, dtype=float))

    return GeneralAdversarySolution(
        inputs=norm_inputs,
        outputs=norm_outputs,
        objective_value=float(problem.value),
        status=str(problem.status),
        u=u_vecs,
        v=v_vecs,
        block_grams=block_grams,
        cross_grams=cross_grams,
        alphabet_values=alphabet_values,
    )


def _make_private_basis_slices(sol: GeneralAdversarySolution) -> tuple[int, dict[tuple[int, str, int], slice]]:
    """Allocate direct-sum private basis blocks (i, direction, query_symbol)."""
    offset = 0
    basis: dict[tuple[int, str, int], slice] = {}
    for i, alphabet in enumerate(sol.alphabet_values):
        r_i = sol.u[i].shape[1]
        for direction in ("up", "down"):
            for symbol in alphabet:
                basis[(i, direction, symbol)] = slice(offset, offset + r_i)
                offset += r_i
    return offset, basis


def build_transducer_state_matrices(sol: GeneralAdversarySolution) -> tuple[Array, Array, int, dict[tuple[int, str, int], slice]]:
    """
    Build source and target state columns for the input-independent unitary.

    source column x is |0> plus the catalyst after the bidirectional query:
        sum_i |i,up,x_i>(u+v)/2 + |i,down,0>(u-v)/2

    target column x is |f(x)> plus the catalyst before the query:
        sum_i |i,up,0>(u+v)/2 + |i,down,x_i>(u-v)/2
    """
    inputs = sol.inputs
    outputs = sol.outputs
    m = len(inputs)
    n = len(inputs[0])
    public_dim = max(max(outputs) + 1, 1)
    private_dim, basis = _make_private_basis_slices(sol)
    total_dim = public_dim + private_dim
    source = np.zeros((total_dim, m), dtype=float)
    target = np.zeros((total_dim, m), dtype=float)

    for x_idx, x in enumerate(inputs):
        source[0, x_idx] = 1.0
        target[outputs[x_idx], x_idx] = 1.0
        for i in range(n):
            up = 0.5 * (sol.u[i][x_idx, :] + sol.v[i][x_idx, :])
            down = 0.5 * (sol.u[i][x_idx, :] - sol.v[i][x_idx, :])

            # after query
            sl = basis[(i, "up", x[i])]
            source[slice(public_dim + sl.start, public_dim + sl.stop), x_idx] += up
            sl = basis[(i, "down", 0)]
            source[slice(public_dim + sl.start, public_dim + sl.stop), x_idx] += down

            # before query
            sl = basis[(i, "up", 0)]
            target[slice(public_dim + sl.start, public_dim + sl.stop), x_idx] += up
            sl = basis[(i, "down", x[i])]
            target[slice(public_dim + sl.start, public_dim + sl.stop), x_idx] += down

    return source, target, public_dim, basis


def _orthonormal_from_gram_columns(A: Array, G_eig_tol: float = 1e-9) -> tuple[Array, Array, Array]:
    """
    For columns A with Gram G, return Q, V, vals with Q = A V diag(vals)^-1/2.
    """
    G = A.conj().T @ A
    G = 0.5 * (G + G.conj().T)
    vals, vecs = la.eigh(G)
    scale = max(1.0, float(np.max(np.abs(vals))) if vals.size else 1.0)
    keep = vals > G_eig_tol * scale
    if not np.any(keep):
        return np.zeros((A.shape[0], 0), dtype=A.dtype), vecs[:, keep], vals[keep]
    vals_keep = vals[keep]
    vecs_keep = vecs[:, keep]
    Q = A @ vecs_keep @ np.diag(1.0 / np.sqrt(vals_keep))
    # One QR pass improves numerical orthonormality without changing the span.
    Q, R = la.qr(Q, mode="economic")
    return Q, vecs_keep, vals_keep


def _complete_basis(Q: Array, tol: float = 1e-10) -> Array:
    """Complete an orthonormal basis Q to a square unitary matrix."""
    n, r = Q.shape
    if r == n:
        return Q
    N = la.null_space(Q.conj().T, rcond=tol)
    E = np.hstack([Q, N])
    # QR cleans up small numerical null_space errors.
    E, _ = la.qr(E, mode="full")
    # The first r columns may be rotated by QR, so preserve Q exactly and re-null.
    N = la.null_space(Q.conj().T, rcond=tol)
    E = np.hstack([Q, N])
    return E


def unitary_from_state_pairs(
    source: Array,
    target: Array,
    *,
    gram_tol: float = 1e-7,
    eig_tol: float = 1e-9,
) -> tuple[Array, float, float, float]:
    """
    Return a unitary U with U @ source approximately target.

    The construction uses the common Gram matrix of the source/target columns:
    if source^* source == target^* target, the column maps define an isometry on
    their span. The isometry is then extended arbitrarily on the orthogonal complement.
    """
    source = np.asarray(source, dtype=complex)
    target = np.asarray(target, dtype=complex)
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    N, _ = source.shape
    Gs = source.conj().T @ source
    Gt = target.conj().T @ target
    gram_error = float(la.norm(Gs - Gt, ord="fro"))

    # Use source Gram eigenvectors to coordinate both spans; for exact SDP data this
    # makes Q_source and Q_target corresponding orthonormal bases.
    G = 0.5 * (Gs + Gs.conj().T)
    vals, vecs = la.eigh(G)
    scale = max(1.0, float(np.max(np.abs(vals))) if vals.size else 1.0)
    keep = vals > eig_tol * scale
    vals_keep = vals[keep]
    V = vecs[:, keep]
    if len(vals_keep) == 0:
        return np.eye(N, dtype=complex), gram_error, float(la.norm(target-source)), 0.0

    Qs = source @ V @ np.diag(1.0 / np.sqrt(vals_keep))
    Qt = target @ V @ np.diag(1.0 / np.sqrt(vals_keep))
    # Re-orthonormalize Qs; rotate Qt by the same inverse R to preserve pairing.
    Qs_qr, R = la.qr(Qs, mode="economic")
    Qt = Qt @ la.inv(R)
    # Orthonormalize Qt by polar only if numerical errors are noticeable.
    # For inconsistent Gram matrices, this yields the closest isometric pairing.
    H = Qt.conj().T @ Qt
    if la.norm(H - np.eye(H.shape[0]), ord="fro") > max(gram_tol, 10 * gram_error):
        vals_h, vecs_h = la.eigh(0.5 * (H + H.conj().T))
        vals_h = np.clip(vals_h, 1e-15, None)
        Qt = Qt @ (vecs_h @ np.diag(1.0 / np.sqrt(vals_h)) @ vecs_h.conj().T)
    Qs = Qs_qr

    Es = _complete_basis(Qs)
    Et = _complete_basis(Qt)
    U = Et @ Es.conj().T
    map_error = float(la.norm(U @ source - target, ord="fro"))
    unitarity_error = float(la.norm(U.conj().T @ U - np.eye(N), ord="fro"))
    return U, gram_error, map_error, unitarity_error


def compute_transducer_unitary(
    sol: GeneralAdversarySolution,
    *,
    gram_tol: float = 1e-7,
    eig_tol: float = 1e-9,
) -> TransducerUnitary:
    """Build the input-independent canonical transducer unitary from an SDP solution."""
    source, target, public_dim, basis = build_transducer_state_matrices(sol)
    U, gram_error, map_error, unitarity_error = unitary_from_state_pairs(
        source, target, gram_tol=gram_tol, eig_tol=eig_tol
    )
    return TransducerUnitary(
        U=U,
        source_states=source,
        target_states=target,
        public_dim=public_dim,
        private_dim=source.shape[0] - public_dim,
        basis_slices=basis,
        gram_error=gram_error,
        map_error=map_error,
        unitarity_error=unitarity_error,
    )


def verify_adversary_constraints(sol: GeneralAdversarySolution) -> dict[str, float]:
    """Return residuals for the SDP equality constraints and objective diagonals."""
    inputs, outputs = sol.inputs, sol.outputs
    m = len(inputs)
    n = len(inputs[0])
    max_eq = 0.0
    fro_sq = 0.0
    for x_idx, x in enumerate(inputs):
        for y_idx, y in enumerate(inputs):
            lhs = 0.0
            for i in range(n):
                if x[i] != y[i]:
                    lhs += float(sol.u[i][x_idx, :] @ sol.v[i][y_idx, :])
            rhs = 1.0 if outputs[x_idx] != outputs[y_idx] else 0.0
            err = lhs - rhs
            max_eq = max(max_eq, abs(err))
            fro_sq += err * err
    costs = []
    for x_idx in range(m):
        cost = 0.0
        for i in range(n):
            cost += np.dot(sol.u[i][x_idx, :], sol.u[i][x_idx, :])
            cost += np.dot(sol.v[i][x_idx, :], sol.v[i][x_idx, :])
        costs.append(0.5 * cost)
    return {
        "max_constraint_abs_error": max_eq,
        "constraint_fro_error": float(np.sqrt(fro_sq)),
        "max_diagonal_cost": float(max(costs) if costs else 0.0),
        "objective_value": float(sol.objective_value),
    }


def demo_or_2() -> None:
    """Small executable demo: OR on two bits."""
    inputs = all_binary_inputs(2)
    outputs = [int(any(x)) for x in inputs]
    sol = solve_general_adversary_sdp(
        inputs,
        outputs,
        solver="CLARABEL",
        solver_kwargs={"tol_gap_abs": 1e-9, "tol_feas": 1e-9, "tol_gap_rel": 1e-9},
    )
    transducer = compute_transducer_unitary(sol)
    print("status:", sol.status)
    print("Adv objective:", sol.objective_value)
    print("constraint residuals:", verify_adversary_constraints(sol))
    print("U shape:", transducer.U.shape)
    print("Gram error:", transducer.gram_error)
    print("map error ||U A - B||_F:", transducer.map_error)
    print("unitarity error ||U*U-I||_F:", transducer.unitarity_error)


# Demo entry point is defined at the end of the file, after all helper functions.


def query_completion_for_input(
    transducer: TransducerUnitary,
    x: Sequence[int],
) -> Array:
    """
    Build a concrete unitary completion of the input query on H plus L.

    The adversary transducer only requires the state-generating action
        |i, up, 0>   -> |i, up, x_i>
        |i, down, x_i> -> |i, down, 0>.
    This routine completes that partial action by the transposition 0 <-> x_i
    on every (i, direction) query-symbol register and fixes all other symbols.

    Returns a matrix Q_x on public_dim + private_dim such that the canonical
    transducer step is S_x = U @ Q_x.
    """
    h = transducer.public_dim
    l = transducer.private_dim
    Q = np.eye(h + l, dtype=complex)
    P = np.zeros((l, l), dtype=complex)

    # Group available query-symbol slices by (index, direction).
    groups: dict[tuple[int, str], dict[int, slice]] = {}
    for (i, direction, symbol), sl in transducer.basis_slices.items():
        groups.setdefault((i, direction), {})[symbol] = sl

    for (i, direction), symbol_to_slice in groups.items():
        xi = int(x[i])
        if 0 not in symbol_to_slice or xi not in symbol_to_slice:
            raise ValueError(
                f"input symbol x[{i}]={xi} is not present in the transducer basis"
            )
        for symbol, sl_src in symbol_to_slice.items():
            if symbol == 0:
                dest_symbol = xi
            elif symbol == xi:
                dest_symbol = 0
            else:
                dest_symbol = symbol
            sl_dst = symbol_to_slice[dest_symbol]
            width = sl_src.stop - sl_src.start
            if sl_dst.stop - sl_dst.start != width:
                raise ValueError("basis slices for a query block have inconsistent widths")
            P[sl_dst, sl_src] = np.eye(width, dtype=complex)

    Q[h:, h:] = P
    return Q


def transducer_step_for_input(
    transducer: TransducerUnitary,
    x: Sequence[int],
) -> Array:
    """Return the concrete transducer unitary S_x = U Q_x on H plus L."""
    return transducer.U @ query_completion_for_input(transducer, x)


def lift_step_to_public_slot(S: Array, public_dim: int, private_dim: int, K: int, slot: int) -> Array:
    """
    Lift S on H plus L to act on the selected public slot H_slot plus shared L.

    The implementation space is (C^K tensor H) plus L, represented as the direct
    sum H_0 plus ... plus H_{K-1} plus L. All public slots except `slot` are fixed.
    """
    if not (0 <= slot < K):
        raise ValueError("slot must be between 0 and K-1")
    h, l = public_dim, private_dim
    N = K * h + l
    if S.shape != (h + l, h + l):
        raise ValueError("S has incompatible shape")
    out = np.eye(N, dtype=complex)

    S_hh = S[:h, :h]
    S_hl = S[:h, h:]
    S_lh = S[h:, :h]
    S_ll = S[h:, h:]

    ps = slice(slot * h, (slot + 1) * h)
    ls = slice(K * h, K * h + l)

    out[ps, ps] = S_hh
    out[ps, ls] = S_hl
    out[ls, ps] = S_lh
    out[ls, ls] = S_ll
    return out


def public_spread_unitary(public_dim: int, private_dim: int, K: int) -> Array:
    """
    Return a unitary that maps |0>_T|psi> to K^{-1/2} sum_t |t>_T|psi>.

    It is the DFT on the K-dimensional slot register tensor I_public, direct-summed
    with identity on the private space.
    """
    if K <= 0:
        raise ValueError("K must be positive")
    h, l = public_dim, private_dim
    omega = np.exp(2j * np.pi / K)
    F = np.array([[omega ** (a * b) for b in range(K)] for a in range(K)], dtype=complex)
    F /= np.sqrt(K)
    public = np.kron(F, np.eye(h, dtype=complex))
    return la.block_diag(public, np.eye(l, dtype=complex))


def repeated_transducer_algorithm_unitary(
    S: Array,
    public_dim: int,
    private_dim: int,
    K: int,
) -> Array:
    """
    Build the K-repeat algorithm implementing the transduction action of S.

    The returned unitary acts on (C^K tensor H) plus L:
      1. spread the input public state over K public slots;
      2. for t = 0, ..., K-1, apply S to slot t and the shared private space;
      3. unspread the public slots.

    Starting from |0>_T|xi> plus zero private state, the first public slot of the
    output approximates the transduction target, with ideal bound 2 sqrt(W / K)
    when a catalyst of norm squared at most W exists.
    """
    h, l = public_dim, private_dim
    Spread = public_spread_unitary(h, l, K)
    A = Spread.copy()
    for slot in range(K):
        A = lift_step_to_public_slot(S, h, l, K, slot) @ A
    A = Spread.conj().T @ A
    return A


def run_repeated_transducer_algorithm(
    transducer: TransducerUnitary,
    x: Sequence[int],
    K: int,
    initial_public_state: Array | None = None,
) -> dict[str, Array | float]:
    """
    Simulate the K-repeat algorithm for one input x.

    Returns the full final vector, the first public output slot, and the norm of
    everything outside that first output slot. For function evaluation, measuring
    the first public slot should yield f(x) with high probability when K is large.
    """
    h, l = transducer.public_dim, transducer.private_dim
    if initial_public_state is None:
        psi = np.zeros(h, dtype=complex)
        psi[0] = 1.0
    else:
        psi = np.asarray(initial_public_state, dtype=complex)
        if psi.shape != (h,):
            raise ValueError("initial_public_state has incompatible shape")
    Sx = transducer_step_for_input(transducer, x)
    A = repeated_transducer_algorithm_unitary(Sx, h, l, K)
    init = np.zeros(K * h + l, dtype=complex)
    init[:h] = psi
    final = A @ init
    output_slot = final[:h]
    garbage_norm = float(la.norm(final[h:]))
    return {
        "algorithm_unitary": A,
        "final_state": final,
        "output_slot": output_slot,
        "garbage_norm": garbage_norm,
    }


def catalyst_norm_bound(transducer: TransducerUnitary) -> float:
    """Return max_x ||before-query catalyst_x||^2 from the constructed columns."""
    h = transducer.public_dim
    private_targets = transducer.target_states[h:, :]
    if private_targets.shape[1] == 0:
        return 0.0
    return float(np.max(np.sum(np.abs(private_targets) ** 2, axis=0)))


def choose_repetition_count(W: float, epsilon: float, safety_factor: float = 4.0) -> int:
    """
    Choose K for the repeated-transducer algorithm.

    If a catalyst has norm squared at most W, the standard conversion gives
    vector error at most 2 * sqrt(W / K).  The default safety_factor=4 gives
    K = ceil(4W/epsilon^2), which makes that ideal bound at most epsilon.
    The function always returns at least 1.
    """
    if W < -1e-12:
        raise ValueError("W must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")
    return max(1, int(np.ceil(safety_factor * max(0.0, W) / (epsilon ** 2))))


def build_repeated_algorithm_for_input(
    transducer: TransducerUnitary,
    x: Sequence[int],
    K: int,
) -> Array:
    """
    Extract the concrete query algorithm for one input x.

    This is the matrix form of the bounded-error algorithm obtained from the
    transducer.  Internally it constructs the input-dependent transducer step

        S_x = U Q_x,

    where U is the input-independent unitary extracted from the SDP witness and
    Q_x is the completed query operation, then returns the K-repeat conversion
    on (C^K tensor H) plus L.

    In a true query-algorithm implementation, Q_x should not be materialized;
    it is the oracle call.  This routine materializes it only for simulation.
    """
    Sx = transducer_step_for_input(transducer, x)
    return repeated_transducer_algorithm_unitary(
        Sx,
        public_dim=transducer.public_dim,
        private_dim=transducer.private_dim,
        K=K,
    )


def output_probabilities_from_slot(output_slot: Array) -> Array:
    """Return computational-basis probabilities in the first public output slot."""
    output_slot = np.asarray(output_slot, dtype=complex)
    return np.abs(output_slot) ** 2


import numpy as np


def show_W_size(sol, *, assume_boolean_query_register=True, print_report=True):
    """
    Report the witness-space dimensions W_i and the transduction complexity W_x.

    Args:
        sol:
            A GeneralAdversarySolution returned by solve_general_adversary_sdp().
        assume_boolean_query_register:
            If True, use q_i = 2 for every input coordinate, matching the standard
            Boolean query-symbol register {0,1}. If False, use the alphabet sizes
            actually stored in sol.alphabet_values.
        print_report:
            If True, print a readable report.

    Returns:
        A dictionary containing:
            - dim_W_i: list of dimensions dim(W_i)
            - dim_W_padded: dimension of a common padded W, max_i dim(W_i)
            - dim_L_i: private-space contribution from each coordinate
            - dim_L_total: total private/catalyst Hilbert-space dimension
            - dim_H: public output-space dimension
            - dim_H_plus_L: transducer Hilbert-space dimension
            - W_x: transduction complexity by input
            - W_max: worst-case transduction complexity
            - sdp_objective: SDP objective value
    """
    inputs = sol.inputs
    outputs = sol.outputs
    n = len(inputs[0])
    m = len(inputs)

    # dim(W_i) is the number of columns in the extracted PSD factor.
    # sol.u[i] has shape (|D|, dim(W_i)).
    dim_W_i = [int(sol.u[i].shape[1]) for i in range(n)]

    # In the paper's padded notation, one can embed all W_i into a common W.
    dim_W_padded = max(dim_W_i) if dim_W_i else 0

    if assume_boolean_query_register:
        q_i = [2 for _ in range(n)]
    else:
        q_i = [len(sol.alphabet_values[i]) for i in range(n)]

    # For each coordinate i:
    # L_i = W_i ⊗ span{up,down} ⊗ Q_i
    # dim(L_i) = dim(W_i) * 2 * q_i.
    dim_L_i = [2 * q_i[i] * dim_W_i[i] for i in range(n)]
    dim_L_total = int(sum(dim_L_i))

    # Public output space H = C^p.
    dim_H = int(max(outputs) + 1) if outputs else 0
    dim_H_plus_L = dim_H + dim_L_total

    # Transduction complexity for each input x:
    #
    # W_x = 1/2 sum_i (||u_{x,i}||^2 + ||v_{x,i}||^2).
    #
    # This equals the squared norm of the catalyst v_x in this adversary transducer.
    W_x = {}
    for x_idx, x in enumerate(inputs):
        cost = 0.0
        for i in range(n):
            cost += np.vdot(sol.u[i][x_idx, :], sol.u[i][x_idx, :]).real
            cost += np.vdot(sol.v[i][x_idx, :], sol.v[i][x_idx, :]).real
        W_x[x] = 0.5 * float(cost)

    W_max = max(W_x.values()) if W_x else 0.0

    report = {
        "num_inputs_|D|": m,
        "num_variables_n": n,
        "query_symbol_dims_q_i": q_i,
        "dim_W_i": dim_W_i,
        "dim_W_padded": dim_W_padded,
        "dim_L_i": dim_L_i,
        "dim_L_total": dim_L_total,
        "dim_H": dim_H,
        "dim_H_plus_L": dim_H_plus_L,
        "W_x": W_x,
        "W_max": W_max,
        "sdp_objective": float(sol.objective_value),
    }

    if print_report:
        print("Witness-space dimensions:")
        for i in range(n):
            print(
                f"  i={i}: dim(W_{i})={dim_W_i[i]}, "
                f"q_i={q_i[i]}, dim(L_{i})=2*q_i*dim(W_i)={dim_L_i[i]}"
            )

        print()
        print(f"Common padded dim(W) = max_i dim(W_i) = {dim_W_padded}")
        print(f"Total private dimension dim(L) = {dim_L_total}")
        print(f"Public dimension dim(H) = {dim_H}")
        print(f"Total transducer dimension dim(H ⊕ L) = {dim_H_plus_L}")

        print()
        print("Transduction complexity by input:")
        for x, val in W_x.items():
            print(f"  W_{x} = {val:.12g}")

        print()
        print(f"W_max = {W_max:.12g}")
        print(f"SDP objective = {float(sol.objective_value):.12g}")
        print(f"|W_max - SDP objective| = {abs(W_max - float(sol.objective_value)):.3e}")

    return report


