import numpy as np
import scipy.linalg as la


def _next_power_of_two(d: int) -> int:
    """Smallest power of two >= d."""
    if d <= 0:
        raise ValueError("dimension must be positive")
    return 1 << int(np.ceil(np.log2(d)))


def _num_qubits_for_dim(d: int) -> int:
    """Number of qubits needed to embed a d-dimensional Hilbert space."""
    return int(np.ceil(np.log2(d)))


def pad_unitary_to_qubits(U: np.ndarray, logical_dim: int | None = None) -> tuple[np.ndarray, int, int]:
    """
    Embed a logical d x d unitary into a 2^q x 2^q qubit unitary.

    Returns:
        padded_U, num_qubits, padded_dim
    """
    U = np.asarray(U, dtype=complex)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be square")

    d = U.shape[0]

    if logical_dim is None:
        logical_dim = d

    if logical_dim != d:
        raise ValueError("logical_dim must match U.shape[0] for this helper")

    padded_dim = _next_power_of_two(d)
    num_qubits = _num_qubits_for_dim(d)

    out = np.eye(padded_dim, dtype=complex)
    out[:d, :d] = U

    return out, num_qubits, padded_dim


def logical_public_spread_unitary(public_dim: int, private_dim: int, K: int) -> np.ndarray:
    """
    Logical spread unitary on

        (C^K tensor H) ⊕ L.

    It maps

        |0>_slot |psi>_H

    to

        1/sqrt(K) sum_t |t>_slot |psi>_H,

    and acts as identity on L.
    """
    if K <= 0:
        raise ValueError("K must be positive")

    h = int(public_dim)
    l = int(private_dim)

    omega = np.exp(2j * np.pi / K)

    F = np.array(
        [[omega ** (a * b) for b in range(K)] for a in range(K)],
        dtype=complex,
    )
    F /= np.sqrt(K)

    public = np.kron(F, np.eye(h, dtype=complex))
    return la.block_diag(public, np.eye(l, dtype=complex))


def logical_lift_step_to_public_slot(
    S: np.ndarray,
    public_dim: int,
    private_dim: int,
    K: int,
    slot: int,
) -> np.ndarray:
    """
    Lift a logical transducer-space unitary S on H ⊕ L so that it acts on

        H_slot ⊕ L

    inside

        H_0 ⊕ ... ⊕ H_{K-1} ⊕ L.

    All other public slots are fixed.
    """
    if not (0 <= slot < K):
        raise ValueError("slot must satisfy 0 <= slot < K")

    S = np.asarray(S, dtype=complex)

    h = int(public_dim)
    l = int(private_dim)

    if S.shape != (h + l, h + l):
        raise ValueError(
            f"S has shape {S.shape}, expected {(h + l, h + l)}"
        )

    N = K * h + l
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


def qiskit_repeated_transducer_circuit_from_oracle_matrix(
    transducer_U: np.ndarray,
    oracle_Q: np.ndarray,
    public_dim: int,
    private_dim: int,
    K: int,
    *,
    measure_all: bool = False,
    name: str = "RepeatedTransducer",
):
    """
    Build a Qiskit circuit for the K-repeat transducer algorithm.

    Args:
        transducer_U:
            The input-independent transducer unitary U on H ⊕ L.
        oracle_Q:
            The query unitary Q_x on H ⊕ L.
            This should be I_H ⊕ oracle_private.
            For the one-vector direct-sum implementation, this is returned by
            one_vector_direct_sum_query_for_input(transducer, x_idx).
        public_dim:
            dim(H).
        private_dim:
            dim(L).
        K:
            Number of repetitions.
        measure_all:
            If True, append measurements of all physical qubits.
        name:
            Circuit name.

    Returns:
        qc:
            A Qiskit QuantumCircuit.
        metadata:
            Dictionary containing logical and padded dimensions.

    Logical action:
        Spread;
        for slot = 0,...,K-1:
            apply lifted oracle Q_x to H_slot ⊕ L;
            apply lifted U to H_slot ⊕ L;
        Unspread.

    The circuit is "oracle based" in the sense that oracle_Q is appended as a
    separate gate at every query location. If you later have a more efficient
    oracle subcircuit, you can replace those dense oracle gates by that subcircuit.
    """
    from qiskit import QuantumCircuit

    U = np.asarray(transducer_U, dtype=complex)
    Q = np.asarray(oracle_Q, dtype=complex)

    h = int(public_dim)
    l = int(private_dim)

    if U.shape != (h + l, h + l):
        raise ValueError(f"transducer_U has shape {U.shape}, expected {(h + l, h + l)}")

    if Q.shape != (h + l, h + l):
        raise ValueError(f"oracle_Q has shape {Q.shape}, expected {(h + l, h + l)}")

    logical_dim = K * h + l
    padded_dim = _next_power_of_two(logical_dim)
    num_qubits = _num_qubits_for_dim(logical_dim)

    qc = QuantumCircuit(num_qubits, name=name)

    qubits = list(range(num_qubits))

    def append_logical_unitary(M: np.ndarray, label: str):
        M = np.asarray(M, dtype=complex)

        if M.shape != (logical_dim, logical_dim):
            raise ValueError(
                f"{label} has shape {M.shape}, expected {(logical_dim, logical_dim)}"
            )

        padded = np.eye(padded_dim, dtype=complex)
        padded[:logical_dim, :logical_dim] = M

        qc.unitary(padded, qubits, label=label)

    # 1. Spread the public input over K slots.
    Spread = logical_public_spread_unitary(h, l, K)
    append_logical_unitary(Spread, "Spread")

    # 2. Apply oracle and U slot by slot.
    for slot in range(K):
        Q_slot = logical_lift_step_to_public_slot(Q, h, l, K, slot)
        U_slot = logical_lift_step_to_public_slot(U, h, l, K, slot)

        append_logical_unitary(Q_slot, f"Oracle[{slot}]")
        append_logical_unitary(U_slot, f"U[{slot}]")

    # 3. Unspread.
    append_logical_unitary(Spread.conj().T, "Unspread")

    if measure_all:
        qc.measure_all()

    metadata = {
        "logical_dim": logical_dim,
        "padded_dim": padded_dim,
        "num_qubits": num_qubits,
        "public_dim": h,
        "private_dim": l,
        "K": K,
        "intended_output_indices": list(range(h)),
        "garbage_indices": list(range(h, logical_dim)),
        "unused_padding_indices": list(range(logical_dim, padded_dim)),
    }

    return qc, metadata


def logical_output_probabilities_from_statevector(
    statevector,
    *,
    public_dim: int,
    logical_dim: int,
) -> dict[str, np.ndarray | float]:
    """
    Decode output probabilities from a Qiskit Statevector.

    The intended output lives in logical indices 0,...,public_dim-1.
    Everything from public_dim to logical_dim-1 is algorithmic garbage.
    Everything from logical_dim onward is unused padding.
    """
    data = np.asarray(statevector.data, dtype=complex)
    probs = np.abs(data) ** 2

    output_probs = probs[:public_dim]
    garbage_prob = float(np.sum(probs[public_dim:logical_dim]))
    padding_prob = float(np.sum(probs[logical_dim:]))

    return {
        "output_probabilities": output_probs,
        "garbage_probability": garbage_prob,
        "padding_probability": padding_prob,
        "total_probability": float(np.sum(probs)),
    }