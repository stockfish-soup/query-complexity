"""Finite-action numerical Wedderburn blocks, with explicit multiplicity lifts.

This reference backend enumerates the permutation group. It supports complex
irreps too, and is intended for validation-sized problems, not large S_n.
"""
from dataclasses import dataclass
import numpy as np


def compose(p, q):
    return tuple(p[i] for i in q)


def inverse(p):
    return tuple(np.argsort(p))


def closure(generators, size, max_order=10000):
    identity = tuple(range(size))
    for p in generators:
        if sorted(p) != list(identity):
            raise ValueError('Each generator must be a permutation of the state indices')
    seen, pending = {identity}, [identity]
    while pending:
        g = pending.pop()
        for h in generators:
            gh = compose(g, h)
            if gh not in seen:
                if len(seen) >= max_order:
                    raise ValueError('Group enumeration exceeds max_order; use a structural backend')
                seen.add(gh)
                pending.append(gh)
    return sorted(seen)


@dataclass
class Sector:
    label: tuple
    degree: int
    multiplicity: int
    basis: np.ndarray
    lift: np.ndarray
    real: bool

    @property
    def projector(self):
        return self.basis @ self.basis.conj().T


def decompose(permutations, tolerance=1e-8):
    """Split the permutation module by central class sums, trivial type first.

    Lift vec(B) to vec(I_degree tensor B) in the original coordinates. Class
    sum eigenvalues provide deterministic labels, ordered lexicographically
    after the trivial sector. Equivalent copies are kept together.
    """
    permutations = sorted(set(tuple(p) for p in permutations))
    order, size = len(permutations), len(permutations[0])
    identity = tuple(range(size))
    group_set = set(permutations)
    if identity not in group_set:
        raise ValueError('Expected a complete finite group')
    matrices = {p: np.eye(size)[:, p] for p in permutations}
    remaining = set(permutations)
    classes = []
    while remaining:
        g = min(remaining)
        conjugates = {compose(compose(h, g), inverse(h)) for h in permutations}
        if not conjugates <= group_set:
            raise ValueError('Expected a complete finite group')
        remaining -= conjugates
        classes.append(sorted(conjugates))
    classes.sort(key=lambda c: c[0])
    central = [sum((matrices[p] for p in c), np.zeros((size, size))) for c in classes]
    spaces = [np.eye(size, dtype=complex)]
    # Real and imaginary Hermitian parts distinguish complex conjugate types.
    for C in central:
        for A in [(C + C.T) / 2, (C - C.T) / (2j)]:
            split = []
            for U in spaces:
                values, vectors = np.linalg.eigh(U.conj().T @ A @ U)
                clusters = np.split(np.arange(len(values)), np.flatnonzero(np.diff(values) > tolerance) + 1)
                split.extend(U @ vectors[:, ids] for ids in clusters)
            spaces = split
    rng = np.random.default_rng(1729)
    random_algebra = sum((rng.normal() * R for R in matrices.values()), np.zeros((size, size)))
    # A generic complex Hermitian group-algebra element splits irrep coordinates.
    random_algebra = random_algebra + 1j * sum((rng.normal() * R for R in matrices.values()), np.zeros((size, size)))
    random_algebra = (random_algebra + random_algebra.conj().T) / 2
    sectors = []
    for U in spaces:
        rank = U.shape[1]
        normalized_character = np.array([np.trace(U.conj().T @ R @ U) / rank for R in matrices.values()])
        degree = round(np.sqrt(order / np.sum(np.abs(normalized_character)**2)))
        if degree < 1 or rank % degree:
            raise ArithmeticError('Unresolved isotypic sector; adjust numerical tolerance')
        multiplicity = rank // degree
        values, vectors = np.linalg.eigh(U.conj().T @ random_algebra @ U)
        ids = np.flatnonzero(np.abs(values - values[0]) < tolerance)
        if len(ids) != multiplicity:
            raise ArithmeticError('Group-algebra split is degenerate at the chosen tolerance')
        V = U @ vectors[:, ids]
        W = np.array([R @ V for R in matrices.values()])
        tensor = degree / order * np.einsum('gia,gjb->ijab', W, W.conj())
        lift = tensor.reshape((size*size, multiplicity*multiplicity), order='F')
        # For real types choose a real multiplicity basis before building the lift.
        if np.max(np.abs(U.imag)) < tolerance:
            real_algebra = random_algebra.real
            values, vectors = np.linalg.eigh(U.real.T @ real_algebra @ U.real)
            ids = np.flatnonzero(np.abs(values - values[0]) < tolerance)
            if len(ids) == multiplicity:
                V = U.real @ vectors[:, ids]
                W = np.array([R @ V for R in matrices.values()])
                lift = (degree / order * np.einsum('gia,gjb->ijab', W, W)).reshape((size*size, multiplicity*multiplicity), order='F')
        trivial = max(abs(normalized_character - 1)) < tolerance
        signature = tuple((round(float(np.trace(U.conj().T @ C @ U).real / rank), 7),
                           round(float(np.trace(U.conj().T @ C @ U).imag / rank), 7)) for C in central)
        sectors.append((not trivial, signature, Sector(signature, degree, multiplicity, U, lift,
                                                      np.max(np.abs(lift.imag)) < tolerance)))
    sectors.sort(key=lambda entry: (entry[0], entry[1]))
    if sectors[0][0]:
        raise ArithmeticError('A permutation module must contain a trivial sector')
    return [s[2] for s in sectors]
