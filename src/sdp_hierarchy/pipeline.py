"""Covariant adversary SDPs restricted to prefixes of stabilizer irreps."""
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from .representation import closure, inverse, decompose


@dataclass
class OracleProblem:
    answers: np.ndarray
    generators: list
    targets: object = None

    def validate(self):
        self.answers = np.asarray(self.answers)
        if self.answers.ndim != 2 or not np.isin(self.answers, [0, 1]).all():
            raise ValueError('answers must be a nonempty binary concepts-by-queries table')
        n, q = self.answers.shape
        if not n or not q:
            raise ValueError('Empty oracle tables are not supported')
        if self.targets is None:
            self.targets = np.arange(n)
        self.targets = np.asarray(self.targets)
        if self.targets.shape != (n,):
            raise ValueError('One target per concept is required')
        separate = self.targets[:, None] != self.targets[None, :]
        for p, s in self.generators:
            if sorted(p) != list(range(n)) or sorted(s) != list(range(q)):
                raise ValueError('Invalid concept/query permutation')
            if not np.array_equal(self.answers[np.ix_(p, s)], self.answers):
                raise ValueError('Oracle table is not equivariant under a generator')
            if not np.array_equal(separate[np.ix_(p, p)], separate):
                raise ValueError('Target equivalence is not group invariant')
        if np.any(separate & np.all(self.answers[:, None, :] == self.answers[None, :, :], axis=2)):
            raise ValueError('Distinct targets have indistinguishable oracles')
        return separate


class Hierarchy:
    """Prepare once, solve level r using the first r+1 present stabilizer types.

    Each query orbit has its own stabilizer and irrep list. Level zero keeps
    exactly its trivial isotypic component. The final level spans the full
    covariant dual SDP. This enumerative backend is for small finite actions.
    """
    def __init__(self, problem, max_order=10000, decomposer=decompose, sector_order=None):
        self.problem = problem
        self.separate = problem.validate()
        self.n, self.q = problem.answers.shape
        generators = [tuple(p) + tuple(self.n + j for j in s) for p, s in problem.generators]
        self.group = closure(generators, self.n + self.q, max_order)
        pending = set(range(self.q))
        self.orbits = []
        while pending:
            rep = min(pending)
            transports = {}
            stabilizer = []
            for g in self.group:
                target = g[self.n + rep] - self.n
                transports.setdefault(target, tuple(g[:self.n]))
                if target == rep:
                    stabilizer.append(tuple(g[:self.n]))
            pending -= set(transports)
            sectors = decomposer(stabilizer)
            if sector_order is not None:
                sectors = sectors[:1] + sorted(sectors[1:], key=sector_order)
            self.orbits.append(dict(representative=rep, transports=transports,
                                    stabilizer=sorted(set(stabilizer)), sectors=sectors))
        self.max_level = max(len(o['sectors']) - 1 for o in self.orbits)

    def block_summary(self):
        return [dict(query=o['representative'], orbit_size=len(o['transports']),
                     sectors=[dict(index=i, degree=s.degree, multiplicity=s.multiplicity,
                                   label=s.label) for i, s in enumerate(o['sectors'])])
                for o in self.orbits]

    def solve(self, level=0, solver='CLARABEL', dense=False, intersection_free=False, validation_tol=1e-6):
        """Solve a hierarchy prefix, or an independent dense covariant benchmark.

        dense=True bypasses the decomposition, imposing stabilizer invariance
        directly on full PSD matrices. Returned matrices are in original
        concept coordinates; diagnostics check the reconstructed witnesses.
        """
        if not isinstance(level, int) or not 0 <= level <= self.max_level:
            raise ValueError(f'level must be an integer in [0,{self.max_level}]')
        if dense and intersection_free:
            raise ValueError('Choose one reference formulation')
        if intersection_free and level != 0:
            raise ValueError('The intersection-free reference is level zero')
        constraints, reps, matrices, variables = [], [], [None]*self.q, []
        for orbit in self.orbits:
            if intersection_free:
                # Independent orbit-indicator construction of the fixed space.
                unseen, columns = set(range(self.n)), []
                while unseen:
                    i = min(unseen)
                    orbit_indices = sorted({p[i] for p in orbit['stabilizer']})
                    unseen -= set(orbit_indices)
                    v = np.zeros(self.n)
                    v[orbit_indices] = 1 / np.sqrt(len(orbit_indices))
                    columns.append(v)
                V = np.column_stack(columns)
                B = cp.Variable((len(columns), len(columns)), symmetric=True)
                variables.append(B)
                constraints.append(B >> 0)
                X = V @ B @ V.T
            elif dense:
                X = cp.Variable((self.n, self.n), symmetric=True)
                constraints.append(X >> 0)
                for p in orbit['stabilizer']:
                    inv = inverse(p)
                    constraints.append(X[np.ix_(inv, inv)] == X)
            else:
                terms = []
                for s in orbit['sectors'][:level+1]:
                    B = cp.Variable((s.multiplicity, s.multiplicity), symmetric=True) if s.real else cp.Variable((s.multiplicity, s.multiplicity), hermitian=True)
                    constraints.append(B >> 0)
                    variables.append(B)
                    lift = s.lift.real if s.real else s.lift
                    terms.append(cp.reshape(lift @ cp.vec(B, order='F'), (self.n, self.n), order='F'))
                X = sum(terms)
            reps.append(X)
            for query, p in orbit['transports'].items():
                inv = inverse(p)
                matrices[query] = X[np.ix_(inv, inv)]
        filtered = sum(cp.multiply(self.problem.answers[:, j, None] != self.problem.answers[None, :, j], X)
                       for j, X in enumerate(matrices))
        if self.separate.any():
            constraints.append(filtered[np.triu(self.separate, 1)] == 1)
        t = cp.Variable()
        load = sum(cp.diag(X) if X.is_real() else cp.real(cp.diag(X)) for X in matrices)
        constraints.extend([load <= t, t >= 0])
        prob = cp.Problem(cp.Minimize(t), constraints)
        options = dict(tol_gap_abs=1e-8, tol_feas=1e-8, max_iter=500) if solver == 'CLARABEL' else {}
        used_solver = solver
        try:
            prob.solve(solver=solver, **options)
        except cp.error.SolverError:
            if solver != 'CLARABEL':
                raise
            used_solver = 'SCS'
            prob.solve(solver='SCS', eps=1e-8, max_iters=100000)

        result = dict(level=level, solver=used_solver, status=prob.status, cost=float(prob.value),
                      block_sizes=[B.shape[0] for B in variables])
        if any(X.value is None for X in matrices):
            result['numerically_feasible'] = False
            return result
        values = np.array([X.value for X in matrices])
        sums = sum((self.problem.answers[:, j, None] != self.problem.answers[None, :, j]) * X
                   for j, X in enumerate(values))
        leakage, covariance = 0., 0.
        for orbit, expr in zip(self.orbits, reps):
            X = expr.value
            P = sum(s.projector for s in orbit['sectors'][:level+1])
            leakage = max(leakage, float(np.linalg.norm(X - P @ X @ P)))
            for p in orbit['stabilizer']:
                inv = inverse(p)
                covariance = max(covariance, float(np.max(abs(X[np.ix_(inv, inv)] - X))))
        result.update(matrices=values, equality_residual=float(np.max(abs(sums[self.separate]-1), initial=0)),
                      minimum_eigenvalue=float(min(np.linalg.eigvalsh((X + X.conj().T)/2).min() for X in values)),
                      maximum_load=float(np.real(np.diagonal(values, axis1=1, axis2=2)).sum(axis=0).max()),
                      support_leakage=None if dense else leakage, covariance_residual=covariance)
        result['numerically_feasible'] = (
            result['equality_residual'] <= validation_tol
            and result['minimum_eigenvalue'] >= -validation_tol
            and result['maximum_load'] <= result['cost'] + validation_tol
            and covariance <= validation_tol
            and (dense or leakage <= validation_tol)
        )
        return result
