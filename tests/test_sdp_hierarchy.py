import math
import numpy as np
import cvxpy as cp
import pytest
from sdp_hierarchy import Hierarchy, OracleProblem, symmetric_evaluation, symmetric_junta
from sdp_hierarchy.representation import closure, compose, decompose, inverse


def assert_feasible(result):
    assert result['status'] in {'optimal', 'optimal_inaccurate'}
    assert result['equality_residual'] < 2e-6
    assert result['minimum_eigenvalue'] > -2e-6
    assert result['maximum_load'] <= result['cost'] + 2e-6
    assert result['covariance_residual'] < 1e-7
    if result['support_leakage'] is not None:
        assert result['support_leakage'] < 1e-7


@pytest.mark.parametrize('problem,expected', [
    (symmetric_evaluation(3,[0,1,1,1]), math.sqrt(3)),
    (symmetric_evaluation(3,[0,0,1,1]), 2),
    (symmetric_evaluation(3,[0,1,0,1]), 3),
    (symmetric_evaluation(4,[0,1,1,1,1]), None),
    (symmetric_junta(4,2,[0,1,1]), 2),
    (symmetric_junta(5,3,[0,0,1,1]), None),
])
def test_prefixes_intersection_free_and_dense_endpoint(problem, expected):
    h = Hierarchy(problem)
    levels = [h.solve(r) for r in range(h.max_level+1)]
    for result in levels:
        assert_feasible(result)
    costs = [r['cost'] for r in levels]
    assert np.max(np.diff(costs), initial=0) < 2e-6
    reference = h.solve(intersection_free=True)
    assert_feasible(reference)
    assert reference['cost'] == pytest.approx(costs[0], abs=2e-6)
    dense = h.solve(dense=True)
    assert_feasible(dense)
    assert dense['cost'] == pytest.approx(costs[-1], abs=2e-6)
    if expected is not None:
        assert costs[-1] == pytest.approx(expected, abs=2e-6)


def test_wedderburn_multiplicities_on_two_regular_S3_modules():
    group = closure([(1,0,2), (0,2,1)], 3)
    lookup = {p:i for i,p in enumerate(group)}
    action = [tuple(lookup[compose(g,h)] + offset for offset in [0,6] for h in group) for g in group]
    sectors = decompose(action)
    assert sorted((s.degree,s.multiplicity) for s in sectors) == [(1,2),(1,2),(2,4)]
    np.testing.assert_allclose(sum(s.projector for s in sectors), np.eye(12), atol=1e-10)
    rng = np.random.default_rng(7)
    for s in sectors:
        A = rng.normal(size=(s.multiplicity,s.multiplicity))
        B = A @ A.T
        X = (s.lift @ B.reshape(-1,order='F')).reshape((12,12),order='F')
        np.testing.assert_allclose(np.trace(X), s.degree*np.trace(B), atol=1e-9)
        assert np.linalg.eigvalsh(X).min() > -1e-9
        np.testing.assert_allclose(X, s.projector @ X @ s.projector, atol=1e-9)
        for p in action:
            inv = inverse(p)
            np.testing.assert_allclose(X[np.ix_(inv,inv)], X, atol=1e-9)
        # Every eigenvalue of B is repeated degree times in its isotypic lift.
        np.testing.assert_allclose(np.linalg.eigvalsh(X)[-s.degree*s.multiplicity:],
                                   np.repeat(np.linalg.eigvalsh(B),s.degree), atol=1e-9)


def test_complex_irreps_are_not_merged():
    group = closure([(1,2,0)],3)
    sectors = decompose(group)
    assert [(s.degree,s.multiplicity) for s in sectors] == [(1,1)]*3
    np.testing.assert_allclose(sum(s.projector for s in sectors), np.eye(3), atol=1e-10)


def test_terminal_matches_completely_unreduced_sdp():
    p = symmetric_evaluation(3,[0,1,0,0])
    h = Hierarchy(p)
    X = [cp.Variable((8,8), symmetric=True) for _ in range(3)]
    t = cp.Variable()
    filtered = sum(cp.multiply(p.answers[:,j,None] != p.answers[None,:,j], X[j]) for j in range(3))
    constraints = [B >> 0 for B in X]
    constraints += [filtered[h.separate] == 1, sum(cp.diag(B) for B in X) <= t]
    prob = cp.Problem(cp.Minimize(t),constraints)
    prob.solve(solver='CLARABEL')
    assert h.solve(h.max_level)['cost'] == pytest.approx(prob.value,abs=2e-6)


def test_constant_and_invalid_models():
    h = Hierarchy(symmetric_evaluation(2,[0,0,0]))
    assert h.solve()['cost'] == pytest.approx(0,abs=1e-7)
    with pytest.raises(ValueError, match='equivariant'):
        Hierarchy(OracleProblem([[0,1],[1,1]],[([1,0],[0,1])]))
    with pytest.raises(ValueError, match='indistinguishable'):
        Hierarchy(symmetric_junta(3,1,[0,0]))
    with pytest.raises(ValueError, match='max_order'):
        Hierarchy(symmetric_evaluation(3,[0,1,1,1]),max_order=2)
    with pytest.raises(ValueError):
        h.solve(-1)


def test_previous_unbiased_intersection_free_witness_embeds_in_level_zero():
    from junta_learning.majority_sdp import solve
    h = Hierarchy(symmetric_junta(5,3,[0,0,1,1]))
    cost,Z,_ = solve(3)
    concepts = [x for x in range(32) if x.bit_count() == 3]
    # Use the builder's combination order, not integer order.
    import itertools
    concepts = [sum(1 << i for i in A) for A in itertools.combinations(range(5),3)]
    pi = np.array([math.comb(3,a)/8 for a in range(4)])
    matrices = []
    for query in range(32):
        a = np.array([(A & query).bit_count() for A in concepts])
        matrices.append(Z[np.ix_(a,a)] / np.sqrt(np.outer(pi[a],pi[a])) / 32)
    for orbit in h.orbits:
        X = matrices[orbit['representative']]
        P = orbit['sectors'][0].projector
        np.testing.assert_allclose(X,P@X@P,atol=1e-8)
    assert h.solve(0)['cost'] <= cost + 1e-6


def test_custom_order_keeps_trivial_first_and_final_value():
    p = symmetric_junta(4,2,[0,1,1])
    h = Hierarchy(p,sector_order=lambda s: (-s.degree,s.label))
    for orbit in h.orbits:
        s = orbit['sectors'][0]
        assert s.degree == 1
        for g in orbit['stabilizer']:
            np.testing.assert_allclose(np.eye(h.n)[:,g] @ s.basis,s.basis,atol=1e-9)
    assert h.solve(h.max_level)['cost'] == pytest.approx(2,abs=2e-6)


def test_complex_stabilizer_blocks_in_a_symmetric_group_action():
    # S3 acts regularly on concepts and through sign on two queries.
    # A query stabilizer is C3, with genuinely complex one-dimensional types.
    group = closure([(1,0,2),(0,2,1)],3)
    indices = {g:i for i,g in enumerate(group)}
    parity = [sum(g[i] > g[j] for i in range(3) for j in range(i+1,3)) % 2 for g in group]
    generators = []
    for g in [(1,0,2),(0,2,1)]:
        generators.append(([indices[compose(g,h)] for h in group],[1,0]))
    problem = OracleProblem(np.array([[p,1-p] for p in parity]),generators,parity)
    h = Hierarchy(problem)
    assert h.max_level == 2
    assert any(not s.real for o in h.orbits for s in o['sectors'])
    for r in range(3):
        result = h.solve(r)
        assert_feasible(result)
        assert result['cost'] == pytest.approx(1,abs=1e-6)
