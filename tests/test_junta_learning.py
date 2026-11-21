"""Check reduced optimizers and lift the explicit witness to original queries."""
import itertools
import math

import numpy as np
import pytest

from junta_learning import bounded_size, majority_sdp, multibias, rank_one, witness_audit


@pytest.mark.parametrize('k', [1, 3, 5, 7])
def test_majority_sdp_matches_nuclear_form_and_rank_one_bound(k):
    value, Z, status = majority_sdp.solve(k)
    nuclear_value, Q, nuclear_status = majority_sdp.solve_cross_block(k)
    assert status == nuclear_status == 'optimal'
    assert np.linalg.eigvalsh(Z).min() >= -1e-7
    assert value == pytest.approx(nuclear_value, abs=2e-6)
    assert value <= rank_one.exact_cost(k) + 2e-6
    z = rank_one.count_witness(k)
    assert z @ z == pytest.approx(rank_one.exact_cost(k), abs=1e-8)
    cut = (k + 1) // 2
    for d in range(1, k + 1):
        kernel = majority_sdp.joint_count_kernel(k, d)[:cut, cut:]
        assert 2 * np.sum(kernel * Z[:cut, cut:]) == pytest.approx(1, abs=1e-7)
        assert 2 * np.sum(kernel * Q) == pytest.approx(1, abs=1e-7)
        assert 2 * np.sum(kernel * np.outer(z[:cut], z[cut:])) == pytest.approx(1, abs=1e-8)


@pytest.mark.parametrize('k', [1, 3, 5])
def test_rank_one_lift_satisfies_original_learning_sdp(k):
    n = k + 2
    z = rank_one.count_witness(k)
    pi = np.array([math.comb(k, a) / 2**k for a in range(k + 1)])
    concepts = [sum(1 << i for i in A) for A in itertools.combinations(range(n), k)]
    separated_sum = np.zeros((len(concepts), len(concepts)))
    loads = np.zeros(len(concepts))
    for S in range(1 << n):
        counts = np.array([(A & S).bit_count() for A in concepts])
        outputs = counts >= (k + 1) // 2
        v = z[counts] / np.sqrt(pi[counts])
        X = np.outer(v, v) / 2**n
        loads += np.diag(X)
        separated_sum += X * (outputs[:, None] != outputs[None, :])
    assert np.max(np.abs(separated_sum - (1 - np.eye(len(concepts))))) < 1e-8
    np.testing.assert_allclose(loads, rank_one.exact_cost(k), atol=1e-8)


def test_mixed_size_sdp_couples_layers_and_reports_feasibility():
    result = bounded_size.solve(3)
    assert result.status == 'optimal'
    assert result.layers == [1, 2, 3]
    assert result.residual < 1e-7
    assert result.minimum_eigenvalue >= -1e-7
    assert max(result.layer_costs) <= result.cost + 1e-7
    assert result.cost >= majority_sdp.solve(3)[0] - 1e-6


def test_multibias_majority_matches_single_bias_count_sdp():
    k = 5
    profile = [int(a >= 3) for a in range(k + 1)]
    result = multibias.solve_general_level0_cvxpy(k, profile, [0.5], solver='CLARABEL')
    assert result['status'] == 'optimal'
    assert result['value'] == pytest.approx(majority_sdp.solve(k)[0], abs=2e-6)


def test_independent_dyadic_and_halfset_checks():
    assert max(witness_audit.construct_witness(7)['residuals'].values()) < 1e-10
    assert witness_audit.common_spectrum_residual(5) < 1e-10
    assert rank_one.exhaustive_best(7)[0] == pytest.approx(rank_one.exact_cost(7))


@pytest.mark.parametrize('k', [0, 2, -1])
def test_majority_optimizer_rejects_invalid_sizes(k):
    with pytest.raises(ValueError):
        majority_sdp.solve_cross_block(k)


def test_mixed_size_joint_counts_match_query_enumeration():
    # |A|=2, |B|=3 and |A intersect B|=1.
    A, B, n = 0b0011, 0b1101, 4
    empirical = np.zeros((3, 4))
    for S in range(1 << n):
        empirical[(A & S).bit_count(), (B & S).bit_count()] += 1 / 2**n
    np.testing.assert_allclose(bounded_size.joint_counts(2, 3, 1), empirical)
