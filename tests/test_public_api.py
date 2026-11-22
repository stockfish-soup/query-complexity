"""Compatibility of the published import surfaces after the src migration."""
import query_complexity as qc
from query_complexity import core
from junta_learning import majority_sdp, rank_one
from sdp_hierarchy import Hierarchy, OracleProblem, triangle_finding


def test_existing_transducer_imports_remain_available():
    assert qc.__version__ == '0.2.0'
    assert qc.AdversarySolution is core.AdversarySolution
    solution = qc.analytic_or_solution(2)
    transducer = qc.build_adversary_transducer(solution)
    assert transducer.private_dim == 2
    assert transducer.map_error < 1e-8


def test_additional_packages_are_public():
    assert callable(majority_sdp.solve)
    assert callable(rank_one.count_witness)
    assert isinstance(triangle_finding(3), OracleProblem)
    assert Hierarchy(triangle_finding(3)).max_level == 1
