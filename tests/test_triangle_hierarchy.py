import math
import pytest
from sdp_hierarchy import Hierarchy
from sdp_hierarchy.examples import triangle_finding


def test_three_vertex_triangle_is_three_edge_and():
    p=triangle_finding(3)
    assert list(p.targets)==[0]*7+[1]
    h=Hierarchy(p)
    for r in range(h.max_level+1):
        s=h.solve(r)
        assert s['numerically_feasible']
        assert s['cost']==pytest.approx(math.sqrt(3),abs=1e-6)


def test_four_vertex_triangle_prefixes_and_dense_endpoint():
    p=triangle_finding(4)
    assert sum(p.targets)==23
    h=Hierarchy(p)
    assert h.max_level==3
    for r in [0,1]:
        assert h.solve(r)['status']=='infeasible'
    assert h.solve(intersection_free=True)['status']=='infeasible'
    for options in [dict(level=2),dict(level=3),dict(dense=True)]:
        s=h.solve(**options)
        assert s['numerically_feasible']
        assert s['cost']==pytest.approx(3,abs=2e-6)
