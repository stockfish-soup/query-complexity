import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('triangle_scaling',Path(__file__).resolve().parents[1]/'experiments/experiment_triangle_level2_scaling.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resource_guard_before_full_graph_allocation():
    assert module.resources(4)['trivial_lift_bytes_lower_bound']<0.5*2**30
    assert module.resources(5)['trivial_lift_bytes_lower_bound']>60*10**9
    assert module.resources(10)['inputs']==35184372088832


def test_stored_bounds_exclude_heuristic_extrapolations():
    bounds=module.stored_bounds()
    assert bounds[4]==3
    assert bounds[8]==6.911788
    assert 9 not in bounds and 10 not in bounds
