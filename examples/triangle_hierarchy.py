"""Evaluate triangle finding on a small full graph domain."""
import argparse
from sdp_hierarchy import Hierarchy, triangle_finding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vertices', type=int, choices=[3, 4], default=4)
    args = parser.parse_args()
    hierarchy = Hierarchy(triangle_finding(args.vertices))
    result = hierarchy.solve(min(2, hierarchy.max_level))
    print(f"Vertices: {args.vertices}; inputs: {hierarchy.n}; queries: {hierarchy.q}")
    print(f"Level: {result['level']}; cost: {result['cost']:.8f}; status: {result['status']}")
    print(f"Numerically feasible: {result['numerically_feasible']}")
    print("This enumerative backend is intended for small graphs.")


if __name__ == "__main__":
    main()
