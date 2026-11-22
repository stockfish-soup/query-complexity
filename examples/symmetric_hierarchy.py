"""Compare irrep prefixes for evaluating three-bit majority."""
from sdp_hierarchy import Hierarchy, symmetric_evaluation


def main():
    hierarchy = Hierarchy(symmetric_evaluation(3, [0, 0, 1, 1]))
    for level in range(hierarchy.max_level + 1):
        result = hierarchy.solve(level)
        print(f"Level {level}: {result['cost']:.8f}; "
              f"validated={result['numerically_feasible']}")
    print(f"Dense reference: {hierarchy.solve(dense=True)['cost']:.8f}")


if __name__ == "__main__":
    main()
