"""Construct an explicit logarithmic-cost majority-learning witness."""
from junta_learning import majority_sdp, rank_one


def main():
    k = 7
    z = rank_one.count_witness(k)
    cost, _, status = majority_sdp.solve(k)
    print(f"Explicit rank-one witness cost: {z @ z:.8f}")
    print(f"Unrestricted count-SDP cost: {cost:.8f} ({status})")
    print("The explicit witness is feasible; it need not minimize the SDP.")


if __name__ == "__main__":
    main()
