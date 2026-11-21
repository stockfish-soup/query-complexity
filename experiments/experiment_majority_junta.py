"""Compare the count SDP optimum with the explicit period-four witness."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from junta_learning import majority_sdp, rank_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5, 7, 15, 31, 63])
    parser.add_argument('--optimize-max-k', type=int, default=15)
    parser.add_argument('--output', type=Path, default=Path('results/junta/majority_comparison.csv'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['k', 'rank_one_cost', 'cost_over_log_k_plus_1', 'rank_one_residual',
              'sdp_cost', 'sdp_status', 'sdp_residual', 'sdp_min_eigenvalue']
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for k in args.ks:
            z = rank_one.count_witness(k)
            cut = (k + 1) // 2
            cross = np.outer(z[:cut], z[cut:])
            kernels = [majority_sdp.joint_count_kernel(k, d)[:cut, cut:] for d in range(1, k+1)]
            cost = rank_one.exact_cost(k)
            row = dict(k=k, rank_one_cost=cost, cost_over_log_k_plus_1=cost/math.log(k+1),
                       rank_one_residual=max(abs(2*np.sum(T*cross)-1) for T in kernels))
            if k <= args.optimize_max_k:
                value, Z, status = majority_sdp.solve(k)
                row.update(sdp_cost=value, sdp_status=status,
                           sdp_residual=max(abs(2*np.sum(T*Z[:cut,cut:])-1) for T in kernels),
                           sdp_min_eigenvalue=np.linalg.eigvalsh(Z).min())
            writer.writerow(row)
            print(row)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
