"""Resource-checked full-domain triangle level-2 sweep; no promise-domain substitution."""
import argparse
import ast
import json
import math
from pathlib import Path
from sdp_hierarchy import Hierarchy, triangle_finding

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'julia/triangle/plot_adversary_evolution.py'


def stored_bounds():
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SERIES' for t in node.targets):
            return dict(ast.literal_eval(node.value)['best MOSEK known'])
    raise ValueError('Stored benchmark series missing')


def resources(n):
    edges=math.comb(n,2)
    inputs=2**edges
    stabilizer_order=2*math.factorial(n-2)
    minimum_trivial_multiplicity=(inputs+stabilizer_order-1)//stabilizer_order
    return dict(vertices=n,edges=edges,inputs=inputs,
                dense_matrix_bytes=8*inputs**2,
                trivial_lift_bytes_lower_bound=8*inputs**2*minimum_trivial_multiplicity**2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-n',type=int,default=10)
    p.add_argument('--budget-gib',type=float,default=0.5)
    p.add_argument('--output',type=Path,default=ROOT/'results/hierarchy/triangle_level2_scaling.json')
    args=p.parse_args()
    if args.max_n<3 or args.budget_gib<=0:p.error('Require max-n >= 3 and positive memory budget')
    bounds=stored_bounds();rows=[]
    for n in range(3,args.max_n+1):
        row=resources(n)
        row.update(stored_adversary_lower_bound=bounds.get(n),
                   stored_bound_source=str(SOURCE.relative_to(ROOT)) if n in bounds else None,
                   query_upper_bound=row['edges'],upper_bound_method='Read every edge; exact deterministic algorithm')
        if row['trivial_lift_bytes_lower_bound']>args.budget_gib*2**30:
            row.update(status='not_run_resource_limit',level2_cost=None,
                       reason='Even the real trivial-sector dense lift exceeds the memory budget; higher blocks and solver memory are additional.')
        else:
            h=Hierarchy(triangle_finding(n))
            level=min(2,h.max_level)
            result=h.solve(level);result.pop('matrices',None)
            row.update(status='solved',requested_level=2,effective_level=level,
                       saturated=level<2,level2_cost=result['cost'],solution=result)
            row['gap_to_stored_lower_bound']=result['cost']-bounds[n] if n in bounds else None
        rows.append(row)
        print(n,row['status'],row['level2_cost'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
