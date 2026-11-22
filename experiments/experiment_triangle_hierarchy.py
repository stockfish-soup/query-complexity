"""Small full-domain triangle-finding irrep hierarchy benchmarks."""
import argparse
import json
import math
from pathlib import Path
from sdp_hierarchy import Hierarchy
from sdp_hierarchy.examples import triangle_finding

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--vertices',nargs='+',type=int,default=[3,4])
p.add_argument('--output',type=Path,default=Path('results/hierarchy/triangle_finding.json'))
args=p.parse_args()
records=[]
for n in args.vertices:
    h=Hierarchy(triangle_finding(n))
    results=[]
    for r in range(h.max_level+1):
        s=h.solve(r);s.pop('matrices',None);results.append(s)
        print(n,r,s,flush=True)
    references={}
    for name,options in [('intersection_free',dict(intersection_free=True)),('dense',dict(dense=True))]:
        s=h.solve(**options);s.pop('matrices',None);references[name]=s
        print(n,name,s,flush=True)
    records.append(dict(vertices=n,inputs=h.n,queries=h.q,group_order=len(h.group),
                        blocks=h.block_summary(),levels=results,**references))
args.output.parent.mkdir(parents=True,exist_ok=True)
def finite_json(value):
    if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list): return [finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value
args.output.write_text(json.dumps(finite_json(records),indent=2,allow_nan=False)+'\n')
