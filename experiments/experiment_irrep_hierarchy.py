"""Reproduce symmetric evaluation and hidden-support hierarchy benchmarks."""
import argparse
import json
from pathlib import Path
from sdp_hierarchy import Hierarchy, symmetric_evaluation, symmetric_junta


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/hierarchy/symmetric_functions.json'))
    args = parser.parse_args()
    examples = [
        ('OR3', symmetric_evaluation(3,[0,1,1,1])),
        ('MAJ3', symmetric_evaluation(3,[0,0,1,1])),
        ('EXACT1_3', symmetric_evaluation(3,[0,1,0,0])),
        ('PARITY3', symmetric_evaluation(3,[0,1,0,1])),
        ('OR4', symmetric_evaluation(4,[0,1,1,1,1])),
        ('OR_junta_n4_k2', symmetric_junta(4,2,[0,1,1])),
        ('MAJ_junta_n5_k3', symmetric_junta(5,3,[0,0,1,1])),
    ]
    records = []
    for name, problem in examples:
        hierarchy = Hierarchy(problem)
        levels = []
        for r in range(hierarchy.max_level+1):
            result = hierarchy.solve(r)
            result.pop('matrices',None)
            levels.append(result)
        reference = hierarchy.solve(intersection_free=True)
        dense = hierarchy.solve(dense=True)
        for result in [reference,dense]:
            result.pop('matrices',None)
        record = dict(name=name, group_order=len(hierarchy.group), blocks=hierarchy.block_summary(),
                      levels=levels, intersection_free=reference, dense=dense)
        records.append(record)
        print(name, [round(s['cost'],8) for s in levels], 'dense', round(dense['cost'],8), flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    run()
