"""Recompute the printed table from hash-pinned archived numeric measurements.

Archival reconstruction is separate from running or qualifying a new runtime.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def regenerate():
    frozen = json.loads((ROOT/'configs/paper/PAPER_FROZEN_CONFIG.json').read_text(encoding='utf-8'))
    workloads = frozen['sections']['dataset_protocol']['workloads']
    cap = frozen['sections']['april_main']['generation']['max_new_tokens']
    methods = ['AR', 'LayerSkip', 'DEL', 'PLD', 'APRIL']
    expected = {(t,m):v['requests'] for t,v in workloads.items() for m in methods}
    identities = json.loads((ROOT/'data/paper_request_plan.json').read_text(encoding='utf-8'))
    ids = {(r['task'], r['prompt_id']):r for r in identities}
    groups = defaultdict(list)
    seen = set()
    for artifact in json.loads((ROOT/'measurements/provenance.json').read_text(encoding='utf-8')):
        path = ROOT/'measurements'/artifact['exported_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['exported_sha256']:
            raise ValueError(f'Measurement digest changed: {path.name}')
        count = 0
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                r = json.loads(line)
                pair = (r['task'],r['prompt_id'])
                key = (*pair, r['method'])
                if pair not in ids or key in seen or r['method'] not in methods:
                    raise ValueError('Unknown or duplicate request/method')
                if r.get('prompt_sha256') != ids[pair]['prompt_sha256']:
                    raise ValueError('Prompt identity mismatch')
                if r['source_index'] != ids[pair]['source_index']:
                    raise ValueError('Source index mismatch')
                n,t = r['committed_tokens'], r['generation_seconds']
                if not isinstance(n,int) or not 0 < n <= cap or not math.isfinite(t) or t <= 0:
                    raise ValueError('Invalid measured token count/time')
                if r.get('profile_status') not in ['not_requested','applied','already_applied']:
                    raise ValueError('Ineligible/missing hardware profile')
                seen.add(key)
                groups[r['task'],r['method']].append(r)
                groups['OVERALL',r['method']].append(r)
                count += 1
        if count != artifact['rows']:
            raise ValueError('Measurement row count changed')
    for key,count in expected.items():
        if len(groups[key]) != count:
            raise ValueError(f'Incomplete population: {key}')
    rows = []
    for task in list(workloads)+['OVERALL']:
        for method in methods:
            rr = sorted(groups[task,method],key=lambda r:(r['task'],r['source_index']))
            n = sum(r['committed_tokens'] for r in rr)
            t = math.fsum(r['generation_seconds'] for r in rr)
            rows.append(dict(task=task,method=method,requests=len(rr),committed_tokens=n,
                generation_seconds=t,tps=n/t,
                historical_validity='ARCHIVED_MEASUREMENT',
                timing_ci='NOT_RECONSTRUCTED'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from april.config import require_frozen_config
    require_frozen_config(a.config)
    rows=regenerate()
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'table2.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'status':'ARCHIVAL_TABLE_REGENERATED',
             'historical_pld_ar_mismatches':3,
             'pld_interpretation':'Retained paper measurement; three AR token mismatches do not invalidate a non-lossless performance claim.',
             'scope':'Numeric reconstruction; not baseline reruns or runtime qualification',
             'rows':rows}
    (a.output/'table2.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for r in rows:
        if r['task']=='OVERALL':print(f"{r['method']}: {r['tps']:.8f} TPS")
    print(summary['status'])


if __name__=='__main__':main()
