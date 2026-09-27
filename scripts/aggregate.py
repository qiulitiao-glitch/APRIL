"""Aggregate complete generated request records using ratio-of-sums TPS."""
import argparse,json,math,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from april.config import require_frozen_config

def aggregate(records):
    groups=defaultdict(lambda:[0,0.0,0]);seen=set()
    for r in records:
        key=(r['task'],r['prompt_id'],r['method'])
        if key in seen:raise ValueError('Duplicate request/method')
        seen.add(key)
        n,t=r['num_generated_tokens'],r['elapsed_time_s']
        if not isinstance(n,int) or n<0 or not math.isfinite(t) or t<=0:raise ValueError('Invalid count/time')
        if not r.get('compute_guard_clean',False):raise ValueError('Hardware observation invalid')
        if r['profile_status'] not in ['not_requested','applied','already_applied']:raise ValueError('Hardware profile invalid')
        for group in [(r['task'],r['method']),('OVERALL',r['method'])]:
            groups[group][0]+=n;groups[group][1]+=t;groups[group][2]+=1
    return [dict(task=k[0],method=k[1],tokens=v[0],seconds=v[1],requests=v[2],tps=v[0]/v[1]) for k,v in sorted(groups.items())]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require_frozen_config(a.config);rows=[]
    for run in a.run:
        summary=json.loads((run/'summary.json').read_text())
        records=[json.loads(x) for x in (run/'results.jsonl').read_text().splitlines() if x.strip()]
        if summary['status']!='COMPLETE' or len(records)!=summary['requests']:raise ValueError('Incomplete run')
        rows.extend(records)
    with a.output.open('x') as f:json.dump(aggregate(rows),f,indent=2)

if __name__=='__main__':main()
