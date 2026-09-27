"""Rebuild the paired 2x2 ablation from archived per-request measurements."""
import argparse,json,hashlib,csv,math,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from april.config import require_frozen_config

def regenerate():
    metadata=json.loads((ROOT/'measurements/table4_provenance.json').read_text())
    file=ROOT/'measurements'/metadata['exported_file']
    if hashlib.sha256(file.read_bytes()).hexdigest()!=metadata['exported_sha256']:raise ValueError('Measurement hash mismatch')
    plan={(r['task'],r['prompt_id']):r for r in json.loads((ROOT/'data/paper_request_plan.json').read_text())}
    groups=defaultdict(list);seen=set();populations=defaultdict(set)
    for line in file.read_text().splitlines():
        r=json.loads(line);key=(r['task'],r['prompt_id']);m=r['method']
        if key not in plan or r['prompt_sha256']!=plan[key]['prompt_sha256'] or r['source_index']!=plan[key]['source_index']:raise ValueError('Identity mismatch')
        if (*key,m) in seen or m not in metadata['expected_methods']:raise ValueError('Duplicate/unknown method')
        if r['completion_status']!='complete' or r['profile_status'] not in ['not_requested','applied','already_applied']:raise ValueError('Ineligible measurement')
        if not 0<=r['committed_tokens']<=256 or not math.isfinite(r['generation_seconds']) or r['generation_seconds']<=0:raise ValueError('Invalid measurement')
        seen.add((*key,m));populations[m].add(key)
        groups[(r['task'],m)].append(r);groups[('OVERALL',m)].append(r)
    if len(seen)!=metadata['rows'] or len(seen)!=1792:raise ValueError('Incomplete population')
    if any(populations[m]!=populations[metadata['expected_methods'][0]] for m in populations):raise ValueError('Unpaired populations')
    tasks={k[0] for k in plan}
    if any(len(groups[(t,m)])!=64 for t in tasks for m in metadata['expected_methods']):raise ValueError('Incomplete workload')
    result=[]
    for (t,m),rows in sorted(groups.items()):
        n=sum(r['committed_tokens'] for r in rows);elapsed=math.fsum(r['generation_seconds'] for r in rows)
        result.append(dict(task=t,method=m,requests=len(rows),tokens=n,seconds=elapsed,tps=n/elapsed))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require_frozen_config(a.config)
    rows=regenerate();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'table4.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (a.output/'table4.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows:
        if r['task']=='OVERALL':print(r['method'],f"{r['tps']:.8f}")

if __name__=='__main__':main()
