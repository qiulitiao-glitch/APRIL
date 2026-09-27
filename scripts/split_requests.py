"""Arrange prepared prompts into the two frozen main controller streams."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from april.config import require_frozen_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require_frozen_config(a.config)
    rows=[json.loads(x) for x in a.input.read_text(encoding='utf-8').splitlines() if x.strip()]
    by_id={(r['task'],r['prompt_id']):r for r in rows}
    plan=json.loads((ROOT/'data/paper_request_plan.json').read_text())
    if len(rows)!=5416 or len(by_id)!=5416:raise ValueError('Complete unique 5416 request population required')
    shards={}
    for expected in plan:
        r=by_id[(expected['task'],expected['prompt_id'])]
        if hashlib.sha256(r['prompt'].encode()).hexdigest()!=expected['prompt_sha256']:raise ValueError('Prompt identity changed')
        shard=expected['shard'];shards.setdefault(shard,[]).append(r)
    a.output.mkdir(parents=True,exist_ok=False)
    for shard,requests in shards.items():
        (a.output/(shard+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests),encoding='utf-8')
    print(json.dumps({s:len(v) for s,v in shards.items()}))

if __name__=='__main__':main()
