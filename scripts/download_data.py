"""Acquire separate public benchmark files; prepare_data verifies paper identity."""
import argparse,hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from april.config import require_frozen_config

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    require_frozen_config(a.config)
    specs=json.loads((ROOT/'data/download_sources.json').read_text())['sources']
    if a.dry_run:
        print(json.dumps({'parquet_bytes':sum(s['bytes'] for s in specs.values()),'sources':specs,'other':['AQuA test.json','sacreBLEU WMT14/full de-en']},indent=2));return
    a.output.mkdir(parents=True,exist_ok=False)
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    for spec in specs.values():
        path=Path(hf_hub_download(repo_id=spec['repo'],repo_type='dataset',revision=spec['revision'],filename=spec['file'],token=False,cache_dir=str(a.output/'cache')))
        if path.stat().st_size!=spec['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Downloaded file identity mismatch')
        with (a.output/(spec['stem']+'.jsonl')).open('x',encoding='utf-8') as f:
            for row in pq.read_table(path).to_pylist():f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with urllib.request.urlopen('https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json',timeout=90) as response:
        content=response.read()
    for line in content.decode('utf-8').splitlines():
        if line.strip():json.loads(line)
    (a.output/'aqua_test.jsonl').write_bytes(content)
    for kind,name in [('src','wmt14_de.txt'),('ref','wmt14_en.txt')]:
        result=subprocess.run([sys.executable,'-m','sacrebleu','-t','wmt14/full','-l','de-en','--echo',kind],capture_output=True,text=True,encoding='utf-8',check=True,timeout=600,
            env={**os.environ,'SACREBLEU':str(a.output/'sacrebleu_cache')})
        (a.output/name).write_text(result.stdout,encoding='utf-8')
    print('Files acquired; run prepare_data.py to verify every frozen hash.')

if __name__=='__main__':main()
