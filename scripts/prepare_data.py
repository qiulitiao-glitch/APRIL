#!/usr/bin/env python3
"""Regenerate prompt/reference content from user-obtained public raw files.

Every request is checked against the actual paper input hash before any final
output is written. Does not download or redistribute datasets. Unknown revisions
fail visibly instead of selecting replacement examples.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def jsonl(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def raw_rows(raw, stem):
    file=raw/(stem+'.jsonl')
    if file.exists():return jsonl(file)
    import pyarrow.parquet as pq
    file=raw/(stem+'.parquet')
    if not file.exists():raise FileNotFoundError(f'Provide {stem}.jsonl or {stem}.parquet under --raw-root')
    return pq.read_table(file).to_pylist()


def core(raw,task):
    if task=='cnn_dm_summarization':
        return {f"cnn_dailymail_3.0.0-test-{r['id']}":('Article:\n'+r['article']+'\n\nSummarize the article in a concise paragraph.',r['highlights']) for r in raw_rows(raw,'cnn_dm_test')}
    if task=='xsum_summarization':
        return {f"xsum-test-{r['id']}":('Document:\n'+r['document']+'\n\nWrite a one-sentence summary of the document.',r['summary']) for r in raw_rows(raw,'xsum_test')}
    if task=='humaneval':
        return {f"humaneval-test-{r['task_id']}":(r['prompt'],r['canonical_solution']) for r in raw_rows(raw,'humaneval_test')}
    if task=='gsm8k':
        source=raw_rows(raw,'gsm8k_test')
        if len(source)!=1319:raise ValueError('Historical GSM8K test population has 1319 rows')
        # Recovered original select_records protocol: sorted random.Random(42)
        # sample of 1000, then IDs enumerated AFTER selection, not raw indices.
        selected=[source[i] for i in sorted(random.Random(42).sample(range(len(source)),1000))]
        return {f"gsm8k-test-{r.get('id',i)}":('Question:\n'+r['question']+"\n\nLet's think step by step.",r['answer']) for i,r in enumerate(selected)}
    if task=='aqua_rat':
        result={}
        for i,r in enumerate(jsonl(raw/'aqua_test.jsonl')):
            options=r.get('options',r.get('choices',''))
            if isinstance(options,list):options='\n'.join(str(x) for x in options)
            elif isinstance(options,dict):options='\n'.join(f'{k}) {v}' for k,v in options.items())
            prompt='Question:\n'+r['question']+'\n\nOptions:\n'+options+"\n\nLet's think step by step and select the correct option."
            result[f'aqua-rat-test-{i}']=(prompt,(r['rationale']+'\nAnswer: '+r['correct']).strip())
        return result
    if task=='wmt14_de_en':
        src=(raw/'wmt14_de.txt').read_text(encoding='utf-8').splitlines()
        ref=(raw/'wmt14_en.txt').read_text(encoding='utf-8').splitlines()
        if len(src)!=len(ref):raise ValueError('WMT source/reference line counts differ')
        return {f'wmt14-de-en-test-{i}':(f'Translate the following German sentence into English:\n\nGerman: {s}\nEnglish:',t) for i,(s,t) in enumerate(zip(src,ref))}
    raise ValueError(task)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',default='all',help='Comma-separated task keys; all for the complete paper data')
    a=p.parse_args()
    from april.config import require_frozen_config
    require_frozen_config(a.config)
    specs=json.loads((ROOT/'data/dataset_identity.json').read_text())
    selected={s['task'] for s in specs} if a.tasks=='all' else set(a.tasks.split(','))
    if not selected <= {s['task'] for s in specs}:raise ValueError('Unknown task')
    prepared=[];checks=[];cnn=None
    for spec in specs:
        task=spec['task']
        if task not in selected:continue
        if task=='cnn_dm_lm':
            cnn=cnn or core(a.raw_root,'cnn_dm_summarization')
            candidates={}
            cnn_spec=next(s for s in specs if s['task']=='cnn_dm_summarization')
            for i,e in enumerate(cnn_spec['requests']):
                text=cnn[e['prompt_id']][0]
                article=text[len('Article:\n'):].split('\n\nSummarize',1)[0]
                words=article.split()
                if len(words)<80:continue
                n=min(160,max(64,len(words)//3))
                if not words[n:n+96]:continue
                candidates[f'cnn-dm-lm-{i}']=('Continue the following news article:\n\n'+' '.join(words[:n]),' '.join(words[n:n+96]))
        else:
            candidates=core(a.raw_root,task)
            if task=='cnn_dm_summarization':cnn=candidates
        for expected in spec['requests']:
            pair=candidates.get(expected['prompt_id'])
            if pair is None:raise ValueError(f"Missing source ID: {task}/{expected['source_index']}")
            prompt,response=pair
            if digest(prompt)!=expected['prompt_sha256'] or digest(response)!=expected['response_sha256']:
                raise ValueError(f"Historical prompt/reference hash mismatch: {task}/{expected['source_index']}; no replacement allowed")
            prepared.append(dict(task=task,source_index=expected['source_index'],prompt_id=expected['prompt_id'],prompt=prompt,response=response))
        checks.append(dict(task=task,requests=spec['count'],content_hashes='PASS'))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    (a.output/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared),encoding='utf-8')
    result=dict(status='PASS',requests=len(prepared),complete_5416=len(prepared)==5416,tasks=checks,
                scope='Prompt/reference content and order; historical JSON serialization and private metadata are not reproduced')
    (a.output/'preparation_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
