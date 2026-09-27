"""Explicit local generation entry point; never executes generated code."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(method='APRIL'):
    from .config import PAPER, paper_snapshot
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--layerskip', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=PAPER['generation']['max_new_tokens'])
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    from .config import require_frozen_config
    require_frozen_config(args.config)
    import fcntl
    identity = subprocess.run(['nvidia-smi','-i',str(args.gpu),
        '--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=30).stdout.strip()
    gpu = next(csv.reader([identity],skipinitialspace=True))
    apps = subprocess.run(['nvidia-smi','-i',gpu[1],'--query-compute-apps=pid',
                           '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=30)
    if apps.stdout.strip():
        raise RuntimeError('Selected GPU has an external compute process')
    # Driver memory/utilization counters may briefly lag a just-finished run.
    # Wait only while no compute process is present; never share an active GPU.
    for attempt in range(11):
        if float(gpu[3]) <= 100 and float(gpu[4]) == 0:
            break
        if attempt == 10:
            raise RuntimeError('Selected GPU has not become idle')
        time.sleep(1)
        identity=subprocess.run(['nvidia-smi','-i',gpu[1],
            '--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
            check=True,capture_output=True,text=True,timeout=30).stdout.strip()
        gpu=next(csv.reader([identity],skipinitialspace=True))
        apps=subprocess.run(['nvidia-smi','-i',gpu[1],'--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=30)
        if apps.stdout.strip():
            raise RuntimeError('Selected GPU has an external compute process')
    lock_dir=Path(os.environ.get('APRIL_LOCK_ROOT','/tmp/april-public-gpu-locks'))
    lock_dir.mkdir(parents=True,exist_ok=True)
    lock=(lock_dir/(gpu[1]+'.lock')).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['CUDA_VISIBLE_DEVICES']=gpu[1]
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_DATASETS_OFFLINE']='1'
    from .model_adapter import load_model, configure_torch
    configure_torch()
    from .generation import AprilGenerator
    from .config import GenerationConfig
    from .hardware import HardwareProfileRecorder
    from .compute_guard import ComputeGuard
    import torch
    inputs=[json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.limit is not None:
        inputs=inputs[:args.limit]
    if not inputs or len({r['prompt_id'] for r in inputs})!=len(inputs):
        raise ValueError('Requests must be nonempty with unique IDs')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(method=method,requests=len(inputs),model=str(args.model),gpu=gpu,
                  frozen_config_status=paper_snapshot()['status'],paper_reproduction_qualified=False,
                  max_new_tokens=args.max_new_tokens,torch=torch.__version__,
                  timing='CUDA-synchronized generation; excludes load, warmup, tokenization, detokenization and record I/O',
                  controller='fresh per process, hybrid request reset, persistent per-shard layer feedback',
                  pid=os.getpid(),requested_profile='not_requested')
    manifest['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    from .config import SNAPSHOT_PATH
    manifest['frozen_config_sha256']=hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest()
    manifest['runtime_source_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).resolve().parent.glob('*.py')}
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    recorder=HardwareProfileRecorder(args.output,gpu[1],sample_interval_s=1.0)
    guard=ComputeGuard(args.output/'hardware',gpu[1])
    rows=[]
    error=None
    try:
        guard.start()
        with recorder:
            model,tokenizer=load_model(args.model,args.layerskip)
            guard.verify_allocated_device()
            model.generate(**tokenizer('This is a warmup prompt',return_tensors='pt').to(model.device),
                           max_new_tokens=10,do_sample=False)
            torch.cuda.synchronize()
            generator=AprilGenerator(model,GenerationConfig(max_new_tokens=args.max_new_tokens),method=method)
            with (args.output/'results.jsonl').open('x',encoding='utf-8') as output:
                for index,request in enumerate(inputs):
                    guard.boundary()
                    torch.manual_seed(PAPER['generation']['seed']+int(request.get('source_index',index)))
                    ids=tokenizer.encode(request['prompt'],add_special_tokens=True)
                    result=generator.generate(ids,[tokenizer.eos_token_id])
                    guard.boundary()
                    result.update(prompt_id=request['prompt_id'],task=request.get('task','custom'),method=method)
                    result['source_index']=request.get('source_index',index)
                    result['prompt_sha256']=hashlib.sha256(request['prompt'].encode('utf-8')).hexdigest()
                    result['output_token_sha256']=hashlib.sha256(json.dumps(result['token_ids'],separators=(',',':')).encode()).hexdigest()
                    result.update(recorder.metadata())
                    output.write(json.dumps(result)+'\n');output.flush()
                    rows.append(result)
                    print(json.dumps(dict(completed=len(rows),total=len(inputs),prompt_id=request['prompt_id'],
                                          tokens=result['num_generated_tokens'],seconds=result['elapsed_time_s'])),flush=True)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        raise
    finally:
        process_audit=guard.close()
        duration=sum(row['elapsed_time_s'] for row in rows)
        tokens=sum(row['num_generated_tokens'] for row in rows)
        imported=[name for name in sys.modules if name=='DEL' or name.startswith('DEL.') or name.startswith('self_speculation')]
        valid=error is None and len(rows)==len(inputs) and process_audit['valid']
        summary=dict(status='COMPLETE' if valid else 'FAILED',error=error,
                     requests=len(rows),tokens=tokens,generation_seconds=duration,tps=tokens/duration if duration else None,
                     restricted_modules_imported=imported,hardware=recorder.summary,
                     lookup_hits=sum((r['lookup'] or {}).get('hit_count',0) for r in rows),
                     self_draft_rounds=sum(r['routes'].get('self_draft',0) for r in rows),
                     controller_updates=sum(r['controller_updated'] for r in rows))
        summary['process_audit']=process_audit
        for row in rows:
            row['compute_guard_clean']=process_audit['valid']
            for field in ['requested_profile','profile_status','actual_power_limit_w','energy_measurement_source','cap_saturated_ratio']:
                row[field]=recorder.summary.get(field)
            row['cap_saturated_ratio_scope']='run-level hardware samples including setup/warmup'
        if (args.output/'results.jsonl').exists():
            temporary=args.output/'results.finalizing.jsonl'
            temporary.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            os.replace(temporary,args.output/'results.jsonl')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        if error is None and not valid:
            raise RuntimeError('Incomplete or hardware-invalid run; see summary.json')
