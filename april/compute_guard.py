"""Observe external GPU compute processes without changing hardware or processes."""
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time


class ComputeGuard:
    def __init__(self, output, gpu_uuid, interval=20.0):
        self.output=Path(output)
        self.gpu_uuid=gpu_uuid
        self.interval=interval
        self.stop_event=threading.Event()
        self.lock=threading.Lock()
        self.rows=[]
        self.failures=[]
        self.thread=None
        self.require_own_process=False

    def sample(self):
        row={'unix_s':time.time(),'own_pid':os.getpid()}
        try:
            result=subprocess.run(['nvidia-smi','-i',self.gpu_uuid,
                '--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],
                check=True,capture_output=True,text=True,timeout=10)
            processes=[]
            for fields in csv.reader(result.stdout.splitlines(),skipinitialspace=True):
                if not fields:continue
                processes.append({'pid':int(fields[0]),'name':fields[1] if len(fields)>1 else ''})
            row['external_processes']=[p for p in processes if p['pid']!=os.getpid()]
            row['own_process_present']=any(p['pid']==os.getpid() for p in processes)
            row['valid']=not row['external_processes'] and (not self.require_own_process or row['own_process_present'])
        except Exception as exc:
            row.update(valid=False,error=type(exc).__name__+': '+str(exc))
        with self.lock:
            self.rows.append(row)
            if not row['valid']:self.failures.append(row)
        return row

    def verify_allocated_device(self):
        """After model allocation, prove this PID is on the monitored GPU."""
        self.require_own_process=True
        self.boundary()

    def boundary(self):
        self.sample()
        with self.lock:
            if self.failures:
                raise RuntimeError('GPU compute contamination or failed observation; run is invalid')

    def _poll(self):
        while not self.stop_event.wait(self.interval):self.sample()

    def start(self):
        self.boundary()
        self.thread=threading.Thread(target=self._poll,daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(timeout=15)
        self.sample()
        self.output.mkdir(parents=True,exist_ok=True)
        with self.lock:
            summary={'valid':not self.failures,'samples':len(self.rows),
                'failed_observations':len(self.failures),'poll_interval_seconds':self.interval,
                'limitation':'Sampled process observations and request boundaries, not continuous OS process tracing.'}
            (self.output/'compute_process_samples.jsonl').write_text(
                ''.join(json.dumps(r)+'\n' for r in self.rows),encoding='utf-8')
            (self.output/'compute_process_audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        return summary
