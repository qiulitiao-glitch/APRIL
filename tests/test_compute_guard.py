from types import SimpleNamespace
import os
import pytest
from april.compute_guard import ComputeGuard


def test_guard_failure_is_sticky(monkeypatch,tmp_path):
    replies=iter([f'{os.getpid()}, python\n999999, external\n',f'{os.getpid()}, python\n'])
    monkeypatch.setattr('april.compute_guard.subprocess.run',lambda *a,**k:SimpleNamespace(stdout=next(replies)))
    guard=ComputeGuard(tmp_path,'test-gpu')
    with pytest.raises(RuntimeError,match='contamination'):guard.boundary()
    audit=guard.close()
    assert audit['valid'] is False
    assert audit['failed_observations']==1
    assert (tmp_path/'compute_process_samples.jsonl').is_file()


def test_failed_gpu_query_is_not_clean(monkeypatch,tmp_path):
    def fail(*a,**k):raise TimeoutError('unavailable')
    monkeypatch.setattr('april.compute_guard.subprocess.run',fail)
    guard=ComputeGuard(tmp_path,'test-gpu')
    audit=guard.close()
    assert audit['valid'] is False
    assert audit['failed_observations']==1


def test_wrong_device_cannot_pass_as_idle(monkeypatch,tmp_path):
    monkeypatch.setattr('april.compute_guard.subprocess.run',lambda *a,**k:SimpleNamespace(stdout=''))
    guard=ComputeGuard(tmp_path,'expected-uuid')
    guard.boundary()  # Before allocation an idle device is expected.
    with pytest.raises(RuntimeError,match='contamination'):
        guard.verify_allocated_device()
    assert guard.close()['valid'] is False
