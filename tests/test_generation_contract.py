"""Token-boundary tests using scripted target logits; no GPU/model download."""
from types import SimpleNamespace
import torch
import pytest
from april import generation
from april.config import GenerationConfig


class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1),requires_grad=False)


def install_target(monkeypatch, token, *, margin=5.0):
    calls=[]
    def forward(model,input_ids,cache):
        calls.append(input_ids.clone())
        logits=torch.zeros((1,input_ids.shape[1],16))
        logits[:,:,token]=margin
        return SimpleNamespace(logits=logits,past_key_values=None)
    monkeypatch.setattr(generation.model_adapter,'forward',forward)
    return calls


def test_ar_immediate_eos_is_not_returned_or_counted(monkeypatch):
    calls=install_target(monkeypatch,2)
    result=generation.AprilGenerator(Target(),GenerationConfig(max_new_tokens=2),method='AR').generate([3], [2])
    assert result['token_ids']==[]
    assert result['num_generated_tokens']==0
    assert result['eos_reached'] is True
    assert len(calls)==1


def test_pld_immediate_eos_is_not_returned_or_counted(monkeypatch):
    calls=install_target(monkeypatch,2)
    result=generation.AprilGenerator(Target(),GenerationConfig(max_new_tokens=2),method='PLD').generate([3], [2])
    assert result['token_ids']==[]
    assert result['pld']['lookup_misses']==1
    assert result['eos_reached'] is True
    assert len(calls)==1


def test_pld_last_position_keeps_numerical_recovery(monkeypatch):
    install_target(monkeypatch,4,margin=0.01)
    recovery_calls=[]
    def recheck(*args):
        recovery_calls.append('recheck')
        logits=torch.zeros(1,1,16);logits[:,:,5]=4.0
        return logits,None,{'target_forward_calls':1}
    def checkpoint(*args):
        recovery_calls.append('checkpoint')
        return None,None,None,{'target_forward_calls':1}
    monkeypatch.setattr(generation,'canonical_fp16_q1_target_recheck',recheck)
    monkeypatch.setattr(generation,'advance_canonical_checkpoint',checkpoint)
    result=generation.AprilGenerator(Target(),GenerationConfig(max_new_tokens=1),method='PLD').generate([3], [2])
    assert result['token_ids']==[5]
    assert recovery_calls==['recheck','checkpoint']
    assert result['pld']['lookup_calls']==1
    assert result['forward_counts']=={'normal_target':1,'shallow':0,'recovery':2}


@pytest.mark.parametrize('cap,expected_budget',[(1,None),(2,1)])
def test_april_reserves_correction_position_and_handles_single_tail(monkeypatch,cap,expected_budget):
    calls=install_target(monkeypatch,4)
    budgets=[]
    def draft(self,current,cache,depth,budget,eos):
        budgets.append(budget)
        return current.new_tensor([[4]*budget])
    monkeypatch.setattr(generation.AprilGenerator,'_draft',draft)
    result=generation.AprilGenerator(Target(),GenerationConfig(max_new_tokens=cap)).generate([3],[2])
    assert result['token_ids']==[4]*cap
    assert budgets==([] if expected_budget is None else [expected_budget])
    assert len(calls)==1
