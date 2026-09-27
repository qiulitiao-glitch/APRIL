from types import SimpleNamespace
import torch
import pytest
from april.generation import AprilGenerator
from april.config import GenerationConfig
from april import model_adapter

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1),requires_grad=False)

def test_shallow_stops_after_appending_low_confidence(monkeypatch):
    calls=[]
    def forward(*args):
        calls.append(args);return SimpleNamespace(logits=torch.tensor([[[0.,0.,0.,0.1]]]),past_key_values=None,exit_query_cache=None)
    monkeypatch.setattr(model_adapter,'shallow_forward',forward)
    g=AprilGenerator(Model());g.controller.reset_request_episode()
    assert g._draft(torch.tensor([[1]]),None,7,4,set()).tolist()==[[3]]
    assert len(calls)==1

def test_budget_and_eos_stop(monkeypatch):
    calls=[]
    def forward(*args):
        calls.append(args);return SimpleNamespace(logits=torch.tensor([[[0.,0.,9.,0.]]]),past_key_values=None,exit_query_cache=None)
    monkeypatch.setattr(model_adapter,'shallow_forward',forward)
    g=AprilGenerator(Model());g.controller.reset_request_episode()
    assert g._draft(torch.tensor([[1]]),None,7,3,set()).tolist()==[[2,2,2]]
    assert g._draft(torch.tensor([[1]]),None,7,3,{2}).tolist()==[[2]]

def test_feedback_conditions_are_additive(monkeypatch):
    # p(top1)=.562: passes base .55; fails bad+.05 + cooldown+.025 - good*.025 = .60.
    calls=[]
    def forward(*args):
        calls.append(args);return SimpleNamespace(logits=torch.tensor([[[0.,0.25]]]),past_key_values=None,exit_query_cache=None)
    monkeypatch.setattr(model_adapter,'shallow_forward',forward)
    g=AprilGenerator(Model());g.controller.reset_request_episode()
    assert g._draft(torch.tensor([[0]]),None,7,3,set()).shape[1]==3
    g.controller.feedback_bad_streak=3;g.controller.feedback_good_streak=2;g.controller.feedback_cooldown_remaining=1
    assert g._draft(torch.tensor([[0]]),None,7,3,set()).shape[1]==1
