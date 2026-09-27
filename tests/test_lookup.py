from collections import deque
import torch
import pytest
from april.lookup import RollingNGramLookup
from april.controller import create_controller

def test_repeated_context_lookup_and_budget():
    prompt=[1,2,3,4,5,6,7,8]*6+[1,2,3,4]
    lookup=RollingNGramLookup(prompt,32,torch.device('cpu'))
    decision=lookup.lookup(8)
    assert decision.hit and decision.k==8
    assert lookup.stage_proposal().tolist()==[[5,6,7,8,1,2,3,4]]
    lookup.record_acceptance(3)
    assert lookup.summary()['accepted_retrieval_tokens']==3
    assert not lookup.lookup(1).hit

def test_lookup_threshold_and_fixed_capacity():
    assert not RollingNGramLookup._qualified(8,4,1,1.0)
    assert RollingNGramLookup._qualified(8,4,2,.75)
    assert not RollingNGramLookup._qualified(8,4,2,.749)
    lookup=RollingNGramLookup([1,2,3],1,torch.device('cpu'))
    lookup.observe_committed([4])
    with pytest.raises(RuntimeError,match='capacity'):lookup.observe_committed([5])

def test_recent_window_is_token_weighted_and_resets():
    c=create_controller();c.confidence_z=0
    c.recent_layer_stats[7]=deque([(1,1),(0,9)],maxlen=8)
    # Remove conservative adjustment independently of the chosen parameter name.
    c._rate_lcb=lambda stat,default:stat.ema
    assert c._recent_rate_lcb(7)==.1
    c.recent_layer_stats[7].extend([(2,2)]*8)
    assert c._recent_rate_lcb(7)==1
    c.reset_request_episode();assert not c.recent_layer_stats


def test_hash_collisions_do_not_consume_eligible_match_budget():
    prompt=[1,2,9,9]*70+[1,2]
    lookup=RollingNGramLookup(prompt,8,torch.device('cpu'),min_ngram=2,max_ngram=2,max_candidates=3)
    # Force unrelated [9,9] windows into the suffix hash bucket. Exact comparison
    # must reject them without spending the three eligible-match slots.
    bucket=lookup._positions[2][lookup._suffix_hash[2]]
    bucket.extend([2]*100)
    decision=lookup.lookup(2)
    assert decision.hit and decision.eligible==3 and decision.support==3
    assert lookup.stage_proposal().tolist()==[[9,9]]
