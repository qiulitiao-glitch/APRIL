from april.controller import create_controller
from copy import deepcopy


def test_frozen_bootstrap_and_persistent_episode():
    controller=create_controller()
    controller.reset_request_episode()
    decision=controller.select(100)
    assert (decision.draft_layer,decision.speculation_length)==(7,4)
    controller.update_after_step(step_id=0,context_length=100,decision=decision,
        num_drafted_tokens=4,num_accepted_by_final=3,proxy_accepted=4,
        anchor_accepted_by_final=3,latency_ms=None,output_tokens=4)
    previous=deepcopy(controller.pair_stats)
    controller.reset_request_episode()
    assert controller.pair_stats==previous
    assert not controller.context_pair_stats
