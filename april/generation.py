"""APRIL loop implemented from the paper's lookup-first decoding specification.

The cache contains consumed committed positions; the last correction/bonus is
the next input and is intentionally not yet cached. No historical runtime is
imported or used as an implementation dependency.
"""
from collections import Counter
import time
import torch
from .config import GenerationConfig, paper_snapshot
from .controller import create_controller
from .lookup import RollingNGramLookup
from . import model_adapter
from .exact_recheck import (
    clone_past_key_values, commit_critical_margin, committed_tokens,
    canonical_fp16_q1_target_recheck, advance_canonical_checkpoint,
)


class AprilGenerator:
    def __init__(self, model, config=None, *, method='APRIL'):
        if method not in {'APRIL', 'AR', 'LayerSkip', 'PLD'}:
            raise ValueError('Unsupported public method')
        self.model = model
        self.config = config or GenerationConfig()
        self.method = method
        self.controller = create_controller() if method == 'APRIL' else None

    def _draft(self, current, committed_cache, depth, budget, eos_ids):
        # Original API composition from the recovered scalar behavior contract.
        stopping = paper_snapshot()['sections']['april_main']['shallow_stopping']
        threshold = self.config.draft_confidence_threshold
        controller = self.controller
        if controller is not None:
            step = stopping['tr_nes_feedback_confidence_step']
            threshold += step * (controller.feedback_bad_streak > 0)
            threshold += (step / 2) * (controller.feedback_cooldown_remaining > 0)
            threshold -= (step / 2) * (controller.feedback_good_streak >= 2)
            threshold = min(stopping['tr_nes_feedback_confidence_max'],
                            max(stopping['tr_nes_feedback_confidence_min'], threshold))
        cache = committed_cache
        current = current.clone()
        hidden = None
        proposal = []
        for position in range(budget):
            result = model_adapter.shallow_forward(self.model, current, cache, depth, hidden)
            cache, hidden = result.past_key_values, result.exit_query_cache
            scores = result.logits[:, -1, :]
            token = int(scores.argmax(-1).item())
            probability = scores.softmax(-1)[0, token].item()
            proposal.append(token)
            if token in eos_ids:
                break
            if self.method == 'APRIL' and threshold > 0 and len(proposal) >= self.config.draft_confidence_stop_after and probability < threshold:
                break
            current = current.new_tensor([[token]])
        return current.new_tensor([proposal])

    @torch.inference_mode()
    def generate(self, prompt_ids, eos_ids=()):
        cfg = self.config
        prompt = [int(x) for x in prompt_ids]
        if not prompt:
            raise ValueError('Empty tokenized prompt')
        device = next(self.model.parameters()).device
        synchronize = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
        synchronize()
        started = time.time()
        if self.controller is not None:
            self.controller.reset_request_episode()
        lookup = RollingNGramLookup(prompt, cfg.max_new_tokens, device,
            min_ngram=cfg.min_ngram, max_ngram=cfg.max_ngram,
            max_candidates=cfg.max_candidates) if self.method == 'APRIL' else None
        current = torch.tensor([prompt], dtype=torch.long, device=device)
        cache = None
        output = []
        checkpoint = None
        checkpoint_output_len = 0
        checkpoint_logits = None
        routes = Counter()
        actions = Counter()
        records = []
        normal_target_forwards = 0
        shallow_forwards = 0
        recovery_forwards = 0
        eos_ids = set(eos_ids)
        eos_reached = False
        pld_stats = Counter()
        pld_proposer = None
        if self.method == 'PLD':
            from transformers.generation.candidate_generator import PromptLookupCandidateGenerator
            pld_config = paper_snapshot()['sections']['pld']
            pld_proposer = PromptLookupCandidateGenerator(
                eos_token_id=torch.tensor(sorted(eos_ids),dtype=torch.long,device=device),
                num_output_tokens=pld_config['max_proposed_tokens'],
                max_matching_ngram_size=pld_config['max_matching_ngram_size'],
                max_length=len(prompt)+cfg.max_new_tokens)
        while len(output) < cfg.max_new_tokens:
            context_length = len(prompt) + len(output)
            remaining = cfg.max_new_tokens - len(output)
            decision = None
            if self.method == 'AR' or (remaining == 1 and self.method != 'PLD'):
                result = model_adapter.forward(self.model, current, cache)
                cache = result.past_key_values
                committed = [int(result.logits[:, -1, :].argmax(-1).item())]
                normal_target_forwards += 1
                route = 'target_only'
                accepted = drafted = 0
                margin = None
            else:
                hit = lookup.lookup(remaining - 1) if lookup is not None else None
                if hit is not None and hit.hit:
                    proposal = lookup.stage_proposal()
                    route = 'lookup'
                elif self.method == 'PLD':
                    history = torch.tensor([prompt+output], dtype=torch.long, device=device)
                    candidate, _ = pld_proposer.get_candidates(history)
                    proposal = candidate[:, history.shape[1]:][:, :max(0,remaining-1)]
                    pld_stats['lookup_calls'] += 1
                    pld_stats['lookup_hits' if proposal.numel() else 'lookup_misses'] += 1
                    pld_stats['proposal_tokens'] += proposal.numel()
                    route = 'pld'
                else:
                    if self.controller is None:
                        depth, budget = 7, min(6, remaining-1)
                    else:
                        decision = self.controller.select(context_length)
                        depth, budget = decision.draft_layer, min(decision.speculation_length, remaining-1)
                        decision.speculation_length = budget
                    target_cache = clone_past_key_values(cache)
                    proposal = self._draft(current, cache, depth, budget, eos_ids)
                    shallow_forwards += proposal.shape[1]
                    route = 'self_draft'
                    actions[f'{depth}:{budget}'] += 1
                drafted = proposal.shape[1]
                if route != 'self_draft':
                    target_cache = clone_past_key_values(cache)
                target = model_adapter.forward(self.model, torch.cat((current, proposal),dim=1).int(), target_cache)
                normal_target_forwards += 1
                logits = target.logits[:, current.shape[1]-1:, :]
                target_ids = logits.argmax(-1)
                margin, accepted = commit_critical_margin(logits, proposal, target_ids)
                recovered = margin <= cfg.margin_threshold
                if recovered:
                    logits, exact_cache, stats = canonical_fp16_q1_target_recheck(
                        self.model, prompt, output, proposal,
                        checkpoint, checkpoint_output_len, checkpoint_logits)
                    recovery_forwards += stats['target_forward_calls']
                    target_ids = logits.argmax(-1)
                    _, accepted = commit_critical_margin(logits, proposal, target_ids)
                    committed = committed_tokens(proposal, target_ids, accepted)
                    cache, checkpoint, checkpoint_logits, stats = advance_canonical_checkpoint(
                        self.model, exact_cache, context_length+accepted, committed[-1])
                    checkpoint_output_len = len(output)+len(committed)
                    recovery_forwards += stats['target_forward_calls']
                else:
                    committed = committed_tokens(proposal, target_ids, accepted)
                    cache = model_adapter.crop_past_key_values(target.past_key_values, context_length+accepted)
                if route == 'lookup':
                    lookup.record_acceptance(accepted)
                elif route == 'pld':
                    pld_stats['accepted_proposal_tokens'] += accepted
                if decision is not None:
                    self.controller.update_after_step(
                        step_id=len(records), context_length=context_length,
                        decision=decision, num_drafted_tokens=drafted,
                        num_accepted_by_final=accepted, proxy_accepted=drafted,
                        anchor_accepted_by_final=accepted, latency_ms=None,
                        output_tokens=len(committed))
            # Stop before committing anything beyond a target-approved EOS.
            for index, token in enumerate(committed):
                if token in eos_ids:
                    # The owned PLD adapter, official AR API and all archived
                    # main EOS-shortened outputs omit the returned stop token.
                    committed = committed[:index]
                    eos_reached = True
                    break
            output.extend(committed)
            if lookup is not None:
                lookup.observe_committed(committed)
            routes[route] += 1
            records.append(dict(route=route,drafted=drafted,accepted=accepted,
                                committed=len(committed),margin=margin))
            if eos_reached:
                break
            current = torch.tensor([[committed[-1]]], dtype=torch.long, device=device)
        synchronize()
        elapsed = time.time()-started
        return dict(token_ids=output, num_generated_tokens=len(output), elapsed_time_s=elapsed,
                    routes=dict(routes),controller_actions=dict(actions),
                    lookup=lookup.summary() if lookup is not None else None,
                    rounds=records, forward_counts=dict(normal_target=normal_target_forwards,
                    shallow=shallow_forwards,recovery=recovery_forwards),
                    pld=dict(pld_stats) if self.method=='PLD' else None,
                    eos_reached=eos_reached,stop_reason='eos' if eos_reached else 'max_token_cap',
                    controller_updated=routes['self_draft'] if self.controller else 0)
