import time
from typing import List, Optional, Tuple

import torch
import transformers

from .model_adapter import crop_past_key_values, forward


RECHECK_MODE = "fp16_q1_current_round_from_canonical_committed_kv"


def clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    cloned = []
    for layer in past_key_values:
        if layer is None:
            cloned.append(None)
            continue
        key, value = layer
        if key is None:
            cloned.append((key, value))
        else:
            cloned.append((key.detach().clone(), value.detach().clone()))
    return tuple(cloned)


def init_recheck_stats(enabled: bool, threshold: float, model) -> dict:
    return {
        "enabled": enabled,
        "threshold": threshold,
        "mode": RECHECK_MODE,
        "model_dtype": str(next(model.parameters()).dtype),
        "total_round_count": 0,
        "recheck_round_count": 0,
        "recheck_round_ratio": 0.0,
        "recheck_target_forward_calls": 0,
        "recheck_q1_forward_calls": 0,
        "recheck_target_input_tokens": 0,
        "canonical_catchup_q1_calls": 0,
        "round_recheck_q1_calls": 0,
        "checkpoint_update_q1_calls": 0,
        "canonical_prefill_tokens": 0,
        "recheck_wall_seconds": 0.0,
        "recheck_time_counted_in_tps": True,
        "round_commit_min_margins": [],
        "triggered_rounds": [],
        "output_changing_rechecks": 0,
    }


def finalize_recheck_stats(stats: dict) -> None:
    total_rounds = stats["total_round_count"]
    stats["recheck_round_ratio"] = (
        stats["recheck_round_count"] / total_rounds if total_rounds else 0.0
    )


def greedy_match_count(
    draft_output_ids: torch.Tensor,
    target_tokens: torch.Tensor,
) -> int:
    verified = draft_output_ids == target_tokens[:, :-1]
    return int(((~verified).cumsum(dim=-1) < 1).sum().item())


def committed_tokens(
    draft_output_ids: torch.Tensor,
    target_tokens: torch.Tensor,
    number_of_matches: int,
) -> List[int]:
    return (
        draft_output_ids[0, :number_of_matches].tolist()
        + target_tokens[0, number_of_matches:number_of_matches + 1].tolist()
    )


def commit_critical_margin(
    verification_logits: torch.Tensor,
    draft_output_ids: torch.Tensor,
    target_tokens: torch.Tensor,
) -> Tuple[float, int]:
    number_of_matches = greedy_match_count(draft_output_ids, target_tokens)
    critical_logits = verification_logits[:, :number_of_matches + 1, :]
    top2 = critical_logits.topk(k=2, dim=-1).values
    margins = top2[..., 0] - top2[..., 1]
    return float(margins.min().item()), number_of_matches


def synchronize_model_device(model) -> None:
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fp16_q1_target_recheck(
    model: transformers.LlamaForCausalLM,
    prefill_token_ids: torch.Tensor,
    prompt_length: int,
    round_start_past_key_values: Optional[
        List[Tuple[torch.Tensor, torch.Tensor]]
    ],
):
    model_dtype = next(model.parameters()).dtype
    if model_dtype != torch.float16:
        raise RuntimeError(
            f"Exact recheck requires FP16 model weights, got {model_dtype}"
        )
    if prompt_length < 1 or prompt_length > prefill_token_ids.size(1):
        raise ValueError(
            f"Invalid prompt_length={prompt_length} for "
            f"sequence length {prefill_token_ids.size(1)}"
        )

    past_key_values = clone_past_key_values(round_start_past_key_values)
    verification_logits = []
    target_forward_calls = 0
    q1_forward_calls = 0

    with torch.inference_mode():
        # Match canonical AR prefill on the first round. Every subsequent
        # verifier position, including all draft tokens, uses query_length=1.
        if past_key_values is None and prompt_length > 1:
            result = forward(
                model,
                prefill_token_ids[:, :prompt_length].int(),
                past_key_values,
            )
            target_forward_calls += 1
            past_key_values = result.past_key_values
            verification_logits.append(result.logits[:, -1:, :])
        else:
            for token_index in range(prompt_length):
                result = forward(
                    model,
                    prefill_token_ids[:, token_index:token_index + 1].int(),
                    past_key_values,
                )
                target_forward_calls += 1
                q1_forward_calls += 1
                past_key_values = result.past_key_values
            verification_logits.append(result.logits[:, -1:, :])

        for token_index in range(prompt_length, prefill_token_ids.size(1)):
            result = forward(
                model,
                prefill_token_ids[:, token_index:token_index + 1].int(),
                past_key_values,
            )
            target_forward_calls += 1
            q1_forward_calls += 1
            past_key_values = result.past_key_values
            verification_logits.append(result.logits[:, -1:, :])

    return (
        torch.cat(verification_logits, dim=1),
        past_key_values,
        {
            "target_forward_calls": target_forward_calls,
            "q1_forward_calls": q1_forward_calls,
            "target_input_tokens": int(prefill_token_ids.numel()),
            "canonical_prefill_tokens": (
                prompt_length
                if round_start_past_key_values is None and prompt_length > 1
                else 0
            ),
        },
    )


def timed_fp16_q1_target_recheck(*args, **kwargs):
    model = kwargs.get("model", args[0] if args else None)
    synchronize_model_device(model)
    start = time.perf_counter()
    result = fp16_q1_target_recheck(*args, **kwargs)
    synchronize_model_device(model)
    return (*result, time.perf_counter() - start)


def canonical_fp16_q1_target_recheck(
    model: transformers.LlamaForCausalLM,
    prompt_token_ids: List[int],
    committed_output_ids: List[int],
    draft_output_ids: torch.Tensor,
    checkpoint_past_key_values,
    checkpoint_output_len: int,
    checkpoint_next_logits: Optional[torch.Tensor],
):
    model_dtype = next(model.parameters()).dtype
    if model_dtype != torch.float16:
        raise RuntimeError(
            f"Exact recheck requires FP16 model weights, got {model_dtype}"
        )
    if checkpoint_output_len < 0 or checkpoint_output_len > len(committed_output_ids):
        raise ValueError(
            f"Invalid checkpoint_output_len={checkpoint_output_len} for "
            f"{len(committed_output_ids)} committed tokens"
        )

    device = next(model.parameters()).device
    past_key_values = clone_past_key_values(checkpoint_past_key_values)
    target_forward_calls = 0
    q1_forward_calls = 0
    catchup_q1_calls = 0
    canonical_prefill_tokens = 0

    with torch.inference_mode():
        if past_key_values is None:
            prompt = torch.tensor([prompt_token_ids], device=device, dtype=torch.long)
            result = forward(model, prompt, None)
            past_key_values = result.past_key_values
            next_logits = result.logits[:, -1:, :]
            target_forward_calls += 1
            canonical_prefill_tokens = len(prompt_token_ids)
            replay_start = 0
        else:
            if checkpoint_next_logits is None:
                raise ValueError("checkpoint_next_logits is required with checkpoint KV")
            next_logits = checkpoint_next_logits
            replay_start = checkpoint_output_len

        for token in committed_output_ids[replay_start:]:
            token_ids = torch.tensor([[token]], device=device, dtype=torch.long)
            result = forward(model, token_ids, past_key_values)
            past_key_values = result.past_key_values
            next_logits = result.logits[:, -1:, :]
            target_forward_calls += 1
            q1_forward_calls += 1
            catchup_q1_calls += 1

        verification_logits = [next_logits]
        for token_index in range(draft_output_ids.size(1)):
            result = forward(
                model,
                draft_output_ids[:, token_index:token_index + 1].int(),
                past_key_values,
            )
            past_key_values = result.past_key_values
            verification_logits.append(result.logits[:, -1:, :])
            target_forward_calls += 1
            q1_forward_calls += 1

    return (
        torch.cat(verification_logits, dim=1),
        past_key_values,
        {
            "target_forward_calls": target_forward_calls,
            "q1_forward_calls": q1_forward_calls,
            "target_input_tokens": (
                canonical_prefill_tokens
                + len(committed_output_ids[replay_start:])
                + int(draft_output_ids.numel())
            ),
            "canonical_catchup_q1_calls": catchup_q1_calls,
            "round_recheck_q1_calls": int(draft_output_ids.numel()),
            "checkpoint_update_q1_calls": 0,
            "canonical_prefill_tokens": canonical_prefill_tokens,
        },
    )


def timed_canonical_fp16_q1_target_recheck(*args, **kwargs):
    model = kwargs.get("model", args[0] if args else None)
    synchronize_model_device(model)
    start = time.perf_counter()
    result = canonical_fp16_q1_target_recheck(*args, **kwargs)
    synchronize_model_device(model)
    return (*result, time.perf_counter() - start)


def advance_canonical_checkpoint(
    model: transformers.LlamaForCausalLM,
    exact_round_past_key_values,
    cache_length_before_commit: int,
    committed_token: int,
):
    model_dtype = next(model.parameters()).dtype
    if model_dtype != torch.float16:
        raise RuntimeError(
            f"Exact recheck requires FP16 model weights, got {model_dtype}"
        )
    official_past_key_values = crop_past_key_values(
        exact_round_past_key_values,
        cache_length_before_commit,
    )
    device = next(model.parameters()).device
    token_ids = torch.tensor([[committed_token]], device=device, dtype=torch.long)
    with torch.inference_mode():
        checkpoint_result = forward(
            model,
            token_ids,
            clone_past_key_values(official_past_key_values),
        )
    return (
        official_past_key_values,
        checkpoint_result.past_key_values,
        checkpoint_result.logits[:, -1:, :].detach(),
        {
            "target_forward_calls": 1,
            "q1_forward_calls": 1,
            "target_input_tokens": 1,
            "canonical_catchup_q1_calls": 0,
            "round_recheck_q1_calls": 0,
            "checkpoint_update_q1_calls": 1,
            "canonical_prefill_tokens": 0,
        },
    )


def timed_advance_canonical_checkpoint(*args, **kwargs):
    model = kwargs.get("model", args[0] if args else None)
    synchronize_model_device(model)
    start = time.perf_counter()
    result = advance_canonical_checkpoint(*args, **kwargs)
    synchronize_model_device(model)
    return (*result, time.perf_counter() - start)
