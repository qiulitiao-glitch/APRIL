"""Low-overhead prompt lookup proposals for APRIL.

Retrieval only proposes token IDs. The caller remains responsible for target
verification, acceptance, and KV commit/crop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


_HASH_BASE = 1_000_003
_HASH_MASK = (1 << 64) - 1
RETRIEVAL_K_CANDIDATES = (2, 4, 6, 8)


@dataclass
class RetrievalDecision:
    hit: bool = False
    k: int = 0
    ngram_length: int = 0
    match_length: int = 0
    support: int = 0
    eligible: int = 0
    uniqueness: float = 0.0
    lookup_ms: float = 0.0
    stage_ms: float = 0.0
    index_update_ms: float = 0.0
    fallback_reason: str = "not_looked_up"


class RollingNGramLookup:
    """Incremental rolling-hash lookup with one fixed proposal buffer."""

    def __init__(
        self,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
        device: torch.device,
        *,
        min_ngram: int = 2,
        max_ngram: int = 4,
        max_candidates: int = 64,
    ) -> None:
        if min_ngram < 1 or max_ngram < min_ngram:
            raise ValueError("invalid retrieval n-gram range")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")

        self.min_ngram = int(min_ngram)
        self.max_ngram = int(max_ngram)
        self.max_candidates = int(max_candidates)
        self.max_k = RETRIEVAL_K_CANDIDATES[-1]
        self._capacity = len(prompt_tokens) + int(max_new_tokens)
        self._tokens: List[int] = [0] * self._capacity
        self._length = 0
        self._positions: Dict[int, Dict[int, List[int]]] = {
            n: {} for n in range(self.min_ngram, self.max_ngram + 1)
        }
        self._suffix_hash: Dict[int, Optional[int]] = {
            n: None for n in range(self.min_ngram, self.max_ngram + 1)
        }
        self._leading_power = {
            n: pow(_HASH_BASE, n - 1, 1 << 64)
            for n in range(self.min_ngram, self.max_ngram + 1)
        }
        self._selected_tokens: List[int] = [0] * self.max_k
        self._decision = RetrievalDecision()

        pin_memory = device.type == "cuda"
        self._host_buffer = torch.empty(
            (1, self.max_k), dtype=torch.long, pin_memory=pin_memory
        )
        self._device_buffer = torch.empty(
            (1, self.max_k), dtype=torch.long, device=device
        )
        self._non_blocking = pin_memory

        self.lookup_count = 0
        self.hit_count = 0
        self.fallback_count = 0
        self.proposal_tokens = 0
        self.accepted_tokens = 0
        self.accepted_prefix_sum = 0
        self.k_counts = {k: 0 for k in RETRIEVAL_K_CANDIDATES}
        self.lookup_latencies_ms: List[float] = []
        self.round_records: List[dict] = []
        self.stage_time_s = 0.0
        self.index_update_time_s = 0.0

        build_start = time.perf_counter()
        self._extend(prompt_tokens)
        self.build_index_ms = (time.perf_counter() - build_start) * 1000.0

    @property
    def decision(self) -> RetrievalDecision:
        return self._decision

    def _hash_window(self, start: int, width: int) -> int:
        value = 0
        for index in range(start, start + width):
            value = (
                value * _HASH_BASE + (int(self._tokens[index]) + 1)
            ) & _HASH_MASK
        return value

    def _append(self, token: int) -> None:
        if self._length >= self._capacity:
            raise RuntimeError("retrieval history exceeded its fixed capacity")
        old_length = self._length
        self._tokens[old_length] = int(token)
        self._length += 1

        for n in range(self.min_ngram, self.max_ngram + 1):
            if self._length < n:
                continue
            if self._length == n:
                hash_value = self._hash_window(0, n)
            else:
                old_hash = self._suffix_hash[n]
                if old_hash is None:
                    raise RuntimeError("rolling hash state is incomplete")
                outgoing_index = self._length - n - 1
                outgoing = int(self._tokens[outgoing_index]) + 1
                hash_value = (
                    (
                        old_hash
                        - outgoing * self._leading_power[n]
                    )
                    * _HASH_BASE
                    + int(token)
                    + 1
                ) & _HASH_MASK
            self._suffix_hash[n] = hash_value
            start = self._length - n
            self._positions[n].setdefault(hash_value, []).append(start)

    def _extend(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            self._append(int(token))

    def observe_committed(self, tokens: Sequence[int]) -> None:
        start = time.perf_counter()
        self._extend(tokens)
        elapsed = time.perf_counter() - start
        self.index_update_time_s += elapsed
        self._decision.index_update_ms = elapsed * 1000.0
        if self.round_records:
            self.round_records[-1]["index_update_ms"] = self._decision.index_update_ms

    def _window_equal(self, left: int, right: int, width: int) -> bool:
        for offset in range(width):
            if self._tokens[left + offset] != self._tokens[right + offset]:
                return False
        return True

    def _context_match_length(self, candidate: int, current: int, width: int) -> int:
        matched = width
        left = candidate - 1
        right = current - 1
        while left >= 0 and right >= 0 and matched < 32:
            if self._tokens[left] != self._tokens[right]:
                break
            matched += 1
            left -= 1
            right -= 1
        return matched

    def _candidate_matches_selected(self, continuation: int, k: int) -> bool:
        for offset in range(k):
            if self._tokens[continuation + offset] != self._selected_tokens[offset]:
                return False
        return True

    @staticmethod
    def _qualified(k: int, match_length: int, support: int, uniqueness: float) -> bool:
        if k == 8:
            return match_length >= 4 and support >= 2 and uniqueness >= 0.75
        if k == 6:
            return match_length >= 3 and uniqueness >= 0.67 and (
                support >= 2 or match_length >= 6
            )
        if k == 4:
            return match_length >= 3 and uniqueness >= 0.67
        return match_length >= 2 and support >= 1 and uniqueness >= 0.50

    def lookup(self, max_k: int) -> RetrievalDecision:
        start_time = time.perf_counter()
        decision = self._decision
        decision.hit = False
        decision.k = 0
        decision.ngram_length = 0
        decision.match_length = 0
        decision.support = 0
        decision.eligible = 0
        decision.uniqueness = 0.0
        decision.stage_ms = 0.0
        decision.index_update_ms = 0.0
        decision.fallback_reason = "no_repeated_ngram"

        allowed_max_k = min(int(max_k), self.max_k)
        if allowed_max_k >= RETRIEVAL_K_CANDIDATES[0]:
            for n in range(self.max_ngram, self.min_ngram - 1, -1):
                if self._length < n:
                    continue
                suffix_hash = self._suffix_hash[n]
                bucket = self._positions[n].get(suffix_hash, [])
                current_start = self._length - n
                token_counts: Dict[int, int] = {}
                best_by_token: Dict[int, tuple] = {}
                inspected = 0

                for candidate in reversed(bucket):
                    if candidate >= current_start:
                        continue
                    continuation = candidate + n
                    available = min(self.max_k, self._length - continuation)
                    if available < RETRIEVAL_K_CANDIDATES[0]:
                        continue
                    if not self._window_equal(candidate, current_start, n):
                        continue
                    inspected += 1
                    next_token = self._tokens[continuation]
                    token_counts[next_token] = token_counts.get(next_token, 0) + 1
                    match_length = self._context_match_length(
                        candidate, current_start, n
                    )
                    score = (match_length, available, candidate)
                    if score > best_by_token.get(next_token, (-1, -1, -1))[:3]:
                        best_by_token[next_token] = (
                            match_length,
                            available,
                            candidate,
                            continuation,
                        )
                    if inspected >= self.max_candidates:
                        break

                if not token_counts:
                    continue
                selected_token = max(
                    token_counts,
                    key=lambda token: (
                        token_counts[token],
                        best_by_token[token][0],
                        best_by_token[token][1],
                        best_by_token[token][2],
                    ),
                )
                match_length, available, _, continuation = best_by_token[selected_token]
                for offset in range(available):
                    self._selected_tokens[offset] = self._tokens[continuation + offset]

                selected_k = 0
                selected_support = 0
                selected_eligible = 0
                selected_uniqueness = 0.0
                for k in reversed(RETRIEVAL_K_CANDIDATES):
                    if k > allowed_max_k or k > available:
                        continue
                    support = 0
                    eligible = 0
                    inspected = 0
                    for candidate in reversed(bucket):
                        if candidate >= current_start:
                            continue
                        candidate_continuation = candidate + n
                        if self._length - candidate_continuation < k:
                            continue
                        if not self._window_equal(candidate, current_start, n):
                            continue
                        inspected += 1
                        eligible += 1
                        if self._candidate_matches_selected(candidate_continuation, k):
                            support += 1
                        if inspected >= self.max_candidates:
                            break
                    uniqueness = support / eligible if eligible else 0.0
                    if self._qualified(k, match_length, support, uniqueness):
                        selected_k = k
                        selected_support = support
                        selected_eligible = eligible
                        selected_uniqueness = uniqueness
                        break

                if selected_k:
                    decision.hit = True
                    decision.k = selected_k
                    decision.ngram_length = n
                    decision.match_length = match_length
                    decision.support = selected_support
                    decision.eligible = selected_eligible
                    decision.uniqueness = selected_uniqueness
                    decision.fallback_reason = ""
                    break
                decision.fallback_reason = "low_confidence"
        else:
            decision.fallback_reason = "remaining_budget_lt_2"

        decision.lookup_ms = (time.perf_counter() - start_time) * 1000.0
        self.lookup_count += 1
        self.lookup_latencies_ms.append(decision.lookup_ms)
        if decision.hit:
            self.hit_count += 1
            self.proposal_tokens += decision.k
            self.k_counts[decision.k] += 1
        else:
            self.fallback_count += 1
        self.round_records.append(
            {
                "round_id": self.lookup_count - 1,
                "hit": decision.hit,
                "k": decision.k,
                "ngram_length": decision.ngram_length,
                "match_length": decision.match_length,
                "support": decision.support,
                "eligible": decision.eligible,
                "uniqueness": decision.uniqueness,
                "lookup_ms": decision.lookup_ms,
                "stage_ms": 0.0,
                "index_update_ms": 0.0,
                "accepted_prefix": None,
                "fallback_reason": decision.fallback_reason,
            }
        )
        return decision

    def stage_proposal(self) -> torch.Tensor:
        decision = self._decision
        if not decision.hit or decision.k not in RETRIEVAL_K_CANDIDATES:
            raise RuntimeError("cannot stage a retrieval miss")
        start = time.perf_counter()
        for index in range(self.max_k):
            value = self._selected_tokens[index] if index < decision.k else 0
            self._host_buffer[0, index] = value
        self._device_buffer.copy_(self._host_buffer, non_blocking=self._non_blocking)
        elapsed = time.perf_counter() - start
        decision.stage_ms = elapsed * 1000.0
        self.stage_time_s += elapsed
        self.round_records[-1]["stage_ms"] = decision.stage_ms
        return self._device_buffer[:, : decision.k]

    def record_acceptance(self, accepted_prefix: int) -> None:
        if not self._decision.hit:
            return
        accepted = max(0, min(int(accepted_prefix), self._decision.k))
        self.accepted_tokens += accepted
        self.accepted_prefix_sum += accepted
        self.round_records[-1]["accepted_prefix"] = accepted

    def summary(self) -> dict:
        sorted_latencies = sorted(self.lookup_latencies_ms)
        if sorted_latencies:
            p95_index = min(
                len(sorted_latencies) - 1,
                max(0, int(0.95 * len(sorted_latencies) + 0.999999) - 1),
            )
            mean_lookup_ms = sum(sorted_latencies) / len(sorted_latencies)
            p95_lookup_ms = sorted_latencies[p95_index]
        else:
            mean_lookup_ms = 0.0
            p95_lookup_ms = 0.0
        hit_denominator = max(1, self.hit_count)
        proposal_denominator = max(1, self.proposal_tokens)
        return {
            "enabled": True,
            "implementation": "rolling_hash_ngram_fixed_buffer_v1",
            "ngram_range": [self.min_ngram, self.max_ngram],
            "k_candidates": list(RETRIEVAL_K_CANDIDATES),
            "fixed_buffer_shape": [1, self.max_k],
            "max_candidates_per_lookup": self.max_candidates,
            "build_index_ms": self.build_index_ms,
            "lookup_count": self.lookup_count,
            "hit_count": self.hit_count,
            "fallback_count": self.fallback_count,
            "coverage": self.hit_count / self.lookup_count if self.lookup_count else 0.0,
            "fallback_ratio": (
                self.fallback_count / self.lookup_count if self.lookup_count else 0.0
            ),
            "lookup_latency_mean_ms": mean_lookup_ms,
            "lookup_latency_p95_ms": p95_lookup_ms,
            "lookup_latency_target_ms": 0.1,
            "lookup_target_met": mean_lookup_ms < 0.1,
            "stage_time_s": self.stage_time_s,
            "index_update_time_s": self.index_update_time_s,
            "total_retrieval_overhead_s": (
                self.build_index_ms / 1000.0
                + sum(self.lookup_latencies_ms) / 1000.0
                + self.stage_time_s
                + self.index_update_time_s
            ),
            "k_counts": dict(self.k_counts),
            "k_usage": {
                str(k): self.k_counts[k] / hit_denominator
                for k in RETRIEVAL_K_CANDIDATES
            },
            "proposal_tokens": self.proposal_tokens,
            "accepted_retrieval_tokens": self.accepted_tokens,
            "proposal_acceptance_rate": self.accepted_tokens / proposal_denominator,
            "accepted_prefix_mean": self.accepted_prefix_sum / hit_denominator,
            "rounds": list(self.round_records),
        }
