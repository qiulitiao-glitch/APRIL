import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple


@dataclass
class NestedExitDecision:
    target_layer: int
    anchor_layer: int
    draft_layer: int
    speculation_length: int
    mode: str
    fallback_reason: Optional[str] = None


@dataclass
class RateStat:
    accepted: int = 0
    total: int = 0
    ema: Optional[float] = None

    def update(self, accepted: int, total: int, smoothing: float) -> None:
        if total <= 0:
            return
        accepted = max(0, min(accepted, total))
        rate = accepted / total
        self.accepted += accepted
        self.total += total
        if self.ema is None:
            self.ema = rate
        else:
            self.ema = smoothing * self.ema + (1.0 - smoothing) * rate

    def value(self, default: float = 0.0) -> float:
        if self.ema is not None:
            return self.ema
        if self.total > 0:
            return self.accepted / self.total
        return default


@dataclass
class CostStat:
    total_ms: float = 0.0
    total_units: int = 0

    def update(self, latency_ms: Optional[float], units: int) -> None:
        if latency_ms is None or units <= 0:
            return
        self.total_ms += latency_ms
        self.total_units += units

    def latency_per_unit(self) -> Optional[float]:
        if self.total_units <= 0:
            return None
        return self.total_ms / self.total_units


@dataclass
class ActionUtilityStat:
    count: int = 0
    total_utility: float = 0.0
    total_utility_sq: float = 0.0
    total_latency_ms: float = 0.0
    total_output_tokens: int = 0
    ema: Optional[float] = None

    def update(
        self,
        utility: float,
        latency_ms: Optional[float],
        output_tokens: int,
        smoothing: float,
    ) -> None:
        if utility <= 0.0:
            return
        self.count += 1
        self.total_utility += utility
        self.total_utility_sq += utility * utility
        if latency_ms is not None and latency_ms > 0.0:
            self.total_latency_ms += latency_ms
        self.total_output_tokens += max(0, int(output_tokens))
        if self.ema is None:
            self.ema = utility
        else:
            self.ema = smoothing * self.ema + (1.0 - smoothing) * utility

    def mean(self) -> Optional[float]:
        if self.count <= 0:
            return None
        return self.total_utility / self.count

    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.total_utility / self.count
        return max(0.0, self.total_utility_sq / self.count - mean * mean)

    def lcb(self, z: float) -> Optional[float]:
        mean = self.mean()
        if mean is None:
            return None
        if z <= 0.0 or self.count <= 1:
            return mean
        return max(0.0, mean - z * math.sqrt(self.variance() / self.count))


def _dedupe_sorted(values: Iterable[int]) -> List[int]:
    return sorted({int(v) for v in values if int(v) > 0})


def parse_int_candidates(value, max_value: int, defaults: List[int]) -> List[int]:
    max_value = max(1, int(max_value))
    if value is None or value == "" or str(value).lower() == "auto":
        candidates = defaults
    elif isinstance(value, str):
        pieces = [
            piece.strip()
            for piece in value.replace(";", ",").replace("-", ",").split(",")
            if piece.strip()
        ]
        candidates = [int(piece) for piece in pieces]
    else:
        candidates = [int(piece) for piece in value]
    parsed = _dedupe_sorted(min(max_value, max(1, item)) for item in candidates)
    if parsed:
        return parsed
    return parse_int_candidates("auto", max_value, defaults)


def parse_layer_candidates(value, total_layers: int, ratios: List[float]) -> List[int]:
    if value is None or value == "" or str(value).lower() == "auto":
        return _dedupe_sorted(
            min(total_layers - 1, max(1, round(total_layers * ratio)))
            for ratio in ratios
        )
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        parsed = [int(piece) for piece in pieces]
    else:
        parsed = [int(piece) for piece in value]
    candidates = _dedupe_sorted(layer for layer in parsed if layer < total_layers)
    if not candidates:
        return parse_layer_candidates("auto", total_layers, ratios)
    return candidates


class TargetRelativeNestedExitController:
    def __init__(
        self,
        total_layers: int,
        anchor_candidates,
        shallow_candidates,
        max_speculation_length: int,
        tau_anchor: float,
        tau_proxy: float,
        tau_final: float,
        smoothing: float = 0.95,
        min_observations: int = 1,
        trace_path: Optional[str] = None,
        enable_probing: bool = True,
        default_anchor_layer: int = -1,
        policy: str = "threshold",
        adaptive_probe_rounds: int = 1,
        adaptive_probe_speculation_length: int = 4,
        adaptive_reprobe_interval: int = 64,
        layer_switch_margin: float = 0.0,
        fixed_draft_layer: int = -1,
        fixed_anchor_layer: int = -1,
        feedback_min_acceptance: float = 0.78,
        feedback_good_acceptance: float = 0.90,
        feedback_cooldown_steps: int = 2,
        feedback_min_speculation_length: int = 10,
        conservative_z: float = 0.75,
        context_bucket_size: int = 128,
        feedback_switch_penalty: float = 0.08,
        recent_rate_mode: str = "min",
        recent_rate_weight: float = 1.0,
        episode_memory_mode: str = "full",
        throttle_enabled: bool = False,
        throttle_initial: float = 0.50,
        throttle_up_step: float = 0.06,
        throttle_down_step: float = 0.14,
        throttle_min_k: int = 4,
        throttle_layer_bias: float = 0.18,
        throttle_prior_weight: float = 0.55,
        utility_ema_decay: float = 0.70,
        underconvert_enabled: bool = False,
        underconvert_min_acceptance: float = 0.62,
        underconvert_min_short_k_fraction: float = 0.40,
        underconvert_short_k: int = 4,
        underconvert_boost: float = 0.45,
        underconvert_utility_floor: float = 0.90,
        layer_retire_enabled: bool = False,
        layer_retire_min_total: int = 64,
        layer_retire_gap: float = 0.08,
        layer_retire_penalty: float = 0.25,
        amortization_guard_enabled: bool = False,
        amortization_min_steps: int = 4,
        amortization_min_acceptance: float = 0.60,
        amortization_min_short_k_fraction: float = 0.55,
        amortization_max_utility_ratio: float = 0.98,
        amortization_cooldown_steps: int = 3,
        amortization_down_boost: float = 0.60,
        amortization_bad_streak_min: int = 1,
        amortization_long_guard_enabled: bool = False,
        amortization_long_max_short_k_fraction: float = 0.35,
        amortization_long_min_k_cap: int = 8,
        amortization_long_down_boost: float = 0.35,
        amortization_long_bad_streak_min: int = 0,
        action_k_candidates="auto",
        action_probe_rounds: int = 1,
        action_episode_probe_budget: int = 3,
        action_reprobe_interval: int = 16,
        action_min_observations: int = 1,
        action_conservative_z: float = 0.50,
        action_explore_bonus: float = 0.05,
        action_switch_penalty: float = 0.04,
        action_candidate_limit: int = 0,
        action_anchor_enabled: bool = False,
        action_aggressive_k: int = 0,
        action_aggressive_acceptance: float = 1.0,
        action_aggressive_lcb_ratio: float = 0.55,
        action_aggressive_min_layer_acceptance: float = 0.0,
    ) -> None:
        self.target_layer = total_layers
        self.anchor_candidates = parse_layer_candidates(
            anchor_candidates, total_layers, [0.64, 0.72, 0.80, 0.88]
        )
        self.shallow_candidates = parse_layer_candidates(
            shallow_candidates, total_layers, [0.24, 0.32, 0.40, 0.48, 0.56, 0.64]
        )
        self.max_speculation_length = max(1, int(max_speculation_length))
        self.tau_anchor = tau_anchor
        self.tau_proxy = tau_proxy
        self.tau_final = tau_final
        self.smoothing = smoothing
        self.min_observations = max(1, int(min_observations))
        self.trace_path = trace_path or os.getenv("TR_NES_TRACE_PATH")
        self.enable_probing = enable_probing
        self.default_anchor_layer = self._normalize_default_anchor(default_anchor_layer)
        self.policy = str(policy or "threshold").lower()
        self.adaptive_probe_rounds = max(1, int(adaptive_probe_rounds))
        self.adaptive_probe_speculation_length = max(
            1,
            min(int(adaptive_probe_speculation_length), self.max_speculation_length),
        )
        self.adaptive_reprobe_interval = max(0, int(adaptive_reprobe_interval))
        self.layer_switch_margin = max(0.0, float(layer_switch_margin))
        self.fixed_draft_layer = self._normalize_fixed_layer(fixed_draft_layer)
        self.fixed_anchor_layer = self._normalize_fixed_layer(fixed_anchor_layer)
        self.feedback_min_acceptance = max(0.0, min(1.0, float(feedback_min_acceptance)))
        self.feedback_good_acceptance = max(
            self.feedback_min_acceptance,
            min(1.0, float(feedback_good_acceptance)),
        )
        self.feedback_cooldown_steps = max(0, int(feedback_cooldown_steps))
        self.feedback_min_speculation_length = max(
            1,
            min(int(feedback_min_speculation_length), self.max_speculation_length),
        )
        self.conservative_z = max(0.0, float(conservative_z))
        self.context_bucket_size = max(1, int(context_bucket_size))
        self.feedback_switch_penalty = max(0.0, min(0.75, float(feedback_switch_penalty)))
        self.recent_rate_mode = str(recent_rate_mode or "min").lower()
        self.recent_rate_weight = max(0.0, min(1.0, float(recent_rate_weight)))
        self.episode_memory_mode = str(episode_memory_mode or "full").lower()
        self.throttle_enabled = bool(throttle_enabled)
        self.throttle_initial = max(0.0, min(1.0, float(throttle_initial)))
        self.throttle_g = self.throttle_initial
        self.throttle_up_step = max(0.0, min(0.50, float(throttle_up_step)))
        self.throttle_down_step = max(0.0, min(0.75, float(throttle_down_step)))
        self.throttle_min_k = max(
            1,
            min(int(throttle_min_k), self.max_speculation_length),
        )
        self.throttle_layer_bias = max(0.0, min(0.75, float(throttle_layer_bias)))
        self.throttle_prior_weight = max(0.0, min(1.0, float(throttle_prior_weight)))
        self.utility_ema_decay = max(0.0, min(0.98, float(utility_ema_decay)))
        self.underconvert_enabled = bool(underconvert_enabled)
        self.underconvert_min_acceptance = max(
            0.0, min(1.0, float(underconvert_min_acceptance))
        )
        self.underconvert_min_short_k_fraction = max(
            0.0, min(1.0, float(underconvert_min_short_k_fraction))
        )
        self.underconvert_short_k = max(
            1,
            min(int(underconvert_short_k), self.max_speculation_length),
        )
        self.underconvert_boost = max(0.0, min(2.0, float(underconvert_boost)))
        self.underconvert_utility_floor = max(
            0.0, min(2.0, float(underconvert_utility_floor))
        )
        self.layer_retire_enabled = bool(layer_retire_enabled)
        self.layer_retire_min_total = max(1, int(layer_retire_min_total))
        self.layer_retire_gap = max(0.0, min(1.0, float(layer_retire_gap)))
        self.layer_retire_penalty = max(0.0, min(1.0, float(layer_retire_penalty)))
        self.amortization_guard_enabled = bool(amortization_guard_enabled)
        self.amortization_min_steps = max(1, int(amortization_min_steps))
        self.amortization_min_acceptance = max(
            0.0, min(1.0, float(amortization_min_acceptance))
        )
        self.amortization_min_short_k_fraction = max(
            0.0, min(1.0, float(amortization_min_short_k_fraction))
        )
        self.amortization_max_utility_ratio = max(
            0.0, min(2.0, float(amortization_max_utility_ratio))
        )
        self.amortization_cooldown_steps = max(0, int(amortization_cooldown_steps))
        self.amortization_down_boost = max(0.0, min(2.0, float(amortization_down_boost)))
        self.amortization_bad_streak_min = max(1, int(amortization_bad_streak_min))
        self.amortization_long_guard_enabled = bool(amortization_long_guard_enabled)
        self.amortization_long_max_short_k_fraction = max(
            0.0, min(1.0, float(amortization_long_max_short_k_fraction))
        )
        self.amortization_long_min_k_cap = max(
            1,
            min(int(amortization_long_min_k_cap), self.max_speculation_length),
        )
        self.amortization_long_down_boost = max(
            0.0, min(2.0, float(amortization_long_down_boost))
        )
        self.amortization_long_bad_streak_min = (
            self.amortization_bad_streak_min
            if int(amortization_long_bad_streak_min) <= 0
            else max(1, int(amortization_long_bad_streak_min))
        )
        default_action_k = [
            self.throttle_min_k,
            min(self.max_speculation_length, 6),
            min(self.max_speculation_length, 8),
            self.feedback_min_speculation_length,
            min(self.max_speculation_length, 12),
            min(self.max_speculation_length, 16),
            self.max_speculation_length,
        ]
        self.action_k_candidates = parse_int_candidates(
            action_k_candidates,
            self.max_speculation_length,
            default_action_k,
        )
        self.action_probe_rounds = max(0, int(action_probe_rounds))
        self.action_episode_probe_budget = max(0, int(action_episode_probe_budget))
        self.action_reprobe_interval = max(0, int(action_reprobe_interval))
        self.action_min_observations = max(1, int(action_min_observations))
        self.action_conservative_z = max(0.0, float(action_conservative_z))
        self.action_explore_bonus = max(0.0, min(1.0, float(action_explore_bonus)))
        self.action_switch_penalty = max(0.0, min(0.75, float(action_switch_penalty)))
        self.action_candidate_limit = max(0, int(action_candidate_limit))
        self.action_anchor_enabled = bool(action_anchor_enabled)
        self.action_aggressive_k = max(0, int(action_aggressive_k))
        self.action_aggressive_acceptance = max(
            0.0, min(1.0, float(action_aggressive_acceptance))
        )
        self.action_aggressive_lcb_ratio = max(
            0.0, min(1.0, float(action_aggressive_lcb_ratio))
        )
        self.action_aggressive_min_layer_acceptance = max(
            0.0, min(1.0, float(action_aggressive_min_layer_acceptance))
        )
        self.throttle_prior_g = self.throttle_initial
        self.throttle_prior_updates = 0
        self.throttle_episode_initial_g = self.throttle_initial
        self.episode_utility_ema: Optional[float] = None
        self.global_utility_ema: Optional[float] = None
        self.throttle_last_utility_ratio: Optional[float] = None
        self.throttle_updates = 0
        self.throttle_up_count = 0
        self.throttle_down_count = 0
        self.throttle_last_delta = 0.0
        self.throttle_last_reason = "initial"
        self.throttle_last_acceptance: Optional[float] = None
        self.throttle_last_reject_position: Optional[int] = None
        self.throttle_last_utility: Optional[float] = None
        self.throttle_last_utility_estimated = False
        self.throttle_last_intensity = 0.0
        self.throttle_episode_resets = 0
        self.amortization_cooldown_remaining = 0
        self.amortization_trigger_count = 0
        self.amortization_bad_streak = 0
        self.amortization_last_reason = "initial"
        self.amortization_last_short_k_fraction = 0.0
        self.amortization_last_utility_ratio: Optional[float] = None
        self.episode_index = 0
        self.episode_step_count = 0
        self.episode_good_streak = 0
        self.episode_bad_streak = 0
        self.episode_recent_outcomes: Deque[Dict[str, float]] = deque(maxlen=6)
        if self.fixed_anchor_layer <= 0 and self.fixed_draft_layer > 0:
            self.fixed_anchor_layer = self.fixed_draft_layer
        if 0 < self.fixed_anchor_layer < self.fixed_draft_layer:
            self.fixed_anchor_layer = self.fixed_draft_layer
        self.pair_stats: Dict[Tuple[int, int], RateStat] = {}
        self.context_pair_stats: Dict[Tuple[int, int, int], RateStat] = {}
        self.cost_stats: Dict[int, CostStat] = {}
        self.anchor_probe_index = 0
        self.shallow_probe_index = 0
        self.direct_probe_index = 0
        self.direct_candidates = _dedupe_sorted(
            layer
            for layer in (
                list(self.shallow_candidates)
                + list(self.anchor_candidates)
                + [self.default_anchor_layer]
            )
            if 0 < layer < self.target_layer
        )
        self.adaptive_probe_counts: Dict[int, int] = {layer: 0 for layer in self.direct_candidates}
        self.adaptive_select_count = 0
        self.adaptive_reprobe_index = 0
        self.mode_counts: Dict[str, int] = {}
        self.draft_layer_counts: Dict[int, int] = {}
        self.speculation_length_counts: Dict[int, int] = {}
        self.layer_acceptance: Dict[int, Dict[str, int]] = {}
        self.recent_layer_stats: Dict[int, Deque[Tuple[int, int]]] = {}
        self.predictability_recent_stats: Deque[Tuple[int, int]] = deque(maxlen=8)
        self.predictability_source_counts: Dict[str, int] = {}
        self.feedback_bad_streak = 0
        self.feedback_good_streak = 0
        self.feedback_cooldown_remaining = 0
        self.feedback_switch_count = 0
        self.feedback_last_layer: Optional[int] = None
        self.controller_select_time_s = 0.0
        self.controller_select_count = 0
        self.action_utility_stats: Dict[Tuple[int, int, int], ActionUtilityStat] = {}
        self.action_episode_probe_counts: Dict[Tuple[int, int, int], int] = {}
        self.action_select_count = 0
        self.action_episode_probe_count = 0
        self.action_reprobe_index = 0
        self.action_last_key: Optional[Tuple[int, int, int]] = None
        self.action_promotion_count = 0

    @classmethod
    def from_generation_config(cls, generation_config, total_layers: int):
        return cls(
            total_layers=total_layers,
            anchor_candidates=getattr(generation_config, "tr_nes_anchor_candidates", "auto"),
            shallow_candidates=getattr(generation_config, "tr_nes_shallow_candidates", "auto"),
            max_speculation_length=getattr(generation_config, "tr_nes_max_speculation_length", 4),
            tau_anchor=getattr(generation_config, "tr_nes_tau_anchor", 0.75),
            tau_proxy=getattr(generation_config, "tr_nes_tau_proxy", 0.75),
            tau_final=getattr(generation_config, "tr_nes_tau_final", 0.75),
            smoothing=getattr(generation_config, "tr_nes_smoothing", 0.95),
            min_observations=getattr(generation_config, "tr_nes_min_observations", 1),
            trace_path=getattr(generation_config, "tr_nes_trace_path", None),
            enable_probing=getattr(generation_config, "tr_nes_enable_probing", True),
            default_anchor_layer=getattr(generation_config, "exit_layer", -1),
            policy=getattr(generation_config, "tr_nes_policy", "threshold"),
            adaptive_probe_rounds=getattr(generation_config, "tr_nes_adaptive_probe_rounds", 1),
            adaptive_probe_speculation_length=getattr(
                generation_config, "tr_nes_adaptive_probe_speculation_length", 4
            ),
            adaptive_reprobe_interval=getattr(
                generation_config, "tr_nes_adaptive_reprobe_interval", 64
            ),
            layer_switch_margin=getattr(generation_config, "tr_nes_layer_switch_margin", 0.0),
            fixed_draft_layer=getattr(generation_config, "tr_nes_fixed_draft_layer", -1),
            fixed_anchor_layer=getattr(generation_config, "tr_nes_fixed_anchor_layer", -1),
            feedback_min_acceptance=getattr(
                generation_config, "tr_nes_feedback_min_acceptance", 0.78
            ),
            feedback_good_acceptance=getattr(
                generation_config, "tr_nes_feedback_good_acceptance", 0.90
            ),
            feedback_cooldown_steps=getattr(
                generation_config, "tr_nes_feedback_cooldown_steps", 2
            ),
            feedback_min_speculation_length=getattr(
                generation_config, "tr_nes_feedback_min_speculation_length", 10
            ),
            conservative_z=getattr(generation_config, "tr_nes_conservative_z", 0.75),
            context_bucket_size=getattr(generation_config, "tr_nes_context_bucket_size", 128),
            feedback_switch_penalty=getattr(
                generation_config, "tr_nes_feedback_switch_penalty", 0.08
            ),
            recent_rate_mode=getattr(generation_config, "tr_nes_recent_rate_mode", "min"),
            recent_rate_weight=getattr(generation_config, "tr_nes_recent_rate_weight", 1.0),
            episode_memory_mode=getattr(generation_config, "tr_nes_episode_memory_mode", "full"),
            throttle_enabled=getattr(generation_config, "tr_nes_throttle_enabled", False),
            throttle_initial=getattr(generation_config, "tr_nes_throttle_initial", 0.50),
            throttle_up_step=getattr(generation_config, "tr_nes_throttle_up_step", 0.06),
            throttle_down_step=getattr(generation_config, "tr_nes_throttle_down_step", 0.14),
            throttle_min_k=getattr(generation_config, "tr_nes_throttle_min_k", 4),
            throttle_layer_bias=getattr(generation_config, "tr_nes_throttle_layer_bias", 0.18),
            throttle_prior_weight=getattr(
                generation_config, "tr_nes_throttle_prior_weight", 0.55
            ),
            utility_ema_decay=getattr(generation_config, "tr_nes_utility_ema_decay", 0.70),
            underconvert_enabled=getattr(
                generation_config, "tr_nes_underconvert_enabled", False
            ),
            underconvert_min_acceptance=getattr(
                generation_config, "tr_nes_underconvert_min_acceptance", 0.62
            ),
            underconvert_min_short_k_fraction=getattr(
                generation_config, "tr_nes_underconvert_min_short_k_fraction", 0.40
            ),
            underconvert_short_k=getattr(
                generation_config, "tr_nes_underconvert_short_k", 4
            ),
            underconvert_boost=getattr(
                generation_config, "tr_nes_underconvert_boost", 0.45
            ),
            underconvert_utility_floor=getattr(
                generation_config, "tr_nes_underconvert_utility_floor", 0.90
            ),
            layer_retire_enabled=getattr(
                generation_config, "tr_nes_layer_retire_enabled", False
            ),
            layer_retire_min_total=getattr(
                generation_config, "tr_nes_layer_retire_min_total", 64
            ),
            layer_retire_gap=getattr(
                generation_config, "tr_nes_layer_retire_gap", 0.08
            ),
            layer_retire_penalty=getattr(
                generation_config, "tr_nes_layer_retire_penalty", 0.25
            ),
            amortization_guard_enabled=getattr(
                generation_config, "tr_nes_amortization_guard_enabled", False
            ),
            amortization_min_steps=getattr(
                generation_config, "tr_nes_amortization_min_steps", 4
            ),
            amortization_min_acceptance=getattr(
                generation_config, "tr_nes_amortization_min_acceptance", 0.60
            ),
            amortization_min_short_k_fraction=getattr(
                generation_config,
                "tr_nes_amortization_min_short_k_fraction",
                0.55,
            ),
            amortization_max_utility_ratio=getattr(
                generation_config, "tr_nes_amortization_max_utility_ratio", 0.98
            ),
            amortization_cooldown_steps=getattr(
                generation_config, "tr_nes_amortization_cooldown_steps", 3
            ),
            amortization_down_boost=getattr(
                generation_config, "tr_nes_amortization_down_boost", 0.60
            ),
            amortization_bad_streak_min=getattr(
                generation_config, "tr_nes_amortization_bad_streak_min", 1
            ),
            amortization_long_guard_enabled=getattr(
                generation_config, "tr_nes_amortization_long_guard_enabled", False
            ),
            amortization_long_max_short_k_fraction=getattr(
                generation_config,
                "tr_nes_amortization_long_max_short_k_fraction",
                0.35,
            ),
            amortization_long_min_k_cap=getattr(
                generation_config, "tr_nes_amortization_long_min_k_cap", 8
            ),
            amortization_long_down_boost=getattr(
                generation_config, "tr_nes_amortization_long_down_boost", 0.35
            ),
            amortization_long_bad_streak_min=getattr(
                generation_config, "tr_nes_amortization_long_bad_streak_min", 0
            ),
            action_k_candidates=getattr(
                generation_config, "tr_nes_action_k_candidates", "auto"
            ),
            action_probe_rounds=getattr(
                generation_config, "tr_nes_action_probe_rounds", 1
            ),
            action_episode_probe_budget=getattr(
                generation_config, "tr_nes_action_episode_probe_budget", 3
            ),
            action_reprobe_interval=getattr(
                generation_config, "tr_nes_action_reprobe_interval", 16
            ),
            action_min_observations=getattr(
                generation_config, "tr_nes_action_min_observations", 1
            ),
            action_conservative_z=getattr(
                generation_config, "tr_nes_action_conservative_z", 0.50
            ),
            action_explore_bonus=getattr(
                generation_config, "tr_nes_action_explore_bonus", 0.05
            ),
            action_switch_penalty=getattr(
                generation_config, "tr_nes_action_switch_penalty", 0.04
            ),
            action_candidate_limit=getattr(
                generation_config, "tr_nes_action_candidate_limit", 0
            ),
            action_anchor_enabled=getattr(
                generation_config, "tr_nes_action_anchor_enabled", False
            ),
            action_aggressive_k=getattr(
                generation_config, "tr_nes_action_aggressive_k", 0
            ),
            action_aggressive_acceptance=getattr(
                generation_config, "tr_nes_action_aggressive_acceptance", 1.0
            ),
            action_aggressive_lcb_ratio=getattr(
                generation_config, "tr_nes_action_aggressive_lcb_ratio", 0.55
            ),
            action_aggressive_min_layer_acceptance=getattr(
                generation_config, "tr_nes_action_aggressive_min_layer_acceptance", 0.0
            ),
        )

    def _normalize_fixed_layer(self, layer: int) -> int:
        try:
            layer = int(layer)
        except Exception:
            return -1
        if 0 < layer < self.target_layer:
            return layer
        return -1

    def _normalize_default_anchor(self, default_anchor_layer: int) -> int:
        if 0 < default_anchor_layer < self.target_layer:
            return default_anchor_layer
        if self.anchor_candidates:
            return self.anchor_candidates[-1]
        return max(1, self.target_layer - 1)

    def record_select_latency(self, elapsed_s: Optional[float]) -> None:
        if elapsed_s is None:
            return
        elapsed_s = float(elapsed_s)
        if elapsed_s < 0.0:
            return
        self.controller_select_time_s += elapsed_s
        self.controller_select_count += 1

    def _pair_stat(self, draft_layer: int, verifier_layer: int) -> Optional[RateStat]:
        return self.pair_stats.get((draft_layer, verifier_layer))

    def _pair_rate(self, draft_layer: int, verifier_layer: int, default: float = 0.0) -> float:
        stat = self._pair_stat(draft_layer, verifier_layer)
        return stat.value(default) if stat is not None else default

    def _context_bucket(self, context_length: int) -> int:
        try:
            context_length = int(context_length)
        except Exception:
            context_length = 0
        return max(0, context_length // self.context_bucket_size)

    def _rate_lcb(self, stat: Optional[RateStat], default: float = 0.0) -> float:
        if stat is None or stat.total <= 0:
            return default
        rate = stat.value(default)
        if self.conservative_z <= 0.0:
            return rate
        uncertainty = math.sqrt(max(rate * (1.0 - rate), 0.0) / max(1, stat.total))
        return max(0.0, rate - self.conservative_z * uncertainty)

    def _recent_rate_lcb(self, draft_layer: int, default: float = 0.0) -> float:
        rows = self.recent_layer_stats.get(draft_layer)
        if not rows:
            return default
        accepted = sum(row[0] for row in rows)
        total = sum(row[1] for row in rows)
        if total <= 0:
            return default
        stat = RateStat(accepted=accepted, total=total, ema=accepted / total)
        return self._rate_lcb(stat, default)

    def _recent_predictability_lcb(self, default: float = 0.75) -> float:
        rows = list(self.predictability_recent_stats)
        if not rows:
            for layer_rows in self.recent_layer_stats.values():
                rows.extend(layer_rows)
        if not rows:
            return default
        accepted = sum(row[0] for row in rows)
        total = sum(row[1] for row in rows)
        if total <= 0:
            return default
        stat = RateStat(accepted=accepted, total=total, ema=accepted / total)
        return self._rate_lcb(stat, default)

    def _predictability_state(self, predictability_lcb: float) -> str:
        if predictability_lcb >= self.feedback_good_acceptance:
            return "aggressive"
        if predictability_lcb >= self.feedback_min_acceptance:
            return "balanced"
        return "conservative"

    def _update_feedback_state(self, step_rate: float) -> None:
        step_rate = max(0.0, min(1.0, float(step_rate)))
        if step_rate < self.feedback_min_acceptance:
            self.feedback_bad_streak += 1
            self.feedback_good_streak = 0
            self.feedback_cooldown_remaining = self.feedback_cooldown_steps
        elif step_rate >= self.feedback_good_acceptance:
            self.feedback_good_streak += 1
            self.feedback_bad_streak = 0
            if self.feedback_cooldown_remaining > 0:
                self.feedback_cooldown_remaining -= 1
        else:
            self.feedback_bad_streak = 0
            if self.feedback_cooldown_remaining > 0:
                self.feedback_cooldown_remaining -= 1

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _episode_recent_acceptance(self) -> Optional[float]:
        drafted = sum(max(0, int(row.get("drafted", 0.0))) for row in self.episode_recent_outcomes)
        if drafted <= 0:
            return None
        accepted = sum(max(0, int(row.get("accepted", 0.0))) for row in self.episode_recent_outcomes)
        return max(0.0, min(1.0, accepted / drafted))

    def _episode_recent_utility(self) -> Optional[float]:
        values = [
            float(row["utility"])
            for row in self.episode_recent_outcomes
            if float(row.get("utility", 0.0)) > 0.0
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _episode_recent_short_k_fraction(self, current_k: Optional[int] = None) -> float:
        values = [
            max(1, int(row.get("drafted", 0.0)))
            for row in self.episode_recent_outcomes
            if int(row.get("drafted", 0.0)) > 0
        ]
        if current_k is not None and current_k > 0:
            values.append(max(1, int(current_k)))
        if not values:
            return 0.0
        short = sum(1 for value in values if value <= self.underconvert_short_k)
        return short / len(values)

    def _has_underconverted_acceptance_support(
        self,
        step_rate: float,
        recent_acceptance: Optional[float],
        utility_ratio: Optional[float],
        early_reject: bool,
        current_k: int,
    ) -> bool:
        if not self.underconvert_enabled or early_reject:
            return False
        acceptance = max(step_rate, recent_acceptance or 0.0)
        if acceptance < self.underconvert_min_acceptance:
            return False
        short_fraction = self._episode_recent_short_k_fraction(current_k=current_k)
        if short_fraction < self.underconvert_min_short_k_fraction:
            return False
        if (
            utility_ratio is not None
            and utility_ratio < self.underconvert_utility_floor
        ):
            return False
        return True

    def reset_request_episode(self) -> None:
        """Reset fast online state so each prompt has its own adaptation episode."""
        self.episode_index += 1
        self.episode_step_count = 0
        self.episode_good_streak = 0
        self.episode_bad_streak = 0
        self.episode_recent_outcomes.clear()
        self.throttle_episode_resets += 1
        if self.episode_memory_mode in {"hybrid", "dual", "slow_prior"} and self.throttle_prior_updates > 0:
            self.throttle_g = self._clamp(
                (1.0 - self.throttle_prior_weight) * self.throttle_initial
                + self.throttle_prior_weight * self.throttle_prior_g,
                0.0,
                1.0,
            )
        else:
            self.throttle_g = self.throttle_initial
        self.throttle_episode_initial_g = self.throttle_g
        self.throttle_last_delta = 0.0
        self.throttle_last_reason = "episode_reset"
        self.throttle_last_acceptance = None
        self.throttle_last_reject_position = None
        self.throttle_last_utility = None
        self.throttle_last_utility_estimated = False
        self.throttle_last_utility_ratio = None
        self.throttle_last_intensity = 0.0
        self.episode_utility_ema = None
        self.amortization_cooldown_remaining = 0
        self.amortization_bad_streak = 0
        self.amortization_last_reason = "episode_reset"
        self.amortization_last_short_k_fraction = 0.0
        self.amortization_last_utility_ratio = None
        self.action_episode_probe_counts.clear()
        self.action_episode_probe_count = 0
        self.action_last_key = None

        full_reset = self.episode_memory_mode not in {"hybrid", "dual", "slow_prior"}
        if full_reset:
            self.pair_stats.clear()
            self.action_utility_stats.clear()
        self.context_pair_stats.clear()
        self.recent_layer_stats.clear()
        self.predictability_recent_stats.clear()
        self.predictability_source_counts.clear()
        self.feedback_bad_streak = 0
        self.feedback_good_streak = 0
        self.feedback_cooldown_remaining = 0
        self.feedback_last_layer = None
        self.anchor_probe_index = 0
        self.shallow_probe_index = 0
        self.direct_probe_index = 0
        if full_reset:
            self.adaptive_probe_counts = {layer: 0 for layer in self.direct_candidates}
        self.adaptive_select_count = 0
        self.adaptive_reprobe_index = 0

    def _throttle_k_cap(self) -> int:
        if not self.throttle_enabled:
            return self.max_speculation_length
        span = max(0, self.max_speculation_length - self.throttle_min_k)
        cap = max(
            1,
            min(
                self.max_speculation_length,
                int(round(self.throttle_min_k + self.throttle_g * span)),
            ),
        )
        recent_acceptance = self._episode_recent_acceptance()
        if recent_acceptance is None:
            return cap

        risk = 0.0
        if self.feedback_min_acceptance > 0.0:
            risk = max(
                risk,
                (self.feedback_min_acceptance - recent_acceptance) / self.feedback_min_acceptance,
            )
        if self.episode_bad_streak > 0:
            risk = max(risk, min(1.0, 0.25 + 0.18 * self.episode_bad_streak))
        if risk <= 0.0:
            return cap

        risk = self._clamp(risk, 0.0, 1.0)
        risk_cap = self.throttle_min_k + (cap - self.throttle_min_k) * (1.0 - risk)
        return max(1, min(cap, int(risk_cap)))

    def _throttle_probe_k(self) -> int:
        return min(
            self.feedback_min_speculation_length,
            self.adaptive_probe_speculation_length,
            self._throttle_k_cap(),
        )

    def _throttle_score_multiplier(self, draft_layer: int, observed_layers: List[int]) -> float:
        if not self.throttle_enabled or len(observed_layers) <= 1 or self.throttle_layer_bias <= 0.0:
            return 1.0
        ordered = sorted(observed_layers)
        try:
            rank = ordered.index(draft_layer)
        except ValueError:
            return 1.0
        depth = rank / max(1, len(ordered) - 1)
        shallow_preference = 1.0 - 2.0 * depth
        multiplier = 1.0 + (self.throttle_g - 0.5) * 2.0 * self.throttle_layer_bias * shallow_preference
        return max(0.25, multiplier)

    def _layer_retire_score_multiplier(
        self,
        draft_layer: int,
        rate_lcb: float,
        layer_rate_lcbs: Dict[int, float],
    ) -> float:
        if (
            not self.layer_retire_enabled
            or len(layer_rate_lcbs) <= 1
            or self.layer_retire_penalty >= 1.0
        ):
            return 1.0
        stat = self._pair_stat(draft_layer, self.target_layer)
        if stat is None or stat.total < self.layer_retire_min_total:
            return 1.0
        best_lcb = max(layer_rate_lcbs.values()) if layer_rate_lcbs else rate_lcb
        if rate_lcb + self.layer_retire_gap < best_lcb:
            return self.layer_retire_penalty
        return 1.0

    def _update_throttle_state(
        self,
        step_rate: float,
        num_accepted_by_final: int,
        num_drafted_tokens: int,
        latency_ms: Optional[float] = None,
        output_tokens: int = 0,
        draft_layer: Optional[int] = None,
    ) -> None:
        if not self.throttle_enabled or num_drafted_tokens <= 0:
            return
        self.episode_step_count += 1
        step_rate = max(0.0, min(1.0, float(step_rate)))
        num_accepted_by_final = max(0, min(int(num_accepted_by_final), int(num_drafted_tokens)))
        num_drafted_tokens = max(1, int(num_drafted_tokens))
        reject_position = (
            None
            if num_accepted_by_final >= num_drafted_tokens
            else max(1, int(num_accepted_by_final) + 1)
        )
        utility = None
        utility_estimated = False
        if latency_ms is not None and latency_ms > 0.0 and output_tokens > 0:
            utility = max(0.0, float(output_tokens) / float(latency_ms))
        elif output_tokens > 0 and draft_layer is not None:
            stat = self.cost_stats.get(int(draft_layer))
            latency_per_unit_ms = stat.latency_per_unit() if stat is not None else None
            if latency_per_unit_ms is not None and latency_per_unit_ms > 0.0:
                estimated_latency_ms = latency_per_unit_ms * max(1, num_drafted_tokens)
                utility = max(0.0, float(output_tokens) / estimated_latency_ms)
                utility_estimated = True
        recent_acceptance = self._episode_recent_acceptance()
        recent_utility = self._episode_recent_utility()
        utility_reference = self.episode_utility_ema
        if utility_reference is None:
            utility_reference = recent_utility
        if utility_reference is None:
            utility_reference = self.global_utility_ema
        utility_ratio = None
        if utility is not None and utility_reference is not None and utility_reference > 0.0:
            utility_ratio = utility / utility_reference

        late_or_full = reject_position is None or reject_position > max(
            1,
            int(math.ceil(num_drafted_tokens * 0.70)),
        )
        early_reject = reject_position is not None and reject_position <= max(
            1,
            int(math.ceil(num_drafted_tokens * 0.35)),
        )
        if step_rate < self.feedback_min_acceptance or early_reject:
            self.episode_bad_streak += 1
            self.episode_good_streak = 0
        elif step_rate >= self.feedback_good_acceptance and late_or_full:
            self.episode_good_streak += 1
            self.episode_bad_streak = 0
        else:
            self.episode_good_streak = 0
            self.episode_bad_streak = 0

        old_g = self.throttle_g
        delta = 0.0
        reason = "hold"
        intensity = 0.0
        underconvert_support = self._has_underconverted_acceptance_support(
            step_rate=step_rate,
            recent_acceptance=recent_acceptance,
            utility_ratio=utility_ratio,
            early_reject=early_reject,
            current_k=num_drafted_tokens,
        )
        short_fraction = self._episode_recent_short_k_fraction(
            current_k=num_drafted_tokens
        )
        effective_acceptance = max(step_rate, recent_acceptance or 0.0)
        amortization_regression = False
        if self.amortization_guard_enabled:
            utility_is_regressing = (
                utility_ratio is not None
                and utility_ratio > 0.0
                and utility_ratio < self.amortization_max_utility_ratio
            )
            common_candidate = (
                self.episode_step_count >= self.amortization_min_steps
                and not early_reject
                and effective_acceptance >= self.amortization_min_acceptance
                and utility_is_regressing
            )
            short_amortization_candidate = (
                common_candidate
                and short_fraction >= self.amortization_min_short_k_fraction
            )
            long_overreach_candidate = (
                common_candidate
                and self.amortization_long_guard_enabled
                and short_fraction <= self.amortization_long_max_short_k_fraction
                and self._throttle_k_cap() >= self.amortization_long_min_k_cap
            )
            amortization_candidate = (
                short_amortization_candidate or long_overreach_candidate
            )
            if amortization_candidate:
                self.amortization_bad_streak += 1
            else:
                self.amortization_bad_streak = 0
            amortization_bad_streak_threshold = (
                self.amortization_long_bad_streak_min
                if long_overreach_candidate
                else self.amortization_bad_streak_min
            )
            amortization_regression = (
                self.amortization_bad_streak >= amortization_bad_streak_threshold
            )
            if amortization_regression:
                self.amortization_cooldown_remaining = max(
                    self.amortization_cooldown_remaining,
                    self.amortization_cooldown_steps,
                )
                self.amortization_trigger_count += 1
                if long_overreach_candidate:
                    self.amortization_last_reason = "long_window_overreach"
                else:
                    self.amortization_last_reason = "utility_not_amortized"
            elif amortization_candidate:
                if long_overreach_candidate:
                    self.amortization_last_reason = "long_candidate"
                else:
                    self.amortization_last_reason = "candidate"
            else:
                self.amortization_last_reason = "clear"
            self.amortization_last_short_k_fraction = short_fraction
            self.amortization_last_utility_ratio = utility_ratio
        if amortization_regression:
            underconvert_support = False

        if amortization_regression:
            ratio_gap = self.amortization_max_utility_ratio - float(utility_ratio or 0.0)
            base_boost = (
                self.amortization_long_down_boost
                if self.amortization_last_reason == "long_window_overreach"
                else self.amortization_down_boost
            )
            locality_term = (
                0.50 * (1.0 - short_fraction)
                if self.amortization_last_reason == "long_window_overreach"
                else 0.50 * short_fraction
            )
            intensity = self._clamp(
                base_boost
                + locality_term
                + 2.0 * max(0.0, ratio_gap),
                0.20,
                2.00,
            )
            delta = -self.throttle_down_step * min(2.00, intensity)
            reason = (
                "long_window_overreach_guard"
                if self.amortization_last_reason == "long_window_overreach"
                else "amortization_guard"
            )
        elif (step_rate < self.feedback_min_acceptance or early_reject) and not underconvert_support:
            acceptance_loss = 0.0
            if self.feedback_min_acceptance > 0.0:
                acceptance_loss = (
                    self.feedback_min_acceptance - step_rate
                ) / self.feedback_min_acceptance
            reject_severity = 0.0
            if reject_position is not None:
                reject_fraction = reject_position / max(1, num_drafted_tokens)
                reject_severity = self._clamp((0.50 - reject_fraction) / 0.50, 0.0, 1.0)
            local_risk = 0.0
            if recent_acceptance is not None and self.feedback_min_acceptance > 0.0:
                local_risk = self._clamp(
                    (self.feedback_min_acceptance - recent_acceptance)
                    / self.feedback_min_acceptance,
                    0.0,
                    1.0,
                )
            intensity = (
                0.55 * self._clamp(acceptance_loss, 0.0, 1.0)
                + 0.30 * reject_severity
                + 0.15 * local_risk
            )
            intensity *= 1.0 + min(0.75, 0.20 * max(0, self.episode_bad_streak - 1))
            delta = -self.throttle_down_step * max(0.20, min(1.75, intensity))
            reason = "quantified_local_risk"
        else:
            denom = max(1e-6, 1.0 - self.feedback_min_acceptance)
            acceptance_gain = self._clamp(
                (step_rate - self.feedback_min_acceptance) / denom,
                0.0,
                1.0,
            )
            late_bonus = 1.0 if late_or_full else 0.0
            local_support = 0.0
            if recent_acceptance is not None:
                denom_good = max(1e-6, 1.0 - self.feedback_good_acceptance)
                local_support = self._clamp(
                    (recent_acceptance - self.feedback_good_acceptance) / denom_good,
                    0.0,
                    1.0,
                )
            streak_support = min(1.0, max(0, self.episode_good_streak - 1) / 3.0)
            intensity = (
                0.45 * acceptance_gain
                + 0.20 * late_bonus
                + 0.20 * local_support
                + 0.15 * streak_support
            )
            if utility_ratio is not None:
                if utility_ratio < 0.92:
                    intensity -= min(0.60, (0.92 - utility_ratio) * 2.0)
                    reason = "utility_regression"
                elif utility_ratio > 1.0:
                    intensity += min(0.35, (utility_ratio - 1.0) * 0.75)
            elif step_rate >= self.feedback_good_acceptance and late_or_full:
                intensity += 0.10
            if underconvert_support:
                short_fraction = self._episode_recent_short_k_fraction(
                    current_k=num_drafted_tokens
                )
                acceptance = max(step_rate, recent_acceptance or 0.0)
                support = self._clamp(
                    (acceptance - self.underconvert_min_acceptance)
                    / max(1e-6, 1.0 - self.underconvert_min_acceptance),
                    0.0,
                    1.0,
                )
                intensity += self.underconvert_boost * (
                    0.50 + 0.30 * short_fraction + 0.20 * support
                )
                reason = "underconverted_acceptance_support"
            if self.feedback_cooldown_remaining > 0:
                intensity *= 0.50
            if intensity > 0.0:
                delta = self.throttle_up_step * min(1.50, intensity)
                if reason == "hold":
                    reason = "quantified_temporal_support"
            elif intensity < -0.05:
                delta = self.throttle_down_step * max(-0.75, intensity)

        self.throttle_g = max(0.0, min(1.0, self.throttle_g + delta))
        self.throttle_updates += 1
        self.throttle_last_delta = self.throttle_g - old_g
        self.throttle_last_reason = reason
        self.throttle_last_acceptance = step_rate
        self.throttle_last_reject_position = reject_position
        self.throttle_last_utility = utility
        self.throttle_last_utility_estimated = utility_estimated
        self.throttle_last_utility_ratio = utility_ratio
        self.throttle_last_intensity = intensity
        if utility is not None and utility > 0.0:
            if self.episode_utility_ema is None:
                self.episode_utility_ema = utility
            else:
                self.episode_utility_ema = (
                    self.utility_ema_decay * self.episode_utility_ema
                    + (1.0 - self.utility_ema_decay) * utility
                )
            if self.global_utility_ema is None:
                self.global_utility_ema = utility
            else:
                global_decay = 0.95
                self.global_utility_ema = global_decay * self.global_utility_ema + (
                    1.0 - global_decay
                ) * utility
        prior_target = self.throttle_g
        if step_rate < self.feedback_min_acceptance or early_reject:
            prior_target = min(prior_target, self.throttle_initial)
        elif utility_ratio is not None and utility_ratio < 0.92:
            prior_target = min(prior_target, old_g)
        if self.throttle_prior_updates <= 0:
            self.throttle_prior_g = prior_target
        else:
            prior_decay = 0.90
            self.throttle_prior_g = prior_decay * self.throttle_prior_g + (
                1.0 - prior_decay
            ) * prior_target
        self.throttle_prior_g = self._clamp(self.throttle_prior_g, 0.0, 1.0)
        self.throttle_prior_updates += 1
        self.episode_recent_outcomes.append(
            {
                "accepted": float(num_accepted_by_final),
                "drafted": float(num_drafted_tokens),
                "utility": float(utility or 0.0),
                "utility_estimated": 1.0 if utility_estimated else 0.0,
                "utility_ratio": float(utility_ratio or 0.0),
                "early_reject": 1.0 if early_reject else 0.0,
            }
        )
        if self.throttle_last_delta > 0:
            self.throttle_up_count += 1
        elif self.throttle_last_delta < 0:
            self.throttle_down_count += 1

    def record_predictability_observation(
        self,
        accepted: int,
        total: int,
        source: str = "tr",
        update_feedback: bool = True,
    ) -> None:
        total = max(1, int(total))
        accepted = max(0, min(int(accepted), total))
        self.predictability_recent_stats.append((accepted, total))
        source = str(source or "unknown")
        self.predictability_source_counts[source] = (
            self.predictability_source_counts.get(source, 0) + 1
        )
        if update_feedback:
            self._update_feedback_state(accepted / total)

    def _contextual_rate_lcb(
        self,
        draft_layer: int,
        verifier_layer: int,
        context_length: int,
        default: float = 0.75,
    ) -> float:
        global_stat = self._pair_stat(draft_layer, verifier_layer)
        if global_stat is None or global_stat.total < self.min_observations:
            return default

        values = [self._rate_lcb(global_stat, default)]
        bucket = self._context_bucket(context_length)
        bucket_stat = self.context_pair_stats.get((bucket, draft_layer, verifier_layer))
        if bucket_stat is not None and bucket_stat.total >= self.min_observations:
            values.append(self._rate_lcb(bucket_stat, values[0]))
        context_lcb = min(values)
        recent = self._recent_rate_lcb(draft_layer, context_lcb)
        if self.recent_rate_mode in {"off", "none", "disabled"}:
            return max(0.0, min(1.0, context_lcb))
        if self.recent_rate_mode in {"soft", "blend"} and recent < context_lcb:
            penalty = (context_lcb - recent) * self.recent_rate_weight
            values.append(context_lcb - penalty)
        elif recent is not None:
            values.append(recent)
        return max(0.0, min(1.0, min(values)))

    def _has_pair_observations(self, draft_layer: int, verifier_layer: int) -> bool:
        stat = self._pair_stat(draft_layer, verifier_layer)
        return stat is not None and stat.total >= self.min_observations

    def _update_pair(self, draft_layer: int, verifier_layer: int, accepted: int, total: int) -> None:
        key = (draft_layer, verifier_layer)
        if key not in self.pair_stats:
            self.pair_stats[key] = RateStat()
        self.pair_stats[key].update(accepted, total, self.smoothing)

    def _update_context_pair(
        self,
        context_length: int,
        draft_layer: int,
        verifier_layer: int,
        accepted: int,
        total: int,
    ) -> None:
        key = (self._context_bucket(context_length), draft_layer, verifier_layer)
        if key not in self.context_pair_stats:
            self.context_pair_stats[key] = RateStat()
        self.context_pair_stats[key].update(accepted, total, self.smoothing)

    def _next_anchor_probe(self) -> int:
        if not self.anchor_candidates:
            return self.default_anchor_layer
        layer = self.anchor_candidates[self.anchor_probe_index % len(self.anchor_candidates)]
        self.anchor_probe_index += 1
        return layer

    def _next_shallow_probe(self, anchor_layer: int) -> Optional[int]:
        candidates = [layer for layer in self.shallow_candidates if layer < anchor_layer]
        if not candidates:
            return None
        layer = candidates[self.shallow_probe_index % len(candidates)]
        self.shallow_probe_index += 1
        return layer

    def _has_expected_gain(self, draft_layer: int, anchor_layer: int) -> bool:
        draft_latency = self.cost_stats.get(draft_layer, CostStat()).latency_per_unit()
        anchor_latency = self.cost_stats.get(anchor_layer, CostStat()).latency_per_unit()
        if draft_latency is not None and anchor_latency is not None:
            return draft_latency < anchor_latency
        return draft_layer < anchor_layer

    def _next_direct_probe(self) -> int:
        if not self.direct_candidates:
            return self.default_anchor_layer
        layer = self.direct_candidates[self.direct_probe_index % len(self.direct_candidates)]
        self.direct_probe_index += 1
        return layer

    def _expected_tokens_for_rate(self, rate: float, speculation_length: int) -> float:
        rate = max(0.0, min(1.0, float(rate)))
        expected = 0.0
        continuation = 1.0
        for _ in range(speculation_length + 1):
            expected += continuation
            continuation *= rate
        return expected

    def _direct_tpl_score(self, draft_layer: int, speculation_length: int, rate: float) -> float:
        expected_tokens = self._expected_tokens_for_rate(rate, speculation_length)
        layer_cost = self.target_layer + draft_layer * speculation_length
        if layer_cost <= 0:
            return 0.0

        score = expected_tokens / layer_cost
        min_cost_units = max(16, self.max_speculation_length)
        stat = self.cost_stats.get(draft_layer)
        if stat is None or stat.total_units < min_cost_units:
            return score
        observed_latency = stat.latency_per_unit()
        if observed_latency is None or observed_latency <= 0.0:
            return score

        observed = [
            other.latency_per_unit()
            for other in self.cost_stats.values()
            if other.total_units >= min_cost_units
            and other.latency_per_unit() is not None
            and other.latency_per_unit() > 0.0
        ]
        if not observed:
            return score

        # Measured latency is used as a relative layer penalty after enough
        # samples. It should not let a single noisy probe decide all routing.
        min_latency = max(min(observed), 1e-9)
        return score / max(1.0, observed_latency / min_latency)

    def _select_direct_tpl(self) -> NestedExitDecision:
        observed_layers = [
            layer
            for layer in self.direct_candidates
            if self._has_pair_observations(layer, self.target_layer)
        ]
        if not observed_layers:
            draft_layer = self._next_direct_probe()
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=draft_layer,
                draft_layer=draft_layer,
                speculation_length=1,
                mode="direct_tpl_probe",
                fallback_reason="no_direct_target_stats",
            )

        best = None
        for draft_layer in sorted(observed_layers):
            rate = self._pair_rate(draft_layer, self.target_layer)
            layer_best = None
            for speculation_length in range(1, self.max_speculation_length + 1):
                score = self._direct_tpl_score(draft_layer, speculation_length, rate)
                candidate = (score, rate, -draft_layer, speculation_length, draft_layer)
                if layer_best is None or candidate > layer_best:
                    layer_best = candidate
            if best is None:
                best = layer_best
                continue
            best_score, _, _, _, best_layer = best
            layer_score, _, _, _, _ = layer_best
            if draft_layer > best_layer:
                if layer_score > best_score * (1.0 + self.layer_switch_margin):
                    best = layer_best
            elif layer_best > best:
                best = layer_best

        _, _, _, speculation_length, draft_layer = best
        return NestedExitDecision(
            target_layer=self.target_layer,
            anchor_layer=draft_layer,
            draft_layer=draft_layer,
            speculation_length=speculation_length,
            mode="direct_tpl",
        )

    def _select_fixed_anchor_tpl(self) -> NestedExitDecision:
        draft_layer = self.fixed_draft_layer if self.fixed_draft_layer > 0 else self.default_anchor_layer
        anchor_layer = self.fixed_anchor_layer if self.fixed_anchor_layer > 0 else draft_layer
        if anchor_layer < draft_layer:
            anchor_layer = draft_layer

        if not self._has_pair_observations(draft_layer, self.target_layer):
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=anchor_layer,
                draft_layer=draft_layer,
                speculation_length=self.adaptive_probe_speculation_length,
                mode="anchor_tpl_probe" if anchor_layer != draft_layer else "direct_action_probe",
                fallback_reason="no_fixed_action_stats",
            )

        direct_rate = self._pair_rate(draft_layer, self.target_layer)
        best = None
        for speculation_length in range(1, self.max_speculation_length + 1):
            rate = direct_rate
            if anchor_layer != draft_layer:
                proxy_rate = self._pair_rate(draft_layer, anchor_layer, direct_rate)
                anchor_rate = self._pair_rate(anchor_layer, self.target_layer, direct_rate)
                rate = min(direct_rate, proxy_rate, anchor_rate)
            score = self._direct_tpl_score(draft_layer, speculation_length, rate)
            candidate = (score, rate, speculation_length)
            if best is None or candidate > best:
                best = candidate

        _, _, speculation_length = best
        return NestedExitDecision(
            target_layer=self.target_layer,
            anchor_layer=anchor_layer,
            draft_layer=draft_layer,
            speculation_length=speculation_length,
            mode="anchor_tpl" if anchor_layer != draft_layer else "direct_action_tpl",
        )

    def _select_layer_adaptive_tpl(self) -> NestedExitDecision:
        self.adaptive_select_count += 1
        if self.enable_probing:
            under_probed = [
                layer
                for layer in self.direct_candidates
                if self.adaptive_probe_counts.get(layer, 0) < self.adaptive_probe_rounds
            ]
            if under_probed:
                draft_layer = under_probed[0]
                self.adaptive_probe_counts[draft_layer] = (
                    self.adaptive_probe_counts.get(draft_layer, 0) + 1
                )
                return NestedExitDecision(
                    target_layer=self.target_layer,
                    anchor_layer=draft_layer,
                    draft_layer=draft_layer,
                    speculation_length=self.adaptive_probe_speculation_length,
                    mode="layer_adaptive_probe",
                    fallback_reason="initial_layer_probe",
                )

            if (
                self.adaptive_reprobe_interval > 0
                and self.direct_candidates
                and self.adaptive_select_count % self.adaptive_reprobe_interval == 0
            ):
                draft_layer = self.direct_candidates[
                    self.adaptive_reprobe_index % len(self.direct_candidates)
                ]
                self.adaptive_reprobe_index += 1
                self.adaptive_probe_counts[draft_layer] = (
                    self.adaptive_probe_counts.get(draft_layer, 0) + 1
                )
                return NestedExitDecision(
                    target_layer=self.target_layer,
                    anchor_layer=draft_layer,
                    draft_layer=draft_layer,
                    speculation_length=self.adaptive_probe_speculation_length,
                    mode="layer_adaptive_reprobe",
                    fallback_reason="periodic_layer_reprobe",
                )

        decision = self._select_direct_tpl()
        if decision.mode == "direct_tpl":
            decision.mode = "layer_adaptive_tpl"
        return decision

    def _feedback_speculation_length(self, rate_lcb: float, max_k: Optional[int] = None) -> int:
        effective_max_k = (
            self.max_speculation_length
            if max_k is None
            else max(1, min(self.max_speculation_length, int(max_k)))
        )
        min_k = min(self.feedback_min_speculation_length, effective_max_k)
        if effective_max_k <= min_k:
            return effective_max_k
        if rate_lcb >= self.feedback_good_acceptance:
            return effective_max_k
        midpoint = max(
            min_k,
            int(round((effective_max_k + min_k) / 2.0)),
        )
        if rate_lcb >= self.feedback_min_acceptance:
            return min(effective_max_k, midpoint)
        return min_k

    def _select_feedback_adaptive_tpl(self, context_length: int) -> NestedExitDecision:
        self.adaptive_select_count += 1
        if self.enable_probing:
            under_probed = [
                layer
                for layer in self.direct_candidates
                if self.adaptive_probe_counts.get(layer, 0) < self.adaptive_probe_rounds
            ]
            if under_probed:
                draft_layer = under_probed[0]
                self.adaptive_probe_counts[draft_layer] = (
                    self.adaptive_probe_counts.get(draft_layer, 0) + 1
                )
                return NestedExitDecision(
                    target_layer=self.target_layer,
                    anchor_layer=draft_layer,
                    draft_layer=draft_layer,
                    speculation_length=self._throttle_probe_k(),
                    mode="feedback_adaptive_probe",
                    fallback_reason="initial_feedback_layer_probe",
                )

            if (
                self.adaptive_reprobe_interval > 0
                and self.direct_candidates
                and self.adaptive_select_count % self.adaptive_reprobe_interval == 0
            ):
                draft_layer = self.direct_candidates[
                    self.adaptive_reprobe_index % len(self.direct_candidates)
                ]
                self.adaptive_reprobe_index += 1
                self.adaptive_probe_counts[draft_layer] = (
                    self.adaptive_probe_counts.get(draft_layer, 0) + 1
                )
                return NestedExitDecision(
                    target_layer=self.target_layer,
                    anchor_layer=draft_layer,
                    draft_layer=draft_layer,
                    speculation_length=self._throttle_probe_k(),
                    mode="feedback_adaptive_reprobe",
                    fallback_reason="periodic_feedback_layer_reprobe",
                )

        observed_layers = [
            layer
            for layer in self.direct_candidates
            if self._has_pair_observations(layer, self.target_layer)
        ]
        if not observed_layers:
            draft_layer = self._next_direct_probe()
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=draft_layer,
                draft_layer=draft_layer,
                speculation_length=min(
                    self.feedback_min_speculation_length,
                    self._throttle_k_cap(),
                ),
                mode="feedback_adaptive_probe",
                fallback_reason="no_feedback_stats",
            )

        shallowest = min(observed_layers)
        has_deeper = any(layer > shallowest for layer in observed_layers)
        k_cap = self._throttle_k_cap()
        layer_rate_lcbs = {}
        for draft_layer in sorted(observed_layers):
            layer_rate_lcbs[draft_layer] = self._contextual_rate_lcb(
                draft_layer,
                self.target_layer,
                context_length,
                default=self._pair_rate(draft_layer, self.target_layer, 0.75),
            )
        best = None
        for draft_layer in sorted(observed_layers):
            rate_lcb = layer_rate_lcbs[draft_layer]
            speculation_length = self._feedback_speculation_length(rate_lcb, max_k=k_cap)
            score = self._direct_tpl_score(draft_layer, speculation_length, rate_lcb)
            score *= self._throttle_score_multiplier(draft_layer, observed_layers)
            score *= self._layer_retire_score_multiplier(
                draft_layer,
                rate_lcb,
                layer_rate_lcbs,
            )
            if (
                self.feedback_cooldown_remaining > 0
                and has_deeper
                and draft_layer == shallowest
            ):
                score *= 0.90
            if (
                self.feedback_last_layer is not None
                and draft_layer != self.feedback_last_layer
                and self.feedback_switch_penalty > 0.0
            ):
                score *= max(0.0, 1.0 - self.feedback_switch_penalty)
            candidate = (score, rate_lcb, -draft_layer, speculation_length, draft_layer)
            if best is None or candidate > best:
                best = candidate

        _, _, _, speculation_length, draft_layer = best
        return NestedExitDecision(
            target_layer=self.target_layer,
            anchor_layer=draft_layer,
            draft_layer=draft_layer,
            speculation_length=speculation_length,
            mode="feedback_adaptive_tpl",
        )

    def _action_key_to_decision(
        self,
        key: Tuple[int, int, int],
        mode: str,
        fallback_reason: Optional[str] = None,
    ) -> NestedExitDecision:
        draft_layer, anchor_layer, speculation_length = key
        return NestedExitDecision(
            target_layer=self.target_layer,
            anchor_layer=anchor_layer,
            draft_layer=draft_layer,
            speculation_length=speculation_length,
            mode=mode,
            fallback_reason=fallback_reason,
        )

    def _action_candidates(self, observed_layers: Optional[List[int]] = None) -> List[Tuple[int, int, int]]:
        layers = observed_layers or list(self.direct_candidates)
        layers = [layer for layer in _dedupe_sorted(layers) if 0 < layer < self.target_layer]
        if not layers:
            layers = [self.default_anchor_layer]
        direct_keys = []
        for k_value in self.action_k_candidates:
            for draft_layer in layers:
                direct_keys.append((draft_layer, draft_layer, k_value))

        anchor_keys = []
        if self.action_anchor_enabled:
            anchors = _dedupe_sorted(list(self.anchor_candidates) + list(self.direct_candidates))
            for k_value in self.action_k_candidates:
                for draft_layer in layers:
                    deeper = [anchor for anchor in anchors if draft_layer < anchor < self.target_layer]
                    if deeper:
                        anchor_keys.append((draft_layer, deeper[0], k_value))

        keys = []
        seen = set()
        for key in direct_keys + anchor_keys:
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
        if self.action_candidate_limit > 0:
            return keys[: self.action_candidate_limit]
        return keys

    def _action_stat_lcb(self, key: Tuple[int, int, int]) -> Optional[float]:
        stat = self.action_utility_stats.get(key)
        if stat is None or stat.count < self.action_min_observations:
            return None
        return stat.lcb(self.action_conservative_z)

    def _select_action_probe(
        self,
        candidates: List[Tuple[int, int, int]],
    ) -> Optional[NestedExitDecision]:
        if (
            not self.enable_probing
            or self.action_probe_rounds <= 0
            or self.action_episode_probe_count >= self.action_episode_probe_budget
            or not candidates
        ):
            return None

        def probe_rank(key: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
            global_count = self.action_utility_stats.get(key, ActionUtilityStat()).count
            episode_count = self.action_episode_probe_counts.get(key, 0)
            draft_layer, anchor_layer, speculation_length = key
            under_global = 0 if global_count < self.action_probe_rounds else 1
            under_episode = 0 if episode_count <= 0 else 1
            return (
                under_global,
                under_episode,
                global_count,
                speculation_length,
                draft_layer,
                anchor_layer,
            )

        ranked = sorted(candidates, key=probe_rank)
        key = ranked[0]
        if (
            self.action_utility_stats.get(key, ActionUtilityStat()).count >= self.action_probe_rounds
            and self.action_episode_probe_counts.get(key, 0) > 0
            and self.action_select_count % max(1, self.action_reprobe_interval or 1) != 0
        ):
            return None

        self.action_episode_probe_counts[key] = self.action_episode_probe_counts.get(key, 0) + 1
        self.action_episode_probe_count += 1
        return self._action_key_to_decision(
            key,
            mode="action_portfolio_probe",
            fallback_reason="online_action_probe",
        )

    def _select_action_portfolio_tpl(self, context_length: int) -> NestedExitDecision:
        self.action_select_count += 1
        self.adaptive_select_count += 1
        observed_layers = [
            layer
            for layer in self.direct_candidates
            if self._has_pair_observations(layer, self.target_layer)
        ]
        candidates = self._action_candidates(None)

        probe = self._select_action_probe(candidates)
        if probe is not None:
            return probe

        best = None
        scored_any = False
        for key in candidates:
            lcb = self._action_stat_lcb(key)
            if lcb is None:
                continue
            scored_any = True
            stat = self.action_utility_stats[key]
            mean = stat.mean() or lcb
            score = lcb
            if self.action_explore_bonus > 0.0:
                score += self.action_explore_bonus * mean / math.sqrt(stat.count + 1.0)
            draft_layer, anchor_layer, speculation_length = key
            if (
                self.action_last_key is not None
                and key != self.action_last_key
                and self.action_switch_penalty > 0.0
            ):
                score *= max(0.0, 1.0 - self.action_switch_penalty)
            rate_lcb = self._contextual_rate_lcb(
                draft_layer,
                self.target_layer,
                context_length,
                default=self._pair_rate(draft_layer, self.target_layer, 0.75),
            )
            model_score = self._direct_tpl_score(draft_layer, speculation_length, rate_lcb)
            candidate = (
                score,
                mean,
                model_score,
                rate_lcb,
                -draft_layer,
                anchor_layer == draft_layer,
                speculation_length,
                key,
            )
            if best is None or candidate > best:
                best = candidate

        if scored_any and best is not None:
            key = best[-1]
            mode = "action_portfolio_tpl"
            if self.action_aggressive_k > 0:
                recent_lcb = self._recent_predictability_lcb(default=None)
                if (
                    recent_lcb is not None
                    and recent_lcb >= self.action_aggressive_acceptance
                    and getattr(self, "feedback_bad_streak", 0) <= 0
                ):
                    draft_layer, anchor_layer, speculation_length = key
                    promoted_key = (
                        draft_layer,
                        anchor_layer,
                        max(int(speculation_length), self.action_aggressive_k),
                    )
                    layer_stat = self.pair_stats.get((draft_layer, self.target_layer))
                    layer_rate = (
                        layer_stat.accepted / layer_stat.total
                        if layer_stat is not None and layer_stat.total > 0
                        else 0.0
                    )
                    if promoted_key in candidates and promoted_key != key:
                        promoted_lcb = self._action_stat_lcb(promoted_key)
                        if (
                            promoted_lcb is not None
                            and layer_rate >= self.action_aggressive_min_layer_acceptance
                            and promoted_lcb >= (
                                float(best[0]) * self.action_aggressive_lcb_ratio
                            )
                        ):
                            key = promoted_key
                            mode = "action_portfolio_promote"
                            self.action_promotion_count += 1
            return self._action_key_to_decision(key, mode=mode)

        decision = self._select_feedback_adaptive_tpl(context_length=context_length)
        decision.mode = "action_portfolio_bootstrap"
        decision.fallback_reason = "no_measured_action_utility"
        return decision

    def _select_online_predictability_tpl(self, context_length: int) -> NestedExitDecision:
        self.adaptive_select_count += 1
        predictability_lcb = self._recent_predictability_lcb(default=self.feedback_min_acceptance)
        state = self._predictability_state(predictability_lcb)

        if self.enable_probing:
            under_probed = [
                layer
                for layer in self.direct_candidates
                if self.adaptive_probe_counts.get(layer, 0) < self.adaptive_probe_rounds
            ]
            if under_probed:
                draft_layer = under_probed[0]
                self.adaptive_probe_counts[draft_layer] = (
                    self.adaptive_probe_counts.get(draft_layer, 0) + 1
                )
                return NestedExitDecision(
                    target_layer=self.target_layer,
                    anchor_layer=draft_layer,
                    draft_layer=draft_layer,
                    speculation_length=self._throttle_probe_k(),
                    mode="online_predictability_probe",
                    fallback_reason="initial_predictability_probe",
                )

            if (
                self.adaptive_reprobe_interval > 0
                and self.direct_candidates
                and self.adaptive_select_count % self.adaptive_reprobe_interval == 0
            ):
                draft_layer = self.direct_candidates[
                    self.adaptive_reprobe_index % len(self.direct_candidates)
                ]
                self.adaptive_reprobe_index += 1
                self.adaptive_probe_counts[draft_layer] = (
                    self.adaptive_probe_counts.get(draft_layer, 0) + 1
                )
                return NestedExitDecision(
                    target_layer=self.target_layer,
                    anchor_layer=draft_layer,
                    draft_layer=draft_layer,
                    speculation_length=self._throttle_probe_k(),
                    mode="online_predictability_reprobe",
                    fallback_reason="periodic_predictability_reprobe",
                )

        observed_layers = [
            layer
            for layer in self.direct_candidates
            if self._has_pair_observations(layer, self.target_layer)
        ]
        if not observed_layers:
            draft_layer = self._next_direct_probe()
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=draft_layer,
                draft_layer=draft_layer,
                speculation_length=min(
                    self.feedback_min_speculation_length,
                    self._throttle_k_cap(),
                ),
                mode="online_predictability_probe",
                fallback_reason="no_predictability_stats",
            )

        deepest = max(observed_layers)
        k_cap = self._throttle_k_cap()
        layer_rate_lcbs = {}
        for draft_layer in sorted(observed_layers):
            layer_rate_lcbs[draft_layer] = self._contextual_rate_lcb(
                draft_layer,
                self.target_layer,
                context_length,
                default=self._pair_rate(draft_layer, self.target_layer, predictability_lcb),
            )
        best = None
        for draft_layer in sorted(observed_layers):
            rate_lcb = layer_rate_lcbs[draft_layer]
            if state == "aggressive":
                speculation_length = self._feedback_speculation_length(rate_lcb, max_k=k_cap)
            elif state == "balanced":
                speculation_length = min(
                    k_cap,
                    max(
                        min(self.feedback_min_speculation_length, k_cap),
                        int(round((k_cap + min(self.feedback_min_speculation_length, k_cap)) / 2.0)),
                    ),
                )
            else:
                speculation_length = min(self.feedback_min_speculation_length, k_cap)

            score = self._direct_tpl_score(draft_layer, speculation_length, rate_lcb)
            score *= self._throttle_score_multiplier(draft_layer, observed_layers)
            score *= self._layer_retire_score_multiplier(
                draft_layer,
                rate_lcb,
                layer_rate_lcbs,
            )
            if state == "conservative":
                if draft_layer < deepest:
                    score *= 0.80
                if self.feedback_bad_streak > 0:
                    score *= 0.90
            elif state == "aggressive":
                if draft_layer == min(observed_layers):
                    score *= 1.05

            if (
                self.feedback_last_layer is not None
                and draft_layer != self.feedback_last_layer
                and self.feedback_switch_penalty > 0.0
            ):
                score *= max(0.0, 1.0 - self.feedback_switch_penalty)
            candidate = (score, rate_lcb, -draft_layer, speculation_length, draft_layer)
            if best is None or candidate > best:
                best = candidate

        _, rate_lcb, _, speculation_length, draft_layer = best
        decision = NestedExitDecision(
            target_layer=self.target_layer,
            anchor_layer=draft_layer,
            draft_layer=draft_layer,
            speculation_length=speculation_length,
            mode=f"online_predictability_{state}",
        )
        if state == "conservative":
            decision.fallback_reason = "low_recent_predictability_lcb"
        elif state == "aggressive":
            decision.fallback_reason = "high_recent_predictability_lcb"
        else:
            decision.fallback_reason = "mid_recent_predictability_lcb"
        return decision

    def select(self, context_length: int) -> NestedExitDecision:
        if self.policy in {"fixed_anchor_tpl", "anchor_tpl", "safe_action_tpl"}:
            return self._select_fixed_anchor_tpl()
        if self.policy in {
            "online_action_portfolio_tpl",
            "action_portfolio_tpl",
            "portfolio_tpl",
        }:
            return self._select_action_portfolio_tpl(context_length=context_length)
        if self.policy in {"online_predictability_tpl", "predictability_tpl", "online_pa_tpl"}:
            return self._select_online_predictability_tpl(context_length=context_length)
        if self.policy in {"feedback_adaptive_tpl", "feedback_tpl", "contextual_feedback_tpl"}:
            return self._select_feedback_adaptive_tpl(context_length=context_length)
        if self.policy in {"layer_adaptive_tpl", "adaptive_tpl", "adaptive_direct_tpl"}:
            return self._select_layer_adaptive_tpl()
        if self.policy in {"direct_tpl", "tpl", "target_relative_tpl"}:
            return self._select_direct_tpl()

        reliable_anchors = []
        for anchor_layer in self.anchor_candidates:
            if not self._has_pair_observations(anchor_layer, self.target_layer):
                continue
            rate = self._pair_rate(anchor_layer, self.target_layer)
            if rate >= self.tau_anchor:
                reliable_anchors.append((anchor_layer, rate))

        if not reliable_anchors:
            anchor_layer = self._next_anchor_probe()
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=anchor_layer,
                draft_layer=anchor_layer,
                speculation_length=1,
                mode="anchor_probe",
                fallback_reason="no_reliable_anchor",
            )

        anchor_layer, _ = max(reliable_anchors, key=lambda item: (item[1], -item[0]))
        shallow_layers = [layer for layer in self.shallow_candidates if layer < anchor_layer]
        if not shallow_layers:
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=anchor_layer,
                draft_layer=anchor_layer,
                speculation_length=self.max_speculation_length,
                mode="anchor_only",
                fallback_reason="no_shallow_candidate_before_anchor",
            )

        valid_shallow = []
        for draft_layer in shallow_layers:
            if not self._has_pair_observations(draft_layer, anchor_layer):
                continue
            proxy_rate = self._pair_rate(draft_layer, anchor_layer)
            if proxy_rate >= self.tau_proxy:
                valid_shallow.append((draft_layer, proxy_rate))

        if not valid_shallow:
            if self.enable_probing:
                probe_layer = self._next_shallow_probe(anchor_layer)
                if probe_layer is not None:
                    return NestedExitDecision(
                        target_layer=self.target_layer,
                        anchor_layer=anchor_layer,
                        draft_layer=probe_layer,
                        speculation_length=1,
                        mode="nested_proxy_probe",
                        fallback_reason="no_proxy_qualified_shallow",
                    )
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=anchor_layer,
                draft_layer=anchor_layer,
                speculation_length=self.max_speculation_length,
                mode="anchor_only",
                fallback_reason="no_proxy_qualified_shallow",
            )

        draft_layer, _ = max(valid_shallow, key=lambda item: (item[1], -item[0]))
        if not self._has_pair_observations(draft_layer, self.target_layer):
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=anchor_layer,
                draft_layer=draft_layer,
                speculation_length=1,
                mode="nested_final_probe",
                fallback_reason="no_final_stats_for_shallow",
            )

        final_rate = self._pair_rate(draft_layer, self.target_layer)
        if final_rate >= self.tau_final and self._has_expected_gain(draft_layer, anchor_layer):
            return NestedExitDecision(
                target_layer=self.target_layer,
                anchor_layer=anchor_layer,
                draft_layer=draft_layer,
                speculation_length=self.max_speculation_length,
                mode="nested",
            )

        return NestedExitDecision(
            target_layer=self.target_layer,
            anchor_layer=anchor_layer,
            draft_layer=anchor_layer,
            speculation_length=self.max_speculation_length,
            mode="anchor_only",
            fallback_reason="shallow_candidate_failed_final_gate",
        )

    def _update_action_utility(
        self,
        decision: NestedExitDecision,
        latency_ms: Optional[float],
        output_tokens: int,
    ) -> None:
        if latency_ms is None or latency_ms <= 0.0 or output_tokens <= 0:
            return
        key = (
            int(decision.draft_layer),
            int(decision.anchor_layer),
            max(1, int(decision.speculation_length)),
        )
        utility = float(output_tokens) / float(latency_ms)
        stat = self.action_utility_stats.setdefault(key, ActionUtilityStat())
        stat.update(
            utility=utility,
            latency_ms=latency_ms,
            output_tokens=output_tokens,
            smoothing=self.utility_ema_decay,
        )
        self.action_last_key = key

    def update_after_step(
        self,
        step_id: int,
        context_length: int,
        decision: NestedExitDecision,
        num_drafted_tokens: int,
        num_accepted_by_final: int,
        proxy_accepted: int,
        anchor_accepted_by_final: int,
        latency_ms: Optional[float],
        output_tokens: int,
        energy_j: Optional[float] = None,
        gpu_power_w: Optional[float] = None,
        memory_clock_mhz: Optional[float] = None,
        temperature_c: Optional[float] = None,
    ) -> Dict:
        if num_drafted_tokens > 0:
            self._update_pair(
                decision.draft_layer,
                decision.target_layer,
                num_accepted_by_final,
                num_drafted_tokens,
            )
            self._update_context_pair(
                context_length,
                decision.draft_layer,
                decision.target_layer,
                num_accepted_by_final,
                num_drafted_tokens,
            )
            self._update_pair(
                decision.draft_layer,
                decision.anchor_layer,
                proxy_accepted,
                num_drafted_tokens,
            )
            self._update_context_pair(
                context_length,
                decision.draft_layer,
                decision.anchor_layer,
                proxy_accepted,
                num_drafted_tokens,
            )
            if decision.anchor_layer != decision.draft_layer:
                self._update_pair(
                    decision.anchor_layer,
                    decision.target_layer,
                    anchor_accepted_by_final,
                    num_drafted_tokens,
                )
                self._update_context_pair(
                    context_length,
                    decision.anchor_layer,
                    decision.target_layer,
                    anchor_accepted_by_final,
                    num_drafted_tokens,
                )
            recent = self.recent_layer_stats.setdefault(decision.draft_layer, deque(maxlen=8))
            recent.append((max(0, int(num_accepted_by_final)), max(1, int(num_drafted_tokens))))
            step_rate = max(0.0, min(1.0, num_accepted_by_final / max(1, num_drafted_tokens)))
            self.record_predictability_observation(
                num_accepted_by_final,
                num_drafted_tokens,
                source="tr",
                update_feedback=True,
            )
            self._update_throttle_state(
                step_rate=step_rate,
                num_accepted_by_final=num_accepted_by_final,
                num_drafted_tokens=num_drafted_tokens,
                latency_ms=latency_ms,
                output_tokens=output_tokens,
                draft_layer=decision.draft_layer,
            )
            self._update_action_utility(
                decision=decision,
                latency_ms=latency_ms,
                output_tokens=output_tokens,
            )
            if self.feedback_last_layer is not None and self.feedback_last_layer != decision.draft_layer:
                self.feedback_switch_count += 1
            self.feedback_last_layer = decision.draft_layer
        self.cost_stats.setdefault(decision.draft_layer, CostStat()).update(
            latency_ms, max(1, num_drafted_tokens)
        )
        self.mode_counts[decision.mode] = self.mode_counts.get(decision.mode, 0) + 1
        self.draft_layer_counts[decision.draft_layer] = (
            self.draft_layer_counts.get(decision.draft_layer, 0) + 1
        )
        self.speculation_length_counts[decision.speculation_length] = (
            self.speculation_length_counts.get(decision.speculation_length, 0) + 1
        )
        layer_stats = self.layer_acceptance.setdefault(
            decision.draft_layer, {"accepted": 0, "drafted": 0}
        )
        layer_stats["accepted"] += max(0, int(num_accepted_by_final))
        layer_stats["drafted"] += max(0, int(num_drafted_tokens))

        if not self.trace_path:
            return {}

        latency_per_token_ms = None
        if latency_ms is not None and output_tokens > 0:
            latency_per_token_ms = latency_ms / output_tokens
        energy_per_token_j = None
        if energy_j is not None and output_tokens > 0:
            energy_per_token_j = energy_j / output_tokens

        row = {
            "step_id": step_id,
            "context_length": context_length,
            "target_layer_L": decision.target_layer,
            "anchor_layer_e1": decision.anchor_layer,
            "draft_layer_e2": decision.draft_layer,
            "speculation_length_k": decision.speculation_length,
            "mode": decision.mode,
            "num_drafted_tokens": num_drafted_tokens,
            "num_accepted_tokens_by_final": num_accepted_by_final,
            "acceptance_rate_e2_L": self._pair_rate(decision.draft_layer, decision.target_layer),
            "proxy_acceptance_rate_e2_e1": self._pair_rate(decision.draft_layer, decision.anchor_layer),
            "anchor_acceptance_rate_e1_L": self._pair_rate(decision.anchor_layer, decision.target_layer),
            "latency_ms": latency_ms,
            "latency_per_token_ms": latency_per_token_ms,
            "energy_j": energy_j,
            "energy_per_token_j": energy_per_token_j,
            "gpu_power_w": gpu_power_w,
            "memory_clock_mhz": memory_clock_mhz,
            "temperature_c": temperature_c,
            "fallback_reason": decision.fallback_reason,
            "feedback_bad_streak": self.feedback_bad_streak,
            "feedback_good_streak": self.feedback_good_streak,
            "feedback_cooldown_remaining": self.feedback_cooldown_remaining,
            "throttle_enabled": self.throttle_enabled,
            "throttle_g": self.throttle_g,
            "throttle_k_cap": self._throttle_k_cap(),
            "throttle_last_delta": self.throttle_last_delta,
            "throttle_last_reason": self.throttle_last_reason,
            "throttle_last_intensity": self.throttle_last_intensity,
            "throttle_last_utility": self.throttle_last_utility,
            "throttle_last_utility_estimated": self.throttle_last_utility_estimated,
            "throttle_last_utility_ratio": self.throttle_last_utility_ratio,
            "throttle_prior_g": self.throttle_prior_g,
            "throttle_prior_updates": self.throttle_prior_updates,
            "throttle_episode_initial_g": self.throttle_episode_initial_g,
            "underconvert_enabled": self.underconvert_enabled,
            "underconvert_min_acceptance": self.underconvert_min_acceptance,
            "underconvert_min_short_k_fraction": self.underconvert_min_short_k_fraction,
            "underconvert_short_k": self.underconvert_short_k,
            "underconvert_boost": self.underconvert_boost,
            "underconvert_utility_floor": self.underconvert_utility_floor,
            "underconvert_recent_short_k_fraction": self._episode_recent_short_k_fraction(
                current_k=None
            ),
            "amortization_guard_enabled": self.amortization_guard_enabled,
            "amortization_min_steps": self.amortization_min_steps,
            "amortization_min_acceptance": self.amortization_min_acceptance,
            "amortization_min_short_k_fraction": self.amortization_min_short_k_fraction,
            "amortization_max_utility_ratio": self.amortization_max_utility_ratio,
            "amortization_cooldown_steps": self.amortization_cooldown_steps,
            "amortization_down_boost": self.amortization_down_boost,
            "amortization_bad_streak_min": self.amortization_bad_streak_min,
            "amortization_bad_streak": self.amortization_bad_streak,
            "amortization_long_guard_enabled": self.amortization_long_guard_enabled,
            "amortization_long_max_short_k_fraction": self.amortization_long_max_short_k_fraction,
            "amortization_long_min_k_cap": self.amortization_long_min_k_cap,
            "amortization_long_down_boost": self.amortization_long_down_boost,
            "amortization_long_bad_streak_min": self.amortization_long_bad_streak_min,
            "amortization_cooldown_remaining": self.amortization_cooldown_remaining,
            "amortization_trigger_count": self.amortization_trigger_count,
            "amortization_last_reason": self.amortization_last_reason,
            "amortization_last_short_k_fraction": self.amortization_last_short_k_fraction,
            "amortization_last_utility_ratio": self.amortization_last_utility_ratio,
            "action_k_candidates": list(self.action_k_candidates),
            "action_select_count": self.action_select_count,
            "action_episode_probe_budget": self.action_episode_probe_budget,
            "action_episode_probe_count": self.action_episode_probe_count,
            "action_last_key": (
                list(self.action_last_key) if self.action_last_key is not None else None
            ),
            "layer_retire_enabled": self.layer_retire_enabled,
            "layer_retire_min_total": self.layer_retire_min_total,
            "layer_retire_gap": self.layer_retire_gap,
            "layer_retire_penalty": self.layer_retire_penalty,
            "episode_memory_mode": self.episode_memory_mode,
            "episode_utility_ema": self.episode_utility_ema,
            "global_utility_ema": self.global_utility_ema,
            "episode_index": self.episode_index,
            "episode_step_count": self.episode_step_count,
            "episode_good_streak": self.episode_good_streak,
            "episode_bad_streak": self.episode_bad_streak,
            "episode_recent_acceptance": self._episode_recent_acceptance(),
            "episode_recent_utility": self._episode_recent_utility(),
        }
        self.append_trace(row)
        return row

    def append_trace(self, row: Dict) -> None:
        if not self.trace_path:
            return
        directory = os.path.dirname(self.trace_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with open(self.trace_path, "a", encoding="utf-8") as trace_file:
            trace_file.write(serialized + "\n")

    def summary(self) -> Dict:
        layer_acceptance = {}
        for layer, values in self.layer_acceptance.items():
            drafted = values.get("drafted", 0)
            accepted = values.get("accepted", 0)
            layer_acceptance[str(layer)] = {
                "accepted": accepted,
                "drafted": drafted,
                "acceptance_rate": accepted / drafted if drafted else None,
            }

        pair_rates = {}
        for (draft_layer, verifier_layer), stat in self.pair_stats.items():
            pair_rates[f"{draft_layer}->{verifier_layer}"] = {
                "accepted": stat.accepted,
                "total": stat.total,
                "rate": stat.value(None),
            }

        cost_stats = {}
        for layer, stat in self.cost_stats.items():
            cost_stats[str(layer)] = {
                "total_ms": stat.total_ms,
                "total_units": stat.total_units,
                "latency_per_unit_ms": stat.latency_per_unit(),
            }

        action_utility_stats = {}
        for (draft_layer, anchor_layer, speculation_length), stat in self.action_utility_stats.items():
            action_utility_stats[
                f"{draft_layer}->{anchor_layer}:k{speculation_length}"
            ] = {
                "count": stat.count,
                "mean_utility": stat.mean(),
                "ema_utility": stat.ema,
                "lcb_utility": stat.lcb(self.action_conservative_z),
                "total_latency_ms": stat.total_latency_ms,
                "total_output_tokens": stat.total_output_tokens,
            }

        context_pair_rates = {}
        for (bucket, draft_layer, verifier_layer), stat in self.context_pair_stats.items():
            context_pair_rates[f"b{bucket}:{draft_layer}->{verifier_layer}"] = {
                "accepted": stat.accepted,
                "total": stat.total,
                "rate": stat.value(None),
                "lcb": self._rate_lcb(stat, None),
            }

        return {
            "policy": self.policy,
            "target_layer": self.target_layer,
            "direct_candidates": list(self.direct_candidates),
            "layer_switch_margin": self.layer_switch_margin,
            "fixed_draft_layer": self.fixed_draft_layer,
            "fixed_anchor_layer": self.fixed_anchor_layer,
            "adaptive_probe_counts": {
                str(layer): count for layer, count in self.adaptive_probe_counts.items()
            },
            "adaptive_select_count": self.adaptive_select_count,
            "mode_counts": dict(self.mode_counts),
            "draft_layer_counts": {
                str(layer): count for layer, count in self.draft_layer_counts.items()
            },
            "speculation_length_counts": {
                str(length): count
                for length, count in self.speculation_length_counts.items()
            },
            "layer_acceptance": layer_acceptance,
            "pair_rates": pair_rates,
            "cost_stats": cost_stats,
            "action_k_candidates": list(self.action_k_candidates),
            "action_utility_stats": action_utility_stats,
            "action_probe_rounds": self.action_probe_rounds,
            "action_episode_probe_budget": self.action_episode_probe_budget,
            "action_reprobe_interval": self.action_reprobe_interval,
            "action_min_observations": self.action_min_observations,
            "action_conservative_z": self.action_conservative_z,
            "action_explore_bonus": self.action_explore_bonus,
            "action_switch_penalty": self.action_switch_penalty,
            "action_candidate_limit": self.action_candidate_limit,
            "action_anchor_enabled": self.action_anchor_enabled,
            "action_aggressive_k": self.action_aggressive_k,
            "action_aggressive_acceptance": self.action_aggressive_acceptance,
            "action_aggressive_lcb_ratio": self.action_aggressive_lcb_ratio,
            "action_aggressive_min_layer_acceptance": self.action_aggressive_min_layer_acceptance,
            "action_promotion_count": self.action_promotion_count,
            "action_select_count": self.action_select_count,
            "action_episode_probe_count": self.action_episode_probe_count,
            "action_last_key": (
                list(self.action_last_key) if self.action_last_key is not None else None
            ),
            "context_bucket_size": self.context_bucket_size,
            "context_pair_rates": context_pair_rates,
            "feedback_min_acceptance": self.feedback_min_acceptance,
            "feedback_good_acceptance": self.feedback_good_acceptance,
            "feedback_cooldown_steps": self.feedback_cooldown_steps,
            "feedback_min_speculation_length": self.feedback_min_speculation_length,
            "feedback_switch_penalty": self.feedback_switch_penalty,
            "recent_rate_mode": self.recent_rate_mode,
            "recent_rate_weight": self.recent_rate_weight,
            "episode_memory_mode": self.episode_memory_mode,
            "predictability_source_counts": dict(self.predictability_source_counts),
            "feedback_bad_streak": self.feedback_bad_streak,
            "feedback_good_streak": self.feedback_good_streak,
            "feedback_cooldown_remaining": self.feedback_cooldown_remaining,
            "feedback_switch_count": self.feedback_switch_count,
            "throttle_enabled": self.throttle_enabled,
            "throttle_initial": self.throttle_initial,
            "throttle_g": self.throttle_g,
            "throttle_k_cap": self._throttle_k_cap(),
            "throttle_min_k": self.throttle_min_k,
            "throttle_up_step": self.throttle_up_step,
            "throttle_down_step": self.throttle_down_step,
            "throttle_layer_bias": self.throttle_layer_bias,
            "throttle_updates": self.throttle_updates,
            "throttle_up_count": self.throttle_up_count,
            "throttle_down_count": self.throttle_down_count,
            "throttle_last_delta": self.throttle_last_delta,
            "throttle_last_reason": self.throttle_last_reason,
            "throttle_last_acceptance": self.throttle_last_acceptance,
            "throttle_last_reject_position": self.throttle_last_reject_position,
            "throttle_last_utility": self.throttle_last_utility,
            "throttle_last_utility_estimated": self.throttle_last_utility_estimated,
            "throttle_last_utility_ratio": self.throttle_last_utility_ratio,
            "throttle_last_intensity": self.throttle_last_intensity,
            "throttle_prior_g": self.throttle_prior_g,
            "throttle_prior_updates": self.throttle_prior_updates,
            "throttle_prior_weight": self.throttle_prior_weight,
            "throttle_episode_initial_g": self.throttle_episode_initial_g,
            "throttle_episode_resets": self.throttle_episode_resets,
            "underconvert_enabled": self.underconvert_enabled,
            "underconvert_min_acceptance": self.underconvert_min_acceptance,
            "underconvert_min_short_k_fraction": self.underconvert_min_short_k_fraction,
            "underconvert_short_k": self.underconvert_short_k,
            "underconvert_boost": self.underconvert_boost,
            "underconvert_utility_floor": self.underconvert_utility_floor,
            "underconvert_recent_short_k_fraction": self._episode_recent_short_k_fraction(
                current_k=None
            ),
            "amortization_guard_enabled": self.amortization_guard_enabled,
            "amortization_min_steps": self.amortization_min_steps,
            "amortization_min_acceptance": self.amortization_min_acceptance,
            "amortization_min_short_k_fraction": self.amortization_min_short_k_fraction,
            "amortization_max_utility_ratio": self.amortization_max_utility_ratio,
            "amortization_cooldown_steps": self.amortization_cooldown_steps,
            "amortization_down_boost": self.amortization_down_boost,
            "amortization_bad_streak_min": self.amortization_bad_streak_min,
            "amortization_bad_streak": self.amortization_bad_streak,
            "amortization_long_guard_enabled": self.amortization_long_guard_enabled,
            "amortization_long_max_short_k_fraction": self.amortization_long_max_short_k_fraction,
            "amortization_long_min_k_cap": self.amortization_long_min_k_cap,
            "amortization_long_down_boost": self.amortization_long_down_boost,
            "amortization_long_bad_streak_min": self.amortization_long_bad_streak_min,
            "amortization_cooldown_remaining": self.amortization_cooldown_remaining,
            "amortization_trigger_count": self.amortization_trigger_count,
            "amortization_last_reason": self.amortization_last_reason,
            "amortization_last_short_k_fraction": self.amortization_last_short_k_fraction,
            "amortization_last_utility_ratio": self.amortization_last_utility_ratio,
            "layer_retire_enabled": self.layer_retire_enabled,
            "layer_retire_min_total": self.layer_retire_min_total,
            "layer_retire_gap": self.layer_retire_gap,
            "layer_retire_penalty": self.layer_retire_penalty,
            "episode_utility_ema": self.episode_utility_ema,
            "global_utility_ema": self.global_utility_ema,
            "utility_ema_decay": self.utility_ema_decay,
            "episode_index": self.episode_index,
            "episode_step_count": self.episode_step_count,
            "episode_good_streak": self.episode_good_streak,
            "episode_bad_streak": self.episode_bad_streak,
            "episode_recent_acceptance": self._episode_recent_acceptance(),
            "episode_recent_utility": self._episode_recent_utility(),
            "recent_predictability_lcb": self._recent_predictability_lcb(default=None),
            "predictability_state": self._predictability_state(
                self._recent_predictability_lcb(default=self.feedback_min_acceptance)
            ),
            "controller_select_count": self.controller_select_count,
            "controller_select_time_s": self.controller_select_time_s,
            "controller_select_avg_us": (
                self.controller_select_time_s * 1_000_000.0 / self.controller_select_count
                if self.controller_select_count
                else None
            ),
        }
