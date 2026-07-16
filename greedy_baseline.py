#!/usr/bin/env python3
"""Forecast-aware one-step greedy baseline for the muni replay simulator.

The baseline scores every currently valid price/quantity action using only
point-in-time information available at the decision time.  It does *not* inspect
the next historical trade.  The action score is a one-step/receding-horizon
expected economic value:

    expected execution value
  + forecast value of expected remaining inventory
  - inventory carrying risk
  - target-schedule shortfall
  - quote smoothness/update costs

This provides a deployable myopic benchmark against the long-horizon neural
Q-policy.  It is intentionally not a trade-oracle benchmark.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hybrid_replay_simulator import (
    HybridMuniReplayEnv,
    ReplayDataset,
    ReplayEpisode,
    RewardConfig,
    SimulatorConfig,
)
from muni_cql_dueling_ddqn import ActionGrid, ActionSpec


@dataclass(frozen=True)
class GreedyBaselineConfig:
    """Configuration for the forecast-aware one-step action score."""

    planning_minutes: float = 30.0
    forward_change_feature_name: str = "forward_price_change_4h"
    forward_confidence_feature_name: str = "forward_model_confidence"
    forecast_weight: float = 1.0
    use_forecast_confidence: bool = True
    confidence_floor: float = 0.0
    confidence_cap: float = 1.0
    expected_fill_multiplier: float = 1.0
    include_inventory_penalty: bool = True
    include_schedule_penalty: bool = True
    include_smoothness_penalty: bool = True
    include_update_cost: bool = True
    tie_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.planning_minutes <= 0:
            raise ValueError("planning_minutes must be positive")
        if self.expected_fill_multiplier < 0:
            raise ValueError("expected_fill_multiplier must be nonnegative")
        if self.confidence_floor > self.confidence_cap:
            raise ValueError("confidence_floor cannot exceed confidence_cap")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance must be nonnegative")


@dataclass(frozen=True)
class GreedyDecision:
    action_id: int
    scores: np.ndarray
    selected_components: Mapping[str, float]
    all_components: Tuple[Mapping[str, float], ...]


class ForecastAwareGreedyPolicy:
    """Select the valid action with the highest one-step expected score."""

    def __init__(
        self,
        action_grid: Optional[ActionGrid] = None,
        config: Optional[GreedyBaselineConfig] = None,
    ) -> None:
        self.action_grid = action_grid or ActionGrid()
        self.config = config or GreedyBaselineConfig()

    def select_action(
        self,
        env: HybridMuniReplayEnv,
        action_mask: Optional[np.ndarray] = None,
        simulator_mode: Optional[str] = None,
    ) -> GreedyDecision:
        mask = env.action_mask() if action_mask is None else np.asarray(action_mask, dtype=bool)
        if mask.shape != (self.action_grid.num_actions,):
            raise ValueError("action_mask has the wrong shape")
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("No valid actions are available")

        scores = np.full(self.action_grid.num_actions, -np.inf, dtype=np.float64)
        components: List[Mapping[str, float]] = [dict() for _ in range(self.action_grid.num_actions)]
        for action_id in valid:
            item = self.score_action(env, int(action_id), simulator_mode=simulator_mode)
            scores[action_id] = item["total_score"]
            components[action_id] = item

        best_score = float(np.max(scores[valid]))
        candidates = valid[scores[valid] >= best_score - self.config.tie_tolerance]
        # Stable deterministic tie-break: prefer the action closest to the prior
        # quote, then the smaller absolute offset, then the lower action ID.
        selected = min(
            (int(action_id) for action_id in candidates),
            key=lambda action_id: self._tie_break_key(env, action_id),
        )
        return GreedyDecision(
            action_id=selected,
            scores=scores.astype(np.float32),
            selected_components=components[selected],
            all_components=tuple(components),
        )

    def score_action(
        self,
        env: HybridMuniReplayEnv,
        action_id: int,
        simulator_mode: Optional[str] = None,
    ) -> Dict[str, float]:
        preview = env.preview_action(action_id, simulator_mode=simulator_mode)
        spec = self.action_grid.decode(action_id)
        reward_cfg = env.reward_config
        sim_cfg = env.config
        starting_inventory = max(env.episode.starting_inventory, 1.0e-12)
        inventory = float(env.inventory)
        fair_mark = float(env.current_snapshot.fair_mark)

        expected_fill = float(
            np.clip(
                preview["expected_fill_quantity"] * self.config.expected_fill_multiplier,
                0.0,
                min(inventory, preview["offer_quantity"]),
            )
        )
        remaining = max(inventory - expected_fill, 0.0)

        offer_price = float(preview["offer_price"])
        execution_value = 0.0
        if not spec.is_no_quote and math.isfinite(offer_price):
            execution_value = (expected_fill / sim_cfg.price_notional_divisor) * (
                offer_price - fair_mark
            )

        raw_forward_change = float(
            env.current_snapshot.features.get(
                self.config.forward_change_feature_name,
                0.0,
            )
        )
        confidence = float(
            env.current_snapshot.features.get(
                self.config.forward_confidence_feature_name,
                1.0,
            )
        )
        confidence = float(
            np.clip(confidence, self.config.confidence_floor, self.config.confidence_cap)
        )
        confidence_multiplier = confidence if self.config.use_forecast_confidence else 1.0
        effective_forward_change = (
            self.config.forecast_weight * confidence_multiplier * raw_forward_change
        )
        forecast_inventory_value = (
            remaining / sim_cfg.price_notional_divisor
        ) * effective_forward_change

        interval_units = self.config.planning_minutes / sim_cfg.base_minutes
        risk_score = max(
            float(
                env.current_snapshot.features.get(
                    reward_cfg.risk_feature_name,
                    1.0,
                )
            ),
            0.0,
        )
        inventory_penalty = 0.0
        if self.config.include_inventory_penalty:
            inventory_penalty = (
                reward_cfg.inventory_lambda
                * (remaining / starting_inventory) ** 2
                * risk_score
                * interval_units
            )

        schedule_penalty = 0.0
        target_time = min(
            env.current_time + pd.Timedelta(minutes=self.config.planning_minutes),
            env.episode.end_time,
        )
        target_inventory = env.target_inventory_at(target_time)
        if self.config.include_schedule_penalty:
            schedule_shortfall = max(remaining - target_inventory, 0.0) / starting_inventory
            schedule_penalty = reward_cfg.schedule_lambda * schedule_shortfall**2 * interval_units

        previous_spec = (
            self.action_grid.decode(env.previous_action_id)
            if env.previous_action_id is not None
            else ActionSpec(-1, 0.0, 0.0, True)
        )
        smoothness_penalty = 0.0
        if self.config.include_smoothness_penalty and env.previous_action_id is not None:
            previous_dollar = (
                env.active_quote.price_offset_dollar if env.active_quote is not None else 0.0
            )
            current_dollar = float(preview["price_offset_dollar"])
            spread_scale = max(float(preview["effective_spread"]), 1.0e-6)
            smoothness_penalty = (
                reward_cfg.price_smooth_lambda
                * abs(
                    float(spec.price_offset_ratio or 0.0)
                    - float(previous_spec.price_offset_ratio or 0.0)
                )
                + reward_cfg.price_dollar_smooth_lambda
                * abs(current_dollar - previous_dollar)
                / spread_scale
                + reward_cfg.quantity_smooth_lambda
                * abs(float(spec.quantity_fraction) - float(previous_spec.quantity_fraction))
            )

        update_cost = 0.0
        if (
            self.config.include_update_cost
            and env.previous_action_id is not None
            and action_id != env.previous_action_id
        ):
            update_cost = reward_cfg.update_cost

        total_score = (
            execution_value
            + forecast_inventory_value
            - inventory_penalty
            - schedule_penalty
            - smoothness_penalty
            - update_cost
        )
        return {
            "action_id": float(action_id),
            "total_score": float(total_score),
            "expected_execution_value": float(execution_value),
            "forecast_inventory_value": float(forecast_inventory_value),
            "inventory_penalty": float(inventory_penalty),
            "schedule_penalty": float(schedule_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "update_cost": float(update_cost),
            "expected_fill_quantity": float(expected_fill),
            "expected_remaining_inventory": float(remaining),
            "offer_price": offer_price,
            "offer_quantity": float(preview["offer_quantity"]),
            "price_offset_ratio": float(preview["price_offset_ratio"]),
            "price_offset_dollar": float(preview["price_offset_dollar"]),
            "raw_price_offset_dollar": float(preview["raw_price_offset_dollar"]),
            "effective_spread": float(preview["effective_spread"]),
            "spread_unit": float(preview["spread_unit"]),
            "offset_was_clipped": float(preview["offset_was_clipped"]),
            "quantity_fraction": float(preview["quantity_fraction"]),
            "pretrade_win_probability": float(preview["win_probability"]),
            "pretrade_participation_share": float(preview["participation_share"]),
            "support_score": float(preview["support_score"]),
            "raw_forward_price_change": float(raw_forward_change),
            "forecast_confidence": float(confidence),
            "effective_forward_price_change": float(effective_forward_change),
            "target_inventory": float(target_inventory),
        }

    def _tie_break_key(self, env: HybridMuniReplayEnv, action_id: int) -> Tuple[float, float, float, int]:
        spec = self.action_grid.decode(action_id)
        if env.previous_action_id is not None:
            previous = self.action_grid.decode(env.previous_action_id)
            change = (
                abs(
                    float(spec.price_offset_ratio or 0.0)
                    - float(previous.price_offset_ratio or 0.0)
                )
                + abs(float(spec.quantity_fraction) - float(previous.quantity_fraction))
            )
        else:
            change = 0.0
        return (
            float(change),
            abs(float(spec.price_offset_ratio or 0.0)),
            float(spec.quantity_fraction),
            int(action_id),
        )


def evaluate_greedy_policy(
    dataset: ReplayDataset,
    episodes: Sequence[ReplayEpisode],
    models: Any,
    action_grid: ActionGrid,
    simulator_config: SimulatorConfig,
    reward_config: RewardConfig,
    greedy_config: GreedyBaselineConfig,
    modes: Sequence[str],
    max_episodes: int,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate the same greedy policy under several fill-simulator assumptions."""
    selected_episodes = list(episodes[: max_episodes if max_episodes > 0 else len(episodes)])
    policy = ForecastAwareGreedyPolicy(action_grid, greedy_config)
    mode_results: Dict[str, Any] = {}
    for mode in modes:
        cfg = dataclasses.replace(simulator_config, simulator_mode=mode, stochastic_fills=False)
        returns: List[float] = []
        ending_inventory_fraction: List[float] = []
        fill_fraction: List[float] = []
        decision_counts: List[int] = []
        action_counts = np.zeros(action_grid.num_actions, dtype=np.int64)
        selected_scores: List[float] = []
        for i, episode in enumerate(selected_episodes):
            env = HybridMuniReplayEnv(
                episode,
                dataset.state_schema,
                models,
                action_grid,
                cfg,
                reward_config,
                seed=seed + i,
            )
            _, mask, _ = env.reset()
            episode_return = 0.0
            while True:
                decision = policy.select_action(env, mask, simulator_mode=mode)
                action_counts[decision.action_id] += 1
                selected_scores.append(float(decision.selected_components["total_score"]))
                result = env.step(decision.action_id)
                episode_return += result.reward
                mask = result.action_mask
                if result.done:
                    break
            returns.append(episode_return)
            ending_inventory_fraction.append(env.inventory / episode.starting_inventory)
            fill_fraction.append(env.cumulative_fill / episode.starting_inventory)
            decision_counts.append(env.decision_count)
        mode_results[mode] = {
            "episodes": len(selected_episodes),
            "mean_return": float(np.mean(returns)) if returns else math.nan,
            "median_return": float(np.median(returns)) if returns else math.nan,
            "return_std": float(np.std(returns)) if returns else math.nan,
            "mean_ending_inventory_fraction": float(np.mean(ending_inventory_fraction)) if returns else math.nan,
            "mean_fill_fraction": float(np.mean(fill_fraction)) if returns else math.nan,
            "mean_decisions": float(np.mean(decision_counts)) if returns else math.nan,
            "mean_selected_one_step_score": float(np.mean(selected_scores)) if selected_scores else math.nan,
            "action_counts": action_counts.tolist(),
        }
    return mode_results


def compare_policy_metrics(
    rl_metrics: Mapping[str, Mapping[str, Any]],
    greedy_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Create compact RL-minus-greedy comparisons by simulator mode."""
    result: Dict[str, Any] = {}
    for mode in sorted(set(rl_metrics) & set(greedy_metrics)):
        rl = rl_metrics[mode]
        greedy = greedy_metrics[mode]
        result[mode] = {
            "mean_return_rl_minus_greedy": float(rl["mean_return"] - greedy["mean_return"]),
            "mean_fill_fraction_rl_minus_greedy": float(
                rl["mean_fill_fraction"] - greedy["mean_fill_fraction"]
            ),
            "mean_ending_inventory_fraction_rl_minus_greedy": float(
                rl["mean_ending_inventory_fraction"]
                - greedy["mean_ending_inventory_fraction"]
            ),
        }
    return result


__all__ = [
    "ForecastAwareGreedyPolicy",
    "GreedyBaselineConfig",
    "GreedyDecision",
    "compare_policy_metrics",
    "evaluate_greedy_policy",
]
