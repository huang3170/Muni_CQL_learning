#!/usr/bin/env python3
"""Simulator-online fine-tuning for the hybrid muni replay environment.

The current policy selects one quote at each decision trigger.  The simulator
then processes chronological market events, updates inventory, and appends the
resulting transition to a changing replay buffer.  An optional fixed historical
transition source can be mixed into each batch for offline-to-simulator-online
fine-tuning.  CQL strength can decay over training.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from hybrid_replay_simulator import (
    HybridMuniReplayEnv,
    ReplayDataset,
    ReplayEpisode,
    RewardConfig,
    SimulatorConfig,
    load_model_bundle,
)
from muni_cql_dueling_ddqn import (
    ActionGrid,
    CQLDuelingDoubleDQNTrainer,
    StateNormalizer,
    TrainConfig,
    TransitionArrayStore,
    atomic_torch_save,
    resolve_device,
    set_seed,
)

LOGGER = logging.getLogger("simulator_online")


@dataclass(frozen=True)
class OnlineConfig:
    seed: int = 2026
    train_episodes: int = 500
    replay_capacity: int = 500_000
    replay_warmup: int = 2_000
    batch_size: int = 512
    updates_per_step: int = 1
    train_every_steps: int = 1
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 100_000
    cql_alpha_start: float = 0.50
    cql_alpha_end: float = 0.05
    cql_decay_steps: int = 100_000
    historical_fraction_start: float = 0.50
    historical_fraction_end: float = 0.10
    historical_fraction_decay_steps: int = 100_000
    normalizer_episodes: int = 20
    normalizer_max_states: int = 100_000
    evaluation_interval_episodes: int = 25
    evaluation_max_episodes: int = 50
    evaluation_modes: Tuple[str, ...] = ("optimistic", "win_only", "partial")
    reward_scale: float = 1.0
    reward_clip: float = 10.0
    max_environment_steps: int = 0
    log_interval_steps: int = 250


class RingReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, num_actions: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.action_masks = np.empty((capacity, num_actions), dtype=bool)
        self.next_action_masks = np.empty((capacity, num_actions), dtype=bool)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
        discount: float,
    ) -> None:
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.action_masks[idx] = action_mask
        self.next_action_masks[idx] = next_action_mask
        self.discounts[idx] = discount
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if n <= 0:
            return _empty_array_batch(self.state_dim, self.num_actions)
        if self.size < n:
            raise ValueError(f"replay buffer has {self.size} rows, cannot sample {n}")
        idx = rng.integers(0, self.size, size=n)
        return {
            "states": self.states[idx].copy(),
            "actions": self.actions[idx].copy(),
            "rewards": self.rewards[idx].copy(),
            "next_states": self.next_states[idx].copy(),
            "dones": self.dones[idx].copy(),
            "action_masks": self.action_masks[idx].copy(),
            "next_action_masks": self.next_action_masks[idx].copy(),
            "discounts": self.discounts[idx].copy(),
        }


class FixedTransitionSource:
    """Uniform random sampler from a fixed offline transition store."""

    def __init__(self, path: str | Path, expected_state_dim: int, expected_num_actions: int) -> None:
        self.data = TransitionArrayStore(path)
        required = (
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
            "action_masks",
            "next_action_masks",
        )
        missing = [name for name in required if name not in self.data]
        if missing:
            raise ValueError(f"offline transition source is missing {missing}")
        self.n = len(self.data["states"])
        self.state_dim = int(self.data["states"].shape[1])
        self.num_actions = int(self.data["action_masks"].shape[1])
        if self.state_dim != expected_state_dim:
            raise ValueError(
                f"offline state_dim={self.state_dim}, simulator state_dim={expected_state_dim}"
            )
        if self.num_actions != expected_num_actions:
            raise ValueError(
                f"offline num_actions={self.num_actions}, simulator num_actions={expected_num_actions}"
            )

    def sample(self, n: int, rng: np.random.Generator, default_discount: float) -> Dict[str, np.ndarray]:
        if n <= 0:
            return _empty_array_batch(self.state_dim, self.num_actions)
        idx = rng.integers(0, self.n, size=n)
        discounts = (
            np.asarray(self.data["discounts"][idx], dtype=np.float32)
            if "discounts" in self.data
            else np.full(n, default_discount, dtype=np.float32)
        )
        return {
            "states": np.asarray(self.data["states"][idx], dtype=np.float32),
            "actions": np.asarray(self.data["actions"][idx], dtype=np.int64),
            "rewards": np.asarray(self.data["rewards"][idx], dtype=np.float32),
            "next_states": np.asarray(self.data["next_states"][idx], dtype=np.float32),
            "dones": np.asarray(self.data["dones"][idx], dtype=np.float32),
            "action_masks": np.asarray(self.data["action_masks"][idx], dtype=bool),
            "next_action_masks": np.asarray(self.data["next_action_masks"][idx], dtype=bool),
            "discounts": discounts,
        }


def _empty_array_batch(state_dim: int, num_actions: int) -> Dict[str, np.ndarray]:
    return {
        "states": np.empty((0, state_dim), dtype=np.float32),
        "actions": np.empty(0, dtype=np.int64),
        "rewards": np.empty(0, dtype=np.float32),
        "next_states": np.empty((0, state_dim), dtype=np.float32),
        "dones": np.empty(0, dtype=np.float32),
        "action_masks": np.empty((0, num_actions), dtype=bool),
        "next_action_masks": np.empty((0, num_actions), dtype=bool),
        "discounts": np.empty(0, dtype=np.float32),
    }


def concatenate_batches(*batches: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    names = (
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "action_masks",
        "next_action_masks",
        "discounts",
    )
    return {
        name: np.concatenate([batch[name] for batch in batches if len(batch[name]) > 0], axis=0)
        for name in names
    }


def to_tensor_batch(
    arrays: Mapping[str, np.ndarray],
    normalizer: StateNormalizer,
    reward_scale: float,
    reward_clip: float,
) -> Tuple[Tensor, ...]:
    states = normalizer.transform(np.asarray(arrays["states"], dtype=np.float32))
    next_states = normalizer.transform(np.asarray(arrays["next_states"], dtype=np.float32))
    rewards = np.asarray(arrays["rewards"], dtype=np.float32) / float(reward_scale)
    if reward_clip > 0:
        rewards = np.clip(rewards, -reward_clip, reward_clip)
    return (
        torch.from_numpy(states),
        torch.from_numpy(np.asarray(arrays["actions"], dtype=np.int64)),
        torch.from_numpy(rewards.astype(np.float32)),
        torch.from_numpy(next_states),
        torch.from_numpy(np.asarray(arrays["dones"], dtype=np.float32)),
        torch.from_numpy(np.asarray(arrays["action_masks"], dtype=bool)),
        torch.from_numpy(np.asarray(arrays["next_action_masks"], dtype=bool)),
        torch.from_numpy(np.asarray(arrays["discounts"], dtype=np.float32)),
    )


def fit_normalizer(states: np.ndarray, clip: float = 10.0) -> StateNormalizer:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or len(states) == 0:
        raise ValueError("normalizer states must be a nonempty [N, state_dim] matrix")
    if not np.isfinite(states).all():
        raise ValueError("normalizer states contain non-finite values")
    mean = states.mean(axis=0)
    std = states.std(axis=0, ddof=1 if len(states) > 1 else 0)
    std = np.maximum(std, 1e-6)
    return StateNormalizer(mean.astype(np.float32), std.astype(np.float32), clip=clip)


def schedule_linear(start: float, end: float, step: int, duration: int) -> float:
    if duration <= 0:
        return float(end)
    weight = min(max(step / duration, 0.0), 1.0)
    return float(start + weight * (end - start))


def select_epsilon_greedy_action(
    trainer: CQLDuelingDoubleDQNTrainer,
    normalizer: StateNormalizer,
    state: np.ndarray,
    action_mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    valid = np.flatnonzero(action_mask)
    if len(valid) == 0:
        raise RuntimeError("state has no valid actions")
    if rng.random() < epsilon:
        return int(rng.choice(valid))
    action, _ = trainer.select_action(normalizer.transform(state), action_mask)
    return action


def collect_normalizer_states(
    episodes: Sequence[ReplayEpisode],
    dataset: ReplayDataset,
    models: Any,
    action_grid: ActionGrid,
    simulator_config: SimulatorConfig,
    reward_config: RewardConfig,
    max_episodes: int,
    max_states: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    collected: List[np.ndarray] = []
    chosen = list(episodes[: max(1, min(max_episodes, len(episodes)))])
    for index, episode in enumerate(chosen):
        env = HybridMuniReplayEnv(
            episode,
            dataset.state_schema,
            models,
            action_grid,
            simulator_config,
            reward_config,
            seed=seed + index,
        )
        state, mask, _ = env.reset()
        while True:
            collected.append(state.copy())
            if len(collected) >= max_states:
                return np.stack(collected)
            action = int(rng.choice(np.flatnonzero(mask)))
            result = env.step(action)
            state, mask = result.state, result.action_mask
            if result.done:
                break
    return np.stack(collected)


def evaluate_policy(
    trainer: CQLDuelingDoubleDQNTrainer,
    normalizer: StateNormalizer,
    dataset: ReplayDataset,
    episodes: Sequence[ReplayEpisode],
    models: Any,
    action_grid: ActionGrid,
    simulator_config: SimulatorConfig,
    reward_config: RewardConfig,
    modes: Sequence[str],
    max_episodes: int,
    seed: int,
) -> Dict[str, Any]:
    selected_episodes = list(episodes[: max_episodes if max_episodes > 0 else len(episodes)])
    mode_results: Dict[str, Any] = {}
    for mode in modes:
        cfg = dataclasses.replace(simulator_config, simulator_mode=mode, stochastic_fills=False)
        returns: List[float] = []
        ending_inventory_fraction: List[float] = []
        fill_fraction: List[float] = []
        decision_counts: List[int] = []
        action_counts = np.zeros(action_grid.num_actions, dtype=np.int64)
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
            state, mask, _ = env.reset()
            episode_return = 0.0
            while True:
                action, _ = trainer.select_action(normalizer.transform(state), mask)
                action_counts[action] += 1
                result = env.step(action)
                episode_return += result.reward
                state, mask = result.state, result.action_mask
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
            "action_counts": action_counts.tolist(),
        }
    return mode_results


def load_warmstart(
    checkpoint_path: Path,
    state_dim: int,
    num_actions: int,
    config: TrainConfig,
    device: torch.device,
) -> Tuple[CQLDuelingDoubleDQNTrainer, StateNormalizer, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["state_dim"]) != state_dim:
        raise ValueError(
            f"warm-start state_dim={checkpoint['state_dim']} but simulator state_dim={state_dim}"
        )
    if int(checkpoint["num_actions"]) != num_actions:
        raise ValueError(
            f"warm-start num_actions={checkpoint['num_actions']} but simulator num_actions={num_actions}"
        )
    trainer = CQLDuelingDoubleDQNTrainer(state_dim, num_actions, config, device)
    trainer.online.load_state_dict(checkpoint["model_state_dict"])
    trainer.target.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
    normalizer = StateNormalizer.from_dict(checkpoint["normalizer"])
    return trainer, normalizer, checkpoint


def checkpoint_payload(
    trainer: CQLDuelingDoubleDQNTrainer,
    normalizer: StateNormalizer,
    action_grid: ActionGrid,
    state_feature_names: Sequence[str],
    train_config: TrainConfig,
    online_config: OnlineConfig,
    simulator_config: SimulatorConfig,
    reward_config: RewardConfig,
    evaluation: Mapping[str, Any],
    episode_index: int,
    environment_steps: int,
) -> Dict[str, Any]:
    return {
        "model_state_dict": trainer.online.state_dict(),
        "target_state_dict": trainer.target.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "state_dim": len(state_feature_names),
        "num_actions": action_grid.num_actions,
        "normalizer": normalizer.to_dict(),
        "action_grid": action_grid.to_json_dict(),
        "state_feature_names": list(state_feature_names),
        "config": asdict(train_config),
        "online_config": asdict(online_config),
        "simulator_config": asdict(simulator_config),
        "reward_config": asdict(reward_config),
        "evaluation": dict(evaluation),
        "episode_index": episode_index,
        "environment_steps": environment_steps,
        "global_step": trainer.global_step,
    }


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    train_data = ReplayDataset.from_directory(args.train_replay_dir)
    valid_data = ReplayDataset.from_directory(args.valid_replay_dir)
    models = load_model_bundle(args.model_factory)
    action_grid = ActionGrid()
    simulator_config = SimulatorConfig(
        clock_minutes=args.clock_minutes,
        base_minutes=args.base_minutes,
        base_discount=args.gamma,
        min_lot=args.min_lot,
        price_notional_divisor=args.price_notional_divisor,
        simulator_mode=args.simulator_mode,
        quantity_model_definition=args.quantity_model_definition,
        stochastic_fills=not args.deterministic_fills,
        allowed_trade_types=tuple(value.upper() for value in args.allowed_trade_types),
        trigger_on_same_cusip_publish=not args.no_trade_publish_trigger,
        trigger_on_own_fill=not args.no_own_fill_trigger,
        min_quote_life_minutes=args.min_quote_life_minutes,
        min_price_delta_from_mark=args.min_price_delta_from_mark,
        max_price_delta_from_mark=args.max_price_delta_from_mark,
        support_threshold=args.support_threshold,
        max_decisions_per_episode=args.max_decisions_per_episode,
    )
    reward_config = RewardConfig(
        inventory_lambda=args.inventory_lambda,
        schedule_lambda=args.schedule_lambda,
        price_smooth_lambda=args.price_smooth_lambda,
        quantity_smooth_lambda=args.quantity_smooth_lambda,
        update_cost=args.update_cost,
        missed_demand_lambda=args.missed_demand_lambda,
        underpricing_lambda=args.underpricing_lambda,
        underpricing_tolerance=args.underpricing_tolerance,
        terminal_lambda=args.terminal_lambda,
        liquidation_concession=args.liquidation_concession,
    )
    online_config = OnlineConfig(
        seed=args.seed,
        train_episodes=args.train_episodes,
        replay_capacity=args.replay_capacity,
        replay_warmup=args.replay_warmup,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        train_every_steps=args.train_every_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        cql_alpha_start=args.cql_alpha_start,
        cql_alpha_end=args.cql_alpha_end,
        cql_decay_steps=args.cql_decay_steps,
        historical_fraction_start=args.historical_fraction_start,
        historical_fraction_end=args.historical_fraction_end,
        historical_fraction_decay_steps=args.historical_fraction_decay_steps,
        normalizer_episodes=args.normalizer_episodes,
        normalizer_max_states=args.normalizer_max_states,
        evaluation_interval_episodes=args.evaluation_interval_episodes,
        evaluation_max_episodes=args.evaluation_max_episodes,
        evaluation_modes=tuple(args.evaluation_modes),
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        max_environment_steps=args.max_environment_steps,
        log_interval_steps=args.log_interval_steps,
    )
    train_config = TrainConfig(
        seed=args.seed,
        device=args.device,
        epochs=1,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        cql_alpha=args.cql_alpha_start,
        cql_temperature=args.cql_temperature,
        target_update_interval=args.target_update_interval,
        gradient_clip_norm=args.gradient_clip_norm,
        trunk_dims=tuple(args.trunk_dims),
        head_dim=args.head_dim,
        torch_num_threads=args.torch_num_threads,
    )

    probe_env = HybridMuniReplayEnv(
        train_data.episodes[0],
        train_data.state_schema,
        models,
        action_grid,
        simulator_config,
        reward_config,
        seed=args.seed,
    )
    probe_state, _, _ = probe_env.reset()
    state_dim = len(probe_state)
    state_feature_names = probe_env.state_feature_names

    if args.warmstart_checkpoint:
        trainer, normalizer, warm_checkpoint = load_warmstart(
            Path(args.warmstart_checkpoint), state_dim, action_grid.num_actions, train_config, device
        )
        checkpoint_names = warm_checkpoint.get("state_feature_names")
        if checkpoint_names is not None and list(checkpoint_names) != list(state_feature_names):
            raise ValueError("warm-start state_feature_names do not match simulator schema")
        LOGGER.info("loaded warm-start checkpoint %s", args.warmstart_checkpoint)
    else:
        LOGGER.info("collecting random-policy states for normalization")
        normalizer_states = collect_normalizer_states(
            train_data.episodes,
            train_data,
            models,
            action_grid,
            simulator_config,
            reward_config,
            max_episodes=online_config.normalizer_episodes,
            max_states=online_config.normalizer_max_states,
            seed=args.seed + 10_000,
        )
        normalizer = fit_normalizer(normalizer_states, clip=args.normalizer_clip)
        trainer = CQLDuelingDoubleDQNTrainer(
            state_dim=state_dim,
            num_actions=action_grid.num_actions,
            config=train_config,
            device=device,
        )

    replay = RingReplayBuffer(online_config.replay_capacity, state_dim, action_grid.num_actions)
    offline_source = (
        FixedTransitionSource(args.offline_transitions, state_dim, action_grid.num_actions)
        if args.offline_transitions
        else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "state_feature_names.json").write_text(
        json.dumps(list(state_feature_names), indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "train_config": asdict(train_config),
                "online_config": asdict(online_config),
                "simulator_config": asdict(simulator_config),
                "reward_config": asdict(reward_config),
                "action_grid": action_grid.to_json_dict(),
                "state_dim": state_dim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_path = output_dir / "training_log.jsonl"
    best_path = output_dir / "best_simulator_online_checkpoint.pt"
    latest_path = output_dir / "latest_simulator_online_checkpoint.pt"
    best_partial_return = -math.inf
    environment_steps = 0
    update_metrics_recent: List[Mapping[str, float]] = []
    shuffled_episodes = list(train_data.episodes)
    rng.shuffle(shuffled_episodes)
    start_wall = time.time()

    for episode_index in range(1, online_config.train_episodes + 1):
        episode = shuffled_episodes[(episode_index - 1) % len(shuffled_episodes)]
        if (episode_index - 1) % len(shuffled_episodes) == 0 and episode_index > 1:
            rng.shuffle(shuffled_episodes)
        env = HybridMuniReplayEnv(
            episode,
            train_data.state_schema,
            models,
            action_grid,
            simulator_config,
            reward_config,
            seed=args.seed + episode_index,
        )
        state, mask, _ = env.reset()
        episode_return = 0.0
        episode_steps = 0
        while True:
            epsilon = schedule_linear(
                online_config.epsilon_start,
                online_config.epsilon_end,
                environment_steps,
                online_config.epsilon_decay_steps,
            )
            action = select_epsilon_greedy_action(
                trainer, normalizer, state, mask, epsilon, rng
            )
            result = env.step(action)
            replay.add(
                state,
                action,
                result.reward,
                result.state,
                result.done,
                mask,
                result.action_mask,
                result.discount,
            )
            state, mask = result.state, result.action_mask
            episode_return += result.reward
            episode_steps += 1
            environment_steps += 1

            if (
                len(replay) >= max(online_config.replay_warmup, 1)
                and environment_steps % online_config.train_every_steps == 0
            ):
                for _ in range(online_config.updates_per_step):
                    historical_fraction = (
                        schedule_linear(
                            online_config.historical_fraction_start,
                            online_config.historical_fraction_end,
                            environment_steps,
                            online_config.historical_fraction_decay_steps,
                        )
                        if offline_source is not None
                        else 0.0
                    )
                    offline_n = int(round(online_config.batch_size * historical_fraction))
                    simulator_n = online_config.batch_size - offline_n
                    simulator_n = min(simulator_n, len(replay))
                    offline_n = online_config.batch_size - simulator_n if offline_source else 0
                    sim_batch = replay.sample(simulator_n, rng)
                    if offline_source is not None and offline_n > 0:
                        offline_batch = offline_source.sample(offline_n, rng, args.gamma)
                        arrays = concatenate_batches(sim_batch, offline_batch)
                    else:
                        arrays = sim_batch
                    permutation = rng.permutation(len(arrays["actions"]))
                    arrays = {name: value[permutation] for name, value in arrays.items()}
                    tensor_batch = to_tensor_batch(
                        arrays,
                        normalizer,
                        reward_scale=online_config.reward_scale,
                        reward_clip=online_config.reward_clip,
                    )
                    cql_alpha = schedule_linear(
                        online_config.cql_alpha_start,
                        online_config.cql_alpha_end,
                        environment_steps,
                        online_config.cql_decay_steps,
                    )
                    update_metrics = trainer.update_batch(tensor_batch, cql_alpha=cql_alpha)
                    update_metrics_recent.append(update_metrics)
                    if len(update_metrics_recent) > 100:
                        update_metrics_recent.pop(0)

            if environment_steps % online_config.log_interval_steps == 0:
                mean_loss = (
                    float(np.mean([m["total_loss"] for m in update_metrics_recent]))
                    if update_metrics_recent
                    else math.nan
                )
                LOGGER.info(
                    "env_steps=%d episode=%d replay=%d epsilon=%.3f return=%.4f "
                    "inventory=%.2f mean_loss=%.5f",
                    environment_steps,
                    episode_index,
                    len(replay),
                    epsilon,
                    episode_return,
                    env.inventory,
                    mean_loss,
                )

            if result.done:
                break
            if online_config.max_environment_steps and environment_steps >= online_config.max_environment_steps:
                break

        episode_record = {
            "type": "train_episode",
            "episode_index": episode_index,
            "episode_id": episode.episode_id,
            "environment_steps": environment_steps,
            "episode_steps": episode_steps,
            "episode_return": episode_return,
            "ending_inventory_fraction": env.inventory / episode.starting_inventory,
            "fill_fraction": env.cumulative_fill / episode.starting_inventory,
            "decisions": env.decision_count,
            "replay_size": len(replay),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode_record) + "\n")

        should_evaluate = (
            episode_index % online_config.evaluation_interval_episodes == 0
            or episode_index == online_config.train_episodes
        )
        if should_evaluate:
            evaluation = evaluate_policy(
                trainer,
                normalizer,
                valid_data,
                valid_data.episodes,
                models,
                action_grid,
                simulator_config,
                reward_config,
                online_config.evaluation_modes,
                online_config.evaluation_max_episodes,
                seed=args.seed + 50_000,
            )
            record = {
                "type": "evaluation",
                "episode_index": episode_index,
                "environment_steps": environment_steps,
                "evaluation": evaluation,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            payload = checkpoint_payload(
                trainer,
                normalizer,
                action_grid,
                state_feature_names,
                train_config,
                online_config,
                simulator_config,
                reward_config,
                evaluation,
                episode_index,
                environment_steps,
            )
            atomic_torch_save(payload, latest_path)
            partial_return = evaluation.get("partial", {}).get("mean_return", -math.inf)
            if math.isfinite(partial_return) and partial_return > best_partial_return:
                best_partial_return = partial_return
                atomic_torch_save(payload, best_path)
            LOGGER.info("evaluation episode=%d %s", episode_index, json.dumps(evaluation))

        if online_config.max_environment_steps and environment_steps >= online_config.max_environment_steps:
            break

    if not latest_path.exists():
        evaluation = evaluate_policy(
            trainer,
            normalizer,
            valid_data,
            valid_data.episodes,
            models,
            action_grid,
            simulator_config,
            reward_config,
            online_config.evaluation_modes,
            online_config.evaluation_max_episodes,
            seed=args.seed + 50_000,
        )
        payload = checkpoint_payload(
            trainer,
            normalizer,
            action_grid,
            state_feature_names,
            train_config,
            online_config,
            simulator_config,
            reward_config,
            evaluation,
            episode_index,
            environment_steps,
        )
        atomic_torch_save(payload, latest_path)
        partial_return = evaluation.get("partial", {}).get("mean_return", -math.inf)
        if math.isfinite(partial_return):
            best_partial_return = partial_return
            atomic_torch_save(payload, best_path)

    summary = {
        "environment_steps": environment_steps,
        "optimizer_steps": trainer.global_step,
        "replay_size": len(replay),
        "best_partial_mean_return": best_partial_return,
        "wall_seconds": time.time() - start_wall,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("simulator-online training complete: %s", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-replay-dir", required=True)
    parser.add_argument("--valid-replay-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offline-transitions", help="Optional fixed NPZ/NPY transition source")
    parser.add_argument("--warmstart-checkpoint", help="Optional offline CQL checkpoint")
    parser.add_argument("--model-factory", help="module:function returning calibrated simulator models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--torch-num-threads", type=int, default=8)

    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--max-environment-steps", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--replay-warmup", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--train-every-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--trunk-dims", type=int, nargs="+", default=(512, 256))
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--normalizer-clip", type=float, default=10.0)
    parser.add_argument("--normalizer-episodes", type=int, default=20)
    parser.add_argument("--normalizer-max-states", type=int, default=100_000)

    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--cql-alpha-start", type=float, default=0.50)
    parser.add_argument("--cql-alpha-end", type=float, default=0.05)
    parser.add_argument("--cql-decay-steps", type=int, default=100_000)
    parser.add_argument("--cql-temperature", type=float, default=1.0)
    parser.add_argument("--historical-fraction-start", type=float, default=0.50)
    parser.add_argument("--historical-fraction-end", type=float, default=0.10)
    parser.add_argument("--historical-fraction-decay-steps", type=int, default=100_000)

    parser.add_argument("--clock-minutes", type=float, default=30.0)
    parser.add_argument("--base-minutes", type=float, default=30.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--min-lot", type=float, default=5.0)
    parser.add_argument("--price-notional-divisor", type=float, default=100.0)
    parser.add_argument("--simulator-mode", choices=("optimistic", "win_only", "partial"), default="partial")
    parser.add_argument(
        "--quantity-model-definition",
        choices=("participation_share", "displayed_quote_ratio"),
        default="participation_share",
    )
    parser.add_argument("--deterministic-fills", action="store_true")
    parser.add_argument("--allowed-trade-types", nargs="+", default=["S"])
    parser.add_argument("--no-trade-publish-trigger", action="store_true")
    parser.add_argument("--no-own-fill-trigger", action="store_true")
    parser.add_argument("--min-quote-life-minutes", type=float, default=0.0)
    parser.add_argument("--min-price-delta-from-mark", type=float, default=-2.0)
    parser.add_argument("--max-price-delta-from-mark", type=float, default=2.0)
    parser.add_argument("--support-threshold", type=float, default=0.0)
    parser.add_argument("--max-decisions-per-episode", type=int, default=500)

    parser.add_argument("--inventory-lambda", type=float, default=0.05)
    parser.add_argument("--schedule-lambda", type=float, default=0.10)
    parser.add_argument("--price-smooth-lambda", type=float, default=0.01)
    parser.add_argument("--quantity-smooth-lambda", type=float, default=0.01)
    parser.add_argument("--update-cost", type=float, default=0.001)
    parser.add_argument("--missed-demand-lambda", type=float, default=0.0)
    parser.add_argument("--underpricing-lambda", type=float, default=0.0)
    parser.add_argument("--underpricing-tolerance", type=float, default=0.125)
    parser.add_argument("--terminal-lambda", type=float, default=1.0)
    parser.add_argument("--liquidation-concession", type=float, default=0.50)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=10.0)

    parser.add_argument("--evaluation-interval-episodes", type=int, default=25)
    parser.add_argument("--evaluation-max-episodes", type=int, default=50)
    parser.add_argument(
        "--evaluation-modes",
        nargs="+",
        default=["optimistic", "win_only", "partial"],
        choices=("optimistic", "win_only", "partial"),
    )
    parser.add_argument("--log-interval-steps", type=int, default=250)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    run_training(args)


if __name__ == "__main__":
    main()
