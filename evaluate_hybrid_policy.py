#!/usr/bin/env python3
"""Evaluate RL and forecast-aware greedy policies in the hybrid replay simulator."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

from greedy_baseline import (
    ForecastAwareGreedyPolicy,
    GreedyBaselineConfig,
    compare_policy_metrics,
    evaluate_greedy_policy,
)
from hybrid_replay_simulator import (
    HybridMuniReplayEnv,
    ReplayDataset,
    RewardConfig,
    SimulatorConfig,
    load_model_bundle,
)
from muni_cql_dueling_ddqn import (
    ActionGrid,
    CQLDuelingDoubleDQNTrainer,
    StateNormalizer,
    TrainConfig,
    resolve_device,
)
from simulator_online_train import evaluate_policy

LOGGER = logging.getLogger("evaluate_hybrid")


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config_payload = dict(checkpoint["config"])
    config_payload["trunk_dims"] = tuple(config_payload["trunk_dims"])
    train_config = TrainConfig(**config_payload)
    trainer = CQLDuelingDoubleDQNTrainer(
        int(checkpoint["state_dim"]), int(checkpoint["num_actions"]), train_config, device
    )
    trainer.online.load_state_dict(checkpoint["model_state_dict"])
    trainer.target.load_state_dict(checkpoint["target_state_dict"])
    normalizer = StateNormalizer.from_dict(checkpoint["normalizer"])
    grid_payload = checkpoint.get("action_grid", {})
    action_grid = ActionGrid(
        price_offsets=tuple(grid_payload.get("price_offsets", ActionGrid().price_offsets)),
        quantity_fractions=tuple(
            grid_payload.get("quantity_fractions", ActionGrid().quantity_fractions)
        ),
        include_no_quote=bool(grid_payload.get("include_no_quote", True)),
    )
    sim_cfg = SimulatorConfig(**checkpoint.get("simulator_config", {}))
    reward_cfg = RewardConfig(**checkpoint.get("reward_config", {}))
    return trainer, normalizer, action_grid, sim_cfg, reward_cfg, checkpoint


def replay_episodes(
    policy_name: str,
    dataset: ReplayDataset,
    models: Any,
    action_grid: ActionGrid,
    simulator_config: SimulatorConfig,
    reward_config: RewardConfig,
    output_path: Path,
    max_episodes: int,
    mode: str,
    seed: int,
    trainer: Optional[CQLDuelingDoubleDQNTrainer] = None,
    normalizer: Optional[StateNormalizer] = None,
    greedy_config: Optional[GreedyBaselineConfig] = None,
) -> None:
    cfg = dataclasses.replace(simulator_config, simulator_mode=mode, stochastic_fills=False)
    greedy = ForecastAwareGreedyPolicy(action_grid, greedy_config) if policy_name == "greedy" else None
    rows: List[Dict[str, Any]] = []
    for episode_index, episode in enumerate(dataset.episodes[:max_episodes]):
        env = HybridMuniReplayEnv(
            episode,
            dataset.state_schema,
            models,
            action_grid,
            cfg,
            reward_config,
            seed=seed + episode_index,
        )
        state, mask, _ = env.reset()
        step_index = 0
        while True:
            if policy_name == "rl":
                if trainer is None or normalizer is None:
                    raise ValueError("RL replay requires trainer and normalizer")
                action_id, action_values = trainer.select_action(normalizer.transform(state), mask)
                selected_value = float(max(action_values))
                greedy_components: Dict[str, float] = {}
            elif policy_name == "greedy":
                assert greedy is not None
                decision = greedy.select_action(env, mask, simulator_mode=mode)
                action_id = decision.action_id
                selected_value = float(decision.selected_components["total_score"])
                greedy_components = dict(decision.selected_components)
            else:
                raise ValueError(f"Unknown policy_name={policy_name}")

            spec = action_grid.decode(action_id)
            result = env.step(action_id)
            step_index += 1
            row: Dict[str, Any] = {
                "policy": policy_name,
                "episode_id": episode.episode_id,
                "cusip": episode.cusip,
                "step": step_index,
                "decision_time": result.info["interval_start"],
                "next_decision_time": result.info["interval_end"],
                "elapsed_minutes": result.elapsed_minutes,
                "action_id": action_id,
                "price_offset": spec.price_offset,
                "quantity_fraction": spec.quantity_fraction,
                "is_no_quote": spec.is_no_quote,
                "policy_action_value": selected_value,
                "reward": result.reward,
                "inventory_before": result.info["inventory_before_interval"],
                "inventory_after": result.info["inventory_after_interval"],
                "trigger_reasons": "|".join(result.info["trigger_reasons"]),
                "fill_quantity": sum(item["filled_quantity"] for item in result.info["fills"]),
                "done": result.done,
            }
            for key, value in greedy_components.items():
                row[f"greedy_{key}"] = value
            rows.append(row)
            state, mask = result.state, result.action_mask
            if result.done:
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def greedy_config_from_args(args: argparse.Namespace) -> GreedyBaselineConfig:
    return GreedyBaselineConfig(
        planning_minutes=args.greedy_planning_minutes,
        forward_change_feature_name=args.greedy_forward_change_feature,
        forward_confidence_feature_name=args.greedy_forward_confidence_feature,
        forecast_weight=args.greedy_forecast_weight,
        use_forecast_confidence=not args.greedy_ignore_forecast_confidence,
        confidence_floor=args.greedy_confidence_floor,
        confidence_cap=args.greedy_confidence_cap,
        expected_fill_multiplier=args.greedy_expected_fill_multiplier,
        include_inventory_penalty=not args.greedy_no_inventory_penalty,
        include_schedule_penalty=not args.greedy_no_schedule_penalty,
        include_smoothness_penalty=not args.greedy_no_smoothness_penalty,
        include_update_cost=not args.greedy_no_update_cost,
    )


def add_greedy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--greedy-planning-minutes", type=float, default=30.0)
    parser.add_argument(
        "--greedy-forward-change-feature", default="forward_price_change_4h"
    )
    parser.add_argument(
        "--greedy-forward-confidence-feature", default="forward_model_confidence"
    )
    parser.add_argument("--greedy-forecast-weight", type=float, default=1.0)
    parser.add_argument("--greedy-ignore-forecast-confidence", action="store_true")
    parser.add_argument("--greedy-confidence-floor", type=float, default=0.0)
    parser.add_argument("--greedy-confidence-cap", type=float, default=1.0)
    parser.add_argument("--greedy-expected-fill-multiplier", type=float, default=1.0)
    parser.add_argument("--greedy-no-inventory-penalty", action="store_true")
    parser.add_argument("--greedy-no-schedule-penalty", action="store_true")
    parser.add_argument("--greedy-no-smoothness-penalty", action="store_true")
    parser.add_argument("--greedy-no-update-cost", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Required when evaluating the RL policy")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--model-factory", help="module:function returning simulator models")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("rl", "greedy"),
        default=["rl", "greedy"],
        help="Policies to evaluate. The greedy policy needs no checkpoint.",
    )
    parser.add_argument("--replay-csv")
    parser.add_argument("--replay-policy", choices=("rl", "greedy"), default="rl")
    parser.add_argument("--replay-csv-episodes", type=int, default=5)
    parser.add_argument(
        "--replay-csv-mode", choices=("optimistic", "win_only", "partial"), default="partial"
    )
    parser.add_argument("--evaluation-max-episodes", type=int, default=100)
    parser.add_argument(
        "--evaluation-modes", nargs="+", default=["optimistic", "win_only", "partial"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    add_greedy_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    device = resolve_device(args.device)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    dataset = ReplayDataset.from_directory(args.replay_dir)
    models = load_model_bundle(args.model_factory)
    requested = list(dict.fromkeys(args.policies))
    if "rl" in requested and not args.checkpoint:
        raise ValueError("--checkpoint is required when --policies includes rl")

    trainer: Optional[CQLDuelingDoubleDQNTrainer] = None
    normalizer: Optional[StateNormalizer] = None
    checkpoint: Optional[Dict[str, Any]] = None
    if args.checkpoint:
        trainer, normalizer, action_grid, sim_cfg, reward_cfg, checkpoint = load_checkpoint(
            Path(args.checkpoint), device
        )
    else:
        action_grid = ActionGrid()
        sim_cfg = SimulatorConfig()
        reward_cfg = RewardConfig()

    probe = HybridMuniReplayEnv(
        dataset.episodes[0], dataset.state_schema, models, action_grid, sim_cfg, reward_cfg, seed=args.seed
    )
    state, _, _ = probe.reset()
    if checkpoint is not None:
        if len(state) != int(checkpoint["state_dim"]):
            raise ValueError("checkpoint state dimension does not match replay dataset")
        expected_names = checkpoint.get("state_feature_names")
        if expected_names is not None and list(expected_names) != list(probe.state_feature_names):
            raise ValueError("checkpoint state feature ordering does not match replay dataset")

    policy_metrics: Dict[str, Any] = {}
    if "rl" in requested:
        assert trainer is not None and normalizer is not None
        policy_metrics["rl"] = evaluate_policy(
            trainer,
            normalizer,
            dataset,
            dataset.episodes,
            models,
            action_grid,
            sim_cfg,
            reward_cfg,
            args.evaluation_modes,
            args.evaluation_max_episodes,
            args.seed,
        )

    greedy_config = greedy_config_from_args(args)
    if "greedy" in requested:
        policy_metrics["greedy"] = evaluate_greedy_policy(
            dataset,
            dataset.episodes,
            models,
            action_grid,
            sim_cfg,
            reward_cfg,
            greedy_config,
            args.evaluation_modes,
            args.evaluation_max_episodes,
            args.seed,
        )

    payload: Dict[str, Any] = {
        "policies": policy_metrics,
        "greedy_config": dataclasses.asdict(greedy_config),
    }
    if "rl" in policy_metrics and "greedy" in policy_metrics:
        payload["rl_vs_greedy"] = compare_policy_metrics(
            policy_metrics["rl"], policy_metrics["greedy"]
        )
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.replay_csv:
        if args.replay_policy not in requested:
            raise ValueError("--replay-policy must be included in --policies")
        replay_episodes(
            args.replay_policy,
            dataset,
            models,
            action_grid,
            sim_cfg,
            reward_cfg,
            Path(args.replay_csv),
            args.replay_csv_episodes,
            args.replay_csv_mode,
            args.seed,
            trainer=trainer,
            normalizer=normalizer,
            greedy_config=greedy_config,
        )
    LOGGER.info("evaluation written to %s", args.output_json)


if __name__ == "__main__":
    main()
