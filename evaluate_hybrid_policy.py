#!/usr/bin/env python3
"""Evaluate or replay a saved hybrid simulator-online policy checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from hybrid_replay_simulator import (
    HybridMuniReplayEnv,
    ReplayDataset,
    RewardConfig,
    SimulatorConfig,
    load_model_bundle,
)
from muni_cql_dueling_ddqn import ActionGrid, StateNormalizer, TrainConfig, CQLDuelingDoubleDQNTrainer, resolve_device
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
        quantity_fractions=tuple(grid_payload.get("quantity_fractions", ActionGrid().quantity_fractions)),
        include_no_quote=bool(grid_payload.get("include_no_quote", True)),
    )
    sim_cfg = SimulatorConfig(**checkpoint.get("simulator_config", {}))
    reward_cfg = RewardConfig(**checkpoint.get("reward_config", {}))
    return trainer, normalizer, action_grid, sim_cfg, reward_cfg, checkpoint


def replay_episodes(
    trainer,
    normalizer,
    action_grid,
    simulator_config,
    reward_config,
    dataset,
    models,
    output_path: Path,
    max_episodes: int,
    mode: str,
    seed: int,
) -> None:
    cfg = dataclasses.replace(simulator_config, simulator_mode=mode)
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
            action_id, q_values = trainer.select_action(normalizer.transform(state), mask)
            spec = action_grid.decode(action_id)
            result = env.step(action_id)
            step_index += 1
            rows.append(
                {
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
                    "reward": result.reward,
                    "inventory_before": result.info["inventory_before_interval"],
                    "inventory_after": result.info["inventory_after_interval"],
                    "trigger_reasons": "|".join(result.info["trigger_reasons"]),
                    "fill_quantity": sum(item["filled_quantity"] for item in result.info["fills"]),
                    "max_q": float(max(q_values)),
                    "done": result.done,
                }
            )
            state, mask = result.state, result.action_mask
            if result.done:
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--model-factory", help="module:function returning simulator models")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--replay-csv")
    parser.add_argument("--replay-csv-episodes", type=int, default=5)
    parser.add_argument("--replay-csv-mode", choices=("optimistic", "win_only", "partial"), default="partial")
    parser.add_argument("--evaluation-max-episodes", type=int, default=100)
    parser.add_argument("--evaluation-modes", nargs="+", default=["optimistic", "win_only", "partial"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    device = resolve_device(args.device)
    dataset = ReplayDataset.from_directory(args.replay_dir)
    models = load_model_bundle(args.model_factory)
    trainer, normalizer, action_grid, sim_cfg, reward_cfg, checkpoint = load_checkpoint(
        Path(args.checkpoint), device
    )
    probe = HybridMuniReplayEnv(
        dataset.episodes[0], dataset.state_schema, models, action_grid, sim_cfg, reward_cfg, seed=args.seed
    )
    state, _, _ = probe.reset()
    if len(state) != int(checkpoint["state_dim"]):
        raise ValueError("checkpoint state dimension does not match replay dataset")
    expected_names = checkpoint.get("state_feature_names")
    if expected_names is not None and list(expected_names) != list(probe.state_feature_names):
        raise ValueError("checkpoint state feature ordering does not match replay dataset")

    metrics = evaluate_policy(
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
    Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.replay_csv:
        replay_episodes(
            trainer,
            normalizer,
            action_grid,
            sim_cfg,
            reward_cfg,
            dataset,
            models,
            Path(args.replay_csv),
            args.replay_csv_episodes,
            args.replay_csv_mode,
            args.seed,
        )
    LOGGER.info("evaluation written to %s", args.output_json)


if __name__ == "__main__":
    main()
