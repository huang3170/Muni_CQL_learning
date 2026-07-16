from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from hybrid_replay_simulator import (
    HybridMuniReplayEnv,
    ModelContext,
    QuoteCandidate,
    ReplayEpisode,
    RewardConfig,
    SimulatorConfig,
    Snapshot,
    StateSchema,
    TradeEvent,
)
from muni_cql_dueling_ddqn import ActionGrid


class DeterministicModels:
    def __init__(self, win_probability: float = 1.0, share: float = 0.60) -> None:
        self.win_probability = win_probability
        self.share = share

    def pricing_anchor(self, context: ModelContext, quantity: float) -> float:
        return 100.50

    def pretrade_win_probability(self, context: ModelContext, quote: QuoteCandidate) -> float:
        return self.win_probability

    def event_win_probability(self, context, quote, trade, demand_price):
        return self.win_probability

    def pretrade_participation_share(self, context, quote):
        return self.share

    def event_participation_share(self, context, quote, trade, demand_price):
        return self.share

    def customer_price_haircut(self, context, trade):
        return 0.0

    def eligibility_tolerance(self, context, trade):
        return 0.0

    def support_score(self, context, quote):
        return 1.0


def make_episode(with_trade: bool = True) -> ReplayEpisode:
    start = pd.Timestamp("2025-01-02 09:30:00")
    snapshots = [
        Snapshot(start, 100.0, {"risk_score": 1.0, "price_scale": 1.0}),
        Snapshot(start + pd.Timedelta(minutes=30), 100.0, {"risk_score": 1.0, "price_scale": 1.0}),
        Snapshot(start + pd.Timedelta(minutes=60), 100.0, {"risk_score": 1.0, "price_scale": 1.0}),
    ]
    trades = []
    if with_trade:
        trades.append(
            TradeEvent(
                "t1",
                start + pd.Timedelta(minutes=10),
                start + pd.Timedelta(minutes=20),
                101.0,
                75.0,
                "S",
            )
        )
    return ReplayEpisode(
        episode_id="e1",
        cusip="TESTCUSIP",
        start_time=start,
        end_time=start + pd.Timedelta(minutes=60),
        starting_inventory=100.0,
        cost_basis=99.5,
        static_features={},
        snapshots=snapshots,
        trades=trades,
    )


class HybridSimulatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = ActionGrid()
        self.schema = StateSchema(snapshot_feature_cols=("risk_score", "price_scale"))
        self.reward = RewardConfig(
            inventory_lambda=0.0,
            schedule_lambda=0.0,
            price_smooth_lambda=0.0,
            quantity_smooth_lambda=0.0,
            update_cost=0.0,
            terminal_lambda=0.0,
            liquidation_concession=0.0,
        )
        self.half_at_anchor = self.grid.encode(3, 2)  # offset 0, quantity fraction 0.50

    def test_partial_fill_uses_capacity_times_share(self) -> None:
        env = HybridMuniReplayEnv(
            make_episode(),
            self.schema,
            DeterministicModels(win_probability=1.0, share=0.60),
            self.grid,
            SimulatorConfig(stochastic_fills=False, min_lot=5.0),
            self.reward,
            seed=1,
        )
        state, mask, _ = env.reset()
        result = env.step(self.half_at_anchor)
        self.assertEqual(result.info["trigger_reasons"], ["own_fill"])
        fill = result.info["fills"][0]
        self.assertEqual(fill["capacity"], 50.0)
        self.assertEqual(fill["filled_quantity"], 30.0)
        self.assertEqual(env.inventory, 70.0)
        self.assertEqual(result.elapsed_minutes, 10.0)
        self.assertAlmostEqual(result.discount, 0.99 ** (10.0 / 30.0), places=8)

    def test_new_quote_is_not_applied_to_trade_that_triggered_refresh(self) -> None:
        env = HybridMuniReplayEnv(
            make_episode(),
            self.schema,
            DeterministicModels(win_probability=1.0, share=0.60),
            self.grid,
            SimulatorConfig(stochastic_fills=False, min_lot=5.0),
            self.reward,
            seed=1,
        )
        _, _, _ = env.reset()
        first = env.step(self.half_at_anchor)
        self.assertEqual(env.inventory, 70.0)
        no_quote = self.grid.no_quote_action_id()
        second = env.step(no_quote)
        self.assertEqual(second.info["trigger_reasons"], ["trade_publish"])
        self.assertEqual(env.inventory, 70.0)
        self.assertEqual(second.elapsed_minutes, 10.0)

    def test_legacy_displayed_quote_ratio_differs_from_participation_share(self) -> None:
        episode = make_episode()
        trade = episode.trades[0]
        episode.trades = [
            TradeEvent(
                trade.event_id,
                trade.execution_time,
                trade.publish_time,
                trade.price,
                20.0,
                trade.trade_type,
            )
        ]
        participation_env = HybridMuniReplayEnv(
            episode,
            self.schema,
            DeterministicModels(win_probability=1.0, share=0.60),
            self.grid,
            SimulatorConfig(
                stochastic_fills=False,
                min_lot=5.0,
                quantity_model_definition="participation_share",
            ),
            self.reward,
            seed=1,
        )
        participation_env.reset()
        participation = participation_env.step(self.half_at_anchor)
        self.assertEqual(participation.info["fills"][0]["filled_quantity"], 10.0)

        legacy_env = HybridMuniReplayEnv(
            episode,
            self.schema,
            DeterministicModels(win_probability=1.0, share=0.60),
            self.grid,
            SimulatorConfig(
                stochastic_fills=False,
                min_lot=5.0,
                quantity_model_definition="displayed_quote_ratio",
            ),
            self.reward,
            seed=1,
        )
        legacy_env.reset()
        legacy = legacy_env.step(self.half_at_anchor)
        self.assertEqual(legacy.info["fills"][0]["filled_quantity"], 20.0)

    def test_unwon_trade_triggers_only_when_published(self) -> None:
        env = HybridMuniReplayEnv(
            make_episode(),
            self.schema,
            DeterministicModels(win_probability=0.0, share=1.0),
            self.grid,
            SimulatorConfig(stochastic_fills=False, min_lot=5.0),
            self.reward,
            seed=1,
        )
        env.reset()
        result = env.step(self.half_at_anchor)
        self.assertEqual(result.info["trigger_reasons"], ["trade_publish"])
        self.assertEqual(result.elapsed_minutes, 20.0)
        self.assertEqual(env.inventory, 100.0)

    def test_clock_trigger_exists_without_trade(self) -> None:
        env = HybridMuniReplayEnv(
            make_episode(with_trade=False),
            self.schema,
            DeterministicModels(),
            self.grid,
            SimulatorConfig(stochastic_fills=False, min_lot=5.0),
            self.reward,
            seed=1,
        )
        env.reset()
        result = env.step(self.half_at_anchor)
        self.assertEqual(result.info["trigger_reasons"], ["clock"])
        self.assertEqual(result.elapsed_minutes, 30.0)


if __name__ == "__main__":
    unittest.main()
