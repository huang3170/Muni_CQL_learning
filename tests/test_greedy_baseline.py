from __future__ import annotations

import unittest

import pandas as pd

from greedy_baseline import ForecastAwareGreedyPolicy, GreedyBaselineConfig, evaluate_greedy_policy
from hybrid_replay_simulator import (
    HybridMuniReplayEnv,
    ModelContext,
    QuoteCandidate,
    ReplayDataset,
    ReplayEpisode,
    RewardConfig,
    SimulatorConfig,
    Snapshot,
    StateSchema,
    TradeEvent,
)
from muni_cql_dueling_ddqn import ActionGrid


class ConstantModels:
    def pricing_anchor(self, context: ModelContext, quantity: float) -> float:
        return context.fair_mark

    def pretrade_win_probability(self, context: ModelContext, quote: QuoteCandidate) -> float:
        return 1.0

    def event_win_probability(self, context, quote, trade, demand_price):
        return 1.0

    def pretrade_participation_share(self, context, quote):
        return 1.0

    def event_participation_share(self, context, quote, trade, demand_price):
        return 1.0

    def customer_price_haircut(self, context, trade):
        return 0.0

    def eligibility_tolerance(self, context, trade):
        return 0.0

    def support_score(self, context, quote):
        return 1.0


def make_episode(forward_change: float, trade_price: float = 101.0) -> ReplayEpisode:
    start = pd.Timestamp("2025-01-02 09:30:00")
    features = {
        "risk_score": 1.0,
        "price_scale": 1.0,
        "forward_price_change_4h": forward_change,
        "forward_model_confidence": 1.0,
    }
    return ReplayEpisode(
        episode_id=f"e_{forward_change}_{trade_price}",
        cusip="TESTCUSIP",
        start_time=start,
        end_time=start + pd.Timedelta(minutes=60),
        starting_inventory=100.0,
        cost_basis=99.5,
        static_features={},
        snapshots=[
            Snapshot(start, 100.0, features),
            Snapshot(start + pd.Timedelta(minutes=30), 100.0, features),
            Snapshot(start + pd.Timedelta(minutes=60), 100.0, features),
        ],
        trades=[
            TradeEvent(
                "t1",
                start + pd.Timedelta(minutes=10),
                start + pd.Timedelta(minutes=20),
                trade_price,
                100.0,
                "S",
            )
        ],
    )


class GreedyBaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = ActionGrid()
        self.schema = StateSchema(
            snapshot_feature_cols=(
                "risk_score",
                "price_scale",
                "forward_price_change_4h",
                "forward_model_confidence",
            )
        )
        self.reward = RewardConfig(
            inventory_lambda=0.0,
            schedule_lambda=0.0,
            price_smooth_lambda=0.0,
            quantity_smooth_lambda=0.0,
            update_cost=0.0,
            terminal_lambda=0.0,
            liquidation_concession=0.0,
        )
        self.sim_cfg = SimulatorConfig(stochastic_fills=False, min_lot=5.0)
        self.models = ConstantModels()
        self.policy = ForecastAwareGreedyPolicy(
            self.grid,
            GreedyBaselineConfig(
                planning_minutes=30.0,
                forecast_weight=1.0,
                use_forecast_confidence=True,
            ),
        )

    def make_env(self, forward_change: float, trade_price: float = 101.0):
        env = HybridMuniReplayEnv(
            make_episode(forward_change, trade_price),
            self.schema,
            self.models,
            self.grid,
            self.sim_cfg,
            self.reward,
            seed=1,
        )
        _, mask, _ = env.reset()
        return env, mask

    def test_action_does_not_depend_on_unobserved_next_trade(self) -> None:
        env_a, mask_a = self.make_env(0.25, trade_price=99.0)
        env_b, mask_b = self.make_env(0.25, trade_price=105.0)
        decision_a = self.policy.select_action(env_a, mask_a, simulator_mode="partial")
        decision_b = self.policy.select_action(env_b, mask_b, simulator_mode="partial")
        self.assertEqual(decision_a.action_id, decision_b.action_id)

    def test_positive_forecast_is_more_conservative_than_negative_forecast(self) -> None:
        env_up, mask_up = self.make_env(1.0)
        env_down, mask_down = self.make_env(-1.0)
        up = self.policy.select_action(env_up, mask_up, simulator_mode="optimistic")
        down = self.policy.select_action(env_down, mask_down, simulator_mode="optimistic")
        up_spec = self.grid.decode(up.action_id)
        down_spec = self.grid.decode(down.action_id)
        up_fraction = 0.0 if up_spec.is_no_quote else up_spec.quantity_fraction
        down_fraction = 0.0 if down_spec.is_no_quote else down_spec.quantity_fraction
        self.assertLess(up_fraction, down_fraction)

    def test_selected_action_is_valid(self) -> None:
        env, mask = self.make_env(0.0)
        decision = self.policy.select_action(env, mask)
        self.assertTrue(mask[decision.action_id])
        self.assertEqual(len(decision.scores), self.grid.num_actions)

    def test_evaluate_greedy_policy(self) -> None:
        episode = make_episode(0.0)
        dataset = ReplayDataset([episode], self.schema)
        metrics = evaluate_greedy_policy(
            dataset,
            dataset.episodes,
            self.models,
            self.grid,
            self.sim_cfg,
            self.reward,
            self.policy.config,
            modes=("partial",),
            max_episodes=1,
            seed=1,
        )
        self.assertEqual(metrics["partial"]["episodes"], 1)
        self.assertEqual(len(metrics["partial"]["action_counts"]), self.grid.num_actions)


if __name__ == "__main__":
    unittest.main()
