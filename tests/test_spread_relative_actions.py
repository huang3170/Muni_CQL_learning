from __future__ import annotations

import unittest

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
)
from muni_cql_dueling_ddqn import ActionGrid
from prepare_muni_transitions_template import nearest_logged_action


class ConstantModels:
    def pricing_anchor(self, context: ModelContext, quantity: float) -> float:
        return 100.0

    def pretrade_win_probability(self, context: ModelContext, quote: QuoteCandidate) -> float:
        return 0.5

    def event_win_probability(self, context, quote, trade, demand_price):
        return 0.5

    def pretrade_participation_share(self, context, quote):
        return 0.5

    def event_participation_share(self, context, quote, trade, demand_price):
        return 0.5

    def customer_price_haircut(self, context, trade):
        return 0.0

    def eligibility_tolerance(self, context, trade):
        return 0.0

    def support_score(self, context, quote):
        return 1.0


def make_episode(spread: float) -> ReplayEpisode:
    start = pd.Timestamp("2025-01-02 09:30:00")
    features = {
        "cep_bid_ask_width": spread,
        "spread_age_minutes": 2.0,
        "risk_score": 1.0,
    }
    return ReplayEpisode(
        episode_id=f"spread_{spread}",
        cusip="TESTCUSIP",
        start_time=start,
        end_time=start + pd.Timedelta(minutes=30),
        starting_inventory=100.0,
        cost_basis=99.0,
        static_features={},
        snapshots=[
            Snapshot(start, 100.0, features),
            Snapshot(start + pd.Timedelta(minutes=30), 100.0, features),
        ],
        trades=[],
    )


class SpreadRelativeActionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = ActionGrid()
        self.schema = StateSchema(
            snapshot_feature_cols=("cep_bid_ask_width", "spread_age_minutes", "risk_score")
        )
        self.reward = RewardConfig(
            inventory_lambda=0.0,
            schedule_lambda=0.0,
            price_smooth_lambda=0.0,
            price_dollar_smooth_lambda=0.0,
            quantity_smooth_lambda=0.0,
            update_cost=0.0,
            terminal_lambda=0.0,
            liquidation_concession=0.0,
        )
        self.ratio_plus_one_half_inventory = self.grid.encode(6, 2)

    def make_env(self, spread: float, **config_kwargs):
        env = HybridMuniReplayEnv(
            make_episode(spread),
            self.schema,
            ConstantModels(),
            self.grid,
            SimulatorConfig(stochastic_fills=False, **config_kwargs),
            self.reward,
            seed=1,
        )
        env.reset()
        return env

    def test_same_ratio_scales_with_cusip_spread(self) -> None:
        narrow = self.make_env(0.40)
        wide = self.make_env(2.00)

        narrow_preview = narrow.preview_action(self.ratio_plus_one_half_inventory)
        wide_preview = wide.preview_action(self.ratio_plus_one_half_inventory)

        self.assertAlmostEqual(narrow_preview["price_offset_ratio"], 1.0)
        self.assertAlmostEqual(wide_preview["price_offset_ratio"], 1.0)
        self.assertAlmostEqual(narrow_preview["price_offset_dollar"], 0.20)
        self.assertAlmostEqual(wide_preview["price_offset_dollar"], 1.00)
        self.assertAlmostEqual(narrow_preview["offer_price"], 100.20)
        self.assertAlmostEqual(wide_preview["offer_price"], 101.00)

    def test_absolute_cap_can_mask_excessively_wide_action(self) -> None:
        env = self.make_env(2.00, max_absolute_price_offset=0.50, mask_if_offset_clipped=True)
        mask = env.action_mask()
        self.assertFalse(mask[self.ratio_plus_one_half_inventory])
        self.assertTrue(mask[self.grid.no_quote_action_id()])

    def test_logged_dollar_offset_is_encoded_as_spread_ratio(self) -> None:
        # Full spread=2.0 -> half-spread unit=1.0. A +0.5 dollar offset maps to +0.5 ratio.
        action_id = nearest_logged_action(
            offer_price=100.5,
            offer_qty=50.0,
            pricing_anchor=100.0,
            inventory_before=100.0,
            effective_spread=2.0,
        )
        spec = self.grid.decode(action_id)
        self.assertAlmostEqual(spec.price_offset_ratio, 0.5)
        self.assertAlmostEqual(spec.quantity_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
