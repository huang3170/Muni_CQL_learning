"""Template for connecting production side models to the replay simulator.

Copy this file into your project, replace the placeholder calculations, and run
simulator_online_train.py with:

    --model-factory model_adapter_template:build_models

All inputs are point-in-time.  ``TradeEvent`` is provided only after the policy
has chosen the quote that is being evaluated against that trade.
"""

from __future__ import annotations

import numpy as np

from hybrid_replay_simulator import ModelContext, QuoteCandidate, TradeEvent


class ProductionModelBundle:
    def __init__(self) -> None:
        # Load your model artifacts here. Avoid loading untrusted pickle files.
        self.pricing_model = None
        self.win_model = None
        self.participation_model = None

    def pricing_anchor(self, context: ModelContext, quantity: float) -> float:
        """Return PricingModel(CUSIP, decision time, quantity)."""
        # Replace with your feature pipeline and PricingModel inference.
        return context.fair_mark + 0.20

    def effective_spread(self, context: ModelContext, quantity: float) -> float:
        """Optional quantity-aware full bid-ask spread used to materialize actions.

        The simulator computes::

            price_offset_dollar = price_offset_ratio * 0.5 * effective_spread

        by default. If this method is omitted, the simulator falls back through
        configured point-in-time snapshot spread features.
        """
        del quantity
        return max(float(context.snapshot_features.get("cep_bid_ask_width", 0.50)), 0.02)

    def pretrade_win_probability(self, context: ModelContext, quote: QuoteCandidate) -> float:
        """Calibrated action-grid score used as a policy-state feature.

        This must not use the next trade price or quantity. For the greedy
        baseline, calibrate it to the configured planning window (30 minutes by
        default). It may be an unconditional any-fill probability, or a
        demand-conditional score combined with a separate arrival estimate, as
        long as the returned value has the intended expected-fill interpretation.
        """
        return 0.02

    def event_win_probability(
        self,
        context: ModelContext,
        quote: QuoteCandidate,
        trade: TradeEvent,
        demand_price: float,
    ) -> float:
        """P(win positive flow | demand, eligible, state, action, event)."""
        # The policy did not see `trade` when choosing `quote`; only the simulator
        # may use it after the action was fixed.
        # The quote also exposes spread-normalized and dollar price locations:
        # quote.price_offset_ratio, quote.price_offset_dollar,
        # quote.effective_spread, and quote.spread_unit.
        price_advantage = demand_price - quote.offer_price
        return float(np.clip(0.10 + 0.40 * max(price_advantage, 0.0), 0.001, 0.999))

    def pretrade_participation_share(self, context: ModelContext, quote: QuoteCandidate) -> float:
        """Expected conditional share used as an action-grid state feature."""
        return 0.50

    def event_participation_share(
        self,
        context: ModelContext,
        quote: QuoteCandidate,
        trade: TradeEvent,
        demand_price: float,
    ) -> float:
        """E[filled / min(inventory, offer quantity, matched demand) | win]."""
        return 0.60

    def customer_price_haircut(self, context: ModelContext, trade: TradeEvent) -> float:
        """Convert observed customer trade price into a conservative demand price."""
        return 0.05 if trade.trade_type == "S" else 0.0

    def eligibility_tolerance(self, context: ModelContext, trade: TradeEvent) -> float:
        """Price matching tolerance based on spread/volatility/uncertainty."""
        width = max(context.snapshot_features.get("cep_bid_ask_width", 0.0), 0.0)
        return 0.02 + 0.05 * width

    def support_score(self, context: ModelContext, quote: QuoteCandidate) -> float:
        """Optional action support score in [0, 1] for action masking."""
        return 1.0


def build_models() -> ProductionModelBundle:
    return ProductionModelBundle()
