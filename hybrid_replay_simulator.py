#!/usr/bin/env python3
"""Hybrid clock-and-trade-triggered historical replay simulator.

The environment implements the Version 2.0 report design:

* One episode is one CUSIP position lifecycle.
* Quotes refresh on a fixed clock, after an own simulated fill, and after a
  relevant external trade becomes observable.
* The quote active *before* a trade processes that trade.  A refreshed quote
  can affect only later events.
* External trades enter the policy state at publication/observable time, not
  execution time.
* Each transition stores its elapsed time and a time-adjusted discount.
* Fill simulation is split into demand arrival, price eligibility, win
  probability, and conditional participation share.

This is a research simulator, not an MSRB execution rule.  The heuristic model
bundle is supplied only for smoke testing; replace it with calibrated internal
Pricing/Win/Participation models for real research.
"""

from __future__ import annotations

import dataclasses
import heapq
import importlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from muni_cql_dueling_ddqn import ActionGrid, ActionSpec


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulatorConfig:
    clock_minutes: float = 30.0
    base_minutes: float = 30.0
    base_discount: float = 0.99
    min_lot: float = 5.0
    price_notional_divisor: float = 100.0
    simulator_mode: str = "partial"  # optimistic | win_only | partial
    quantity_model_definition: str = "participation_share"  # participation_share | displayed_quote_ratio
    stochastic_fills: bool = True
    allowed_trade_types: Tuple[str, ...] = ("S",)
    trigger_on_same_cusip_publish: bool = True
    trigger_on_own_fill: bool = True
    min_quote_life_minutes: float = 0.0
    clock_overrides_min_quote_life: bool = True
    own_fill_overrides_min_quote_life: bool = True

    # Spread-relative price action mapping. The default action unit is one
    # half of the effective full bid-ask spread.
    spread_feature_names: Tuple[str, ...] = (
        "cep_bid_ask_width",
        "predicted_bid_ask_spread",
        "segment_bid_ask_spread",
    )
    spread_age_feature_name: str = "spread_age_minutes"
    spread_floor: float = 0.02
    spread_cap: float = 4.0
    fallback_spread: float = 0.50
    spread_unit_multiplier: float = 0.50
    max_absolute_price_offset: float = 2.0
    mask_if_offset_clipped: bool = True

    min_price_delta_from_mark: float = -2.0
    max_price_delta_from_mark: float = 2.0
    support_threshold: float = 0.0
    max_decisions_per_episode: int = 500
    rolling_trade_minutes: float = 30.0
    state_clip_abs: float = 1.0e8

    def __post_init__(self) -> None:
        if self.clock_minutes <= 0 or self.base_minutes <= 0:
            raise ValueError("clock_minutes and base_minutes must be positive")
        if not 0 < self.base_discount <= 1:
            raise ValueError("base_discount must be in (0, 1]")
        if self.min_lot <= 0:
            raise ValueError("min_lot must be positive")
        if self.simulator_mode not in {"optimistic", "win_only", "partial"}:
            raise ValueError("simulator_mode must be optimistic, win_only, or partial")
        if self.quantity_model_definition not in {"participation_share", "displayed_quote_ratio"}:
            raise ValueError(
                "quantity_model_definition must be participation_share or displayed_quote_ratio"
            )
        if not self.spread_feature_names:
            raise ValueError("spread_feature_names must contain at least one feature")
        if self.spread_floor <= 0 or self.spread_cap < self.spread_floor:
            raise ValueError("spread_floor must be positive and spread_cap >= spread_floor")
        if self.fallback_spread <= 0:
            raise ValueError("fallback_spread must be positive")
        if self.spread_unit_multiplier <= 0:
            raise ValueError("spread_unit_multiplier must be positive")
        if self.max_absolute_price_offset <= 0:
            raise ValueError("max_absolute_price_offset must be positive")


@dataclass(frozen=True)
class RewardConfig:
    inventory_lambda: float = 0.05
    schedule_lambda: float = 0.10
    price_smooth_lambda: float = 0.01  # applied to offset-ratio changes
    price_dollar_smooth_lambda: float = 0.0  # optional spread-normalized dollar change
    quantity_smooth_lambda: float = 0.01
    update_cost: float = 0.001
    missed_demand_lambda: float = 0.0
    underpricing_lambda: float = 0.0
    underpricing_tolerance: float = 0.125
    terminal_lambda: float = 1.0
    liquidation_concession: float = 0.50
    risk_feature_name: str = "risk_score"
    price_scale_feature_name: str = "price_scale"


@dataclass(frozen=True)
class StateSchema:
    static_feature_cols: Tuple[str, ...] = ()
    snapshot_feature_cols: Tuple[str, ...] = ()
    include_action_grid_features: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "StateSchema":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            static_feature_cols=tuple(payload.get("static_feature_cols", ())),
            snapshot_feature_cols=tuple(payload.get("snapshot_feature_cols", ())),
            include_action_grid_features=bool(payload.get("include_action_grid_features", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Replay data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeEvent:
    event_id: str
    execution_time: pd.Timestamp
    publish_time: pd.Timestamp
    price: float
    quantity: float
    trade_type: str = "S"


@dataclass(frozen=True)
class Snapshot:
    observable_time: pd.Timestamp
    fair_mark: float
    features: Mapping[str, float]


@dataclass
class ReplayEpisode:
    episode_id: str
    cusip: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    starting_inventory: float
    cost_basis: float
    static_features: Mapping[str, float]
    snapshots: List[Snapshot]
    trades: List[TradeEvent]

    def validate(self) -> None:
        if self.start_time >= self.end_time:
            raise ValueError(f"Episode {self.episode_id}: start_time must be before end_time")
        if self.starting_inventory <= 0:
            raise ValueError(f"Episode {self.episode_id}: starting_inventory must be positive")
        if not self.snapshots:
            raise ValueError(f"Episode {self.episode_id}: at least one snapshot is required")
        if self.snapshots[0].observable_time > self.start_time:
            raise ValueError(
                f"Episode {self.episode_id}: first snapshot must be observable by episode start"
            )
        if any(a.observable_time > b.observable_time for a, b in zip(self.snapshots, self.snapshots[1:])):
            raise ValueError(f"Episode {self.episode_id}: snapshots are not sorted")
        if any(a.execution_time > b.execution_time for a, b in zip(self.trades, self.trades[1:])):
            raise ValueError(f"Episode {self.episode_id}: trades are not sorted")
        for trade in self.trades:
            if trade.publish_time < trade.execution_time:
                raise ValueError(
                    f"Episode {self.episode_id}, trade {trade.event_id}: publish_time precedes execution_time"
                )
            if trade.quantity < 0:
                raise ValueError("trade quantity must be nonnegative")


@dataclass
class ReplayDataset:
    episodes: List[ReplayEpisode]
    state_schema: StateSchema

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ReplayDataset":
        directory = Path(directory)
        schema_path = directory / "schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(schema_path)
        schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
        state_schema = StateSchema(
            static_feature_cols=tuple(schema_payload.get("static_feature_cols", ())),
            snapshot_feature_cols=tuple(schema_payload.get("snapshot_feature_cols", ())),
            include_action_grid_features=bool(schema_payload.get("include_action_grid_features", True)),
        )
        positions = _read_table(directory, "positions")
        snapshots = _read_table(directory, "snapshots")
        trades = _read_table(directory, "trades")

        required_positions = {
            "episode_id",
            "cusip",
            "start_time",
            "end_time",
            "starting_inventory",
            "cost_basis",
        }
        required_snapshots = {"episode_id", "observable_time", "fair_mark"}
        required_trades = {
            "episode_id",
            "event_id",
            "execution_time",
            "publish_time",
            "trade_price",
            "trade_quantity",
            "trade_type",
        }
        _require_columns(positions, required_positions, "positions")
        _require_columns(snapshots, required_snapshots, "snapshots")
        _require_columns(trades, required_trades, "trades")

        for col in ("start_time", "end_time"):
            positions[col] = pd.to_datetime(positions[col], errors="raise")
        snapshots["observable_time"] = pd.to_datetime(snapshots["observable_time"], errors="raise")
        trades["execution_time"] = pd.to_datetime(trades["execution_time"], errors="raise")
        trades["publish_time"] = pd.to_datetime(trades["publish_time"], errors="raise")

        snapshot_groups = {str(k): v.sort_values("observable_time") for k, v in snapshots.groupby("episode_id")}
        trade_groups = {str(k): v.sort_values(["execution_time", "event_id"]) for k, v in trades.groupby("episode_id")}
        episodes: List[ReplayEpisode] = []
        for row in positions.itertuples(index=False):
            episode_id = str(row.episode_id)
            static_features = {
                col: _finite_float(getattr(row, col), f"positions.{col}")
                for col in state_schema.static_feature_cols
            }
            snapshot_rows = snapshot_groups.get(episode_id)
            if snapshot_rows is None:
                raise ValueError(f"No snapshots for episode_id={episode_id}")
            episode_snapshots = [
                Snapshot(
                    observable_time=pd.Timestamp(r.observable_time),
                    fair_mark=_finite_float(r.fair_mark, "snapshots.fair_mark"),
                    features={
                        col: _finite_float(getattr(r, col), f"snapshots.{col}")
                        for col in state_schema.snapshot_feature_cols
                    },
                )
                for r in snapshot_rows.itertuples(index=False)
            ]
            trade_rows = trade_groups.get(episode_id)
            episode_trades: List[TradeEvent] = []
            if trade_rows is not None:
                episode_trades = [
                    TradeEvent(
                        event_id=str(r.event_id),
                        execution_time=pd.Timestamp(r.execution_time),
                        publish_time=pd.Timestamp(r.publish_time),
                        price=_finite_float(r.trade_price, "trades.trade_price"),
                        quantity=max(_finite_float(r.trade_quantity, "trades.trade_quantity"), 0.0),
                        trade_type=str(r.trade_type).upper(),
                    )
                    for r in trade_rows.itertuples(index=False)
                ]
            episode = ReplayEpisode(
                episode_id=episode_id,
                cusip=str(row.cusip),
                start_time=pd.Timestamp(row.start_time),
                end_time=pd.Timestamp(row.end_time),
                starting_inventory=_finite_float(row.starting_inventory, "positions.starting_inventory"),
                cost_basis=_finite_float(row.cost_basis, "positions.cost_basis"),
                static_features=static_features,
                snapshots=episode_snapshots,
                trades=episode_trades,
            )
            episode.validate()
            episodes.append(episode)
        if not episodes:
            raise ValueError("Replay dataset contains no episodes")
        return cls(episodes=episodes, state_schema=state_schema)

    def by_id(self) -> Dict[str, ReplayEpisode]:
        return {episode.episode_id: episode for episode in self.episodes}


def _read_table(directory: Path, stem: str) -> pd.DataFrame:
    for suffix, reader in (
        (".parquet", pd.read_parquet),
        (".csv", pd.read_csv),
        (".pkl", pd.read_pickle),
    ):
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return reader(path)
    raise FileNotFoundError(f"Expected {stem}.parquet, {stem}.csv, or {stem}.pkl in {directory}")


def _require_columns(df: pd.DataFrame, required: Iterable[str], table: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{table} is missing columns: {missing}")


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


# ---------------------------------------------------------------------------
# Side-model interfaces and smoke-test heuristics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelContext:
    episode_id: str
    cusip: str
    time: pd.Timestamp
    fair_mark: float
    inventory: float
    starting_inventory: float
    cost_basis: float
    static_features: Mapping[str, float]
    snapshot_features: Mapping[str, float]
    observable_trade_features: Mapping[str, float]


@dataclass(frozen=True)
class QuoteCandidate:
    action: ActionSpec
    offer_price: float
    offer_quantity: float
    price_offset_ratio: float
    price_offset_dollar: float
    raw_price_offset_dollar: float
    effective_spread: float
    spread_unit: float
    spread_source_index: float
    spread_is_fallback: float
    offset_was_clipped: bool


class SimulatorModelBundle(Protocol):
    def pricing_anchor(self, context: ModelContext, quantity: float) -> float: ...

    def pretrade_win_probability(self, context: ModelContext, quote: QuoteCandidate) -> float: ...

    def event_win_probability(
        self, context: ModelContext, quote: QuoteCandidate, trade: TradeEvent, demand_price: float
    ) -> float: ...

    def pretrade_participation_share(self, context: ModelContext, quote: QuoteCandidate) -> float: ...

    def event_participation_share(
        self, context: ModelContext, quote: QuoteCandidate, trade: TradeEvent, demand_price: float
    ) -> float: ...

    def customer_price_haircut(self, context: ModelContext, trade: TradeEvent) -> float: ...

    def eligibility_tolerance(self, context: ModelContext, trade: TradeEvent) -> float: ...

    def support_score(self, context: ModelContext, quote: QuoteCandidate) -> float: ...


@dataclass
class HeuristicModelBundle:
    """Runnable placeholder models for smoke tests only."""

    base_offer_spread: float = 0.20
    quantity_curve: float = 0.05
    win_intercept: float = -1.0
    win_price_slope: float = 5.0
    share_base: float = 0.65
    share_price_slope: float = 0.50
    customer_haircut_value: float = 0.05
    tolerance_value: float = 0.03

    def pricing_anchor(self, context: ModelContext, quantity: float) -> float:
        relative_size = quantity / max(context.starting_inventory, 1.0)
        return context.fair_mark + self.base_offer_spread + self.quantity_curve * math.log1p(relative_size)

    def pretrade_win_probability(self, context: ModelContext, quote: QuoteCandidate) -> float:
        aggressiveness = context.fair_mark + self.base_offer_spread - quote.offer_price
        liquidity = context.snapshot_features.get("liquidity_score", 0.0)
        return _sigmoid(self.win_intercept + 2.0 * aggressiveness + 0.2 * liquidity)

    def event_win_probability(
        self, context: ModelContext, quote: QuoteCandidate, trade: TradeEvent, demand_price: float
    ) -> float:
        price_advantage = demand_price - quote.offer_price
        size_pressure = quote.offer_quantity / max(trade.quantity, 1.0)
        liquidity = context.snapshot_features.get("liquidity_score", 0.0)
        logit = self.win_intercept + self.win_price_slope * price_advantage - 0.25 * size_pressure + 0.2 * liquidity
        return _sigmoid(logit)

    def pretrade_participation_share(self, context: ModelContext, quote: QuoteCandidate) -> float:
        size_fraction = quote.offer_quantity / max(context.inventory, 1.0)
        return float(np.clip(self.share_base - 0.15 * size_fraction, 0.05, 1.0))

    def event_participation_share(
        self, context: ModelContext, quote: QuoteCandidate, trade: TradeEvent, demand_price: float
    ) -> float:
        advantage = max(demand_price - quote.offer_price, 0.0)
        size_fraction = quote.offer_quantity / max(context.inventory, 1.0)
        share = self.share_base + self.share_price_slope * advantage - 0.15 * size_fraction
        return float(np.clip(share, 0.05, 1.0))

    def customer_price_haircut(self, context: ModelContext, trade: TradeEvent) -> float:
        return self.customer_haircut_value if trade.trade_type == "S" else 0.0

    def eligibility_tolerance(self, context: ModelContext, trade: TradeEvent) -> float:
        width = max(context.snapshot_features.get("cep_bid_ask_width", 0.0), 0.0)
        volatility = max(context.snapshot_features.get("realized_volatility_30d", 0.0), 0.0)
        return self.tolerance_value + 0.05 * width + 0.02 * volatility

    def support_score(self, context: ModelContext, quote: QuoteCandidate) -> float:
        price_distance = abs(quote.offer_price - context.fair_mark)
        return float(np.clip(1.0 - price_distance / 3.0, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_model_bundle(factory_spec: Optional[str]) -> SimulatorModelBundle:
    """Load ``module:function`` returning a SimulatorModelBundle.

    If ``factory_spec`` is omitted, return the smoke-test heuristic bundle.
    """

    if not factory_spec:
        return HeuristicModelBundle()
    if ":" not in factory_spec:
        raise ValueError("model factory must be written as module:function")
    module_name, function_name = factory_spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    bundle = factory()
    return bundle


# ---------------------------------------------------------------------------
# Environment internals
# ---------------------------------------------------------------------------


@dataclass
class ActiveQuote:
    action_id: int
    price_offset_ratio: float
    price_offset_dollar: float
    effective_spread: float
    spread_unit: float
    quantity_fraction: float
    offer_price: float
    initial_quantity: float
    live_quantity: float
    posted_time: pd.Timestamp


@dataclass(order=True, frozen=True)
class TimelineEvent:
    time: pd.Timestamp
    priority: int
    sequence: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False, default=None)


@dataclass
class StepResult:
    state: np.ndarray
    action_mask: np.ndarray
    reward: float
    done: bool
    discount: float
    elapsed_minutes: float
    info: Dict[str, Any]


class HybridMuniReplayEnv:
    """One-position hybrid historical replay environment."""

    DYNAMIC_FEATURE_NAMES: Tuple[str, ...] = (
        "inventory",
        "inventory_fraction_remaining",
        "position_age_minutes",
        "position_age_fraction",
        "time_to_horizon_minutes",
        "time_to_horizon_fraction",
        "delta_since_last_decision_minutes",
        "fair_mark",
        "cost_basis_minus_mark",
        "cumulative_fill_fraction",
        "live_quote_fraction_of_inventory",
        "quote_age_minutes",
        "effective_spread",
        "spread_unit",
        "log_effective_spread",
        "spread_age_minutes",
        "spread_source_index",
        "spread_is_fallback",
        "previous_price_offset_ratio",
        "previous_price_offset_dollar",
        "previous_quantity_fraction",
        "quote_update_count",
        "last_own_fill_fraction",
    )
    OBSERVABLE_TRADE_FEATURE_NAMES: Tuple[str, ...] = (
        "last_trade_price",
        "last_trade_quantity",
        "last_trade_type_sign",
        "minutes_since_last_published_trade",
        "rolling_trade_count",
        "rolling_trade_volume",
        "rolling_trade_vwap",
        "rolling_trade_direction_imbalance",
    )
    TRIGGER_FEATURE_NAMES: Tuple[str, ...] = (
        "trigger_clock",
        "trigger_own_fill",
        "trigger_trade_publish",
        "trigger_reset",
    )
    ACTION_GRID_FEATURE_NAMES: Tuple[str, ...] = (
        "anchor_minus_mark",
        "offer_minus_mark",
        "price_offset_ratio",
        "price_offset_dollar",
        "effective_spread",
        "spread_unit",
        "pretrade_win_probability",
        "pretrade_participation_share",
        "expected_fill_fraction",
    )

    def __init__(
        self,
        episode: ReplayEpisode,
        state_schema: StateSchema,
        models: SimulatorModelBundle,
        action_grid: Optional[ActionGrid] = None,
        simulator_config: Optional[SimulatorConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        seed: int = 2026,
    ) -> None:
        episode.validate()
        self.episode = episode
        self.state_schema = state_schema
        self.models = models
        self.action_grid = action_grid or ActionGrid()
        self.config = simulator_config or SimulatorConfig()
        self.reward_config = reward_config or RewardConfig()
        self.rng = np.random.default_rng(seed)

        self._events: List[TimelineEvent] = []
        self._event_cursor = 0
        self._published_trades: Deque[Tuple[pd.Timestamp, TradeEvent]] = deque()
        self._last_published_trade: Optional[Tuple[pd.Timestamp, TradeEvent]] = None
        self._snapshot_by_time: Dict[pd.Timestamp, Snapshot] = {}
        self.current_snapshot: Snapshot = episode.snapshots[0]
        self.current_time: pd.Timestamp = episode.start_time
        self.last_decision_time: pd.Timestamp = episode.start_time
        self.inventory: float = episode.starting_inventory
        self.active_quote: Optional[ActiveQuote] = None
        self.previous_action_id: Optional[int] = None
        self.cumulative_fill: float = 0.0
        self.last_own_fill: float = 0.0
        self.quote_update_count: int = 0
        self.decision_count: int = 0
        self.done: bool = False
        self.last_trigger_reasons: set[str] = {"reset"}
        self._interval_reward_components: Dict[str, float] = {}
        self._last_accrual_time: pd.Timestamp = episode.start_time
        self._build_timeline()
        self._set_latest_snapshot_at_or_before(self.current_time, accrue_mtm=False)

        # Build names once and validate state dimension deterministically.
        self.state_feature_names = self._make_state_feature_names()

    # ---------------------------- public API ----------------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        self._event_cursor = 0
        self._published_trades.clear()
        self._last_published_trade = None
        self.current_time = self.episode.start_time
        self.last_decision_time = self.current_time
        self.inventory = self.episode.starting_inventory
        self.active_quote = None
        self.previous_action_id = None
        self.cumulative_fill = 0.0
        self.last_own_fill = 0.0
        self.quote_update_count = 0
        self.decision_count = 0
        self.done = False
        self.last_trigger_reasons = {"reset"}
        self._last_accrual_time = self.current_time
        self._set_latest_snapshot_at_or_before(self.current_time, accrue_mtm=False)
        self._interval_reward_components = self._empty_reward_components()
        state = self.build_state()
        mask = self.action_mask()
        return state, mask, self._base_info()

    def step(self, action_id: int) -> StepResult:
        if self.done:
            raise RuntimeError("step() called after episode termination")
        mask = self.action_mask()
        if action_id < 0 or action_id >= self.action_grid.num_actions or not bool(mask[action_id]):
            raise ValueError(f"Invalid action_id={action_id} at {self.current_time}")
        self._interval_reward_components = self._empty_reward_components()
        self._post_quote(action_id)
        interval_start = self.current_time
        start_inventory = self.inventory
        self.decision_count += 1

        trigger_reasons: set[str] = set()
        fills: List[Dict[str, Any]] = []
        while not self.done:
            if self._event_cursor >= len(self._events):
                self._terminate_at(self.episode.end_time, "event_queue_exhausted")
                trigger_reasons.add("terminal")
                break

            group_time = self._events[self._event_cursor].time
            self._accrue_carry_to(group_time)
            self.current_time = group_time
            group: List[TimelineEvent] = []
            while self._event_cursor < len(self._events) and self._events[self._event_cursor].time == group_time:
                group.append(self._events[self._event_cursor])
                self._event_cursor += 1

            # 1) observable snapshots update the mark/features first.
            for event in group:
                if event.kind == "snapshot":
                    self._apply_snapshot(event.payload)

            # 2) all trades executed at the same timestamp use the quote that was
            # active before that timestamp.  Live quantity is reduced sequentially.
            had_own_fill = False
            for event in group:
                if event.kind == "trade_exec":
                    fill_info = self._process_trade_execution(event.payload)
                    fills.append(fill_info)
                    if fill_info["filled_quantity"] > 0:
                        had_own_fill = True

            if had_own_fill and self.config.trigger_on_own_fill:
                if self.config.own_fill_overrides_min_quote_life or self._quote_life_allows_trigger("own_fill"):
                    trigger_reasons.add("own_fill")

            # 3) external trade information becomes policy-visible only at publish time.
            for event in group:
                if event.kind == "trade_publish":
                    self._publish_trade(event.payload)
                    if self.config.trigger_on_same_cusip_publish and self._quote_life_allows_trigger(
                        "trade_publish"
                    ):
                        trigger_reasons.add("trade_publish")

            # 4) mandatory clock checkpoints.
            if any(event.kind == "clock" for event in group):
                if self.config.clock_overrides_min_quote_life or self._quote_life_allows_trigger("clock"):
                    trigger_reasons.add("clock")

            if any(event.kind == "terminal" for event in group):
                self._terminate_at(group_time, "horizon")
                trigger_reasons.add("terminal")

            if self.inventory <= 1e-12:
                self.inventory = 0.0
                self.done = True
                trigger_reasons.add("inventory_zero")

            if self.decision_count >= self.config.max_decisions_per_episode:
                self._terminate_at(group_time, "max_decisions")
                trigger_reasons.add("terminal")

            if trigger_reasons or self.done:
                self.current_time = group_time
                break

        self.current_time = min(self.current_time, self.episode.end_time)
        elapsed_minutes = max((self.current_time - interval_start).total_seconds() / 60.0, 0.0)
        self._apply_schedule_penalty(elapsed_minutes)
        if self.done and self.inventory > 0:
            self._apply_terminal_liquidation()

        discount = self.config.base_discount ** (elapsed_minutes / self.config.base_minutes)
        reward = float(sum(self._interval_reward_components.values()))
        self.last_trigger_reasons = set(trigger_reasons) or {"terminal"}
        self.last_decision_time = self.current_time

        if self.done:
            next_state = self.build_state()
            next_mask = np.zeros(self.action_grid.num_actions, dtype=bool)
        else:
            next_state = self.build_state()
            next_mask = self.action_mask()

        info = self._base_info()
        info.update(
            {
                "interval_start": str(interval_start),
                "interval_end": str(self.current_time),
                "elapsed_minutes": elapsed_minutes,
                "discount": discount,
                "trigger_reasons": sorted(self.last_trigger_reasons),
                "reward_components": dict(self._interval_reward_components),
                "fills": fills,
                "inventory_before_interval": start_inventory,
                "inventory_after_interval": self.inventory,
            }
        )
        return StepResult(
            state=next_state,
            action_mask=next_mask,
            reward=reward,
            done=self.done,
            discount=float(discount),
            elapsed_minutes=float(elapsed_minutes),
            info=info,
        )

    def preview_action(self, action_id: int, simulator_mode: Optional[str] = None) -> Dict[str, float]:
        """Return point-in-time, pre-trade action diagnostics without mutation.

        Price actions are spread-relative. The returned diagnostics expose both
        the dimensionless ratio and the materialized dollar offset.
        """
        mask = self.action_mask()
        if action_id < 0 or action_id >= self.action_grid.num_actions or not bool(mask[action_id]):
            raise ValueError(f"Invalid action_id={action_id} at {self.current_time}")
        mode = simulator_mode or self.config.simulator_mode
        if mode not in {"optimistic", "win_only", "partial"}:
            raise ValueError("simulator_mode must be optimistic, win_only, or partial")

        context = self._model_context()
        spec = self.action_grid.decode(action_id)
        if spec.is_no_quote:
            spread = self._effective_spread_info(context, max(min(self.inventory, self.config.min_lot), 0.0))
            return {
                "action_id": float(action_id),
                "is_no_quote": 1.0,
                "price_offset_ratio": 0.0,
                "price_offset_dollar": 0.0,
                "raw_price_offset_dollar": 0.0,
                "effective_spread": spread["effective_spread"],
                "spread_unit": spread["spread_unit"],
                "spread_source_index": spread["spread_source_index"],
                "spread_is_fallback": spread["spread_is_fallback"],
                "offset_was_clipped": 0.0,
                "quantity_fraction": 0.0,
                "offer_price": math.nan,
                "offer_quantity": 0.0,
                "win_probability": 0.0,
                "participation_share": 0.0,
                "expected_fill_quantity": 0.0,
                "expected_remaining_inventory": float(self.inventory),
                "support_score": 1.0,
            }

        quote = self._candidate_from_spec(context, spec)
        if mode == "optimistic":
            p_win = 1.0
            share = 1.0
        else:
            p_win = _clip_probability(self.models.pretrade_win_probability(context, quote))
            share = (
                1.0
                if mode == "win_only"
                else float(np.clip(self.models.pretrade_participation_share(context, quote), 0.0, 1.0))
            )
        expected_fill = min(self.inventory, quote.offer_quantity) * p_win * share
        expected_fill = max(min(expected_fill, self.inventory, quote.offer_quantity), 0.0)
        support = float(np.clip(self.models.support_score(context, quote), 0.0, 1.0))
        return {
            "action_id": float(action_id),
            "is_no_quote": 0.0,
            "price_offset_ratio": float(quote.price_offset_ratio),
            "price_offset_dollar": float(quote.price_offset_dollar),
            "raw_price_offset_dollar": float(quote.raw_price_offset_dollar),
            "effective_spread": float(quote.effective_spread),
            "spread_unit": float(quote.spread_unit),
            "spread_source_index": float(quote.spread_source_index),
            "spread_is_fallback": float(quote.spread_is_fallback),
            "offset_was_clipped": float(quote.offset_was_clipped),
            "quantity_fraction": float(spec.quantity_fraction),
            "offer_price": float(quote.offer_price),
            "offer_quantity": float(quote.offer_quantity),
            "win_probability": float(p_win),
            "participation_share": float(share),
            "expected_fill_quantity": float(expected_fill),
            "expected_remaining_inventory": float(max(self.inventory - expected_fill, 0.0)),
            "support_score": support,
        }

    def target_inventory_at(self, time: pd.Timestamp) -> float:
        """Public read-only wrapper for the episode's linear inventory schedule."""
        return float(self._target_inventory(time))

    def build_state(self) -> np.ndarray:
        context = self._model_context()
        values: List[float] = []
        values.extend(float(self.episode.static_features[col]) for col in self.state_schema.static_feature_cols)
        values.extend(float(self.current_snapshot.features[col]) for col in self.state_schema.snapshot_feature_cols)

        horizon_minutes = max((self.episode.end_time - self.episode.start_time).total_seconds() / 60.0, 1.0)
        age_minutes = max((self.current_time - self.episode.start_time).total_seconds() / 60.0, 0.0)
        time_to_horizon = max((self.episode.end_time - self.current_time).total_seconds() / 60.0, 0.0)
        delta_minutes = max((self.current_time - self.last_decision_time).total_seconds() / 60.0, 0.0)
        quote_age = (
            max((self.current_time - self.active_quote.posted_time).total_seconds() / 60.0, 0.0)
            if self.active_quote is not None
            else 0.0
        )
        live_fraction = (
            self.active_quote.live_quantity / max(self.inventory, 1e-12)
            if self.active_quote is not None and self.inventory > 0
            else 0.0
        )
        previous_spec = (
            self.action_grid.decode(self.previous_action_id)
            if self.previous_action_id is not None
            else ActionSpec(-1, 0.0, 0.0, True)
        )
        spread = self._effective_spread_info(context, max(self.inventory, self.config.min_lot))
        previous_dollar_offset = (
            self.active_quote.price_offset_dollar if self.active_quote is not None else 0.0
        )
        dynamic = (
            self.inventory,
            self.inventory / self.episode.starting_inventory,
            age_minutes,
            age_minutes / horizon_minutes,
            time_to_horizon,
            time_to_horizon / horizon_minutes,
            delta_minutes,
            self.current_snapshot.fair_mark,
            self.episode.cost_basis - self.current_snapshot.fair_mark,
            self.cumulative_fill / self.episode.starting_inventory,
            live_fraction,
            quote_age,
            spread["effective_spread"],
            spread["spread_unit"],
            math.log(max(spread["effective_spread"], 1.0e-12)),
            spread["spread_age_minutes"],
            spread["spread_source_index"],
            spread["spread_is_fallback"],
            float(previous_spec.price_offset_ratio or 0.0),
            float(previous_dollar_offset),
            float(previous_spec.quantity_fraction),
            float(self.quote_update_count),
            self.last_own_fill / self.episode.starting_inventory,
        )
        values.extend(dynamic)
        observable = self._observable_trade_features()
        values.extend(observable[name] for name in self.OBSERVABLE_TRADE_FEATURE_NAMES)
        values.extend(
            (
                float("clock" in self.last_trigger_reasons),
                float("own_fill" in self.last_trigger_reasons),
                float("trade_publish" in self.last_trigger_reasons),
                float("reset" in self.last_trigger_reasons),
            )
        )

        if self.state_schema.include_action_grid_features:
            for action_id in range(self.action_grid.num_actions):
                spec = self.action_grid.decode(action_id)
                if spec.is_no_quote or self.inventory <= 0:
                    values.extend((0.0,) * len(self.ACTION_GRID_FEATURE_NAMES))
                    continue
                quote = self._candidate_from_spec(context, spec)
                anchor = self.models.pricing_anchor(context, quote.offer_quantity)
                p_win = _clip_probability(self.models.pretrade_win_probability(context, quote))
                share = float(np.clip(self.models.pretrade_participation_share(context, quote), 0.0, 1.0))
                expected_fill_fraction = (
                    min(quote.offer_quantity, self.inventory) * p_win * share / max(self.inventory, 1e-12)
                )
                values.extend(
                    (
                        anchor - context.fair_mark,
                        quote.offer_price - context.fair_mark,
                        quote.price_offset_ratio,
                        quote.price_offset_dollar,
                        quote.effective_spread,
                        quote.spread_unit,
                        p_win,
                        share,
                        expected_fill_fraction,
                    )
                )

        state = np.asarray(values, dtype=np.float32)
        if state.shape != (len(self.state_feature_names),):
            raise RuntimeError(
                f"State shape {state.shape} does not match feature names {len(self.state_feature_names)}"
            )
        if not np.isfinite(state).all():
            bad = [self.state_feature_names[i] for i in np.flatnonzero(~np.isfinite(state))[:10]]
            raise FloatingPointError(f"State contains non-finite values in: {bad}")
        if self.config.state_clip_abs > 0:
            state = np.clip(state, -self.config.state_clip_abs, self.config.state_clip_abs)
        return state

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_grid.num_actions, dtype=bool)
        if self.inventory <= 0 or self.done:
            return mask
        context = self._model_context()
        for action_id in range(self.action_grid.num_actions):
            spec = self.action_grid.decode(action_id)
            if spec.is_no_quote:
                mask[action_id] = True
                continue
            quote = self._candidate_from_spec(context, spec)
            price_delta = quote.offer_price - context.fair_mark
            valid = (
                quote.offer_quantity >= self.config.min_lot
                and quote.offer_quantity <= self.inventory + 1e-9
                and self.config.min_price_delta_from_mark <= price_delta <= self.config.max_price_delta_from_mark
                and (not self.config.mask_if_offset_clipped or not quote.offset_was_clipped)
                and self.models.support_score(context, quote) >= self.config.support_threshold
            )
            mask[action_id] = bool(valid)
        if not mask.any():
            no_quote = self.action_grid.no_quote_action_id()
            if no_quote is None:
                raise RuntimeError("No valid actions and no no-quote action configured")
            mask[no_quote] = True
        return mask

    # -------------------------- state names -----------------------------

    def _make_state_feature_names(self) -> Tuple[str, ...]:
        names: List[str] = []
        names.extend(f"static.{name}" for name in self.state_schema.static_feature_cols)
        names.extend(f"snapshot.{name}" for name in self.state_schema.snapshot_feature_cols)
        names.extend(f"dynamic.{name}" for name in self.DYNAMIC_FEATURE_NAMES)
        names.extend(f"trade.{name}" for name in self.OBSERVABLE_TRADE_FEATURE_NAMES)
        names.extend(f"event.{name}" for name in self.TRIGGER_FEATURE_NAMES)
        if self.state_schema.include_action_grid_features:
            for action_id in range(self.action_grid.num_actions):
                for feature in self.ACTION_GRID_FEATURE_NAMES:
                    names.append(f"action_{action_id}.{feature}")
        return tuple(names)

    # -------------------------- timeline -------------------------------

    def _build_timeline(self) -> None:
        events: List[TimelineEvent] = []
        sequence = 0
        for snapshot in self.episode.snapshots:
            if self.episode.start_time < snapshot.observable_time <= self.episode.end_time:
                events.append(TimelineEvent(snapshot.observable_time, 0, sequence, "snapshot", snapshot))
                sequence += 1
        allowed = set(self.config.allowed_trade_types)
        for trade in self.episode.trades:
            if trade.trade_type not in allowed:
                continue
            if self.episode.start_time < trade.execution_time <= self.episode.end_time:
                events.append(TimelineEvent(trade.execution_time, 1, sequence, "trade_exec", trade))
                sequence += 1
            if self.episode.start_time < trade.publish_time <= self.episode.end_time:
                events.append(TimelineEvent(trade.publish_time, 2, sequence, "trade_publish", trade))
                sequence += 1
        clock_time = self.episode.start_time + pd.Timedelta(minutes=self.config.clock_minutes)
        while clock_time < self.episode.end_time:
            events.append(TimelineEvent(clock_time, 3, sequence, "clock", None))
            sequence += 1
            clock_time += pd.Timedelta(minutes=self.config.clock_minutes)
        events.append(TimelineEvent(self.episode.end_time, 4, sequence, "terminal", None))
        events.sort()
        self._events = events

    def _set_latest_snapshot_at_or_before(self, time: pd.Timestamp, accrue_mtm: bool) -> None:
        candidates = [s for s in self.episode.snapshots if s.observable_time <= time]
        if not candidates:
            raise ValueError(f"No snapshot observable by {time}")
        snapshot = candidates[-1]
        if accrue_mtm:
            self._apply_snapshot(snapshot)
        else:
            self.current_snapshot = snapshot

    def _apply_snapshot(self, snapshot: Snapshot) -> None:
        old_mark = self.current_snapshot.fair_mark
        new_mark = snapshot.fair_mark
        mtm = (self.inventory / self.config.price_notional_divisor) * (new_mark - old_mark)
        self._interval_reward_components["remaining_inventory_mtm"] += mtm
        self.current_snapshot = snapshot

    def _accrue_carry_to(self, new_time: pd.Timestamp) -> None:
        if new_time < self._last_accrual_time:
            raise RuntimeError("Timeline moved backwards")
        delta_minutes = (new_time - self._last_accrual_time).total_seconds() / 60.0
        if delta_minutes <= 0:
            return
        risk = max(
            float(self.current_snapshot.features.get(self.reward_config.risk_feature_name, 1.0)),
            0.0,
        )
        interval_units = delta_minutes / self.config.base_minutes
        penalty = (
            self.reward_config.inventory_lambda
            * (self.inventory / self.episode.starting_inventory) ** 2
            * risk
            * interval_units
        )
        self._interval_reward_components["inventory_penalty"] -= penalty
        self._last_accrual_time = new_time

    # -------------------------- quoting -------------------------------

    def _post_quote(self, action_id: int) -> None:
        context = self._model_context()
        spec = self.action_grid.decode(action_id)
        previous_spec = (
            self.action_grid.decode(self.previous_action_id)
            if self.previous_action_id is not None
            else ActionSpec(-1, 0.0, 0.0, True)
        )
        changed = self.previous_action_id is not None and action_id != self.previous_action_id
        candidate = None if spec.is_no_quote else self._candidate_from_spec(context, spec)
        previous_dollar = self.active_quote.price_offset_dollar if self.active_quote is not None else 0.0
        current_dollar = candidate.price_offset_dollar if candidate is not None else 0.0
        current_spread = (
            candidate.effective_spread
            if candidate is not None
            else (self.active_quote.effective_spread if self.active_quote is not None else 1.0)
        )
        smooth = 0.0 if self.previous_action_id is None else (
            self.reward_config.price_smooth_lambda
            * abs(float(spec.price_offset_ratio or 0.0) - float(previous_spec.price_offset_ratio or 0.0))
            + self.reward_config.price_dollar_smooth_lambda
            * abs(current_dollar - previous_dollar)
            / max(current_spread, 1.0e-6)
            + self.reward_config.quantity_smooth_lambda
            * abs(float(spec.quantity_fraction) - float(previous_spec.quantity_fraction))
        )
        self._interval_reward_components["smoothness_penalty"] -= smooth
        if changed:
            self._interval_reward_components["quote_update_cost"] -= self.reward_config.update_cost

        if spec.is_no_quote:
            self.active_quote = None
        else:
            assert candidate is not None
            self.active_quote = ActiveQuote(
                action_id=action_id,
                price_offset_ratio=float(candidate.price_offset_ratio),
                price_offset_dollar=float(candidate.price_offset_dollar),
                effective_spread=float(candidate.effective_spread),
                spread_unit=float(candidate.spread_unit),
                quantity_fraction=float(spec.quantity_fraction),
                offer_price=candidate.offer_price,
                initial_quantity=candidate.offer_quantity,
                live_quantity=candidate.offer_quantity,
                posted_time=self.current_time,
            )
        self.previous_action_id = action_id
        self.quote_update_count += 1
        self.last_own_fill = 0.0

    def _effective_spread_info(self, context: ModelContext, quantity: float) -> Dict[str, float]:
        raw_spread: Optional[float] = None
        source_index = -1.0
        is_fallback = 0.0

        model_method = getattr(self.models, "effective_spread", None)
        if callable(model_method):
            try:
                candidate = float(model_method(context, quantity))
            except (TypeError, ValueError):
                candidate = math.nan
            if math.isfinite(candidate) and candidate > 0:
                raw_spread = candidate
                source_index = -1.0

        if raw_spread is None:
            for index, feature_name in enumerate(self.config.spread_feature_names):
                value = context.snapshot_features.get(feature_name)
                if value is None:
                    continue
                candidate = float(value)
                if math.isfinite(candidate) and candidate > 0:
                    raw_spread = candidate
                    source_index = float(index)
                    break

        if raw_spread is None:
            raw_spread = self.config.fallback_spread
            source_index = float(len(self.config.spread_feature_names))
            is_fallback = 1.0

        effective_spread = float(np.clip(raw_spread, self.config.spread_floor, self.config.spread_cap))
        spread_unit = effective_spread * self.config.spread_unit_multiplier
        age_value = context.snapshot_features.get(self.config.spread_age_feature_name, 0.0)
        try:
            spread_age_candidate = float(age_value)
        except (TypeError, ValueError):
            spread_age_candidate = 0.0
        spread_age = spread_age_candidate if math.isfinite(spread_age_candidate) else 0.0
        return {
            "raw_spread": float(raw_spread),
            "effective_spread": effective_spread,
            "spread_unit": float(max(spread_unit, 1.0e-12)),
            "spread_source_index": source_index,
            "spread_is_fallback": is_fallback,
            "spread_age_minutes": max(spread_age, 0.0),
        }

    def _candidate_from_spec(self, context: ModelContext, spec: ActionSpec) -> QuoteCandidate:
        if spec.is_no_quote:
            return QuoteCandidate(spec, math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, False)
        quantity = _round_down_lot(
            min(context.inventory, spec.quantity_fraction * context.inventory), self.config.min_lot
        )
        anchor = self.models.pricing_anchor(context, quantity)
        spread = self._effective_spread_info(context, quantity)
        ratio = float(spec.price_offset_ratio or 0.0)
        raw_dollar_offset = ratio * spread["spread_unit"]
        dollar_offset = float(
            np.clip(
                raw_dollar_offset,
                -self.config.max_absolute_price_offset,
                self.config.max_absolute_price_offset,
            )
        )
        clipped = not math.isclose(raw_dollar_offset, dollar_offset, rel_tol=0.0, abs_tol=1.0e-12)
        offer_price = anchor + dollar_offset
        return QuoteCandidate(
            action=spec,
            offer_price=float(offer_price),
            offer_quantity=float(quantity),
            price_offset_ratio=ratio,
            price_offset_dollar=dollar_offset,
            raw_price_offset_dollar=float(raw_dollar_offset),
            effective_spread=spread["effective_spread"],
            spread_unit=spread["spread_unit"],
            spread_source_index=spread["spread_source_index"],
            spread_is_fallback=spread["spread_is_fallback"],
            offset_was_clipped=clipped,
        )

    def _quote_life_allows_trigger(self, trigger: str) -> bool:
        if self.active_quote is None or self.config.min_quote_life_minutes <= 0:
            return True
        if trigger == "own_fill" and self.config.own_fill_overrides_min_quote_life:
            return True
        age = (self.current_time - self.active_quote.posted_time).total_seconds() / 60.0
        return age >= self.config.min_quote_life_minutes

    # -------------------------- fill simulator ------------------------

    def _process_trade_execution(self, trade: TradeEvent) -> Dict[str, Any]:
        context = self._model_context(time_override=trade.execution_time)
        result: Dict[str, Any] = {
            "event_id": trade.event_id,
            "execution_time": str(trade.execution_time),
            "publish_time": str(trade.publish_time),
            "trade_price": trade.price,
            "trade_quantity": trade.quantity,
            "trade_type": trade.trade_type,
            "demand_arrival": 1,
            "eligible": 0,
            "win_probability": 0.0,
            "won": 0.0,
            "participation_share": 0.0,
            "quantity_model_definition": self.config.quantity_model_definition,
            "capacity": 0.0,
            "filled_quantity": 0.0,
            "execution_price": math.nan,
        }
        if self.active_quote is None or self.active_quote.live_quantity < self.config.min_lot:
            return result

        spec = self.action_grid.decode(self.active_quote.action_id)
        raw_offset = self.active_quote.price_offset_ratio * self.active_quote.spread_unit
        quote = QuoteCandidate(
            action=spec,
            offer_price=self.active_quote.offer_price,
            offer_quantity=self.active_quote.live_quantity,
            price_offset_ratio=self.active_quote.price_offset_ratio,
            price_offset_dollar=self.active_quote.price_offset_dollar,
            raw_price_offset_dollar=raw_offset,
            effective_spread=self.active_quote.effective_spread,
            spread_unit=self.active_quote.spread_unit,
            spread_source_index=0.0,
            spread_is_fallback=0.0,
            offset_was_clipped=not math.isclose(
                raw_offset, self.active_quote.price_offset_dollar, rel_tol=0.0, abs_tol=1.0e-12
            ),
        )
        haircut = max(float(self.models.customer_price_haircut(context, trade)), 0.0)
        tolerance = max(float(self.models.eligibility_tolerance(context, trade)), 0.0)
        demand_price = trade.price - haircut
        eligible = quote.offer_price <= demand_price + tolerance
        result.update(
            {
                "demand_price": demand_price,
                "haircut": haircut,
                "tolerance": tolerance,
                "eligible": int(eligible),
            }
        )
        if not eligible:
            self._apply_optional_missed_demand_penalty(trade, quote, capacity=0.0)
            return result

        capacity = max(min(self.inventory, self.active_quote.live_quantity, trade.quantity), 0.0)
        capacity = _round_down_lot(capacity, self.config.min_lot)
        result["capacity"] = capacity
        if capacity < self.config.min_lot:
            return result

        if self.config.simulator_mode == "optimistic":
            p_win = 1.0
            won_factor = 1.0
            share = 1.0
        else:
            p_win = _clip_probability(self.models.event_win_probability(context, quote, trade, demand_price))
            if self.config.stochastic_fills:
                won_factor = float(self.rng.random() < p_win)
            else:
                won_factor = p_win
            if self.config.simulator_mode == "win_only":
                share = 1.0
            else:
                share = float(
                    np.clip(
                        self.models.event_participation_share(context, quote, trade, demand_price),
                        0.0,
                        1.0,
                    )
                )

        if self.config.quantity_model_definition == "participation_share":
            raw_filled = capacity * won_factor * share
        else:
            # Existing FillRatioModel interpretation:
            # ratio = filled quantity / displayed quote quantity.
            raw_filled = won_factor * min(
                self.inventory,
                trade.quantity,
                self.active_quote.live_quantity * share,
            )
        filled = _round_down_lot(raw_filled, self.config.min_lot)
        filled = min(filled, capacity, self.inventory, self.active_quote.live_quantity)
        result.update(
            {
                "win_probability": p_win,
                "won": won_factor,
                "participation_share": share,
                "filled_quantity": filled,
                "execution_price": quote.offer_price if filled > 0 else math.nan,
            }
        )
        if filled <= 0:
            self._apply_optional_missed_demand_penalty(trade, quote, capacity=capacity)
            return result

        execution_pnl = (filled / self.config.price_notional_divisor) * (
            quote.offer_price - self.current_snapshot.fair_mark
        )
        self._interval_reward_components["execution_pnl"] += execution_pnl
        underpricing_gap = max(
            demand_price - quote.offer_price - self.reward_config.underpricing_tolerance, 0.0
        )
        underpricing_penalty = (
            self.reward_config.underpricing_lambda
            * (filled / self.config.price_notional_divisor)
            * underpricing_gap**2
        )
        self._interval_reward_components["underpricing_penalty"] -= underpricing_penalty

        self.inventory = max(self.inventory - filled, 0.0)
        self.active_quote.live_quantity = max(self.active_quote.live_quantity - filled, 0.0)
        self.cumulative_fill += filled
        self.last_own_fill += filled
        return result

    def _apply_optional_missed_demand_penalty(
        self, trade: TradeEvent, quote: QuoteCandidate, capacity: float
    ) -> None:
        if self.reward_config.missed_demand_lambda <= 0:
            return
        offered = quote.offer_quantity
        missed = max(min(self.inventory, trade.quantity) - offered, 0.0)
        urgency = self._urgency_score()
        penalty = (
            self.reward_config.missed_demand_lambda
            * urgency
            * (missed / self.episode.starting_inventory) ** 2
        )
        self._interval_reward_components["missed_demand_penalty"] -= penalty

    def _publish_trade(self, trade: TradeEvent) -> None:
        self._last_published_trade = (trade.publish_time, trade)
        self._published_trades.append((trade.publish_time, trade))
        cutoff = trade.publish_time - pd.Timedelta(minutes=self.config.rolling_trade_minutes)
        while self._published_trades and self._published_trades[0][0] < cutoff:
            self._published_trades.popleft()

    # -------------------------- reward and terminal --------------------

    def _apply_schedule_penalty(self, elapsed_minutes: float) -> None:
        target = self._target_inventory(self.current_time)
        shortfall = max(self.inventory - target, 0.0) / self.episode.starting_inventory
        interval_units = elapsed_minutes / self.config.base_minutes
        penalty = self.reward_config.schedule_lambda * shortfall**2 * interval_units
        self._interval_reward_components["schedule_penalty"] -= penalty

    def _apply_terminal_liquidation(self) -> None:
        liquidation_price = self.current_snapshot.fair_mark - self.reward_config.liquidation_concession
        terminal_execution = (self.inventory / self.config.price_notional_divisor) * (
            liquidation_price - self.current_snapshot.fair_mark
        )
        terminal_penalty = self.reward_config.terminal_lambda * (
            self.inventory / self.episode.starting_inventory
        ) ** 2
        self._interval_reward_components["terminal_liquidation"] += terminal_execution
        self._interval_reward_components["terminal_inventory_penalty"] -= terminal_penalty

    def _terminate_at(self, time: pd.Timestamp, reason: str) -> None:
        self.current_time = min(max(time, self.current_time), self.episode.end_time)
        self.done = True

    def _target_inventory(self, time: pd.Timestamp) -> float:
        total = max((self.episode.end_time - self.episode.start_time).total_seconds(), 1.0)
        elapsed = min(max((time - self.episode.start_time).total_seconds(), 0.0), total)
        return self.episode.starting_inventory * (1.0 - elapsed / total)

    def _urgency_score(self) -> float:
        total = max((self.episode.end_time - self.episode.start_time).total_seconds(), 1.0)
        remaining = max((self.episode.end_time - self.current_time).total_seconds(), 0.0)
        return 1.0 + (1.0 - remaining / total)

    # -------------------------- observable state -----------------------

    def _model_context(self, time_override: Optional[pd.Timestamp] = None) -> ModelContext:
        return ModelContext(
            episode_id=self.episode.episode_id,
            cusip=self.episode.cusip,
            time=time_override or self.current_time,
            fair_mark=self.current_snapshot.fair_mark,
            inventory=self.inventory,
            starting_inventory=self.episode.starting_inventory,
            cost_basis=self.episode.cost_basis,
            static_features=self.episode.static_features,
            snapshot_features=self.current_snapshot.features,
            observable_trade_features=self._observable_trade_features(),
        )

    def _observable_trade_features(self) -> Dict[str, float]:
        cutoff = self.current_time - pd.Timedelta(minutes=self.config.rolling_trade_minutes)
        while self._published_trades and self._published_trades[0][0] < cutoff:
            self._published_trades.popleft()

        if self._last_published_trade is None:
            return {
                "last_trade_price": self.current_snapshot.fair_mark,
                "last_trade_quantity": 0.0,
                "last_trade_type_sign": 0.0,
                "minutes_since_last_published_trade": 1.0e4,
                "rolling_trade_count": 0.0,
                "rolling_trade_volume": 0.0,
                "rolling_trade_vwap": self.current_snapshot.fair_mark,
                "rolling_trade_direction_imbalance": 0.0,
            }
        last_publish, last_trade = self._last_published_trade
        quantities = np.asarray([trade.quantity for _, trade in self._published_trades], dtype=float)
        prices = np.asarray([trade.price for _, trade in self._published_trades], dtype=float)
        signs = np.asarray([_trade_type_sign(trade.trade_type) for _, trade in self._published_trades], dtype=float)
        total_volume = float(quantities.sum()) if len(quantities) else 0.0
        vwap = (
            float(np.dot(prices, quantities) / total_volume)
            if total_volume > 0
            else self.current_snapshot.fair_mark
        )
        imbalance = float(np.dot(signs, quantities) / total_volume) if total_volume > 0 else 0.0
        return {
            "last_trade_price": float(last_trade.price),
            "last_trade_quantity": float(last_trade.quantity),
            "last_trade_type_sign": _trade_type_sign(last_trade.trade_type),
            "minutes_since_last_published_trade": max(
                (self.current_time - last_publish).total_seconds() / 60.0, 0.0
            ),
            "rolling_trade_count": float(len(self._published_trades)),
            "rolling_trade_volume": total_volume,
            "rolling_trade_vwap": vwap,
            "rolling_trade_direction_imbalance": imbalance,
        }

    def _base_info(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "cusip": self.episode.cusip,
            "time": str(self.current_time),
            "inventory": self.inventory,
            "starting_inventory": self.episode.starting_inventory,
            "fair_mark": self.current_snapshot.fair_mark,
            "decision_count": self.decision_count,
            "quote_update_count": self.quote_update_count,
            "active_quote": dataclasses.asdict(self.active_quote) if self.active_quote else None,
        }

    @staticmethod
    def _empty_reward_components() -> Dict[str, float]:
        return {
            "execution_pnl": 0.0,
            "remaining_inventory_mtm": 0.0,
            "inventory_penalty": 0.0,
            "schedule_penalty": 0.0,
            "smoothness_penalty": 0.0,
            "quote_update_cost": 0.0,
            "missed_demand_penalty": 0.0,
            "underpricing_penalty": 0.0,
            "terminal_liquidation": 0.0,
            "terminal_inventory_penalty": 0.0,
        }


def _round_down_lot(quantity: float, lot: float) -> float:
    if quantity <= 0:
        return 0.0
    return math.floor((quantity + 1e-12) / lot) * lot


def _clip_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _trade_type_sign(trade_type: str) -> float:
    trade_type = trade_type.upper()
    if trade_type == "S":
        return 1.0
    if trade_type == "P":
        return -1.0
    return 0.0


__all__ = [
    "ActionGrid",
    "ActiveQuote",
    "HeuristicModelBundle",
    "HybridMuniReplayEnv",
    "ModelContext",
    "QuoteCandidate",
    "ReplayDataset",
    "ReplayEpisode",
    "RewardConfig",
    "SimulatorConfig",
    "SimulatorModelBundle",
    "Snapshot",
    "StateSchema",
    "StepResult",
    "TradeEvent",
    "load_model_bundle",
]
