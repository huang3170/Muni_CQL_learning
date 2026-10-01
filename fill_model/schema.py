"""Map quote dataframe columns to model fields without fitting any statistics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuoteSchema:
    quantity_multiplier: float = 1.0
    maturity_unit: str = "days"
    liquidity_kind: str = "numeric"
    quote_end_is_first_fill: bool = True

    def __post_init__(self):
        if not np.isfinite(self.quantity_multiplier) or self.quantity_multiplier <= 0:
            raise ValueError("quantity_multiplier must be finite and positive.")
        if self.maturity_unit not in {"days", "years"}:
            raise ValueError("maturity_unit must be days or years, not bond duration.")
        if self.liquidity_kind not in {"numeric", "categorical"}:
            raise ValueError("liquidity_kind must be numeric or categorical.")


def _nonblank(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype("string").str.strip().ne("").fillna(False)


def _number(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").astype(float)
    if (_nonblank(values) & result.isna()).any():
        raise ValueError(f"{name} contains nonnumeric values; check units or categorical encoding.")
    if np.isinf(result).any():
        raise ValueError(f"{name} contains infinite values.")
    return result


def local_time(values: pd.Series) -> pd.Series:
    """Parse naive New York clock times without attaching or converting zones."""
    try:
        parsed = pd.to_datetime(values, format="mixed", errors="coerce")
    except ValueError as exc:
        raise ValueError("Use timezone-naive New York local timestamps consistently.") from exc
    if isinstance(parsed.dtype, pd.DatetimeTZDtype) or not pd.api.types.is_datetime64_dtype(parsed):
        raise ValueError("Use timezone-naive New York local timestamps; no timezone conversion is performed.")
    return parsed


def normalize_quote_states(raw: pd.DataFrame, schema: QuoteSchema | None = None) -> pd.DataFrame:
    """Accept current user names or canonical fields, including unlabeled inference.

    Canonical columns take precedence so prepared frames are idempotent. This
    function never reads end times, fill labels, exposure or episode outcomes.
    """
    schema = schema or QuoteSchema()
    df = raw.copy()
    for source, target in [("cycle_time", "start_time"), ("cep_time", "cep_asof_time")]:
        if target not in df and source in df:
            df[target] = local_time(df[source])
    for target in ["start_time", "cep_asof_time"]:
        if target in df:
            df[target] = local_time(df[target])
    for source, target in [("mid_price", "cep_mid"), ("rating", "rating_bucket")]:
        if target not in df and source in df:
            df[target] = df[source]
    if "quote_quantity" not in df and "quantity" in df:
        df["quote_quantity"] = _number(df.quantity, "quantity") * schema.quantity_multiplier
    if "time_to_maturity_years" not in df and "time_to_maturity" in df:
        df["time_to_maturity_years"] = _number(df.time_to_maturity, "time_to_maturity") / (
            365.25 if schema.maturity_unit == "days" else 1.)
    if "liquidity" in df:
        target = "liquidity_score" if schema.liquidity_kind == "numeric" else "liquidity_bucket"
        if target not in df:
            df[target] = _number(df.liquidity, "liquidity") if schema.liquidity_kind == "numeric" else df.liquidity
    for name in ["cep_mid", "l1_price", "l2_price", "l3_price", "l1_active", "l2_active", "l3_active",
                 "quote_quantity", "time_to_maturity_years", "coupon", "liquidity_score"]:
        if name in df:
            df[name] = _number(df[name], name)
    if "cycle_time" in raw:
        # Some source feeds retain an old price for an inactive level. It is
        # not an offered price; remove it before deriving the level gaps.
        for level in [1, 2, 3]:
            price, mask = f"l{level}_price", f"l{level}_active"
            if {price, mask}.issubset(df.columns):
                df[price] = df[price].mask(df[mask].eq(0))
    if "market_spread" not in df:
        width = _number(df.cep_bid_ask_width, "cep_bid_ask_width") if "cep_bid_ask_width" in df else pd.Series(np.nan, index=df.index)
        if {"bid_price", "ask_price"}.issubset(df.columns):
            computed = _number(df.ask_price, "ask_price") - _number(df.bid_price, "bid_price")
            if (width.notna() & computed.notna() & (width - computed).abs().gt(1e-6)).any():
                raise ValueError("cep_bid_ask_width must equal ask_price - bid_price in price points.")
            width = width.fillna(computed)
        df["market_spread"] = width
    # Always derive action geometry from the current prices, never from outcome
    # fields or all three vs-mid columns, which would bypass the price bound.
    if {"l1_price", "cep_mid"}.issubset(df.columns) and "delta_l1" not in df:
        df["delta_l1"] = df.l1_price - df.cep_mid
    for level in [2, 3]:
        target = f"gap_l{level}"
        if target not in df and {f"l{level}_price", f"l{level}_active", "l1_price"}.issubset(df.columns):
            df[target] = (df[f"l{level}_price"] - df.l1_price).where(df[f"l{level}_active"].eq(1))
    return df


def normalize_quote_training(raw: pd.DataFrame, schema: QuoteSchema | None = None) -> pd.DataFrame:
    """Adapt the user's interval schema; do not invent censoring mechanisms.

    On filled rows quote_end_time/exposure must stop at the first execution.
    When real execution IDs are absent, the generated key detects repeated
    (CUSIP, execution time), but cannot validate trade-ledger linkage.
    """
    schema = schema or QuoteSchema()
    df = normalize_quote_states(raw, schema)
    user_schema = "cycle_time" in raw
    if "end_time" not in df and "quote_end_time" in df:
        df["end_time"] = local_time(df.quote_end_time)
    if "position_episode_id" not in df and "episode_id" in df:
        df["position_episode_id"] = df.episode_id
    if "first_fill_level" in df:
        # Presence defines the arrival label, regardless of level encoding.
        # Level validity belongs to the later routing model, not this model.
        present = _nonblank(df.first_fill_level)
        observed_event = present.astype(int)
        if "event" in df and _number(df.event, "event").ne(observed_event).any():
            raise ValueError("event conflicts with first_fill_level presence.")
        df["event"] = observed_event
        df["fill_level"] = df.first_fill_level.where(present)
    if "fill_par" not in df and "first_fill_quantity" in df:
        # Kept in source quantity units; not a par-dollar claim or a feature.
        df["fill_quantity"] = _number(df.first_fill_quantity, "first_fill_quantity")
        if "event" in df and (df.event.eq(0) & df.fill_quantity.fillna(0).ne(0)).any():
            raise ValueError("A no-fill row cannot have nonzero first_fill_quantity.")
    if user_schema:
        required = {"cusip", "start_time", "end_time", "event", "position_episode_id"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing quote interval fields: {sorted(required - set(df.columns))}")
        if "interval_id" not in df:
            df["interval_id"] = df.cusip.astype("string") + "|" + df.start_time.astype("string")
        if "quote_config_id" not in df:
            # Audit placeholder only; never used to infer configuration age.
            df["quote_config_id"] = df.interval_id
        if "end_reason" not in df:
            df["end_reason"] = np.where(df.event.eq(1), "FILL", "CENSORED")
        if "train_eligible" not in df:
            df["train_eligible"] = 1
        if "fill_time" not in df:
            if df.event.eq(1).any() and not schema.quote_end_is_first_fill:
                raise ValueError("Supply actual fill_time or confirm quote_end_is_first_fill=True; "
                                 "positive exposure must stop at the first fill.")
            df["fill_time"] = df.end_time.where(df.event.eq(1))
        if "fill_event_id" not in df:
            df["fill_event_id"] = (df.cusip.astype("string") + "|FILL|" +
                                   df.fill_time.astype("string")).where(df.event.eq(1))
    return df
