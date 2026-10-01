"""Reproducible SYNTHETIC rows in the same schema as the supplied dataframe."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_data(episodes: int = 2400, days: int = 50, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-07-15", periods=days)
    rows = []
    for e in range(episodes):
        start = dates[e % days] + pd.Timedelta(hours=9, minutes=30)
        quantity = float(rng.choice([25000, 50000, 100000, 250000, 500000]))
        delta = float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]))
        l3_active = int(rng.random() < 0.6)
        liquidity = float(rng.uniform(0, 1))
        mid = float(rng.uniform(98, 104))
        width = float(rng.uniform(0.05, 0.30))
        for cycle in range(8):
            rate = 0.0025 * np.exp(-4 * delta + 0.15 * np.log(quantity / 100000)
                                    + 0.45 * l3_active + 0.25 * liquidity)
            waiting = rng.exponential(1 / rate)
            event = waiting <= 30
            exposure = min(waiting, 30.0)
            end = start + pd.Timedelta(minutes=exposure)
            filled = float(quantity if rng.random() < 0.4 else quantity * 0.4) if event else np.nan
            rows.append({
                "cusip": f"SYN{e:06d}", "quantity": quantity,
                "l1_price": mid + delta, "l2_price": mid + delta + 0.075,
                "l3_price": mid + delta - 0.060,
                "l1_active": 1, "l2_active": 1, "l3_active": l3_active,
                "cycle_time": start, "exposure_minutes": exposure,
                "episode_id": f"SYN_E{e}", "quote_end_time": end,
                "first_fill_level": int(rng.choice([1, 2, 3] if l3_active else [1, 2])) if event else None,
                "first_fill_quantity": filled,
                "cep_time": start - pd.Timedelta(minutes=2),
                "bid_price": mid - width / 2, "ask_price": mid + width / 2,
                "mid_price": mid, "cep_age_min": 2.0,
                "cep_bid_ask_width": width,
                "l1_vs_mid": delta, "l2_vs_mid": delta + 0.075,
                "l3_vs_mid": delta - 0.060,
                "time_to_maturity": float(365 * (5 + e % 10)),
                "rating": ["AA", "A", "BBB"][e % 3],
                "liquidity": liquidity, "coupon": float(3 + e % 4),
            })
            if event:
                quantity -= filled
                if quantity < 1e-6:
                    break
            # After an early fill, quoting resumes at the next scheduled cycle.
            start += pd.Timedelta(minutes=30)
    df = pd.DataFrame(rows)
    # Only ordinary input features receive missing values. Labels and interval
    # timestamps stay complete so the example has known exposure/outcomes.
    ordinary = ["quantity", "l1_price", "l2_price", "l3_price", "l1_active",
                "l2_active", "l3_active", "bid_price", "ask_price", "mid_price",
                "cep_age_min", "cep_bid_ask_width", "l1_vs_mid", "l2_vs_mid",
                "l3_vs_mid", "time_to_maturity", "rating", "coupon"]
    for column in ordinary + ["liquidity"]:
        fraction = 0.24 if column == "liquidity" else 0.02
        missing = rng.choice(df.index, size=round(fraction * len(df)), replace=False)
        df.loc[missing, column] = np.nan
    return df
