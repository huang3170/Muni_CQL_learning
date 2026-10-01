"""Reproducible SYNTHETIC data only. Never mix these rows with trading data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_data(episodes: int = 2400, days: int = 50, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-07-15", periods=days, tz="America/New_York")
    rows = []
    for e in range(episodes):
        date = dates[e % days]
        start = (date + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
        inv = float(rng.choice([25000, 50000, 100000, 250000, 500000]))
        delta = float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]))
        l3 = int(rng.random() < 0.6)
        config, age = 0, 0.0
        for cycle in range(8):
            rate = 0.0018 * np.exp(-4 * delta + 0.15 * np.log(inv / 100000) + 0.45 * l3 - 0.18 * np.log1p(age / 30))
            waiting = rng.exponential(1 / rate)
            event = int(waiting <= 30)
            exposure = min(waiting, 30.0)
            end = start + pd.Timedelta(minutes=exposure)
            filled = float(inv if rng.random() < 0.4 else inv * 0.4) if event else 0.0
            identifier = f"SYN_E{e}_I{cycle}"
            rows.append({
                "interval_id": identifier, "position_episode_id": f"SYN_E{e}",
                "inventory_segment_id": f"SYN_E{e}_S0", "quote_config_id": f"SYN_E{e}_Q{config}",
                "cusip": f"SYN{e:06d}", "start_time_utc": start, "end_time_utc": end,
                "exposure_minutes": exposure, "end_reason": "FILL" if event else ("QUOTE_INACTIVE" if cycle == 7 else "TIME_SLICE_END"),
                "event": event, "fill_event_id": identifier + "_F" if event else None,
                "fill_time_utc": end if event else None,
                "fill_level": int(rng.choice([1, 2, 3] if l3 else [1, 2])) if event else None,
                "fill_par": filled, "inventory_par_start": inv,
                "l1_active": 1, "l2_active": 1, "l3_active": l3,
                "l1_price": 100 + delta, "l2_price": 100 + delta + 0.075,
                "l3_price": 100 + delta - 0.060 if l3 else None,
                "cep_mid": 100.0, "cep_asof_time_utc": start - pd.Timedelta(minutes=2),
                "delta_l1": delta, "gap_l2": 0.075, "gap_l3": -0.060 if l3 else None,
                "config_age_minutes": age, "l1_venue_set": "A", "l2_venue_set": "B|C",
                "l3_venue_set": "R" if l3 else "", "train_eligible": 1, "quality_reason": "OK",
                "duration_years": float(5 + e % 10), "rating_bucket": ["AA", "A", "BBB"][e % 3],
                "sector": ["GO", "Revenue"][e % 2],
            })
            if event:
                inv -= filled
                if inv < 1e-6:
                    break
                config += 1
                age = 0.0
            else:
                age += exposure
            # Synthetic assumption: after an early fill the next valid quote
            # starts at the next scheduled boundary; the intervening gap is idle.
            start += pd.Timedelta(minutes=30)
    return pd.DataFrame(rows)
