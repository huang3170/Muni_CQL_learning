#!/usr/bin/env python3
"""Prepare canonical data for the hybrid muni historical replay simulator.

The canonical output directory contains:

* positions.parquet
* snapshots.parquet
* trades.parquet
* schema.json

The simulator generates 30-minute clock events itself.  Trade execution and
publication remain separate timestamps so that the active quote can process the
trade before the trade becomes policy-visible.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("prepare_hybrid_replay")


CANONICAL_POSITION_COLUMNS = (
    "episode_id",
    "cusip",
    "start_time",
    "end_time",
    "starting_inventory",
    "cost_basis",
)
CANONICAL_SNAPSHOT_COLUMNS = ("episode_id", "observable_time", "fair_mark")
CANONICAL_TRADE_COLUMNS = (
    "episode_id",
    "event_id",
    "execution_time",
    "publish_time",
    "trade_price",
    "trade_quantity",
    "trade_type",
)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file type: {path}")



def write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    """Write parquet when available; fall back to pickle for lightweight demos."""
    parquet_path = output_dir / f"{stem}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        pickle_path = output_dir / f"{stem}.pkl"
        LOGGER.warning("pyarrow/fastparquet unavailable; writing %s", pickle_path)
        df.to_pickle(pickle_path)
        return pickle_path


def apply_column_map(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    """Rename source columns to canonical names.

    The JSON mapping is ``{"source_name": "canonical_name"}``.
    """

    return df.rename(columns=dict(mapping))


def require_columns(df: pd.DataFrame, required: Iterable[str], table: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{table} is missing columns: {missing}")


def coerce_numeric_finite(df: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{table}.{column} contains non-finite values")


def assign_episode_ids_to_trades(trades: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Assign a trade to the unique position episode covering its CUSIP/time."""

    if "episode_id" in trades.columns and trades["episode_id"].notna().all():
        trades["episode_id"] = trades["episode_id"].astype(str)
        return trades
    require_columns(trades, ("cusip", "execution_time"), "trades without episode_id")
    positions_by_cusip = {
        str(cusip): group.sort_values("start_time") for cusip, group in positions.groupby("cusip")
    }
    assigned: list[str] = []
    for row in trades.itertuples(index=False):
        candidates = positions_by_cusip.get(str(row.cusip))
        if candidates is None:
            assigned.append("")
            continue
        hit = candidates[
            (candidates["start_time"] <= row.execution_time)
            & (row.execution_time <= candidates["end_time"])
        ]
        if len(hit) > 1:
            raise ValueError(
                f"Trade at {row.execution_time} for {row.cusip} matches multiple episodes"
            )
        assigned.append(str(hit.iloc[0]["episode_id"]) if len(hit) == 1 else "")
    trades = trades.copy()
    trades["episode_id"] = assigned
    dropped = int((trades["episode_id"] == "").sum())
    if dropped:
        LOGGER.warning("dropping %d trades that do not map to a position episode", dropped)
        trades = trades[trades["episode_id"] != ""].copy()
    return trades


def assign_episode_ids_to_snapshots(snapshots: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    if "episode_id" in snapshots.columns and snapshots["episode_id"].notna().all():
        snapshots["episode_id"] = snapshots["episode_id"].astype(str)
        return snapshots
    require_columns(snapshots, ("cusip", "observable_time"), "snapshots without episode_id")
    positions_by_cusip = {
        str(cusip): group.sort_values("start_time") for cusip, group in positions.groupby("cusip")
    }
    output_rows: list[pd.Series] = []
    for _, row in snapshots.iterrows():
        candidates = positions_by_cusip.get(str(row["cusip"]))
        if candidates is None:
            continue
        # Allow a pre-start snapshot to seed an episode.  The latest such row is
        # retained later for every matching episode.
        for _, position in candidates.iterrows():
            if row["observable_time"] <= position["end_time"]:
                copied = row.copy()
                copied["episode_id"] = str(position["episode_id"])
                output_rows.append(copied)
    if not output_rows:
        raise ValueError("No snapshots mapped to episodes")
    return pd.DataFrame(output_rows)


def prepare_dataset(
    positions: pd.DataFrame,
    snapshots: pd.DataFrame,
    trades: pd.DataFrame,
    schema: Mapping[str, Any],
    output_dir: Path,
    default_publish_lag_minutes: float,
    allowed_trade_types: set[str],
) -> None:
    positions = positions.copy()
    snapshots = snapshots.copy()
    trades = trades.copy()

    require_columns(positions, CANONICAL_POSITION_COLUMNS, "positions")
    for col in ("start_time", "end_time"):
        positions[col] = pd.to_datetime(positions[col], errors="raise")
    positions["episode_id"] = positions["episode_id"].astype(str)
    positions["cusip"] = positions["cusip"].astype(str)
    coerce_numeric_finite(positions, ("starting_inventory", "cost_basis"), "positions")
    if positions["episode_id"].duplicated().any():
        dupes = positions.loc[positions["episode_id"].duplicated(), "episode_id"].head().tolist()
        raise ValueError(f"duplicate episode_id values: {dupes}")
    if (positions["start_time"] >= positions["end_time"]).any():
        raise ValueError("every position start_time must be before end_time")
    if (positions["starting_inventory"] <= 0).any():
        raise ValueError("starting_inventory must be positive")

    snapshots["observable_time"] = pd.to_datetime(snapshots["observable_time"], errors="raise")
    snapshots = assign_episode_ids_to_snapshots(snapshots, positions)
    require_columns(snapshots, CANONICAL_SNAPSHOT_COLUMNS, "snapshots")
    coerce_numeric_finite(snapshots, ("fair_mark",), "snapshots")

    trades["execution_time"] = pd.to_datetime(trades["execution_time"], errors="raise")
    if "publish_time" not in trades.columns:
        trades["publish_time"] = pd.NaT
    trades["publish_time"] = pd.to_datetime(trades["publish_time"], errors="coerce")
    trades["publish_time"] = trades["publish_time"].fillna(
        trades["execution_time"] + pd.to_timedelta(default_publish_lag_minutes, unit="m")
    )
    if "event_id" not in trades.columns:
        trades["event_id"] = [f"trade_{i}" for i in range(len(trades))]
    trades["trade_type"] = trades["trade_type"].astype(str).str.upper()
    trades = trades[trades["trade_type"].isin(allowed_trade_types)].copy()
    trades = assign_episode_ids_to_trades(trades, positions)
    require_columns(trades, CANONICAL_TRADE_COLUMNS, "trades")
    coerce_numeric_finite(trades, ("trade_price", "trade_quantity"), "trades")
    if (trades["publish_time"] < trades["execution_time"]).any():
        raise ValueError("publish_time cannot precede execution_time")
    if (trades["trade_quantity"] < 0).any():
        raise ValueError("trade_quantity must be nonnegative")

    static_cols = list(schema.get("static_feature_cols", []))
    snapshot_cols = list(schema.get("snapshot_feature_cols", []))
    require_columns(positions, static_cols, "positions static features")
    require_columns(snapshots, snapshot_cols, "snapshots state features")
    coerce_numeric_finite(positions, static_cols, "positions")
    coerce_numeric_finite(snapshots, snapshot_cols, "snapshots")

    position_lookup = positions.set_index("episode_id")
    filtered_snapshot_parts: list[pd.DataFrame] = []
    for episode_id, group in snapshots.groupby("episode_id"):
        if episode_id not in position_lookup.index:
            continue
        position = position_lookup.loc[episode_id]
        group = group.sort_values("observable_time")
        before = group[group["observable_time"] <= position["start_time"]].tail(1)
        during = group[
            (group["observable_time"] > position["start_time"])
            & (group["observable_time"] <= position["end_time"])
        ]
        combined = pd.concat([before, during], ignore_index=True)
        if combined.empty or combined.iloc[0]["observable_time"] > position["start_time"]:
            raise ValueError(
                f"episode {episode_id} has no snapshot observable by its start_time"
            )
        filtered_snapshot_parts.append(combined)
    snapshots = pd.concat(filtered_snapshot_parts, ignore_index=True)

    trades = trades.merge(
        positions[["episode_id", "start_time", "end_time"]], on="episode_id", how="inner"
    )
    trades = trades[
        (trades["execution_time"] > trades["start_time"])
        & (trades["execution_time"] <= trades["end_time"])
    ].drop(columns=["start_time", "end_time"])

    position_output_cols = list(CANONICAL_POSITION_COLUMNS) + static_cols
    snapshot_output_cols = list(CANONICAL_SNAPSHOT_COLUMNS) + snapshot_cols
    trade_output_cols = list(CANONICAL_TRADE_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(
        positions[position_output_cols].sort_values("episode_id"), output_dir, "positions"
    )
    write_table(
        snapshots[snapshot_output_cols].sort_values(["episode_id", "observable_time"]),
        output_dir,
        "snapshots",
    )
    write_table(
        trades[trade_output_cols].sort_values(["episode_id", "execution_time", "event_id"]),
        output_dir,
        "trades",
    )
    schema_output = {
        "static_feature_cols": static_cols,
        "snapshot_feature_cols": snapshot_cols,
        "include_action_grid_features": bool(schema.get("include_action_grid_features", True)),
        "quantity_unit": schema.get("quantity_unit", "par_thousands_or_internal_consistent_unit"),
        "price_unit": schema.get("price_unit", "dollar_price_per_100_par"),
        "default_publish_lag_minutes_used_when_missing": default_publish_lag_minutes,
        "allowed_trade_types": sorted(allowed_trade_types),
    }
    (output_dir / "schema.json").write_text(json.dumps(schema_output, indent=2), encoding="utf-8")

    summary = {
        "positions": len(positions),
        "snapshots": len(snapshots),
        "trades": len(trades),
        "episode_start": str(positions["start_time"].min()),
        "episode_end": str(positions["end_time"].max()),
        "trade_type_counts": trades["trade_type"].value_counts().to_dict(),
    }
    (output_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("wrote canonical replay dataset to %s", output_dir)


def make_demo(output_dir: Path, episodes: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    position_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    base_date = pd.Timestamp("2025-01-02 09:30:00")
    for i in range(episodes):
        day = base_date + pd.Timedelta(days=i)
        start = day
        end = day + pd.Timedelta(hours=6, minutes=30)
        episode_id = f"demo_{i:04d}"
        cusip = f"DEMO{i:05d}"
        starting_inventory = float(rng.choice([50, 75, 100, 150, 200]))
        base_mark = 100.0 + rng.normal(scale=0.7)
        position_rows.append(
            {
                "episode_id": episode_id,
                "cusip": cusip,
                "start_time": start,
                "end_time": end,
                "starting_inventory": starting_inventory,
                "cost_basis": base_mark - rng.normal(scale=0.25),
                "coupon": rng.uniform(2.0, 5.5),
                "duration": rng.uniform(2.0, 14.0),
                "rating_numeric": rng.uniform(1.0, 8.0),
                "amount_outstanding_log": rng.uniform(10.0, 15.0),
            }
        )
        mark = base_mark
        for minute in range(0, 391, 15):
            time = start + pd.Timedelta(minutes=minute)
            mark += rng.normal(scale=0.025)
            snapshot_rows.append(
                {
                    "episode_id": episode_id,
                    "observable_time": time,
                    "fair_mark": mark,
                    "cep_bid_ask_width": max(rng.normal(0.35, 0.08), 0.05),
                    "realized_volatility_30d": max(rng.normal(0.45, 0.12), 0.05),
                    "liquidity_score": rng.normal(),
                    "risk_score": max(rng.normal(1.0, 0.15), 0.25),
                    "price_scale": 1.0,
                    "forward_price_change_4h": rng.normal(scale=0.12),
                    "forward_price_up_probability": rng.uniform(0.25, 0.75),
                    "forward_price_down_probability": rng.uniform(0.25, 0.75),
                    "forward_model_confidence": rng.uniform(0.3, 0.9),
                }
            )
        n_trades = int(rng.poisson(7))
        trade_minutes = sorted(rng.choice(np.arange(5, 385), size=max(n_trades, 1), replace=False))
        for j, minute in enumerate(trade_minutes):
            execution = start + pd.Timedelta(minutes=int(minute))
            nearest_snapshot = min(snapshot_rows[-27:], key=lambda r: abs(r["observable_time"] - execution))
            trade_price = nearest_snapshot["fair_mark"] + rng.normal(0.18, 0.18)
            quantity = float(rng.choice([10, 20, 25, 50, 75, 100]))
            lag = int(rng.integers(1, 16))
            trade_rows.append(
                {
                    "episode_id": episode_id,
                    "event_id": f"{episode_id}_trade_{j}",
                    "execution_time": execution,
                    "publish_time": execution + pd.Timedelta(minutes=lag),
                    "trade_price": trade_price,
                    "trade_quantity": quantity,
                    "trade_type": "S",
                }
            )

    schema = {
        "static_feature_cols": [
            "coupon",
            "duration",
            "rating_numeric",
            "amount_outstanding_log",
        ],
        "snapshot_feature_cols": [
            "cep_bid_ask_width",
            "realized_volatility_30d",
            "liquidity_score",
            "risk_score",
            "price_scale",
            "forward_price_change_4h",
            "forward_price_up_probability",
            "forward_price_down_probability",
            "forward_model_confidence",
        ],
        "include_action_grid_features": True,
        "quantity_unit": "demo_internal_units",
        "price_unit": "dollar_price_per_100_par",
    }
    prepare_dataset(
        positions=pd.DataFrame(position_rows),
        snapshots=pd.DataFrame(snapshot_rows),
        trades=pd.DataFrame(trade_rows),
        schema=schema,
        output_dir=output_dir,
        default_publish_lag_minutes=15.0,
        allowed_trade_types={"S"},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Normalize real tables into the replay directory format")
    prep.add_argument("--positions", required=True)
    prep.add_argument("--snapshots", required=True)
    prep.add_argument("--trades", required=True)
    prep.add_argument("--schema-json", required=True)
    prep.add_argument("--column-map-json", help="Optional source->canonical column rename mapping")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--default-publish-lag-minutes", type=float, default=15.0)
    prep.add_argument("--allowed-trade-types", nargs="+", default=["S"])

    demo = sub.add_parser("make-demo", help="Create a synthetic hybrid replay dataset")
    demo.add_argument("--output-dir", required=True)
    demo.add_argument("--episodes", type=int, default=40)
    demo.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if args.command == "make-demo":
        make_demo(Path(args.output_dir), args.episodes, args.seed)
        return

    positions = read_table(args.positions)
    snapshots = read_table(args.snapshots)
    trades = read_table(args.trades)
    if args.column_map_json:
        mapping = json.loads(Path(args.column_map_json).read_text(encoding="utf-8"))
        positions = apply_column_map(positions, mapping.get("positions", {}))
        snapshots = apply_column_map(snapshots, mapping.get("snapshots", {}))
        trades = apply_column_map(trades, mapping.get("trades", {}))
    schema = json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
    prepare_dataset(
        positions=positions,
        snapshots=snapshots,
        trades=trades,
        schema=schema,
        output_dir=Path(args.output_dir),
        default_publish_lag_minutes=args.default_publish_lag_minutes,
        allowed_trade_types={value.upper() for value in args.allowed_trade_types},
    )


if __name__ == "__main__":
    main()
