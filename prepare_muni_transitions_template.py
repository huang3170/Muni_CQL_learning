#!/usr/bin/env python3
"""Template: convert decision-level muni quote data into offline RL transitions.

This file is intentionally explicit rather than fully generic. Rename the input
columns in `ColumnConfig`, and edit the state feature list in a JSON file.

The output is a directory of memory-mapped-friendly NPY arrays consumed by
`muni_cql_dueling_ddqn.py`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PRICE_OFFSETS: Tuple[float, ...] = (-0.50, -0.25, -0.125, 0.0, 0.125, 0.25, 0.50)
QUANTITY_FRACTIONS: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
NUM_GRID_ACTIONS = len(PRICE_OFFSETS) * len(QUANTITY_FRACTIONS)
NO_QUOTE_ACTION_ID = NUM_GRID_ACTIONS
NUM_ACTIONS = NUM_GRID_ACTIONS + 1


@dataclass(frozen=True)
class ColumnConfig:
    episode_id: str = "episode_id"
    decision_time: str = "decision_time"
    inventory_before: str = "inventory_before"
    starting_inventory: str = "starting_inventory"
    logged_offer_price: str = "logged_offer_price"
    logged_offer_qty: str = "logged_offer_qty"
    pricing_anchor_logged_qty: str = "pricing_anchor_logged_qty"
    filled_qty: str = "filled_qty"
    execution_price: str = "execution_price"
    fair_mark: str = "fair_mark"
    target_inventory_next: str = "target_inventory_next"
    risk_score: str = "risk_score"
    urgency_score: str = "urgency_score"
    demand_qty: str = "demand_qty"
    demand_trade_price: str = "demand_trade_price"
    price_scale: str = "price_scale"


@dataclass(frozen=True)
class RewardConfig:
    inventory_lambda: float = 0.05
    schedule_lambda: float = 0.10
    price_smooth_lambda: float = 0.01
    quantity_smooth_lambda: float = 0.01
    missed_demand_lambda: float = 0.01
    underpricing_lambda: float = 0.0
    underpricing_tolerance: float = 0.125
    base_discount: float = 0.99
    base_minutes: float = 30.0
    min_lot: float = 5.0


def action_id_from_indices(price_index: int, quantity_index: int) -> int:
    return price_index * len(QUANTITY_FRACTIONS) + quantity_index


def decode_action(action_id: int) -> Tuple[Optional[float], float, bool]:
    if action_id == NO_QUOTE_ACTION_ID:
        return None, 0.0, True
    price_index = action_id // len(QUANTITY_FRACTIONS)
    quantity_index = action_id % len(QUANTITY_FRACTIONS)
    return PRICE_OFFSETS[price_index], QUANTITY_FRACTIONS[quantity_index], False


def nearest_logged_action(
    offer_price: float,
    offer_qty: float,
    pricing_anchor: float,
    inventory_before: float,
) -> int:
    if not np.isfinite(offer_qty) or offer_qty <= 0 or inventory_before <= 0:
        return NO_QUOTE_ACTION_ID
    price_offset = offer_price - pricing_anchor
    quantity_fraction = np.clip(offer_qty / inventory_before, 0.0, 1.0)
    price_index = int(np.argmin(np.abs(np.asarray(PRICE_OFFSETS) - price_offset)))
    quantity_index = int(np.argmin(np.abs(np.asarray(QUANTITY_FRACTIONS) - quantity_fraction)))
    return action_id_from_indices(price_index, quantity_index)


def round_down_lot(quantity: float, lot: float) -> float:
    if lot <= 0:
        return max(quantity, 0.0)
    return max(np.floor(quantity / lot) * lot, 0.0)


def build_action_mask(inventory: float, min_lot: float) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    if inventory <= 0:
        return mask
    for price_index in range(len(PRICE_OFFSETS)):
        for quantity_index, fraction in enumerate(QUANTITY_FRACTIONS):
            quote_qty = round_down_lot(fraction * inventory, min_lot)
            action_id = action_id_from_indices(price_index, quantity_index)
            mask[action_id] = quote_qty >= min_lot and quote_qty <= inventory
    mask[NO_QUOTE_ACTION_ID] = True
    return mask


def safe_series(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def validate_point_in_time_features(df: pd.DataFrame, state_cols: Sequence[str]) -> None:
    missing = [column for column in state_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing state columns: {missing}")
    feature_frame = df.loc[:, state_cols].apply(pd.to_numeric, errors="coerce")
    if feature_frame.isna().any().any():
        bad = feature_frame.columns[feature_frame.isna().any()].tolist()
        raise ValueError(
            "State features must be imputed before transition creation. "
            f"Columns with missing/non-numeric values: {bad}"
        )
    values = feature_frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("State features contain non-finite values")


def construct_transitions(
    raw_df: pd.DataFrame,
    state_cols: Sequence[str],
    columns: ColumnConfig,
    reward_cfg: RewardConfig,
) -> Dict[str, np.ndarray]:
    df = raw_df.copy()
    required = [
        columns.episode_id,
        columns.decision_time,
        columns.inventory_before,
        columns.starting_inventory,
        columns.logged_offer_price,
        columns.logged_offer_qty,
        columns.pricing_anchor_logged_qty,
        columns.filled_qty,
        columns.execution_price,
        columns.fair_mark,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[columns.decision_time] = pd.to_datetime(df[columns.decision_time], errors="raise")
    df = df.sort_values([columns.episode_id, columns.decision_time]).reset_index(drop=True)
    validate_point_in_time_features(df, state_cols)

    group = df.groupby(columns.episode_id, sort=False, group_keys=False)
    states = df.loc[:, state_cols].to_numpy(dtype=np.float32)
    next_state_df = group[list(state_cols)].shift(-1)
    is_last = group.cumcount(ascending=False).eq(0).to_numpy()
    next_state_df.loc[is_last, :] = df.loc[is_last, state_cols].to_numpy()
    next_states = next_state_df.to_numpy(dtype=np.float32)

    inventory_before = pd.to_numeric(df[columns.inventory_before], errors="raise").to_numpy(float)
    starting_inventory = pd.to_numeric(df[columns.starting_inventory], errors="raise").to_numpy(float)
    filled_qty = pd.to_numeric(df[columns.filled_qty], errors="coerce").fillna(0.0).to_numpy(float)
    inventory_after = np.maximum(inventory_before - filled_qty, 0.0)

    offer_price = pd.to_numeric(df[columns.logged_offer_price], errors="coerce").to_numpy(float)
    offer_qty = pd.to_numeric(df[columns.logged_offer_qty], errors="coerce").fillna(0.0).to_numpy(float)
    pricing_anchor = pd.to_numeric(df[columns.pricing_anchor_logged_qty], errors="coerce").to_numpy(float)
    execution_price = pd.to_numeric(df[columns.execution_price], errors="coerce").to_numpy(float)
    fair_mark = pd.to_numeric(df[columns.fair_mark], errors="raise").to_numpy(float)
    next_fair_mark = group[columns.fair_mark].shift(-1).fillna(df[columns.fair_mark]).to_numpy(float)

    actions = np.fromiter(
        (
            nearest_logged_action(p, q, anchor, inv)
            for p, q, anchor, inv in zip(offer_price, offer_qty, pricing_anchor, inventory_before)
        ),
        dtype=np.int64,
        count=len(df),
    )

    action_masks = np.stack(
        [build_action_mask(inv, reward_cfg.min_lot) for inv in inventory_before], axis=0
    )
    # Logged actions must be valid. If historical quantities do not align with the
    # proposed lot rules, explicitly retain their nearest bucket for offline learning.
    action_masks[np.arange(len(df)), actions] = True

    next_inventory_before = group[columns.inventory_before].shift(-1).to_numpy(float)
    next_inventory_before = np.where(is_last, inventory_after, next_inventory_before)
    next_action_masks = np.stack(
        [build_action_mask(inv, reward_cfg.min_lot) for inv in next_inventory_before], axis=0
    )
    next_action_masks[is_last, :] = False

    # Previous logged action, used only to calculate quote-change penalties here.
    action_series = pd.Series(actions, index=df.index)
    prev_action = action_series.groupby(df[columns.episode_id], sort=False).shift(1)
    prev_action = prev_action.fillna(action_series).to_numpy(dtype=np.int64)
    current_offsets = np.array([decode_action(int(a))[0] or 0.0 for a in actions], dtype=float)
    current_fractions = np.array([decode_action(int(a))[1] for a in actions], dtype=float)
    previous_offsets = np.array([decode_action(int(a))[0] or 0.0 for a in prev_action], dtype=float)
    previous_fractions = np.array([decode_action(int(a))[1] for a in prev_action], dtype=float)

    target_inventory_next = safe_series(df, columns.target_inventory_next, 0.0).to_numpy(float)
    risk_score = safe_series(df, columns.risk_score, 1.0).clip(lower=0.0).to_numpy(float)
    urgency = safe_series(df, columns.urgency_score, 1.0).clip(lower=0.0).to_numpy(float)
    demand_qty = safe_series(df, columns.demand_qty, 0.0).clip(lower=0.0).to_numpy(float)
    demand_trade_price = safe_series(df, columns.demand_trade_price, np.nan).to_numpy(float)
    price_scale = safe_series(df, columns.price_scale, 1.0).clip(lower=1e-6).to_numpy(float)

    next_time = group[columns.decision_time].shift(-1)
    delta_minutes = (
        (next_time - df[columns.decision_time]).dt.total_seconds().div(60.0).fillna(reward_cfg.base_minutes)
    ).clip(lower=1.0).to_numpy(float)
    discounts = np.power(
        reward_cfg.base_discount,
        delta_minutes / reward_cfg.base_minutes,
    ).astype(np.float32)

    # Economic wealth change. For no fill, execution contribution is zero.
    valid_execution_price = np.where(np.isfinite(execution_price), execution_price, fair_mark)
    execution_pnl = filled_qty * (valid_execution_price - fair_mark)
    remaining_inventory_mtm = inventory_after * (next_fair_mark - fair_mark)
    wealth_change = execution_pnl + remaining_inventory_mtm

    safe_starting_inventory = np.maximum(starting_inventory, 1e-6)
    interval_units = delta_minutes / reward_cfg.base_minutes
    inventory_penalty = (
        reward_cfg.inventory_lambda
        * np.square(inventory_after / safe_starting_inventory)
        * risk_score
        * interval_units
    )
    schedule_penalty = reward_cfg.schedule_lambda * np.square(
        np.maximum(inventory_after - target_inventory_next, 0.0) / safe_starting_inventory
    )
    smooth_penalty = (
        reward_cfg.price_smooth_lambda * np.abs(current_offsets - previous_offsets) / price_scale
        + reward_cfg.quantity_smooth_lambda * np.abs(current_fractions - previous_fractions)
    )
    missed_qty = np.maximum(np.minimum(inventory_before, demand_qty) - offer_qty, 0.0)
    missed_demand_penalty = (
        reward_cfg.missed_demand_lambda
        * urgency
        * np.square(missed_qty / safe_starting_inventory)
    )
    underpricing_gap = np.maximum(
        np.where(np.isfinite(demand_trade_price), demand_trade_price, valid_execution_price)
        - valid_execution_price
        - reward_cfg.underpricing_tolerance,
        0.0,
    )
    underpricing_penalty = (
        reward_cfg.underpricing_lambda
        * filled_qty
        * np.square(underpricing_gap / price_scale)
    )

    rewards = (
        wealth_change
        - inventory_penalty
        - schedule_penalty
        - smooth_penalty
        - missed_demand_penalty
        - underpricing_penalty
    ).astype(np.float32)

    dones = is_last.astype(np.float32)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "action_masks": action_masks,
        "next_action_masks": next_action_masks,
        "discounts": discounts,
    }


def save_npy_directory(arrays: Mapping[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array, allow_pickle=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", required=True, help="Input .parquet, .csv, or .pkl file")
    parser.add_argument("--feature-config", required=True, help="JSON containing a `state_cols` list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inventory-lambda", type=float, default=0.05)
    parser.add_argument("--schedule-lambda", type=float, default=0.10)
    parser.add_argument("--price-smooth-lambda", type=float, default=0.01)
    parser.add_argument("--quantity-smooth-lambda", type=float, default=0.01)
    parser.add_argument("--missed-demand-lambda", type=float, default=0.01)
    parser.add_argument("--underpricing-lambda", type=float, default=0.0)
    parser.add_argument("--underpricing-tolerance", type=float, default=0.125)
    parser.add_argument("--base-discount", type=float, default=0.99)
    parser.add_argument("--base-minutes", type=float, default=30.0)
    parser.add_argument("--min-lot", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.feature_config, "r", encoding="utf-8") as handle:
        feature_config = json.load(handle)
    state_cols = feature_config["state_cols"]
    input_path = Path(args.input_data)
    suffix = input_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(input_path)
    else:
        raise ValueError("--input-data must end in .parquet, .csv, .pkl, or .pickle")
    reward_cfg = RewardConfig(
        inventory_lambda=args.inventory_lambda,
        schedule_lambda=args.schedule_lambda,
        price_smooth_lambda=args.price_smooth_lambda,
        quantity_smooth_lambda=args.quantity_smooth_lambda,
        missed_demand_lambda=args.missed_demand_lambda,
        underpricing_lambda=args.underpricing_lambda,
        underpricing_tolerance=args.underpricing_tolerance,
        base_discount=args.base_discount,
        base_minutes=args.base_minutes,
        min_lot=args.min_lot,
    )
    arrays = construct_transitions(df, state_cols, ColumnConfig(), reward_cfg)
    output_dir = Path(args.output_dir)
    save_npy_directory(arrays, output_dir)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "state_cols": state_cols,
                "price_offsets": PRICE_OFFSETS,
                "quantity_fractions": QUANTITY_FRACTIONS,
                "no_quote_action_id": NO_QUOTE_ACTION_ID,
                "reward_config": reward_cfg.__dict__,
                "rows": len(df),
            },
            handle,
            indent=2,
        )
    print(f"Wrote {len(df):,} transitions to {output_dir}")


if __name__ == "__main__":
    main()
