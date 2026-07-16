# Changelog

## v4 — Spread-relative price actions

- Replaced fixed dollar price-offset actions with dimensionless ratios.
- Default ratio grid: `[-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]`.
- Materializes dollar offset as `ratio * spread_unit`, with `spread_unit = 0.5 * effective_spread` by default.
- Added ordered spread-source fallback, spread floor/cap, fallback spread, and absolute offset cap.
- Added action masking when the raw spread-relative offset exceeds the absolute cap.
- Added effective spread, spread unit, source/fallback diagnostics, ratio, and dollar offset to state/action diagnostics.
- Updated greedy and RL policies to share the same action materialization.
- Updated quote smoothness to operate primarily on ratio changes, with optional spread-normalized dollar changes.
- Updated offline logged-action encoding to use half-spread ratios.
- Added tests for cross-CUSIP spread scaling, cap masking, and logged-action encoding.

## v3 — Forecast-aware greedy baseline

- Added deployable one-step greedy expected-value baseline.
- Added RL-versus-greedy held-out evaluation and replay exports.

## v2 — Hybrid simulator-online replay

- Added fixed-clock and trade-triggered quote decisions, four-layer fill simulation, variable-time discounts, and simulator-online Dueling Double DQN training.
