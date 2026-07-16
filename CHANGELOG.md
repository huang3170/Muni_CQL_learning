# Changelog

## v3.0

- Added a forecast-aware one-step greedy expected-value baseline over all 36 actions.
- Added non-mutating, point-in-time `HybridMuniReplayEnv.preview_action()` diagnostics.
- Added RL-versus-greedy evaluation under optimistic, win-only and partial-fill assumptions.
- Added greedy replay CSV exports with per-action score components.
- Added automatic held-out greedy evaluation during simulator-online training.
- Added CLI controls for forecast weight/confidence and penalty ablations.
- Added tests proving the greedy policy does not inspect the next trade and reacts directionally to forward-price signals.

## v2.0

- Added hybrid 30-minute, own-fill, and trade-publication quote triggers.
- Added execution-time versus publication-time event ordering.
- Added semi-Markov elapsed-time discounts.
- Added four-layer demand/eligibility/win/quantity simulation.
- Added participation-share and legacy displayed-quote-ratio quantity definitions.
- Added dynamic live quote quantity and inventory state transitions.
- Added simulator-online replay-buffer training with epsilon-greedy exploration.
- Added optional fixed historical replay mixing and decaying CQL.
- Added held-out evaluation across optimistic, win-only, and partial-fill simulators.
- Added production side-model adapter template and canonical replay-data preparation.
- Added unit tests for leakage-sensitive timeline behavior and fill capacity.
