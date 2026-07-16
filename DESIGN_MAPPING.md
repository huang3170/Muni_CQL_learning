# Report-to-Code Mapping — v4

| Design item | Implementation |
|---|---|
| One CUSIP position lifecycle per episode | `ReplayEpisode`, `HybridMuniReplayEnv` |
| Fixed clock + own-fill + published-trade decisions | timeline construction and trigger handling in `hybrid_replay_simulator.py` |
| Old quote processes current trade | `_process_trade_execution()` runs before any triggered re-quote |
| Spread-relative price action | `ActionGrid.price_offset_ratios` and `_candidate_from_spec()` |
| Effective spread hierarchy | `_effective_spread_info()` |
| Half-spread materialization | `spread_unit_multiplier=0.5` in `SimulatorConfig` |
| Absolute business cap and mask | `max_absolute_price_offset`, `mask_if_offset_clipped`, `action_mask()` |
| Same action mapping for RL and greedy | both call `HybridMuniReplayEnv.preview_action()` / `_candidate_from_spec()` |
| Logged quote-to-action mapping | `nearest_logged_action()` in `prepare_muni_transitions_template.py` |
| Four-layer fill simulator | demand timeline, eligibility, win, participation share in `_process_trade_execution()` |
| Variable elapsed-time discount | `discount = base_discount ** (elapsed/base_minutes)` |
| Dueling Double DQN + optional CQL | `muni_cql_dueling_ddqn.py`, `simulator_online_train.py` |
| Forecast-aware greedy baseline | `greedy_baseline.py` |
| Point-in-time leakage controls | execution/publish separation and `preview_action()` |
