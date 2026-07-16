# Report-to-Code Mapping

| Report section | Code implementation |
|---|---|
| Episode and hybrid timeline | `ReplayEpisode`, `HybridMuniReplayEnv._build_timeline` |
| Old quote handles current trade | Timestamp-group processing in `HybridMuniReplayEnv.step` |
| Execution vs publication time | Separate `trade_exec` and `trade_publish` timeline events |
| Semi-Markov transition | `StepResult.elapsed_minutes` and `StepResult.discount` |
| State design | `HybridMuniReplayEnv.build_state` and `state_feature_names` |
| 36-action grid | `ActionGrid` in `muni_cql_dueling_ddqn.py` |
| Action masking | `HybridMuniReplayEnv.action_mask` |
| Demand arrival | Relevant `TradeEvent` execution while quote is active |
| Price eligibility | `_process_trade_execution`: demand price, haircut and tolerance |
| Win probability | `SimulatorModelBundle.event_win_probability` |
| Participation share | `SimulatorModelBundle.event_participation_share` |
| Legacy FillRatio option | `SimulatorConfig.quantity_model_definition` |
| Wealth, risk and quote costs | Environment reward-component accumulator |
| Terminal liquidation | `_apply_terminal_liquidation` |
| Dueling Double DQN | `DuelingQNetwork`, `CQLDuelingDoubleDQNTrainer` |
| CQL warm start / decay | offline trainer plus `simulator_online_train.py` schedules |
| Mixed replay | `FixedTransitionSource` + `RingReplayBuffer` |
| Simulator variants | `optimistic`, `win_only`, `partial` |
| Out-of-time evaluation | `evaluate_hybrid_policy.py` and training-time validation |
| Production model integration | `model_adapter_template.py` |

| Forecast-aware greedy baseline | `greedy_baseline.ForecastAwareGreedyPolicy` |
| Point-in-time action preview | `HybridMuniReplayEnv.preview_action` |
| RL-versus-greedy held-out comparison | `evaluate_hybrid_policy.py` and training-time `greedy_baseline_evaluation.json` |
