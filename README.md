# Municipal Bond Offer Pricing RL - Hybrid Replay Codebase v4

This codebase implements the **hybrid clock-and-trade-triggered simulator-online design** described in the Version 2.0 report, with spread-relative price actions for cross-CUSIP comparability.

It supports two learning stages and one explicit benchmark:

1. **Optional offline CQL warm start** from fixed historical internal quote transitions.
2. **Simulator-online Dueling Double DQN fine-tuning** in a historical MSRB replay environment, with optional mixing of historical transitions and a decaying CQL penalty.
3. **Forecast-aware one-step greedy baseline** that evaluates all 36 valid actions from point-in-time Pricing/Fill/ForwardPrice signals without looking at the next historical trade.

The code is a research starter, not a production execution system. The included heuristic Pricing/Win/Participation models exist only so the pipeline can run end to end. Replace them with calibrated internal models before interpreting results.

## Core event-ordering rule

The central leakage control is:

```text
quote active before trade
        -> processes that trade
        -> fill and inventory are updated
        -> external trade becomes observable at publish time
        -> policy may refresh the quote
        -> refreshed quote affects only later events
```

An own simulated fill is internally observable immediately. Another dealer's trade enters the policy state at its publication/observable time, not automatically at execution time.

## What changed through v4

The original v1 code trained only from fixed transition arrays. Version 2 adds:

- one-CUSIP position-lifecycle episodes;
- mandatory 30-minute quote refreshes;
- immediate re-quote after an own simulated fill;
- re-quote when a same-CUSIP trade becomes observable;
- separate trade execution time and publication time;
- irregular semi-Markov transitions with per-transition discounts;
- four-layer fill simulation: demand, eligibility, win, participation share;
- optimistic, win-only, and partial-fill simulator variants;
- dynamic inventory and live-quote quantity updates;
- quote-update and smoothness costs;
- simulator replay buffer and epsilon-greedy interaction;
- optional historical/simulator mixed replay;
- decaying CQL during simulator-online fine-tuning;
- held-out replay evaluation across simulator assumptions;
- tests for event ordering and fill-capacity logic;
- a deployable forecast-aware greedy baseline and RL-versus-greedy reporting;
- spread-relative price actions, where one action ratio is materialized using each CUSIP's point-in-time effective spread;
- spread floors, caps, fallbacks, source diagnostics, and an absolute business cap;
- offline logged-action encoding in half-spread units.

## File map

| File | Purpose |
|---|---|
| `hybrid_replay_simulator.py` | Hybrid historical replay environment, state construction, action masks, four-layer fill logic, reward and terminal handling. |
| `simulator_online_train.py` | Replay buffer, simulator-online Dueling Double DQN training, mixed historical replay, CQL decay, evaluation and checkpoints. |
| `prepare_hybrid_replay_data.py` | Converts position, point-in-time snapshot and trade tables into the canonical replay directory. Also creates demo data. |
| `model_adapter_template.py` | Interface for connecting your PricingModel, WinModel and ParticipationShareModel. |
| `evaluate_hybrid_policy.py` | Standalone RL/greedy held-out evaluation, comparison metrics and episode-level replay export. |
| `greedy_baseline.py` | Forecast-aware one-step expected-value policy, action diagnostics and held-out baseline evaluation. |
| `muni_cql_dueling_ddqn.py` | Offline CQL-Dueling-Double-DQN trainer and reusable neural Q-function trainer. |
| `prepare_muni_transitions_template.py` | Existing template for fixed offline transition construction. |
| `replay_schema.example.json` | Static and snapshot feature configuration for simulator states. |
| `column_map.example.json` | Example source-to-canonical column mapping. |
| `feature_config.example.json` | Existing fixed-transition feature configuration. |
| `tests/test_hybrid_simulator.py` | Timeline, publication delay, capacity and discount tests. |
| `tests/test_spread_relative_actions.py` | Cross-CUSIP spread scaling, absolute-cap masking and logged-action encoding tests. |

## Environment formulation

### Episode

```text
(CUSIP, position start time, starting inventory)
```

The episode ends when:

- inventory reaches zero;
- the maximum position horizon is reached;
- the event queue ends;
- the maximum decision count is reached.

### Decision triggers

```text
T_decision = T_clock U T_own_fill U T_trade_publish
```

The first version uses:

- every 30 minutes;
- immediately after a positive simulated fill;
- when a relevant same-CUSIP trade becomes observable.

### Action

The default action grid is dimensionless in price:

```text
price offset ratios: [-1.00, -0.50, -0.25, 0, 0.25, 0.50, 1.00]
quantity fractions:  [0.10, 0.25, 0.50, 0.75, 1.00]
plus one no-quote action
```

There are `7 x 5 + 1 = 36` actions. The actual dollar adjustment is state-dependent:

```text
effective_spread = clip(first valid point-in-time spread source, spread_floor, spread_cap)
spread_unit      = spread_unit_multiplier * effective_spread
                  # default multiplier=0.5, so the unit is one half-spread
raw_offset_$     = price_offset_ratio * spread_unit
offset_$         = clip(raw_offset_$, -max_absolute_price_offset, +max_absolute_price_offset)
quote quantity   = RoundLot(min(inventory, quantity_fraction * inventory))
quote price      = PricingModel(CUSIP, time, quote quantity) + offset_$
```

For example, ratio `+0.5` means half of one half-spread more passive than the PricingModel anchor. A CUSIP with a `0.40` full spread gets a `+0.10` dollar offset, while a CUSIP with a `2.00` spread gets `+0.50`.

The default spread source hierarchy is:

```text
cep_bid_ask_width
-> predicted_bid_ask_spread
-> segment_bid_ask_spread
-> fallback_spread
```

The simulator records the effective spread, spread unit, source index, fallback flag, ratio, and materialized dollar offset in state/action diagnostics. By default, an action is masked if its raw spread-relative offset would exceed the absolute business cap.

### Logged historical action encoding

For offline CQL warm-start data, historical dollar quotes are converted to the same ratio grid:

```text
logged_ratio = (historical_offer_price - pricing_anchor)
             / (spread_unit_multiplier * effective_spread)
```

The closest ratio and quantity-fraction buckets are used. `prepare_muni_transitions_template.py` now requires an `effective_bid_ask_spread` column and writes `price_offset_ratios` metadata. Old checkpoints trained with absolute-dollar action semantics should not be reused as if the action IDs had the new meaning.

### Four-layer fill simulator

For a relevant trade event:

```text
Demand:       D = 1
Eligibility:  E = 1[offer_price <= trade_price - haircut + tolerance]
Win:          Z ~ Bernoulli(p_win)
Capacity:     C = min(inventory, live_offer_quantity, trade_quantity)
Fill:         q_fill = D * E * Z * RoundLot(C * participation_share)
```

Use `--quantity-model-definition participation_share` for the report's preferred capacity-share label, or `displayed_quote_ratio` to retain a legacy label defined as filled quantity divided by displayed quote quantity.

The simulator execution price is the **agent offer price**, not the higher historical trade price.

### Reward

The environment accumulates:

```text
execution P&L
+ mark-to-market of remaining inventory
- time-scaled inventory risk
- target-inventory schedule shortfall
- price/quantity smoothness
- quote-update cost
- optional underpricing and missed-demand terms
- terminal liquidation cost and terminal inventory penalty
```

ForwardPriceModel predictions belong in the state. Realized future mark changes belong in the reward.

## Forecast-aware greedy baseline

The greedy policy is a **deployable myopic baseline**, not a trade oracle. At each decision time it scores every valid action using only information currently observable:

```text
expected execution value
+ forecast value of expected remaining inventory
- inventory carrying risk
- target-schedule shortfall
- quote smoothness and update cost
```

For action `a`:

```text
expected_fill(a) = offer_quantity(a)
                 * pretrade_win_probability(a)
                 * pretrade_participation_share(a)

greedy_score(a) = expected_fill(a) / 100
                * (offer_price(a) - fair_mark)
                + expected_remaining_inventory(a) / 100
                * forward_price_change_4h
                * forward_model_confidence
                - one_step_penalties(a)
```

The baseline does **not** inspect the next trade price or quantity. The environment's `preview_action()` method uses only current point-in-time features and pre-trade side-model outputs. In contrast, the neural Q-policy estimates long-horizon discounted value across future inventory states and quote opportunities.

Training automatically writes `greedy_baseline_evaluation.json` unless `--skip-greedy-baseline-evaluation` is supplied. Validation logs include RL-minus-greedy return, fill and ending-inventory differences.

### Variable time discount

Each transition stores elapsed time:

```text
discount_k = gamma_30m ** (elapsed_minutes / 30)
```

## Spread-relative configuration

Key training flags:

```bash
--spread-feature-names cep_bid_ask_width predicted_bid_ask_spread segment_bid_ask_spread
--spread-floor 0.02
--spread-cap 4.0
--fallback-spread 0.50
--spread-unit-multiplier 0.50
--max-absolute-price-offset 2.0
--price-smooth-lambda 0.01
--price-dollar-smooth-lambda 0.0
```

Set floors and caps from training-period distributions and business limits rather than treating the example defaults as universal.

## Canonical replay data

The prepared replay directory contains three tables and one schema.

### `positions`

Required columns:

```text
episode_id
cusip
start_time
end_time
starting_inventory
cost_basis
```

Static features listed in `schema.json` are also required.

### `snapshots`

Required columns:

```text
episode_id
observable_time
fair_mark
```

Every feature here must be point-in-time and observable at `observable_time`. A snapshot observable by the episode start is required. Any feature named in `spread_feature_names` must also be included in `snapshot_feature_cols`, because replay loading retains only configured snapshot features.

### `trades`

Required columns:

```text
episode_id
event_id
execution_time
publish_time
trade_price
trade_quantity
trade_type
```

For a dealer selling inventory, `S` trades are the default demand proxy. `D` trades should normally be calibrated separately. Quantity and price units must be consistent across inventory, quotes and trades.

## Install

```bash
pip install -r requirements.txt
```

If `pyarrow` is unavailable, the preparation script falls back to pickle files for demos. Parquet is recommended for real datasets.

## Run unit tests

```bash
python -m unittest discover -s tests -v
```

The tests verify:

- the old quote handles the trade that triggers a refresh;
- external trade information triggers a decision at publish time;
- partial fill equals feasible capacity times participation share;
- no-trade clock decisions remain in the episode;
- variable elapsed time produces the expected discount.

## End-to-end demo

```bash
./run_demo.sh
```

A smaller smoke run:

```bash
python prepare_hybrid_replay_data.py make-demo \
  --output-dir demo_train \
  --episodes 12

python prepare_hybrid_replay_data.py make-demo \
  --output-dir demo_valid \
  --episodes 5 \
  --seed 3026

python simulator_online_train.py \
  --train-replay-dir demo_train \
  --valid-replay-dir demo_valid \
  --output-dir demo_output \
  --train-episodes 10 \
  --replay-warmup 100 \
  --batch-size 64 \
  --evaluation-interval-episodes 5 \
  --device auto
```

## Prepare real replay data

First edit `replay_schema.example.json`, then normalize your source tables:

```bash
python prepare_hybrid_replay_data.py prepare \
  --positions positions.parquet \
  --snapshots point_in_time_snapshots.parquet \
  --trades msrb_trades.parquet \
  --schema-json replay_schema.example.json \
  --column-map-json column_map.example.json \
  --output-dir replay/train \
  --default-publish-lag-minutes 15 \
  --allowed-trade-types S
```

Prepare a strictly later validation period separately:

```bash
python prepare_hybrid_replay_data.py prepare \
  --positions valid_positions.parquet \
  --snapshots valid_snapshots.parquet \
  --trades valid_msrb_trades.parquet \
  --schema-json replay_schema.example.json \
  --column-map-json column_map.example.json \
  --output-dir replay/valid \
  --allowed-trade-types S
```

## Connect production side models

Copy `model_adapter_template.py` and replace the placeholders.

The adapter must provide:

- `pricing_anchor(context, quantity)`;
- pre-trade win/share scores for action-grid state features;
- event-conditioned win probability and participation share;
- customer-price haircut and eligibility tolerance;
- optional support score for action masking.

Run with:

```bash
python simulator_online_train.py \
  --train-replay-dir replay/train \
  --valid-replay-dir replay/valid \
  --model-factory my_model_adapter:build_models \
  --output-dir checkpoints/hybrid_v3 \
  --train-episodes 1000 \
  --device cuda
```

## Optional offline-to-simulator-online workflow

### 1. Train the existing offline warm start

```bash
python muni_cql_dueling_ddqn.py train \
  --train-npz transitions/train \
  --valid-npz transitions/valid \
  --output-dir checkpoints/offline_cql \
  --epochs 30 \
  --cql-alpha 1.0 \
  --device cuda
```

The offline transition state schema must exactly match the simulator state schema if the checkpoint or historical transitions are reused.

### 2. Fine-tune in the simulator

```bash
python simulator_online_train.py \
  --train-replay-dir replay/train \
  --valid-replay-dir replay/valid \
  --warmstart-checkpoint checkpoints/offline_cql/best_checkpoint.pt \
  --offline-transitions transitions/train \
  --model-factory my_model_adapter:build_models \
  --output-dir checkpoints/hybrid_v3 \
  --cql-alpha-start 0.5 \
  --cql-alpha-end 0.05 \
  --historical-fraction-start 0.5 \
  --historical-fraction-end 0.1 \
  --device cuda
```

This progressively increases near-policy simulator experience while reducing, rather than abruptly removing, offline pessimism.

## Output artifacts

`simulator_online_train.py` writes:

```text
best_simulator_online_checkpoint.pt
latest_simulator_online_checkpoint.pt
training_log.jsonl
training_summary.json
run_config.json
state_feature_names.json
greedy_baseline_evaluation.json
```

A saved checkpoint can be compared with the greedy baseline using the same held-out episodes, simulator assumptions and reward:

```bash
python evaluate_hybrid_policy.py \
  --checkpoint checkpoints/hybrid_v3/best_simulator_online_checkpoint.pt \
  --replay-dir replay/test \
  --model-factory my_model_adapter:build_models \
  --policies rl greedy \
  --output-json checkpoints/hybrid_v3/test_metrics.json \
  --replay-csv checkpoints/hybrid_v3/greedy_replay.csv \
  --replay-policy greedy \
  --device cuda
```

Evaluate the greedy baseline alone, without an RL checkpoint:

```bash
python evaluate_hybrid_policy.py \
  --replay-dir replay/test \
  --model-factory my_model_adapter:build_models \
  --policies greedy \
  --output-json checkpoints/greedy_only_metrics.json
```

Useful baseline controls include `--greedy-forecast-weight`, `--greedy-forward-change-feature`, `--greedy-forward-confidence-feature`, and switches that remove inventory, schedule, smoothness or update-cost terms for ablation studies.

The best checkpoint is selected using mean return in the held-out **partial-fill** simulator. This is still not evidence of real-market value. Policy promotion requires chronological backtesting, cold-CUSIP evaluation, simulator sensitivity analysis and shadow mode.

## Important modeling cautions

1. **MSRB trade quantity is demand-event capacity, not your guaranteed fill.**
2. **A trade must not enter policy state before it is observable.**
3. **The new quote cannot be evaluated against the trade that caused the refresh.**
4. **Do not fit side models and evaluate the RL policy on the same time period.**
5. **Run optimistic, win-only and partial-fill variants.** A policy that works only under full-capacity fills is simulator-dependent.
6. **Check cold-CUSIP generalization.** A neural Q-function enables generalization but does not prove it.
7. **Add uncertainty controls.** Use support masks, model ensembles, calibration residuals or conservative reward adjustments when the policy enters weakly supported regions.
8. **Control quote flickering.** Use update cost, smoothness penalties, minimum quote life and operational rate limits.
9. **Keep a fallback policy and kill switch** before any restricted live test.

## Primary method references

- Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning*, NeurIPS 2020.
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016.
- Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning*, ICML 2016.
- Sutton, Precup and Singh, *Between MDPs and Semi-MDPs*, Artificial Intelligence 1999.
- Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.
