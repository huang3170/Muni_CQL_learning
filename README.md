# Muni Offer Pricing：CQL-Regularized Dueling Double DQN 第一版

这是一套可运行的 **offline reinforcement learning** starter code，目标是学习：

\[
a_t=(\Delta p_t,\rho_t)
\]

其中：

- `Δp`：相对 PricingModel 输出的 offer-price offset；
- `ρ`：当前 inventory 的 offer quantity fraction；
- 额外包含一个 `no quote` action。

默认 action grid：

- Price offsets：`[-0.50, -0.25, -0.125, 0, 0.125, 0.25, 0.50]`
- Quantity fractions：`[0.10, 0.25, 0.50, 0.75, 1.00]`
- Action 数量：`7 × 5 + 1 = 36`

## 文件

- `muni_cql_dueling_ddqn.py`：主训练、验证、checkpoint 和单状态 inference。
- `prepare_muni_transitions_template.py`：把 decision-level dataframe 转成 offline transition arrays 的模板。
- `feature_config.example.json`：state feature 顺序示例。
- `requirements.txt`：依赖。

## 模型结构

### Dueling Q-network

网络分为：

- `V(s)`：当前 inventory/market state 的整体价值；
- `A(s,a)`：某个报价 action 相对其他 action 的优势。

合并方式：

\[
Q(s,a)=V(s)+A(s,a)-\operatorname{mean}_{a'\in A_{valid}}A(s,a')
\]

### Double DQN target

Online network 选择 next action，target network 估值：

\[
a^*=\arg\max_{a'}Q_\theta(s',a')
\]

\[
y=r+\gamma Q_{\bar\theta}(s',a^*)
\]

### Discrete CQL regularizer

\[
L_{CQL}=\alpha\left[\tau\log\sum_{a\in A_{valid}}
\exp(Q(s,a)/\tau)-Q(s,a_{logged})\right]
\]

最终损失：

\[
L=L_{Huber\ TD}+L_{CQL}
\]

## Transition 数据格式

训练脚本支持两种输入：

1. 单个 `.npz` 文件；
2. 一个目录，每个 array 单独保存为 `.npy`。**大数据推荐第二种**，因为 `.npy` 可 memory-map。

必须包含：

| Array | Shape | 类型 | 说明 |
|---|---:|---|---|
| `states` | `[N, state_dim]` | float32 | 当前 state |
| `actions` | `[N]` | int64 | 历史 logged action ID |
| `rewards` | `[N]` | float32 | 该 transition reward |
| `next_states` | `[N, state_dim]` | float32 | 下一 state |
| `dones` | `[N]` | float/bool | episode 是否结束 |
| `action_masks` | `[N, 36]` | bool | 当前 state 有效 action |
| `next_action_masks` | `[N, 36]` | bool | 下一 state 有效 action |
| `discounts` | `[N]` | float32 | 可选，event-driven 时间折扣 |

对于不规则时间间隔，建议：

\[
\gamma_t=\gamma_{base}^{\Delta t/base\_minutes}
\]

## State 建议顺序

你的 state vector 可以包括：

1. bond static：coupon、maturity、duration、rating、sector、state、tax status、call features；
2. market：CEP、curve、recent MSRB trade、volume、imbalance、volatility、liquidity regime；
3. inventory：current inventory、fraction remaining、position age、cost、unrealized P&L、target inventory；
4. time：time of day、time to close、time to deadline、delta time；
5. PricingModel grid outputs；
6. FillModel calibrated probability/fill-ratio grid；
7. ForwardPriceModel output and confidence；
8. previous quote/action/fill history。

所有 state feature 必须是 **point-in-time**，且训练/验证 feature 顺序必须一致。

## Reward 模板

`prepare_muni_transitions_template.py` 默认使用：

\[
R^{wealth}_t=
q^{fill}_t(p^{exec}_t-m_t)
+I_{t+1}(m_{t+1}-m_t)
\]

然后扣除：

- inventory-risk penalty；
- target-inventory schedule penalty；
- quote price/quantity instability；
- observed demand 下的 missed-quantity penalty；
- 可选 underpricing penalty。

第一版建议将 `underpricing_lambda=0`，先避免与 execution P&L 重复惩罚。

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. Smoke test

生成 synthetic data：

```bash
python muni_cql_dueling_ddqn.py make-demo \
  --output-dir demo_data \
  --train-rows 50000 \
  --valid-rows 10000 \
  --state-dim 64
```

训练：

```bash
python muni_cql_dueling_ddqn.py train \
  --train-npz demo_data/demo_train.npz \
  --valid-npz demo_data/demo_valid.npz \
  --output-dir demo_output \
  --epochs 10 \
  --batch-size 1024 \
  --cql-alpha 1.0 \
  --device auto
```

## 3. 准备真实 muni transition data

先编辑 `feature_config.example.json`，确保 state columns 都是 numeric、point-in-time、已经完成 missing-value handling。

```bash
python prepare_muni_transitions_template.py \
  --input-data train_decisions.parquet \
  --feature-config feature_config.example.json \
  --output-dir transitions/train
```

对 validation period 单独运行：

```bash
python prepare_muni_transitions_template.py \
  --input-data valid_decisions.parquet \
  --feature-config feature_config.example.json \
  --output-dir transitions/valid
```

然后训练：

```bash
python muni_cql_dueling_ddqn.py train \
  --train-npz transitions/train \
  --valid-npz transitions/valid \
  --output-dir checkpoints/muni_cql_v1 \
  --epochs 30 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --cql-alpha 1.0 \
  --target-update-interval 1000 \
  --reward-scale 100.0 \
  --reward-clip 10 \
  --device cuda
```

`reward-scale` 应使大部分 scaled reward 落在大约 `[-5, 5]`，而不是机械使用示例中的 100。

## 4. Inference

输入：

- `state.npy`：一个原始、未标准化 state；
- `mask.npy`：一个 bool action mask。

```bash
python muni_cql_dueling_ddqn.py infer \
  --checkpoint checkpoints/muni_cql_v1/best_checkpoint.pt \
  --state-npy state.npy \
  --mask-npy mask.npy \
  --device cuda
```

输出包括 `action_id`、对应 price offset/quantity fraction 和全部 masked Q-values。

## 第一轮建议调参

建议小范围搜索：

- `cql_alpha`: `[0.1, 0.5, 1.0, 2.0, 5.0]`
- `learning_rate`: `[1e-4, 3e-4]`
- `batch_size`: `[1024, 2048, 4096]`
- `target_update_interval`: `[500, 1000, 2500]`
- reward penalty weights：先以 P&L 为主，逐个增加 penalty，不要一次全部调大。

监控：

- validation TD loss；
- CQL gap；
- policy/behavior action agreement；
- learned action distribution；
- Q-value scale；
- chronological historical replay P&L；
- fill、inventory age、terminal inventory、quote aggressiveness；
- action support：policy 是否集中选择历史上很少出现的动作。

**不要用 validation loss 代替 policy-value backtest。** 最终模型选择需要 chronological replay、shadow simulation，最好再加 FQE 或其他离线 policy evaluation。

## 需要按你的数据修改的地方

1. `ColumnConfig` 的列名；
2. `feature_config.example.json` 中的 state columns；
3. action price-offset grid 和 quantity grid；
4. quantity lot rounding；
5. reward penalty coefficients；
6. action mask 中的业务价格上下限、inventory/lot/risk constraints；
7. terminal liquidation reward。当前 transition template 把 episode 最后一行标记为 terminal，但没有替你构造强制 liquidation cash flow。

## 参考论文

- Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning*, NeurIPS 2020.
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016.
- Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning*, ICML 2016.
