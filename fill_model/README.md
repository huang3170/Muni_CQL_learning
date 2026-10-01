# Municipal bond first-fill model

直接使用你提供的 pandas DataFrame 列名。一行对应某 CUSIP 一段有效报价观察区间，最多包含一次首次成交。模型输出每分钟的首次成交到达率，以及指定窗口内的成交概率。

## 输入列与特征

`run_experiment(df, ...)` 不需要 `prepare_data` 或列名适配函数。代码不生成额外的 interval、configuration 或 fill ID，也不读取旧版库存、duration、sector、venue 等字段。

| 输入列 | 用途 |
|---|---|
| `quantity` | `log_quantity = log1p(quantity)`，保留输入数量单位 |
| `l1_vs_mid` | 单独的价格项，其系数限制为非正 |
| `l2_vs_mid`、`l3_vs_mid` | 减去 `l1_vs_mid` 得到 `gap_l2`、`gap_l3`；inactive level 不提供有效 gap |
| `l1_price`、`l2_price`、`l3_price` | 对应 `l*_vs_mid` 缺失时，用报价减 `mid_price` 补充价格偏移 |
| `l1_active`、`l2_active`、`l3_active` | `active_template` 类别特征；缺失 mask 用 `?` 表示 |
| `mid_price` | 数值状态特征 |
| `cep_bid_ask_width` | 数值状态特征；缺失时由 `ask_price - bid_price` 补充，bid/ask 不重复进入状态矩阵 |
| `cycle_time` | 区间起点、按日切分及日内 sin/cos 特征 |
| `cep_age_min` | `log_cep_age = log1p(cep_age_min)`，直接使用给定值 |
| `cep_time` | 检查 CEP 是否晚于报价起点；不重算 age，不进入状态矩阵 |
| `time_to_maturity` | 数值状态特征，保留输入单位 |
| `rating` | 类别状态特征 |
| `liquidity`、`coupon` | 数值状态特征，保留输入单位 |
| `first_fill_level` | 非空且不是空白字符串时 `event=1`，否则为 0 |
| `first_fill_quantity` | 成交结果，不进入预测特征 |
| `quote_end_time`、`exposure_minutes` | 区间终点和有效暴露分钟，用于检查、似然及评估 |
| `cusip`、`episode_id` | 分组与审计，不作为模型特征 |

`state_features(df)` 只产生明确列出的状态特征。ID、成交结果、结束时间、exposure 都不进入 X；价格项单独加入价格模型，避免当前价格偏移重复进入无约束状态项。价格模型假定报价提高时成交到达率不增加，价格及价格偏移的单位必须在训练和预测时一致。

普通数值特征使用**训练集的中位数**处理缺失，并为训练期出现缺失的列添加缺失指示变量，包括约 24% 缺失的 `liquidity`。不做全表 `dropna`，不把缺失 liquidity 填成 0，也不使用验证/测试数据计算填充值。训练期全空的普通数值特征不参与拟合；价格模型的价格项若训练期全空则报错。类别缺失独立编码，未知类别在预测时可处理。

`cycle_time`、`quote_end_time`、`cep_time` 应已是 pandas datetime dtype。代码直接使用输入时间，不转换 UTC、不附加或切换时区，也不转换 DataFrame 中的字符串时间。数量、maturity 和 coupon 同样不自动换单位。CSV 本身只能存文本，因此 CLI 在读取 CSV 时用 `parse_dates` 恢复这三个 datetime 列；这只是文件反序列化，没有时区转换。

## 使用

在包含 `fill_model` 文件夹的项目目录执行：

```bash
python -m pip install -r requirements.txt
```

```python
from fill_model import TrainingConfig, run_experiment, state_features

# df 使用上表中的原始列名，时间列已为 datetime dtype。
config = TrainingConfig(
    train_fraction=0.60,
    validation_fraction=0.20,
    alphas=(0.0001, 0.001, 0.01, 0.1),
    compute_ipcw=False,
    bootstrap_repetitions=300,
)
result = run_experiment(df, output_dir="runs/fill_model", config=config)
print(state_features(df).columns)
print(result.split_audit)
print(result.metrics[[
    "model", "split", "events",
    "event_time_nll_per_30_exposure_minutes",
    "integrated_hazard_to_events",
]])
print("验证期选中的模型:", result.selected_model_name)
print(result.feature_missingness.query("feature == 'liquidity'"))

# 预测只需当前状态列，不需要成交结果、结束时间、exposure 或 ID。
# p30 = result.selected_model.predict_proba(new_states, horizon_minutes=30)
```

显式指定切分日期时，两个边界必须一起提供：

```python
config = TrainingConfig(validation_start="2026-09-01", test_start="2026-09-16")
result = run_experiment(df, "runs/fill_model", config)
```

读取 CSV 时保留 ID 的前导零，并将时间文本反序列化：

```python
import pandas as pd

df = pd.read_csv(
    "fill_model_intervals.csv",
    dtype={"cusip": "string", "episode_id": "string"},
    parse_dates=["cycle_time", "quote_end_time", "cep_time"],
)
```

```bash
python -m fill_model --data fill_model_intervals.csv --output runs/fill_model
python -m fill_model --demo --output runs/synthetic_demo
python -m unittest discover -s tests -p 'test_fill_model.py' -v
```

Parquet 读取需要 `pyarrow`。Demo 使用相同列名，包含约 24% liquidity 缺失和约 2% 普通状态特征缺失；数据全部为虚构数据。

## 模型与似然

| 模型 | 定义 |
|---|---|
| `template_baseline` | active template 的事件数 / 暴露时间，向总体到达率收缩 |
| `no_price` | 仅使用状态特征的正则化分段指数模型 |
| `price` | 状态模型加有界的 `l1_vs_mid` 价格项 |

```text
lambda = exp(intercept + X_state @ beta + beta_price * l1_vs_mid / price_scale)
beta_price <= 0

loss = sum(lambda_i * exposure_i - event_i * log(lambda_i))
       / (sum(exposure_i) / 30) + alpha / 2 * sum(beta_j²)
```

默认 `price_scale=0.10`；截距不惩罚。损失按总 exposure 而非行数归一化，在状态与到达率保持一致时，拆分区间不会改变事件似然与正则项的相对权重。

预测概率为 `-expm1(-lambda * horizon_minutes)`，假设窗口内报价及状态保持不变。训练使用完整事件和 exposure，不做正负样本平衡或无权重负样本下采样。优化器未收敛时明确报错。此模型只拟合 any-first-fill，不拟合成交 level、成交量或价格的因果效果。

## 数据检查与时间切分

输入检查直接使用原列名。`exposure_minutes` 必须大于 0 且不能超过 `quote_end_time - cycle_time`；CEP 不能来自未来。时间和 exposure 等定义观察区间的字段不能通过特征中位数修补。普通状态字段的缺失由模型预处理处理。

有成交的区间应在首次成交时结束；空 `first_fill_level` 表示区间内确实无首次成交。缺失匹配结果或未知 outcome 应先在上游处理。`first_fill_quantity` 不决定 event。

同一 CUSIP 的区间不能重叠，以免重复计算整套报价的 exposure。代码默认每个 CUSIP 同时只有一套合并报价；如有独立 desk/book，应在上游定义合适的实体。`strict=True` 时数据检查失败会报错；显式设置 `strict=False` 可排除问题行，并应检查 `exclusions.csv`。

默认按输入中实际出现的日期约 60% / 20% / 20% 划分，不随机拆行。只删除自身跨边界的区间：`cycle_time < cut < quote_end_time`。终点恰好等于边界的行留在前一集合，起点等于边界的行进入后一集合。`episode_id` 不做整组 purge，同一 episode 可以在多个时期出现。这评估未来日期的报价，包括已有持仓的后续报价。

Exposure 只包含实际有效挂单分钟。收盘结束且未成交的区间，其 `first_fill_level` 为空；次日恢复报价应建立新行，不能把休市时间算作有效暴露。代码不会自动截断跨日行或推测其中成交归属。

预处理只在训练期拟合，验证期选择 alpha 和三种模型中的 winner；测试期只用于最终评估。保存的模型仍是原训练期拟合的模型，不以 train+validation 重训。

## 结果与诊断

| 文件 | 内容 |
|---|---|
| `metrics.csv` | 各模型在 train/validation/test 的事件似然与校准指标 |
| `tuning_results.csv` | 各 alpha 的验证分数及优化器状态 |
| `split_audit.csv` | 各期及跨界排除的行数、事件、episode 和 exposure |
| `exclusions.csv` | 数据检查排除的行及原因 |
| `predictions.csv` | 验证/测试逐行到达率、p30、实际 exposure 及事件损失 |
| `calibration.csv` | 按到达率、日期、价格、数量、template 等分组的诊断 |
| `bootstrap_intervals.csv` | 按原 episode 分组重采样得到的测试指标区间 |
| `coefficients.csv` | 特征名称与系数 |
| `feature_missingness.csv` | 各集合中填补前的特征缺失行数及比例 |
| `metadata.json` | 切分边界、模型选择、配置及假设 |
| `models.joblib` | 三个模型及选择结果 |

`event_time_nll_per_30_exposure_minutes` 越小越好，是按 exposure 归一化的事件似然，不是 binary log loss。只应在相同数据和时间单位下比较。

`integrated_hazard_to_events = sum(lambda_i * observed_exposure_i) / sum(event_i)`，理想值约为 1。这是实际观察区间上的累计强度诊断，不等于 `sum(p30)/events`。按组查看可定位偏差，但事件稀少的组会不稳定。

Bootstrap 的 NLL difference 负数表示前一个模型更好，例如 `price-no_price`。重采样保留同一 CUSIP/episode 内的行关联，区间以已拟合模型为条件，不覆盖重训、选参或跨 episode 的市场冲击。

`compute_ipcw=True` 时附加输出 IPCW Brier；默认关闭。训练期拟合删失 Kaplan-Meier，在窗口内成交的行按 `1/G(event_time-)` 加权，完整观察窗口且无事件的行按 `1/G(horizon-)` 加权，提前删失的行贡献权重 0。缺少删失支持时返回不可用状态。这要求训练与评估期删失分布可迁移，并满足独立删失假设；主动改价或撤单可能违背该假设。

## 保存后预测与价格曲线

```python
import joblib
from fill_model import score_price_grid

bundle = joblib.load("runs/fill_model/models.joblib")
model = bundle["models"][bundle["selected_model_name"]]
# p30 = model.predict_proba(new_states, horizon_minutes=30)
# curve = score_price_grid(bundle["models"]["price"], new_states.head(10),
#                          deltas=[0.05, 0.10, 0.15, 0.20, 0.30])
```

只加载可信来源的模型文件。价格曲线保持其他状态和 level gaps 不变，同步改变各 level 的价格及相对 mid 偏移，用于检查模型的数学价格响应。它不模拟改价后的客户行为或状态变化，不能单凭此曲线判断报价效果。
