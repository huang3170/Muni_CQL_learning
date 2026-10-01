# Municipal bond first-fill model v1

输入是已经构建好的 interval pandas DataFrame。一行对应某 CUSIP 整套有效报价的一段可观测挂单时间，最多一个有效首次成交事件。输出是每分钟成交到达率和给定窗口（默认 30 分钟）的首次成交概率。

本版本完成 any-first-fill 到达率训练与事实评估。Level routing、fill size、价格优化、RL 和因果价格效果不在本版本中。报价数据及源文件不会被修改。

## 当前 dataframe 列名可直接使用

现在 `run_experiment(df, ...)` 直接接受你提供的列名。所有时间都是不带时区的纽约当地钟面时间，代码不附加时区、不转换 UTC，也没有时区配置参数；09:30 保持 09:30。`cycle_time`、`quote_end_time`、`cep_time` 可直接使用普通 datetime 或无时区字符串。有成交的 row 在首次成交时结束；`quantity` 为挂单数量；`time_to_maturity` 为天数；`liquidity` 为数值。

内部及输出的时间列为 `start_time`、`end_time`、`cep_asof_time`、`fill_time`。输入的 `cycle_time`、`quote_end_time`、`cep_time` 无需改名。旧版已转换为 UTC 的数据不应直接改列名后重用；请从原始纽约当地时间 dataframe 重新运行。新版不自动处理带时区的值，以免悄悄改变时间含义。

| 原始列 | 用途 / 模型中的表示 |
|---|---|
| `quantity` | `log_quote_quantity = log1p(quantity)`；不当作持仓库存 |
| `l1_price`、`mid_price` | `delta_l1 = l1_price - mid_price`，作为有约束的价格动作 |
| `l2_price`、`l3_price` | 相对 L1 的 level gaps；仅 active level 的价格有效 |
| `l1_active`、`l2_active`、`l3_active` | active template 类别特征 |
| `mid_price` | 当前市场价格状态 `cep_mid` |
| `cep_bid_ask_width`、`bid_price`、`ask_price` | `market_spread`；宽度缺失时从 ask-bid 派生 |
| `cycle_time` | interval 起点与当地日内 sin/cos 特征 |
| `cep_time`、`cep_age_min` | 从 cycle_time-cep_time 重算 `log_cep_age`；缓存的 cep_age_min 不重复入模 |
| `time_to_maturity` | 除以 365.25 得到 `time_to_maturity_years`；不同于久期 duration |
| `rating` | 类别特征，缺失单独编码 |
| `liquidity` | 数值特征 `liquidity_score`，保留缺失标记 |
| `coupon` | 数值特征，保持输入单位；训练与预测必须一致 |
| `first_fill_level` | 非空/非空白字符串即 `event=1`；不进入 X |
| `first_fill_quantity` | 成交结果，保留供检查；不进入 X |
| `quote_end_time`、`exposure_minutes` | 观察终点与有效挂单分钟，用于标签/似然/评估；不进入 X |
| `cusip`、`episode_id` | 时间重叠检查与 episode 分组；不做 ID 类别特征 |

`l1_vs_mid`、`l2_vs_mid`、`l3_vs_mid` 不直接进入状态矩阵：从价格重新计算 L1 shift 和 gaps，避免把同一价格动作放入多个无约束特征，绕开 `beta_price <= 0`。所有价格和宽度应使用每 $100 par 的价格点，不是收益率 bp。

当前没有库存总量或 configuration age，因此不从 quantity 或 exposure 猜测这两项；旧格式中若有真实的 `inventory_par_start` / `config_age_minutes`，仍可使用。未提供 `interval_id` 时按 CUSIP+起点生成；`quote_config_id` 缺失时只是 interval 的占位标识；缺真实 `fill_event_id` 时按 CUSIP+首次成交时间生成检查键。这不能替代与原始成交账本的唯一匹配。

未提供 `end_reason` 时，成交行标 `FILL`，已知无成交的行标 `CENSORED`。`CENSORED` 只表示观察结束前无目标成交，不推断它是改价、RFQ、收盘或数据截止。若你已有真实 end_reason，继续传入即可。按你的数据口径，空 first_fill_level 表示已确认无目标成交；未观测/未匹配的结果必须先在上游区分，不能混作这种空值。

普通数值状态缺失（包括 liquidity 的约 24%，以及 quantity/coupon/maturity 等少量缺失）使用**训练集的中位数 + 训练时出现缺失的列的 missing indicator**；不做全表 dropna、不把 liquidity 缺失填成 0、不从未来回填。rating 缺失编码为 `__MISSING__`。训练期全空的可选特征不拟合；验证/预测不重新计算填补值。

时间、exposure、标签、active masks、active level 的报价和 mid 等定义样本的关键字段不能用中位数修补，仍做严格检查。当前列名下 inactive level 中残留的价格会屏蔽为空；第一版仍要求 L1 active。CEP 时间缺失无法完成来源时间检查，也会进入数据质量检查，而不是用别的行填补。

## 安装与直接使用 DataFrame

将整个 `fill_model_v1` 文件夹放在 notebook 或项目目录下，保留文件夹名称。在项目目录执行：

```bash
python -m pip install -r fill_model_v1/requirements.txt
```

```python
from fill_model_v1 import TrainingConfig, run_experiment

# df = 你的 interval dataframe，不需要先做随机拆分、one-hot 或标准化。
config = TrainingConfig(
    maturity_unit="days",
    quantity_multiplier=1.0,      # 保留挂单数量的原始单位；并未假设是面值美元
    liquidity_kind="numeric",
    quote_end_is_first_fill=True,
    train_fraction=0.60,
    validation_fraction=0.20,
    alphas=(0.0001, 0.001, 0.01, 0.1),
    compute_ipcw=False,
    bootstrap_repetitions=300,
)

result = run_experiment(df, output_dir="runs/fill_v1", config=config)

print(result.split_audit)
print(result.metrics[[
    "model", "split", "events",
    "event_time_nll_per_30_exposure_minutes",
    "integrated_hazard_to_events",
]])
print("验证期选中的模型:", result.selected_model_name)
print(result.feature_missingness.query("feature == 'liquidity_score'"))

# 预测也支持当前列名，无需成交字段、quote_end_time、exposure 或 episode_id。
# 提供 cycle_time、报价状态、CEP 来源时间和当时可得的债券属性。
# new_states = ...
# p30 = result.selected_model.predict_proba(new_states, horizon_minutes=30)
```

也可以显式指定当地日期边界，两个日期必须一起提供。以下日期只是调用示例，应根据自己的真实日期范围设置：

```python
config = TrainingConfig(
    validation_start="2026-09-01",  # 纽约当地 00:00 起属于 validation
    test_start="2026-09-16",        # 纽约当地 00:00 起属于 test
)
result = run_experiment(df, "runs/fill_v1", config)
```

CSV 中 CUSIP 和 ID 列用 string 读取，避免丢失前导零：

```python
import pandas as pd
df = pd.read_csv("fill_model_intervals.csv", dtype={
    "cusip": "string", "interval_id": "string",
    "position_episode_id": "string", "episode_id": "string", "quote_config_id": "string",
    "fill_event_id": "string",
})
```

命令行与模拟数据示例：

```bash
python -m fill_model_v1 --data fill_model_intervals.csv --output runs/fill_v1
python -m fill_model_v1 --demo --output runs/synthetic_demo
python -m unittest discover -s fill_model_v1/tests -v
```

读取 parquet 需要另外安装 `pyarrow`。Demo 全部使用虚构数据，不代表真实交易效果。

## 三个模型

| 名称 | 定义 |
|---|---|
| `template_baseline` | 每种 active-level template 的事件数 / 暴露时间，向总体到达率收缩；未见 template 回退总体 |
| `no_price` | 状态特征的正则化分段指数模型，不含当前价格动作 |
| `price` | 同样的状态模型，加 `delta_l1`，系数限制为非正 |

模型定义：`lambda = exp(intercept + X_state @ beta + beta_price * delta_l1 / 0.10)`，价格系数 `beta_price <= 0`。时间单位是有效挂单分钟。

目标函数：

```text
sum(lambda_i * exposure_i - event_i * log(lambda_i))
-------------------------------------------------  + alpha / 2 * sum(beta_j²)
           sum(exposure_i) / 30
```

截距不惩罚。以总 exposure 而非行数归一化，避免单纯拆行改变似然相对正则项的权重。在相同特征/到达率下，72 分钟一次成交与 30+30+12 分钟的 0/0/1 具有相同事件似然。预处理、不同的时间特征和数据切分仍会影响最终拟合；拆行不会创造独立成交信息。

预测概率为 `-expm1(-lambda * horizon_minutes)`。它假设预测窗口内报价和状态保持不变。模型采用完整事件与 exposure，不做正负样本平衡、SMOTE 或无权重负样本下采样。

SciPy L-BFGS-B 同时优化线性参数和有界价格系数。若没有收敛，会明确报错；不会把未收敛结果当作可用模型。

## 输入与数据检查

`schema.py` 先适配当前列名，`core.prepare_data` 再检查内部字段。`required` 列表是适配后的必需字段，不需要你手工补齐全部内部列。旧格式的非时间字段仍可使用；内部时间列已移除 `_utc` 后缀，并要求纽约当地钟面值。无需 size 标签也能训练 arrival，但执行时间及归属必须可信。

- 保留用户原来的 parent episode。`inventory_segment_id` 可以留在输入中，但不能替代 parent episode 的分组。
- 价格为美元 / $100 par；库存为美元 par；`delta_l1 = l1_price - cep_mid`。代码会派生 delta 和 gaps，并核对已有值。
- 第一版要求 L1 有效作为共同价格 anchor。L1 不存在的配置会被明确标记，而不是静默用别的价格替代。
- inactive level 的价格必须为空，不能填 0。`l*_active` 为 0/1。
- `exposure_minutes > 0`，且不能大于两个时间戳之间的墙上时钟时长。
- `event=1` 当且仅当 `end_reason=FILL`。部分成交也是 FILL；成交时间等于区间终点。
- 同一个 fill event 不能重复归属。多个 venue 不能产生重叠的整仓训练行。
- DATA_GAP 前缀只有在报价有效、成交信息完整时才允许标 0；本代码无法代替源数据覆盖检查。
- `train_eligible=0` 的记录会写入 exclusions。标为可训练但检查失败的记录默认报错。
- 如需审计后主动剔除问题记录，可以设置 `strict=False`，并检查 `exclusions.csv`。不可把此选项用于掩盖系统性匹配问题。
- `UNKNOWN` 或不可信 outcome 不能当作负样本。缺 level 或 size 不会自动删除可信的 arrival 正样本。
- 所有时间字段使用无时区的纽约当地时间；不使用 `Z` 或时区偏移后缀。
- `cep_asof_time` 必须不晚于起点。其他市场特征也必须在起点已经可用；代码无法仅凭聚合数值发现报告延迟或未来信息。

此版本假设每个 CUSIP 同时只有一个合并后的可售持仓。如果同一 CUSIP 有多个独立 desk/book，需先把 book 标识纳入实体键再使用，不能忽略重叠检查。

## 预测特征与泄漏防护

代码使用明确 allowlist，绝不自动把 dataframe 的所有列送入模型。

当前状态：log 挂单数量、level gaps、active template、日内 sin/cos、CEP age、mid、market spread、到期年数、rating、liquidity、coupon。旧格式的库存、configuration age、duration、sector、venue、历史成交等字段若提供仍可使用。缺少可选字段可运行，训练期完全缺失的数值特征不拟合。

中位数填补、缺失指示、标准化、低频分类合并全部只在训练期拟合。Exposure 用于损失和训练标准化权重，不作为预测特征。`end_reason`、event、fill level/size、结束时间、episode 最终时长和成交后库存均不进入特征矩阵。

`no_price` 仍包含 level gaps，因为它们描述报价结构；排除的是当前 common shift，不代表排除所有市场定价信息。

## 划分、调参和模型选择

默认按实际出现的纽约交易日期约 60% / 20% / 20% 划分，边界为输入日期的 00:00。没有随机行级拆分，也没有时区换算。显式边界推荐只传 `YYYY-MM-DD`。

现在只删除单条 interval 自身跨边界的记录：`start_time < cut < end_time`。结束时间恰好等于边界的记录留在前一集合；起点等于边界的记录进入后一集合。同一个 parent episode 或 configuration ID 可按各行时间出现在多个集合，不再整组 purge，也不需要改变真实 episode 的定义。

对于仅在当地 07:00–17:00 active 的报价，只要每条 interval 都在当日闭市前结束，边界 purge 应为 0。建表时当天最后一条记录最晚在 17:00 结束；若因收盘结束且没有 fill，则 `end_reason=MARKET_CLOSE, event=0`，仅累计实际 active exposure。次日恢复报价时重新开始 interval，并取当时可用的状态；即使价格/数量未变，也不能把休市时间当作连续 active 挂单。代码不会自动截断跨日行或猜测其中的成交归属；实际跨界行会被删除并提醒检查。

此协议评估未来日期的 quote，包括既有库存的后续报价，不是未见过的 episode 或 CUSIP 泛化测试。特征必须是起点当时可得，训练标签在切分边界已经可知；episode 最终信息不能作为特征。完整 episode 隔离可以作为另外的评估协议，但不是此版本的默认切分。

`split_audit.csv` 记录各集合的行数、事件数与 exposure。`metadata.json` 的 `split` 记录不带时区的日期边界、被删除的行/事件/exposure 比例、跨集合的 episode 数。一个 episode 可在多个集合计数，因此各集合的 episode 数不能相加作为总 episode 数。

只在训练期拟合，验证期 NLL 选择各模型 alpha，再在三种模型中选验证期最优者。基准或无价格模型胜出时照实返回，不强行选价格模型。测试期不用于调参、校准或决定 winner。保存的是原训练期拟合的模型，本次不会再用 train+validation 重训。

这是首轮单一 chronological holdout，尚未做 expanding-window 多轮验证。反复看 test 再改特征会消耗这个测试集；下一阶段应使用新的后续时期。

## 评估文件如何看

| 文件 | 含义 |
|---|---|
| `metrics.csv` | 三个模型在 train/validation/test 的 NLL、暴露量和事件校准诊断 |
| `tuning_results.csv` | 各 alpha 的验证分数和优化器状态 |
| `split_audit.csv` | 各期及跨界排除的行数、事件数、episode 数与暴露量 |
| `exclusions.csv` | 不可用数据及原因；与跨界排除分开 |
| `predictions.csv` | 验证/测试逐行到达率、p30、实际 exposure 和事件损失 |
| `calibration.csv` | 按预测率分箱、日期、价差、库存、配置年龄、template 的校准诊断 |
| `bootstrap_intervals.csv` | 按 parent episode 成对重采样的测试指标区间 |
| `coefficients.csv` | 特征名称与系数 |
| `feature_missingness.csv` | 各集合中各状态特征填补前的缺失行数/比例，包含 liquidity |
| `metadata.json` | 日期边界、选择规则、超参数、版本、假设和价格系数 |
| `models.joblib` | 三个模型和选择结果；仅加载自己信任的模型文件 |

优先比较 `event_time_nll_per_30_exposure_minutes`，越小越好。它是事件似然按暴露量归一化，不是 binary log loss；单位变更或不同数据集合的绝对数值不能直接比较。

`integrated_hazard_to_events = sum(lambda_i * observed_exposure_i) / sum(event_i)`，理想情况下接近 1。它使用成交停止后的实际 exposure，是累计强度/事件校准诊断，**不能称为未来窗口的 expected fill count，也不是 sum(p30)/events**。各分组检查可定位系统性偏差，稀少事件分组的比值会很不稳定。

bootstrap 的 NLL difference 负数表示前一个模型更好。例如 `price-no_price`。重采样时把测试期内同一 parent episode 的所有行放在一起，保持内部关联；不会涵盖跨 episode 的共同市场日冲击，也不包含重训/选参不确定性及共享 episode 引起的训练/测试依赖。结果是在已拟合模型条件下的初步区间，不能证明价格因果效果。

### 可选的 30 分钟 IPCW Brier

默认关闭，设置 `compute_ipcw=True` 才输出。不能把观察了 8 分钟后改价的未成交区间标成完整 30 分钟负样本，也不能只删掉这种负样本后计算普通 Brier。

可选实现用训练期拟合的 censoring Kaplan-Meier：

- 30 分钟内发生事件：用 `1/G(event_time-)` 加权。
- 已完整观察到 30 分钟仍无事件：用 `1/G(30-)` 加权。
- 30 分钟前删失、未成交：权重 0，但总体分母仍是全部评估行数。
- 正好在 30 分钟结束的 TIME_SLICE_END，是可观察完整窗口，不会因为删失分布在 30 分钟跳到 0 而丢失。
- 缺少 censoring support 时返回不可用状态，不硬截极大权重。

这要求训练与评估期删失分布具有可迁移性，并满足边际独立删失。主动 RFQ、撤单、加仓或改价往往与流动性/隐含需求相关，可能违反假设。IPCW 在这里是附加诊断，不是模型正确性的保证；条件删失模型、竞争风险或后续前瞻验证需要另行设计。

## 保存后预测与价格曲线

```python
import joblib
from fill_model_v1 import score_price_grid

bundle = joblib.load("runs/fill_v1/models.joblib")
model = bundle["models"][bundle["selected_model_name"]]
# p30 = model.predict_proba(new_states, horizon_minutes=30)

# 仅检查模型在固定状态下的数学价格响应。
# curve = score_price_grid(bundle["models"]["price"], new_states.head(10),
#                          deltas=[0.05, 0.10, 0.15, 0.20, 0.30])
```

价格曲线保持其他输入不变，并同步重算所有 level 价格。它没有模拟真实改价后 configuration age、客户响应或后续状态的变化，也没有判断候选价的条件历史支持，不能直接用作报价建议。

实现参考：[SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)、[scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)、[分段指数模型与 Poisson exposure 的关系](https://grodri.github.io/glms/notes/c7s4)。
