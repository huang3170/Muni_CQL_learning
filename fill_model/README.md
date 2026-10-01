# Municipal bond first-fill model v1

输入是已经构建好的 interval pandas DataFrame。一行对应某 CUSIP 整套有效报价的一段可观测挂单时间，最多一个有效首次成交事件。输出是每分钟成交到达率和给定窗口（默认 30 分钟）的首次成交概率。

本版本完成 any-first-fill 到达率训练与事实评估。Level routing、fill size、价格优化、RL 和因果价格效果不在本版本中。报价数据及源文件不会被修改。

## 安装与直接使用 DataFrame

将整个 `fill_model_v1` 文件夹放在 notebook 或项目目录下，保留文件夹名称。在项目目录执行：

```bash
python -m pip install -r fill_model_v1/requirements.txt
```

```python
from fill_model_v1 import TrainingConfig, run_experiment

# df = 你的 interval dataframe，不需要先做随机拆分、one-hot 或标准化。
config = TrainingConfig(
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

# 若 new_states 的列和输入状态相同，可不包含 event/end_reason 等标签字段。
# 必须包含开始时间、库存、active masks、CEP 来源时间、delta、gaps、configuration age。
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
    "position_episode_id": "string", "quote_config_id": "string",
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

对照之前的 32 列模板。`core.prepare_data` 的 `required` 列表是本代码的最小必需输入。无需 routing 或 size 标签也能训练 arrival，但 `fill_event_id` 和执行时间必须可信。

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
- UTC 列推荐 timezone-aware 或带 `Z` 的时间字符串。代码把 UTC 列中的 naive 时间解释为 UTC，不会假定为纽约时间。
- `cep_asof_time_utc` 必须不晚于起点。其他市场特征也必须在起点已经可用；代码无法仅凭聚合数值发现报告延迟或未来信息。

此版本假设每个 CUSIP 同时只有一个合并后的可售持仓。如果同一 CUSIP 有多个独立 desk/book，需先把 book 标识纳入实体键再使用，不能忽略重叠检查。

## 预测特征与泄漏防护

代码使用明确 allowlist，绝不自动把 dataframe 的所有列送入模型。

核心状态：log inventory、log configuration age、level gaps、active template、日内 sin/cos、CEP age、venue 构成。可选状态：duration、rating、sector、历史市场成交次数/面值、最近成交年龄、距加仓时间、market spread、历史 CEP 变化/波动。缺少可选字段可运行，完全缺失的数值特征不拟合。

中位数填补、缺失指示、标准化、低频分类合并全部只在训练期拟合。Exposure 用于损失和训练标准化权重，不作为预测特征。`end_reason`、event、fill level/size、结束时间、episode 最终时长和成交后库存均不进入特征矩阵。

`no_price` 仍包含 level gaps，因为它们描述报价结构；排除的是当前 common shift，不代表排除所有市场定价信息。

## 划分、调参和模型选择

默认按实际出现的纽约交易日期约 60% / 20% / 20% 划分。没有随机行级拆分。

一个 parent episode 或一个 quote configuration 跨边界时，整个组从本次训练/验证/测试中排除。跨界损失写入 split audit；超过 10% 会提醒。长 episode 更容易被排除，必须检查这种选择偏差。若损失过多，应重新设计日期和评估协议，而不是直接接受当前结果。

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
| `metadata.json` | 日期边界、选择规则、超参数、版本、假设和价格系数 |
| `models.joblib` | 三个模型和选择结果；仅加载自己信任的模型文件 |

优先比较 `event_time_nll_per_30_exposure_minutes`，越小越好。它是事件似然按暴露量归一化，不是 binary log loss；单位变更或不同数据集合的绝对数值不能直接比较。

`integrated_hazard_to_events = sum(lambda_i * observed_exposure_i) / sum(event_i)`，理想情况下接近 1。它使用成交停止后的实际 exposure，是累计强度/事件校准诊断，**不能称为未来窗口的 expected fill count，也不是 sum(p30)/events**。各分组检查可定位系统性偏差，稀少事件分组的比值会很不稳定。

bootstrap 的 NLL difference 负数表示前一个模型更好。例如 `price-no_price`。重采样单位是完整 parent episode，保持内部关联；不会涵盖跨 episode 的共同市场日冲击，也不包含重训/选参不确定性。结果是初步条件区间，不能证明价格因果效果。

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
