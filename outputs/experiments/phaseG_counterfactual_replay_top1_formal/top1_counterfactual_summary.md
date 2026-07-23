# Top-1 反事实 Value-Swap 重放汇总

## 实验设置

- 任务：multi-entity delayed recall
- 字段：color, city, age, job
- 每个样本实体数：4
- Delay：32
- Core slots：16
- 模型变体：specialized_core
- 路由方式：Top-1 hard slot routing
- 干预方式：固定 entity、field 和 position，只替换被查询 fact 的 value，以及对应 answer value
- 每个 checkpoint 重放样本数：64
- 每个 checkpoint 反事实样本数：576

## 结果

| 条件 | Seed | 写槽不变率 | 读槽不变率 | 写槽 L1 变化 | 读槽 L1 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定字段顺序 | 42 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 固定字段顺序 | 43 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 打乱字段顺序 | 42 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 打乱字段顺序 | 43 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## 解读

Top-1 重放强化了之前 soft-routing 的结果。在 value-only 干预下，所有 checkpoint、所有被测试的反事实样本中，选中的写槽和读槽都保持不变。

由于这里使用的是 hard Top-1 路由，路由权重的 L1 变化为 `0.0000` 是预期结果：argmax 之后只有“选中槽改变”或“选中槽不变”两种情况，不再存在 soft 权重微调。实验结果显示，value-only swap 没有改变选中的 slot。

当前证据支持下面这个更严格、边界清楚的结论：

> 在四字段 multi-entity delayed recall、16 slots、delay=32 设置下，无论 soft routing 还是 Top-1 hard routing，slot selection 都对 value-only counterfactual swap 保持不变。

机制层面的解释是：

- value 内容不决定主槽分配；
- entity、field、position 或模型学到的结构上下文主导 slot selection；
- value 仍可能影响下游 hidden activation 或 logits，但在这组干预下不改变被选中的 slot。

## 关联产物

- 固定字段顺序 seed 42：`baseline_slots16/replay_seed42/counterfactual_value_swap_replay.md`
- 固定字段顺序 seed 43：`baseline_slots16/replay_seed43/counterfactual_value_swap_replay.md`
- 打乱字段顺序 seed 42：`shuffle_fields_slots16/replay_seed42/counterfactual_value_swap_replay.md`
- 打乱字段顺序 seed 43：`shuffle_fields_slots16/replay_seed43/counterfactual_value_swap_replay.md`
