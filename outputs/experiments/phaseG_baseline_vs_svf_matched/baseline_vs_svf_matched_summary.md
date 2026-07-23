# Matched Baseline vs SVF 对照汇总

## 实验目的

本实验用于检验 SVF-Transformer 是否不仅具备结构可解释性，还在 delayed recall 性能上优于参数量接近的普通 Transformer baseline。

为了避免“SVF 参数更多所以更强”的混淆，baseline 使用 `--match-baseline-to specialized_core` 自动放大到与 `specialized_core` 接近的参数量。

## 设置

- 任务：multi-entity delayed recall
- 字段：color, city, age, job
- 每个样本实体数：4
- Delay：32
- Seeds：42, 43
- Steps：300
- Batch size：16
- Train samples：512
- Val samples：128
- SVF 变体：specialized_core
- Baseline：无 memory、无 persistent core、无 structural loss 的普通 Transformer
- 参数量：
  - baseline：约 138,240
  - specialized_core：约 136,064

## 结果

### 固定字段顺序

| Variant | Runs | Val CE Loss | Answer Exact Acc | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2 | 2.5508 | 0.1055 | 138,240 |
| specialized_core | 2 | 2.6851 | 0.1055 | 136,064 |

结论：固定字段顺序下，二者 answer accuracy 持平；baseline 的 validation CE 更低。

### 打乱字段顺序

| Variant | Runs | Val CE Loss | Answer Exact Acc | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2 | 2.9279 | 0.1211 | 138,240 |
| specialized_core | 2 | 3.0606 | 0.1016 | 136,064 |

结论：打乱字段顺序下，baseline 在 validation CE 和 answer accuracy 上都优于 specialized_core。

## 当前判断

这组 matched 对照不支持“SVF 在当前 delayed recall 设置下有性能优势”。

更准确的结论是：

> 当前 SVF 优势主要体现在机制可解释性和反事实可重放性，而不是短程 delayed recall 的直接性能提升。

这和 Top-1 / soft routing 的 counterfactual replay 结果并不矛盾：

- replay 证明 SVF 的 slot selection 具有稳定结构；
- matched baseline 对照说明这种结构在当前训练规模和任务难度下没有转化成更高 answer accuracy；
- 因此 SVF 当前是“更可解释”，不是“已证明更强”。

## 后续更合理的性能检验

如果要继续寻找 SVF 的性能优势，应扩大任务压力，而不是只在当前短训练设置下比较：

1. 更长 delay：64、128、256
2. 更多实体：6、8、10
3. 更多字段：5、6、7
4. 多字符 value：value_length 2 或 3
5. 更长训练：1000 到 3000 steps
6. 比较 degradation curve：随着 delay 增加，baseline 是否比 SVF 更快退化

当前最值得做的下一组实验：

```powershell
python train_delayed_recall.py --variants baseline specialized_core --match-baseline-to specialized_core --task-type multi_entity --delays 32 64 128 --seeds 42 43 --steps 1000 --batch-size 16 --train-samples 1024 --val-samples 256 --entities-per-sample 4 --fields color city age job --value-length 1 --d-model 64 --d-ff 128 --layers 1 --heads 4 --core-slots 16 --eval-interval 500 --device cuda --output-dir outputs/experiments/phaseH_delay_scaling_matched/fixed_order
```

如果 SVF 真有长期结构记忆优势，应该在 delay 增大时表现为：

- answer accuracy 下降更慢；
- validation CE 增长更慢；
- core norm / slot structure 保持稳定；
- baseline 在长 delay 下更早退化。

## 关联产物

- 固定字段顺序 aggregate：`baseline_fixed_order/recall_aggregate_20260724_053739.json`
- 打乱字段顺序 aggregate：`shuffle_fields/recall_aggregate_20260724_053757.json`
