# 实验主日志

> 规范见 [`EXPERIMENT_TRACE.md`](../EXPERIMENT_TRACE.md)。所有指标来自真实训练/推理日志，未编造。

---

## exp-001 — Qwen2.5-VL Grounding SFT — 2026-05-31

**状态**：已完成

**动机**：基座模型 Qwen2.5-VL-3B-Instruct 在开放词汇检测任务上输出格式不稳定（JSON / XML 混用、解析失败、重复生成），需通过 SFT 对齐 grounding 专用 token 格式，提升可解析性与检测覆盖。

**数据版本**：data-v1

**配置**：

| 项 | 值 |
|----|-----|
| run_name | grounding-sft-0 |
| base model | Qwen2.5-VL-3B-Instruct (3B) |
| 微调范围 | Vision + MLP + LLM 全参数 |
| train samples | 1000 |
| epochs | 10 |
| batch (per device × grad accum) | 4 × 4 = 有效 16 |
| learning_rate | 5e-5 (projector 5e-5, vision 5e-6) |
| GPU | NVIDIA A800 80GB × 1 |
| attention | flash_attention_2 |

**配置快照**：[`configs/snapshots/exp-001.yaml`](../configs/snapshots/exp-001.yaml)

**命令**：
```bash
cd /root/autodl-tmp/demo1
bash finetuning/scripts/sft_gnd.sh
```

**结果**：
- 产出路径：`work_dirs/grounding-sft-0/`（最终权重 + `checkpoint-620`）
- 训练时长：约 68 分钟（`train_runtime=4070.96s`）
- global_step：620
- 平均 train_loss：**1.81**（首 step loss 14.88 → 末 step loss 2.38，最低单步 0.83）
- 可训练参数量：3,756,115,968（约 3.76B）

**微调前后对比**（5 张固定测试图，脚本 `run_comparison.py`）：

| 指标 | 微调前（基座） | 微调后（SFT） |
|------|---------------|---------------|
| 解析成功率 | 3/5 | **5/5** |
| 总检测框数 | 22 | **65** |
| 输出格式 | JSON / points_xml 混用 | 统一 sft_special_tokens |
| 详细报告 | [`logs/comparison/COMPARISON.md`](../logs/comparison/COMPARISON.md) | 同上 |

**观察与结论**：
1. SFT 后模型稳定输出 grounding 专用 token，解析器 100% 成功。
2. 原先完全失败的 `cafe.jpg`、`gui.png` 在微调后可正常出框。
3. 基座模型在 `cafe.jpg` 上出现 JSON 重复循环（6565 chars），SFT 后消除。
4. 推理速度提升：cafe 从 142s → 7.3s（输出长度大幅缩短）。

**下一步**：
- 可选：在更大数据集（RefCOCO / ODinW）上评估 mAP
- 可选：LoRA 对比实验降低显存占用

---

## exp-000 — 基座模型推理基线 — 2026-05-30

**状态**：已完成

**动机**：记录微调前基座模型在固定 benchmark 上的表现，作为 exp-001 对照。

**数据版本**：—（推理阶段，无训练数据变更）

**命令**：
```bash
python inference/grounding_sft_models/run_comparison.py --phase before
```

**结果**：
- 模型：`pretrained/Qwen2.5-VL-3B-Instruct`
- 解析成功 3/5，总框 22
- 原始输出：`logs/comparison/before/results.json`
- 可视化：`logs/comparison/before/visualizations/`

**关联实验**：exp-001（after 阶段对比）

---
