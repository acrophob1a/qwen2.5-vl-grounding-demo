# 简历 / 作品集描述

> 基于 [`records/experiments.md`](records/experiments.md) 真实实验记录生成。可直接复制到简历、GitHub About、面试作品集。

**项目仓库**：https://github.com/acrophob1a/qwen2.5-vl-grounding-demo

---

## 一句话版本（简历项目标题下）

基于 Qwen2.5-VL-3B 完成视觉 Grounding SFT 微调，将开放词汇检测解析成功率从 60% 提升至 100%，并统一输出格式为可解析的 bounding box token。

---

## 标准版（3–4 行，适合简历「项目经历」）

**Qwen2.5-VL 视觉 Grounding 微调** | 个人项目 | 2026.05

- 在 A800 GPU 上对 Qwen2.5-VL-3B-Instruct 进行全参数 SFT（Vision + LLM），使用 1000 条 Grounding 标注数据训练 10 epoch（约 68 分钟）
- 设计微调前后自动化对比 pipeline（`run_comparison.py`），在 5 类场景（人物、咖啡馆、GUI、文档版面、桌面物品）上量化评估
- 微调后解析成功率 3/5 → **5/5**，输出格式从 JSON/XML 混用统一为 grounding 专用 token，消除基座模型的重复生成问题
- 技术栈：PyTorch、Transformers、Flash Attention 2、Qwen2.5-VL、Gradio 可视化

---

## 详细版（作品集 / GitHub README / 面试口述）

### 背景

Qwen2.5-VL 基座模型具备视觉理解能力，但在开放词汇目标检测（Visual Grounding）任务上，输出格式不稳定：有时返回 JSON bbox，有时返回 XML points，甚至出现 JSON 重复循环导致解析失败。

### 方案

1. 基于 [Brilliant-B/awesome-demos](https://huggingface.co/datasets/Brilliant-B/awesome-demos) 提供的 Grounding SFT 框架，使用 Grounding-ToyData（1000 样本）进行监督微调
2. 启用 grounding 专用 token（`<|object_ref_start|>`, `<|box_start|>` 等），全参数微调 Vision Encoder + Projector + LLM
3. 构建可复现的 before/after 对比 benchmark，固定 5 张测试图与类别 prompt

### 关键结果（真实日志）

| 指标 | 微调前 | 微调后 |
|------|--------|--------|
| 解析成功率 | 3/5 (60%) | **5/5 (100%)** |
| 总检测框 | 22 | 65 |
| 训练 loss | — | 1.81 (avg) |
| 训练时长 | — | ~68 min / 620 steps |

### 个人贡献

- 完成环境搭建（flash-attn 编译、依赖最小化安装）
- 执行 SFT 训练并解决首次 OOM（调整 `save_total_limit`）
- 编写自动化对比评估脚本与可视化 pipeline
- 建立实验留痕体系（配置快照、实验日志、对比图）

### 可展示材料

- 对比可视化：[`records/results/exp-001/`](records/results/exp-001/)
- 完整实验日志：[`records/experiments.md`](records/experiments.md)
- 对比报告：[`logs/comparison/COMPARISON.md`](logs/comparison/COMPARISON.md)

---

## 面试追问预备

**Q: 为什么解析成功率不是 mAP？**
A: 本项目是 demo 级玩具数据集，没有 GT bbox 标注用于 mAP 计算。解析成功率衡量的是模型输出能否被下游 parser 稳定消费——这是 grounding 落地的前置条件。基座模型 2/5 完全无法解析，SFT 后 5/5 可解析，说明格式对齐有效。

**Q: 为什么 cafe 基座失败了？**
A: 基座输出了 6565 字符的 JSON，其中 chair 条目重复循环，parser 无法提取有效 bbox。SFT 后输出 1099 字符的结构化 token，7.3 秒完成推理。

**Q: 训练参数怎么选的？**
A: 沿用项目默认配置（lr=5e-5, cosine, 10 epochs），全参数微调 3B 模型在 A800 80GB 上 batch=4×accum4 可跑通。首次训练因 `save_total_limit=3` 磁盘不足 OOM，改为 1 后成功。

---

## 英文版（可选）

**Visual Grounding SFT on Qwen2.5-VL-3B** | Personal Project | May 2026

- Fine-tuned Qwen2.5-VL-3B-Instruct (full params: vision + LLM) on 1K grounding samples for 10 epochs (~68 min on A800)
- Built automated before/after benchmark on 5 diverse scenes; improved parse success rate from 60% to 100%
- Unified model output to grounding-specific tokens, eliminating JSON/XML format inconsistency and repetition loops
- Stack: PyTorch, Transformers, Flash Attention 2, Qwen2.5-VL
