# Qwen2.5-VL 视觉 Grounding 微调

基于 Qwen2.5-VL-3B-Instruct 的开放词汇目标检测（Visual Grounding）微调实践：在 1K 标注数据上完成监督微调（SFT），解决基座模型输出格式不稳定、重复生成、解析失败等问题，实现结构化、可消费的检测结果输出。

## 效果展示

**人物检测** — 微调前仅检出 person/boy 两类，微调后覆盖所有指定类别（person / boy / ball / grass / tree），框更完整：



**咖啡馆场景** — 基座模型陷入 JSON 重复循环（输出 6565 字符、耗时 142 秒、解析失败），微调后 7.3 秒输出结构化结果，检出 6 类目标：


### 量化对比

| 指标 | 基座模型 | 微调后 |
|------|---------|--------|
| 解析成功率 | 3/5（60%） | **5/5（100%）** |
| 检测框总数 | 22 | **65** |
| 输出格式 | JSON / XML 混用、重复循环 | 统一 grounding 专用 token |
| cafe 场景推理耗时 | 142s（失败） | **7.3s** |

## 方法概述

1. **基座模型**：Qwen2.5-VL-3B-Instruct（3B 参数），原生具备视觉理解能力但未针对 grounding 任务做格式对齐
2. **微调数据**：Grounding-ToyData（1000 条标注样本），覆盖人物、日常物体、室内场景等多类检测目标
3. **训练配置**：全参数微调 Vision Encoder + Projector + LLM，A800 80GB 单卡，有效 batch size 16，10 epochs，lr=5e-5（vision encoder 5e-6），训练时长约 68 分钟
4. **核心改动**：启用 grounding 专用特殊 token（`<|object_ref_start|>`、`<|box_start|>` 等），将模型输出从自由格式 JSON/XML 对齐为结构化坐标 token，从根本上消除重复生成和解析失败问题

## 项目结构

```
├── finetuning/          # SFT 训练代码、配置与启动脚本
├── inference/           # 推理脚本与测试图片
├── visualizer/          # 检测框解析与可视化模块
├── datasets/            # Grounding-ToyData（图片需另行下载）
└── records/results/     # 微调前后对比可视化
```

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install -e .
pip install mmengine liger-kernel ujson
pip install --no-deps -e finetuning

# 下载基座模型（约 7GB）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir pretrained/Qwen2.5-VL-3B-Instruct
```

> 数据集图片 `toy_data.images.tsv`（约 121MB）因 GitHub 大小限制未纳入仓库，可从 [awesome-demos](https://huggingface.co/datasets/Brilliant-B/awesome-demos) 获取 `demo1.tar.gz`，解压至 `datasets/Grounding-ToyData/`。

### 推理

```bash
python inference/grounding_sft_models/detection_inference.py
```

对 `inference/test_images/` 下的图片批量推理，输出带检测框的可视化结果。

### 微调

```bash
bash finetuning/scripts/sft_gnd.sh
```

checkpoint 保存至 `work_dirs/grounding-sft-0/`。

## 技术栈

- **模型**：Qwen2.5-VL-3B-Instruct
- **训练框架**：PyTorch、Transformers、Flash Attention 2
- **数据处理**：TSV 数据集、自定义 collator
- **可视化**：OpenCV、Pillow

## 致谢

训练框架基于 [Brilliant-B/awesome-demos](https://huggingface.co/datasets/Brilliant-B/awesome-demos) 提供的 Grounding SFT 方案。
