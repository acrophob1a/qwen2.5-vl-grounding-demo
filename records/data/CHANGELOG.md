# 数据集变更日志

## data-v1 — 2026-05-30

**变更类型**：新增（项目初始数据）

**摘要**：使用 Grounding-ToyData 玩具数据集进行 grounding SFT 训练。

**详情**：
- 来源/路径：`datasets/Grounding-ToyData/`
- 标注文件：`toy_data.annotations.tsv`（1000 条）
- 图像索引：`toy_data.images.tsv`（约 121MB，未进 Git，需从 [awesome-demos](https://huggingface.co/datasets/Brilliant-B/awesome-demos) 获取）
- 任务格式：开放词汇检测，输出 `<|object_ref_start|>…<|box_start|>…` 特殊 token 坐标
- 关键参数：`image_min_pixels=12544`, `image_max_pixels=2007040`

**关联实验**：exp-001

**验证命令**：
```bash
wc -l datasets/Grounding-ToyData/toy_data.annotations.tsv
# 输出: 1000
```
