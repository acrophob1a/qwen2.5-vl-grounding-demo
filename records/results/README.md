# 实验结果索引

| 文件 | 实验 | 说明 |
|------|------|------|
| `exp-001/*_compare.jpg` | exp-001 | **并排对比图**（推荐展示 / 简历） |
| `exp-001/boys_before.jpg` | exp-000 / exp-001 | 基座模型 — 人物检测（boys.jpg） |
| `exp-001/boys_after.jpg` | exp-001 | SFT 后 — 人物检测，框更完整 |
| `exp-001/cafe_after.jpg` | exp-001 | SFT 后 — 咖啡馆场景（基座完全失败，无 before 图） |
| `exp-001/gui_after.jpg` | exp-001 | SFT 后 — GUI 元素 grounding（基座输出 points 无法解析） |
| `exp-001/layout_before.jpg` | exp-000 / exp-001 | 基座 — 文档版面检测 |
| `exp-001/layout_after.jpg` | exp-001 | SFT 后 — 文档版面检测 |
| `exp-001/gndtest1_before.jpg` | exp-000 / exp-001 | 基座 — 桌面物品检测 |
| `exp-001/gndtest1_after.jpg` | exp-001 | SFT 后 — 桌面物品检测 |

完整对比报告（含原始模型输出）：[`logs/comparison/COMPARISON.md`](../../logs/comparison/COMPARISON.md)

复现对比：
```bash
python inference/grounding_sft_models/run_comparison.py --phase before
python inference/grounding_sft_models/run_comparison.py --phase after \
  --model-path work_dirs/grounding-sft-0
```
