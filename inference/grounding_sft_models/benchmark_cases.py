"""Shared benchmark cases for before/after SFT comparison."""

from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# (relative path from project root, task, categories)
BENCHMARK_CASES = [
    (
        "inference/test_images/boys.jpg",
        "detection",
        ["person", "boy", "ball", "grass", "tree"],
    ),
    (
        "inference/test_images/cafe.jpg",
        "detection",
        ["cup", "coffee", "table", "chair", "person", "laptop"],
    ),
    (
        "inference/test_images/gui.png",
        "gui_grounding",
        ["button", "search bar", "menu", "icon", "text field", "sidebar"],
    ),
    (
        "inference/test_images/layout.jpg",
        "detection",
        ["title", "paragraph", "image", "table", "header", "footer"],
    ),
    (
        "testfiles/gndtest1.png",
        "detection",
        ["laptop", "potted plant", "white mug", "notebook", "pen", "pencil holder"],
    ),
]

DEFAULT_BASE_MODEL = "pretrained/Qwen2.5-VL-3B-Instruct"
DEFAULT_AFTER_MODEL = "work_dirs/grounding-sft-0"
LOG_ROOT = Path("logs/comparison")
