import json
from pathlib import Path

from PIL import Image
from visualizer import DoVisualize, Wrapper

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEST_IMAGES_DIR = Path("inference/test_images")

# Per-image detection categories (open-vocabulary; adjust as needed)
TEST_IMAGE_CONFIG = {
    "boys.jpg": {
        "task": "detection",
        "categories": ["person", "boy", "ball", "grass", "tree"],
    },
    "cafe.jpg": {
        "task": "detection",
        "categories": ["cup", "coffee", "table", "chair", "person", "laptop"],
    },
    "gui.png": {
        "task": "gui_grounding",
        "categories": ["button", "search bar", "menu", "icon", "text field", "sidebar"],
    },
    "layout.jpg": {
        "task": "detection",
        "categories": ["title", "paragraph", "image", "table", "header", "footer"],
    },
}


def build_visualize_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_visualize.jpg")


def list_input_images(image_dir: Path) -> list[Path]:
    images = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem.endswith("_visualize"):
            continue
        images.append(path)
    return images


def run_inference_on_image(model: Wrapper, image_path: Path) -> None:
    config = TEST_IMAGE_CONFIG.get(
        image_path.name,
        {
            "task": "detection",
            "categories": ["person", "car", "dog"],
        },
    )

    image = Image.open(image_path).convert("RGB")
    results = model.inference(
        images=image,
        task=config["task"],
        categories=config["categories"],
    )
    result = results[0]

    print(f"\n{'=' * 60}")
    print(f"Image: {image_path}")
    print(f"Task: {config['task']}")
    print(f"Categories: {config['categories']}")

    if not result["success"]:
        print(f"Inference failed: {result['error']}")
        return

    print("\n=== Model raw output ===")
    print(result["raw_output"])

    predictions = result["extracted_predictions"]
    print("\n=== Parsed predictions ===")
    print(json.dumps(predictions, indent=2, ensure_ascii=False))

    vis_image = DoVisualize(
        image=image,
        predictions=predictions,
        font_size=20,
        draw_width=5,
        show_labels=True,
    )
    output_path = build_visualize_output_path(image_path)
    vis_image.save(output_path)
    print(f"\nVisualization saved to: {output_path}")


def main():
    model_path = "pretrained/Qwen2.5-VL-3B-Instruct"
    image_dir = TEST_IMAGES_DIR

    image_paths = list_input_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {image_dir}")

    print(f"Found {len(image_paths)} images in {image_dir}")

    model = Wrapper(
        model_path=model_path,
        backend="transformers",
        attn_implementation="sdpa",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    for image_path in image_paths:
        run_inference_on_image(model, image_path)


if __name__ == "__main__":
    main()
