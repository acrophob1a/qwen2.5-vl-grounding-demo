#!/usr/bin/env python3
"""
Run grounding benchmark inference and write before/after comparison logs.

Usage:
  # Before SFT (base model)
  python inference/grounding_sft_models/run_comparison.py --phase before

  # After SFT (use latest checkpoint or explicit path)
  python inference/grounding_sft_models/run_comparison.py --phase after \\
      --model-path work_dirs/grounding-sft-0/checkpoint-630

  # Regenerate markdown report only
  python inference/grounding_sft_models/run_comparison.py --report-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualizer import DoVisualize, Wrapper

from inference.grounding_sft_models.benchmark_cases import (
    BENCHMARK_CASES,
    DEFAULT_AFTER_MODEL,
    DEFAULT_BASE_MODEL,
    LOG_ROOT,
)


def detect_output_format(raw_output: str) -> str:
    text = raw_output or ""
    if "<|object_ref_start|>" in text:
        return "sft_special_tokens"
    if re.search(r"```json|bbox_2d", text):
        return "json"
    if "<points" in text or "x1=" in text:
        return "points_xml"
    if text.strip():
        return "other_text"
    return "empty"


def count_boxes(predictions: dict) -> int:
    total = 0
    for annotations in predictions.values():
        total += len(annotations)
    return total


def summarize_case(result: dict) -> dict:
    predictions = result.get("extracted_predictions") or {}
    raw = result.get("raw_output") or ""
    boxes = count_boxes(predictions)
    return {
        "image": result["image"],
        "task": result["task"],
        "success": result["success"],
        "output_format": detect_output_format(raw),
        "categories_requested": result["categories"],
        "categories_parsed": list(predictions.keys()),
        "num_boxes": boxes,
        "parse_ok": boxes > 0,
        "raw_output_chars": len(raw),
        "inference_time_s": round(result.get("inference_time", 0), 2),
        "visualization": result.get("visualization"),
    }


def run_phase(
    model_path: str,
    phase: str,
    attn_implementation: str = "flash_attention_2",
) -> dict:
    log_dir = PROJECT_ROOT / LOG_ROOT / phase
    vis_dir = log_dir / "visualizations"
    log_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = Wrapper(
        model_path=model_path,
        backend="transformers",
        attn_implementation=attn_implementation,
        max_tokens=512,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.2,
    )

    cases = []
    for rel_path, task, categories in BENCHMARK_CASES:
        image_path = PROJECT_ROOT / rel_path
        image = Image.open(image_path).convert("RGB")
        results = model.inference(images=image, task=task, categories=categories)
        result = results[0]

        vis_rel = None
        predictions = result.get("extracted_predictions") or {}
        if result.get("success"):
            vis_path = vis_dir / f"{image_path.stem}_visualize.jpg"
            if predictions:
                vis_image = DoVisualize(
                    image=image,
                    predictions=predictions,
                    font_size=20,
                    draw_width=5,
                    show_labels=True,
                )
                vis_image.save(vis_path)
                vis_rel = str(vis_path.relative_to(PROJECT_ROOT))

        case_record = {
            "image": rel_path,
            "task": task,
            "categories": categories,
            "success": result.get("success", False),
            "error": result.get("error"),
            "raw_output": result.get("raw_output"),
            "extracted_predictions": predictions,
            "inference_time": result.get("inference_time"),
            "visualization": vis_rel,
            "summary": {},
        }
        case_record["summary"] = summarize_case(case_record)
        cases.append(case_record)
        print(
            f"[{phase}] {rel_path}: format={case_record['summary']['output_format']} "
            f"boxes={case_record['summary']['num_boxes']}"
        )

    payload = {
        "phase": phase,
        "model_path": model_path,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attn_implementation": attn_implementation,
        "cases": cases,
        "aggregate": {
            "total_images": len(cases),
            "parse_ok_count": sum(1 for c in cases if c["summary"]["parse_ok"]),
            "total_boxes": sum(c["summary"]["num_boxes"] for c in cases),
            "output_formats": {
                fmt: sum(1 for c in cases if c["summary"]["output_format"] == fmt)
                for fmt in sorted({c["summary"]["output_format"] for c in cases})
            },
        },
    }

    json_path = log_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(PROJECT_ROOT)}")
    return payload


def load_results(phase: str) -> dict | None:
    path = PROJECT_ROOT / LOG_ROOT / phase / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generate_report() -> Path:
    before = load_results("before")
    after = load_results("after")

    lines = [
        "# Grounding SFT Before / After Comparison",
        "",
        "Automated benchmark log for open-vocabulary detection on fixed test images.",
        "",
        "## How this log is produced",
        "",
        "```bash",
        "# Before fine-tuning",
        "python inference/grounding_sft_models/run_comparison.py --phase before",
        "",
        "# After fine-tuning (replace checkpoint path)",
        "python inference/grounding_sft_models/run_comparison.py --phase after \\",
        "  --model-path work_dirs/grounding-sft-0/checkpoint-XXX",
        "```",
        "",
        "Artifacts:",
        "- `logs/comparison/before/results.json` — base model raw outputs",
        "- `logs/comparison/after/results.json` — SFT model raw outputs",
        "- `logs/comparison/*/visualizations/` — side-by-side visualization images",
        "- This file (`COMPARISON.md`) — human-readable summary",
        "",
    ]

    def section(title: str, data: dict | None, pending: bool = False) -> list[str]:
        out = [f"## {title}", ""]
        if data is None:
            out.append("*Pending — run the command above after fine-tuning.*" if pending else "*Not recorded yet.*")
            out.append("")
            return out
        out.extend(
            [
                f"- **Model**: `{data['model_path']}`",
                f"- **Recorded (UTC)**: {data['timestamp_utc']}",
                f"- **Attention**: {data.get('attn_implementation', 'n/a')}",
                f"- **Images**: {data['aggregate']['total_images']}",
                f"- **Parse OK (boxes drawn)**: {data['aggregate']['parse_ok_count']}/{data['aggregate']['total_images']}",
                f"- **Total boxes**: {data['aggregate']['total_boxes']}",
                f"- **Output formats**: {data['aggregate']['output_formats']}",
                "",
                "| Image | Format | Boxes | Parse OK | Viz |",
                "|-------|--------|------:|----------|-----|",
            ]
        )
        for case in data["cases"]:
            s = case["summary"]
            viz = case.get("visualization") or "—"
            if viz != "—":
                viz = f"`{viz}`"
            ok = "yes" if s["parse_ok"] else "no"
            out.append(
                f"| `{case['image']}` | {s['output_format']} | {s['num_boxes']} | {ok} | {viz} |"
            )
        out.append("")
        return out

    lines.extend(section("Before SFT (base model)", before))
    lines.extend(section("After SFT", after, pending=True))

    if before and after:
        lines.extend(["## Side-by-side delta", ""])
        after_by_image = {c["image"]: c for c in after["cases"]}
        lines.extend(
            [
                "| Image | Before boxes | After boxes | Before format | After format |",
                "|-------|-------------:|------------:|---------------|--------------|",
            ]
        )
        for bc in before["cases"]:
            img = bc["image"]
            ac = after_by_image.get(img)
            if not ac:
                continue
            bs, as_ = bc["summary"], ac["summary"]
            delta = as_["num_boxes"] - bs["num_boxes"]
            delta_str = f"{as_['num_boxes']} ({delta:+d})"
            lines.append(
                f"| `{img}` | {bs['num_boxes']} | {delta_str} | {bs['output_format']} | {as_['output_format']} |"
            )
        lines.append("")

    lines.extend(["## Per-case raw outputs", ""])
    for phase_name, data in [("before", before), ("after", after)]:
        lines.append(f"### {phase_name}")
        lines.append("")
        if not data:
            lines.append("*Not available.*")
            lines.append("")
            continue
        for case in data["cases"]:
            lines.extend(
                [
                    f"#### `{case['image']}`",
                    "",
                    f"- Categories: `{case['categories']}`",
                    f"- Summary: {json.dumps(case['summary'], ensure_ascii=False)}",
                    "",
                    "<details>",
                    "<summary>Raw model output</summary>",
                    "",
                    "```",
                    (case.get("raw_output") or "").strip()[:8000],
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )

    report_path = PROJECT_ROOT / LOG_ROOT / "COMPARISON.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path.relative_to(PROJECT_ROOT)}")
    return report_path


def resolve_after_model(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return str(p)
    if p.is_dir() and not (p / "config.json").exists():
        checkpoints = sorted(p.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
        if checkpoints:
            return str(checkpoints[-1])
    return str(p)


def main():
    parser = argparse.ArgumentParser(description="Grounding SFT before/after comparison log")
    parser.add_argument("--phase", choices=["before", "after"], help="Record base or SFT model")
    parser.add_argument("--model-path", help="Model or checkpoint directory")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--report-only", action="store_true", help="Rebuild COMPARISON.md only")
    args = parser.parse_args()

    if args.report_only:
        generate_report()
        return

    if not args.phase:
        parser.error("--phase is required unless --report-only")

    if args.model_path:
        model_path = args.model_path
    elif args.phase == "before":
        model_path = DEFAULT_BASE_MODEL
    else:
        model_path = resolve_after_model(DEFAULT_AFTER_MODEL)

    run_phase(model_path, args.phase, args.attn_implementation)
    generate_report()


if __name__ == "__main__":
    main()
