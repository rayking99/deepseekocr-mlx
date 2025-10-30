"""Simple DeepSeek-OCR demo script.

Run with:
    /Users/jso/code/chi/dsocr/.venv/bin/python run_demo.py
The script loads the cached Hugging Face checkpoint, preprocesses
``assets/demo.jpg``, and prints the generated markdown transcription.
"""

from __future__ import annotations

import ast
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, cast
import time

import mlx.core as mx
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from deepseek_ocr_mlx import DeepSeekOCRPreprocessor, GenerationConfig, generate
from deepseek_ocr_mlx.load import load as load_model, load_config

PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
IMAGE_PATH = Path("assets/test1.jpeg")
HUGGING_FACE_ID = "deepseek-ai/DeepSeek-OCR"
OUTPUT_DIR = Path("demo_output")


@dataclass
class Detection:
    raw: str
    label: str
    boxes: List[Tuple[float, float, float, float]]


_DET_PATTERN = re.compile(
    r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)",
    re.DOTALL,
)


def _parse_detections(text: str) -> List[Detection]:
    detections: List[Detection] = []
    for full, label, coords in _DET_PATTERN.findall(text):
        label_clean = label.strip()
        try:
            parsed = ast.literal_eval(coords.strip())
        except (ValueError, SyntaxError):
            parsed = []

        boxes: List[Tuple[float, float, float, float]] = []
        if isinstance(parsed, (list, tuple)):
            for item in parsed:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 4
                    and all(isinstance(val, (int, float)) for val in item)
                ):
                    values = tuple(float(val) for val in item)
                    boxes.append(cast(Tuple[float, float, float, float], values))

        detections.append(Detection(raw=full, label=label_clean, boxes=boxes))
    return detections


def _scale_box(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        return 0, 0, 0, 0

    x1, y1, x2, y2 = box
    scale = 999.0
    left = max(0, min(width, int(round(x1 / scale * width))))
    top = max(0, min(height, int(round(y1 / scale * height))))
    right = max(0, min(width, int(round(x2 / scale * width))))
    bottom = max(0, min(height, int(round(y2 / scale * height))))
    return left, top, right, bottom


def _save_image_crops(
    image: Image.Image,
    detections: Sequence[Detection],
    images_dir: Path,
) -> dict[str, str]:
    images_dir.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    counter = 0
    replacements: dict[str, str] = {}

    for detection in detections:
        if detection.label.lower() != "image":
            continue

        replacement = ""
        for box in detection.boxes:
            left, top, right, bottom = _scale_box(box, width, height)
            if right <= left or bottom <= top:
                continue
            crop = image.crop((left, top, right, bottom))
            crop_path = images_dir / f"{counter}.jpg"
            crop.save(crop_path)
            replacement = f"![](images/{counter}.jpg)\n"
            counter += 1
            break

        replacements[detection.raw] = replacement

    return replacements


def _annotate_image(
    image: Image.Image,
    detections: Sequence[Detection],
) -> Image.Image:
    annotated = image.copy()
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(annotated)
    overlay_draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    palette = [
        (229, 57, 53),
        (30, 136, 229),
        (67, 160, 71),
        (255, 179, 0),
        (142, 36, 170),
        (0, 150, 136),
    ]

    width, height = annotated.size
    for idx, detection in enumerate(detections):
        color = palette[idx % len(palette)]
        fill = (*color, 40)
        line_width = 4 if detection.label.lower() == "title" else 2

        for box in detection.boxes:
            left, top, right, bottom = _scale_box(box, width, height)
            if right <= left or bottom <= top:
                continue
            draw.rectangle([left, top, right, bottom], outline=color, width=line_width)
            overlay_draw.rectangle([left, top, right, bottom], fill=fill)

            label_text = detection.label or "text"
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = left
            text_y = max(0, top - text_height - 2)
            draw.rectangle(
                [text_x, text_y, text_x + text_width + 4, text_y + text_height + 2],
                fill=(255, 255, 255),
            )
            draw.text((text_x + 2, text_y + 1), label_text, font=font, fill=color)

    annotated.paste(overlay, (0, 0), overlay)
    return annotated


def _render_markdown(
    raw_text: str,
    detections: Sequence[Detection],
    image_replacements: dict[str, str],
) -> str:
    cleaned = raw_text
    for detection in detections:
        if detection.label.lower() == "image":
            replacement = image_replacements.get(detection.raw, "")
        else:
            replacement = ""
        cleaned = cleaned.replace(detection.raw, replacement, 1)

    cleaned = cleaned.replace("<｜end▁of▁sentence｜>", "")
    cleaned = cleaned.replace("\\coloneqq", ":=")
    cleaned = cleaned.replace("\\eqqcolon", "=:")
    return cleaned.strip()


def save_demo_outputs(image: Image.Image, raw_text: str, output_dir: Path) -> str:
    detections = _parse_detections(raw_text)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    image_rgb = image.convert("RGB")
    replacements = _save_image_crops(image_rgb, detections, images_dir)
    annotated = (
        _annotate_image(image_rgb, detections) if detections else image_rgb.copy()
    )
    annotated.save(output_dir / "result_with_boxes.jpg")

    markdown = _render_markdown(raw_text, detections, replacements)
    (output_dir / "result.md").write_text(markdown + "\n", encoding="utf-8")

    return markdown


def _silence_model_debug_logs() -> None:
    from deepseek_ocr_mlx import model as model_impl  # Local import to avoid cycles

    setattr(model_impl, "print", lambda *args, **kwargs: None)


def main() -> None:
    start = time.time()
    print(f"Using model weights from: {HUGGING_FACE_ID}")

    _silence_model_debug_logs()

    model_dir = Path(snapshot_download(HUGGING_FACE_ID))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = load_config(model_dir)
    preprocessor = DeepSeekOCRPreprocessor(tokenizer, config)

    with Image.open(IMAGE_PATH) as img:
        image = img.convert("RGB")
    batch = preprocessor.prepare_single(PROMPT, [image])

    model = load_model(model_dir, lazy=False)
    mx.eval(model.parameters())

    eos_ids_attr = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_ids_attr, int):
        stop_ids = [eos_ids_attr]
    elif eos_ids_attr:
        stop_ids = list(eos_ids_attr)
    else:
        stop_ids = []

    if not stop_ids and tokenizer.eos_token_id is not None:
        stop_ids = [tokenizer.eos_token_id]

    gen_config = GenerationConfig(
        max_new_tokens=1024,
        temperature=0.2,
        eos_token_id=tokenizer.eos_token_id,
        stop_token_ids=stop_ids,
        skip_special_tokens=False,
    )

    print(f"Setup completed in {time.time() - start:.2f}s. Running generation…")
    decode_start = time.time()
    result = generate(model, tokenizer, batch, gen_config)
    print(f"Generation finished in {time.time() - decode_start:.2f}s.")

    markdown = save_demo_outputs(image, result.text, OUTPUT_DIR)

    print("\n--- Markdown Output ---\n")
    print(markdown)
    print(f"\nArtifacts saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
