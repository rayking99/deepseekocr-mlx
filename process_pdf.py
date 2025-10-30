"""PDF-to-Markdown OCR pipeline using DeepSeek-OCR (MLX).

This script accepts a single PDF file or a directory of PDFs, renders pages to
images, runs the MLX DeepSeek-OCR model per page (optionally in parallel),
stores per-page artifacts (annotated image, crops, result.md), and writes one
aggregated Markdown file per PDF.

Usage examples:
  - Single PDF to out dir with 3 workers:
      .venv/bin/python process_pdf.py assets/ocr-guide-to-testing.pdf test_batch --workers 3

  - Directory of PDFs (recursively) to out dir:
      .venv/bin/python process_pdf.py assets/ test_batch
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import mlx.core as mx
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from deepseek_ocr_mlx import (
    DeepSeekOCRPreprocessor,
    GenerationConfig,
    generate,
    save_ocr_outputs,
)
from deepseek_ocr_mlx.load import load as load_model, load_config


DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
PDF_SUFFIXES = {".pdf"}


@dataclass
class PageTask:
    pdf_path: Path
    pdf_stem: str
    page_index: int
    image_bytes: bytes
    output_page_dir: Path


@dataclass
class WorkerState:
    tokenizer: PreTrainedTokenizerBase
    preprocessor: DeepSeekOCRPreprocessor
    model: object
    gen_config: GenerationConfig
    prompt: str


_WORKER_STATE: Optional[WorkerState] = None


def _discover_pdfs(root: Path) -> List[Path]:
    if root.is_file() and root.suffix.lower() in PDF_SUFFIXES:
        return [root.resolve()]
    pdfs: List[Path] = []
    if root.is_dir():
        for path in sorted(root.rglob("*.pdf")):
            if path.is_file():
                pdfs.append(path.resolve())
    return pdfs


def _render_pdf_to_images(pdf_path: Path, zoom: float = 2.0) -> List[Image.Image]:
    """Render a PDF to PIL images using PyMuPDF.

    zoom=2.0 roughly corresponds to ~144 DPI when the source is ~72 DPI. Increase
    for higher quality at the cost of memory and compute.
    """

    doc = fitz.open(pdf_path)
    try:
        images: List[Image.Image] = []
        mat = fitz.Matrix(zoom, zoom)
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            images.append(img)
        return images
    finally:
        doc.close()


def _init_worker(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    stop_on_eos: bool,
) -> None:
    global _WORKER_STATE

    model_dir = Path(snapshot_download(model_id))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = load_config(model_dir)
    preprocessor = DeepSeekOCRPreprocessor(tokenizer, config)
    model = load_model(model_dir, lazy=False)
    mx.eval(model.parameters())

    eos_ids_attr = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_ids_attr, int):
        stop_ids = [eos_ids_attr]
    elif eos_ids_attr:
        stop_ids = list(eos_ids_attr)
    else:
        stop_ids = []

    if stop_on_eos and not stop_ids and tokenizer.eos_token_id is not None:
        stop_ids = [tokenizer.eos_token_id]

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id if stop_on_eos else None,
        stop_token_ids=stop_ids if stop_on_eos else None,
        skip_special_tokens=False,
    )

    _WORKER_STATE = WorkerState(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        model=model,
        gen_config=gen_config,
        prompt=prompt,
    )


def _process_page(task: PageTask) -> Tuple[int, str]:
    if _WORKER_STATE is None:
        raise RuntimeError("Worker state not initialised")

    state = _WORKER_STATE

    # Recreate PIL image from bytes in the worker process
    with Image.open(io.BytesIO(task.image_bytes)) as img:
        image = img.convert("RGB")

    batch = state.preprocessor.prepare_single(state.prompt, [image])
    result = generate(state.model, state.tokenizer, batch, state.gen_config)

    # Persist per-page artifacts
    markdown = save_ocr_outputs(image, result.text, task.output_page_dir)
    return task.page_index, markdown


def _rewrite_image_paths(md: str, page_idx: int, base_prefix: str = "") -> str:
    """Rewrite relative image links to include the page folder.

    The per-page markdown produced by ``save_ocr_outputs`` uses links like
    ``![](images/0.jpg)`` that are relative to the page directory. When we
    aggregate pages, we need to prefix these with the page subfolder,
    e.g. ``page-000/images/0.jpg``. If ``base_prefix`` is provided (e.g. the
    PDF stem), it will be prepended as well: ``{pdf_stem}/page-000/images/0.jpg``.
    """

    import re

    page_prefix = f"{base_prefix}page-{page_idx:03d}/"

    # Replace variations like (images/0.jpg) or (./images/0.jpg)
    pattern = re.compile(r"\]\((?:\./)?images/([^)]+)\)")
    return pattern.sub(lambda m: f"]({page_prefix}images/{m.group(1)})", md)


def _aggregate_markdown(
    pdf_stem: str, page_markdowns: Sequence[Tuple[int, str]], base_prefix: str = ""
) -> str:
    parts: List[str] = [f"# {pdf_stem}"]
    for page_idx, md in sorted(page_markdowns, key=lambda x: x[0]):
        md_rewritten = _rewrite_image_paths(md, page_idx, base_prefix=base_prefix)
        parts.append(f"\n\n## Page {page_idx + 1}\n\n{md_rewritten.strip()}\n")
    return "\n---\n".join(parts).strip() + "\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSeek-OCR PDF to Markdown")
    p.add_argument("input", type=Path, help="PDF file or directory of PDFs")
    p.add_argument("output_dir", type=Path, help="Directory for outputs")
    p.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 2))
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--no-stop", action="store_true", help="Do not stop on EOS")
    p.add_argument("--zoom", type=float, default=2.0, help="Rendering zoom (1-3)")
    return p.parse_args()


def _process_single_pdf(
    pdf_path: Path,
    out_root: Path,
    workers: int,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    stop_on_eos: bool,
    zoom: float,
) -> None:
    pdf_stem = pdf_path.stem
    pdf_out_dir = out_root / pdf_stem
    pdf_out_dir.mkdir(parents=True, exist_ok=True)

    images = _render_pdf_to_images(pdf_path, zoom=zoom)
    if not images:
        print(f"[SKIP] {pdf_path} has no pages")
        return

    # Build page tasks with in-memory images
    tasks: List[PageTask] = []
    for i, image in enumerate(images):
        page_dir = pdf_out_dir / f"page-{i:03d}"
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        tasks.append(
            PageTask(
                pdf_path=pdf_path,
                pdf_stem=pdf_stem,
                page_index=i,
                image_bytes=buf.getvalue(),
                output_page_dir=page_dir,
            )
        )

    print(f"Processing {pdf_path.name}: {len(tasks)} page(s) with {workers} worker(s)")

    page_markdowns: List[Tuple[int, str]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(model, prompt, max_new_tokens, temperature, stop_on_eos),
    ) as ex:
        futures = {ex.submit(_process_page, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            task = futures[fut]
            try:
                page_idx, md = fut.result()
                page_markdowns.append((page_idx, md))
                print(f"[OK]  {pdf_path.name} page {page_idx + 1}")
            except Exception as e:  # pragma: no cover
                print(f"[FAIL] {pdf_path.name} page {task.page_index + 1}: {e}")

    # Aggregate markdown and write to two convenient locations:
    # Write combined markdown inside the PDF output directory with links relative to it
    combined_local = _aggregate_markdown(pdf_stem, page_markdowns, base_prefix="")
    (pdf_out_dir / "combined.md").write_text(combined_local, encoding="utf-8")

    # Also write a convenience copy at the root output directory. Since this file
    # sits one level above ``pdf_out_dir``, rewrite links to include the pdf stem.
    combined_root = _aggregate_markdown(
        pdf_stem, page_markdowns, base_prefix=f"{pdf_stem}/"
    )
    (out_root / f"{pdf_stem}.md").write_text(combined_root, encoding="utf-8")
    print(f"[DONE] {pdf_path.name} -> {pdf_stem}.md")


def main() -> None:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pdfs = _discover_pdfs(input_path)
    if not pdfs:
        raise SystemExit(f"No PDF(s) found at: {input_path}")

    for pdf in pdfs:
        _process_single_pdf(
            pdf,
            out_root,
            workers=args.workers,
            model=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop_on_eos=not args.no_stop,
            zoom=args.zoom,
        )


if __name__ == "__main__":
    main()
