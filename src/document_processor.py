from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    chunk_index: int
    page: int | None


def chunk_settings_for_file_size(
    size_bytes: int,
    default_chunk: int,
    default_overlap: int,
) -> tuple[int, int]:
    """Use larger chunks for big files to cut API calls while keeping search usable."""
    size_mb = size_bytes / (1024 * 1024)
    if size_mb >= 15:
        return max(default_chunk, 3500), max(default_overlap, 350)
    if size_mb >= 5:
        return max(default_chunk, 2500), max(default_overlap, 300)
    return default_chunk, default_overlap


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> list[tuple[str, int]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[tuple[str, int]] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = _normalize_whitespace(page.extract_text() or "")
        if page_text:
            pages.append((page_text, index))
    if not pages:
        raise ValueError(f"No extractable text found in {filename}.")
    return pages


def extract_text_from_txt(file_bytes: bytes, filename: str) -> list[tuple[str, int]]:
    for encoding in ("utf-8", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode text file {filename}.")

    normalized = _normalize_whitespace(text)
    if not normalized:
        raise ValueError(f"File {filename} is empty.")
    return [(normalized, 1)]


def chunk_pages(
    pages: list[tuple[str, int]],
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page_text, page_number in pages:
        start = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            piece = page_text[start:end].strip()
            if piece:
                chunks.append(
                    DocumentChunk(
                        text=piece,
                        source=source,
                        chunk_index=chunk_index,
                        page=page_number,
                    )
                )
                chunk_index += 1
            if end >= len(page_text):
                break
            start = max(end - chunk_overlap, start + 1)

    return chunks


def process_upload(
    filename: str,
    file_bytes: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    chunk_size, chunk_overlap = chunk_settings_for_file_size(
        len(file_bytes), chunk_size, chunk_overlap
    )
    lower = filename.lower()
    if lower.endswith(".pdf"):
        pages = extract_text_from_pdf(file_bytes, filename)
    elif lower.endswith(".txt"):
        pages = extract_text_from_txt(file_bytes, filename)
    else:
        raise ValueError("Only .pdf and .txt files are supported.")

    return chunk_pages(pages, source=filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
