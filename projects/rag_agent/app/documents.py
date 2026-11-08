"""Document validation, extraction, and chunking for grounded RAG queries."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_CHARS = 2_000_000
DEFAULT_CHUNK_SIZE = 1_200
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    document_id: str
    chunk_id: str
    text: str
    page: int | None = None


def validate_upload(filename: str | None, content: bytes) -> str:
    if not filename:
        raise ValueError("A filename is required")
    suffix = PurePath(filename).suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF, TXT, and Markdown files are supported")
    if not content:
        raise ValueError("The uploaded document is empty")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError("The uploaded document exceeds the 10 MB limit")
    return suffix


def extract_pages(filename: str, content: bytes) -> list[tuple[int | None, str]]:
    suffix = validate_upload(filename, content)
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
    else:
        pages = [(None, content.decode("utf-8-sig"))]
    total_chars = sum(len(text) for _, text in pages)
    if total_chars > MAX_DOCUMENT_CHARS:
        raise ValueError("The extracted document text exceeds the 2 million character limit")
    if not any(text.strip() for _, text in pages):
        raise ValueError("No readable text was found in the document")
    return [(page, re.sub(r"\s+", " ", text).strip()) for page, text in pages if text.strip()]


def document_id(filename: str, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()[:16]
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", PurePath(filename).stem).strip("-").lower()
    return f"{stem or 'document'}-{digest}"


def chunk_pages(
    doc_id: str,
    pages: list[tuple[int | None, str]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks: list[DocumentChunk] = []
    for page, text in pages:
        start = 0
        page_chunks = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        document_id=doc_id,
                        chunk_id=f"{doc_id}-{page or 0}-{page_chunks}",
                        text=chunk_text,
                        page=page,
                    )
                )
                page_chunks += 1
            if end == len(text):
                break
            start = end - overlap
    return chunks
