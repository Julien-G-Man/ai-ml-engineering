"""In-memory document metadata registry for the learning project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    filename: str
    chunks: int


class DocumentStore:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}

    def add(self, record: DocumentRecord) -> DocumentRecord:
        self._documents[record.document_id] = record
        return record

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._documents.get(document_id)

    def list(self) -> list[DocumentRecord]:
        return list(self._documents.values())


document_store = DocumentStore()
