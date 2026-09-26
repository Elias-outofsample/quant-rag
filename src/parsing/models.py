"""Backend-independent canonical RAG models."""
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class CanonicalDocument:
    document_id: str; source_path: str; filename: str; sha256: str
    title: str | None; subtitle: str | None; authors: list[str]
    editors: list[str] = field(default_factory=list)
    publication_year: int | None = None
    edition: str | None = None
    publisher: str | None = None
    page_count: int = 0; parser_backend: str = ''; parser_version: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
    def as_dict(self): return asdict(self)

@dataclass
class CanonicalBlock:
    block_id: str; document_id: str; page_idx: int; block_type: str; text: str
    bbox: list[float] | None; heading_level: int | None; parent_heading: str | None
    content_type: str; raw_text: str | None = None
    part: str | None = None; chapter: str | None = None
    section: str | None = None; subsection: str | None = None
    image_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    def as_dict(self): return asdict(self)

@dataclass
class CanonicalChunk:
    chunk_id: str; document_id: str; parent_id: str; previous_chunk_id: str | None
    next_chunk_id: str | None; page_start: int; page_end: int; part: str | None
    chapter: str | None; section: str | None; subsection: str | None; text: str
    content_type: str; token_count: int; rag_eligible: bool
    title_path: str | None = None
    image_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    def as_dict(self): return asdict(self)
