"""Document ingestion: PDF/text/Markdown -> clean, chunked passages.

Uses a recursive splitter (paragraph -> line -> word boundaries) rather than a
naive fixed-character split, so chunks respect natural structure. Each chunk
carries source metadata for citation and per-namespace scoping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    namespace: str = "default"
    metadata: dict = field(default_factory=dict)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_document(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported document type: {suffix} (expected .pdf/.txt/.md)")


def chunk_text(
    text: str,
    source: str,
    *,
    namespace: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split text into overlapping, structure-aware chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]
    return [
        Chunk(
            id=f"{namespace}:{source}:{i}",
            text=piece,
            source=source,
            namespace=namespace,
            metadata={"source": source, "namespace": namespace, "chunk_index": i},
        )
        for i, piece in enumerate(pieces)
    ]


def ingest_file(
    path: str | Path,
    *,
    namespace: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    path = Path(path)
    text = read_document(path)
    return chunk_text(
        text,
        source=path.name,
        namespace=namespace,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
