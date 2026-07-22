"""PDF ingestion: the real _read_pdf path via a small committed fixture PDF
(no PDF-writing dependency needed in CI — the fixture is checked in)."""

from pathlib import Path

from app.ingest import ingest_file, read_document

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"


def test_read_pdf_extracts_text():
    text = read_document(FIXTURE)
    assert "Photosynthesis" in text
    assert "chlorophyll" in text


def test_ingest_pdf_produces_chunks_with_source():
    chunks = ingest_file(FIXTURE, namespace="default", chunk_size=120, chunk_overlap=20)
    assert chunks, "expected at least one chunk from the PDF"
    assert all(c.source == "sample.pdf" for c in chunks)
    assert any("glucose" in c.text.lower() for c in chunks)


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "notes.rtf"
    bad.write_text("hello")
    try:
        read_document(bad)
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported type")
