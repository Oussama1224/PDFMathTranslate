"""PDF ingestion: validation, decryption and normalisation of the input file."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from pt2en.errors import InvalidDocumentError

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"


@dataclass
class LoadedDocument:
    doc: pymupdf.Document
    source_bytes: bytes
    page_count: int
    metadata: dict
    notes: list[str] = field(default_factory=list)


def looks_like_pdf(data: bytes) -> bool:
    return PDF_MAGIC in data[:1024]


def load_pdf(
    path_or_bytes: Path | str | bytes,
    *,
    max_pages: int = 0,
    password: str = "",
) -> LoadedDocument:
    """Open and normalise a PDF for processing.

    Normalisation bakes page rotation into the content stream so every later
    stage can work in a single, unrotated coordinate system. The visual
    appearance of the pages is unchanged.
    """
    if isinstance(path_or_bytes, (str, Path)):
        data = Path(path_or_bytes).read_bytes()
    else:
        data = bytes(path_or_bytes)
    if not data:
        raise InvalidDocumentError("The uploaded file is empty.")
    if not looks_like_pdf(data):
        raise InvalidDocumentError("The uploaded file is not a PDF document.")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # pragma: no cover - depends on corrupt inputs
        raise InvalidDocumentError(
            "The PDF could not be opened; it may be corrupted.", detail=str(exc)
        ) from exc

    notes: list[str] = []
    if doc.needs_pass:
        if not doc.authenticate(password or ""):
            raise InvalidDocumentError(
                "The PDF is password-protected. Remove the password and try again."
            )
        notes.append("Document was encrypted and has been decrypted.")
    if doc.is_repaired:
        notes.append("The PDF structure was damaged and has been repaired.")
    if doc.page_count == 0:
        raise InvalidDocumentError("The PDF has no pages.")
    if max_pages and doc.page_count > max_pages:
        raise InvalidDocumentError(
            f"The PDF has {doc.page_count} pages; the limit is {max_pages}."
        )

    # Work on a clean copy (decrypted, garbage-collected) so the original bytes
    # stay untouched for previews and validation.
    clean = pymupdf.open(stream=doc.tobytes(garbage=1, deflate=True), filetype="pdf")
    for page in clean:
        if page.rotation:
            try:
                page.remove_rotation()
            except Exception:  # pragma: no cover - very old PyMuPDF
                log.warning("Could not normalise rotation of page %s", page.number)
    if any(p.rotation for p in clean):
        notes.append("Some pages are rotated; coordinates were normalised.")
    normalised = clean.tobytes(garbage=1, deflate=True)
    clean.close()
    work = pymupdf.open(stream=normalised, filetype="pdf")

    metadata = dict(doc.metadata or {})
    doc.close()
    return LoadedDocument(
        doc=work,
        source_bytes=normalised,
        page_count=work.page_count,
        metadata=metadata,
        notes=notes,
    )
